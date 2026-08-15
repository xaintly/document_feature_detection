#!/usr/bin/env python3
"""
docfeatures_fix_paths.py — Correct stale capitalization in files.file_path

MySQL's default collation makes `files.file_path` lookups case-insensitive,
so "FILE.TXT" and "file.txt" already resolve to the same file_id -- but the
stored string itself is whatever casing was first seen. If a corpus
originally had two files differing only by case and one was later deleted
or renamed (e.g. to fix a Windows-incompatible duplicate), the survivor's
on-disk casing can drift from what's stored in `files`. Two symptoms follow:

  - docfeatures.py / docfeatures_batch.py can mis-treat the file as "never
    processed" (a case-sensitive Python string comparison disagreeing with
    the DB's case-insensitive identity) and then fail inserting a duplicate
    (run_name, file_id) row. filter_pending() in lib/docfeatures_lib.py
    fixes that going forward, but doesn't repair already-stored paths.
  - docfeatures_web.py can fail to read the file for preview/export, since
    a case-sensitive filesystem needs the exact on-disk casing.

This tool scans every `files` row, and for any whose stored path doesn't
exist exactly as stored but a uniquely-matching case-insensitive file does
in the same directory, corrects the stored path to match. Rows with no
on-disk match at all (files genuinely removed from the corpus -- a separate,
unrelated situation) or with more than one case-insensitive match
(an unresolved true duplicate) are reported but left alone.

Usage:
    python docfeatures_fix_paths.py --dry-run
    python docfeatures_fix_paths.py
    python docfeatures_fix_paths.py --corpus-base /path/to/corpus --dry-run
"""

import argparse
import os
import sys

import pymysql

from lib.docfeatures_lib import get_connection


def find_case_correct_path(base, stored_path):
    """Compare a stored files.file_path against disk.

    Returns (corrected_path_or_None, status), where status is one of:
      'ok'        -- exists exactly as stored, nothing to do
      'fixable'   -- missing as stored, but exactly one case-insensitive
                      match exists in the same directory
      'ambiguous' -- missing as stored, but more than one case-insensitive
                      match exists (an unresolved true duplicate)
      'missing'   -- missing as stored, and no case-insensitive match either
                      (the file was removed from the corpus entirely --
                      unrelated to capitalization, nothing this tool can fix)
    """
    full = stored_path if os.path.isabs(stored_path) else os.path.join(base, stored_path)
    if os.path.exists(full):
        return None, "ok"

    d = os.path.dirname(full)
    target = os.path.basename(full).lower()
    if not os.path.isdir(d):
        return None, "missing"

    matches = [entry for entry in os.listdir(d) if entry.lower() == target]
    if not matches:
        return None, "missing"
    if len(matches) > 1:
        return None, "ambiguous"

    corrected_full = os.path.join(d, matches[0])
    corrected = corrected_full if os.path.isabs(stored_path) else os.path.relpath(corrected_full, base)
    return corrected, "fixable"


def main():
    parser = argparse.ArgumentParser(
        description="Correct files.file_path rows whose stored capitalization no longer "
                    "matches the on-disk filename.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --dry-run                          # preview, no changes
  %(prog)s                                     # apply corrections
  %(prog)s --corpus-base /path/to/corpus -n 5 --dry-run
        """,
    )
    parser.add_argument(
        "--corpus-base", default=os.environ.get("CORPUS_BASE_PATH"),
        help="Base directory that relative files.file_path values are resolved against "
             "(default: CORPUS_BASE_PATH from .env).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    parser.add_argument("-n", "--limit", type=int, help="Stop after N corrections.")
    args = parser.parse_args()

    if not args.corpus_base:
        parser.error("No corpus base directory. Pass --corpus-base or set CORPUS_BASE_PATH in .env.")

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT file_id, file_path FROM files ORDER BY file_id")
        rows = cur.fetchall()

    print(f"Scanning {len(rows)} files row(s) against {args.corpus_base} ...", file=sys.stderr)

    already_ok = 0
    fixed = 0
    ambiguous = 0
    missing = 0

    for row in rows:
        corrected, status = find_case_correct_path(args.corpus_base, row["file_path"])

        if status == "ok":
            already_ok += 1
            continue
        if status == "missing":
            missing += 1
            continue
        if status == "ambiguous":
            ambiguous += 1
            print(f"  [AMBIGUOUS] file_id={row['file_id']} {row['file_path']!r} has more than one "
                  f"case-insensitive match on disk -- skipped, resolve manually.", file=sys.stderr)
            continue

        # status == "fixable"
        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"  {prefix}file_id={row['file_id']}  {row['file_path']!r} -> {corrected!r}")
        if not args.dry_run:
            try:
                with conn.cursor() as cur:
                    cur.execute("UPDATE files SET file_path=%s WHERE file_id=%s", (corrected, row["file_id"]))
            except pymysql.err.IntegrityError as e:
                print(f"    Could not update file_id={row['file_id']}: {e}", file=sys.stderr)
                continue
        fixed += 1
        if args.limit and fixed >= args.limit:
            break

    print()
    print(f"Already correct : {already_ok}")
    print(f"{'Would fix' if args.dry_run else 'Fixed'}       : {fixed}")
    print(f"Ambiguous       : {ambiguous}  (multiple case-insensitive matches on disk -- resolve manually)")
    print(f"Not found       : {missing}  (no on-disk match at all -- removed from the corpus, unrelated to capitalization)")
    conn.close()


if __name__ == "__main__":
    main()
