#!/usr/bin/env python3
"""
docfeatures_dedupe.py — Deduplicate documents within a run by file_hash.

Finds documents in a run that share an identical file_hash (byte-for-byte
identical content — e.g. boilerplate "removed for privacy" placeholders, or
the same file present under multiple paths). For each group of duplicates,
one document is kept and the rest are deleted:

  - Keeper selection: prefers a document with status='complete' (it has
    features); among candidates, the lowest doc_id wins. Ties are otherwise
    arbitrary, per spec.
  - Tags (document_features) on documents being deleted are merged onto the
    keeper: a feature name the keeper doesn't already have is copied over;
    a feature name the keeper already has is left as-is (the keeper's value
    wins) and the discarded value is reported as a conflict.
  - Deleting a document cascades to its document_features and chunk_results
    rows (FK ON DELETE CASCADE).
  - Optionally (--delete-files), the duplicate's underlying file is also
    deleted from disk, so future runs over the same corpus won't rediscover
    it as a "new" duplicate. A missing file (e.g. already removed while
    deduping a different run_name against the same corpus) is silently
    treated as already-handled.

Usage:
    python docfeatures_dedupe.py --run-name lung_v1 --dry-run
    python docfeatures_dedupe.py --run-name lung_v1
    python docfeatures_dedupe.py --run-name lung_v1 --limit 20 --report dedupe.log
    python docfeatures_dedupe.py --run-name lung_v1 --delete-files
"""

import argparse
import os
import sys
from datetime import datetime

import pymysql
from dotenv import load_dotenv

load_dotenv()

DEFAULT_REPORT_PATH = "dedupe_report.txt"


def get_connection():
    """Connect to MySQL using credentials from environment / .env file."""
    return pymysql.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "docfeatures"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def run_exists(conn, run_name):
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM runs WHERE run_name=%s", (run_name,))
        return cur.fetchone() is not None


def find_duplicate_hashes(conn, run_name):
    """Return list of file_hash values that occur more than once in the run."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_hash FROM documents "
            "WHERE run_name=%s AND file_hash IS NOT NULL "
            "GROUP BY file_hash HAVING COUNT(*) > 1 "
            "ORDER BY file_hash",
            (run_name,),
        )
        return [row["file_hash"] for row in cur.fetchall()]


def get_docs_for_hash(conn, run_name, file_hash):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, file_path, status FROM documents "
            "WHERE run_name=%s AND file_hash=%s ORDER BY doc_id",
            (run_name, file_hash),
        )
        return cur.fetchall()


def get_features(conn, doc_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT feature_name, value_text FROM document_features "
            "WHERE doc_id=%s",
            (doc_id,),
        )
        return {row["feature_name"]: row["value_text"] for row in cur.fetchall()}


def pick_keeper(docs):
    """Prefer a 'complete' document; among candidates, lowest doc_id."""
    complete = [d for d in docs if d["status"] == "complete"]
    pool = complete if complete else docs
    return min(pool, key=lambda d: d["doc_id"])


def copy_feature(conn, doc_id, feature_name, value_text):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO document_features (doc_id, feature_name, value_text) "
            "VALUES (%s,%s,%s)",
            (doc_id, feature_name, value_text),
        )


def delete_document(conn, doc_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))


def delete_source_file(file_path):
    """Delete a file from disk.

    Returns (result, err) where result is 'removed', 'missing', or 'failed'.
    A missing file is treated as already-handled (e.g. a previous dedupe run
    against a different run_name already deleted it), not an error.
    """
    try:
        os.remove(file_path)
        return "removed", None
    except FileNotFoundError:
        return "missing", None
    except OSError as e:
        return "failed", str(e)


def dedupe_run(conn, run_name, dry_run, limit, delete_files, report_lines):
    hashes = find_duplicate_hashes(conn, run_name)

    groups_seen = 0
    docs_deleted = 0
    tags_copied = 0
    tags_conflicted = 0
    files_removed = 0
    files_already_missing = 0
    file_delete_failures = 0
    limit_hit = False

    for file_hash in hashes:
        if limit is not None and docs_deleted >= limit:
            limit_hit = True
            break

        docs = get_docs_for_hash(conn, run_name, file_hash)
        keeper = pick_keeper(docs)
        dupes = [d for d in docs if d["doc_id"] != keeper["doc_id"]]
        if not dupes:
            continue

        groups_seen += 1
        keeper_features = get_features(conn, keeper["doc_id"])

        report_lines.append(
            f"Hash {file_hash[:12]}... ({len(docs)} docs) -> "
            f"KEEP doc_id={keeper['doc_id']} ({keeper['file_path']})"
        )

        for dupe in dupes:
            if limit is not None and docs_deleted >= limit:
                limit_hit = True
                break

            dupe_features = get_features(conn, dupe["doc_id"])
            action_lines = []

            for name, value in dupe_features.items():
                if name not in keeper_features:
                    if not dry_run:
                        copy_feature(conn, keeper["doc_id"], name, value)
                    keeper_features[name] = value
                    tags_copied += 1
                    action_lines.append(f"    copied: {name} = {value}")
                elif keeper_features[name] != value:
                    tags_conflicted += 1
                    action_lines.append(
                        f"    conflict: kept {name}={keeper_features[name]}, "
                        f"discarded {name}={value}"
                    )
                # identical value on both sides: no action, not reported

            file_note = ""
            proceed_with_db_delete = True
            if delete_files:
                if dry_run:
                    file_note = " [+ would delete source file]"
                else:
                    result, err = delete_source_file(dupe["file_path"])
                    if result == "removed":
                        files_removed += 1
                        file_note = " [+ source file removed]"
                    elif result == "missing":
                        files_already_missing += 1
                        file_note = " [source file already missing, skipped]"
                    else:
                        file_delete_failures += 1
                        proceed_with_db_delete = False
                        file_note = (
                            f" [FILE DELETE FAILED ({err}) — DB record kept]"
                        )

            if not proceed_with_db_delete:
                report_lines.append(
                    f"  SKIPPED doc_id={dupe['doc_id']} "
                    f"({dupe['file_path']}){file_note}"
                )
                report_lines.extend(action_lines)
                continue

            if not dry_run:
                delete_document(conn, dupe["doc_id"])
            docs_deleted += 1

            verb = "WOULD DELETE" if dry_run else "DELETED"
            report_lines.append(
                f"  {verb} doc_id={dupe['doc_id']} "
                f"({dupe['file_path']}){file_note}"
            )
            report_lines.extend(action_lines)

        if limit_hit:
            break

    return {
        "groups": groups_seen,
        "docs_deleted": docs_deleted,
        "tags_copied": tags_copied,
        "tags_conflicted": tags_conflicted,
        "files_removed": files_removed,
        "files_already_missing": files_already_missing,
        "file_delete_failures": file_delete_failures,
        "limit_hit": limit_hit,
        "total_dup_hashes": len(hashes),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate documents within a run by file_hash.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Preview what would happen, no changes made
  %(prog)s --run-name lung_v1 --dry-run

  # Actually dedupe (asks for confirmation)
  %(prog)s --run-name lung_v1

  # Only handle the first 20 duplicate documents, custom report path
  %(prog)s --run-name lung_v1 --limit 20 --report dedupe.log
        """,
    )
    parser.add_argument("-r", "--run-name", required=True, help="Run to deduplicate.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would happen without deleting or modifying anything.",
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=None, metavar="N",
        help="Stop after handling N duplicate documents.",
    )
    parser.add_argument(
        "--report", default=DEFAULT_REPORT_PATH, metavar="PATH",
        help=f"Report file to append to (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--delete-files", action="store_true",
        help="Also delete each duplicate's underlying file from disk (not "
             "just the DB record), so the same content isn't picked up "
             "again in future runs. If the file no longer exists (e.g. it "
             "was already removed while deduping a different run), this is "
             "silently skipped.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation prompt (only relevant without --dry-run).",
    )
    args = parser.parse_args()

    try:
        conn = get_connection()
    except pymysql.err.OperationalError as e:
        code, msg = e.args
        print(f"Connection failed: {msg}", file=sys.stderr)
        sys.exit(1)

    if not run_exists(conn, args.run_name):
        print(f"Run '{args.run_name}' not found.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and not args.yes:
        file_warning = " and their underlying files on disk" if args.delete_files else ""
        confirm = input(
            f"This will permanently delete duplicate documents (and their "
            f"tags/chunks){file_warning} in run '{args.run_name}'. "
            f"Continue? [y/N] "
        )
        if confirm.strip().lower() != "y":
            print("Cancelled.")
            conn.close()
            return

    header = (
        f"=== Dedup run: {datetime.now().isoformat(timespec='seconds')} "
        f"run_name={args.run_name} "
        f"mode={'dry-run' if args.dry_run else 'live'}"
        + (f" limit={args.limit}" if args.limit is not None else "")
        + " ==="
    )
    report_lines = [header]

    stats = dedupe_run(
        conn, args.run_name, args.dry_run, args.limit, args.delete_files,
        report_lines,
    )

    summary = (
        f"--- summary: {stats['total_dup_hashes']} duplicate hash group(s) found, "
        f"{stats['groups']} processed, {stats['docs_deleted']} document(s) "
        f"{'would be ' if args.dry_run else ''}deleted, "
        f"{stats['tags_copied']} tag(s) copied, "
        f"{stats['tags_conflicted']} conflict(s)"
    )
    if args.delete_files and not args.dry_run:
        summary += (
            f", {stats['files_removed']} file(s) removed, "
            f"{stats['files_already_missing']} already missing, "
            f"{stats['file_delete_failures']} file delete failure(s)"
        )
    summary += (
        (", LIMIT REACHED (more duplicates remain)" if stats["limit_hit"] else "")
        + " ---"
    )
    report_lines.append(summary)
    report_lines.append("")

    with open(args.report, "a") as f:
        f.write("\n".join(report_lines) + "\n")

    for line in report_lines:
        print(line, file=sys.stderr)
    print(f"\nReport appended to {args.report}", file=sys.stderr)

    conn.close()


if __name__ == "__main__":
    main()
