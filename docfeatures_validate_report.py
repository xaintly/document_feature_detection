#!/usr/bin/env python3
"""
docfeatures_validate_report.py — Diagnose enum-validation failures

docfeatures.py retries a chunk when the LLM returns an enum value outside
its declared options, and usually recovers on a corrective retry -- which
is good for throughput, but means most failures never show up as a
document-level error anymore (see log_validation_failure() in
lib/docfeatures_lib.py). This tool reports on the validation_failures table
that captures every rejected attempt regardless of whether it was later
corrected, and classifies each invalid value to point at what's actually
wrong: a missing enum option, two features the model is confusing with each
other, or a genuinely novel/hallucinated value.

Two modes:

  --backfill   One-time (idempotent) import of validation failures from
               *pre-existing* document_runs.error_message text (status=
               'error' rows), for data captured before this table existed
               or before a document was reprocessed and its error_message
               overwritten. Only recovers documents that never got past
               retry exhaustion -- it can't recover attempts that were
               corrected on retry and never became a document-level error,
               since that information was never persisted anywhere.

  (default)    Report mode: classified breakdown of invalid values per
               feature, with example documents, for a --run-name.

Usage:
    python docfeatures_validate_report.py --backfill
    python docfeatures_validate_report.py --backfill --run-name v3
    python docfeatures_validate_report.py --run-name v3
    python docfeatures_validate_report.py --run-name v3 --feature malignancy_likelihood
"""

import argparse
import ast
import difflib
import re
import sys
from collections import Counter, defaultdict

import yaml

from lib.docfeatures_lib import get_connection

DEFAULT_TOP = 10
DEFAULT_EXAMPLES = 2

# Prefixes/suffixes stripped when checking whether an invalid value looks
# like a *derived* form of another feature's name (e.g. "has_lung_cancer"
# -> "lung_cancer"), not just an exact match.
_STRIP_PREFIXES = ("has_", "is_", "shows_", "contains_", "with_", "no_")
_STRIP_SUFFIXES = ("_present", "_finding", "_flag", "_noted", "_positive")

_LEGACY_ERROR_RE = re.compile(
    r"Feature '(?P<feature>[^']+)'(?:\s*\(chunk (?P<chunk_idx>\d+)/\d+\))?:\s*"
    r"LLM returned (?P<value>.+?), which is not one of the declared options"
)


def _clean_repr_value(raw):
    """The original error message embeds the value via Python's !r (repr),
    so ast.literal_eval is the exact correct inverse for the common cases
    (quoted strings, None, numbers) -- falls back to the raw text for
    anything that isn't a valid literal (rare, but not worth failing on)."""
    raw = raw.strip()
    try:
        return str(ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return raw


def parse_legacy_error_message(message):
    """Extract (feature_name, chunk_index, invalid_value) from an old-style
    document_runs.error_message string, or None if it doesn't match the
    enum-validation shape (a JSON-parse failure, connection error, etc. --
    not this table's concern)."""
    m = _LEGACY_ERROR_RE.search(message or "")
    if not m:
        return None
    feature = m.group("feature")
    # error messages show 1-based "chunk N/M"; log_validation_failure()'s
    # chunk_index (like chunk_results.chunk_index) is 0-based.
    chunk_idx = int(m.group("chunk_idx")) - 1 if m.group("chunk_idx") else None
    value = _clean_repr_value(m.group("value"))
    return feature, chunk_idx, value


def classify_invalid_value(value, feature_name, features_config):
    """Return (category, related_feature_or_None) explaining *why* an
    invalid value probably showed up, by cross-referencing the run's own
    feature schema -- not just reporting the raw string."""
    value_norm = str(value).strip().lower()
    fdef = features_config.get(feature_name, {})
    own_options = [o.lower() for o in fdef.get("options", [])]

    close = difflib.get_close_matches(value_norm, own_options, n=1, cutoff=0.6)
    if close:
        return "own_near_miss", close[0]

    for other_name, other_fdef in features_config.items():
        if other_name == feature_name or other_fdef.get("type") != "enum":
            continue
        if value_norm in [o.lower() for o in other_fdef.get("options", [])]:
            return "other_enum_value", other_name

    for other_name in features_config:
        if other_name == feature_name:
            continue
        if value_norm in (other_name.lower(), other_name.lower().replace("_", " ")):
            return "other_field_name", other_name

    for other_name in features_config:
        if other_name == feature_name:
            continue
        stripped = other_name.lower()
        for p in _STRIP_PREFIXES:
            if stripped.startswith(p):
                stripped = stripped[len(p):]
        for s in _STRIP_SUFFIXES:
            if stripped.endswith(s):
                stripped = stripped[: -len(s)]
        if stripped and (
            value_norm == stripped
            or difflib.SequenceMatcher(None, value_norm, stripped).ratio() >= 0.75
        ):
            return "modified_field_name", other_name

    return "novel", None


_CATEGORY_LABELS = {
    "own_near_miss": "near-miss of its own valid options (validator/formatting issue, not a prompt problem)",
    "other_enum_value": "matches an option from a DIFFERENT enum feature (features may be confusable)",
    "other_field_name": "matches another feature's NAME exactly (prompt likely too dense/ambiguous)",
    "modified_field_name": "resembles a derived form of another feature's NAME (same as above)",
    "novel": "not traceable to anything else in the schema (possible missing enum option)",
}


def cmd_backfill(args):
    conn = get_connection()
    with conn.cursor() as cur:
        if args.run_name:
            cur.execute(
                "SELECT doc_id, run_name, file_id, error_message FROM document_runs "
                "WHERE run_name=%s AND status='error'",
                (args.run_name,),
            )
        else:
            cur.execute(
                "SELECT doc_id, run_name, file_id, error_message FROM document_runs "
                "WHERE status='error'"
            )
        rows = cur.fetchall()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT doc_id FROM validation_failures WHERE source='backfill'"
        )
        already_backfilled = {r["doc_id"] for r in cur.fetchall()}

    scanned = len(rows)
    matched = 0
    skipped_done = 0
    skipped_no_match = 0

    for row in rows:
        if row["doc_id"] in already_backfilled:
            skipped_done += 1
            continue
        parsed = parse_legacy_error_message(row["error_message"])
        if parsed is None:
            skipped_no_match += 1
            continue
        feature_name, chunk_idx, invalid_value = parsed
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO validation_failures "
                "(run_name, file_id, doc_id, chunk_index, feature_name, invalid_value, "
                "error_message, attempt, source) VALUES (%s,%s,%s,%s,%s,%s,%s,NULL,'backfill')",
                (row["run_name"], row["file_id"], row["doc_id"], chunk_idx,
                 feature_name, invalid_value, row["error_message"]),
            )
        matched += 1

    print(f"Scanned      : {scanned} error document(s)"
          + (f" for run '{args.run_name}'" if args.run_name else " across all runs"))
    print(f"Backfilled   : {matched}")
    print(f"Already done : {skipped_done} (idempotent -- safe to re-run)")
    print(f"Not enum-related (skipped): {skipped_no_match}")
    conn.close()


def cmd_report(args):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT config_yaml FROM runs WHERE run_name=%s", (args.run_name,))
        row = cur.fetchone()
    if not row:
        print(f"No run named '{args.run_name}'.", file=sys.stderr)
        sys.exit(1)
    features_config = yaml.safe_load(row["config_yaml"]).get("features", {})

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM chunk_results cr JOIN document_runs dr "
            "ON cr.doc_id = dr.doc_id WHERE dr.run_name=%s",
            (args.run_name,),
        )
        total_chunks = cur.fetchone()["n"]

    with conn.cursor() as cur:
        if args.feature:
            cur.execute(
                "SELECT feature_name, invalid_value FROM validation_failures "
                "WHERE run_name=%s AND feature_name=%s",
                (args.run_name, args.feature),
            )
        else:
            cur.execute(
                "SELECT feature_name, invalid_value FROM validation_failures WHERE run_name=%s",
                (args.run_name,),
            )
        failures = cur.fetchall()

    if not failures:
        print(f"No validation failures recorded for run '{args.run_name}'"
              + (f", feature '{args.feature}'" if args.feature else "") + ".")
        print("(If this run predates the validation_failures table, try --backfill first.)")
        conn.close()
        return

    by_feature = defaultdict(Counter)
    for f in failures:
        by_feature[f["feature_name"]][f["invalid_value"]] += 1

    print(f"Run '{args.run_name}': {len(failures)} logged validation failure(s) across "
          f"{len(by_feature)} feature(s) (~{total_chunks:,} chunks processed total)")
    print("=" * 70)

    for feature_name, counter in sorted(by_feature.items(), key=lambda kv: -sum(kv[1].values())):
        feature_total = sum(counter.values())
        rate = f"{feature_total / total_chunks * 100:.2f}%" if total_chunks else "n/a"
        print(f"\n{feature_name}  ({feature_total} failure(s), ~{rate} of all processed chunks)")
        print("-" * 70)

        category_totals = Counter()
        for value, count in counter.most_common(args.top):
            category, related = classify_invalid_value(value, feature_name, features_config)
            category_totals[category] += count
            tag = f"{category}" + (f" -> {related}" if related else "")
            print(f"  {count:>6}  {value!r:<30} [{tag}]")

            if args.examples:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT f.file_path FROM validation_failures vf "
                        "JOIN files f ON vf.file_id = f.file_id "
                        "WHERE vf.run_name=%s AND vf.feature_name=%s AND vf.invalid_value=%s "
                        "LIMIT %s",
                        (args.run_name, feature_name, value, args.examples),
                    )
                    for ex in cur.fetchall():
                        print(f"           e.g. {ex['file_path']}")

        if len(counter) > args.top:
            print(f"  ... and {len(counter) - args.top} more distinct value(s) (--top to show more)")

        print("\n  Category breakdown:")
        for category, count in category_totals.most_common():
            print(f"    {count:>6}  {_CATEGORY_LABELS[category]}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose docfeatures.py enum-validation failures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --backfill                          # one-time import of pre-existing error data, all runs
  %(prog)s --backfill --run-name v3            # ... scoped to one run
  %(prog)s --run-name v3                       # classified report
  %(prog)s --run-name v3 --feature malignancy_likelihood --top 20 --examples 3
        """,
    )
    parser.add_argument("--backfill", action="store_true",
                         help="Import validation failures from existing document_runs.error_message "
                              "text instead of generating a report. Idempotent.")
    parser.add_argument("-r", "--run-name",
                         help="Required for the report. Optional filter for --backfill (all runs if omitted).")
    parser.add_argument("--feature", help="Limit the report to one feature.")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                         help="Show the top N distinct invalid values per feature (default: %(default)s).")
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES,
                         help="Example source documents per invalid value, 0 to disable (default: %(default)s).")
    args = parser.parse_args()

    if args.backfill:
        cmd_backfill(args)
        return

    if not args.run_name:
        parser.error("--run-name is required for the report (or use --backfill).")
    cmd_report(args)


if __name__ == "__main__":
    main()
