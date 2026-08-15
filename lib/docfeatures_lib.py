#!/usr/bin/env python3
"""
docfeatures_lib.py — Shared library for docfeatures.py and docfeatures_batch.py

Corpus-agnostic pieces used by both the synchronous (local LLM) and batch
(AWS Bedrock) processing tools: database access, text sanitization/chunking,
prompt generation, response parsing/merging, file discovery, and config
loading. Deliberately has no dependency on `boto3` or `requests` so that
importing it doesn't pull in either tool's transport-specific dependencies.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pymysql
import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Chunk target: ~128k token context minus prompt/output overhead.
# At ~3 chars/token for clinical text, 350k chars ≈ 117k tokens,
# leaving room for the prompt (~1k tokens) and response (~500 tokens).
# Override with --chunk-size if your model has a different context window.
CHUNK_TARGET_CHARS = os.environ.get("CHUNK_TARGET_CHARS", 150_000)
TEXT_EXTENSIONS = {".txt", ".html", ".htm", ".md", ".text"}


# ===========================================================================
# Database
# ===========================================================================

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


def get_or_create_run(conn, run_name, config, config_hash, host, model, temperature):
    with conn.cursor() as cur:
        cur.execute("SELECT run_name FROM runs WHERE run_name = %s", (run_name,))
        if cur.fetchone():
            return
        desc = config.get("run_description", "")
        cur.execute(
            "INSERT INTO runs (run_name, config_hash, config_yaml, "
            "description, llm_host, llm_model, llm_temperature) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (run_name, config_hash, yaml.dump(config), desc, host, model, temperature),
        )


def cleanup_incomplete(conn, run_name):
    """Remove docs stuck in 'processing' (interrupted mid-flight).

    Deliberately does NOT touch 'batch_pending' rows — those are claimed by
    an in-flight or awaiting-submission Bedrock batch job (see
    docfeatures_batch.py) and must survive a sync-tool run starting up.
    """
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_runs WHERE run_name=%s AND status='processing'",
            (run_name,),
        )
        if cur.rowcount:
            print(
                f"  Cleaned up {cur.rowcount} interrupted document(s) "
                "from previous session.",
                file=sys.stderr,
            )


def get_finished_paths(conn, run_name, retry_errors=False):
    """Return set of file_paths already finished or claimed for this run --
    i.e. NOT eligible to be picked up as pending again.

    Always excludes 'complete' and 'batch_pending' (claimed by a Bedrock
    batch job that hasn't been imported or cancelled yet — prevents the
    sync tool and a batch `prepare` from double-claiming the same file, and
    two concurrent batch preps from double-claiming each other's files) from
    the pending pool. 'error' is excluded too (left un-retried) unless
    retry_errors is True, in which case errored documents are left OUT of
    this "finished" set so they come back as pending. Pass
    retry_errors=args.retry_errors directly -- no extra negation, so as not
    to repeat the sign-inversion bug this parameter used to have.
    """
    statuses = "('complete','batch_pending')" if retry_errors else "('complete','error','batch_pending')"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT f.file_path FROM document_runs dr "
            "JOIN files f ON dr.file_id = f.file_id "
            f"WHERE dr.run_name=%s AND dr.status IN {statuses}",
            (run_name,),
        )
        return {row["file_path"] for row in cur.fetchall()}


def filter_pending(all_files, finished):
    """Return the subset of all_files not already in finished, comparing
    case-insensitively.

    MySQL's default collation makes `files.file_path` identity
    case-insensitive already (get_or_create_file resolves "FILE.TXT" and
    "file.txt" to the same row), so a case-sensitive Python string
    comparison here can disagree with the database: if a file's on-disk
    capitalization has drifted from what's stored (e.g. a corpus originally
    had two files differing only by case and one was later deleted or
    renamed), the file would look "pending" even though its file_id already
    has a complete row for this run -- and the subsequent INSERT would fail
    on the (run_name, file_id) unique constraint. Comparing case-
    insensitively here keeps this in agreement with the database's own
    notion of file identity. See also docfeatures_fix_paths.py, which
    corrects the stored casing so on-disk reads (e.g. the web app's preview)
    keep working too.
    """
    finished_casefold = {f.casefold() for f in finished}
    return [f for f in all_files if f.casefold() not in finished_casefold]


def validate_filter(conn, filter_config):
    """Check that the source run exists and that referenced features are valid.
    Returns the source run's feature config for cross-reference."""
    from_run = filter_config.get("from_run")
    if not from_run:
        raise ValueError("Filter config must include 'from_run'.")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT config_yaml FROM runs WHERE run_name = %s", (from_run,)
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(
                f"Filter references run '{from_run}', but it does not exist. "
                f"Use --list-runs to see available runs."
            )
        source_config = yaml.safe_load(row["config_yaml"])
        source_features = set(source_config.get("features", {}).keys())

        # Validate that all referenced feature names exist in the source run
        for section_name in ("require", "exclude"):
            section = filter_config.get(section_name, {})
            for feat_name in section:
                if feat_name not in source_features:
                    raise ValueError(
                        f"Filter {section_name} references feature "
                        f"'{feat_name}', but run '{from_run}' does not "
                        f"have that feature. Available: "
                        f"{', '.join(sorted(source_features))}"
                    )

        # Check that the source run has completed documents
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM document_runs "
            "WHERE run_name=%s AND status='complete'",
            (from_run,),
        )
        count = cur.fetchone()["cnt"]
        if count == 0:
            raise ValueError(
                f"Run '{from_run}' has no completed documents to filter."
            )

    return source_config


def get_filtered_paths(conn, filter_config):
    """Build a JOIN-based query to select file_paths matching the filter.

    Uses INNER JOINs for 'require' criteria and LEFT JOIN + IS NULL for
    'exclude' criteria. Designed for corpora with hundreds of millions of
    rows where IN (SELECT ...) subqueries would be prohibitively slow.

    Returns a list of file_path strings.
    """
    from_run = filter_config["from_run"]
    require = filter_config.get("require", {})
    exclude = filter_config.get("exclude", {})

    # Start building the query
    # d = source document_runs table, joined to files for file_path
    joins = []
    where_clauses = ["d.run_name = %s", "d.status = 'complete'"]
    params = []

    # --- REQUIRE: INNER JOIN for each required feature ---
    for i, (feat_name, feat_value) in enumerate(require.items()):
        alias = f"req{i}"
        if isinstance(feat_value, bool) and feat_value is True:
            # Boolean true: row must exist (we only store positive values)
            joins.append(
                f"INNER JOIN document_features {alias} "
                f"ON d.doc_id = {alias}.doc_id "
                f"AND {alias}.feature_name = %s"
            )
            params.append(feat_name)
        elif isinstance(feat_value, bool) and feat_value is False:
            # Boolean false: row must NOT exist (same as exclude)
            joins.append(
                f"LEFT JOIN document_features {alias} "
                f"ON d.doc_id = {alias}.doc_id "
                f"AND {alias}.feature_name = %s"
            )
            params.append(feat_name)
            where_clauses.append(f"{alias}.id IS NULL")
        elif isinstance(feat_value, list):
            # Enum: row must exist with one of the listed values
            placeholders = ", ".join(["%s"] * len(feat_value))
            joins.append(
                f"INNER JOIN document_features {alias} "
                f"ON d.doc_id = {alias}.doc_id "
                f"AND {alias}.feature_name = %s "
                f"AND {alias}.value_text IN ({placeholders})"
            )
            params.append(feat_name)
            params.extend(str(v) for v in feat_value)
        else:
            # Single enum value (string)
            joins.append(
                f"INNER JOIN document_features {alias} "
                f"ON d.doc_id = {alias}.doc_id "
                f"AND {alias}.feature_name = %s "
                f"AND {alias}.value_text = %s"
            )
            params.append(feat_name)
            params.append(str(feat_value))

    # --- EXCLUDE: LEFT JOIN + IS NULL for each excluded feature ---
    for i, (feat_name, feat_value) in enumerate(exclude.items()):
        alias = f"exc{i}"
        if isinstance(feat_value, bool) and feat_value is True:
            # Exclude documents where this feature is true (row exists)
            joins.append(
                f"LEFT JOIN document_features {alias} "
                f"ON d.doc_id = {alias}.doc_id "
                f"AND {alias}.feature_name = %s"
            )
            params.append(feat_name)
            where_clauses.append(f"{alias}.id IS NULL")
        elif isinstance(feat_value, list):
            # Exclude documents with any of these values
            placeholders = ", ".join(["%s"] * len(feat_value))
            joins.append(
                f"LEFT JOIN document_features {alias} "
                f"ON d.doc_id = {alias}.doc_id "
                f"AND {alias}.feature_name = %s "
                f"AND {alias}.value_text IN ({placeholders})"
            )
            params.append(feat_name)
            params.extend(str(v) for v in feat_value)
            where_clauses.append(f"{alias}.id IS NULL")
        else:
            # Exclude documents with this specific value
            joins.append(
                f"LEFT JOIN document_features {alias} "
                f"ON d.doc_id = {alias}.doc_id "
                f"AND {alias}.feature_name = %s "
                f"AND {alias}.value_text = %s"
            )
            params.append(feat_name)
            params.append(str(feat_value))
            where_clauses.append(f"{alias}.id IS NULL")

    sql = (
        "SELECT f.file_path FROM document_runs d\n"
        "JOIN files f ON d.file_id = f.file_id\n"
        + "\n".join(joins)
        + "\nWHERE " + " AND ".join(where_clauses)
        + "\nORDER BY f.file_path"
    )
    params.append(from_run)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [row["file_path"] for row in cur.fetchall()]


def get_or_create_file(conn, file_path, file_hash, file_size):
    """Return file_id for file_path, creating the 'files' row on first sight.

    file_hash/file_size_bytes are the file's identity and are set once, on
    first sight, and never overwritten. If a later run sees a different
    hash/size for the same path, the file changed on disk between runs —
    that's logged as an anomaly (it may affect the validity of earlier
    runs' results) rather than silently treated as the new canonical value.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_id, file_hash, file_size_bytes FROM files "
            "WHERE file_path=%s",
            (file_path,),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO files (file_path, file_hash, file_size_bytes) "
                "VALUES (%s,%s,%s)",
                (file_path, file_hash, file_size),
            )
            return cur.lastrowid

        if row["file_hash"] != file_hash or row["file_size_bytes"] != file_size:
            print(
                f"  [ANOMALY] {file_path} differs from its first-seen "
                f"hash/size — the file changed on disk since an earlier "
                f"run processed it. Keeping the original as canonical; "
                f"features from earlier runs may no longer reflect the "
                f"current file contents.",
                file=sys.stderr,
            )
        return row["file_id"]


def upsert_document(conn, run_name, file_id, total_chunks, status="processing", batch_job_id=None):
    """Insert or reset a document_runs row. Returns doc_id.

    status/batch_job_id let docfeatures_batch.py create rows in
    'batch_pending' state tied to a batch_jobs row, instead of the sync
    tool's 'processing'.
    """
    with conn.cursor() as cur:
        # Delete any prior incomplete row (cascade cleans chunks/features)
        cur.execute(
            "DELETE FROM document_runs "
            "WHERE run_name=%s AND file_id=%s AND status != 'complete'",
            (run_name, file_id),
        )
        cur.execute(
            "INSERT INTO document_runs "
            "(run_name, file_id, total_chunks, status, batch_job_id) "
            "VALUES (%s,%s,%s,%s,%s)",
            (run_name, file_id, total_chunks, status, batch_job_id),
        )
        return cur.lastrowid


def save_chunk_result(conn, doc_id, chunk_index, raw_json_str):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chunk_results (doc_id, chunk_index, raw_json) "
            "VALUES (%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE raw_json=VALUES(raw_json)",
            (doc_id, chunk_index, raw_json_str),
        )


def save_document_features(conn, doc_id, file_id, features, features_config):
    """Save only positive/non-default feature values. Skips False booleans,
    the lowest (first) enum option, null text, and null integers.
    Completeness is provable via the document_runs table (status='complete').
    file_id is denormalized here so features for a document can be looked
    up across every run without joining back through document_runs."""

    with conn.cursor() as cur:
        for name, value in features.items():
            fdef = features_config.get(name, {})
            ftype = fdef.get("type", "boolean")

            # Skip false booleans
            if ftype == "boolean" and value is False:
                continue

            # Skip the default (first/lowest) enum value
            if ftype == "enum":
                default_val = fdef.get("options", [""])[0]
                if str(value).lower().strip() == default_val.lower().strip():
                    continue

            # Skip null text values
            if ftype == "text" and (value is None or str(value).strip() == ""):
                continue

            # Skip null integer values
            if ftype == "integer" and value is None:
                continue

            cur.execute(
                "INSERT INTO document_features (doc_id, file_id, feature_name, value_text) "
                "VALUES (%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE value_text=VALUES(value_text)",
                (doc_id, file_id, name, str(value)),
            )


def mark_document(conn, doc_id, status, elapsed=None, error=None):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE document_runs SET status=%s, processing_secs=%s, "
            "error_message=%s WHERE doc_id=%s",
            (status, elapsed, error, doc_id),
        )


def list_runs_db(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT r.run_name, r.description, r.llm_model, r.llm_temperature, r.created_at,
                   COUNT(d.doc_id)                       AS total_docs,
                   SUM(CASE WHEN d.status='complete' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN d.status='error'    THEN 1 ELSE 0 END) AS errors
            FROM runs r
            LEFT JOIN document_runs d ON r.run_name = d.run_name
            GROUP BY r.run_name
            ORDER BY r.created_at DESC
        """)
        return cur.fetchall()


def purge_run_db(conn, run_name):
    with conn.cursor() as cur:
        cur.execute("SELECT run_name FROM runs WHERE run_name=%s", (run_name,))
        if not cur.fetchone():
            print(f"Run '{run_name}' not found.", file=sys.stderr)
            return False
        cur.execute("DELETE FROM runs WHERE run_name=%s", (run_name,))
        print(f"Purged run '{run_name}' and all associated data.")
        return True


# ===========================================================================
# Text Sanitization
# ===========================================================================

# Control characters that are illegal in JSON (and useless to the LLM)
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

# HTML tag pattern (keeps text content, strips markup)
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)

# Collapse runs of whitespace
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def sanitize_text(text):
    """Clean document text for LLM consumption.

    - Strips HTML tags (keeps text content)
    - Removes control characters that break JSON encoding
    - Decodes common HTML entities
    - Normalizes excessive whitespace

    This handles Word-generated HTML, malformed markup, and documents
    with embedded control characters.
    """
    # Strip HTML tags if present (check before expensive regex)
    if "<" in text and ">" in text:
        # Decode common HTML entities first
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&rsquo;", "’")
        text = text.replace("&ldquo;", "“")
        text = text.replace("&rdquo;", "”")
        text = text.replace("&mdash;", "—")
        text = text.replace("&ndash;", "–")
        # Strip HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        # Strip style/script blocks entirely
        text = re.sub(
            r"<(style|script)[^>]*>.*?</\1>", "", text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Strip remaining tags
        text = _HTML_TAG_RE.sub(" ", text)

    # Remove control characters
    text = _CONTROL_CHAR_RE.sub("", text)

    # Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


# ===========================================================================
# Chunking
# ===========================================================================

def split_into_sections(text, target_chars):
    """Split text into sections using cascading strategies.

    Tries each strategy in order; any section still over *target_chars*
    is re-split with the next finer strategy. Final fallback is a hard
    character-boundary split.

    Strategy hierarchy:
      1. HTML headers (<h1>–<h3>)
      2. Markdown headers (# ## ###)
      3. Paragraph breaks (double newline)
      4. Sentence boundaries (after . ! ?)
      5. Single line breaks
      6. Hard split at target_chars (last resort)
    """
    strategies = [
        re.compile(r"(?=<h[1-3][\s>])", re.IGNORECASE),
        re.compile(r"(?=^#{1,3}\s)", re.MULTILINE),
        re.compile(r"\n\s*\n"),
        re.compile(r"(?<=[.!?])\s+"),
        re.compile(r"\n"),
    ]

    sections = [text]

    for pattern in strategies:
        # Stop early if everything already fits
        if all(len(s) <= target_chars for s in sections):
            break

        refined = []
        for section in sections:
            if len(section) <= target_chars:
                refined.append(section)
                continue

            # Attempt to split the oversized section
            parts = pattern.split(section)
            parts = [p for p in parts if p.strip()]

            if len(parts) > 1:
                refined.extend(parts)
            else:
                # Strategy didn't help — pass through for the next one
                refined.append(section)

        sections = refined

    # Final fallback: hard split any remaining oversized sections
    final = []
    for section in sections:
        if len(section) <= target_chars:
            final.append(section)
        else:
            # Split at target_chars, trying to break at a space
            pos = 0
            while pos < len(section):
                end = pos + target_chars
                if end < len(section):
                    # Look back up to 200 chars for a space to break on
                    space = section.rfind(" ", end - 200, end)
                    if space > pos:
                        end = space
                chunk = section[pos:end].strip()
                if chunk:
                    final.append(chunk)
                pos = end

    return final if final else [text]


def build_chunks(text, target_chars=CHUNK_TARGET_CHARS):
    """Pack sections into chunks up to *target_chars*, never splitting
    mid-section. A single oversized section becomes its own chunk."""
    if len(text) <= target_chars:
        return [text]

    sections = split_into_sections(text, target_chars)
    chunks = []
    buf = []
    buf_len = 0

    for sec in sections:
        sec_len = len(sec)
        # Cost of adding this section: its length + 2 for "\n\n" if not first
        add_len = sec_len + (2 if buf else 0)
        if buf and buf_len + add_len > target_chars:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0
            add_len = sec_len  # first in new buffer, no separator

        buf.append(sec)
        buf_len += add_len

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks or [text]


# ===========================================================================
# Prompt Generation
# ===========================================================================

def build_prompt(features_config, text, chunk_info=None, correction=None):
    """Assemble the extraction prompt from feature definitions + document.

    correction, if given (see build_correction_note()), is appended just
    before the final "JSON output:" cue -- describes a previous invalid
    attempt so a retry can ask the model to fix that specific problem
    instead of resending an identical prompt and hoping sampling gives a
    different answer.
    """
    parts = [
        "You are a clinical document analyst. Given the document text below, "
        "extract the requested features.",
        "",
        "Respond with ONLY a valid JSON object - no explanation, no markdown "
        "fencing, no commentary, no additional text whatsoever.",
        "",
    ]

    if chunk_info:
        idx, total = chunk_info
        parts.append(
            f"NOTE: This is section {idx} of {total} from a larger document. "
            "Evaluate features for THIS section only."
        )
        parts.append("")

    parts.append("Features to extract:")
    parts.append("")

    for name, fdef in features_config.items():
        ftype = fdef.get("type", "boolean")
        if ftype == "boolean":
            hint = "respond with true or false"
        elif ftype == "enum":
            opts = ", ".join(fdef["options"])
            hint = f"respond with exactly one of: {opts}"
        elif ftype == "text":
            max_len = fdef.get("max_length")
            if max_len:
                hint = f"respond with a text string of at most {max_len} characters, or null if not found"
            else:
                hint = "respond with a text string, or null if not found"
        elif ftype == "integer":
            hint = "respond with an integer, or null if not applicable"
        else:
            hint = "respond with true or false"

        desc = fdef.get("description", "").strip()
        parts.append(f"- {name} ({hint})")
        if desc:
            parts.append(f"  {desc}")
        parts.append("")

    parts += ["Document text:", "---", text, "---", ""]
    if correction:
        parts.append(correction)
        parts.append("")
    parts.append("JSON output:")
    return "\n".join(parts)


def build_correction_note(previous_response_text, error_message):
    """Build the build_prompt(correction=...) addendum for a retry: what the
    model answered last time, and specifically why it was rejected. Works
    regardless of temperature -- unlike a blind reroll (which only has a
    chance of a different answer when sampling is stochastic), this changes
    the input itself, so it's a meaningful retry even at temperature 0.

    previous_response_text is a plain string, not a dict -- callers pass
    json.dumps(parsed) for a parsed-but-invalid response, or a raw text
    excerpt if the previous attempt didn't produce parseable JSON at all
    (parse_json_response failures are retryable through the same path).
    """
    return (
        f"Your previous response was: {previous_response_text}\n"
        f"This was invalid: {error_message}\n"
        f"Provide a corrected JSON response following the instructions above."
    )


# ===========================================================================
# Response Parsing
# ===========================================================================

def parse_json_response(raw):
    """Extract a JSON object (dict) from the LLM response, tolerating
    markdown fences, chain-of-thought preamble, and other wrapping.

    Raises ValueError with diagnostic detail if parsing fails.
    """
    if raw is None:
        raise ValueError("LLM returned None (empty response).")

    # Strip markdown code fences
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)

    def _validate(obj):
        """Ensure the parsed JSON is a dict, not null/list/string."""
        if obj is None:
            raise ValueError(
                "LLM returned JSON null instead of an object. "
                "The model may have failed to extract features from "
                "this document."
            )
        if not isinstance(obj, dict):
            raise ValueError(
                f"LLM returned JSON {type(obj).__name__} instead of "
                f"an object: {str(obj)[:200]}"
            )
        return obj

    # Direct parse
    try:
        return _validate(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # Greedy search for outermost { ... }
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return _validate(json.loads(cleaned[start : i + 1]))
                except json.JSONDecodeError:
                    start = None

    # Attempt to repair truncated JSON (response hit token limit)
    if start is not None and depth > 0:
        # We found an opening { but never closed it — try closing it
        fragment = cleaned[start:]
        # Close any open strings, then close braces
        repair = fragment.rstrip()
        if repair.endswith(","):
            repair = repair[:-1]
        # Close any open string
        if repair.count('"') % 2 == 1:
            repair += '"'
        # Close braces
        repair += "}" * depth
        try:
            return _validate(json.loads(repair))
        except json.JSONDecodeError:
            pass

    # Build a diagnostic message
    preview = raw[:500]
    if len(raw) > 500:
        preview += f"\n... ({len(raw)} chars total)"
    raise ValueError(f"Could not parse JSON from LLM response:\n{raw}")


# ===========================================================================
# Feature Merging (across chunks)
# ===========================================================================

def validate_enum_values(parsed, features_config, chunk_info=None):
    """Raise ValueError if any enum feature in a single chunk's parsed
    response isn't one of its declared options (case-insensitive). LLMs
    occasionally hallucinate an option that was never offered; catching it
    here fails the document fast (before spending more LLM calls on it)
    instead of letting merge_chunk_results silently store the made-up
    value. Booleans/text/integers aren't validated -- this hasn't been an
    observed problem for those types.
    """
    where = f" (chunk {chunk_info[0]}/{chunk_info[1]})" if chunk_info else ""
    for name, fdef in features_config.items():
        if fdef.get("type", "boolean") != "enum" or name not in parsed:
            continue
        value = parsed[name]
        options = fdef.get("options", [])
        if str(value).strip().lower() not in [o.lower() for o in options]:
            raise ValueError(
                f"Feature '{name}'{where}: LLM returned {value!r}, which is "
                f"not one of the declared options {options}"
            )


def merge_chunk_results(chunk_jsons, features_config):
    """Combine per-chunk extractions into a single feature dict.

    - boolean:  OR (any chunk True → document True)
    - enum:     MAX by option-list position (later = stronger)
    - text:     configurable via 'strategy': last-chunk (default),
                first-chunk, or concatenate
    - integer:  MAX of non-null values; null if all chunks are null
    """
    merged = {}

    for name, fdef in features_config.items():
        ftype = fdef.get("type", "boolean")
        values = [cj[name] for cj in chunk_jsons if name in cj]

        if not values:
            if ftype == "boolean":
                merged[name] = False
            elif ftype == "enum":
                merged[name] = fdef.get("options", ["unknown"])[0]
            elif ftype in ("text", "integer"):
                merged[name] = None
            else:
                merged[name] = False
            continue

        if ftype == "boolean":
            merged[name] = any(
                v is True or (isinstance(v, str) and v.lower() == "true")
                for v in values
            )

        elif ftype == "enum":
            options = [o.lower() for o in fdef.get("options", [])]
            best_idx = -1
            best_val = None
            for v in values:
                v_lower = str(v).lower().strip()
                if v_lower in options:
                    idx = options.index(v_lower)
                    if idx > best_idx:
                        best_idx = idx
                        best_val = fdef["options"][idx]
            if best_val is None:
                # Every chunk's value was outside the declared options.
                # validate_enum_values() should already have caught this
                # per-chunk and aborted the document; this is a safety net,
                # not the primary check -- never silently store a made-up
                # value.
                raise ValueError(
                    f"Feature '{name}': none of the LLM's returned values "
                    f"{values!r} match the declared options {fdef.get('options', [])}"
                )
            merged[name] = best_val

        elif ftype == "text":
            strategy = fdef.get("strategy", "last-chunk")
            # Filter out null / None / empty / "not found" values
            non_empty = [
                str(v) for v in values
                if v is not None
                and str(v).strip() != ""
                and str(v).strip().lower() not in ("null", "not found", "n/a", "none")
            ]
            if not non_empty:
                merged[name] = None
            elif strategy == "first-chunk":
                merged[name] = non_empty[0]
            elif strategy == "concatenate":
                merged[name] = " ".join(non_empty)
            else:  # last-chunk (default)
                merged[name] = non_empty[-1]

        elif ftype == "integer":
            # Parse to int, skip nulls
            int_values = []
            for v in values:
                if v is None or str(v).strip().lower() in ("null", "none", "n/a", "-1"):
                    continue
                try:
                    int_values.append(int(float(str(v))))
                except (ValueError, TypeError):
                    continue
            merged[name] = max(int_values) if int_values else None

        else:
            merged[name] = values[0]

    return merged


# ===========================================================================
# File Discovery
# ===========================================================================

def discover_files(corpus_paths):
    """Yield deduplicated paths of text files under one or more corpus paths.
    *corpus_paths* can be a single string/Path or a list of them.
    Each entry can be a directory (searched recursively) or a single file.
    Files are deduplicated by resolved path so overlapping directories
    don't cause duplicate processing.
    """
    if isinstance(corpus_paths, (str, Path)):
        corpus_paths = [corpus_paths]

    seen = set()
    for cp in corpus_paths:
        root = Path(cp)
        if root.is_file():
            resolved = str(root.resolve())
            if resolved not in seen:
                seen.add(resolved)
                yield str(root)
            continue
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS:
                resolved = str(f.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    yield str(f)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ===========================================================================
# Formatting helpers
# ===========================================================================

def fmt_feature_value(v):
    """Short display string for a feature value. Shared by docfeatures.py's
    per-document progress line and docfeatures_batch.py's `import --verbose`."""
    if isinstance(v, bool):
        return "Y" if v else "n"
    if v is None:
        return "–"
    s = str(v)
    if len(s) > 40:
        return s[:37] + "..."
    return s


# ===========================================================================
# Config loading + validation
# ===========================================================================

def load_and_validate_config(path):
    """Load a YAML feature-config file and validate its structure.

    Shared by docfeatures.py and docfeatures_batch.py so both tools accept
    (and reject) the same configs the same way. Raises ValueError with a
    human-readable message on any problem; callers turn that into
    parser.error() or similar.
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    if "features" not in config or not config["features"]:
        raise ValueError("Config must contain a 'features' section with at least one feature.")

    has_filter = "filter" in config and config["filter"]
    if has_filter:
        fc = config["filter"]
        if not fc.get("from_run"):
            raise ValueError("Filter section must include 'from_run'.")
        if not fc.get("require") and not fc.get("exclude"):
            raise ValueError(
                "Filter section must include at least one 'require' or 'exclude' entry."
            )

    valid_types = ("boolean", "enum", "text", "integer")
    valid_strategies = ("first-chunk", "last-chunk", "concatenate")
    for name, fdef in config["features"].items():
        ftype = fdef.get("type", "boolean")
        if ftype not in valid_types:
            raise ValueError(
                f"Feature '{name}': unsupported type '{ftype}'. "
                f"Use one of: {', '.join(valid_types)}."
            )
        if ftype == "enum" and not fdef.get("options"):
            raise ValueError(f"Feature '{name}': enum type requires an 'options' list.")
        if ftype == "text":
            strategy = fdef.get("strategy", "last-chunk")
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Feature '{name}': unsupported strategy '{strategy}'. "
                    f"Use one of: {', '.join(valid_strategies)}."
                )

    return config
