#!/usr/bin/env python3
"""
docfeatures.py — Document Feature Identification Tool

Scans a corpus of text documents using a local LLM (via OpenAI-compatible API)
to identify researcher-defined features. Results are stored in MySQL.

Setup:
    pip install pymysql pyyaml requests python-dotenv
    cp .env.example .env   # edit with your DB credentials

Usage:
    python docfeatures.py --config features.yaml --corpus /data/notes/ --run-name v1 --limit 10
    python docfeatures.py --config features.yaml --corpus /data/notes/ --run-name v1
    python docfeatures.py --list-runs
    python docfeatures.py --purge-run v1
"""

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import pymysql
import requests
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
CHUNK_TARGET_CHARS = 350_000
DEFAULT_LLM_HOST = "http://192.168.86.33:11433"
DEFAULT_LLM_MODEL = "qwen3.5:35b"
TEXT_EXTENSIONS = {".txt", ".html", ".htm", ".md", ".text"}

# ---------------------------------------------------------------------------
# Graceful Ctrl+C
# ---------------------------------------------------------------------------
_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    if _interrupted:
        sys.exit(1)  # second Ctrl+C = immediate
    _interrupted = True
    print("\n[Ctrl+C] Finishing current document, then stopping...", file=sys.stderr)


signal.signal(signal.SIGINT, _handle_sigint)


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

def get_or_create_run(conn, run_name, config, config_hash, host, model):
    with conn.cursor() as cur:
        cur.execute("SELECT run_name FROM runs WHERE run_name = %s", (run_name,))
        if cur.fetchone():
            return
        desc = config.get("run_description", "")
        cur.execute(
            "INSERT INTO runs (run_name, config_hash, config_yaml, "
            "description, llm_host, llm_model) VALUES (%s,%s,%s,%s,%s,%s)",
            (run_name, config_hash, yaml.dump(config), desc, host, model),
        )


def cleanup_incomplete(conn, run_name):
    """Remove docs stuck in 'processing' (interrupted mid-flight)."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM documents WHERE run_name=%s AND status='processing'",
            (run_name,),
        )
        if cur.rowcount:
            print(
                f"  Cleaned up {cur.rowcount} interrupted document(s) "
                "from previous session.",
                file=sys.stderr,
            )


def get_finished_paths(conn, run_name, include_errors=False):
    """Return set of file_paths already finished for this run."""
    statuses = "('complete','error')" if not include_errors else "('complete')"
    # We skip 'complete' always; we skip 'error' unless retrying
    with conn.cursor() as cur:
        if include_errors:
            # retrying errors — only skip 'complete'
            cur.execute(
                "SELECT file_path FROM documents "
                "WHERE run_name=%s AND status='complete'",
                (run_name,),
            )
        else:
            # default — skip both 'complete' and 'error'
            cur.execute(
                "SELECT file_path FROM documents "
                "WHERE run_name=%s AND status IN ('complete','error')",
                (run_name,),
            )
        return {row["file_path"] for row in cur.fetchall()}


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
            "SELECT COUNT(*) AS cnt FROM documents "
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
    # d = source documents table
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
        "SELECT d.file_path FROM documents d\n"
        + "\n".join(joins)
        + "\nWHERE " + " AND ".join(where_clauses)
        + "\nORDER BY d.file_path"
    )
    params.append(from_run)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [row["file_path"] for row in cur.fetchall()]


def upsert_document(conn, run_name, file_path, file_hash, file_size, total_chunks):
    """Insert or reset a document row. Returns doc_id."""
    with conn.cursor() as cur:
        # Delete any prior incomplete row (cascade cleans chunks/features)
        cur.execute(
            "DELETE FROM documents "
            "WHERE run_name=%s AND file_path=%s AND status != 'complete'",
            (run_name, file_path),
        )
        cur.execute(
            "INSERT INTO documents "
            "(run_name, file_path, file_hash, file_size_bytes, total_chunks, status) "
            "VALUES (%s,%s,%s,%s,%s,'processing')",
            (run_name, file_path, file_hash, file_size, total_chunks),
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


def save_document_features(conn, doc_id, features, features_config):
    """Save only positive/non-default feature values. Skips False booleans,
    the lowest (first) enum option, null text, and null integers.
    Completeness is provable via the documents table (status='complete')."""
										   
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
                "INSERT INTO document_features (doc_id, feature_name, value_text) "
                "VALUES (%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE value_text=VALUES(value_text)",
                (doc_id, name, str(value)),
            )


def mark_document(conn, doc_id, status, elapsed=None, error=None):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET status=%s, processing_secs=%s, "
            "error_message=%s WHERE doc_id=%s",
            (status, elapsed, error, doc_id),
        )


def list_runs_db(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT r.run_name, r.description, r.llm_model, r.created_at,
                   COUNT(d.doc_id)                       AS total_docs,
                   SUM(CASE WHEN d.status='complete' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN d.status='error'    THEN 1 ELSE 0 END) AS errors
            FROM runs r
            LEFT JOIN documents d ON r.run_name = d.run_name
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
        text = text.replace("&rsquo;", "\u2019")
        text = text.replace("&ldquo;", "\u201c")
        text = text.replace("&rdquo;", "\u201d")
        text = text.replace("&mdash;", "\u2014")
        text = text.replace("&ndash;", "\u2013")
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

def build_prompt(features_config, text, chunk_info=None):
    """Assemble the extraction prompt from feature definitions + document."""
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

    parts += ["Document text:", "---", text, "---", "", "JSON output:"]
    return "\n".join(parts)


# ===========================================================================
# LLM Interaction
# ===========================================================================

# Retry configuration
RETRY_DELAY_SECS = 15           # wait between retries on 503 / transient errors
RETRY_MAX_ATTEMPTS = 12         # give up after ~3 minutes of retries
RETRY_HTTP_CODES = {502, 503}   # codes that trigger a retry
HALT_ON_CONN_FAILURE = False    # Connection failure = exit script vs. just wait


class LLMServerDead(Exception):
    """Raised when the LLM server is unreachable (connection refused)."""
    pass


def call_llm(host, model, prompt):
    """Send prompt to llama-server with retry on transient errors.

    - 502/503: server restarting → retry up to RETRY_MAX_ATTEMPTS
    - ConnectionError: server dead → raise LLMServerDead immediately
    - Other HTTP errors: raise normally (per-document error)
    """
    url = f"{host.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        # "max_completion_tokens": 2048,
    }

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        if _interrupted:
            raise KeyboardInterrupt

        try:
            resp = requests.post(url, json=payload, timeout=600)
        except (requests.ConnectionError, requests.exceptions.ConnectionError) as e:
          if HALT_ON_CONN_FAILURE is False and attempt < RETRY_MAX_ATTEMPTS:
              print(
                f"  [RETRY {attempt}/{RETRY_MAX_ATTEMPTS}] "
                f"Server returned connection error, "
                f"waiting {RETRY_DELAY_SECS}s...",
                file=sys.stderr,
              )
              time.sleep(RETRY_DELAY_SECS)
              continue
          else:										  
            raise LLMServerDead(
                f"Cannot connect to LLM server at {host} — server may be down. "
                f"({e})"
            ) from e

        if resp.status_code not in RETRY_HTTP_CODES:
            break

        # Transient error — wait and retry
        if attempt < RETRY_MAX_ATTEMPTS:
            print(
                f"  [RETRY {attempt}/{RETRY_MAX_ATTEMPTS}] "
                f"Server returned {resp.status_code}, "
                f"waiting {RETRY_DELAY_SECS}s...",
                file=sys.stderr,
            )
            time.sleep(RETRY_DELAY_SECS)
        else:
            raise LLMServerDead(
                f"Server returned {resp.status_code} after "
                f"{RETRY_MAX_ATTEMPTS} retries (~{RETRY_MAX_ATTEMPTS * RETRY_DELAY_SECS}s). "
                f"Halting run."
            )

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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
            best_val = str(values[0])
            for v in values:
                v_lower = str(v).lower().strip()
                if v_lower in options:
                    idx = options.index(v_lower)
                    if idx > best_idx:
                        best_idx = idx
                        best_val = fdef["options"][idx]
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

def fmt_duration(secs):
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{int(secs)//60}m {int(secs)%60}s"
    h = int(secs) // 3600
    m = (int(secs) % 3600) // 60
    return f"{h}h {m}m"


def fmt_feature_value(v):
    """Short display string for a feature value."""
    if isinstance(v, bool):
        return "Y" if v else "n"
    if v is None:
        return "–"
    s = str(v)
    if len(s) > 40:
        return s[:37] + "..."
    return s


# ===========================================================================
# Main processing loop
# ===========================================================================

def process_corpus(args, config):
    features_config = config["features"]
    host = args.host or config.get("llm", {}).get("host", DEFAULT_LLM_HOST)
    model = args.model or config.get("llm", {}).get("model", DEFAULT_LLM_MODEL)

    conn = get_connection()

    config_hash = hashlib.sha256(yaml.dump(config).encode()).hexdigest()
    get_or_create_run(conn, args.run_name, config, config_hash, host, model)
    cleanup_incomplete(conn, args.run_name)

    skip_errors = not args.retry_errors
    finished = get_finished_paths(conn, args.run_name, include_errors=skip_errors)

    # --- Resolve corpus paths: CLI overrides YAML ---
    corpus_paths = args.corpus  # list or None (from action="append")
    if not corpus_paths:
        # Fall back to config YAML
        yaml_corpus = config.get("corpus", [])
        if isinstance(yaml_corpus, str):
            yaml_corpus = [yaml_corpus]
        corpus_paths = yaml_corpus if yaml_corpus else None

    # --- File discovery: filter mode vs. corpus mode ---
    filter_config = config.get("filter")
    if filter_config:
        validate_filter(conn, filter_config)
        all_files = get_filtered_paths(conn, filter_config)
        source_label = f"filter from run '{filter_config['from_run']}'"

        # If corpus also specified, intersect with filesystem
        if corpus_paths:
            corpus_files = set(discover_files(corpus_paths))
            all_files = [f for f in all_files if f in corpus_files]
            source_label += f" ∩ [{', '.join(corpus_paths)}]"
    else:
        all_files = list(discover_files(corpus_paths))
        if len(corpus_paths) == 1:
            source_label = f"{corpus_paths[0]}  ({len(all_files)} files)"
        else:
            source_label = (
                f"{len(corpus_paths)} paths  ({len(all_files)} files)\n"
                + "".join(f"               {p}\n" for p in corpus_paths)
            ).rstrip()
    pending = [f for f in all_files if f not in finished]

    print(f"Run          : {args.run_name}", file=sys.stderr)
    print(f"Config       : {args.config}", file=sys.stderr)
    print(f"Source       : {source_label}", file=sys.stderr)
    if filter_config:
        fc = filter_config
        print(f"  from_run   : {fc['from_run']}", file=sys.stderr)
        if fc.get("require"):
            for k, v in fc["require"].items():
                print(f"  require    : {k} = {v}", file=sys.stderr)
        if fc.get("exclude"):
            for k, v in fc["exclude"].items():
                print(f"  exclude    : {k} = {v}", file=sys.stderr)
    print(f"Matched      : {len(all_files)}", file=sys.stderr)
    print(f"Already done : {len(finished)}", file=sys.stderr)
    print(f"Pending      : {len(pending)}", file=sys.stderr)
    print(f"LLM          : {model} @ {host}", file=sys.stderr)
    if args.limit:
        pending = pending[: args.limit]
        print(f"Batch limit  : {args.limit}", file=sys.stderr)
    if args.cooldown and args.cooldown > 0:
        print(f"Cooldown     : {args.cooldown}s between documents", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    session_start = time.time()
    processed = 0
    errors = 0
    total_chunks = 0

    for file_path in pending:
        if _interrupted:
            break

        doc_start = time.time()
        doc_id = None
        try:
            raw_bytes = Path(file_path).read_bytes()
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = raw_bytes.decode("cp1252")
                except:
                    text = raw_bytes.decode("utf-8", errors="replace")
            fhash = file_hash(file_path)
            fsize = os.path.getsize(file_path)

            # Chunk on raw text (preserves HTML structure for splitting),
            # then sanitize each chunk individually before sending to LLM
            chunks = build_chunks(text, target_chars=args.chunk_size)
            num_chunks = len(chunks)

            doc_id = upsert_document(
                conn, args.run_name, file_path, fhash, fsize, num_chunks
            )

            chunk_results_list = []
            for ci, chunk_text in enumerate(chunks):
                if _interrupted:
                    break
                clean_chunk = sanitize_text(chunk_text)
                if not clean_chunk:
                    continue  # skip empty chunks (e.g., pure HTML boilerplate)
                chunk_info = (ci + 1, num_chunks) if num_chunks > 1 else None
                prompt = build_prompt(features_config, clean_chunk, chunk_info)
                raw = call_llm(host, model, prompt)
                parsed = parse_json_response(raw)
                save_chunk_result(conn, doc_id, ci, json.dumps(parsed))
                chunk_results_list.append(parsed)

            if _interrupted:
                # leave as 'processing'; cleanup_incomplete will handle next run
                break

            merged = merge_chunk_results(chunk_results_list, features_config)
            save_document_features(conn, doc_id, merged, features_config)

            elapsed = time.time() - doc_start
            mark_document(conn, doc_id, "complete", elapsed=elapsed)

            processed += 1
            total_chunks += num_chunks

            feat_str = "  ".join(
                f"{k}={fmt_feature_value(v)}" for k, v in merged.items()
            )
            print(
                f"  [{processed}/{len(pending)}] {Path(file_path).name}  "
                f"({num_chunks} chunk{'s' if num_chunks != 1 else ''}, "
                f"{elapsed:.1f}s)  {feat_str}",
                file=sys.stderr,
            )

            # Cooldown pause to mitigate thermal throttling on compact hardware
            if args.cooldown and args.cooldown > 0 and not _interrupted:
                time.sleep(args.cooldown)

        except LLMServerDead as e:
            elapsed = time.time() - doc_start
            errors += 1
            if doc_id:
                mark_document(conn, doc_id, "error", elapsed=elapsed, error=str(e))
            print(f"\n  [FATAL] {e}", file=sys.stderr)
            print("  Halting run. Resume with the same --run-name once "
                  "the server is back.", file=sys.stderr)
            break

        except Exception as e:
            elapsed = time.time() - doc_start
            errors += 1
            if doc_id:
                mark_document(conn, doc_id, "error", elapsed=elapsed, error=str(e))
            print(f"  [ERROR] {Path(file_path).name}: {e}", file=sys.stderr)

    # ---- Session summary ----
    session_elapsed = time.time() - session_start
    total_done = len(finished) + processed
    remaining = len(all_files) - total_done

    print("", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f'Run "{args.run_name}"', file=sys.stderr)
    print(f"  Processed this session : {processed}", file=sys.stderr)
    print(f"  Previously completed   : {len(finished)}", file=sys.stderr)
    print(f"  Total complete         : {total_done} / {len(all_files)}", file=sys.stderr)
    print(f"  Errors this session    : {errors}", file=sys.stderr)
    print(
        f"  Chunks this session    : {total_chunks} "
        f"({total_chunks / max(processed, 1):.1f}/doc avg)",
        file=sys.stderr,
    )
    if processed > 0:
        avg = session_elapsed / processed
        print(f"  Elapsed                : {fmt_duration(session_elapsed)}", file=sys.stderr)
        print(f"  Avg per document       : {avg:.1f}s", file=sys.stderr)
        if remaining > 0:
            print(
                f"  Est. remaining         : {fmt_duration(remaining * avg)} "
                f"({remaining} docs)",
                file=sys.stderr,
            )
    elif session_elapsed > 0:
        print(f"  Elapsed                : {fmt_duration(session_elapsed)}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    conn.close()


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Document Feature Identification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Test run on 10 docs
  %(prog)s -c features.yaml --corpus /data/notes/ -r lung_v1 -n 10

  # Full run (resumes automatically)
  %(prog)s -c features.yaml --corpus /data/notes/ -r lung_v1

  # Multiple corpus directories
  %(prog)s -c features.yaml --corpus /data/2023/ --corpus /data/2024/ -r lung_v1

  # Corpus paths in YAML config (no --corpus needed)
  %(prog)s -c features.yaml -r lung_v1

  # CLI --corpus overrides YAML corpus paths
  %(prog)s -c features.yaml --corpus /data/subset/ -r lung_test

  # Filtered run — config YAML contains a 'filter' section
  %(prog)s -c lung_details.yaml -r lung_details_v1 -n 10

  # Retry documents that errored
  %(prog)s -c features.yaml --corpus /data/notes/ -r lung_v1 --retry-errors

  # List all runs
  %(prog)s --list-runs

  # Delete a test run
  %(prog)s --purge-run lung_v1
        """,
    )

    # Processing arguments
    parser.add_argument("-c", "--config", help="YAML feature-definition file.")
    parser.add_argument(
        "--corpus", action="append", default=None,
        help="Path to document directory or file. Can be specified multiple "
             "times. Overrides corpus paths in the YAML config if provided.",
    )
    parser.add_argument("-r", "--run-name", help="Name for this run (used for resume).")
    parser.add_argument(
        "-n", "--limit", type=int, help="Stop after N documents."
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-process documents that errored in a previous session.",
    )
    parser.add_argument(
        "--cooldown", type=float, default=0, metavar="SECS",
        help="Pause N seconds between documents to reduce thermal load. "
             "Recommended: 3-5s for DGX Spark or similar compact hardware.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_TARGET_CHARS, metavar="CHARS",
        help=f"Max characters per chunk (default: {CHUNK_TARGET_CHARS:,}). "
             "Reduce for models with smaller context windows. "
             "Rule of thumb: context_tokens × 3 for clinical text.",
    )

    # LLM overrides (take precedence over config file)
    parser.add_argument(
        "--host", help=f"LLM server URL (default: from config or {DEFAULT_LLM_HOST})"
    )
    parser.add_argument(
        "-m", "--model",
        help=f"Model name (default: from config or '{DEFAULT_LLM_MODEL}')",
    )

    # Management commands
    parser.add_argument(
        "--list-runs", action="store_true", help="Show all runs in the database."
    )
    parser.add_argument(
        "--purge-run", metavar="NAME", help="Delete a run and all its results."
    )

    args = parser.parse_args()

    # ---- list-runs ----
    if args.list_runs:
        conn = get_connection()
        runs = list_runs_db(conn)
        conn.close()
        if not runs:
            print("No runs found.")
            return
        print(
            f"{'Run Name':<30} {'Model':<20} "
            f"{'Done':>6} {'Errs':>5} {'Total':>6}  Created"
        )
        print("-" * 100)
        for r in runs:
            print(
                f"{r['run_name']:<30} {(r['llm_model'] or '?'):<20} "
                f"{r['completed'] or 0:>6} {r['errors'] or 0:>5} "
                f"{r['total_docs']:>6}  {r['created_at']}"
            )
        return

    # ---- purge-run ----
    if args.purge_run:
        conn = get_connection()
        confirm = input(f"Delete run '{args.purge_run}' and all associated data? [y/N] ")
        if confirm.strip().lower() == "y":
            purge_run_db(conn, args.purge_run)
        else:
            print("Cancelled.")
        conn.close()
        return

    # ---- processing mode ----
    if not args.config or not args.run_name:
        parser.error("--config and --run-name are required for processing.")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if "features" not in config or not config["features"]:
        parser.error("Config must contain a 'features' section with at least one feature.")

    # --corpus is required unless the config has corpus paths or a filter section
    has_filter = "filter" in config and config["filter"]
    has_yaml_corpus = bool(config.get("corpus"))
    has_cli_corpus = bool(args.corpus)
    if not has_cli_corpus and not has_yaml_corpus and not has_filter:
        parser.error(
            "No corpus specified. Provide --corpus on the command line, "
            "'corpus' in the YAML config, or a 'filter' section."
        )
    # Validate filter section if present
    if has_filter:
        fc = config["filter"]
        if not fc.get("from_run"):
            parser.error("Filter section must include 'from_run'.")
        if not fc.get("require") and not fc.get("exclude"):
            parser.error(
                "Filter section must include at least one 'require' or 'exclude' entry."
            )

    # Validate feature definitions
    valid_types = ("boolean", "enum", "text", "integer")
    valid_strategies = ("first-chunk", "last-chunk", "concatenate")
    for name, fdef in config["features"].items():
        ftype = fdef.get("type", "boolean")
        if ftype not in valid_types:
            parser.error(
                f"Feature '{name}': unsupported type '{ftype}'. "
                f"Use one of: {', '.join(valid_types)}."
            )
        if ftype == "enum" and not fdef.get("options"):
            parser.error(
                f"Feature '{name}': enum type requires an 'options' list."
            )
        if ftype == "text":
            strategy = fdef.get("strategy", "last-chunk")
            if strategy not in valid_strategies:
                parser.error(
                    f"Feature '{name}': unsupported strategy '{strategy}'. "
                    f"Use one of: {', '.join(valid_strategies)}."
                )

    process_corpus(args, config)


if __name__ == "__main__":
    main()
