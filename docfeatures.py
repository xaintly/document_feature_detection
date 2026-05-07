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
# CHUNK_TARGET_CHARS = 40_000  # ~10k tokens
CHUNK_TARGET_CHARS = 400_000  # ~100k tokens
DEFAULT_LLM_HOST = "http://localhost:8080"
DEFAULT_LLM_MODEL = "default"
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


def init_db(conn):
    """Create tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_name        VARCHAR(64) PRIMARY KEY,
                config_hash     CHAR(64),
                config_yaml     MEDIUMTEXT,
                description     TEXT,
                llm_host        VARCHAR(512),
                llm_model       VARCHAR(255),
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                          ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id          INT AUTO_INCREMENT PRIMARY KEY,
                run_name        VARCHAR(64) NOT NULL,
                file_path       VARCHAR(256) NOT NULL,
                file_hash       CHAR(64),
                file_size_bytes INT,
                total_chunks    INT DEFAULT 1,
                status          ENUM('processing','complete','error')
                                    DEFAULT 'processing',
                error_message   TEXT,
                processing_secs FLOAT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_run_path (run_name, file_path),
                FOREIGN KEY (run_name) REFERENCES runs(run_name)
                    ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunk_results (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                doc_id          INT NOT NULL,
                chunk_index     INT NOT NULL,
                raw_json        MEDIUMTEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_chunk (doc_id, chunk_index),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                    ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_features (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                doc_id          INT NOT NULL,
                feature_name    VARCHAR(255) NOT NULL,
                value_text      VARCHAR(1024),
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_feature (doc_id, feature_name),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                    ON DELETE CASCADE
            )
        """)


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
    """Save only positive/non-default feature values. Skips False booleans
    and the lowest (first) enum option. Completeness is provable via the
    documents table (status='complete')."""
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
# Chunking
# ===========================================================================

def split_into_sections(text):
    """Split text at the best available structural delimiter."""
    # HTML headers
    if re.search(r"<h[1-3][\s>]", text, re.IGNORECASE):
        parts = re.split(r"(?=<h[1-3][\s>])", text, flags=re.IGNORECASE)
        return [p for p in parts if p.strip()]

    # Markdown headers
    if re.search(r"^#{1,3}\s", text, re.MULTILINE):
        parts = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
        return [p for p in parts if p.strip()]

    # Paragraph breaks (double newline)
    parts = re.split(r"\n\s*\n", text)
    return [p for p in parts if p.strip()]


def build_chunks(text, target_chars=CHUNK_TARGET_CHARS):
    """Pack sections into chunks up to *target_chars*, never splitting
    mid-section. A single oversized section becomes its own chunk."""
    if len(text) <= target_chars:
        return [text]

    sections = split_into_sections(text)
    chunks = []
    buf = []
    buf_len = 0

    for sec in sections:
        sec_len = len(sec)
        if buf and buf_len + sec_len > target_chars:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0
        buf.append(sec)
        buf_len += sec_len

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
        "identify whether each listed feature is present.",
        "",
        "Respond with ONLY a valid JSON object — no explanation, no markdown "
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

    parts.append("Features to identify:")
    parts.append("")

    for name, fdef in features_config.items():
        ftype = fdef.get("type", "boolean")
        if ftype == "boolean":
            hint = "respond with true or false"
        elif ftype == "enum":
            opts = ", ".join(fdef["options"])
            hint = f"respond with exactly one of: {opts}"
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

def call_llm(host, model, prompt):
    """Send prompt to llama-server (OpenAI-compatible chat completions)."""
    url = f"{host.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(raw):
    """Extract a JSON object from the LLM response, tolerating markdown
    fences, chain-of-thought preamble, and other wrapping."""
    # Strip markdown code fences
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)

    # Direct parse
    try:
        return json.loads(cleaned)
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
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")


# ===========================================================================
# Feature Merging (across chunks)
# ===========================================================================

def merge_chunk_results(chunk_jsons, features_config):
    """Combine per-chunk extractions into a single feature dict.

    - boolean: OR (any chunk True → document True)
    - enum:    MAX by option-list position (later = stronger)
    """
    merged = {}

    for name, fdef in features_config.items():
        ftype = fdef.get("type", "boolean")
        values = [cj[name] for cj in chunk_jsons if name in cj]

        if not values:
            merged[name] = (
                False if ftype == "boolean"
                else fdef.get("options", ["unknown"])[0]
            )
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

        else:
            merged[name] = values[0]

    return merged


# ===========================================================================
# File Discovery
# ===========================================================================

def discover_files(corpus_path):
    """Yield paths of text files under *corpus_path* (recursively)."""
    root = Path(corpus_path)
    if root.is_file():
        yield str(root)
        return
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS:
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
    return str(v)


# ===========================================================================
# Main processing loop
# ===========================================================================

def process_corpus(args, config):
    features_config = config["features"]
    host = args.host or config.get("llm", {}).get("host", DEFAULT_LLM_HOST)
    model = args.model or config.get("llm", {}).get("model", DEFAULT_LLM_MODEL)

    conn = get_connection()
    init_db(conn)

    config_hash = hashlib.sha256(yaml.dump(config).encode()).hexdigest()
    get_or_create_run(conn, args.run_name, config, config_hash, host, model)
    cleanup_incomplete(conn, args.run_name)

    skip_errors = not args.retry_errors
    finished = get_finished_paths(conn, args.run_name, include_errors=skip_errors)

    all_files = list(discover_files(args.corpus))
    pending = [f for f in all_files if f not in finished]

    print(f"Run          : {args.run_name}", file=sys.stderr)
    print(f"Config       : {args.config}", file=sys.stderr)
    print(f"Corpus       : {args.corpus}  ({len(all_files)} files)", file=sys.stderr)
    print(f"Already done : {len(finished)}", file=sys.stderr)
    print(f"Pending      : {len(pending)}", file=sys.stderr)
    print(f"LLM          : {model} @ {host}", file=sys.stderr)
    if args.limit:
        pending = pending[: args.limit]
        print(f"Batch limit  : {args.limit}", file=sys.stderr)
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
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
            fhash = file_hash(file_path)
            fsize = os.path.getsize(file_path)

            chunks = build_chunks(text)
            num_chunks = len(chunks)

            doc_id = upsert_document(
                conn, args.run_name, file_path, fhash, fsize, num_chunks
            )

            chunk_results_list = []
            for ci, chunk_text in enumerate(chunks):
                if _interrupted:
                    break
                chunk_info = (ci + 1, num_chunks) if num_chunks > 1 else None
                prompt = build_prompt(features_config, chunk_text, chunk_info)
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
    parser.add_argument("--corpus", help="Path to document directory (or single file).")
    parser.add_argument("-r", "--run-name", help="Name for this run (used for resume).")
    parser.add_argument(
        "-n", "--limit", type=int, help="Stop after N documents."
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-process documents that errored in a previous session.",
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
        init_db(conn)
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
        init_db(conn)
        confirm = input(f"Delete run '{args.purge_run}' and all associated data? [y/N] ")
        if confirm.strip().lower() == "y":
            purge_run_db(conn, args.purge_run)
        else:
            print("Cancelled.")
        conn.close()
        return

    # ---- processing mode ----
    if not args.config or not args.corpus or not args.run_name:
        parser.error("--config, --corpus, and --run-name are required for processing.")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if "features" not in config or not config["features"]:
        parser.error("Config must contain a 'features' section with at least one feature.")

    # Validate feature definitions
    for name, fdef in config["features"].items():
        ftype = fdef.get("type", "boolean")
        if ftype not in ("boolean", "enum"):
            parser.error(
                f"Feature '{name}': unsupported type '{ftype}'. Use 'boolean' or 'enum'."
            )
        if ftype == "enum" and not fdef.get("options"):
            parser.error(
                f"Feature '{name}': enum type requires an 'options' list."
            )

    process_corpus(args, config)


if __name__ == "__main__":
    main()