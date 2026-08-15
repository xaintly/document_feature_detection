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
import signal
import sys
import time
from pathlib import Path

import requests
import yaml

from lib.docfeatures_lib import (
    CHUNK_TARGET_CHARS,
    build_chunks,
    build_correction_note,
    build_prompt,
    cleanup_incomplete,
    discover_files,
    file_hash,
    filter_pending,
    fmt_feature_value,
    get_connection,
    get_finished_paths,
    get_filtered_paths,
    get_or_create_file,
    get_or_create_run,
    list_runs_db,
    load_and_validate_config,
    mark_document,
    merge_chunk_results,
    parse_json_response,
    purge_run_db,
    sanitize_text,
    save_chunk_result,
    save_document_features,
    upsert_document,
    validate_enum_values,
    validate_filter,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LLM_HOST = os.environ.get("DEFAULT_LLM_HOST", "http://127.0.0.1:11433")
DEFAULT_LLM_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "default")
# Bearer token for commercial OpenAI-compatible endpoints (OpenAI itself, or
# any other hosted provider using the same API shape). Deliberately NOT
# readable from the YAML feature config (unlike host/model/temperature) --
# that config gets persisted verbatim into runs.config_yaml and can be
# displayed back (e.g. via docfeatures_web.py), so routing a secret through
# it would mean storing it in the database. --api-key / API_KEY only.
DEFAULT_API_KEY = os.environ.get("API_KEY")
# Default for --chunk-retry-max-attempts: how many times to retry a chunk
# when the LLM returns an enum value outside its declared options, before
# giving up on the document. Applies regardless of temperature -- each retry
# tells the model what it answered and why that was rejected (see
# build_correction_note), so it's a meaningful retry even at temperature 0,
# not a blind reroll hoping sampling gives a different answer. Set to 1 to
# disable retrying (e.g. to study the model's raw first-attempt failure
# rate rather than triage around it).
CHUNK_RETRY_MAX_ATTEMPTS = 3

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
# LLM Interaction
# ===========================================================================

# Retry configuration
RETRY_DELAY_SECS = 15           # wait between retries on 503 / transient errors
RETRY_MAX_ATTEMPTS = 12         # give up after ~3 minutes of retries
RETRY_HTTP_CODES = {502, 503}   # codes that trigger a retry


class LLMServerDead(Exception):
    """Raised when the LLM server is unreachable (connection refused)."""
    pass


def call_llm(host, model, prompt, temperature=0.0, halt_on_conn_failure=False, api_key=None):
    """Send prompt to llama-server with retry on transient errors.

    - 502/503: server restarting → retry up to RETRY_MAX_ATTEMPTS
    - ConnectionError: retried the same as 502/503 by default, since the
      server may just be mid-restart or there's a transient network blip.
      Pass halt_on_conn_failure=True to instead raise LLMServerDead on the
      first failure (--halt-on-conn-failure).
    - Other HTTP errors: raise normally (per-document error)

    api_key, if given, is sent as `Authorization: Bearer <api_key>` -- for
    commercial OpenAI-compatible endpoints (OpenAI itself, or any other
    hosted provider using the same API shape) that require it. Local
    servers (ollama/vLLM/llama-server) typically ignore the header if sent,
    but api_key is None by default so it's simply omitted for them.
    """
    url = f"{host.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        # "max_completion_tokens": 2048,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        if _interrupted:
            raise KeyboardInterrupt

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=600)
        except requests.exceptions.ConnectionError as e:
            if not halt_on_conn_failure and attempt < RETRY_MAX_ATTEMPTS:
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


# ===========================================================================
# Main processing loop
# ===========================================================================

def process_corpus(args, config):
    features_config = config["features"]
    host = args.host or config.get("llm", {}).get("host", DEFAULT_LLM_HOST)
    model = args.model or config.get("llm", {}).get("model", DEFAULT_LLM_MODEL)
    temperature = (
        args.temperature if args.temperature is not None
        else config.get("llm", {}).get("temperature", 0.0)
    )
    # Deliberately not readable from the YAML config -- see DEFAULT_API_KEY.
    api_key = args.api_key or DEFAULT_API_KEY

    conn = get_connection()

    config_hash = hashlib.sha256(yaml.dump(config).encode()).hexdigest()
    if not args.dry_run:
        get_or_create_run(conn, args.run_name, config, config_hash, host, model, temperature)
        cleanup_incomplete(conn, args.run_name)

    finished = get_finished_paths(conn, args.run_name, retry_errors=args.retry_errors)

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
    pending = filter_pending(all_files, finished)

    print(f"Run          : {args.run_name}" + ("  [DRY RUN -- nothing will be saved]" if args.dry_run else ""), file=sys.stderr)
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
    print(f"LLM          : {model} @ {host}  (temperature={temperature})", file=sys.stderr)
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

            if args.dry_run:
                file_id = None
            else:
                file_id = get_or_create_file(conn, file_path, fhash, fsize)
                doc_id = upsert_document(conn, args.run_name, file_id, num_chunks)

            chunk_results_list = []
            for ci, chunk_text in enumerate(chunks):
                if _interrupted:
                    break
                clean_chunk = sanitize_text(chunk_text)
                if not clean_chunk:
                    continue  # skip empty chunks (e.g., pure HTML boilerplate)
                chunk_info = (ci + 1, num_chunks) if num_chunks > 1 else None

                max_attempts = args.chunk_retry_max_attempts
                correction = None
                for attempt in range(1, max_attempts + 1):
                    prompt = build_prompt(features_config, clean_chunk, chunk_info, correction=correction)
                    raw = call_llm(
                        host, model, prompt, temperature,
                        halt_on_conn_failure=args.halt_on_conn_failure,
                        api_key=api_key,
                    )
                    parsed = None  # reset each attempt -- must not leak a stale value into the fallback below
                    try:
                        parsed = parse_json_response(raw)
                        validate_enum_values(parsed, features_config, chunk_info)
                    except ValueError as e:
                        if max_attempts == 1:
                            raise e
                        elif attempt == max_attempts:
                            raise ValueError(f"Chunk retries limit exceeded: {e}")
                        else:
                            print(f"  [ERROR] {Path(file_path).name}: attempt {attempt}/{max_attempts} {e}", file=sys.stderr)
                            # Retry with the actual error fed back, not a blind
                            # reroll -- this changes the input, so it's a
                            # meaningful retry even at temperature 0. Fall back
                            # to a raw-text excerpt if parsing itself failed
                            # (nothing to json.dumps in that case).
                            prev_text = json.dumps(parsed) if parsed is not None else (raw[:500] if raw else "(empty response)")
                            correction = build_correction_note(prev_text, str(e))
                            continue
                    break
                if not args.dry_run:
                    save_chunk_result(conn, doc_id, ci, json.dumps(parsed))
                chunk_results_list.append(parsed)

            if _interrupted:
                # leave as 'processing'; cleanup_incomplete will handle next run
                break

            merged = merge_chunk_results(chunk_results_list, features_config)
            elapsed = time.time() - doc_start
            if not args.dry_run:
                save_document_features(conn, doc_id, file_id, merged, features_config)
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
    print(f'Run "{args.run_name}"' + ("  [DRY RUN -- nothing was saved]" if args.dry_run else ""), file=sys.stderr)
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
        "--dry-run", action="store_true",
        help="Call the LLM and print results to the screen, but don't create "
             "a run or write anything to the database. Useful for iterating "
             "on feature definitions against a small corpus. Combine with "
             "-n/--limit to bound it, or Ctrl+C to stop early.",
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
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="LLM sampling temperature (default: 0.0, or from config's "
             "llm.temperature). Use >0 to check agreement across repeated "
             "runs of the same model.",
    )
    parser.add_argument(
        "--api-key", default=DEFAULT_API_KEY,
        help="Bearer token for commercial OpenAI-compatible endpoints (OpenAI, or any other "
             "hosted provider using the same API shape). Sent as 'Authorization: Bearer <key>'. "
             "Default: API_KEY from .env. Not settable via the YAML config, since that gets "
             "stored in the database (runs.config_yaml) -- use --api-key or .env only.",
    )
    parser.add_argument(
        "--halt-on-conn-failure", action="store_true",
        help="Treat a refused/unreachable LLM connection as fatal and halt "
             "the run immediately. Default: retry it the same as a 502/503 "
             "(up to ~3 minutes), since the server may just be mid-restart.",
    )
    parser.add_argument(
        "--chunk-retry-max-attempts", type=int, default=CHUNK_RETRY_MAX_ATTEMPTS,
        metavar="N",
        help="Max attempts for a chunk that fails to parse as JSON or returns an "
             "enum value outside its declared options (default: %(default)s). "
             "Each retry tells the model what it answered and why that was "
             "rejected, so this applies regardless of --temperature -- it's not "
             "a blind reroll. Set to 1 to disable retrying entirely (e.g. to "
             "study the model's raw first-attempt failure rate).",
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
            f"{'Run Name':<30} {'Model':<20} {'Temp':>5} "
            f"{'Done':>6} {'Errs':>5} {'Total':>6}  Created"
        )
        print("-" * 100)
        for r in runs:
            print(
                f"{r['run_name']:<30} {(r['llm_model'] or '?'):<20} "
                f"{r['llm_temperature'] if r['llm_temperature'] is not None else 0.0:>5.2f} "
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

    try:
        config = load_and_validate_config(args.config)
    except ValueError as e:
        parser.error(str(e))

    # --corpus is required unless the config has corpus paths or a filter section
    has_filter = "filter" in config and config["filter"]
    has_yaml_corpus = bool(config.get("corpus"))
    has_cli_corpus = bool(args.corpus)
    if not has_cli_corpus and not has_yaml_corpus and not has_filter:
        parser.error(
            "No corpus specified. Provide --corpus on the command line, "
            "'corpus' in the YAML config, or a 'filter' section."
        )

    if args.chunk_retry_max_attempts < 1:
        parser.error("--chunk-retry-max-attempts must be at least 1.")

    process_corpus(args, config)


if __name__ == "__main__":
    main()
