# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

docfeatures scans a corpus of text documents (originally built for radiology/clinical notes) through a
locally-hosted, OpenAI-compatible LLM (ollama / vLLM / llama.cpp) to extract researcher-defined features
— booleans, enums, free text, integers — and stores results in MySQL, one row per feature per document.
Researchers define features in a YAML config, iterate with small `--limit` runs, then run the full corpus.
A second "filtered run" pass can extract additional features only from documents matching criteria from a
prior run. A Flask web app provides search/browse/verify/export over the results.

There is no test suite, linter, or build step in this repo. "Testing" a change in practice means a
`--dry-run` or small `--limit` run against real or sample documents pointed at a real LLM server, or
`docfeatures_initdb.py --check` for schema changes.

## Setup / running

```bash
pip install -r requirements.txt
cp env.example .env                     # fill in DB_HOST/PORT/USER/PASSWORD/NAME (+ CORPUS_BASE_PATH, FLASK_SECRET_KEY for the web app)

python docfeatures_initdb.py            # create schema (safe/idempotent — run once per new DB)
python docfeatures_initdb.py --check    # verify schema without changes
python docfeatures_initdb.py --migrate  # upgrade a pre-files/document_runs-split DB in place

# core pipeline
python docfeatures.py -c features.yaml --corpus /data/notes/ -r v1 -n 10   # small test run
python docfeatures.py -c features.yaml --corpus /data/notes/ -r v1         # resumes automatically, full corpus
python docfeatures.py -c features.yaml --corpus /data/notes/ -r v1 --dry-run -n 5   # print LLM output, write nothing
python docfeatures.py -c features.yaml --corpus /data/notes/ -r v1 --retry-errors
python docfeatures.py --list-runs
python docfeatures.py --purge-run v1

# post-processing
python docfeatures_dedupe.py --run-name v1 --dry-run     # find/merge byte-identical duplicate documents

# web UI
python docfeatures_web.py                                 # dev server
gunicorn docfeatures_web:app -b 0.0.0.0:5000               # prod-style
```

Note: `docfeatures.py --help`'s CLI reference in README.md is out of date — the actual parser also has
`--dry-run`, `--chunk-size`, and `--temperature`; check `main()` in docfeatures.py, not just the README.

## Architecture

### Pipeline (docfeatures.py, `process_corpus`)

For each pending file: read bytes → decode (utf-8, falling back to cp1252, then utf-8 with
`errors="replace"`) → `build_chunks()` splits oversized text at the largest structural boundary that
fits (HTML headers → markdown headers → paragraph breaks → sentences → lines → hard char split;
`split_into_sections` cascades through these) → each chunk is sanitized (`sanitize_text` — strips HTML,
control chars, normalizes whitespace) *after* chunking so structural markup survives the split →
`build_prompt()` turns the YAML `features` block into instructions → `call_llm()` posts to
`{host}/v1/chat/completions` → `parse_json_response()` tolerates markdown fences, chain-of-thought
preamble, and truncated/unclosed JSON (brace-repair) → `validate_enum_values()` re-rolls a chunk (up to
`CHUNK_RETRY_MAX_ATTEMPTS`, only when `temperature > 0`) if the model returns an enum value outside its
declared `options` → all chunk results for a document are combined by `merge_chunk_results()`.

Merge semantics per feature type (chunks → one document-level value):
- `boolean`: OR (any chunk true → true)
- `enum`: strongest wins — options are ordered weakest→strongest, merge takes the rightmost/highest-index value seen
- `text`: `strategy: first-chunk | last-chunk (default) | concatenate`
- `integer`: MAX of non-null values, else null

Only non-default values are persisted (`save_document_features`): false booleans, the first/lowest enum
option, and null text/integer are all omitted. A `document_runs.status='complete'` row with no
`document_features` rows means every feature evaluated to its default — this is what keeps large corpora
cheap to store.

### Database schema (docfeatures_initdb.py is the single source of truth)

```
files (file_id) ──< document_runs (doc_id, run_name, file_id) ──< chunk_results (doc_id, chunk_index)
                                                              └──< document_features (doc_id, file_id, feature_name)
                                                              └──< feature_verifications (doc_id, feature_name)
runs (run_name) ──< document_runs
```

- `files` is file identity (path/hash/size), set once on first sight and never overwritten
  (`get_or_create_file` logs an `[ANOMALY]` if a later run sees a changed hash for the same path, but
  keeps the original as canonical — earlier runs' results stay attributable).
- `document_runs` is one row per (file, run) — status (`processing`/`complete`/`error`), chunk count,
  timing, error text. A crash/Ctrl+C leaves a row `processing`; the next invocation for that run name
  deletes it via `cleanup_incomplete()` before resuming (cascades clean up its chunks/features).
- `document_features.file_id` is denormalized (duplicated from `document_runs`) specifically so "all
  features for this file across every run" doesn't require a join through `document_runs`/`runs` — this
  is what `docfeatures_web.py`'s cross-run views and `docfeatures_dedupe.py`'s tag-merging rely on.
- `feature_verifications` (not mentioned in README.md) backs the web app's edit/verify mode: a
  human-corrected value per (doc_id, feature_name), distinct from the LLM-extracted `document_features`
  value.
- `docfeatures_initdb.py --migrate` converts an older single-`documents`-table schema (file identity and
  run status combined in one row) into the current `files`/`document_runs` split. It's idempotent —
  `detect_schema_state()` classifies the DB as fresh/legacy/migrated/partial and each of the 9 migration
  steps checks its own precondition, so a partial/interrupted migration can be re-run safely.

### Filtered runs

A YAML config's `filter` section (`from_run`, `require`, `exclude`) makes `--corpus` optional — the file
list is derived from a prior run's completed documents instead of walking the filesystem.
`get_filtered_paths()` builds this as one query with an `INNER JOIN`/`LEFT JOIN...IS NULL` per filter
criterion against `document_features` (not `IN (SELECT ...)` subqueries), explicitly so it stays fast
against corpora with hundreds of millions of rows. `validate_filter()` checks the source run exists and
that every referenced feature name is defined in *that run's stored config_yaml* before the run starts.

### LLM server resilience (docfeatures.py constants near `call_llm`)

502/503 responses retry up to `RETRY_MAX_ATTEMPTS` (12) with `RETRY_DELAY_SECS` (15s) pauses — these two
are hardcoded, not CLI flags. A refused/unreachable connection is retried the same way by default; pass
`--halt-on-conn-failure` to instead raise `LLMServerDead` and halt the run on the first failure. Chunk
retries for out-of-options enum values (`--chunk-retry-max-attempts`, default 20) only kick in when
`--temperature > 0` — deterministic temperature-0 output won't change on retry, so `max_attempts` is
forced to 1 regardless of the flag in that case.

### Web app (docfeatures_web.py)

Flask app exposing JSON APIs (`/api/runs`, `/api/browse`, `/api/search`, `/api/chart_stats`,
`/api/document/<id>/content`, `/api/document/<id>/features`, `/api/verify`, `/api/export`) over the same
schema, rendered by the single-page `templates/index.html`. `CORPUS_BASE_PATH` + `safe_file_path()` is
the path-traversal boundary for document preview/download — any file_path served must resolve to a
descendant of `CORPUS_BASE_PATH`; treat that function as security-relevant when touching preview/export
code. `MAX_PREVIEW_BYTES` (2MB) caps in-browser document preview.

### Other scripts

- `docfeatures_dedupe.py` — operates per `--run-name` on `document_runs` rows sharing an identical
  `files.file_hash` (byte-identical content). Keeps one document (prefers `status='complete'`, else
  lowest `doc_id`), merges the discarded documents' `document_features` onto the keeper when the keeper
  lacks that feature (conflicts — keeper already has a different value — are reported, not overwritten),
  then deletes the duplicates (FK cascade cleans their `chunk_results`/`document_features`). All schema
  assumptions here depend on `docfeatures_initdb.py`'s files/document_runs split.
- `query.py` — standalone ad hoc CLI for querying an Ollama model directly (prompt/system/image
  attachment/temperature/context). Not part of the docfeatures data pipeline or schema.

## YAML feature config

Four feature `type`s: `boolean`, `enum` (needs `options`, ordered weakest→strongest), `text` (optional
`strategy` and `max_length`), `integer` (use cautiously — LLMs are unreliable counters; prefer a boolean
+ manual review). `llm.host` / `llm.model` / `llm.temperature` in the YAML are defaults, overridable by
`--host`/`-m`/`--temperature`. See `examples/example_features.yaml` and
`examples/example_filtered_features.yaml` for full syntax including the `filter` section.
