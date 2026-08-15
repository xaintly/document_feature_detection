# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

docfeatures scans a corpus of text documents (originally built for radiology/clinical notes) through an
LLM to extract researcher-defined features — booleans, enums, free text, integers — and stores results in
MySQL, one row per feature per document. Researchers define features in a YAML config, iterate with small
`--limit` runs, then run the full corpus. A second "filtered run" pass can extract additional features
only from documents matching criteria from a prior run. A Flask web app provides search/browse/verify/
export over the results.

Two interchangeable processing paths share the same schema and feature-config YAML: `docfeatures.py`
sends documents one at a time to a locally-hosted, OpenAI-compatible LLM (ollama / vLLM / llama.cpp);
`docfeatures_batch.py` stages them as an AWS Bedrock batch inference job instead (cheaper/no server to
run, at the cost of async latency). Both are thin CLIs over shared logic in `lib/docfeatures_lib.py`.

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

# core pipeline (local LLM)
python docfeatures.py -c features.yaml --corpus /data/notes/ -r v1 -n 10   # small test run
python docfeatures.py -c features.yaml --corpus /data/notes/ -r v1         # resumes automatically, full corpus
python docfeatures.py -c features.yaml --corpus /data/notes/ -r v1 --dry-run -n 5   # print LLM output, write nothing
python docfeatures.py -c features.yaml --corpus /data/notes/ -r v1 --retry-errors
python docfeatures.py --list-runs
python docfeatures.py --purge-run v1

# alternative: AWS Bedrock batch (needs `pip install -r requirements-batch.txt`)
python query_bedrock.py -m <model-id> -p "hello"          # sanity-check model ID/access before a real batch job
python query_bedrock.py -m <model-id> --check-access       # free control-plane check, no invoke
python docfeatures_batch.py prepare -c features.yaml --corpus /data/notes/ -r v1 -n 100  # stage .jsonl, no AWS calls
python docfeatures_batch.py submit --job-name v1-x   # --model-id/--s3-bucket/--role-arn from .env if set, else pass them here
python docfeatures_batch.py status --job-name v1-x
python docfeatures_batch.py import --job-name v1-x --dry-run   # preview, writes nothing to the DB
python docfeatures_batch.py import --job-name v1-x -vv          # once Completed/PartiallyCompleted, for real
python docfeatures_batch.py cleanup --job-name v1-x    # delete local .jsonl + S3 input/output once you're done with it
python docfeatures_batch.py cancel --job-name v1-x      # returns its documents to the pending pool
python docfeatures_batch.py list-jobs -r v1

# post-processing
python docfeatures_dedupe.py --run-name v1 --dry-run     # find/merge byte-identical duplicate documents
python docfeatures_fix_paths.py --dry-run                 # repair files.file_path rows with stale on-disk casing

# web UI
python docfeatures_web.py                                 # dev server
gunicorn docfeatures_web:app -b 0.0.0.0:5000               # prod-style
```

Note: `docfeatures.py --help`'s CLI reference in README.md is out of date — the actual parser also has
`--dry-run`, `--chunk-size`, and `--temperature`; check `main()` in docfeatures.py, not just the README.

## Architecture

### Shared libraries (`lib/docfeatures_lib.py`, `lib/docfeatures_bedrock_lib.py`)

Only the two importable library modules live under `lib/` — every CLI entry point (`docfeatures.py`,
`docfeatures_batch.py`, `docfeatures_web.py`, `docfeatures_dedupe.py`, `docfeatures_initdb.py`,
`docfeatures_fix_paths.py`, `query.py`, `query_bedrock.py`) stays at the repo root and is invoked exactly
as its name suggests (`python docfeatures.py ...`). This split was deliberate: Python only puts a running
script's *own* directory on the import path, so moving the CLI scripts into a subfolder too would break
their `from lib... import` lines (or require `sys.path` shims / `python -m` invocation everywhere) for
essentially cosmetic benefit. `lib/` has an `__init__.py`; note the project's `.gitignore` originally
had a blanket `lib/`/`lib64/` rule inherited from the standard Python template (meant for setuptools
build artifacts) — that had to be removed for this directory to be tracked at all.

`lib/docfeatures_lib.py`: DB access, `sanitize_text`/`build_chunks`, `build_prompt`,
`parse_json_response`/`validate_enum_values`/`merge_chunk_results`, `discover_files`/`file_hash`,
`filter_pending()`, and `load_and_validate_config()` — so `docfeatures.py` (sync) and
`docfeatures_batch.py` (Bedrock batch) process documents identically up to the point of actually calling
an LLM. Deliberately has no `requests` or `boto3` import — `docfeatures.py` owns the former (`call_llm`),
the Bedrock tools the latter, so nothing drags in a transport dependency it doesn't need.

`lib/docfeatures_bedrock_lib.py`: the Converse-request-shape helpers (`build_converse_input`,
`build_image_block`, `extract_converse_text`, `apply_disable_thinking`) shared by `docfeatures_batch.py`
and `query_bedrock.py` — split out from `docfeatures_lib.py` because it's Bedrock-specific, not
corpus-processing-generic. `extract_converse_text` works on both a live `converse()` response and a
Bedrock batch output record, since AWS uses the identical shape for both.

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
files (file_id) ──< document_runs (doc_id, run_name, file_id, batch_job_id) ──< chunk_results (doc_id, chunk_index)
                                                              │              └──< document_features (doc_id, file_id, feature_name)
                                                              │              └──< feature_verifications (doc_id, feature_name)
runs (run_name) ──< document_runs                            │
             └──< batch_jobs (batch_job_id, run_name) ────────┘
```

- `files` is file identity (path/hash/size), set once on first sight and never overwritten
  (`get_or_create_file` logs an `[ANOMALY]` if a later run sees a changed hash for the same path, but
  keeps the original as canonical — earlier runs' results stay attributable).
  **`file_path` identity is case-insensitive** (MySQL's default collation on this DB is
  `utf8mb4_0900_ai_ci`) even though the stored string preserves whatever casing was first seen. This bit
  us once already: a corpus that originally had two files differing only by case, where one was later
  deleted/renamed (e.g. to fix a Windows-incompatible duplicate), leaves the survivor's on-disk casing
  drifted from the stored value. `filter_pending()` (in `lib/docfeatures_lib.py`, used by both
  `process_corpus` and `docfeatures_batch.py`'s `cmd_prepare`) compares case-insensitively for exactly
  this reason — a case-*sensitive* Python string comparison there previously disagreed with the DB's own
  identity and caused a duplicate-key crash on `document_runs`'s `(run_name, file_id)` unique constraint.
  `docfeatures_fix_paths.py` is the companion one-off/repeatable tool that corrects the *stored* casing to
  match disk (needed separately, since a case-sensitive filesystem still needs the exact on-disk name for
  `docfeatures_web.py` to actually read the file for preview/export).
- `document_runs` is one row per (file, run) — status (`processing`/`complete`/`error`/`batch_pending`),
  chunk count, timing, error text. A crash/Ctrl+C leaves a row `processing`; the next invocation for that
  run name deletes it via `cleanup_incomplete()` before resuming (cascades clean up its chunks/features).
  `cleanup_incomplete()` deliberately never touches `batch_pending` — that's a Bedrock batch job's claim
  on the file, not a crashed sync-tool session (see below).
- `document_features.file_id` is denormalized (duplicated from `document_runs`) specifically so "all
  features for this file across every run" doesn't require a join through `document_runs`/`runs` — this
  is what `docfeatures_web.py`'s cross-run views and `docfeatures_dedupe.py`'s tag-merging rely on.
- `feature_verifications` (not mentioned in README.md) backs the web app's edit/verify mode: a
  human-corrected value per (doc_id, feature_name), distinct from the LLM-extracted `document_features`
  value.
- `batch_jobs` tracks AWS Bedrock batch job lifecycle (see next section). Its `status` column is a plain
  `VARCHAR`, not a MySQL `ENUM` — it stores Bedrock's own status strings verbatim, which aren't ours to
  constrain a schema to.
- `docfeatures_initdb.py --migrate` converts an older single-`documents`-table schema (file identity and
  run status combined in one row) into the current `files`/`document_runs` split. It's idempotent —
  `detect_schema_state()` classifies the DB as fresh/legacy/migrated/partial and each of the 9 migration
  steps checks its own precondition, so a partial/interrupted migration can be re-run safely. Separately,
  `TABLES`/`INDEXES`/`COLUMNS`/`ENUM_ADDITIONS` are each idempotent additive-schema-change lists checked
  on every `do_init()` run (no `--migrate` needed) — `ENUM_ADDITIONS` exists because appending a value to
  an existing `ENUM` column doesn't fit the `ADD COLUMN`-shaped `COLUMNS` list.

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

### AWS Bedrock batch path (docfeatures_batch.py)

Alternative to the sync loop above, sharing `lib/docfeatures_lib.py` for everything except the actual LLM
call. Lifecycle: `prepare` (offline, no AWS calls) → `submit` → `status` (poll) → `import` → optionally
`cancel` at any point.

- **`prepare`**'s per-file loop body is wrapped in `try/except Exception`, mirroring
  `process_corpus`'s per-document handling — one bad file logs `[ERROR]` and moves on instead of crashing
  a run that may have already staged thousands of others. If a `document_runs` row was already created
  (`status='batch_pending'`) before the failure, it's marked `'error'` in the handler rather than left
  stuck — an uncleaned `batch_pending` row would otherwise be permanently "claimed" (see `filter_pending`
  above) with no batch job ever able to resolve it.
- **`prepare`** chunks/sanitizes documents exactly like `docfeatures.py`, but instead of calling an LLM,
  writes one Converse-format JSONL record per chunk (`build_converse_input`) and creates each document's
  `document_runs` row with `status='batch_pending'`, `batch_job_id` pointing at a new `batch_jobs` row
  (`status='preparing'`). `recordId` is `f"{doc_id}:{chunk_index}"` — `doc_id` is a global autoincrement,
  so this is unique with no separate recordId-mapping table needed; `import` parses it straight back into
  `(doc_id, chunk_index)`. A document that sanitizes to zero non-empty chunks (pure boilerplate) is
  completed immediately in `prepare` itself with default feature values, the same outcome the sync path
  produces for such documents, and never enters the batch.
- `-n`/`--limit` bounds the *document* count, but `--batch-max-records` bounds the *record* (chunk) count
  for the whole job — they're not the same axis, and a `-n` that looks safely under the quota can still
  produce a job over it if documents chunk into more than one record each. `prepare` tracks
  `writer.total_records` as a running total and checks `writer.total_records + total_chunks >
  args.batch_max_records` *before* staging each document (i.e. before `upsert_document`), breaking out of
  the `pending` loop rather than staging past the limit and letting `submit`/Bedrock validation fail on
  it later. `stopped_at_limit` drives a summary line reporting documents-staged-of-requested; the
  unstaged remainder is untouched (not `batch_pending`, not errored) so a follow-up `prepare` (same
  `-r`, new `--job-name`) picks them back up normally. A single document whose own chunk count exceeds
  `--batch-max-records` raises inside the per-file `try` (caught by the existing error handler below) —
  it can never fit regardless of what else is in the job.
- Because `get_finished_paths()` (shared lib) treats `batch_pending` as claimed, the sync tool and a
  second `prepare` (same or different run) won't re-stage a file that's already inside an unresolved
  batch job — this is what prevents double-processing across the two tools or concurrent batches.
  `get_finished_paths(conn, run_name, retry_errors=...)` — pass `args.retry_errors` straight through, no
  extra negation. It used to be named `include_errors` and be called via a `skip_errors = not
  args.retry_errors` intermediate at both call sites; that negation was backwards (confirmed against the
  very first commit — pre-existing, not introduced by any batch-tool work) and meant `docfeatures.py
  --retry-errors` had *never* worked as documented: errors were retried every run by default, and
  explicitly passing `--retry-errors` actually suppressed retrying them. Fixed in both
  `docfeatures.py` and `docfeatures_batch.py prepare` (which gained a `--retry-errors` flag it didn't have
  before — needed because `import` marks failed documents `'error'`, not `'batch_pending'`, so `cancel`'s
  cleanup (which only deletes `'batch_pending'` rows) can't return them to the pool once a job's been
  imported).
- **`submit`** uploads `prepare`'s staged `.jsonl` file(s) to S3 and calls `create_model_invocation_job`.
  Model selection happens here, not in `prepare` — Converse's whole point is a model-agnostic payload, so
  nothing about the staged records is model-specific. `--model-id` sometimes needs to be a cross-region
  inference profile ID (`us.anthropic...`), not the bare foundation-model ID some Claude models reject for
  on-demand use — validate with `query_bedrock.py` first rather than discovering this at submit time.
  `--model-id`/`--s3-bucket`/`--role-arn` each fall back to `BEDROCK_MODEL_ID`/`BEDROCK_S3_BUCKET`/
  `BEDROCK_ROLE_ARN` from `.env` (`DEFAULT_BEDROCK_*` module constants) when omitted — not `required=True`
  in argparse anymore, so the "at least one of CLI/env must supply it" check happens as an explicit
  post-parse validation in `main()` instead, naming exactly which ones are missing.
- **`status`** calls `get_model_invocation_job` and writes Bedrock's own status string (`Submitted`,
  `Validating`, `Scheduled`, `InProgress`, `Completed`, `PartiallyCompleted`, `Failed`, `Expired`,
  `Stopping`, `Stopped`) straight into `batch_jobs.status` — no local remapping.
  `IMPORTABLE_STATUSES = {Completed, PartiallyCompleted}` gates `import`.
- **`import`** downloads every object under the job's output S3 prefix, matches each line's `recordId`
  back to `(doc_id, chunk_index)`, and runs the *same* `parse_json_response`/`validate_enum_values` path
  the sync tool uses. Once every chunk for a `doc_id` has arrived, `merge_chunk_results` +
  `save_document_features` + `mark_document(..., 'complete')` runs — identical to the sync tool's ending.
  A document missing any chunk (errored record, or a `PartiallyCompleted` job that never got to it) is
  marked `'error'` instead; there's no post-hoc retry. Idempotency guard is `batch_jobs.imported_at`
  (not `status`, which stays Bedrock's own value so `status` can still be re-checked after import) —
  `import --force` re-runs it. `-v`/`--verbose` is a count (`action="count"`), not a boolean:
  `verbosity = max(args.verbose, 2 if args.dry_run else 0)` — level 1 (`-v`) prints each successfully-
  completed document's merged feature values as they're written (`fmt_feature_value`-formatted, same
  style as `docfeatures.py`'s own progress line; `fmt_feature_value` lives in `lib/docfeatures_lib.py`,
  shared by both); level 2 (`-vv`) additionally prints per-record parse/validation errors as they're
  found rather than only counting them. `--dry-run` forces verbosity to at least 2 and skips every DB
  write (`save_chunk_result`, `save_document_features`, `mark_document`, and
  `update_batch_job(imported_at=...)` all become no-ops via `if not args.dry_run:` guards) — it also
  bypasses the `imported_at`-without-`--force` guard, since a dry run changes nothing and is safe to
  repeat freely, including against a job that's already been really imported (though `pending_docs` will
  be empty by then, since those rows are no longer `'batch_pending'`). This is the "view a completed run"
  feature: run it before a real `import` to inspect results, or after a failure to tell a total loss from
  a partial success before deciding whether to `cancel`-and-retry.
- Every DB write in `import` (`save_chunk_result`, and `save_document_features`+`mark_document` together)
  is wrapped in `try/except pymysql.err.IntegrityError`, counted separately as `integrity_errors` and
  reported in the summary, rather than letting one bad row crash the whole import. Root cause: both
  `chunk_results` and `document_features` FK to `document_runs(doc_id)`, and a `doc_id` embedded in a
  batch job's S3 output can go stale if a *later* `prepare` for the same run re-stages the same file
  (`upsert_document` deletes-then-reinserts, handing it a new `doc_id`) before this job's `import` gets
  around to it — the old `doc_id` this job's output still references no longer exists in `document_runs`
  at all. `save_chunk_result`'s failure can't report a file name (the doc_id→file_id lookup goes through
  the now-gone `document_runs` row); `save_document_features`'s can, since `pending_docs` already joined
  `files` before the race window. Avoid triggering this by not re-running `prepare` for a run while an
  older batch job for that same run is still unimported.
- **`cancel`** calls `stop_model_invocation_job` if the job is still AWS-active, then unconditionally
  `DELETE`s the `document_runs` rows tied to that `batch_job_id` (FK cascade cleans up any partial
  `chunk_results`), returning those files to the pending pool, and sets `batch_jobs.status='cancelled'`
  (a local-only terminal state, distinct from Bedrock's own `Stopped`). This is the same path whether the
  job was never submitted, is still running, or failed on AWS — the "return files to the pool" ask this
  feature was built around.
- **`cleanup`** deletes the local staged `.jsonl` directory (`work_dir/run_name/job_name/`) and/or the
  job's S3 input/output prefixes (`--keep-local`/`--keep-s3-input`/`--keep-s3-output` opt out of each
  individually; default is delete all three). Refuses if `status in ACTIVE_STATUSES` — the same active-set
  `cancel` uses to decide whether to call `stop_model_invocation_job` — so it can't be used to yank the
  input/output out from under a job Bedrock is still processing. Doesn't touch the `batch_jobs`/
  `document_runs` rows at all, only files; the DB record stays as a historical record after cleanup.
  `delete_s3_keys()` batches `delete_objects` calls at 1000 keys (the API's per-call limit).
- `prepare --disable-thinking` sets `thinking: {type: disabled}`. **Opt-in, off by default** — this
  briefly defaulted to *on* (batch is one-shot per chunk with no retry, and `merge_chunk_results` discards
  reasoning tokens anyway, so disabling looked like a pure win), but real usage found Bedrock Batch's
  Converse validation rejecting the field outright (`extraneous key [thinking] is not permitted`) for a
  model that accepts the identical field fine via a live `query_bedrock.py` invoke — i.e. batch's Converse
  support is not a strict superset of live Converse's, at least not for this field, and a payload that
  validates live is not proof it'll be accepted by a real batch job. Reverted to opt-in given a forced
  default can silently fail 100% of a job's records. `query_bedrock.py` has always defaulted the other way
  (thinking on, `--disable-thinking` to turn off) since seeing the reasoning is usually the point of a
  one-off query — the two tools' defaults no longer match by design, not by oversight. Reasoning control
  itself (`llm.additional_model_request_fields` passthrough into Converse's `additionalModelRequestFields`)
  stays generic rather than a hardcoded per-model table — Bedrock hosts many model families with their own
  reasoning-control conventions and AWS ships new models often; a table would rot and wouldn't have caught
  this failure mode anyway since it's a batch-vs-live inconsistency, not a per-model difference.

### Web app (docfeatures_web.py)

Flask app exposing JSON APIs (`/api/runs`, `/api/browse`, `/api/search`, `/api/chart_stats`,
`/api/document/<id>/content`, `/api/document/<id>/features`, `/api/verify`, `/api/export`) over the same
schema, rendered by the single-page `templates/index.html`. `CORPUS_BASE_PATH` + `safe_file_path()` is
the path-traversal boundary for document preview/download — any file_path served must resolve to a
descendant of `CORPUS_BASE_PATH`; treat that function as security-relevant when touching preview/export
code. `MAX_PREVIEW_BYTES` (2MB) caps in-browser document preview.

### Other scripts

- `docfeatures_batch.py` — AWS Bedrock batch inference alternative to the sync loop; see its own
  architecture section above.
- `query_bedrock.py` — standalone ad hoc CLI for querying a Bedrock model directly via `converse()`
  (mirrors `query.py`'s role, but for Bedrock instead of a local Ollama server). Not part of the
  docfeatures data pipeline or schema — no DB connection at all. Exists because Bedrock's console "Model
  access" page was retired in favor of auto-provisioning on first invoke (up to ~15 min, failing with
  `AccessDeniedException` in the meantime); `--check-access`/`--subscribe` call the
  `ListFoundationModelAgreementOffers`/`CreateFoundationModelAgreement` control-plane APIs directly
  instead of guessing at retry timing. `-t/--timeout` retries a real invoke through that provisioning
  window rather than requiring the user to re-run the command by hand.
- `docfeatures_dedupe.py` — operates per `--run-name` on `document_runs` rows sharing an identical
  `files.file_hash` (byte-identical content). Keeps one document (prefers `status='complete'`, else
  lowest `doc_id`), merges the discarded documents' `document_features` onto the keeper when the keeper
  lacks that feature (conflicts — keeper already has a different value — are reported, not overwritten),
  then deletes the duplicates (FK cascade cleans their `chunk_results`/`document_features`). All schema
  assumptions here depend on `docfeatures_initdb.py`'s files/document_runs split. This is about
  byte-identical *content* duplicates — a different situation from `docfeatures_fix_paths.py` below,
  which is about one file whose stored path casing drifted from disk, not true duplicates.
- `docfeatures_fix_paths.py` — corrects `files.file_path` rows whose stored capitalization no longer
  matches the on-disk filename (see the `files` bullet under Database schema above for how this happens).
  For each row missing exactly-as-stored, checks the same directory for a unique case-insensitive match
  and updates the stored path to it; reports (but doesn't touch) rows with either no on-disk match at all
  (file removed from the corpus entirely — unrelated) or more than one case-insensitive match (an
  unresolved true duplicate, needs manual attention). `--corpus-base` defaults to `CORPUS_BASE_PATH` from
  `.env` — the same env var `docfeatures_web.py` already uses to resolve stored paths for preview/export.
- `query.py` — standalone ad hoc CLI for querying an Ollama model directly (prompt/system/image
  attachment/temperature/context). Not part of the docfeatures data pipeline or schema.

## YAML feature config

Four feature `type`s: `boolean`, `enum` (needs `options`, ordered weakest→strongest), `text` (optional
`strategy` and `max_length`), `integer` (use cautiously — LLMs are unreliable counters; prefer a boolean
+ manual review). `llm.host` / `llm.model` / `llm.temperature` in the YAML are defaults, overridable by
`--host`/`-m`/`--temperature`. See `examples/example_features.yaml` and
`examples/example_filtered_features.yaml` for full syntax including the `filter` section.
