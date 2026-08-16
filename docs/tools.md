# Tools

Standalone scripts that aren't part of getting the core pipeline running, but are useful once you're
operating docfeatures day to day: quick model testing, and cleaning up corpus/database drift that
accumulates over time.

For the core pipeline (`docfeatures.py`, `docfeatures_batch.py`, `docfeatures_initdb.py`,
`docfeatures_web.py`), see the main [README](../README.md).

## query.py

Ad hoc CLI for testing a prompt against a local Ollama model directly — no database, no feature config,
just "does this model respond the way I expect." Useful when you're picking a model or debugging odd
output before wiring it into a full docfeatures run.

```bash
python query.py -m qwen3.5:35b -p "hello"
python query.py -m qwen3.5:35b -p "describe this image" -a photo.png
echo "some clinical text" | python query.py -m qwen3.5:35b -s "You are a radiologist."
```

Key flags: `-m/--model`, `-p/--prompt` (reads stdin if omitted), `-s/--system`, `-a/--attach` (repeatable,
png/jpg/gif/webp), `-e/--temperature`, `-c/--context` (context window size — computed automatically from
prompt length if omitted), `--host` (default `http://127.0.0.1:11434`).

For AWS Bedrock instead of a local Ollama server, use `query_bedrock.py` — see
[Batch Processing](batch-processing.md#testing-model-access-query_bedrockpy), since it's tightly tied to
that workflow (validating a model ID/access before a real batch job, working around the retired "Model
access" console page).

## docfeatures_dedupe.py

Finds `document_runs` rows within a run that share an identical `files.file_hash` (byte-for-byte identical
content, e.g. the same file present under multiple paths, or boilerplate placeholder text). Keeps one,
merges the others' feature tags onto it, deletes the rest.

```bash
python docfeatures_dedupe.py --run-name v1 --dry-run
python docfeatures_dedupe.py --run-name v1
```

## docfeatures_fix_paths.py

Corrects `files.file_path` rows whose *stored* capitalization no longer matches the file's *current*
on-disk name. This happens because `files.file_path` identity is case-insensitive at the database level
(so "FILE.TXT" and "file.txt" already resolve to the same file), but a case-sensitive filesystem still
needs the exact on-disk casing to actually open the file — if a corpus originally had two files differing
only by case and one was later deleted or renamed (e.g. to fix a Windows-incompatible duplicate), the
survivor's casing can drift from what's stored. Left uncorrected, this can both break
`docfeatures_web.py` document preview/export (file not found on disk) and, previously, caused
`docfeatures.py`/`docfeatures_batch.py` to mis-treat the file as unprocessed and fail with a duplicate-key
error (fixed via `filter_pending()` — but the *stored* casing still needs correcting so on-disk reads
work).

```bash
python docfeatures_fix_paths.py --dry-run   # preview
python docfeatures_fix_paths.py             # apply
```

Resolves relative `file_path` values against `CORPUS_BASE_PATH` from `.env` by default (override with
`--corpus-base`) — the same variable `docfeatures_web.py` uses for the same purpose. Rows with no
on-disk match at all, or more than one case-insensitive match, are reported but left untouched.

## docfeatures_validate_report.py

Diagnoses *why* chunks fail enum validation, rather than just reporting that they do. Every rejected
chunk-validation attempt (whether or not it was later corrected on retry — see
[Validation and Retries](validation-and-retries.md)) is logged to a `validation_failures` table as it
happens. This tool reads that table and classifies each invalid value against the run's own feature
schema instead of just listing raw strings and counts:

```bash
python docfeatures_validate_report.py --run-name v1
python docfeatures_validate_report.py --run-name v1 --feature malignancy_likelihood --top 20 --examples 3
```

Each invalid value is bucketed into one of:
- **near-miss of its own valid options** — a formatting/validator issue (whitespace, casing, singular vs.
  plural), not a prompt problem.
- **matches an option from a different enum feature** — the model is confusing two features; consider
  clarifying the prompt or separating them.
- **matches (or closely resembles) another feature's name** — often a boolean feature's key name leaking
  into an enum answer, usually because too many similarly-shaped features are crammed into one prompt.
- **novel** — not traceable to anything else in the schema. This is the strongest signal that an enum is
  missing a legitimate option (e.g. no `none`/`n/a`/`unclear` catch-all for documents where the feature
  genuinely doesn't apply).

See your team's feature definition guide (Confluence) for researcher-facing advice on interpreting these
categories and writing enums that avoid them.

For runs (or documents) that predate this table — or where a document's `error_message` got overwritten
by a later successful reprocessing — recover what's still recoverable from `document_runs.error_message`
with `--backfill` (idempotent, safe to re-run):

```bash
python docfeatures_validate_report.py --backfill              # all runs
python docfeatures_validate_report.py --backfill --run-name v1
```

Backfill can only recover documents that *never* got corrected on retry (i.e. still `status='error'`) —
it has no way to know about attempts that failed and then succeeded, since that was never persisted
anywhere before this table existed. Run it before reprocessing/retrying old errors if you want to keep
that history, since a successful reprocess overwrites `error_message`.
