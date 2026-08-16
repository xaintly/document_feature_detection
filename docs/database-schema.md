# Database

docfeatures uses MySQL. Configure credentials via `.env` file (see `.env.example`). Use one database per document corpus.

Tables are created automatically on first run (`python docfeatures_initdb.py`):

- **`runs`** — run metadata, config YAML, model info
- **`files`** — stable file identity (path, hash, size), one row per unique file regardless of how many runs have processed it
- **`document_runs`** — per-(file, run) status, chunk count, timing, errors
- **`batch_jobs`** — AWS Bedrock batch job metadata/lifecycle (see [Batch Processing](batch-processing.md))
- **`chunk_results`** — raw JSON from each LLM call (per-chunk)
- **`document_features`** — merged feature values (sparse: only positive/non-default values stored)
- **`feature_verifications`** — human-corrected values from the web app's edit/verify mode
- **`validation_failures`** — every rejected chunk-validation attempt, corrected on retry or not (see [Validation and Retries](validation-and-retries.md) and `docfeatures_validate_report.py` in [Tools](tools.md))

## Sparse Storage

Only positive/non-default values are stored in `document_features`:
- `false` booleans are omitted
- The first (lowest) enum option is omitted
- `null` text and integer values are omitted

A document with `status='complete'` and no feature rows means all features were negative/default. This dramatically reduces storage for large corpora where most features are absent from most documents.

## Keeping the schema clean

Two tools address database/corpus drift that accumulates over time as files get renamed, deleted, or
reprocessed — see [Tools](tools.md) for full usage:

- **`docfeatures_dedupe.py`** — merges byte-identical duplicate documents within a run.
- **`docfeatures_fix_paths.py`** — corrects stored file paths whose capitalization has drifted from the
  actual on-disk filename.
