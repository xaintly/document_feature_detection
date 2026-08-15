# docfeatures

**Scan large document corpora for researcher-defined features using a local LLM.**

docfeatures is a command-line tool that feeds text documents through a locally-hosted language model to identify specific features — clinical findings, metadata, document characteristics — and stores the results in MySQL for analysis. It's designed for medical/clinical research workflows where a researcher needs to triage thousands of documents before manual review.

## How It Works

![data flow diagram](https://github.com/xaintly/document_feature_detection/blob/main/examples/data_flow.png?raw=true)

1. You define the features you're looking for in a YAML config file.
2. docfeatures reads each document, sends it to a local LLM with a generated prompt, and parses the structured JSON response.
3. Results are stored in MySQL, one row per feature per document.
4. You query the results with SQL to find documents matching your criteria.

Large documents are automatically chunked at structural boundaries (HTML headers, markdown headers, paragraph breaks) and results are merged across chunks.

## Quick Start

```bash
pip install -r requirements.txt 

# Configure database credentials
cp env.example .env
# Edit .env with your MySQL host, user, password, database

# Create a database (one per corpus)
mysql -e "CREATE DATABASE my_corpus"

# Test on 10 documents
python docfeatures.py \
  --config my_features.yaml \
  --corpus /path/to/documents/ \
  --run-name v1 \
  --host http://your-llm-server:11434 \
  --limit 10

# Check results, tweak config, iterate...
python docfeatures.py --config my_features.yaml --corpus /path/to/documents/ --run-name v2 --limit 10

# Happy with the output — full run
python docfeatures.py --config my_features.yaml --corpus /path/to/documents/ --run-name v2_final
```

## Feature Types

Define features in a YAML config file. Four types are supported:

### boolean

Is this feature present? `true` / `false`.

Chunks merge with **OR**: if any chunk says true, the document is true.

```yaml
lung_consolidation:
  type: boolean
  description: >
    Consolidation or airspace opacity. Indicated by air bronchograms,
    lobar or segmental opacification, or explicit mention of consolidation.
```

### enum

Which category does this document fall into? Exactly one of the listed options.

Options are ordered weakest → strongest. Chunks merge by taking the **strongest** (rightmost) value.

```yaml
malignancy_likelihood:
  type: enum
  options: [none, possible, probable, definite]
  description: >
    Overall suspicion for malignancy based on report language.
```

### text

Free text extraction — summaries, names, descriptions.

For chunked documents, set `strategy` to control merging:
- `first-chunk` — use the first non-empty value (good for header metadata like author names)
- `last-chunk` — use the last non-empty value (default; good for summaries and conclusions)
- `concatenate` — join all non-empty values with spaces

Optional: `max_length` gives the LLM a character limit hint.

```yaml
signing_radiologist:
  type: text
  strategy: first-chunk
  description: >
    Full name of the signing radiologist, or null if not found.

document_summary:
  type: text
  strategy: last-chunk
  max_length: 200
  description: >
    One to two sentence summary of primary findings.
```

### integer

Numeric extraction — counts, measurements. Null means "not applicable."

Chunks merge by taking the **MAX** non-null value.

**Use with caution.** LLMs are unreliable counters. Best used as triage approximations, not precise measurements. Consider using a boolean for detection, then reviewing matching documents manually.

```yaml
enlarged_node_count:
  type: integer
  description: >
    Number of enlarged lymph nodes, or null if not examined.
```

## Filtered Runs

After a broad scan, you can run a second pass that extracts additional features only from documents matching specific criteria. Add a `filter` section to the YAML config:

```yaml
filter:
  from_run: "lung_v3"
  require:
    lung_consolidation: true
    malignancy_likelihood: [probable, definite]
  exclude:
    pneumothorax: true

features:
  consolidation_location:
    type: enum
    options: [upper_lobe, middle_lobe, lower_lobe, multilobar, unspecified]
    description: ...
```

When a filter is present, `--corpus` is optional — the file list comes from the source run's completed documents. If `--corpus` is also specified, results are intersected.

Filter logic uses JOIN-based queries suitable for corpora with hundreds of millions of documents.

## LLM Backend Compatibility

docfeatures communicates via the OpenAI-compatible `/v1/chat/completions` endpoint, which is supported by all major local LLM serving tools:

| Backend | Default Port | Notes |
|---|---|---|
| **ollama** | 11434 | Simplest setup. Set `num_ctx` for large documents. |
| **vLLM** | 8000 | Best for multi-user concurrency (continuous batching). |
| **llama.cpp** (llama-server) | 8080 | Lightweight, single-user. |

```bash
# ollama
python docfeatures.py --host http://server:11434 -m qwen3.5:35b ...

# vLLM
python docfeatures.py --host http://server:8000 -m Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 ...

# llama-server
python docfeatures.py --host http://server:8080 ...
```

The `--host` and `--model` flags can also be set in the YAML config under the `llm` section to avoid repeating them:

```yaml
llm:
  host: "http://my-server:11434"
  model: "qwen3.5-35b-long"
```

## Model Selection Notes

Model recommendations change rapidly. Some general guidance as of early 2026:

**Disable reasoning/thinking mode.** Models like Qwen3, DeepSeek-R1, and other "reasoning" models generate an internal chain-of-thought before answering. This can produce thousands of wasted tokens per document and occasionally degenerate into infinite loops. For structured extraction tasks, disable thinking at the server level or via prompt tags (e.g., `/no_think` for Qwen3). docfeatures benefits from fast, direct answers — not deliberation.

**Instruction following matters more than model size.** A well-tuned 8B model that reliably produces clean JSON is more useful than a 70B model that occasionally returns malformed output or ignores the schema. Test with `--limit 10` and inspect the raw `chunk_results` table to verify output quality before committing to a large run.

**LLMs are extractors, not oracles.** When the model is pulling information from text you provide (feature extraction, classification, name extraction), accuracy is generally good. When the model must generate facts from memory (medical knowledge, dates, statistics), hallucination rates are high. Design your features to extract from the document, not to quiz the model.

## CLI Reference

```
usage: docfeatures.py [-h] [-c CONFIG] [--corpus CORPUS] [-r RUN_NAME]
                      [-n LIMIT] [--retry-errors] [--cooldown SECS]
                      [--host HOST] [-m MODEL]
                      [--list-runs] [--purge-run NAME]

Processing:
  -c, --config        YAML feature config file
  --corpus            Path to document directory or single file
  -r, --run-name      Name for this run (used for resume and comparison)
  -n, --limit         Stop after N documents
  --retry-errors      Re-process documents that errored previously
  --cooldown SECS     Pause between documents (thermal mitigation)

LLM overrides:
  --host              LLM server URL (overrides config file)
  -m, --model         Model name (overrides config file)

Management:
  --list-runs         Show all runs in the database
  --purge-run NAME    Delete a run and all its results
```

## Resume and Batching

Runs resume automatically. Re-running with the same `--run-name` skips completed documents and picks up where it left off. Documents interrupted mid-processing are cleaned up and retried.

The intended workflow for iterating on feature definitions:

```bash
# Test
python docfeatures.py -c features.yaml --corpus /data/ -r v1 -n 10
# Tweak config...
python docfeatures.py -c features.yaml --corpus /data/ -r v2 -n 10
# Compare v1 and v2 in MySQL, tweak again...
python docfeatures.py -c features.yaml --corpus /data/ -r v3 -n 10
# Satisfied — full run
python docfeatures.py -c features.yaml --corpus /data/ -r v3_final
# Clean up test runs
python docfeatures.py --purge-run v1
python docfeatures.py --purge-run v2
python docfeatures.py --purge-run v3
```

Ctrl+C stops gracefully after the current document, prints throughput statistics, and estimates time remaining for the rest of the corpus.

## Server Resilience

docfeatures handles LLM server instability:

- **503/502 (server restarting):** Retries up to 12 times with 15-second pauses (~3 minutes). If the server comes back, processing continues seamlessly.
- **Connection refused (server down):** By default, retried the same as a 502/503 (the server may just be mid-restart or there's a transient network blip). Pass `--halt-on-conn-failure` to instead treat this as fatal and halt the run immediately — resume with the same `--run-name` once the server is back.
- **Thermal throttling (compact hardware):** Use `--cooldown 3` to inject pauses between documents, reducing sustained heat load on devices like the NVIDIA DGX Spark.
- **Invalid enum values (temperature > 0 only):** If the LLM returns an enum value outside its declared `options`, the chunk is re-rolled up to `--chunk-retry-max-attempts` times (default 20). Only applies when sampling temperature is above 0 — a temperature-0 model is deterministic, so retrying would just reproduce the same invalid answer.

## Batch Processing (AWS Bedrock)

For large corpora, `docfeatures_batch.py` is an alternative to the synchronous `docfeatures.py` loop: it
stages documents as a model-agnostic [Converse-format](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
`.jsonl` file, submits it as an AWS Bedrock batch inference job, and imports the results once the job
completes. No server to keep running, and batch inference is typically cheaper than on-demand — the
tradeoff is latency (a job can take minutes to ~24h) and no per-chunk retry (each record gets one shot).

It shares chunking, prompt generation, response parsing, and the MySQL schema with `docfeatures.py` — the
same feature-config YAML works for both. `boto3` is only required for this tool; `docfeatures.py` doesn't
need it:

```bash
pip install -r requirements-batch.txt
```

### Workflow

```bash
# 0. Sanity-check the model ID and your access to it with a single cheap call
#    (see "Testing model access" below) before committing to a min-100-record batch job
python query_bedrock.py -m us.anthropic.claude-haiku-4-5-20251001-v1:0 -p "hello"

# 1. Chunk documents and stage them as local .jsonl file(s) — no AWS calls yet
python docfeatures_batch.py prepare -c features.yaml --corpus /data/notes/ -r v1 -n 100

# 2. Upload to S3 and create the Bedrock batch inference job
python docfeatures_batch.py submit --job-name v1-20260814 \
    --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0 \
    --s3-bucket my-batch-bucket --role-arn arn:aws:iam::123456789012:role/BedrockBatchRole

# 3. Poll status (updates the local DB's record of the job too)
python docfeatures_batch.py status --job-name v1-20260814

# 4. Once Completed/PartiallyCompleted, pull results into MySQL
python docfeatures_batch.py import --job-name v1-20260814

# List locally-known jobs (no AWS call)
python docfeatures_batch.py list-jobs -r v1

# Stop the job (if still running) and return its documents to the pending pool --
# for a job you decide not to run, or one that failed
python docfeatures_batch.py cancel --job-name v1-20260814
```

Each `prepare` command creates a `batch_jobs` row (status `preparing`) and moves the documents it stages
into `document_runs.status = 'batch_pending'` — a state the sync tool and other batch `prepare` runs both
treat as claimed, so nothing gets double-processed across the two tools or concurrent batches for the same
run. `cancel` is the release valve: it deletes those `batch_pending` rows (returning the files to the
pending pool for the run) whether the job was ever submitted, is still running, or failed on AWS.

`prepare` splits staged records across multiple `.jsonl` files according to `--batch-max-records` and
`--batch-max-file-bytes`, and warns (without blocking) if the total is under `--batch-min-records`. The
defaults (100 min / 100,000 max records per job, ~1GB max per file) match this project's confirmed AWS
Service Quotas — override them via flags if your account's quotas differ.

### Testing model access (`query_bedrock.py`)

Bedrock's old console "Model access" page has been retired — models are auto-enabled per account, and
invoking one for the first time triggers a background marketplace subscription instead (can take up to
~15 minutes; calls during that window fail with `AccessDeniedException`). `query_bedrock.py` is a small
standalone tool (same role as `query.py`, but for Bedrock instead of a local Ollama server) for poking a
model directly — useful before committing to a real batch job, or just to test a prompt:

```bash
# Send a prompt, print the response text
python query_bedrock.py -m us.anthropic.claude-haiku-4-5-20251001-v1:0 -p "hello"

# Check whether a model needs a marketplace agreement, without invoking it (no cost)
python query_bedrock.py -m anthropic.claude-haiku-4-5-20251001-v1:0 --check-access

# Accept the agreement ahead of time -- the closest replacement for the old "Model access" button
python query_bedrock.py -m anthropic.claude-haiku-4-5-20251001-v1:0 --subscribe

# Print the full raw Converse response (stopReason, usage, thinking blocks, ...)
python query_bedrock.py -m us.anthropic.claude-haiku-4-5-20251001-v1:0 -p "hello" --raw
```

If a model is still auto-provisioning, `query_bedrock.py` retries through `AccessDeniedException` for up
to `--timeout` minutes (default 5) instead of making you re-run the command by hand.

**Two IDs, two purposes**: `--check-access`/`--subscribe` use the bare foundation-model ID (e.g.
`anthropic.claude-haiku-4-5-20251001-v1:0`) — marketplace agreements are keyed to the underlying model.
Actually invoking (`-p`) several current Claude models, though, requires a **cross-region inference
profile ID** instead (e.g. `us.anthropic.claude-haiku-4-5-20251001-v1:0`) — the bare ID fails with
*"Invocation of model ID ... with on-demand throughput isn't supported."* Find the right one with:

```bash
aws bedrock list-inference-profiles --type-equals SYSTEM_DEFINED --region <region>
```

`docfeatures_batch.py submit --model-id` takes whichever ID actually works for a live invoke — test it
with `query_bedrock.py` first.

### Model "thinking" / reasoning

`prepare` sets `additionalModelRequestFields.thinking={type: disabled}` **by default**. Batch gives each
chunk exactly one shot with no retry loop, and any reasoning/chain-of-thought tokens are discarded by
`merge_chunk_results` regardless — so there's no upside to paying for them in a batch run (they're still
worth seeing for one-off exploratory prompts, which is what `query_bedrock.py` is for, and it defaults to
leaving thinking on).

**Adaptive-thinking-only models reject an explicit `disabled`** (as of this writing: Claude Mythos 5,
Claude Fable 5, Claude Opus 4.7, Claude Mythos Preview) — they return an HTTP 400 for it, so the default
will break `prepare` for those specifically. Pass `--enable-thinking` to suppress the forced `disabled`
injection (needed for those models; also available any time you just want reasoning on). To cap rather
than fully disable reasoning on an adaptive-only model, combine `--enable-thinking` with
`llm.additional_model_request_fields` in the feature-config YAML — passed straight through into every
record's Converse `additionalModelRequestFields`, so it works for any current or future Bedrock
model/provider without this project hardcoding a model-name table:

```yaml
llm:
  additional_model_request_fields:
    output_config:
      effort: low
```

Bedrock does **not** validate that `modelInput` matches a model's actual schema at submit time — a bad
thinking config only shows up as a per-record `error` in the job's output. Test with a small `-n`-bounded
`prepare` + `submit` first, the same iterate-small-first workflow described above for the sync tool.

### One-time AWS setup

`docfeatures_batch.py` expects an S3 bucket and an IAM **service role** to already exist — it doesn't
provision either itself (that needs broader IAM permissions than the rest of this project). Two separate
sets of permissions are involved; don't merge them onto the same policy:

1. **The service role** — the identity Bedrock itself assumes (via `sts:AssumeRole`) to read your input
   `.jsonl` and write output to S3 while the job runs. This is the `--role-arn` you pass to `submit`.
2. **Your own IAM user/credentials** — whatever AWS credentials are active when *you* run
   `docfeatures_batch.py submit/status/import/cancel` need `bedrock:CreateModelInvocationJob`,
   `bedrock:GetModelInvocationJob`, `bedrock:ListModelInvocationJobs`, `bedrock:StopModelInvocationJob` —
   typically already covered if you're an account admin, otherwise attach those actions to your own user/role.

**Create the service role** (IAM console → Roles → Create role → **Custom trust policy**, not "AWS
service" — Bedrock batch isn't a preset use case in that dropdown):

Trust policy (who can assume this role — replace `{{ACCOUNT_ID}}` and `{{REGION}}`):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": { "Service": "bedrock.amazonaws.com" },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": { "aws:SourceAccount": "{{ACCOUNT_ID}}" },
                "ArnEquals": { "aws:SourceArn": "arn:aws:bedrock:{{REGION}}:{{ACCOUNT_ID}}:model-invocation-job/*" }
            }
        }
    ]
}
```

Permissions policy attached to that same role (what it's allowed to do once assumed — replace
`{{BUCKET}}`):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3Access",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": ["arn:aws:s3:::{{BUCKET}}", "arn:aws:s3:::{{BUCKET}}/*"]
        }
    ]
}
```

Copy the resulting role's ARN (`arn:aws:iam::{{ACCOUNT_ID}}:role/...`) — that's the `--role-arn` for
`docfeatures_batch.py submit`.

See [AWS's batch inference permissions docs](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-permissions.html)
and [service role setup](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html) for the
full/current version of both policies.

## Database

docfeatures uses MySQL. Configure credentials via `.env` file (see `.env.example`). Use one database per document corpus.

Tables are created automatically on first run:

- **`runs`** — run metadata, config YAML, model info
- **`files`** — stable file identity (path, hash, size), one row per unique file regardless of how many runs have processed it
- **`document_runs`** — per-(file, run) status, chunk count, timing, errors
- **`batch_jobs`** — AWS Bedrock batch job metadata/lifecycle (see [Batch Processing](#batch-processing-aws-bedrock) above)
- **`chunk_results`** — raw JSON from each LLM call (per-chunk)
- **`document_features`** — merged feature values (sparse: only positive/non-default values stored)

### Sparse Storage

Only positive/non-default values are stored in `document_features`:
- `false` booleans are omitted
- The first (lowest) enum option is omitted
- `null` text and integer values are omitted

A document with `status='complete'` and no feature rows means all features were negative/default. This dramatically reduces storage for large corpora where most features are absent from most documents.

## Maintenance Tools

Two small standalone scripts for cleaning up corpus/database drift that accumulates over time:

- **`docfeatures_dedupe.py`** — finds `document_runs` rows within a run that share an identical
  `files.file_hash` (byte-for-byte identical content, e.g. the same file present under multiple paths, or
  boilerplate placeholder text). Keeps one, merges the others' feature tags onto it, deletes the rest.
  ```bash
  python docfeatures_dedupe.py --run-name v1 --dry-run
  python docfeatures_dedupe.py --run-name v1
  ```

- **`docfeatures_fix_paths.py`** — corrects `files.file_path` rows whose *stored* capitalization no
  longer matches the file's *current* on-disk name. This happens because `files.file_path` identity is
  case-insensitive at the database level (so "FILE.TXT" and "file.txt" already resolve to the same file),
  but a case-sensitive filesystem still needs the exact on-disk casing to actually open the file — if a
  corpus originally had two files differing only by case and one was later deleted or renamed (e.g. to
  fix a Windows-incompatible duplicate), the survivor's casing can drift from what's stored. Left
  uncorrected, this can both break `docfeatures_web.py` document preview/export (file not found on disk)
  and, previously, caused `docfeatures.py`/`docfeatures_batch.py` to mis-treat the file as unprocessed and
  fail with a duplicate-key error (fixed as of this version — see `filter_pending()` — but the *stored*
  casing still needs correcting so on-disk reads work).
  ```bash
  python docfeatures_fix_paths.py --dry-run   # preview
  python docfeatures_fix_paths.py             # apply
  ```
  Resolves relative `file_path` values against `CORPUS_BASE_PATH` from `.env` by default (override with
  `--corpus-base`) — the same variable `docfeatures_web.py` uses for the same purpose. Rows with no
  on-disk match at all, or more than one case-insensitive match, are reported but left untouched.

## Dependencies

```
pymysql
pyyaml
requests
python-dotenv
```

`docfeatures_batch.py` and `query_bedrock.py` additionally need `boto3` — install via
`requirements-batch.txt` (kept separate so the local-LLM path doesn't force it on everyone).

Python 3.8+.

## License

MIT License
