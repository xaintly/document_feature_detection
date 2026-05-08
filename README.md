# docfeatures

**Scan large document corpora for researcher-defined features using a local LLM.**

docfeatures is a command-line tool that feeds text documents through a locally-hosted language model to identify specific features — clinical findings, metadata, document characteristics — and stores the results in MySQL for analysis. It's designed for medical/clinical research workflows where a researcher needs to triage thousands of documents before manual review.

## How It Works

1. You define the features you're looking for in a YAML config file.
2. docfeatures reads each document, sends it to a local LLM with a generated prompt, and parses the structured JSON response.
3. Results are stored in MySQL, one row per feature per document.
4. You query the results with SQL to find documents matching your criteria.

Large documents are automatically chunked at structural boundaries (HTML headers, markdown headers, paragraph breaks) and results are merged across chunks.

## Quick Start

```bash
pip install pymysql pyyaml requests python-dotenv

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
- **Connection refused (server down):** Halts the run immediately. Resume with the same `--run-name` once the server is back.  Change HALT_ON_CONN_FAILURE internally to False to treat as if 503/502.
- **Thermal throttling (compact hardware):** Use `--cooldown 3` to inject pauses between documents, reducing sustained heat load on devices like the NVIDIA DGX Spark.

## Database

docfeatures uses MySQL. Configure credentials via `.env` file (see `.env.example`). Use one database per document corpus.

Tables are created automatically on first run:

- **`runs`** — run metadata, config YAML, model info
- **`documents`** — per-document status, file path, hash, processing time
- **`chunk_results`** — raw JSON from each LLM call (per-chunk)
- **`document_features`** — merged feature values (sparse: only positive/non-default values stored)

### Sparse Storage

Only positive/non-default values are stored in `document_features`:
- `false` booleans are omitted
- The first (lowest) enum option is omitted
- `null` text and integer values are omitted

A document with `status='complete'` and no feature rows means all features were negative/default. This dramatically reduces storage for large corpora where most features are absent from most documents.

## Dependencies

```
pymysql
pyyaml
requests
python-dotenv
```

Python 3.8+.

## License

MIT License
