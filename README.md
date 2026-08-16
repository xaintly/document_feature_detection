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
python docfeatures_initdb.py

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

See your team's feature definition guide (Confluence) for advice on writing enums that avoid common
failure patterns (missing catch-all options, overlapping vocabulary between features), and
[Tools](docs/tools.md#docfeatures_validate_reportpy) for a tool that diagnoses these automatically from a
real run.

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

## CLI Reference

```
usage: docfeatures.py [-h] [-c CONFIG] [--corpus CORPUS] [-r RUN_NAME]
                      [-n LIMIT] [--dry-run] [--retry-errors]
                      [--cooldown SECS] [--chunk-size CHARS] [--host HOST]
                      [-m MODEL] [--temperature TEMPERATURE]
                      [--api-key API_KEY] [--halt-on-conn-failure]
                      [--chunk-retry-max-attempts N] [--list-runs]
                      [--purge-run NAME]

Processing:
  -c, --config            YAML feature config file
  --corpus                Path to document directory or file (repeatable, overrides YAML 'corpus')
  -r, --run-name          Name for this run (used for resume and comparison)
  -n, --limit             Stop after N documents
  --dry-run               Print results without writing to the database
  --retry-errors          Re-process documents that errored previously
  --cooldown SECS         Pause between documents (thermal mitigation)
  --chunk-size CHARS      Max characters per chunk (default: 150,000)

LLM overrides (also settable in the YAML config's 'llm' section, except --api-key):
  --host                  LLM server URL
  -m, --model             Model name
  --temperature           Sampling temperature (default: 0.0)
  --api-key               Bearer token for commercial endpoints (or API_KEY in .env) -- .env only, not YAML
  --halt-on-conn-failure  Treat a refused connection as fatal instead of retrying it
  --chunk-retry-max-attempts N
                          Max corrective retries for a chunk that fails validation (default: 3)

Management:
  --list-runs             Show all runs in the database
  --purge-run NAME        Delete a run and all its results
```

## Documentation

- [LLM Backend Compatibility](docs/llm-backends.md) — ollama/vLLM/llama.cpp, commercial/hosted endpoints, model selection notes
- [Batch Processing (AWS Bedrock)](docs/batch-processing.md) — `docfeatures_batch.py`, `query_bedrock.py`, AWS setup
- [Database Schema](docs/database-schema.md) — tables, sparse storage
- [Validation and Retries](docs/validation-and-retries.md) — resume behavior, server resilience, corrective retries
- [Tools](docs/tools.md) — `query.py`, `docfeatures_dedupe.py`, `docfeatures_fix_paths.py`, `docfeatures_validate_report.py`

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
