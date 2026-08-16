# LLM Backend Compatibility

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

## Commercial / hosted endpoints

The same OpenAI-compatible `/v1/chat/completions` shape works for hosted providers, not just local
servers — including OpenAI itself. Pass `--api-key` (or set `API_KEY` in `.env`) and it's sent as
`Authorization: Bearer <key>`:

```bash
python docfeatures.py --host https://api.openai.com -m gpt-5 --api-key sk-... ...
```

`--api-key` is deliberately **not** settable from the YAML config the way `--host`/`--model` are — that
config gets stored verbatim in the database (`runs.config_yaml`) and can be displayed back (e.g. via
`docfeatures_web.py`), so routing a secret through it would mean persisting it in the DB. Use `--api-key`
or `.env` only. Local servers that don't expect an `Authorization` header are unaffected — the header is
simply omitted when no key is configured.

For AWS Bedrock instead of an OpenAI-compatible endpoint, see [Batch Processing](batch-processing.md) —
Bedrock uses a different client (`boto3`) and a different tool (`docfeatures_batch.py`), not `--host`.

## Model Selection Notes

Model recommendations change rapidly. Some general guidance as of early 2026:

**Disable reasoning/thinking mode.** Models like Qwen3, DeepSeek-R1, and other "reasoning" models generate an internal chain-of-thought before answering. This can produce thousands of wasted tokens per document and occasionally degenerate into infinite loops. For structured extraction tasks, disable thinking at the server level or via prompt tags (e.g., `/no_think` for Qwen3). docfeatures benefits from fast, direct answers — not deliberation.

**Instruction following matters more than model size.** A well-tuned 8B model that reliably produces clean JSON is more useful than a 70B model that occasionally returns malformed output or ignores the schema. Test with `--limit 10` and inspect the raw `chunk_results` table to verify output quality before committing to a large run.

**LLMs are extractors, not oracles.** When the model is pulling information from text you provide (feature extraction, classification, name extraction), accuracy is generally good. When the model must generate facts from memory (medical knowledge, dates, statistics), hallucination rates are high. Design your features to extract from the document, not to quiz the model.

See also [Validation and Retries](validation-and-retries.md) for what happens when a model doesn't follow the schema, and [Diagnosing Recurring Validation Failures](tools.md#docfeatures_validate_reportpy) for finding systematic problems in your feature definitions rather than the model.
