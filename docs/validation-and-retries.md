# Validation and Retries

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
- **Malformed JSON or an invalid enum value:** the chunk is retried up to `--chunk-retry-max-attempts` times (default 3) — but not by resending an identical prompt and hoping for a different roll. Each retry tells the model what it answered and specifically why that was rejected, and asks for a correction. Because the retry prompt is different from the original (not just resampled), this works the same regardless of `--temperature`, including the default of 0. Set `--chunk-retry-max-attempts 1` to disable retrying entirely — useful if you're studying the model's raw first-attempt failure rate rather than triaging around it.

## Diagnosing recurring validation failures

Because most rejected chunks now get corrected on retry, they no longer show up as a document-level
error — which is good for throughput, but means `document_runs.error_message` stops being a useful place
to look for *patterns* in why the model picks invalid enum values. Every rejected attempt (corrected or
not) is instead logged to a `validation_failures` table as it happens (see [Database Schema](database-schema.md)).

`docfeatures_validate_report.py` reads that table and classifies each invalid value against the run's own
feature schema, instead of just listing raw strings and counts — see [Tools](tools.md#docfeatures_validate_reportpy)
for full usage, including how to recover pre-existing error data with `--backfill`.
