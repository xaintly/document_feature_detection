#!/usr/bin/env python3
"""
docfeatures_batch.py — AWS Bedrock batch inference for docfeatures

Alternative to docfeatures.py's synchronous per-document loop: stages
documents as Converse-format .jsonl files, submits them as an AWS Bedrock
batch inference job, and imports the results once the job completes. Shares
chunking/prompt/parsing/merge logic and the MySQL schema with docfeatures.py
via docfeatures_lib.py. Along with query_bedrock.py, this is one of the two
files in the project that import boto3 -- docfeatures.py and
docfeatures_lib.py stay free of it.

One-time AWS setup (S3 bucket + IAM service role) is documented in the
README, not automated here. Before submitting a real (min-100-record) batch
job, it's worth validating --model-id actually works with a single cheap
call via query_bedrock.py first -- it'll surface both missing model access
and the "needs an inference profile ID, not the bare model ID" error some
models (several current Claude models included) return for on-demand use.

Usage:
    # 0. Sanity-check the model ID/access first (see query_bedrock.py)
    python query_bedrock.py -m us.anthropic.claude-haiku-4-5-20251001-v1:0 -p "hello"

    # 1. Stage chunked documents as local .jsonl file(s)
    python docfeatures_batch.py prepare -c features.yaml --corpus /data/notes/ -r v1 -n 100

    # 2. Upload to S3 and create the Bedrock batch inference job
    python docfeatures_batch.py submit --job-name v1-20260814 \\
        --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0 \\
        --s3-bucket my-batch-bucket --role-arn arn:aws:iam::123456789012:role/BedrockBatchRole

    # 3. Poll status
    python docfeatures_batch.py status --job-name v1-20260814

    # 4. Preview results without writing to the DB, or pull them in for real
    python docfeatures_batch.py import --job-name v1-20260814 --dry-run
    python docfeatures_batch.py import --job-name v1-20260814

    # 5. Delete the local .jsonl staging and S3 input/output for a finished job
    python docfeatures_batch.py cleanup --job-name v1-20260814

    # Return files to the pending pool if you cancel or a job fails
    python docfeatures_batch.py cancel --job-name v1-20260814

    python docfeatures_batch.py list-jobs -r v1
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pymysql
import yaml

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    print(
        "docfeatures_batch.py requires boto3, which isn't installed.\n"
        "Install it with:\n"
        "  pip install -r requirements-batch.txt",
        file=sys.stderr,
    )
    sys.exit(1)

from lib.docfeatures_lib import (
    CHUNK_TARGET_CHARS,
    build_chunks,
    build_prompt,
    discover_files,
    file_hash,
    filter_pending,
    fmt_feature_value,
    get_connection,
    get_filtered_paths,
    get_finished_paths,
    get_or_create_file,
    get_or_create_run,
    load_and_validate_config,
    mark_document,
    merge_chunk_results,
    parse_json_response,
    sanitize_text,
    save_chunk_result,
    save_document_features,
    upsert_document,
    validate_enum_values,
    validate_filter,
)
from lib.docfeatures_bedrock_lib import (
    apply_disable_thinking,
    build_converse_input,
    extract_converse_text,
)

# ---------------------------------------------------------------------------
# Constants (defaults confirmed against the AWS console's Service Quotas
# for this account -- override via CLI flags if your quota differs)
# ---------------------------------------------------------------------------
DEFAULT_BATCH_MIN_RECORDS = 100
DEFAULT_BATCH_MAX_RECORDS = 100_000
DEFAULT_BATCH_MAX_FILE_BYTES = 950_000_000  # AWS max is 1GB; leave margin
DEFAULT_WORK_DIR = "./batch_work"

# submit's --model-id/--s3-bucket/--role-arn fall back to these if set, so
# they don't need to be repeated on every invocation.
DEFAULT_BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID")
DEFAULT_BEDROCK_S3_BUCKET = os.environ.get("BEDROCK_S3_BUCKET")
DEFAULT_BEDROCK_ROLE_ARN = os.environ.get("BEDROCK_ROLE_ARN")

# Bedrock's own job status values (from GetModelInvocationJob) that mean
# the job is still active on AWS and eligible for StopModelInvocationJob.
ACTIVE_STATUSES = {"Submitted", "Validating", "Scheduled", "InProgress"}
# Statuses from which results can be imported.
IMPORTABLE_STATUSES = {"Completed", "PartiallyCompleted"}


# ===========================================================================
# batch_jobs DB helpers (specific to this tool, not shared with the sync path)
# ===========================================================================

def create_batch_job(conn, run_name, job_name):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO batch_jobs (run_name, job_name, status) VALUES (%s,%s,'preparing')",
            (run_name, job_name),
        )
        return cur.lastrowid


def get_batch_job(conn, job_name):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM batch_jobs WHERE job_name=%s", (job_name,))
        return cur.fetchone()


def get_latest_preparing_job(conn, run_name):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM batch_jobs WHERE run_name=%s AND status='preparing' "
            "ORDER BY created_at DESC LIMIT 1",
            (run_name,),
        )
        return cur.fetchone()


def list_batch_jobs(conn, run_name=None):
    with conn.cursor() as cur:
        if run_name:
            cur.execute(
                "SELECT * FROM batch_jobs WHERE run_name=%s ORDER BY created_at DESC",
                (run_name,),
            )
        else:
            cur.execute("SELECT * FROM batch_jobs ORDER BY created_at DESC")
        return cur.fetchall()


def update_batch_job(conn, batch_job_id, **fields):
    if not fields:
        return
    set_clause = ", ".join(f"{k}=%s" for k in fields)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE batch_jobs SET {set_clause} WHERE batch_job_id=%s",
            (*fields.values(), batch_job_id),
        )


def get_run_config(conn, run_name):
    """Load a run's feature config back out of the DB (stored at prepare
    time), so import/cancel/status don't need --config re-supplied."""
    with conn.cursor() as cur:
        cur.execute("SELECT config_yaml FROM runs WHERE run_name=%s", (run_name,))
        row = cur.fetchone()
        return yaml.safe_load(row["config_yaml"]) if row else None


# ===========================================================================
# Converse record construction
# ===========================================================================

def build_additional_model_request_fields(config, disable_thinking):
    """Merge the config's llm.additional_model_request_fields passthrough
    with --disable-thinking sugar. Opt-in, off by default: Bedrock Batch's
    Converse validation has been observed rejecting
    additionalModelRequestFields.thinking outright ("extraneous key") for a
    model that accepts the identical field fine via a live converse() call
    -- so a payload validated with query_bedrock.py is not proof it'll be
    accepted by a real batch job, and forcing this by default risks a
    100%-failed run. Generic by design otherwise -- Bedrock hosts many model
    families with their own reasoning-control conventions, so this avoids
    hardcoding a model-name table that would rot as AWS ships new models
    (see README for further per-model caveats, e.g. adaptive-thinking-only
    Claude models reject an explicit 'disabled' even via live Converse)."""
    fields = config.get("llm", {}).get("additional_model_request_fields") or {}
    fields = apply_disable_thinking(fields, disable_thinking)
    return fields or None


# ===========================================================================
# JSONL file writer (splits on record count and byte size quotas)
# ===========================================================================

class BatchFileWriter:
    """Writes JSONL records across multiple part files, rolling to a new
    file when either --batch-max-records or --batch-max-file-bytes would be
    exceeded. Never splits mid-record."""

    def __init__(self, work_dir, max_records, max_bytes):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.part_index = 0
        self._file = None
        self.records_in_file = 0
        self.bytes_in_file = 0
        self.total_records = 0
        self.written_paths = []

    def _open_new(self):
        if self._file:
            self._file.close()
        self.part_index += 1
        path = self.work_dir / f"part-{self.part_index:04d}.jsonl"
        self._file = open(path, "w", encoding="utf-8")
        self.written_paths.append(path)
        self.records_in_file = 0
        self.bytes_in_file = 0

    def write(self, record):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if (
            self._file is None
            or self.records_in_file >= self.max_records
            or self.bytes_in_file + line_bytes > self.max_bytes
        ):
            self._open_new()
        self._file.write(line)
        self.records_in_file += 1
        self.bytes_in_file += line_bytes
        self.total_records += 1

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


# ===========================================================================
# S3 helpers
# ===========================================================================

def parse_s3_uri(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri!r}")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    return bucket, prefix


def list_s3_objects(s3, bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def delete_s3_keys(s3, bucket, keys):
    """Batch-delete via DeleteObjects (max 1000 keys per call)."""
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in batch]})


# ===========================================================================
# prepare
# ===========================================================================

def sanitize_job_name(name):
    """Bedrock jobName must match [a-zA-Z0-9]{1,63}(-*[a-zA-Z0-9+\\-.]){0,63}."""
    name = re.sub(r"[^a-zA-Z0-9+.\-]", "-", name)
    name = re.sub(r"^-+", "", name)
    return name[:63] or "docfeatures-batch"


def cmd_prepare(args):
    config = load_and_validate_config(args.config)

    if args.model_invocation_type != "Converse":
        print(
            "Only --model-invocation-type Converse is supported by "
            "docfeatures_batch.py prepare. InvokeModel needs a different "
            "payload per model family and isn't implemented.",
            file=sys.stderr,
        )
        sys.exit(1)

    temperature = args.temperature if args.temperature is not None else config.get("llm", {}).get("temperature", 0.0)
    additional_fields = build_additional_model_request_fields(config, args.disable_thinking)

    job_name = sanitize_job_name(args.job_name or f"{args.run_name}-{int(time.time())}")

    conn = get_connection()

    config_hash = hashlib.sha256(yaml.dump(config).encode()).hexdigest()
    get_or_create_run(conn, args.run_name, config, config_hash, host="bedrock", model=None, temperature=temperature)

    existing = get_batch_job(conn, job_name)
    if existing:
        print(f"Job name '{job_name}' already exists (status={existing['status']}). "
              f"Pick a different --job-name.", file=sys.stderr)
        sys.exit(1)
    batch_job_id = create_batch_job(conn, args.run_name, job_name)

    finished = get_finished_paths(conn, args.run_name, retry_errors=args.retry_errors)

    corpus_paths = args.corpus
    if not corpus_paths:
        yaml_corpus = config.get("corpus", [])
        if isinstance(yaml_corpus, str):
            yaml_corpus = [yaml_corpus]
        corpus_paths = yaml_corpus if yaml_corpus else None

    filter_config = config.get("filter")
    if filter_config:
        validate_filter(conn, filter_config)
        all_files = get_filtered_paths(conn, filter_config)
        if corpus_paths:
            corpus_files = set(discover_files(corpus_paths))
            all_files = [f for f in all_files if f in corpus_files]
    else:
        if not corpus_paths:
            print(
                "No corpus specified. Provide --corpus, 'corpus' in the YAML "
                "config, or a 'filter' section.",
                file=sys.stderr,
            )
            sys.exit(1)
        all_files = list(discover_files(corpus_paths))

    pending = filter_pending(all_files, finished)
    if args.limit:
        pending = pending[: args.limit]

    print(f"Job          : {job_name}", file=sys.stderr)
    print(f"Run          : {args.run_name}", file=sys.stderr)
    print(f"Matched      : {len(all_files)}", file=sys.stderr)
    print(f"Already done : {len(finished)}", file=sys.stderr)
    print(f"Staging      : {len(pending)}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    work_dir = Path(args.work_dir) / args.run_name / job_name / "input"
    writer = BatchFileWriter(work_dir, args.batch_max_records, args.batch_max_file_bytes)

    features_config = config["features"]
    staged = 0
    auto_completed = 0
    errors = 0

    for file_path in pending:
        doc_id = None
        try:
            raw_bytes = Path(file_path).read_bytes()
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = raw_bytes.decode("cp1252")
                except Exception:
                    text = raw_bytes.decode("utf-8", errors="replace")

            fhash = file_hash(file_path)
            fsize = os.path.getsize(file_path)
            file_id = get_or_create_file(conn, file_path, fhash, fsize)

            raw_chunks = build_chunks(text, target_chars=args.chunk_size)
            clean_chunks = [c for c in (sanitize_text(rc) for rc in raw_chunks) if c]

            if not clean_chunks:
                # Document sanitized to nothing (e.g. pure boilerplate) -- same
                # outcome as the sync tool: complete immediately with default
                # (empty) feature values, never enters the batch at all.
                doc_id = upsert_document(conn, args.run_name, file_id, 0, status="processing")
                merged = merge_chunk_results([], features_config)
                save_document_features(conn, doc_id, file_id, merged, features_config)
                mark_document(conn, doc_id, "complete", elapsed=0)
                auto_completed += 1
                continue

            total_chunks = len(clean_chunks)
            doc_id = upsert_document(
                conn, args.run_name, file_id, total_chunks,
                status="batch_pending", batch_job_id=batch_job_id,
            )

            for ci, clean_chunk in enumerate(clean_chunks):
                chunk_info = (ci + 1, total_chunks) if total_chunks > 1 else None
                prompt = build_prompt(features_config, clean_chunk, chunk_info)
                model_input = build_converse_input(prompt, temperature, additional_fields)
                writer.write({"recordId": f"{doc_id}:{ci}", "modelInput": model_input})

            staged += 1

        except Exception as e:
            # One bad file shouldn't crash a prepare run that may have
            # already staged thousands of others -- log it, clean up any
            # half-made document_runs row (so it doesn't sit stuck in
            # 'batch_pending' forever, unreachable by any future run), and
            # move on. Mirrors docfeatures.py's per-document error handling.
            errors += 1
            if doc_id is not None:
                mark_document(conn, doc_id, "error", error=str(e))
            print(f"  [ERROR] {Path(file_path).name}: {e}", file=sys.stderr)

    writer.close()
    update_batch_job(conn, batch_job_id, total_records=writer.total_records)

    print(f"Staged       : {staged} document(s), {writer.total_records} record(s) "
          f"across {len(writer.written_paths)} file(s)", file=sys.stderr)
    if auto_completed:
        print(f"Auto-completed (empty after sanitization): {auto_completed}", file=sys.stderr)
    if errors:
        print(f"Errors (skipped, marked 'error'): {errors}", file=sys.stderr)
    for p in writer.written_paths:
        print(f"  {p}", file=sys.stderr)

    if writer.total_records == 0:
        print("\nNothing to submit -- no records were staged.", file=sys.stderr)
    elif writer.total_records < args.batch_min_records:
        print(
            f"\nWARNING: {writer.total_records} record(s) is below the "
            f"--batch-min-records quota of {args.batch_min_records}. "
            f"Bedrock will reject this job at submit time unless you stage "
            f"more documents or lower the quota (if your account's actual "
            f"minimum differs).",
            file=sys.stderr,
        )
    else:
        print(f"\nNext: python {Path(sys.argv[0]).name} submit --job-name {job_name} "
              f"--model-id <bedrock-model-id> --s3-bucket <bucket> --role-arn <role-arn>",
              file=sys.stderr)

    conn.close()


# ===========================================================================
# submit
# ===========================================================================

def cmd_submit(args):
    conn = get_connection()

    job = get_batch_job(conn, args.job_name) if args.job_name else get_latest_preparing_job(conn, args.run_name)
    if not job:
        print("No matching batch job found. Run `prepare` first, or check --job-name/--run-name.", file=sys.stderr)
        sys.exit(1)
    if job["status"] != "preparing":
        print(f"Job '{job['job_name']}' is in status '{job['status']}', not 'preparing' -- "
              f"it's already been submitted (or cancelled).", file=sys.stderr)
        sys.exit(1)

    work_dir = Path(args.work_dir) / job["run_name"] / job["job_name"] / "input"
    jsonl_files = sorted(work_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No staged .jsonl files found under {work_dir}. Run `prepare` first.", file=sys.stderr)
        sys.exit(1)

    if not (24 <= args.timeout_hours <= 168):
        print("--timeout-hours must be between 24 and 168 (Bedrock's valid range).", file=sys.stderr)
        sys.exit(1)

    s3 = boto3.client("s3", region_name=args.region)
    bedrock = boto3.client("bedrock", region_name=args.region)

    s3_prefix = f"{args.s3_prefix.strip('/')}/{job['job_name']}"
    for f in jsonl_files:
        key = f"{s3_prefix}/input/{f.name}"
        print(f"  Uploading {f} -> s3://{args.s3_bucket}/{key}", file=sys.stderr)
        s3.upload_file(str(f), args.s3_bucket, key)

    s3_input_uri = f"s3://{args.s3_bucket}/{s3_prefix}/input/"
    s3_output_uri = f"s3://{args.s3_bucket}/{s3_prefix}/output/"

    resp = bedrock.create_model_invocation_job(
        jobName=job["job_name"],
        roleArn=args.role_arn,
        modelId=args.model_id,
        modelInvocationType=job["model_invocation_type"],
        inputDataConfig={"s3InputDataConfig": {"s3Uri": s3_input_uri}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": s3_output_uri}},
        timeoutDurationInHours=args.timeout_hours,
    )
    job_arn = resp["jobArn"]

    update_batch_job(
        conn, job["batch_job_id"],
        job_arn=job_arn, model_id=args.model_id,
        s3_input_uri=s3_input_uri, s3_output_uri=s3_output_uri,
        role_arn=args.role_arn, status="Submitted",
        submitted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"\nSubmitted '{job['job_name']}' -> {job_arn}", file=sys.stderr)
    print(f"Check progress with: python {Path(sys.argv[0]).name} status --job-name {job['job_name']}", file=sys.stderr)
    conn.close()


# ===========================================================================
# status / list-jobs
# ===========================================================================

def _print_jobs_table(jobs):
    print(f"{'Job Name':<30} {'Run Name':<20} {'Status':<20} {'Records':>15}  Created")
    print("-" * 105)
    for j in jobs:
        records = f"{j.get('total_records') or 0}"
        print(f"{j['job_name']:<30} {j['run_name']:<20} {j['status']:<20} {records:>15}  {j['created_at']}")


def cmd_status(args):
    conn = get_connection()
    jobs = [get_batch_job(conn, args.job_name)] if args.job_name else list_batch_jobs(conn, args.run_name)
    jobs = [j for j in jobs if j]
    if not jobs:
        print("No matching batch job(s) found.")
        conn.close()
        return

    bedrock = boto3.client("bedrock", region_name=args.region)
    for job in jobs:
        if not job["job_arn"]:
            continue  # never submitted (still 'preparing') or already terminal-local ('cancelled')
        resp = bedrock.get_model_invocation_job(jobIdentifier=job["job_arn"])
        update_batch_job(
            conn, job["batch_job_id"],
            status=resp["status"],
            last_checked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            error_message=resp.get("message"),
            total_records=resp.get("totalRecordCount") or job.get("total_records"),
        )
        job["status"] = resp["status"]
        job["total_records"] = resp.get("totalRecordCount") or job.get("total_records")
        print(
            f"{job['job_name']}: {resp['status']}  "
            f"(processed {resp.get('processedRecordCount', 0)}/{resp.get('totalRecordCount', 0)}, "
            f"{resp.get('successRecordCount', 0)} ok, {resp.get('errorRecordCount', 0)} errored)"
        )
        if resp.get("message"):
            print(f"  {resp['message']}")
        if resp["status"] in IMPORTABLE_STATUSES:
            print(f"  Ready to import: python {Path(sys.argv[0]).name} import --job-name {job['job_name']}")

    conn.close()


def cmd_list_jobs(args):
    conn = get_connection()
    jobs = list_batch_jobs(conn, args.run_name)
    conn.close()
    if not jobs:
        print("No batch jobs found.")
        return
    _print_jobs_table(jobs)


# ===========================================================================
# import
# ===========================================================================

def cmd_import(args):
    conn = get_connection()
    job = get_batch_job(conn, args.job_name)
    if not job:
        print(f"No batch job named '{args.job_name}'.", file=sys.stderr)
        sys.exit(1)
    if job["imported_at"] and not args.force and not args.dry_run:
        print(f"Job '{args.job_name}' was already imported at {job['imported_at']}. "
              f"Use --force to re-import, or --dry-run to view without re-importing.", file=sys.stderr)
        sys.exit(1)
    if job["status"] not in IMPORTABLE_STATUSES:
        print(f"Job '{args.job_name}' is in status '{job['status']}', not "
              f"Completed/PartiallyCompleted. Run `status` first.", file=sys.stderr)
        sys.exit(1)

    # --dry-run never writes to the DB and always shows at least as much
    # detail as -vv (successes AND per-record errors as they're found, not
    # just counted in the summary) -- it exists specifically to let you
    # inspect a job's results before committing to a real import.
    verbosity = max(args.verbose, 2 if args.dry_run else 0)
    tag = "[DRY RUN] " if args.dry_run else ""

    config = get_run_config(conn, job["run_name"])
    features_config = config["features"]

    bucket, prefix = parse_s3_uri(job["s3_output_uri"])
    s3 = boto3.client("s3", region_name=args.region)

    chunk_data = {}   # doc_id -> {chunk_index: parsed_json}
    doc_errors = {}   # doc_id -> [error strings]
    records_seen = 0
    integrity_errors = 0   # writes rejected by a FK constraint -- see save_chunk_result below

    for key in list_s3_objects(s3, bucket, prefix):
        if key.endswith("manifest.json.out"):
            continue
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # not an output record line (defensive -- unexpected file in the output prefix)

            record_id = obj.get("recordId", "")
            if ":" not in record_id:
                continue
            doc_id_str, chunk_index_str = record_id.rsplit(":", 1)
            try:
                doc_id, chunk_index = int(doc_id_str), int(chunk_index_str)
            except ValueError:
                continue
            records_seen += 1

            if "error" in obj:
                err = f"chunk {chunk_index}: {obj['error'].get('errorMessage', obj['error'])}"
                doc_errors.setdefault(doc_id, []).append(err)
                if verbosity >= 2:
                    print(f"  {tag}[ERROR] doc_id={doc_id} {err}", file=sys.stderr)
                continue

            try:
                raw_text = extract_converse_text(obj["modelOutput"])
                parsed = parse_json_response(raw_text)
                validate_enum_values(parsed, features_config)
            except (KeyError, ValueError) as e:
                err = f"chunk {chunk_index}: {e}"
                doc_errors.setdefault(doc_id, []).append(err)
                if verbosity >= 2:
                    print(f"  {tag}[ERROR] doc_id={doc_id} {err}", file=sys.stderr)
                continue

            if not args.dry_run:
                try:
                    save_chunk_result(conn, doc_id, chunk_index, json.dumps(parsed))
                except pymysql.err.IntegrityError as e:
                    # FK violation: no document_runs row exists for this doc_id anymore.
                    # Can happen if the same file got re-staged (new doc_id) by a later
                    # `prepare` for this run before this job's import got around to it --
                    # the old doc_id's document_runs row was deleted out from under this
                    # job. We can't recover a file name here (that lookup goes through the
                    # now-gone document_runs row); report what we have and move on instead
                    # of taking down the rest of the import.
                    integrity_errors += 1
                    print(f"  [ERROR] doc_id={doc_id} chunk={chunk_index}: could not save chunk "
                          f"result -- no document_runs row for this doc_id (likely re-staged by "
                          f"a later run before this import completed): {e}", file=sys.stderr)
                    continue
            chunk_data.setdefault(doc_id, {})[chunk_index] = parsed

    with conn.cursor() as cur:
        cur.execute(
            "SELECT dr.doc_id, dr.file_id, dr.total_chunks, f.file_path FROM document_runs dr "
            "JOIN files f ON dr.file_id = f.file_id "
            "WHERE dr.batch_job_id=%s AND dr.status='batch_pending'",
            (job["batch_job_id"],),
        )
        pending_docs = cur.fetchall()

    completed = 0
    errored = 0
    for row in pending_docs:
        doc_id, file_id, total_chunks = row["doc_id"], row["file_id"], row["total_chunks"]
        if doc_id in doc_errors:
            if not args.dry_run:
                mark_document(conn, doc_id, "error", error="; ".join(doc_errors[doc_id][:5]))
            errored += 1
            continue
        got = chunk_data.get(doc_id, {})
        if len(got) == total_chunks:
            ordered = [got[i] for i in range(total_chunks)]
            merged = merge_chunk_results(ordered, features_config)
            if not args.dry_run:
                try:
                    save_document_features(conn, doc_id, file_id, merged, features_config)
                    mark_document(conn, doc_id, "complete")
                except pymysql.err.IntegrityError as e:
                    # Same race as save_chunk_result above, just later: this doc_id's
                    # document_runs row existed when `pending_docs` was fetched but was
                    # deleted (re-staged by a concurrent prepare/run) before we got here.
                    integrity_errors += 1
                    print(f"  [ERROR] doc_id={doc_id} ({Path(row['file_path']).name}): could not "
                          f"save document features -- document_runs row no longer exists: {e}",
                          file=sys.stderr)
                    continue
            completed += 1
            if verbosity >= 1:
                feat_str = "  ".join(f"{k}={fmt_feature_value(v)}" for k, v in merged.items())
                print(f"  {tag}[{completed + errored}/{len(pending_docs)}] {Path(row['file_path']).name}  "
                      f"({total_chunks} chunk{'s' if total_chunks != 1 else ''})  {feat_str}")
        else:
            err = f"only {len(got)}/{total_chunks} chunks returned by Bedrock"
            if not args.dry_run:
                mark_document(conn, doc_id, "error", error=err)
            errored += 1
            if verbosity >= 2:
                print(f"  {tag}[ERROR] doc_id={doc_id} ({Path(row['file_path']).name}) {err}", file=sys.stderr)

    if not args.dry_run:
        update_batch_job(conn, job["batch_job_id"], imported_at=time.strftime("%Y-%m-%d %H:%M:%S"))

    verb = "Would import" if args.dry_run else "Imported"
    print(f"{tag}{verb} job '{args.job_name}': {records_seen} record(s) read, "
          f"{completed} document(s) complete, {errored} document(s) errored"
          + (f", {integrity_errors} skipped (database integrity error)" if integrity_errors else "")
          + ".")
    conn.close()


# ===========================================================================
# cancel
# ===========================================================================

def cmd_cancel(args):
    conn = get_connection()
    job = get_batch_job(conn, args.job_name)
    if not job:
        print(f"No batch job named '{args.job_name}'.", file=sys.stderr)
        sys.exit(1)

    if job["job_arn"] and job["status"] in ACTIVE_STATUSES:
        bedrock = boto3.client("bedrock", region_name=args.region)
        bedrock.stop_model_invocation_job(jobIdentifier=job["job_arn"])
        print(f"Requested stop on AWS for '{job['job_name']}'.", file=sys.stderr)

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_runs WHERE batch_job_id=%s AND status='batch_pending'",
            (job["batch_job_id"],),
        )
        returned = cur.rowcount

    update_batch_job(conn, job["batch_job_id"], status="cancelled")
    print(f"Cancelled '{job['job_name']}': {returned} document(s) returned to the "
          f"pending pool for run '{job['run_name']}'.")
    conn.close()


# ===========================================================================
# cleanup
# ===========================================================================

def cmd_cleanup(args):
    conn = get_connection()
    job = get_batch_job(conn, args.job_name)
    if not job:
        print(f"No batch job named '{args.job_name}'.", file=sys.stderr)
        sys.exit(1)

    if job["status"] in ACTIVE_STATUSES:
        print(f"Job '{args.job_name}' is still '{job['status']}' on AWS -- refusing to clean up an "
              f"in-progress job. `cancel` it first (which also stops it), or wait for it to finish.",
              file=sys.stderr)
        sys.exit(1)

    tag = "[DRY RUN] " if args.dry_run else ""
    s3 = None

    if not args.keep_local:
        work_dir = Path(args.work_dir) / job["run_name"] / job["job_name"]
        if work_dir.exists():
            print(f"{tag}Removing local directory: {work_dir}")
            if not args.dry_run:
                shutil.rmtree(work_dir)
        else:
            print(f"Local directory already gone: {work_dir}")

    for label, keep, uri_field in (
        ("input", args.keep_s3_input, "s3_input_uri"),
        ("output", args.keep_s3_output, "s3_output_uri"),
    ):
        if keep:
            continue
        uri = job[uri_field]
        if not uri:
            print(f"No S3 {label} URI recorded for this job -- nothing to remove.")
            continue
        if s3 is None:
            s3 = boto3.client("s3", region_name=args.region)
        bucket, prefix = parse_s3_uri(uri)
        keys = list(list_s3_objects(s3, bucket, prefix))
        print(f"{tag}Removing {len(keys)} S3 {label} object(s): {uri}")
        if not args.dry_run and keys:
            delete_s3_keys(s3, bucket, keys)

    conn.close()


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AWS Bedrock batch inference for docfeatures (prepare / submit / status / import / cancel).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s prepare -c features.yaml --corpus /data/notes/ -r v1 -n 100
  %(prog)s submit --job-name v1-20260814 --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0 \\
      --s3-bucket my-batch-bucket --role-arn arn:aws:iam::123456789012:role/BedrockBatchRole
  %(prog)s status --job-name v1-20260814
  %(prog)s import --job-name v1-20260814
  %(prog)s cancel --job-name v1-20260814
  %(prog)s cleanup --job-name v1-20260814
  %(prog)s list-jobs -r v1
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Chunk documents and stage them as local .jsonl file(s).")
    p_prepare.add_argument("-c", "--config", required=True, help="YAML feature-definition file.")
    p_prepare.add_argument("-r", "--run-name", required=True, help="Run name (shared with docfeatures.py).")
    p_prepare.add_argument("--corpus", action="append", default=None, help="Document directory or file. Repeatable.")
    p_prepare.add_argument("-n", "--limit", type=int, help="Stage at most N documents.")
    p_prepare.add_argument("--chunk-size", type=int, default=CHUNK_TARGET_CHARS, metavar="CHARS")
    p_prepare.add_argument("--model-invocation-type", choices=["Converse", "InvokeModel"], default="Converse",
                            help="Only Converse is currently implemented.")
    p_prepare.add_argument("--temperature", type=float, default=None)
    p_prepare.add_argument("--disable-thinking", action="store_true",
                            help="Set additionalModelRequestFields.thinking={type: disabled}. Off by "
                                 "default -- Bedrock Batch's Converse validation has been observed "
                                 "rejecting this field outright ('extraneous key') for a model that "
                                 "accepts it fine via a live query_bedrock.py invoke, so verify with a "
                                 "small (-n 100) batch test before relying on it for a full run.")
    p_prepare.add_argument("--retry-errors", action="store_true",
                            help="Re-stage documents that errored in a previous run (including ones from "
                                 "a batch job that already ran through `import`). Same semantics as "
                                 "docfeatures.py --retry-errors.")
    p_prepare.add_argument("--job-name", help="Defaults to '<run-name>-<unix-timestamp>'.")
    p_prepare.add_argument("--batch-min-records", type=int, default=DEFAULT_BATCH_MIN_RECORDS,
                            help="Warn if staged records fall below this (default: %(default)s).")
    p_prepare.add_argument("--batch-max-records", type=int, default=DEFAULT_BATCH_MAX_RECORDS,
                            help="Split into multiple files above this many records (default: %(default)s).")
    p_prepare.add_argument("--batch-max-file-bytes", type=int, default=DEFAULT_BATCH_MAX_FILE_BYTES,
                            help="Split into multiple files above this size (default: %(default)s).")
    p_prepare.add_argument("--work-dir", default=DEFAULT_WORK_DIR, help="Local staging directory.")
    p_prepare.set_defaults(func=cmd_prepare)

    p_submit = sub.add_parser("submit", help="Upload staged .jsonl file(s) to S3 and create the Bedrock job.")
    p_submit.add_argument("--job-name", help="Job to submit. Defaults to the most recent 'preparing' job for --run-name.")
    p_submit.add_argument("-r", "--run-name", help="Used to find the job if --job-name is omitted.")
    p_submit.add_argument("--model-id", default=DEFAULT_BEDROCK_MODEL_ID,
                           help="Bedrock model ID or ARN to run the job against "
                                "(default: BEDROCK_MODEL_ID from .env).")
    p_submit.add_argument("--s3-bucket", default=DEFAULT_BEDROCK_S3_BUCKET,
                           help="(default: BEDROCK_S3_BUCKET from .env)")
    p_submit.add_argument("--s3-prefix", default="docfeatures-batch")
    p_submit.add_argument("--role-arn", default=DEFAULT_BEDROCK_ROLE_ARN,
                           help="IAM service role ARN, see README for required policy "
                                "(default: BEDROCK_ROLE_ARN from .env).")
    p_submit.add_argument("--region", help="AWS region (default: boto3's normal resolution chain).")
    p_submit.add_argument("--timeout-hours", type=int, default=24, help="24-168 (default: %(default)s).")
    p_submit.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    p_submit.set_defaults(func=cmd_submit)

    p_status = sub.add_parser("status", help="Refresh and print job status from AWS.")
    p_status.add_argument("--job-name")
    p_status.add_argument("-r", "--run-name")
    p_status.add_argument("--region")
    p_status.set_defaults(func=cmd_status)

    p_list = sub.add_parser("list-jobs", help="List locally-known batch jobs (no AWS call).")
    p_list.add_argument("-r", "--run-name")
    p_list.set_defaults(func=cmd_list_jobs)

    p_import = sub.add_parser("import", help="Pull a Completed/PartiallyCompleted job's results into MySQL.")
    p_import.add_argument("--job-name", required=True)
    p_import.add_argument("--region")
    p_import.add_argument("--force", action="store_true", help="Re-import a job that was already imported.")
    p_import.add_argument("-v", "--verbose", action="count", default=0,
                           help="Print each successfully-imported document's merged feature values as "
                                "they're written, same style as docfeatures.py's progress line. Repeat "
                                "(-vv) to also print per-record parse/validation errors as they're found, "
                                "not just counted in the final summary.")
    p_import.add_argument("--dry-run", action="store_true",
                           help="Parse and print the job's output (implies -vv) without writing anything "
                                "to the database -- view results before committing to a real import. "
                                "Ignores --force's guard since nothing is changed.")
    p_import.set_defaults(func=cmd_import)

    p_cancel = sub.add_parser("cancel", help="Stop the job (if active) and return its documents to the pending pool.")
    p_cancel.add_argument("--job-name", required=True)
    p_cancel.add_argument("--region")
    p_cancel.set_defaults(func=cmd_cancel)

    p_cleanup = sub.add_parser(
        "cleanup", help="Delete a job's local staged .jsonl file(s) and/or S3 input/output objects.",
    )
    p_cleanup.add_argument("--job-name", required=True)
    p_cleanup.add_argument("--region")
    p_cleanup.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    p_cleanup.add_argument("--keep-local", action="store_true", help="Don't delete the local staged .jsonl directory.")
    p_cleanup.add_argument("--keep-s3-input", action="store_true", help="Don't delete the job's S3 input objects.")
    p_cleanup.add_argument("--keep-s3-output", action="store_true", help="Don't delete the job's S3 output objects.")
    p_cleanup.add_argument("--dry-run", action="store_true", help="Report what would be deleted without deleting.")
    p_cleanup.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()

    if args.command == "submit":
        if not args.job_name and not args.run_name:
            parser.error("submit requires --job-name or --run-name.")
        missing = [
            flag for flag, val in [
                ("--model-id", args.model_id),
                ("--s3-bucket", args.s3_bucket),
                ("--role-arn", args.role_arn),
            ] if not val
        ]
        if missing:
            parser.error(
                f"Missing {', '.join(missing)}. Pass on the command line, or set "
                f"BEDROCK_MODEL_ID/BEDROCK_S3_BUCKET/BEDROCK_ROLE_ARN in .env."
            )

    try:
        args.func(args)
    except ClientError as e:
        print(f"AWS error: {e.response.get('Error', {}).get('Message', e)}", file=sys.stderr)
        sys.exit(1)
    except BotoCoreError as e:
        print(f"AWS error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
