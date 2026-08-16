# Batch Processing (AWS Bedrock)

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

## Workflow

`--model-id`/`--s3-bucket`/`--role-arn` for `submit` can be set once via `.env` (`BEDROCK_MODEL_ID`,
`BEDROCK_S3_BUCKET`, `BEDROCK_ROLE_ARN` — see `env.example`) instead of repeating them on every
invocation; CLI flags still override them when passed.

```bash
# 0. Sanity-check the model ID and your access to it with a single cheap call
#    (see "Testing model access" below) before committing to a min-100-record batch job
python query_bedrock.py -m us.anthropic.claude-haiku-4-5-20251001-v1:0 -p "hello"

# 1. Chunk documents and stage them as local .jsonl file(s) — no AWS calls yet
python docfeatures_batch.py prepare -c features.yaml --corpus /data/notes/ -r v1 -n 100

# 2. Upload to S3 and create the Bedrock batch inference job
python docfeatures_batch.py submit --job-name v1-20260814
# (or, with everything explicit instead of relying on .env defaults:)
python docfeatures_batch.py submit --job-name v1-20260814 \
    --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0 \
    --s3-bucket my-batch-bucket --role-arn arn:aws:iam::123456789012:role/BedrockBatchRole

# 3. Poll status (updates the local DB's record of the job too)
python docfeatures_batch.py status --job-name v1-20260814

# 4. Once Completed/PartiallyCompleted, preview results without touching the database...
python docfeatures_batch.py import --job-name v1-20260814 --dry-run
# ...then pull them in for real
python docfeatures_batch.py import --job-name v1-20260814
python docfeatures_batch.py import --job-name v1-20260814 -v    # also print each document's feature values as they're written
python docfeatures_batch.py import --job-name v1-20260814 -vv   # also print per-record parse/validation errors as they're found

# 5. Once you're done with a job, delete its local .jsonl staging and S3 input/output objects
python docfeatures_batch.py cleanup --job-name v1-20260814 --dry-run   # preview first
python docfeatures_batch.py cleanup --job-name v1-20260814

# List locally-known jobs (no AWS call)
python docfeatures_batch.py list-jobs -r v1

# Stop the job (if still running) and return its documents to the pending pool --
# for a job you decide not to run, or one that failed
python docfeatures_batch.py cancel --job-name v1-20260814
```

`cleanup` refuses to touch a job that's still active on AWS (`Submitted`/`Validating`/`Scheduled`/
`InProgress` — run `status` first if unsure, or `cancel` it instead). By default it deletes all three of
the local staged `.jsonl` directory, the S3 input objects, and the S3 output objects; opt out of any of
those individually with `--keep-local`, `--keep-s3-input`, `--keep-s3-output`. It only touches files —
the `batch_jobs`/`document_runs` database rows are left alone as a historical record.

Each `prepare` command creates a `batch_jobs` row (status `preparing`) and moves the documents it stages
into `document_runs.status = 'batch_pending'` — a state the sync tool and other batch `prepare` runs both
treat as claimed, so nothing gets double-processed across the two tools or concurrent batches for the same
run. `cancel` is the release valve: it deletes those `batch_pending` rows (returning the files to the
pending pool for the run) whether the job was ever submitted, is still running, or failed on AWS.

`--batch-max-records` caps the **whole job's total record count** — every chunk across every staged
document counts as one record, not one per document, so a `-n`/`--limit` document count doesn't
translate 1:1 into records. `prepare` tracks the running total as it stages and stops *before* it would
push the job over the limit, rather than staging everything `-n` asked for and letting the job fail
validation at submit time. If it stops early, the summary says how many documents were actually staged
out of how many were requested — the rest are left pending (not touched, not errored) for a follow-up
`prepare` (same `-r`, a new `--job-name`) to pick up into another job. A single document whose own chunk
count exceeds `--batch-max-records` can never fit in any job at that limit and is marked `'error'`
immediately with a message saying so, rather than silently produced anyway or hung on forever.
`--batch-max-file-bytes` is a separate, per-*file* concern (multiple `.jsonl` files can make up one job)
and warns (without blocking) if the total is under `--batch-min-records`. The defaults (100 min / 100,000
max records per job, ~1GB max per file) match this project's confirmed AWS Service Quotas — override them
via flags if your account's quotas differ.

## Testing model access (`query_bedrock.py`)

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

## Model "thinking" / reasoning

`prepare --disable-thinking` sets `additionalModelRequestFields.thinking={type: disabled}`. **Off by
default** — reasoning tokens are wasted cost in a batch run (each chunk gets one shot with no retry, and
`merge_chunk_results` discards them regardless), so disabling looks like a good default in theory, but in
practice we hit a real case where it broke every record in a live job: Bedrock Batch's Converse validation
rejected `additionalModelRequestFields.thinking` outright —
*`Malformed input request: #: extraneous key [thinking] is not permitted`* — for a model
(`us.anthropic.claude-haiku-4-5-20251001-v1:0`) that accepts the exact same field without complaint via a
live `query_bedrock.py` invoke. **Batch's Converse implementation is not a strict match for live
Converse's accepted fields**, at least not for this one, and there's no reliable way to know which models
are affected ahead of time short of testing.

Given that, treat `--disable-thinking` as something to verify with a small (`-n 100`, the minimum job
size) real batch test before relying on it for a full run — a successful `query_bedrock.py` invoke with
the same flag is *not* sufficient proof it'll be accepted by an actual batch job. If a job comes back with
every record erroring on the same "extraneous key" message, drop `--disable-thinking` and retry (see
"Cleaning up and retrying" below).

Separately, **adaptive-thinking-only models reject an explicit `disabled`** even via live Converse (as of
this writing: Claude Mythos 5, Claude Fable 5, Claude Opus 4.7, Claude Mythos Preview) — don't pass
`--disable-thinking` for those at all. To cap (not fully disable) reasoning on one of those, use
`llm.additional_model_request_fields` in the feature-config YAML instead — passed straight through into
every record's Converse `additionalModelRequestFields`, so it works for any current or future Bedrock
model/provider without this project hardcoding a model-name table:

```yaml
llm:
  additional_model_request_fields:
    output_config:
      effort: low
```

Bedrock does **not** validate that `modelInput` matches a model's actual schema at submit time — a bad
config only shows up as a per-record `error` in the job's output, or (as above) an entire failed job.

## Cleaning up and retrying

A job that fails outright (e.g. IAM permissions) or completes with every record errored (e.g. a rejected
`additionalModelRequestFields` field) leaves its documents claimed and going nowhere on their own.
`import --dry-run` is the safe way to check first — it shows exactly what a real `import` would do
(including per-record errors at the detail `--dry-run` always shows) without writing anything to the
database, so you can tell a total loss from a partial success before committing to anything:

```bash
python docfeatures_batch.py import --job-name v1-20260814 --dry-run
```

If it shows no usable output, don't run a real `import` — `import` marks every document `'error'`, and
unlike `cancel`, that status is **not** automatically re-picked-up by a plain `prepare` (see
`--retry-errors` below). For a fully-failed job, go straight to:

```bash
python docfeatures_batch.py cancel --job-name <job-name>   # returns its documents to the pending pool
python docfeatures_batch.py prepare -c features.yaml --corpus /data/notes/ -r v1 -n 100   # re-stage, fixing whatever broke
python docfeatures_batch.py submit --job-name <new-name> ...
```

If a job *was* already imported (so its documents are sitting at `status='error'` in `document_runs`,
not `'batch_pending'` — `cancel` won't touch those), use `prepare --retry-errors` instead to re-stage them
(same semantics as `docfeatures.py --retry-errors`):

```bash
python docfeatures_batch.py prepare -c features.yaml --corpus /data/notes/ -r v1 --retry-errors -n 100
```

## One-time AWS setup

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

Permissions policy attached to that same role — **two statements, both required** (replace `{{BUCKET}}`).
S3 access alone isn't enough; the service role also has to be allowed to actually call the model, or jobs
fail partway through with `"Customer doesn't have permissions to invokeModel"` even though S3 upload/
validation succeeded:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3Access",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": ["arn:aws:s3:::{{BUCKET}}", "arn:aws:s3:::{{BUCKET}}/*"]
        },
        {
            "Sid": "InvokeModel",
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel"],
            "Resource": "*"
        }
    ]
}
```

**Figuring out the right `InvokeModel` resource ARN(s)** is the part that's easy to get wrong, and the
official docs undersell it as "optional... for inference profiles" — but several current Claude models
*require* an inference profile ID for on-demand/batch use in the first place (see "Testing model access"
above), which makes this non-optional for them in practice. If `--model-id` is a cross-region inference
profile (`us.`/`global.`-prefixed), you need the profile's own ARN *and* every regional foundation-model
ARN it can route to, or the job fails with the permissions error above the moment it tries to actually
invoke:

```bash
aws bedrock get-inference-profile --inference-profile-identifier us.anthropic.claude-haiku-4-5-20251001-v1:0
# returns the profile ARN plus each underlying regional foundation-model ARN -- list all of them
# as Resource entries in the InvokeModel statement above
```

If `--model-id` is a bare foundation-model ID instead (no `us.`/`global.` prefix), a single
`arn:aws:bedrock:{{REGION}}::foundation-model/{{MODEL_ID}}` resource entry is enough.

Copy the resulting role's ARN (`arn:aws:iam::{{ACCOUNT_ID}}:role/...`) — that's the `--role-arn` for
`docfeatures_batch.py submit`.

See [AWS's batch inference permissions docs](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-permissions.html)
and [service role setup](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html) for the
full/current version of both policies.
