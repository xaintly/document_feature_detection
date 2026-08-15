#!/usr/bin/env python3
"""
query_bedrock.py — Query an AWS Bedrock model via the Converse API

The Bedrock console's old "Model access" page has been retired (as of this
writing, AWS auto-enables serverless models and triggers a background
marketplace subscription on first use instead). This tool is the CLI
equivalent of poking a model in the Chat Playground: send it a prompt and
see what happens. A few roles in one:

  - Quick prompt testing against a Bedrock model (mirrors query.py, but
    for Bedrock instead of a local Ollama server).
  - Dumping the raw Converse response (--raw) when you need to see
    stopReason/usage/thinking blocks, not just the text.
  - Checking/requesting access ahead of time (--check-access / --subscribe)
    instead of finding out via a failed real call.
  - Just invoking the model at all: the first call to a model you haven't
    used before is what triggers Bedrock's background auto-subscription --
    this tool retries through the AccessDeniedException window for you
    (up to --timeout minutes) instead of making you re-run it by hand.

Note: some models (several current Claude models included) reject on-demand invocation by their
bare foundation-model ID and require a cross-region inference profile ID instead -- if -p/--raw
invocation fails with "Invocation of model ID ... with on-demand throughput isn't supported",
run `aws bedrock list-inference-profiles` (or the boto3 equivalent) and use the matching
`us.<provider>...` / `global.<provider>...` ID instead. --check-access/--subscribe use the bare
model ID either way -- marketplace agreements are keyed to the underlying model, not the profile.

Usage:
    python query_bedrock.py -m us.anthropic.claude-haiku-4-5-20251001-v1:0 -p "hello"
    python query_bedrock.py -m anthropic.claude-haiku-4-5-20251001-v1:0 --check-access
    python query_bedrock.py -m anthropic.claude-haiku-4-5-20251001-v1:0 --subscribe
    python query_bedrock.py -m us.anthropic.claude-haiku-4-5-20251001-v1:0 -p "hello" --raw
    echo "long document text..." | python query_bedrock.py -m <model-id> --disable-thinking
"""

import argparse
import json
import sys
import time

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    print(
        "query_bedrock.py requires boto3, which isn't installed.\n"
        "Install it with:\n"
        "  pip install -r requirements-batch.txt",
        file=sys.stderr,
    )
    sys.exit(1)

from lib.docfeatures_bedrock_lib import (
    apply_disable_thinking,
    build_converse_input,
    build_image_block,
    extract_converse_text,
)

DEFAULT_TIMEOUT_MINUTES = 5
RETRY_DELAY_SECS = 20


def _aws_error_message(e):
    """ClientError carries a structured .response; other botocore errors
    (NoCredentialsError, EndpointConnectionError, ...) don't -- fall back
    to str(e) for those."""
    if isinstance(e, ClientError):
        return e.response.get("Error", {}).get("Message", str(e))
    return str(e)


def cmd_check_access(bedrock, model_id):
    """Report whether model_id needs a marketplace agreement, without
    invoking it (no cost)."""
    try:
        resp = bedrock.list_foundation_model_agreement_offers(modelId=model_id)
    except (ClientError, BotoCoreError) as e:
        print(f"Could not check offers for '{model_id}': {_aws_error_message(e)}", file=sys.stderr)
        return
    offers = resp.get("offers", [])
    if not offers:
        print(f"No marketplace agreement offers found for '{model_id}' -- it's likely auto-enabled "
              f"with no separate subscription step. A plain invoke should work (may still take a few "
              f"minutes to auto-provision on first use).")
    else:
        print(f"{len(offers)} marketplace offer(s) available for '{model_id}':")
        for o in offers:
            print(f"  offerId={o['offerId']}")
        print("\nRun with --subscribe to accept an offer ahead of time.")


def cmd_subscribe(bedrock, model_id):
    """Accept model_id's marketplace agreement (if any) -- the closest
    equivalent to the retired console 'Model access' request button."""
    try:
        resp = bedrock.list_foundation_model_agreement_offers(modelId=model_id)
    except (ClientError, BotoCoreError) as e:
        print(f"Could not list offers for '{model_id}': {_aws_error_message(e)}", file=sys.stderr)
        return
    offers = resp.get("offers", [])
    if not offers:
        print(f"No offers found for '{model_id}' -- nothing to subscribe to (it's likely already "
              f"auto-enabled, or the model ID is wrong).")
        return
    try:
        bedrock.create_foundation_model_agreement(modelId=model_id, offerToken=offers[0]["offerToken"])
    except (ClientError, BotoCoreError) as e:
        print(f"Could not accept agreement for '{model_id}': {_aws_error_message(e)}", file=sys.stderr)
        return
    print(f"Accepted the marketplace agreement for '{model_id}'. Provisioning can still take a few "
          f"minutes -- try a plain invoke shortly.")
    print("\nNote: some Anthropic models additionally require a one-time 'intended use' form, which "
          "this tool doesn't submit for you -- see Bedrock console -> Model catalog -> the model.")


def main():
    parser = argparse.ArgumentParser(
        description="Query an AWS Bedrock model via the Converse API.",
        epilog="""
examples:
  %(prog)s -m us.anthropic.claude-haiku-4-5-20251001-v1:0 -p "hello"
  %(prog)s -m anthropic.claude-haiku-4-5-20251001-v1:0 --check-access
  %(prog)s -m anthropic.claude-haiku-4-5-20251001-v1:0 --subscribe
  %(prog)s -m us.anthropic.claude-haiku-4-5-20251001-v1:0 -p "hello" --raw --disable-thinking
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--model-id", required=True,
                         help="Bedrock model ID or ARN (same value docfeatures_batch.py submit --model-id takes).")
    parser.add_argument("-p", "--prompt", help="Prompt text. If omitted, reads from stdin.")
    parser.add_argument("-s", "--system", default=None, help="System prompt.")
    parser.add_argument("-a", "--attach", action="append", default=[],
                         help="Image file to attach (can be repeated). Supported: png, jpg/jpeg, gif, webp.")
    parser.add_argument("-e", "--temperature", type=float, default=None, help="Model temperature parameter.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max response tokens (default: %(default)s).")
    parser.add_argument("--disable-thinking", action="store_true",
                         help="Set additionalModelRequestFields.thinking={type: disabled}. Same caveats as "
                              "docfeatures_batch.py --disable-thinking (see README).")
    parser.add_argument("--extra-fields", metavar="JSON",
                         help="Raw JSON object merged into additionalModelRequestFields (applied before --disable-thinking).")
    parser.add_argument("--region", help="AWS region (default: boto3's normal resolution chain).")
    parser.add_argument(
        "-t", "--timeout", type=float, default=DEFAULT_TIMEOUT_MINUTES, metavar="MINUTES",
        help="Minutes to keep retrying if Bedrock is still auto-provisioning access to a model used "
             f"for the first time (0 to disable retrying, default: {DEFAULT_TIMEOUT_MINUTES}).",
    )
    parser.add_argument("--raw", action="store_true", help="Print the full raw Converse response JSON instead of just the text.")
    parser.add_argument("--check-access", action="store_true",
                         help="Report whether --model-id needs a marketplace agreement, without invoking it.")
    parser.add_argument("--subscribe", action="store_true",
                         help="Accept --model-id's marketplace agreement ahead of time, without invoking it "
                              "-- the closest replacement for the retired 'Model access' console page.")
    args = parser.parse_args()

    if args.check_access or args.subscribe:
        bedrock = boto3.client("bedrock", region_name=args.region)
        if args.check_access:
            cmd_check_access(bedrock, args.model_id)
        if args.subscribe:
            cmd_subscribe(bedrock, args.model_id)
        if args.prompt is None:
            return  # pure access-management invocation, nothing to query

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        parser.error("No prompt provided.")

    if args.extra_fields:
        try:
            additional_fields = json.loads(args.extra_fields)
        except json.JSONDecodeError as e:
            parser.error(f"--extra-fields is not valid JSON: {e}")
    else:
        additional_fields = None
    additional_fields = apply_disable_thinking(additional_fields, args.disable_thinking)

    images = None
    if args.attach:
        try:
            images = [build_image_block(p) for p in args.attach]
        except (ValueError, OSError) as e:
            parser.error(str(e))

    model_input = build_converse_input(
        prompt,
        args.temperature if args.temperature is not None else 0.0,
        additional_fields,
        system=args.system,
        max_tokens=args.max_tokens,
        images=images,
    )

    runtime = boto3.client("bedrock-runtime", region_name=args.region)

    deadline = time.time() + args.timeout * 60 if args.timeout > 0 else None
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = runtime.converse(modelId=args.model_id, **model_input)
            break
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            message = e.response.get("Error", {}).get("Message", str(e))
            if code == "AccessDeniedException" and deadline is not None and time.time() < deadline:
                print(
                    f"[attempt {attempt}] Access denied -- likely still auto-provisioning access "
                    f"(first use of a model can take up to ~15 min). Retrying in {RETRY_DELAY_SECS}s... "
                    f"({message})",
                    file=sys.stderr,
                )
                time.sleep(RETRY_DELAY_SECS)
                continue
            if code == "AccessDeniedException":
                print(
                    f"Error: {message}\n\n"
                    f"If this is the first time this account has used '{args.model_id}', Bedrock may "
                    f"still be auto-provisioning access in the background (can take up to ~15 min) -- "
                    f"try again shortly, or re-run with a longer --timeout to wait it out here instead. "
                    f"Otherwise, run with --check-access, or confirm your IAM identity has "
                    f"bedrock:InvokeModel permission for this model.",
                    file=sys.stderr,
                )
            else:
                print(f"Error ({code or 'unknown'}): {message}", file=sys.stderr)
            sys.exit(1)
        except BotoCoreError as e:
            print(f"Error: {_aws_error_message(e)}", file=sys.stderr)
            sys.exit(1)

    if args.raw:
        print(json.dumps(resp, indent=2, default=str))
        return

    try:
        print(extract_converse_text(resp))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
