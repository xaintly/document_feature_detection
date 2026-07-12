#!/usr/bin/env python3
"""CLI tool to query Ollama models via the REST API."""

import argparse
import base64
import json
import sys
import requests

DEFAULT_MODEL = "qwen3.5:122b"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 5

def main():
    parser = argparse.ArgumentParser(description="Query an Ollama model.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("-p", "--prompt", help="Prompt text. If omitted, reads from stdin.")
    parser.add_argument("-s", "--system", default=None, help="System prompt.")
    parser.add_argument("-a", "--attach", action="append", default=[], help="Image file to attach (can be repeated). Supported: png, jpg, gif, webp.")
    parser.add_argument("-t", "--timeout", default=DEFAULT_TIMEOUT, help=f"Server wait timeout, in minutes (0 to disable, default: {DEFAULT_TIMEOUT} minutes)")
    parser.add_argument("-e", "--temperature", default=None, help=f"Model temperature parameter")
    parser.add_argument("-c", "--context", default=None, help=f"Context window size")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama server URL (default: {DEFAULT_HOST})")
    args = parser.parse_args()
    
    

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        parser.error("No prompt provided.")

    context = None
    if args.context != None:
        context = int(args.context)
    else:
        # require as much context as the prompt size, rounded up to next 4k size
        context= ((len(prompt) // 4096) + 1) * 4096
        
    if int(args.timeout) < 1:
        args.timeout = None
    else:
        args.timeout = int(args.timeout) * 60
        

    # Build the request payload (streaming off for simplicity)
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": context
        }
    }

    if args.system:
        payload["system"] = args.system

    if args.temperature != None and float(args.temperature) >= 0.0 and float(args.temperature) <= 2.0:
        payload["temperature"] = float(args.temperature)

    # Ollama's /api/generate supports an "images" field: a list of base64-encoded
    # images. This works with multimodal/vision models (e.g. llava, llama3.2-vision).
    # For non-vision models the images are silently ignored.
    if args.attach:
        images = []
        for path in args.attach:
            with open(path, "rb") as f:
                images.append(base64.b64encode(f.read()).decode())
        payload["images"] = images

    url = f"{args.host.rstrip('/')}/api/generate"

    try:
        resp = requests.post(url, json=payload, timeout=args.timeout)
        resp.raise_for_status()
    except requests.ConnectionError:
        print(f"Error: cannot connect to Ollama at {args.host}", file=sys.stderr)
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"Error: {e}\n{resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    print(data.get("response", ""))


if __name__ == "__main__":
    main()
