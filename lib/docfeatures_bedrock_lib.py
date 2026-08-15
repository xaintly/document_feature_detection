#!/usr/bin/env python3
"""
docfeatures_bedrock_lib.py — Shared Converse-format helpers for AWS Bedrock tools

Used by docfeatures_batch.py and query_bedrock.py. Separate from
docfeatures_lib.py (which stays free of any transport-specific dependency,
including boto3) because this module is specifically about the shape of
Bedrock Converse API requests/responses, not corpus processing.
"""


def apply_disable_thinking(additional_fields, disable_thinking):
    """Merge a --disable-thinking flag into an additionalModelRequestFields
    dict, preserving any other keys already present (e.g. a user-supplied
    'output_config'). Returns the dict unchanged if disable_thinking is
    falsy.

    Works on most current Claude models. Adaptive-thinking-only models
    (e.g. Claude Mythos 5, Claude Fable 5, Claude Opus 4.7, Claude Mythos
    Preview as of this writing) reject an explicit 'disabled' with a 400 --
    for those, omit 'thinking' entirely and use additionalModelRequestFields
    = {"output_config": {"effort": "low"}} instead.
    """
    if not disable_thinking:
        return additional_fields
    fields = dict(additional_fields or {})
    thinking = dict(fields.get("thinking") or {})
    thinking["type"] = "disabled"
    fields["thinking"] = thinking
    return fields


def build_converse_input(prompt, temperature, additional_fields=None, system=None,
                          max_tokens=None, images=None):
    """Build a Converse-format modelInput/request body.

    *images* is an optional list of Converse image content blocks (see
    build_image_block()), placed before the text block -- the usual
    convention for multimodal prompts.
    """
    content = list(images) if images else []
    content.append({"text": prompt})

    body = {
        "messages": [{"role": "user", "content": content}],
        "inferenceConfig": {"temperature": temperature},
    }
    if max_tokens is not None:
        body["inferenceConfig"]["maxTokens"] = max_tokens
    if system:
        body["system"] = [{"text": system}]
    if additional_fields:
        body["additionalModelRequestFields"] = additional_fields
    return body


# Converse's supported image formats, keyed by common file extensions.
_IMAGE_FORMATS = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".gif": "gif",
    ".webp": "webp",
}


def build_image_block(path):
    """Read a local image file into a Converse image content block.
    Raises ValueError if the extension isn't one Converse supports."""
    import os

    ext = os.path.splitext(path)[1].lower()
    fmt = _IMAGE_FORMATS.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unsupported image extension '{ext}' for {path!r}. "
            f"Converse supports: {', '.join(sorted(set(_IMAGE_FORMATS.values())))}."
        )
    with open(path, "rb") as f:
        data = f.read()
    return {"image": {"format": fmt, "source": {"bytes": data}}}


def extract_converse_text(model_output):
    """Pull the text content block out of a Converse response, skipping any
    non-text blocks (e.g. reasoningContent, if thinking wasn't disabled).
    *model_output* is the raw Converse response shape: {"output": {"message":
    {"content": [...]}}, ...} -- the same shape whether it came from a live
    bedrock-runtime.converse() call or a Bedrock batch output record."""
    content = model_output.get("output", {}).get("message", {}).get("content", [])
    for block in content:
        if "text" in block:
            return block["text"]
    raise ValueError(f"No text content block in Converse output: {model_output!r}")
