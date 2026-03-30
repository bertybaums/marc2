"""LLM API abstraction — OpenAI-compatible (MindRouter) for subject model testing.

Claude-powered work (Phases 1-3, 6, 8) runs via Claude Code subagents, not this module.
This module handles MindRouter API calls for subject model testing (Phases 4-8).

Supports a two-pass approach for reasoning models:
  Pass 1: Let the model reason freely (unstructured)
  Pass 2: Feed reasoning back and extract structured JSON output
"""

import base64
import json
import os
import time

import httpx


def call_llm(model_config, messages):
    """Call an LLM and return (raw_response_json, response_text, latency_ms).

    Messages follow OpenAI format: [{"role": ..., "content": ...}]
    """
    api_key = _get_api_key(model_config)
    endpoint = model_config["endpoint"].rstrip("/")
    url = f"{endpoint}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model_config["model_id"],
        "messages": messages,
        "temperature": model_config.get("temperature", 0.0),
    }
    if "max_tokens" in model_config:
        body["max_tokens"] = model_config["max_tokens"]
    if "reasoning_effort" in model_config:
        body["reasoning_effort"] = model_config["reasoning_effort"]

    start = time.monotonic()
    with httpx.Client(timeout=model_config.get("timeout", 300.0)) as client:
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
    latency_ms = int((time.monotonic() - start) * 1000)

    raw = resp.text
    data = resp.json()
    msg = data["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return raw, text, latency_ms


def call_llm_two_pass(model_config, messages, extraction_prompt_fn,
                      extraction_model_config=None):
    """Two-pass LLM call: reasoning then extraction.

    Pass 1: Send messages, get unstructured reasoning.
    Pass 2: Send extraction prompt with the reasoning, get structured output.
    """
    raw1, text1, latency1 = call_llm(model_config, messages)

    # Get full reasoning (may be in reasoning_content for reasoning models)
    raw1_data = json.loads(raw1) if isinstance(raw1, str) else raw1
    msg = raw1_data.get("choices", [{}])[0].get("message", {})
    reasoning = msg.get("reasoning_content") or text1 or ""

    pass2_config = extraction_model_config or model_config
    extraction_messages = extraction_prompt_fn(reasoning)
    raw2, text2, latency2 = call_llm(pass2_config, extraction_messages)

    return raw1, reasoning, raw2, text2, latency1 + latency2


def _get_api_key(config):
    env_var = config.get("api_key_env")
    if not env_var:
        return None
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"Environment variable {env_var} not set")
    return key


def make_image_content_block(image_bytes, media_type="image/png"):
    """Create an OpenAI-compatible image_url content block from raw bytes."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64}"},
    }
