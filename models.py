"""LLM API abstraction — OpenAI-compatible (MindRouter) for subject model testing.

Claude-powered work (Phases 1-3, 6, 8) runs via Claude Code subagents, not this module.
This module handles MindRouter API calls for subject model testing (Phases 4-8).

Supports a two-pass approach for reasoning models:
  Pass 1: Let the model reason freely (unstructured)
  Pass 2: Feed reasoning back and extract structured JSON output

Includes a process-global token-bucket rate limiter shared across all worker
threads. MindRouter enforces a 200 req/min per-account cap (admin-set). Every
outbound request — including 429 retries — must acquire a token first, so a
fast 429-retry cascade can't blow past the cap.
"""

import base64
import json
import os
import threading
import time

import httpx


# --- Process-global rate limiter ---------------------------------------------
#
# The compression project (~/Documents/_RCDS/compression) learned the hard way
# that retries-on-429 fire so quickly (~200ms response time) that without a
# shared limiter, retries can amplify outgoing rate 2-5x and self-reinforce
# 429 cascades. Solution: every HTTP call acquires a token from a single
# process-wide bucket before firing — including retries.

class _TokenBucket:
    """Thread-safe token bucket. acquire() blocks until a token is available."""

    def __init__(self, rate_per_sec: float, burst: float):
        self.rate = rate_per_sec
        self.burst = max(1.0, burst)
        self._tokens = self.burst
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.burst, self._tokens + (now - self._last_refill) * self.rate
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_time = (1.0 - self._tokens) / self.rate
            time.sleep(wait_time)


_RATE_LIMITER: _TokenBucket | None = None
_RATE_LOCK = threading.Lock()


def set_rate_limit(rpm: float, burst: float = 10.0) -> None:
    """Configure the process-global rate limiter. Idempotent: last call wins."""
    global _RATE_LIMITER
    with _RATE_LOCK:
        _RATE_LIMITER = _TokenBucket(rate_per_sec=rpm / 60.0, burst=burst)


def configure_rate_limit_from_config(config: dict) -> None:
    """Read mindrouter.max_req_per_minute from config and set the limiter."""
    mr = config.get("mindrouter") or {}
    rpm = mr.get("max_req_per_minute", 100)  # conservative default
    burst = mr.get("burst_capacity", 10)
    set_rate_limit(rpm, burst)


def _acquire_token() -> None:
    if _RATE_LIMITER is not None:
        _RATE_LIMITER.acquire()


# --- HTTP call ---------------------------------------------------------------

_RATE_LIMIT_MAX_RETRIES = 5
_RATE_LIMIT_BACKOFF_BASE = 2.0  # seconds


def call_llm(model_config, messages):
    """Call an LLM and return (raw_response_json, response_text, latency_ms).

    Messages follow OpenAI format: [{"role": ..., "content": ...}]
    Acquires a rate-limit token before each request (including 429 retries).
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
    # Qwen-style thinking-mode toggle (e.g. qwen3.6-27b). MR honors this via
    # chat_template_kwargs.enable_thinking — false skips the hidden reasoning
    # trace and is ~16x faster, at the cost of more output-format violations.
    if "enable_thinking" in model_config:
        body["chat_template_kwargs"] = {"enable_thinking": model_config["enable_thinking"]}

    timeout = model_config.get("timeout", 300.0)
    start = time.monotonic()
    last_retryable_body = None
    last_retryable_status = None
    # Retry on rate limits (429), transient server errors (5xx), and socket-
    # level errors (DNS failures, connection resets, timeouts). MR's qwen
    # backends have shown cold-start 502 storms; client wifi blips have
    # produced [Errno 8] DNS errors that previously became permanent failures.
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(_RATE_LIMIT_MAX_RETRIES):
            _acquire_token()
            try:
                resp = client.post(url, json=body, headers=headers)
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                    httpx.RemoteProtocolError, httpx.PoolTimeout,
                    httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_retryable_body = f"{type(e).__name__}: {e}"[:200]
                last_retryable_status = "network"
                wait = _RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
                continue
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_retryable_body = resp.text[:200]
                last_retryable_status = resp.status_code
                wait = _RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            if last_retryable_status == "network":
                raise httpx.ConnectError(
                    f"network error after {_RATE_LIMIT_MAX_RETRIES} retries: {last_retryable_body}",
                )
            raise httpx.HTTPStatusError(
                f"{last_retryable_status} after {_RATE_LIMIT_MAX_RETRIES} retries: {last_retryable_body}",
                request=resp.request, response=resp,
            )
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
