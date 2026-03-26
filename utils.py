"""Shared utilities used across pipeline scripts."""

import json
from pathlib import Path

from tasks import load_config, resolve_arc_path


def find_model_config(config, model_name):
    """Find a model configuration by name."""
    for m in config["models"]:
        if m["name"] == model_name:
            return m
    raise ValueError(f"Model '{model_name}' not found in config")


def get_extraction_model_config(config, model_name):
    """Get extraction model config for two-pass protocol.

    Uses gpt-oss-120b-extract for extraction unless the subject model is
    already gpt-oss-120b.
    """
    if "120b" in model_name:
        return None
    for m in config["models"]:
        if m.get("role") == "extraction":
            return m
    return None


def load_arc(config, arc_name):
    """Load an ARC-AGI2 task JSON file."""
    arc_path = resolve_arc_path(config, arc_name)
    if arc_path is None:
        raise FileNotFoundError(f"ARC task file not found for {arc_name}")
    with open(arc_path) as f:
        return json.load(f)


def serialize_prompt(messages):
    """Serialize messages to a string for storage (strip binary image data)."""
    cleaned = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            parts = []
            for block in msg["content"]:
                if block.get("type") == "image_url":
                    parts.append({"type": "image_url", "image_url": {"url": "[base64 image]"}})
                else:
                    parts.append(block)
            cleaned.append({**msg, "content": parts})
        else:
            cleaned.append(msg)
    return json.dumps(cleaned, ensure_ascii=False)
