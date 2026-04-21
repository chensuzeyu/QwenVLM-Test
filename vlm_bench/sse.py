from __future__ import annotations

import json
from typing import Any, Iterator


def iter_chat_completion_chunks(sse_lines: Iterator[str] | Iterator[bytes]) -> Iterator[dict[str, Any]]:
    """Parse OpenAI-style SSE stream (lines starting with 'data: ')."""

    for raw in sse_lines:
        if not raw:
            continue
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="replace").strip()
        else:
            line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def extract_delta_text(chunk: dict[str, Any]) -> str:
    """Return visible assistant text delta (excludes optional reasoning channel)."""

    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = (choices[0] or {}).get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content
    return ""


def extract_any_delta_text(chunk: dict[str, Any]) -> str:
    """First non-empty delta across content and common reasoning fields (for TTFT)."""

    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = (choices[0] or {}).get("delta") or {}
    for key in ("content", "reasoning_content", "reasoning"):
        val = delta.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def extract_usage(chunk: dict[str, Any]) -> dict[str, Any] | None:
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        return usage
    return None
