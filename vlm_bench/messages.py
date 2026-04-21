from __future__ import annotations

from typing import Any


def text_user_message(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def multimodal_user_message(
    text: str,
    *,
    image_url: str | None = None,
    image_base64: str | None = None,
    mime: str = "image/png",
) -> dict[str, Any]:
    """
    Build an OpenAI-style multimodal user message.

    image_url: full URL or data URL.
    image_base64: raw base64 (without data: prefix); combined with mime.
    """

    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if image_url:
        parts.append({"type": "image_url", "image_url": {"url": image_url}})
    elif image_base64:
        data_url = f"data:{mime};base64,{image_base64}"
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
    return {"role": "user", "content": parts}
