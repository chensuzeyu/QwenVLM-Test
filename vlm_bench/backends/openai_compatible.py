from __future__ import annotations

from typing import Any

from vlm_bench.client import stream_chat_completion
from vlm_bench.config import InvokeSettings
from vlm_bench.routing import Backend
from vlm_bench.types import StreamMetrics


class OpenAICompatibleBackend:
    """Default backend: any server that speaks OpenAI Chat Completions + SSE."""

    def __init__(self, settings: InvokeSettings) -> None:
        self._settings = settings

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        model: str | None = None,
        proxy_backend: Backend | None = None,
    ) -> StreamMetrics:
        return stream_chat_completion(
            self._settings,
            messages,
            model=model,
            proxy_backend=proxy_backend,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )
