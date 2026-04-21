from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vlm_bench.types import StreamMetrics


@runtime_checkable
class StreamingChatBackend(Protocol):
    """Implement for new providers; keep `StreamMetrics` as the common result type."""

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float | None,
        enable_thinking: bool | None,
        extra_body: dict[str, Any] | None,
    ) -> StreamMetrics: ...
