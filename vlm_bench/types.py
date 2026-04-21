from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamMetrics:
    """Timing and throughput for one streamed completion."""

    ttft_s: float
    """Wall time from sending the HTTP request until the first content delta arrives."""

    subsequent_tokens_per_s: float | None
    """Tokens per second after the first token (requires token count from API or heuristic)."""

    total_elapsed_s: float
    total_tokens_reported: int | None
    total_chars: int
    text: str
    enable_thinking: bool | None
    raw_usage: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
