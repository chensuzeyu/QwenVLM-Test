"""Pluggable backends for non-OpenAI-compatible services."""

from vlm_bench.backends.openai_compatible import OpenAICompatibleBackend
from vlm_bench.backends.protocol import StreamingChatBackend

__all__ = ["OpenAICompatibleBackend", "StreamingChatBackend"]
