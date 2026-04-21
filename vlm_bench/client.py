from __future__ import annotations

import time
from typing import Any, Iterator

import httpx

from vlm_bench.config import InvokeSettings
from vlm_bench.routing import Backend, build_chat_request_body, resolve_model_for_call
from vlm_bench.sse import (
    extract_any_delta_text,
    extract_delta_text,
    extract_usage,
    iter_chat_completion_chunks,
)
from vlm_bench.types import StreamMetrics


def _rough_token_estimate(text: str) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / 2.5)))


def stream_chat_completion(
    settings: InvokeSettings,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    proxy_backend: Backend | None = None,
    max_tokens: int = 4096,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
    extra_body: dict[str, Any] | None = None,
) -> StreamMetrics:
    """
    POST Chat Completions（流式），聚合 TTFT 与后续 tokens/s。

    - **proxy** 模式：根据 ``proxy_backend`` 组请求体（``backend``、EAS 用 ``chat_template_kwargs``、
      DashScope 用顶层 ``enable_thinking``）。
    - **direct_eas**：沿用顶层 ``enable_thinking``，不注入 ``backend``。
    """

    if settings.kind == "direct_eas":
        pb: Backend | None = None
    else:
        pb = proxy_backend if proxy_backend is not None else settings.default_proxy_backend

    effective_model = (model or settings.default_model).strip()
    wire_model = resolve_model_for_call(effective_model, invoke_kind=settings.kind, proxy_backend=pb)

    body = build_chat_request_body(
        invoke_kind=settings.kind,
        model=wire_model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        proxy_backend=pb,
        enable_thinking=enable_thinking,
        extra_body=extra_body,
    )

    headers = {
        "Authorization": settings.authorization_header,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    t_send = time.perf_counter()
    first_token_s: float | None = None
    parts: list[str] = []
    last_usage: dict[str, Any] | None = None

    timeout = httpx.Timeout(settings.timeout_s, connect=30.0)

    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            settings.chat_completions_url,
            headers=headers,
            json=body,
        ) as resp:
            resp.raise_for_status()
            chunks = iter_chat_completion_chunks(resp.iter_lines())
            for chunk in chunks:
                if first_token_s is None:
                    piece = extract_any_delta_text(chunk)
                    if piece:
                        first_token_s = time.perf_counter()
                vis = extract_delta_text(chunk)
                if vis:
                    parts.append(vis)
                u = extract_usage(chunk)
                if u is not None:
                    last_usage = u

    t_end = time.perf_counter()
    text = "".join(parts)
    total_elapsed_s = t_end - t_send

    if first_token_s is None:
        ttft_s = total_elapsed_s
    else:
        ttft_s = first_token_s - t_send

    total_tokens_reported: int | None = None
    if last_usage:
        ct = last_usage.get("completion_tokens")
        if isinstance(ct, int):
            total_tokens_reported = ct

    if total_tokens_reported is None:
        total_tokens_reported = _rough_token_estimate(text)

    subsequent: float | None = None
    if first_token_s is not None and total_tokens_reported is not None and total_tokens_reported > 1:
        gen_s = t_end - first_token_s
        if gen_s > 0:
            subsequent = (total_tokens_reported - 1) / gen_s

    return StreamMetrics(
        ttft_s=ttft_s,
        subsequent_tokens_per_s=subsequent,
        total_elapsed_s=total_elapsed_s,
        total_tokens_reported=total_tokens_reported,
        total_chars=len(text),
        text=text,
        enable_thinking=enable_thinking,
        raw_usage=last_usage,
        extra={
            "endpoint": settings.chat_completions_url,
            "invoke_kind": settings.kind,
            "proxy_backend": pb,
            "model": wire_model,
        },
    )


def stream_chat_completion_events(
    settings: InvokeSettings,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    proxy_backend: Backend | None = None,
    max_tokens: int = 4096,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
    extra_body: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    if settings.kind == "direct_eas":
        pb = None
    else:
        pb = proxy_backend if proxy_backend is not None else settings.default_proxy_backend

    effective_model = (model or settings.default_model).strip()
    wire_model = resolve_model_for_call(effective_model, invoke_kind=settings.kind, proxy_backend=pb)

    body = build_chat_request_body(
        invoke_kind=settings.kind,
        model=wire_model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        proxy_backend=pb,
        enable_thinking=enable_thinking,
        extra_body=extra_body,
    )

    headers = {
        "Authorization": settings.authorization_header,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    timeout = httpx.Timeout(settings.timeout_s, connect=30.0)

    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            settings.chat_completions_url,
            headers=headers,
            json=body,
        ) as resp:
            resp.raise_for_status()
            yield from iter_chat_completion_chunks(resp.iter_lines())
