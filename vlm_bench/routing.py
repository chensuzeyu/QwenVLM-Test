from __future__ import annotations

from typing import Any, Literal

Backend = Literal["eas", "dashscope"]

# 与团队约定一致：经 Qwen Proxy 时 EAS 仅该 id；直连 PAI-EAS 时也应使用同名部署模型。
EAS_MODEL = "Qwen3.5-397B-A17B"

DASHSCOPE_MODELS: frozenset[str] = frozenset(
    {
        "qwen3.5-plus",
        "qwen3.5-27b",
        "qwen3.5-35b-a3b",
        "qwen3.5-122b-a10b",
        "qwen3.5-397b-a17b",
        "qwen3-vl-235b-a22b-instruct",
    }
)


def normalize_proxy_backend(name: str) -> Backend:
    n = name.strip().lower()
    if n in ("eas", "dashscope"):
        return n  # type: ignore[return-value]
    raise ValueError(f"backend must be 'eas' or 'dashscope', got {name!r}.")


def resolve_model_for_call(model: str, *, invoke_kind: str, proxy_backend: Backend | None) -> str:
    """Return the model string to send on the wire after validation."""

    raw = model.strip()
    if invoke_kind == "direct_eas":
        if raw != EAS_MODEL:
            raise ValueError(
                f"直连 EAS 仅支持模型 {EAS_MODEL!r}（当前为 {raw!r}）。"
                "若需 DashScope 多模型，请配置 PROXY_BASE_URL。"
            )
        return raw

    assert proxy_backend is not None
    if proxy_backend == "eas":
        if raw != EAS_MODEL:
            raise ValueError(
                f"Proxy backend=eas 仅支持 {EAS_MODEL!r}（当前为 {raw!r}）。"
                "换用 DashScope 模型请加 --backend dashscope 并选用对应 model id。"
            )
        return raw

    key = raw.lower()
    if key not in DASHSCOPE_MODELS:
        raise ValueError(
            f"Proxy backend=dashscope 时 model 须为 {sorted(DASHSCOPE_MODELS)} 之一，当前为 {raw!r}。"
        )
    return key


def build_chat_request_body(
    *,
    invoke_kind: Literal["proxy", "direct_eas"],
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    stream: bool,
    temperature: float | None,
    proxy_backend: Backend | None,
    enable_thinking: bool | None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble JSON body; proxy 模式下 EAS / DashScope 的 thinking 字段形状不同。"""

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if temperature is not None:
        body["temperature"] = temperature

    if invoke_kind == "direct_eas":
        if enable_thinking is not None:
            body["enable_thinking"] = enable_thinking
    else:
        assert proxy_backend is not None
        body["backend"] = proxy_backend
        if proxy_backend == "eas":
            # 团队约定：关 thinking 时不传 chat_template_kwargs；开 thinking 时嵌套 kwargs。
            if enable_thinking is True:
                body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            if enable_thinking is True:
                body["enable_thinking"] = True

    if extra_body:
        body.update(extra_body)
    return body
