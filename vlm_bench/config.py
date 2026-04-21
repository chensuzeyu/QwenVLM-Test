from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

from vlm_bench.routing import Backend, EAS_MODEL, normalize_proxy_backend

load_dotenv()


@dataclass(frozen=True)
class InvokeSettings:
    """
    HTTP 调用配置。

    - 若设置 ``PROXY_BASE_URL``：走团队 **Qwen Proxy**（OpenAI 兼容 ``/v1`` + ``POST .../chat/completions``）。
    - 否则：直连 **PAI-EAS**（在 ``EAS_BASE_URL`` 后拼接 ``/v1/chat/completions``）。
    """

    kind: Literal["proxy", "direct_eas"]
    chat_completions_url: str
    authorization_header: str
    default_model: str
    timeout_s: float
    default_proxy_backend: Backend

    @classmethod
    def from_env(cls) -> InvokeSettings:
        proxy_base = os.environ.get("PROXY_BASE_URL", "").strip()
        timeout_s = float(os.environ.get("LLM_TIMEOUT_S", os.environ.get("EAS_TIMEOUT_S", "120")))

        if proxy_base:
            key = os.environ.get("PROXY_API_KEY", "").strip()
            if not key:
                raise ValueError("已设置 PROXY_BASE_URL，但缺少 PROXY_API_KEY（见 .env.example）。")

            use_bearer = os.environ.get("PROXY_AUTH_BEARER", "").strip().lower() in ("1", "true", "yes")
            auth = f"Bearer {key}" if use_bearer else key

            url = proxy_base.rstrip("/") + "/chat/completions"
            model = os.environ.get("PROXY_MODEL", EAS_MODEL).strip()
            raw_backend = os.environ.get("PROXY_DEFAULT_BACKEND", "eas").strip()
            default_proxy_backend = normalize_proxy_backend(raw_backend)

            return cls(
                kind="proxy",
                chat_completions_url=url,
                authorization_header=auth,
                default_model=model,
                timeout_s=timeout_s,
                default_proxy_backend=default_proxy_backend,
            )

        base = os.environ.get("EAS_BASE_URL", "").strip()
        token = os.environ.get("EAS_API_TOKEN", "").strip()
        model = os.environ.get("EAS_MODEL", EAS_MODEL).strip()
        if not base:
            raise ValueError("请设置 PROXY_BASE_URL（推荐）或 EAS_BASE_URL。见 .env.example。")
        if not token:
            raise ValueError("直连模式需要 EAS_API_TOKEN。见 .env.example。")

        url = base.rstrip("/") + "/v1/chat/completions"
        auth = f"Bearer {token}"

        return cls(
            kind="direct_eas",
            chat_completions_url=url,
            authorization_header=auth,
            default_model=model,
            timeout_s=timeout_s,
            default_proxy_backend="eas",
        )


# 旧名兼容（少量外部引用时可继续 import EndpointConfig）
EndpointConfig = InvokeSettings
