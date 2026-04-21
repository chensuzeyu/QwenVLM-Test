from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from vlm_bench.client import stream_chat_completion
from vlm_bench.config import InvokeSettings
from vlm_bench.messages import multimodal_user_message, text_user_message
from vlm_bench.routing import Backend, normalize_proxy_backend
from vlm_bench.scenario import load_scenario
from vlm_bench.types import StreamMetrics


def _print_metrics(label: str, m: StreamMetrics) -> None:
    print(f"\n=== {label} ===")
    print(f"TTFT (send → first token): {m.ttft_s * 1000:.1f} ms")
    if m.subsequent_tokens_per_s is not None:
        print(f"Subsequent speed: {m.subsequent_tokens_per_s:.2f} tokens/s")
    else:
        print("Subsequent speed: n/a (single token or missing timing)")
    print(f"Total time: {m.total_elapsed_s:.2f} s")
    tok_src = "API usage" if m.raw_usage and "completion_tokens" in m.raw_usage else "estimate"
    print(f"Tokens ({tok_src}): {m.total_tokens_reported}")
    print(f"Visible chars: {m.total_chars}")
    ex = m.extra or {}
    if ex.get("model"):
        print(f"Model (wire): {ex['model']}")
    if m.text:
        preview = m.text[:500] + ("…" if len(m.text) > 500 else "")
        print(f"Reply preview:\n{preview}")


def build_messages_cli(args: argparse.Namespace) -> list[dict]:
    msgs: list[dict] = []
    if args.system:
        msgs.append({"role": "system", "content": args.system})
    if args.image_url or args.image_b64_file:
        b64: str | None = None
        if args.image_b64_file:
            b64 = Path(args.image_b64_file).read_text(encoding="utf-8").strip()
        msgs.append(
            multimodal_user_message(
                args.prompt,
                image_url=args.image_url,
                image_base64=b64,
                mime=args.image_mime,
            )
        )
    else:
        msgs.append(text_user_message(args.prompt))
    return msgs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Qwen 流式压测：支持团队 Qwen Proxy（EAS / DashScope）或直连 PAI-EAS。"
    )
    p.add_argument(
        "--scenario",
        default=None,
        metavar="PATH",
        help="JSON/YAML 场景文件；可含 model / backend（仅 PROXY）；指定后忽略 --prompt / --system / --image-*",
    )
    p.add_argument("--prompt", default="你叫什么？用一句话回答。", help="未使用 --scenario 时的用户文案")
    p.add_argument("--system", default=None, help="未使用 --scenario 时的 system 消息")
    p.add_argument("--model", default=None, help="覆盖 .env 默认模型（PROXY 下 eas / dashscope 校验规则见 README）")
    p.add_argument(
        "--backend",
        choices=["eas", "dashscope"],
        default=None,
        help="仅 PROXY 模式：请求体中的 backend（默认 PROXY_DEFAULT_BACKEND；可被场景文件覆盖）",
    )
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--no-thinking", action="store_true", dest="no_thinking")
    p.add_argument("--thinking", action="store_true", dest="thinking")
    p.add_argument("--image-url", default=None, dest="image_url", help="未使用 --scenario 时生效")
    p.add_argument("--image-b64-file", default=None, help="未使用 --scenario 时生效")
    p.add_argument("--image-mime", default="image/png", help="CLI 多模态 MIME")
    p.add_argument("--json", action="store_true", help="每轮一行 JSON 摘要")

    args = p.parse_args(argv)

    try:
        settings = InvokeSettings.from_env()
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    scenario_model: str | None = None
    scenario_backend: str | None = None

    if args.scenario:
        try:
            loaded = load_scenario(Path(args.scenario))
            messages = loaded.build_messages()
            scenario_model = loaded.scenario.model
            scenario_backend = loaded.scenario.backend
        except (OSError, ValueError, TypeError) as e:
            print(f"场景文件错误: {e}", file=sys.stderr)
            return 2
    else:
        messages = build_messages_cli(args)

    if settings.kind == "direct_eas":
        if args.backend is not None and args.backend != "eas":
            print("直连 PAI-EAS 仅支持 EAS；不能使用 --backend dashscope。请设置 PROXY_BASE_URL。", file=sys.stderr)
            return 2
        if scenario_backend is not None and scenario_backend != "eas":
            print("直连模式下场景文件不能将 backend 设为 dashscope。", file=sys.stderr)
            return 2
        proxy_backend_eff: Backend | None = None
    else:
        raw_pb = args.backend or scenario_backend
        if raw_pb is None:
            proxy_backend_eff = settings.default_proxy_backend
        else:
            try:
                proxy_backend_eff = normalize_proxy_backend(raw_pb)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 2

    model_eff = (args.model or scenario_model or settings.default_model).strip()
    if not model_eff:
        print("model 不能为空（请设 --model、场景文件或 .env 默认）。", file=sys.stderr)
        return 2

    if args.no_thinking and args.thinking:
        runs: list[tuple[str, bool | None]] = [
            ("off", False),
            ("on", True),
        ]
    elif args.no_thinking:
        runs = [("off", False)]
    elif args.thinking:
        runs = [("on", True)]
    else:
        runs = [("off", False), ("on", True)]

    last: StreamMetrics | None = None
    for thinking_label, et in runs:
        try:
            m = stream_chat_completion(
                settings,
                messages,
                model=model_eff,
                proxy_backend=proxy_backend_eff,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                enable_thinking=et,
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        except httpx.ConnectError as e:
            url = settings.chat_completions_url
            print(
                "网络错误：无法与服务器建立 TCP 连接（连接被拒绝或不可达）。\n"
                f"  请求 URL: {url}\n"
                "  常见原因：\n"
                "  · 使用 Qwen Proxy 时：本机或内网代理进程未启动、IP/端口填错、防火墙拦截。\n"
                "  · 使用直连 EAS 时：公网/VPC 地址与当前网络不匹配。\n"
                f"  底层信息: {e}",
                file=sys.stderr,
            )
            return 1
        except httpx.HTTPStatusError as e:
            url = str(e.request.url)
            snippet = ""
            try:
                snippet = (e.response.text or "")[:800]
            except OSError:
                pass
            print(
                f"HTTP {e.response.status_code}：{url}\n"
                "  若为 404：核对控制台「调用地址」与服务名（如 qwen3_5_vllm）是否与 EAS_BASE_URL 一致，"
                "且服务已启用 OpenAI 兼容路由。\n"
                f"  响应片段: {snippet!r}",
                file=sys.stderr,
            )
            return 1
        except httpx.TimeoutException as e:
            print(f"请求超时（>{settings.timeout_s}s）：{e}", file=sys.stderr)
            return 1

        last = m
        if settings.kind == "proxy":
            human = f"backend={proxy_backend_eff} thinking={thinking_label} (explicit={et})"
        else:
            human = f"direct_eas thinking={thinking_label} (explicit={et})"

        if args.json:
            row = {
                "label": human,
                "enable_thinking": et,
                "invoke_kind": m.extra.get("invoke_kind") if m.extra else None,
                "proxy_backend": m.extra.get("proxy_backend") if m.extra else None,
                "model": m.extra.get("model") if m.extra else None,
                "ttft_ms": round(m.ttft_s * 1000, 3),
                "subsequent_tokens_per_s": m.subsequent_tokens_per_s,
                "total_elapsed_s": round(m.total_elapsed_s, 4),
                "total_tokens": m.total_tokens_reported,
                "usage": m.raw_usage,
            }
            print(json.dumps(row, ensure_ascii=False))
        else:
            _print_metrics(human, m)

    return 0 if last is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
