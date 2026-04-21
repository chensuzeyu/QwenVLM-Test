# QwenVLM-Test · VLM 流式压测

面向 **OpenAI 兼容 Chat Completions** 的轻量压测工具：测 **非思考 / 思考** 两种模式，输出 **首 token 时延（TTFT）** 与 **首 token 之后的生成速度（tokens/s）**。

支持两种接入方式（二选一，由环境变量自动切换）：

1. **团队 Qwen Proxy（推荐）**：单一 `PROXY_BASE_URL`，请求体里用 `backend` 切换 **EAS** 与 **DashScope**，思考字段形状与官方示例一致。  
2. **直连 PAI-EAS**：保留原有 `EAS_BASE_URL` + `Bearer` Token 方式（无 `backend` 字段）。

## 功能概览

- **流式请求**：`stream: true`，真实测量首包时间与后续吞吐。
- **Qwen Proxy**：自动组包 `backend`、`chat_template_kwargs.enable_thinking`（EAS）或顶层 `enable_thinking`（DashScope）；关思考时不传多余字段（与网关默认一致）。
- **多模态**：可选 `image_url` 或 base64 文件（直连与 Proxy 均依赖下游是否支持）。
- **场景文件**：JSON/YAML 中可写 `model`、`backend`，与命令行解耦；**优先级**：`--model` / `--backend` > 场景文件 > `.env` 默认。
- **可扩展**：`StreamingChatBackend` 协议 + `OpenAICompatibleBackend`。

## 调用方式与模型

### 经 Qwen Proxy（`PROXY_BASE_URL` 已设置）

| `backend` | 可用 `model` | 说明 |
|-----------|----------------|------|
| `eas` | 仅 `Qwen3.5-397B-A17B` | 路演优先，板子快。 |
| `dashscope` | `qwen3.5-plus`、`qwen3.5-27b`、`qwen3.5-35b-a3b`、`qwen3.5-122b-a10b`、`qwen3.5-397b-a17b`、`qwen3-vl-235b-a22b-instruct` | 备份线路；程序会将 id **规范为小写** 再请求。 |

默认后端由 `PROXY_DEFAULT_BACKEND` 控制（建议 `eas`）。

### 直连 PAI-EAS（未设置 `PROXY_BASE_URL`）

仅支持 **`Qwen3.5-397B-A17B`**（与部署名一致），请求 **无** `backend` 字段；思考仍用顶层 `enable_thinking` 传布尔值（与旧版网关一致）。

## 思考模式（Thinking）

| 线路 | 开启思考 | 关闭思考 |
|------|----------|----------|
| Proxy + **EAS** | `"chat_template_kwargs": {"enable_thinking": true}` | 不传 `chat_template_kwargs` |
| Proxy + **DashScope** | `"enable_thinking": true` | 不传 `enable_thinking` |
| 直连 EAS | `"enable_thinking": true/false` | 同上 |

与 OpenAI Python SDK 对齐时：扩展字段用 `extra_body` 传入上述键（见团队示例脚本）。

## 快速开始

```powershell
cd D:\Knowin\QwenVLM-Test
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
# 编辑 .env：优先配置 PROXY_BASE_URL + PROXY_API_KEY
python -m vlm_bench
```

默认会 **依次** 跑两轮：思考关、思考开（Proxy 下对 EAS/DashScope 自动用正确 JSON 形状）。

## 环境变量

### Qwen Proxy（推荐）

| 变量 | 说明 |
|------|------|
| `PROXY_BASE_URL` | OpenAI 根路径，**须含 `/v1`**，例如 `http://host:5018/v1`。程序请求 `{PROXY_BASE_URL}/chat/completions`。 |
| `PROXY_API_KEY` | 鉴权；默认请求头为 `Authorization: <key>`（与部分网关一致）。 |
| `PROXY_AUTH_BEARER` | 设为 `1` / `true` 时使用 `Authorization: Bearer <key>`。 |
| `PROXY_DEFAULT_BACKEND` | `eas` 或 `dashscope`，默认 `eas`。 |
| `PROXY_MODEL` | 默认模型 id，须与默认 backend 匹配。 |
| `LLM_TIMEOUT_S` | 超时（秒），默认 `120`（未设时也可用 `EAS_TIMEOUT_S`）。 |

### 直连 PAI-EAS

| 变量 | 说明 |
|------|------|
| `EAS_BASE_URL` | 控制台给出的服务根路径，**不含** `/v1/chat/completions`。 |
| `EAS_API_TOKEN` | Bearer Token。 |
| `EAS_MODEL` | 默认 `Qwen3.5-397B-A17B`。 |

未设置 `PROXY_BASE_URL` 时必须配置 `EAS_*`；设置了 `PROXY_BASE_URL` 时 **忽略** `EAS_*`。

## 场景文件（解耦 Prompt / 模型 / 后端）

将 **系统提示、用户文本、图像、可选 `model` / `backend`** 写在 `scenarios/*.yaml` 或 `*.json` 中（路径相对于场景文件解析）。**若 CLI 与场景同时给出 `model` / `backend`，以 CLI 为准。**

```powershell
python -m vlm_bench --scenario scenarios/text_only.yaml --no-thinking
python -m vlm_bench --scenario scenarios/text_only.json --backend dashscope --model qwen3.5-plus
```

| 字段 | 说明 |
|------|------|
| `system` | 可选；系统消息。 |
| `user_text` | 用户文案；YAML 可用多行 `\|` / `>`。 |
| `user_text_file` | 可选；相对场景目录的 `.txt`。**若设置，以文件为准**（覆盖 `user_text`）。 |
| `image_url` / `image_base64_file` / `image_file` / `image_mime` | 多模态：`image_file` 为相对场景目录的 **二进制** 图片路径（PNG/JPEG 等）；`image_base64_file` 仍为纯 base64 文本文件。三者与 `image_url` **至多选一**。 |
| `model` | 可选；无 CLI `--model` 时用作默认模型。 |
| `backend` | 可选；`eas` \| `dashscope`（仅 Proxy；无 CLI `--backend` 时使用）。 |

仓库示例：`scenarios/text_only.yaml`、`scenarios/text_only.json`、`scenarios/multimodal.example.yaml`、`scenarios/image_local.example.yaml`、`scenarios/prompts/sample_user.txt`。

### 本地图片目录（不提交到 git）

将待测图片放在仓库根目录的 **`local_images/`**：该目录下除 `.gitkeep` 外的文件默认被 `.gitignore` 忽略，避免误传隐私或大文件。在场景 YAML 中用 `image_file: ../local_images/你的图.png`（路径相对 **场景文件所在目录**）并设置正确的 `image_mime`（如 `image/jpeg`）。

**问答示例（可写入 `user_text` 或 `prompts/*.txt`）**

| 目的 | 示例问法 |
|------|----------|
| 整体描述 | 「请用一两句中文描述这张图片的主要内容。」 |
| 物体/计数 | 「图中有哪些主要物体？大约各有多少个？」 |
| 文字 OCR | 「图片里可见的文字是什么？请按阅读顺序列出。」 |
| 细节推理 | 「图中人物大概在做什么？依据画面中的哪些线索？」 |

**运行示例**

```powershell
# 先将图片保存为 local_images\my_test.png，并复制 scenarios\image_local.example.yaml 按需改名修改
python -m vlm_bench --scenario scenarios/image_local.example.yaml --no-thinking
python -m vlm_bench --scenario scenarios/image_local.example.yaml --no-thinking --backend dashscope --model qwen3.5-397b-a17b
```

指定 `--scenario` 时仍忽略 `--prompt`、`--system`、`--image-*`（避免重复定义）。

## 命令行

| 场景 | 示例 |
|------|------|
| 默认（关思考 + 开思考各一次） | `python -m vlm_bench` |
| 仅非思考 | `python -m vlm_bench --no-thinking` |
| 仅思考 | `python -m vlm_bench --thinking` |
| 两轮对比 | `python -m vlm_bench --no-thinking --thinking` |
| Proxy + DashScope | `python -m vlm_bench --backend dashscope --model qwen3.5-plus --no-thinking` |
| JSON 行输出 | `python -m vlm_bench --json` |
| 场景文件 | `python -m vlm_bench --no-thinking --scenario scenarios/text_only.yaml` |
| 临时一行（无场景） | `python -m vlm_bench --prompt "你好"` |

安装后也可：`vlm-bench`。

## 指标说明

- **TTFT**：从发起 POST 到第一个非空 delta（含 `reasoning_content` 等思考通道）。
- **Subsequent tokens/s**：\((N-1) / (t_{\text{结束}} - t_{\text{首 token}})\)；\(N\) 优先取 `usage.completion_tokens`，否则对可见文本粗估。

## 在代码里调用

```python
from vlm_bench.config import InvokeSettings
from vlm_bench.client import stream_chat_completion
from vlm_bench.messages import text_user_message

settings = InvokeSettings.from_env()
messages = [text_user_message("你好")]
m = stream_chat_completion(settings, messages, enable_thinking=False)
# 已配置 PROXY_BASE_URL 时，可选：model=..., proxy_backend="dashscope" | "eas"
print(m.ttft_s, m.subsequent_tokens_per_s, m.text, m.extra)
```

原始 SSE chunk：`stream_chat_completion_events()`（`vlm_bench/client.py`）。

## 安全

- 密钥只放 **`.env`**；勿提交仓库。
- 曾泄露的 Key 请向 **@Yangkai** 或云平台流程申请轮换。

## 目录结构

```
.
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── local_images/            # 本地测试图（内容默认 gitignore，仅保留 .gitkeep）
├── scenarios/
│   ├── text_only.yaml
│   ├── text_only.json
│   ├── multimodal.example.yaml
│   ├── image_local.example.yaml
│   └── prompts/
│       └── sample_user.txt
└── vlm_bench/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── client.py
    ├── config.py              # InvokeSettings（兼容别名 EndpointConfig）
    ├── routing.py             # 模型校验 + Proxy/Direct 请求体规则
    ├── messages.py
    ├── scenario.py
    ├── sse.py
    ├── types.py
    └── backends/
        ├── __init__.py
        ├── openai_compatible.py
        └── protocol.py
```
