from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vlm_bench.messages import multimodal_user_message, text_user_message


@dataclass
class Scenario:
    """Prompt + optional image, loaded from a JSON or YAML file on disk."""

    system: str | None = None
    user_text: str = ""
    user_text_file: str | None = None
    image_url: str | None = None
    image_base64_file: str | None = None
    image_file: str | None = None
    image_mime: str = "image/png"
    model: str | None = None
    backend: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Scenario:
        def opt_str(key: str) -> str | None:
            v = data.get(key)
            if v is None:
                return None
            if isinstance(v, str):
                return v
            raise TypeError(f"{key} must be a string or null, got {type(v).__name__}")

        def opt_path(key: str) -> str | None:
            v = opt_str(key)
            if v is None or not v.strip():
                return None
            return v.strip()

        ut = data.get("user_text", "")
        if ut is None:
            ut = ""
        if not isinstance(ut, str):
            raise TypeError("user_text must be a string or null")

        sys_msg = opt_str("system")
        if sys_msg is not None and not sys_msg.strip():
            sys_msg = None

        mime = opt_path("image_mime") or "image/png"

        model_s = opt_str("model")
        model = model_s.strip() if model_s and model_s.strip() else None
        backend_raw = opt_str("backend")
        backend: str | None = None
        if backend_raw is not None:
            b = backend_raw.strip().lower()
            if b not in ("eas", "dashscope"):
                raise ValueError("scenario.backend 必须是 'eas' 或 'dashscope'")
            backend = b

        return cls(
            system=sys_msg,
            user_text=ut,
            user_text_file=opt_path("user_text_file"),
            image_url=opt_path("image_url"),
            image_base64_file=opt_path("image_base64_file"),
            image_file=opt_path("image_file"),
            image_mime=mime,
            model=model,
            backend=backend,
        )


@dataclass
class LoadedScenario:
    """Scenario plus the directory used to resolve relative paths."""

    scenario: Scenario
    config_dir: Path

    def build_messages(self) -> list[dict[str, Any]]:
        s = self.scenario
        base = self.config_dir

        user_body = s.user_text
        if s.user_text_file:
            path = (base / s.user_text_file).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"user_text_file not found: {path}")
            user_body = path.read_text(encoding="utf-8")

        img_sources = sum(
            1 for x in (s.image_url, s.image_base64_file, s.image_file) if x
        )
        if img_sources > 1:
            raise ValueError("image_url、image_base64_file、image_file 至多设置其一。")

        has_image = bool(s.image_url or s.image_base64_file or s.image_file)
        if not has_image and not user_body.strip():
            raise ValueError(
                "Scenario must set non-empty user_text / user_text_file, or provide image_url / image_base64_file / image_file."
            )

        msgs: list[dict[str, Any]] = []
        if s.system and str(s.system).strip():
            msgs.append({"role": "system", "content": s.system})

        b64: str | None = None
        if s.image_base64_file:
            ip = (base / s.image_base64_file).resolve()
            if not ip.is_file():
                raise FileNotFoundError(f"image_base64_file not found: {ip}")
            b64 = ip.read_text(encoding="utf-8").strip()
        elif s.image_file:
            ip = (base / s.image_file).resolve()
            if not ip.is_file():
                raise FileNotFoundError(f"image_file not found: {ip}")
            b64 = base64.b64encode(ip.read_bytes()).decode("ascii")

        if s.image_url or b64 is not None:
            msgs.append(
                multimodal_user_message(
                    user_body,
                    image_url=s.image_url,
                    image_base64=b64,
                    mime=s.image_mime,
                )
            )
        else:
            msgs.append(text_user_message(user_body))

        return msgs


def load_scenario(path: Path) -> LoadedScenario:
    p = path.expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Scenario file not found: {p.resolve()}")

    raw = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}") from e
    elif suffix == ".json":
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    else:
        raise ValueError(f"Unsupported scenario format: {suffix} (use .json, .yaml, .yml)")

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError("Scenario root must be a mapping (JSON object / YAML dict).")

    scenario = Scenario.from_mapping(data)
    return LoadedScenario(scenario=scenario, config_dir=p.resolve().parent)
