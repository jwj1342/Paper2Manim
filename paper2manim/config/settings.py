from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from paper2manim.infrastructure.llm.client import LLMConfig


@dataclass(frozen=True)
class ProviderDefaults:
    base_url: str
    model: str
    api_key_env: str
    api_style: str


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig
    vlm: VLMConfig | None
    visual_review: VisualReviewConfig
    output_dir: Path = Path("runs/mvp1")
    render_enabled: bool = True
    fallback_video_enabled: bool = False
    duration_mode: str = "medium"
    target_duration_seconds: int | None = None
    render_fix_max_retries: int = 2
    max_failed_scenes: int = 0
    continue_on_scene_failure: bool = False


@dataclass(frozen=True)
class VLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: int = 120
    max_retries: int = 2
    temperature: float = 0.0


@dataclass(frozen=True)
class VisualReviewConfig:
    enabled: bool = False
    max_retries: int = 2
    strict: bool = False
    frame_sample_count: int = 5
    use_mock_vlm: bool = False


def load_dotenv(path: Path = Path(".env")) -> None:
    """Seed os.environ from ``path`` without clobbering already-set keys.

    Thin shim over ``python-dotenv`` (already a dependency) — we keep this
    wrapper so callers and re-exports in ``paper2manim.config`` don't have to
    care about the upstream import path.
    """
    from dotenv import load_dotenv as _dotenv_load

    _dotenv_load(path, override=False)


def load_settings(
    config_path: Path = Path("config.yaml"),
    *,
    require_api_key: bool = True,
) -> Settings:
    load_dotenv()
    raw_config = _load_yaml_config(config_path)
    provider_defaults = _load_provider_defaults(raw_config)
    llm_config = _section(raw_config, "llm")
    vlm_config = _section(raw_config, "vlm")
    visual_review_config = _section(raw_config, "visual_review")
    render_config = _section(raw_config, "render")
    output_config = _section(raw_config, "output")
    planning_config = _section(raw_config, "planning")

    provider = _optional_string(llm_config.get("provider")) or "deepseek"
    provider = provider.lower()
    if provider not in provider_defaults:
        supported = ", ".join(sorted(provider_defaults))
        raise RuntimeError(
            "Unsupported LLM provider "
            f"'{provider}'. Use one of: {supported}."
        )

    defaults = provider_defaults[provider]
    api_key_env = _optional_string(llm_config.get("api_key_env")) or defaults.api_key_env
    api_key = _first_env("LLM_API_KEY", api_key_env)
    if require_api_key and not api_key:
        raise RuntimeError(
            f"Missing API key. Set LLM_API_KEY or {api_key_env} "
            "in .env or export it before running."
        )

    base_url = _optional_string(llm_config.get("base_url")) or defaults.base_url
    model = _optional_string(llm_config.get("model")) or defaults.model
    api_style = _optional_string(llm_config.get("api_style")) or defaults.api_style
    timeout = _optional_int(_env("LLM_TIMEOUT"), llm_config.get("timeout"), 120)
    max_retries = _optional_int(
        _env("LLM_MAX_RETRIES"),
        llm_config.get("max_retries"),
        2,
    )
    if not base_url:
        raise RuntimeError("Missing llm.base_url for this provider.")
    if not model:
        raise RuntimeError("Missing llm.model for this provider.")

    extra_body = _section(llm_config, "extra_body")
    thinking = _optional_string(llm_config.get("thinking"))
    if provider == "deepseek" and not thinking:
        thinking = "disabled"
    if provider == "deepseek" and thinking:
        extra_body = {**extra_body, "thinking": {"type": thinking}}

    vlm_settings = _load_vlm_config(vlm_config)
    visual_review_settings = _load_visual_review_config(visual_review_config, vlm_settings)

    return Settings(
        llm=LLMConfig(
            provider=provider,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            api_style=api_style,
            extra_body=extra_body,
        ),
        vlm=vlm_settings,
        visual_review=visual_review_settings,
        output_dir=Path(str(output_config.get("dir") or "runs/mvp1")),
        render_enabled=bool(render_config.get("enabled", True)),
        fallback_video_enabled=bool(render_config.get("fallback_video", False)),
        duration_mode=_optional_string(planning_config.get("duration_mode")) or "medium",
        target_duration_seconds=(
            _optional_int("", planning_config.get("target_duration_seconds"), 0)
            if planning_config.get("target_duration_seconds") is not None
            else None
        ),
        render_fix_max_retries=_optional_int(
            _env("RENDER_FIX_MAX_RETRIES"),
            render_config.get("fix_max_retries"),
            2,
        ),
        max_failed_scenes=_optional_int(
            _env("MAX_FAILED_SCENES"),
            render_config.get("max_failed_scenes"),
            0,
        ),
        continue_on_scene_failure=_optional_bool(
            _env("CONTINUE_ON_SCENE_FAILURE"),
            render_config.get("continue_on_scene_failure"),
            False,
        ),
    )


def _load_vlm_config(config: dict[str, Any]) -> VLMConfig | None:
    enabled = _optional_bool(_env("VLM_ENABLE"), config.get("enable"), False)
    use_mock = _optional_bool(_env("USE_MOCK_VLM"), config.get("use_mock"), False)
    if use_mock:
        enabled = True
    if not enabled:
        return None

    provider = _optional_string(_env("VLM_PROVIDER")) or _optional_string(
        config.get("provider")
    )
    provider = provider or "volcengine_ark"
    api_key_env = _optional_string(config.get("api_key_env")) or "VLM_API_KEY"
    api_key = _first_env("VLM_API_KEY", api_key_env)
    if not api_key and not use_mock:
        raise RuntimeError(
            f"Missing VLM API key. Set VLM_API_KEY or {api_key_env} in .env or env."
        )
    base_url = _optional_string(_env("VLM_BASE_URL")) or _optional_string(
        config.get("base_url")
    )
    model = _optional_string(_env("VLM_MODEL")) or _optional_string(
        config.get("model")
    )
    if not base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
    if not model:
        model = "doubao-seed-2-0-pro-260215"

    timeout = _optional_int(_env("VLM_TIMEOUT_SECONDS"), config.get("timeout"), 120)
    max_retries = _optional_int(_env("VLM_MAX_RETRIES"), config.get("max_retries"), 2)
    temperature = _optional_float(_env("VLM_TEMPERATURE"), config.get("temperature"), 0.0)
    return VLMConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
    )


def _load_visual_review_config(
    config: dict[str, Any],
    vlm_config: VLMConfig | None,
) -> VisualReviewConfig:
    use_mock = _optional_bool(_env("USE_MOCK_VLM"), config.get("use_mock"), False)
    enabled = _optional_bool(_env("ENABLE_VISUAL_REVIEW"), config.get("enable"), False)
    if use_mock:
        enabled = True
    if enabled and vlm_config is None:
        enabled = False
    max_retries = _optional_int(_env("VISUAL_REVISION_MAX_RETRIES") or _env("VISUAL_REVIEW_MAX_RETRIES"), config.get("max_retries"), 2)
    strict = _optional_bool(_env("STRICT_VISUAL_REVIEW"), config.get("strict"), False)
    frame_sample_count = _optional_int(_env("FRAME_SAMPLE_COUNT"), config.get("frame_sample_count"), 5)
    return VisualReviewConfig(
        enabled=enabled,
        max_retries=max_retries,
        strict=strict,
        frame_sample_count=frame_sample_count,
        use_mock_vlm=use_mock,
    )


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must contain a YAML mapping: {path}")
    return data


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Config section '{key}' must be a mapping.")
    return dict(value)


def _load_provider_defaults(config: dict[str, Any]) -> dict[str, ProviderDefaults]:
    raw_providers = config.get("llm_providers")
    if raw_providers is None:
        raise RuntimeError("Missing 'llm_providers' section in config.yaml.")
    if not isinstance(raw_providers, dict):
        raise RuntimeError("Config section 'llm_providers' must be a mapping.")
    defaults: dict[str, ProviderDefaults] = {}
    for name, value in raw_providers.items():
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Provider '{name}' in 'llm_providers' must be a mapping."
            )
        defaults[str(name).lower()] = ProviderDefaults(
            base_url=_optional_string(value.get("base_url")),
            model=_optional_string(value.get("model")),
            api_key_env=_optional_string(value.get("api_key_env")) or "LLM_API_KEY",
            api_style=_optional_string(value.get("api_style")) or "openai",
        )
    return defaults


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _first_env(*names: str) -> str:
    for name in names:
        if not name:
            continue
        value = _env(name)
        if value:
            return value
    return ""


def _optional_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_int(env_value: str, config_value: Any, default: int) -> int:
    value = env_value or config_value
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Expected integer config value, got {value!r}.") from exc


def _optional_float(env_value: str, config_value: Any, default: float) -> float:
    value = env_value or config_value
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Expected float config value, got {value!r}.") from exc


def _optional_bool(env_value: str, config_value: Any, default: bool) -> bool:
    value = env_value or config_value
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Expected boolean config value, got {value!r}.")
