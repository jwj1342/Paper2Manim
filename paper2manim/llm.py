"""Chat-model factory routed by YAML config (or MiMo env fallback).

Two layers, picked at call time:

1. **YAML-driven (preferred)** — if ``<project_root>/config.yaml`` exists, model
   definitions and role→model bindings come from there via
   :func:`paper2manim.config.config_loader.load_model_settings`. Canonical role
   names match :data:`CANONICAL_ROLES` and are referenced both by graph nodes
   and by ``model_roles:`` keys in YAML.

2. **MiMo env fallback (legacy)** — if no ``config.yaml`` is present, build a
   MiMo ``ChatOpenAI`` from ``.env`` (``MIMO_API_KEY``). Legacy aliases
   ``flash`` / ``pro`` / ``v2`` / ``v2-omni`` still work for backwards
   compatibility with the existing agents.

``get_vlm()`` is YAML-only — it requires a model with ``supports_vision=true``
bound to the ``vision_checker`` role.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from paper2manim.config import PROJECT_ROOT, get_settings
from paper2manim.config.config_loader import load_model_settings
from paper2manim.config.model_config import ModelConfig, ModelSettings

log = logging.getLogger(__name__)
_T = TypeVar("_T", bound=BaseModel)

CANONICAL_ROLES: tuple[str, ...] = (
    "global_reader",
    "scene_planner",
    "scene_coder",
    "render_fixer",
    "final_summarizer",
    "visual_reviser",
    "vision_checker",
)

# Legacy aliases used by agents pre-YAML; map onto canonical roles so older
# agent code keeps working without a rewrite.
_LEGACY_ALIAS_TO_ROLE: dict[str, str] = {
    "flash": "scene_coder",
    "pro": "final_summarizer",
    "v2": "scene_coder",
    "v2-omni": "scene_coder",
}

_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Fallback model map for MiMo env mode. Keys cover both legacy aliases and
# canonical roles so the same agent code paths resolve in either mode.
_MIMO_FALLBACK_MODEL: dict[str, str] = {
    "flash": "mimo-v2.5",
    "pro": "mimo-v2.5-pro",
    "v2": "mimo-v2-pro",
    "v2-omni": "mimo-v2-omni",
    "global_reader": "mimo-v2.5-pro",
    "scene_planner": "mimo-v2.5",
    "scene_coder": "mimo-v2.5",
    "render_fixer": "mimo-v2.5",
    "final_summarizer": "mimo-v2.5-pro",
    "visual_reviser": "mimo-v2.5",
}


def _seed_env_from_dotenv() -> None:
    """Make $VAR references inside config.yaml resolvable.

    pydantic-settings only injects fields it explicitly declares; anything else
    in ``.env`` (e.g. ``AZURE_CLAUDE_API_KEY``) is ignored. The YAML loader,
    however, reads ``os.environ`` directly. Delegate to ``python-dotenv``
    (already a dependency) with ``override=False`` so process env wins over
    ``.env``.
    """
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)


@lru_cache(maxsize=1)
def _yaml_settings() -> ModelSettings | None:
    if not _CONFIG_PATH.exists():
        return None
    _seed_env_from_dotenv()
    try:
        return load_model_settings(_CONFIG_PATH)
    except Exception as exc:
        # Don't silently fall back to env-MiMo: the user put a config.yaml on
        # disk so they expect it to be honored. A typo'd $ENV_VAR or malformed
        # model entry should surface, not get masked by a fallback that suddenly
        # routes traffic to a different provider.
        raise RuntimeError(
            f"[llm] config.yaml at {_CONFIG_PATH} is present but failed to load: "
            f"{type(exc).__name__}: {exc}. Fix the file or delete it to fall "
            "back to the env-MiMo path."
        ) from exc


def reload_config() -> None:
    """Drop the cached :class:`ModelSettings` so the next call re-reads YAML."""
    _yaml_settings.cache_clear()


def vision_checker_config() -> ModelConfig:
    """Return the cached :class:`ModelConfig` bound to the ``vision_checker`` role.

    Raises ``RuntimeError`` if no ``config.yaml`` is present or no model is bound
    to ``vision_checker``. Used by the VLM agent to avoid re-parsing YAML on
    every review (one parse per process instead of one per scene-revision).
    """
    settings = _yaml_settings()
    if settings is None:
        raise RuntimeError(
            "VLM review requires config.yaml with a supports_vision=true model "
            "bound to role 'vision_checker'."
        )
    try:
        return settings.model_for_role("vision_checker")
    except KeyError as exc:
        raise RuntimeError(
            f"config.yaml has no model bound to role 'vision_checker'. {exc}"
        ) from exc


def _resolve_role(name: str) -> str:
    return _LEGACY_ALIAS_TO_ROLE.get(name, name)


def _build_yaml_client(
    cfg: ModelConfig,
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra: dict[str, Any],
) -> BaseChatModel:
    """Materialize a langchain chat client from a :class:`ModelConfig`."""
    if not cfg.api_key:
        raise RuntimeError(
            f"Model '{cfg.name}' has empty api_key — fill the $ENV_VAR referenced "
            "by config.yaml or inline a key."
        )
    if cfg.provider == "openai_compatible":
        oai_kwargs: dict[str, Any] = {
            "model": cfg.model,
            "api_key": cfg.api_key,
            "base_url": cfg.base_url,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if not cfg.omit_temperature:
            oai_kwargs["temperature"] = temperature
        oai_kwargs.update(extra)
        return ChatOpenAI(**oai_kwargs)
    if cfg.provider == "anthropic":
        # Imported lazily — langchain-anthropic is an optional dep until you
        # actually point a role at an Anthropic model.
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "anthropic_api_url": cfg.base_url,
            "anthropic_api_key": cfg.api_key,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if not cfg.omit_temperature:
            kwargs["temperature"] = temperature
        # Azure-hosted Claude uses Authorization: Bearer instead of the default
        # x-api-key header; flip via default_headers when auth_style says so.
        if cfg.auth_style == "bearer":
            kwargs["default_headers"] = {
                "Authorization": f"Bearer {cfg.api_key}",
            }
        kwargs.update(extra)
        return ChatAnthropic(**kwargs)
    raise RuntimeError(
        f"Unsupported provider '{cfg.provider}' on model '{cfg.name}'. "
        "Choose 'openai_compatible' or 'anthropic'."
    )


def _build_mimo_fallback(
    role_or_alias: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra: dict[str, Any],
) -> ChatOpenAI:
    if role_or_alias not in _MIMO_FALLBACK_MODEL:
        raise ValueError(
            f"Unknown model alias / role '{role_or_alias}'. Either copy "
            "config.example.yaml to config.yaml or use a known alias "
            f"({', '.join(sorted(_MIMO_FALLBACK_MODEL))})."
        )
    s = get_settings()
    if not s.MIMO_API_KEY:
        raise RuntimeError(
            "No config.yaml found and MIMO_API_KEY is empty. Either create "
            "config.yaml (copy config.example.yaml) or fill MIMO_API_KEY in .env."
        )
    return ChatOpenAI(
        model=_MIMO_FALLBACK_MODEL[role_or_alias],
        api_key=s.MIMO_API_KEY,
        base_url=s.MIMO_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **extra,
    )


def get_llm(
    model: str = "flash",
    *,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: float = 120,
    **kwargs,
) -> BaseChatModel:
    """Return a chat client for ``model`` (role name or legacy alias)."""
    role = _resolve_role(model)
    settings = _yaml_settings()
    if settings is None:
        return _build_mimo_fallback(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra=kwargs,
        )
    try:
        cfg = settings.model_for_role(role)
    except KeyError as exc:
        raise RuntimeError(f"config.yaml has no model bound to role '{role}'. {exc}") from exc
    return _build_yaml_client(
        cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra=kwargs,
    )


def get_vlm(
    *,
    role: str = "vision_checker",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float = 120,
    **kwargs,
) -> BaseChatModel:
    """Return a vision-capable chat client. Raises if no such model is configured."""
    settings = _yaml_settings()
    if settings is None:
        raise RuntimeError(
            "get_vlm() needs config.yaml with a supports_vision=true model "
            f"bound to role '{role}'. No config.yaml found."
        )
    try:
        cfg = settings.model_for_role(role)
    except KeyError as exc:
        raise RuntimeError(f"config.yaml has no model bound to role '{role}'. {exc}") from exc
    if not cfg.supports_vision:
        raise RuntimeError(
            f"Model '{cfg.name}' bound to role '{role}' has supports_vision=false; "
            "point the role at a vision-capable model."
        )
    return _build_yaml_client(
        cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra=kwargs,
    )


def current_provider() -> str:
    """Return the YAML provider for ``scene_coder`` role, or 'mimo' in fallback."""
    s = _yaml_settings()
    if s is None:
        return "mimo"
    try:
        return s.model_for_role("scene_coder").provider
    except KeyError:
        return "<unbound>"


def current_model(model: str = "flash") -> str:
    role = _resolve_role(model)
    s = _yaml_settings()
    if s is None:
        return _MIMO_FALLBACK_MODEL.get(model, _MIMO_FALLBACK_MODEL.get(role, "?"))
    try:
        return s.model_for_role(role).model
    except KeyError:
        return "<unbound>"


def is_vision_capable(model: str = "vision_checker") -> bool:
    role = _resolve_role(model)
    s = _yaml_settings()
    if s is None:
        return False
    try:
        return s.model_for_role(role).supports_vision
    except KeyError:
        return False


def safe_structured_invoke(
    llm: BaseChatModel,
    model_cls: type[_T],
    messages: Sequence[tuple[str, str]],
    *,
    retries: int = 1,
) -> _T:
    """``llm.with_structured_output(model_cls)`` with one auto-retry on parse fail.

    Uses ``method="function_calling"`` because Volcengine Ark (Doubao) rejects
    ``response_format={"type":"json_schema"|"json_object"}``, while all four
    supported provider styles (mimo / deepseek / doubao / openai / anthropic)
    accept tool calling.

    Raises
    ------
    OutputParserException / ValidationError
        If both attempts fail; callers should catch and convert to ``fatal_error``.
    """
    structured = llm.with_structured_output(model_cls, method="function_calling")
    last_exc: Exception | None = None
    msgs: list[tuple[str, str]] = list(messages)
    for attempt in range(retries + 1):
        try:
            return structured.invoke(msgs)  # type: ignore[return-value]
        except (OutputParserException, ValidationError) as exc:
            last_exc = exc
            log.warning(
                "[llm.safe_structured_invoke] attempt %d/%d failed for %s: %s",
                attempt + 1,
                retries + 1,
                model_cls.__name__,
                str(exc)[:300],
            )
            if attempt < retries and msgs and msgs[-1][0] == "user":
                msgs = msgs[:-1] + [
                    (
                        "user",
                        msgs[-1][1]
                        + "\n\nIMPORTANT: respond with STRICT JSON matching the schema. "
                        "No prose, no commentary, no markdown fences.",
                    )
                ]
    assert last_exc is not None
    raise last_exc
