"""Multi-provider LLM client factory — thin wrapper over langchain-openai.ChatOpenAI.

Each provider exposes an OpenAI-compatible /chat/completions endpoint, so a single
``ChatOpenAI`` instance can target any of them by varying ``base_url`` and ``model``.
The provider for the current process is resolved once from ``Settings.LLM_PROVIDER``
(or auto-detected from whichever ``*_API_KEY`` is set), and ``get_llm(role)`` builds
clients on demand.

The role layer (``flash`` / ``pro``) is kept for backwards compatibility with the
4 production agents (storyboarder / coder / summarizer / reviewer). Per-role model
overrides live in ``LLM_MODEL_FLASH`` / ``LLM_MODEL_PRO``.

Vision capability is declared per provider via ``supports_vision``. Callers that
require an image-capable client (future ``vlm_scene_reviewer`` integration) should
pass ``require_vision=True`` — mis-configured providers raise at construction time
rather than failing mid-graph.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from paper2manim.config import get_settings

log = logging.getLogger(__name__)
_T = TypeVar("_T", bound=BaseModel)

Role = Literal["flash", "pro"]

# Provider registry. ``flash``/``pro`` give a default model id per role; if the
# value is ``None`` we resolve it lazily (currently only doubao, which reads
# ``Settings.VLM_MODEL`` so users with a pre-existing Volcengine setup don't have
# to duplicate the value). ``key_envs`` is the ordered fallback chain of
# ``Settings`` attributes consulted when ``LLM_API_KEY`` is not set.
PROVIDER_TABLE: dict[str, dict[str, Any]] = {
    "mimo": {
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "flash": "mimo-v2.5",
        "pro": "mimo-v2.5-pro",
        "supports_vision": False,
        "key_envs": ("MIMO_API_KEY",),
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "flash": "deepseek-chat",
        "pro": "deepseek-reasoner",
        "supports_vision": False,
        "key_envs": ("DEEPSEEK_API_KEY",),
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "flash": None,  # resolved from settings.VLM_MODEL at call time
        "pro": None,
        "supports_vision": True,
        "key_envs": ("DOUBAO_API_KEY", "VLM_API_KEY"),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "flash": "gpt-4o-mini",
        "pro": "gpt-4o",
        "supports_vision": True,
        "key_envs": ("OPENAI_API_KEY",),
    },
}

# Auto-detect order: first non-empty *_API_KEY wins when LLM_PROVIDER is unset.
_AUTODETECT_ORDER = ("mimo", "deepseek", "doubao", "openai")


def _resolve_provider() -> str:
    s = get_settings()
    explicit = (s.LLM_PROVIDER or "").strip().lower()
    if explicit:
        if explicit not in PROVIDER_TABLE:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER '{explicit}'. "
                f"Choose one of: {', '.join(sorted(PROVIDER_TABLE))}."
            )
        return explicit
    # Auto-detect: pick whichever provider has at least one key set.
    for name in _AUTODETECT_ORDER:
        if _resolve_key_for(name)[0]:
            return name
    raise RuntimeError(
        "No LLM provider configured. Set LLM_PROVIDER and one of "
        f"{', '.join(sorted(set(env for p in PROVIDER_TABLE.values() for env in p['key_envs'])))} "
        "in .env (or copy from .env.example)."
    )


def _resolve_key_for(provider: str) -> tuple[str, str]:
    """Return ``(key, source_env_name)`` for the given provider; key is empty if unset."""
    s = get_settings()
    # Generic LLM_API_KEY beats provider-specific
    generic = (s.LLM_API_KEY or "").strip()
    if generic:
        return generic, "LLM_API_KEY"
    for env in PROVIDER_TABLE[provider]["key_envs"]:
        value = (getattr(s, env, "") or "").strip()
        if value:
            return value, env
    return "", ""


def _resolve_model_for(provider: str, role: Role) -> str:
    s = get_settings()
    override = (
        s.LLM_MODEL_FLASH if role == "flash" else s.LLM_MODEL_PRO
    ).strip()
    if override:
        return override
    default = PROVIDER_TABLE[provider][role]
    if default is None:
        # doubao currently — fall back to VLM_MODEL when set
        if provider == "doubao":
            fallback = (s.VLM_MODEL or "").strip()
            if fallback:
                return fallback
        raise RuntimeError(
            f"No default model for provider '{provider}' role '{role}'. "
            f"Set LLM_MODEL_{role.upper()} or VLM_MODEL in .env."
        )
    return default


def _resolve_base_url(provider: str) -> str:
    s = get_settings()
    override = (s.LLM_BASE_URL or "").strip()
    if override:
        return override
    # provider=doubao: also accept legacy VLM_BASE_URL when LLM_BASE_URL unset
    if provider == "doubao":
        vlm_url = (s.VLM_BASE_URL or "").strip()
        if vlm_url:
            return vlm_url
    return PROVIDER_TABLE[provider]["base_url"]


def get_llm(
    role: Role = "flash",
    *,
    require_vision: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: float = 120,
    **kwargs,
) -> ChatOpenAI:
    """Build a ChatOpenAI client targeting the configured provider.

    Parameters
    ----------
    role : "flash" | "pro"
        Selects the per-role model id from ``PROVIDER_TABLE`` or
        ``LLM_MODEL_FLASH``/``LLM_MODEL_PRO`` overrides.
    require_vision : bool
        If True, raise when the resolved provider does not declare
        ``supports_vision=True``. Use this from agents that pass image inputs.

    Examples
    --------
    >>> get_llm("flash").invoke("hi").content
    >>> get_llm("pro").with_structured_output(MyPydanticModel).invoke([("system", ...), ("user", ...)])
    """
    if role not in ("flash", "pro"):
        raise ValueError(f"Unknown role '{role}'. Choose 'flash' or 'pro'.")
    provider = _resolve_provider()
    if require_vision and not PROVIDER_TABLE[provider]["supports_vision"]:
        vision_options = [
            name for name, cfg in PROVIDER_TABLE.items() if cfg["supports_vision"]
        ]
        raise RuntimeError(
            f"Role '{role}' on provider '{provider}' is text-only "
            f"(supports_vision=False). Set LLM_PROVIDER to one of: "
            f"{', '.join(sorted(vision_options))}."
        )
    api_key, key_source = _resolve_key_for(provider)
    if not api_key:
        envs = " or ".join(PROVIDER_TABLE[provider]["key_envs"])
        raise RuntimeError(
            f"No API key for provider '{provider}'. "
            f"Set LLM_API_KEY or {envs} in .env."
        )
    base_url = _resolve_base_url(provider)
    model = _resolve_model_for(provider, role)
    log.debug(
        "[llm] provider=%s role=%s model=%s base_url=%s key_source=%s",
        provider, role, model, base_url, key_source,
    )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )


def is_vision_capable(role: Role = "flash") -> bool:
    """Return True iff the configured provider declares ``supports_vision``.

    ``role`` is accepted for symmetry with ``get_llm`` but currently unused —
    capability is a provider-level property.
    """
    del role  # currently provider-wide; reserved for future per-role capability
    return PROVIDER_TABLE[_resolve_provider()]["supports_vision"]


def current_provider() -> str:
    """Return the resolved provider name (auto-detected if LLM_PROVIDER unset)."""
    return _resolve_provider()


def current_model(role: Role = "flash") -> str:
    """Return the resolved model id for the given role under the current provider."""
    return _resolve_model_for(_resolve_provider(), role)


def current_key_source() -> str:
    """Return the env variable name actually providing the API key, or '' if none."""
    return _resolve_key_for(_resolve_provider())[1]


def safe_structured_invoke(
    llm: ChatOpenAI,
    model_cls: type[_T],
    messages: Sequence[tuple[str, str]],
    *,
    retries: int = 1,
) -> _T:
    """Invoke ``llm.with_structured_output(model_cls)`` with one automatic retry on
    schema/parse failures.

    Wraps the common pattern where an LLM occasionally returns commentary plus JSON,
    or JSON that fails pydantic validation. We retry once with a strict reminder.

    Raises
    ------
    OutputParserException / ValidationError
        If both attempts fail; callers should catch and convert to ``fatal_error``.
    """
    structured = llm.with_structured_output(model_cls)
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
