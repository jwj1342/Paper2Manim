"""Smoke tests for the multi-provider LLM factory — never call any real API."""

from __future__ import annotations

import pytest

from paper2manim.llm import (
    PROVIDER_TABLE,
    current_key_source,
    current_model,
    current_provider,
    get_llm,
    is_vision_capable,
)


def _reload_settings(monkeypatch):
    """Clear and rebuild the cached Settings singleton after env changes."""
    from paper2manim import config

    config.get_settings.cache_clear()
    config.settings = config.get_settings()  # type: ignore[attr-defined]


def _clear_all_provider_keys(monkeypatch):
    """Force every provider key env to an empty string so the developer's local
    .env file values do not leak into tests (pydantic-settings falls back to the
    .env file when an env var is unset, so we set "" explicitly to override)."""
    for name in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL",
                 "LLM_MODEL_FLASH", "LLM_MODEL_PRO", "VLM_MODEL"):
        monkeypatch.setenv(name, "")
    for env in {e for cfg in PROVIDER_TABLE.values() for e in cfg["key_envs"]}:
        monkeypatch.setenv(env, "")


# ---------------- legacy/back-compat (MiMo path) ----------------

def test_get_llm_requires_key(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mimo")
    _reload_settings(monkeypatch)
    with pytest.raises(RuntimeError, match="MIMO_API_KEY"):
        get_llm("flash")


def test_get_llm_returns_chat_openai(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mimo")
    monkeypatch.setenv("MIMO_API_KEY", "tp-test")
    _reload_settings(monkeypatch)
    llm = get_llm("flash")
    assert llm.model_name == "mimo-v2.5" or getattr(llm, "model", None) == "mimo-v2.5"


def test_get_llm_unknown_role():
    with pytest.raises(ValueError, match="role"):
        get_llm("ultra")  # type: ignore[arg-type]


# ---------------- multi-provider ----------------

def test_unknown_provider_raises(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    _reload_settings(monkeypatch)
    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        get_llm("flash")


def test_deepseek_provider_routes_correctly(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    _reload_settings(monkeypatch)
    llm = get_llm("flash")
    assert current_provider() == "deepseek"
    assert current_model("flash") == "deepseek-chat"
    assert "deepseek.com" in str(getattr(llm, "openai_api_base", "") or llm.root_client.base_url)


def test_doubao_reads_vlm_model_default(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "doubao")
    monkeypatch.setenv("VLM_API_KEY", "ark-test")
    monkeypatch.setenv("VLM_MODEL", "doubao-test-1.5")
    _reload_settings(monkeypatch)
    assert current_provider() == "doubao"
    assert current_model("flash") == "doubao-test-1.5"
    assert current_model("pro") == "doubao-test-1.5"
    assert current_key_source() == "VLM_API_KEY"


def test_unified_key_overrides_provider_specific(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-specific")
    monkeypatch.setenv("LLM_API_KEY", "sk-unified")
    _reload_settings(monkeypatch)
    assert current_key_source() == "LLM_API_KEY"


def test_role_model_override(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mimo")
    monkeypatch.setenv("MIMO_API_KEY", "tp-x")
    monkeypatch.setenv("LLM_MODEL_FLASH", "mimo-custom-flash")
    monkeypatch.setenv("LLM_MODEL_PRO", "mimo-custom-pro")
    _reload_settings(monkeypatch)
    assert current_model("flash") == "mimo-custom-flash"
    assert current_model("pro") == "mimo-custom-pro"


# ---------------- vision capability ----------------

def test_require_vision_rejects_text_only_provider(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    _reload_settings(monkeypatch)
    assert is_vision_capable() is False
    with pytest.raises(RuntimeError, match="text-only"):
        get_llm("flash", require_vision=True)


def test_require_vision_accepts_vision_capable_provider(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "doubao")
    monkeypatch.setenv("VLM_API_KEY", "ark-x")
    monkeypatch.setenv("VLM_MODEL", "doubao-vision-1")
    _reload_settings(monkeypatch)
    assert is_vision_capable() is True
    llm = get_llm("flash", require_vision=True)
    assert llm is not None


# ---------------- auto-detect ----------------

def test_provider_autodetect_from_env(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    # No LLM_PROVIDER, only DOUBAO_API_KEY set → should auto-pick doubao.
    monkeypatch.setenv("DOUBAO_API_KEY", "ark-x")
    monkeypatch.setenv("VLM_MODEL", "doubao-auto-1")
    _reload_settings(monkeypatch)
    assert current_provider() == "doubao"


def test_autodetect_picks_first_in_order(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("MIMO_API_KEY", "tp-x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    _reload_settings(monkeypatch)
    # mimo comes first in _AUTODETECT_ORDER
    assert current_provider() == "mimo"


def test_no_keys_raises(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    _reload_settings(monkeypatch)
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        get_llm("flash")
