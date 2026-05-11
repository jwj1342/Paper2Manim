"""Smoke test for llm.get_llm — does NOT call the real MiMo API."""

from unittest.mock import patch

import pytest

from paper2manim.llm import get_llm


def test_get_llm_requires_key(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "")
    from paper2manim import config

    config.get_settings.cache_clear()
    config.settings = config.get_settings()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="MIMO_API_KEY"):
        get_llm("flash")


def test_get_llm_returns_chat_openai(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "tp-test")
    from paper2manim import config

    config.get_settings.cache_clear()
    config.settings = config.get_settings()  # type: ignore[attr-defined]
    llm = get_llm("flash")
    # Use attribute access compatible with langchain ChatOpenAI; either works
    assert llm.model_name == "mimo-v2.5" or getattr(llm, "model", None) == "mimo-v2.5"


def test_get_llm_unknown_alias():
    with pytest.raises(ValueError):
        get_llm("ultra")  # type: ignore[arg-type]
