"""Smoke test for llm.get_llm — does NOT call the real MiMo API."""


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


def test_malformed_config_yaml_raises_loudly(monkeypatch, tmp_path):
    """A present-but-broken config.yaml must NOT silently drop back to env-MiMo."""
    bad = tmp_path / "config.yaml"
    bad.write_text("models:\n  - not-a-mapping\n", encoding="utf-8")
    from paper2manim import llm as llm_mod

    monkeypatch.setattr(llm_mod, "_CONFIG_PATH", bad)
    llm_mod.reload_config()
    with pytest.raises(RuntimeError, match="config.yaml.*failed to load"):
        get_llm("scene_coder")
