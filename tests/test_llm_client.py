"""Model configuration routing without making API calls."""

from unittest.mock import MagicMock

import pytest
import yaml

from paper2manim import llm as llm_mod


def test_missing_config_requires_explicit_setup(monkeypatch, tmp_path):
    # An example beside the missing config must never be loaded automatically.
    (tmp_path / "config.example.yaml").write_text("models: []\n", encoding="utf-8")
    monkeypatch.setattr(llm_mod, "_CONFIG_PATH", tmp_path / "missing.yaml")
    llm_mod.reload_config()
    constructor = MagicMock()
    monkeypatch.setattr(llm_mod, "ChatOpenAI", constructor)

    with pytest.raises(RuntimeError, match="No config.yaml.*Copy config.example.yaml"):
        llm_mod.get_llm()
    constructor.assert_not_called()
    assert llm_mod.current_provider() == "<unconfigured>"
    assert all(
        llm_mod.current_model(role) == "<unconfigured>" for role in llm_mod.CANONICAL_ROLES
    )
    assert llm_mod.is_vision_capable() is False
    assert llm_mod.tts_config() is None


@pytest.mark.parametrize("factory", [llm_mod.get_vlm, llm_mod.vision_checker_config])
def test_missing_config_has_no_vision_default(factory, monkeypatch, tmp_path):
    monkeypatch.setattr(llm_mod, "_CONFIG_PATH", tmp_path / "missing.yaml")
    llm_mod.reload_config()
    with pytest.raises(RuntimeError, match="config.yaml"):
        factory()


def test_get_llm_uses_configured_role_model_and_endpoint():
    config_path = llm_mod._CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"].append(
        {
            "name": "reader",
            "provider": "openai_compatible",
            "model": "reader-model",
            "api_key": "reader-key",
            "base_url": "http://localhost:0/reader/v1",
        }
    )
    config["model_roles"]["global_reader"] = "reader"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    llm_mod.reload_config()

    coder = llm_mod.get_llm()
    reader = llm_mod.get_llm("global_reader")
    assert coder.model_name == "test-model"
    assert coder.openai_api_base == "http://localhost:0/v1"
    assert reader.model_name == "reader-model"
    assert reader.openai_api_base == "http://localhost:0/reader/v1"
    assert reader.openai_api_key.get_secret_value() == "reader-key"
    assert llm_mod.current_provider() == "openai_compatible"
    assert llm_mod.current_model("global_reader") == "reader-model"


@pytest.mark.parametrize("role", ["flash", "pro", "v2", "v2-omni", "unknown_role"])
def test_unbound_role_is_not_translated_or_defaulted(role, monkeypatch):
    constructor = MagicMock()
    monkeypatch.setattr(llm_mod, "ChatOpenAI", constructor)
    with pytest.raises(RuntimeError, match=f"no model bound to role '{role}'"):
        llm_mod.get_llm(role)
    constructor.assert_not_called()
    assert llm_mod.current_model(role) == "<unbound>"


def test_malformed_config_yaml_raises_loudly(monkeypatch, tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("models:\n  - not-a-mapping\n", encoding="utf-8")
    monkeypatch.setattr(llm_mod, "_CONFIG_PATH", bad)
    llm_mod.reload_config()
    constructor = MagicMock()
    monkeypatch.setattr(llm_mod, "ChatOpenAI", constructor)
    with pytest.raises(RuntimeError, match="config.yaml.*failed to load"):
        llm_mod.get_llm("scene_coder")
    constructor.assert_not_called()


def test_empty_configured_key_is_rejected(monkeypatch):
    config_path = llm_mod._CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"][0]["api_key"] = ""
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    llm_mod.reload_config()
    constructor = MagicMock()
    monkeypatch.setattr(llm_mod, "ChatOpenAI", constructor)
    with pytest.raises(RuntimeError, match="empty api_key"):
        llm_mod.get_llm()
    constructor.assert_not_called()
