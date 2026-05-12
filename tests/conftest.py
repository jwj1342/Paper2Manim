"""Pytest fixtures for paper2manim."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolated_runs_dir(tmp_path, monkeypatch):
    """Force PAPER2MANIM_RUNS_DIR to a tmp dir for every test.

    Also force the YAML LLM factory into env-mimo fallback by pointing its
    config path at a guaranteed-absent file. Without this, a developer's local
    ``config.yaml`` (used for real experiments) would silently bypass the
    ``MIMO_API_KEY`` injection below and break every test that asserts on the
    fallback path.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("PAPER2MANIM_RUNS_DIR", str(runs))
    monkeypatch.setenv("MIMO_API_KEY", os.environ.get("MIMO_API_KEY", "tp-test-key"))
    # Force config reload
    from paper2manim import config

    config.get_settings.cache_clear()
    config.settings = config.get_settings()  # type: ignore[attr-defined]

    # Drop yaml-cache and steer the loader at a non-existent path so tests
    # always exercise the env-mimo fallback regardless of whether the dev has
    # a local config.yaml.
    from paper2manim import llm as llm_mod

    monkeypatch.setattr(llm_mod, "_CONFIG_PATH", tmp_path / "config_absent.yaml")
    llm_mod.reload_config()

    yield runs
    # Restore module-level settings after test
    config.get_settings.cache_clear()
    config.settings = config.get_settings()  # type: ignore[attr-defined]
    llm_mod.reload_config()


@pytest.fixture
def fake_storyboard():
    return {
        "title": "Pythagoras",
        "scenes": [
            {
                "name": "PythagorasIntro",
                "description": "Title fades in; right triangle drawn; equation shown.",
                "duration_hint": 12.0,
            }
        ],
    }


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch paper2manim.llm.get_llm to return a controllable mock."""
    mock = MagicMock()

    def fake_get_llm(model="flash", **kw):
        return mock

    monkeypatch.setattr("paper2manim.llm.get_llm", fake_get_llm)
    monkeypatch.setattr("paper2manim.agents.storyboarder.get_llm", fake_get_llm)
    monkeypatch.setattr("paper2manim.agents.coder.get_llm", fake_get_llm)
    monkeypatch.setattr("paper2manim.agents.summarizer.get_llm", fake_get_llm)
    monkeypatch.setattr("paper2manim.agents.reviewer.get_llm", fake_get_llm)
    return mock
