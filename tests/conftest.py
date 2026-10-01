"""Pytest fixtures for paper2manim."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture(autouse=True)
def _isolated_runs_dir(tmp_path, monkeypatch):
    """Force PAPER2MANIM_RUNS_DIR to a tmp dir for every test.

    Use a temporary model configuration with test credentials and a local
    endpoint, independent of a developer's config.yaml.

    NOTE on the rebind dance: ``paper2manim.config.env.settings`` is a
    module-level singleton built at first import, and ``paper2manim.artifacts``
    captures it via ``from paper2manim.config.env import settings``. Just
    setting ``PAPER2MANIM_RUNS_DIR`` env var here would silently leave
    ``artifacts.settings.PAPER2MANIM_RUNS_DIR`` pointing at the original
    project ``runs/`` directory — and trace.jsonl + attempts/* would
    accumulate across test sessions, contaminating any test that calls
    ``parse_trace`` (e.g. EMB consolidation tests). We rebind the attribute on
    every relevant module that already imported it, so ``run_dir()`` writes
    into ``tmp_path``.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("PAPER2MANIM_RUNS_DIR", str(runs))

    from paper2manim.config import env as env_mod

    env_mod.get_settings.cache_clear()
    fresh = env_mod.get_settings()
    monkeypatch.setattr(env_mod, "settings", fresh)
    # Re-bind the captured-at-import-time references in every module that did
    # ``from paper2manim.config.env import settings``. Add new modules to this
    # list if they ever fail to see the tmp dir.
    for mod_path in (
        "paper2manim.artifacts",
        "paper2manim.cli",
        "paper2manim.cli_emb",
    ):
        import importlib
        try:
            mod = importlib.import_module(mod_path)
        except Exception:  # noqa: BLE001 — module may not yet be importable in some test
            continue
        if hasattr(mod, "settings"):
            monkeypatch.setattr(mod, "settings", fresh)

    from paper2manim import llm as llm_mod

    model_config = tmp_path / "config.yaml"
    model_config.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "name": "test",
                        "provider": "openai_compatible",
                        "model": "test-model",
                        "api_key": "test-key",
                        "base_url": "http://localhost:0/v1",
                        "supports_vision": True,
                    }
                ],
                "model_roles": {role: "test" for role in llm_mod.CANONICAL_ROLES},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_mod, "_CONFIG_PATH", model_config)
    llm_mod.reload_config()

    yield runs
    # monkeypatch teardown automatically reverts settings rebinds; we just need
    # to clear lru_cache so the next test gets a fresh build.
    env_mod.get_settings.cache_clear()
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

    def fake_get_llm(model="scene_coder", **kw):
        return mock

    monkeypatch.setattr("paper2manim.llm.get_llm", fake_get_llm)
    monkeypatch.setattr("paper2manim.agents.storyboarder.get_llm", fake_get_llm)
    monkeypatch.setattr("paper2manim.agents.coder.get_llm", fake_get_llm)
    monkeypatch.setattr("paper2manim.agents.summarizer.get_llm", fake_get_llm)
    monkeypatch.setattr("paper2manim.agents.reviewer.get_llm", fake_get_llm)
    return mock
