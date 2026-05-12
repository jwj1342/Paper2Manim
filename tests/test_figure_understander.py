"""Tests for paper2manim.agents.figure_understander.

All cases use MockVLMClient or monkeypatch get_vlm — no real VLM HTTP calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper2manim.agents.figure_understander import (
    _safe_parse_understanding,
    figure_understander_node,
)
from paper2manim.infrastructure.vlm.mock_vlm_client import MockVLMClient


def _wrap(semantics: dict, recipe: dict | None) -> str:
    """Helper: build a FigureUnderstanding JSON string."""
    return json.dumps({"semantics": semantics, "recipe": recipe})


_SEM_SCHEMATIC = {
    "fig_type": "schematic",
    "redrawable": True,
    "salience": "high",
    "one_line_summary": "encoder-decoder architecture overview",
    "key_elements": ["encoder", "decoder", "attention"],
}
_SEM_HEATMAP = {
    "fig_type": "heatmap",
    "redrawable": False,
    "salience": "medium",
    "one_line_summary": "attention pattern between source and target tokens",
    "key_elements": ["source tokens", "target tokens", "attention weights"],
}
_RECIPE_BASIC = {
    "template": "vertical_stack",
    "nodes": [
        {"id": "n1", "label": "Encoder", "shape": "rect"},
        {"id": "n2", "label": "Decoder", "shape": "rect"},
    ],
    "edges": [{"source": "n1", "target": "n2", "label": None, "style": "solid"}],
    "annotations": ["Encoder ×6"],
    "animation_hint": "Reveal top-to-bottom",
}

_GOOD_JSON = _wrap(_SEM_SCHEMATIC, _RECIPE_BASIC)
_HEATMAP_JSON = _wrap(_SEM_HEATMAP, None)


# ---- _safe_parse_understanding ----------------------------------------------

def test_parse_understanding_with_recipe():
    sem, rec = _safe_parse_understanding(_GOOD_JSON)
    assert sem is not None
    assert sem["fig_type"] == "schematic"
    assert rec is not None
    assert rec["template"] == "vertical_stack"
    assert len(rec["nodes"]) == 2


def test_parse_understanding_recipe_null_when_not_redrawable():
    sem, rec = _safe_parse_understanding(_HEATMAP_JSON)
    assert sem is not None
    assert sem["redrawable"] is False
    assert rec is None


def test_parse_understanding_drops_recipe_when_inconsistent():
    """If VLM contradicts itself (redrawable=False but recipe present), drop the recipe."""
    bad = _wrap(_SEM_HEATMAP, _RECIPE_BASIC)
    sem, rec = _safe_parse_understanding(bad)
    assert sem is not None
    assert sem["redrawable"] is False
    assert rec is None  # dropped for consistency


def test_parse_understanding_with_fence_and_prose():
    raw = f"Sure! Here is the output:\n```json\n{_GOOD_JSON}\n```\nLet me know."
    sem, rec = _safe_parse_understanding(raw)
    assert sem is not None
    assert sem["fig_type"] == "schematic"
    assert rec is not None


def test_parse_understanding_invalid_enum_value():
    bad = _wrap(
        {**_SEM_SCHEMATIC, "fig_type": "not_a_real_type"},
        _RECIPE_BASIC,
    )
    sem, rec = _safe_parse_understanding(bad)
    assert sem is None
    assert rec is None


def test_parse_understanding_missing_required_field():
    bad = json.dumps({"semantics": {"fig_type": "schematic"}})  # missing redrawable etc.
    sem, rec = _safe_parse_understanding(bad)
    assert sem is None
    assert rec is None


def test_parse_understanding_no_json():
    sem, rec = _safe_parse_understanding("Sorry, I cannot answer.")
    assert sem is None
    assert rec is None


def test_parse_understanding_redrawable_without_recipe_is_ok():
    """VLM may set redrawable=True but produce no recipe (acceptable; coder will embed)."""
    raw = _wrap(_SEM_SCHEMATIC, None)
    sem, rec = _safe_parse_understanding(raw)
    assert sem is not None
    assert sem["redrawable"] is True
    assert rec is None


# ---- figure_understander_node short-circuits --------------------------------

def test_no_figures_returns_empty(monkeypatch):
    # Even with a working VLM, empty figures means do-nothing
    monkeypatch.setattr(
        "paper2manim.agents.figure_understander.get_vlm",
        lambda: MockVLMClient(result=_GOOD_JSON),
    )
    out = figure_understander_node({"run_id": None, "figures": []})
    assert out == {}


def test_no_vlm_returns_empty(monkeypatch):
    monkeypatch.setattr("paper2manim.agents.figure_understander.get_vlm", lambda: None)
    out = figure_understander_node(
        {"run_id": None, "figures": [{"fig_id": "fig_001", "path": "/nonexistent"}]}
    )
    # When VLM is disabled, we return {} — figures stay as parser left them
    assert out == {}


def test_state_with_missing_figures_key(monkeypatch):
    """The node tolerates state without a `figures` key at all (MVP1 path)."""
    monkeypatch.setattr(
        "paper2manim.agents.figure_understander.get_vlm",
        lambda: MockVLMClient(result=_GOOD_JSON),
    )
    out = figure_understander_node({"run_id": None})
    assert out == {}


# ---- figure_understander_node happy + degraded paths ------------------------

def test_happy_path_attaches_semantics_and_recipe(monkeypatch, tmp_path):
    fig_path = tmp_path / "fig_001.png"
    fig_path.write_bytes(b"FAKE-PNG")

    monkeypatch.setattr(
        "paper2manim.agents.figure_understander.get_vlm",
        lambda: MockVLMClient(result=_GOOD_JSON),
    )
    state = {
        "run_id": None,  # disables trace writes
        "figures": [
            {"fig_id": "fig_001", "path": str(fig_path), "source_name": "img.png", "caption": None}
        ],
    }
    out = figure_understander_node(state)
    assert "figures" in out
    assert len(out["figures"]) == 1
    enriched = out["figures"][0]
    sem = enriched["semantics"]
    rec = enriched["recipe"]
    assert sem is not None
    assert sem["fig_type"] == "schematic"
    assert sem["redrawable"] is True
    assert rec is not None
    assert rec["template"] == "vertical_stack"
    # original fields preserved
    assert enriched["fig_id"] == "fig_001"
    assert enriched["caption"] is None


def test_happy_path_heatmap_has_no_recipe(monkeypatch, tmp_path):
    """A non-redrawable figure: semantics populated, recipe stays None."""
    fig_path = tmp_path / "fig_001.png"
    fig_path.write_bytes(b"FAKE-PNG")

    monkeypatch.setattr(
        "paper2manim.agents.figure_understander.get_vlm",
        lambda: MockVLMClient(result=_HEATMAP_JSON),
    )
    state = {
        "run_id": None,
        "figures": [{"fig_id": "fig_001", "path": str(fig_path)}],
    }
    out = figure_understander_node(state)
    enriched = out["figures"][0]
    assert enriched["semantics"]["fig_type"] == "heatmap"
    assert enriched["semantics"]["redrawable"] is False
    assert enriched["recipe"] is None


def test_missing_image_path_marks_both_none(monkeypatch):
    monkeypatch.setattr(
        "paper2manim.agents.figure_understander.get_vlm",
        lambda: MockVLMClient(result=_GOOD_JSON),
    )
    state = {
        "run_id": None,
        "figures": [{"fig_id": "fig_001", "path": "/does/not/exist.png"}],
    }
    out = figure_understander_node(state)
    assert out["figures"][0]["semantics"] is None
    assert out["figures"][0]["recipe"] is None


def test_vlm_raises_marks_both_none(monkeypatch, tmp_path):
    """A VLM exception is logged and the figure's semantics/recipe stay None — no crash."""
    fig_path = tmp_path / "fig_001.png"
    fig_path.write_bytes(b"FAKE-PNG")

    class BoomVLM:
        def review_scene(self, *a, **kw):
            raise RuntimeError("boom")

        def review_images(self, *a, **kw):  # pragma: no cover — unused here
            raise RuntimeError("boom")

    monkeypatch.setattr("paper2manim.agents.figure_understander.get_vlm", lambda: BoomVLM())
    state = {"run_id": None, "figures": [{"fig_id": "fig_001", "path": str(fig_path)}]}
    out = figure_understander_node(state)
    assert out["figures"][0]["semantics"] is None
    assert out["figures"][0]["recipe"] is None


def test_partial_success_mixes_some_none(monkeypatch, tmp_path):
    """Two figures: one with a valid path, one without — only the first gets semantics."""
    fig_path = tmp_path / "fig_001.png"
    fig_path.write_bytes(b"FAKE-PNG")

    monkeypatch.setattr(
        "paper2manim.agents.figure_understander.get_vlm",
        lambda: MockVLMClient(result=_GOOD_JSON),
    )
    state = {
        "run_id": None,
        "figures": [
            {"fig_id": "fig_001", "path": str(fig_path)},
            {"fig_id": "fig_002", "path": "/missing.png"},
        ],
    }
    out = figure_understander_node(state)
    assert out["figures"][0]["semantics"] is not None
    assert out["figures"][1]["semantics"] is None


def test_trace_written_when_run_id_present(monkeypatch, tmp_path):
    """When run_id is set, a parser-style trace event is appended."""
    fig_path = tmp_path / "fig_001.png"
    fig_path.write_bytes(b"FAKE-PNG")
    runs_dir = tmp_path / "runs"

    # Repoint settings.PAPER2MANIM_RUNS_DIR so artifacts.run_dir lands in tmp_path
    from paper2manim.config import env as cfg

    monkeypatch.setattr(cfg.settings, "PAPER2MANIM_RUNS_DIR", str(runs_dir))
    monkeypatch.setattr(
        "paper2manim.agents.figure_understander.get_vlm",
        lambda: MockVLMClient(result=_GOOD_JSON),
    )

    out = figure_understander_node(
        {
            "run_id": "test_run",
            "figures": [{"fig_id": "fig_001", "path": str(fig_path)}],
        }
    )
    assert out["figures"][0]["semantics"] is not None
    assert out["figures"][0]["recipe"] is not None  # _GOOD_JSON has redrawable=True + recipe

    trace_path = runs_dir / "test_run" / "trace.jsonl"
    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["node"] == "figure_understander"
    assert rec["n_figures"] == 1
    assert rec["n_understood"] == 1
    assert rec["n_with_recipe"] == 1
