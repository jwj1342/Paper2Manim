"""Tests for the storyboarder agent."""

from unittest.mock import MagicMock

from paper2manim.agents.storyboarder import storyboarder_node
from paper2manim.schemas import StoryboardModel


def test_storyboarder_returns_storyboard(mock_llm):
    sb_obj = StoryboardModel(
        title="Pythagoras",
        scenes=[
            {
                "name": "PythagorasIntro",
                "description": "shows the theorem",
                "duration_hint": 12.0,
            }
        ],
    )
    structured = MagicMock()
    structured.invoke.return_value = sb_obj
    mock_llm.with_structured_output.return_value = structured

    state = {"run_id": "test-run", "raw_text": "Pythagorean theorem", "attempts": []}
    out = storyboarder_node(state)
    assert out["storyboard"]["title"] == "Pythagoras"
    assert out["current_scene_idx"] == 0
    assert out["iter_count"] == 0


def test_storyboarder_fatal_when_no_input(mock_llm):
    state = {"run_id": "test-run", "attempts": []}
    out = storyboarder_node(state)
    assert "fatal_error" in out


def test_storyboarder_fatal_on_validation_error(monkeypatch):
    """C3 regression: structured-output failure must surface as fatal_error,
    not raise through the graph."""
    from pydantic import ValidationError

    def fake_safe_invoke(*args, **kwargs):
        # Simulate the model returning a JSON that fails StoryboardModel validation
        try:
            StoryboardModel.model_validate({"title": "x"})  # missing 'scenes'
        except ValidationError as e:
            raise e
        raise AssertionError("expected ValidationError")

    monkeypatch.setattr(
        "paper2manim.agents.storyboarder.safe_structured_invoke", fake_safe_invoke
    )
    monkeypatch.setattr("paper2manim.agents.storyboarder.get_llm", lambda *a, **kw: object())

    state = {"run_id": "test-run", "raw_text": "Pythagorean theorem", "attempts": []}
    out = storyboarder_node(state)
    assert "fatal_error" in out
    assert "storyboarder" in out["fatal_error"]
    assert "ValidationError" in out["fatal_error"]
