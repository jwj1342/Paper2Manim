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
