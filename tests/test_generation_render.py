"""API-free integration test with real Manim, frame sampling, and assembly."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper2manim.artifacts import run_dir
from paper2manim.emb.manager import build_in_memory_emb
from paper2manim.graphs.generation import build_generation_graph
from paper2manim.schemas import StoryboardModel


@pytest.mark.slow
def test_text_to_video_and_memory_with_real_renderer(monkeypatch):
    storyboard = StoryboardModel(
        title="Shape transformation",
        scenes=[
            {
                "name": "ShapeScene",
                "description": "Create a circle and transform it into a square.",
                "paper_claim": "Shapes can transform.",
                "paper_evidence": "The supplied transformation example.",
                "final_takeaway": "The same object changes its geometry.",
                "duration_hint": 2,
            }
        ],
    )
    planner = MagicMock()
    planner.with_structured_output.return_value.invoke.return_value = storyboard
    coder = MagicMock()
    coder.invoke.return_value.content = """from manim import *
class ShapeScene(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        self.play(Create(circle), run_time=0.4)
        self.play(Transform(circle, Square(color=GREEN)), run_time=0.4)
        self.wait(0.2)
"""
    monkeypatch.setattr("paper2manim.agents.storyboarder.get_llm", lambda *args, **kwargs: planner)
    monkeypatch.setattr("paper2manim.agents.coder.get_llm", lambda *args, **kwargs: coder)
    monkeypatch.setattr(
        "paper2manim.graphs.scene_graph.review_scene",
        lambda *args, **kwargs: {
            "decision": "pass",
            "scores": {"logic_flow": 92, "layout_occlusion": 92, "accuracy": 92},
            "average_score": 92,
        },
    )
    monkeypatch.setattr(
        "paper2manim.agents.rationale_writer.write_rationale_llm",
        lambda *args: "A staged creation and transformation keeps the visual focus clear.",
    )
    emb = build_in_memory_emb()
    out = build_generation_graph().invoke(
        {
            "run_id": "real-render",
            "input_kind": "text",
            "raw_text": "Create a circle, then transform it into a square.",
            "scene_role": "METHOD",
            "quality": "l",
            "max_retries": 2,
            "max_visual_revisions": 2,
            "vlm_enabled": True,
            "emb_enabled": True,
            "emb_instance": emb,
        }
    )
    assert not out.get("fatal_error"), out
    assert Path(out["final_video_path"]).stat().st_size > 1000
    assert (run_dir("real-render") / "vlm_frames" / "ShapeScene_v0.png").is_file()
    assert out["scene_reports"][0]["reflection_rounds"] == 0
    assert emb.stats()["success"] == 1
    record = emb.all(polarity="success")[0]
    assert record.body.frame_hash
    assert record.context.scene_role == "METHOD"
