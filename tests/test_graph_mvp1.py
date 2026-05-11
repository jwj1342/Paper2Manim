"""End-to-end smoke test for the MVP 1.0 graph (mocks LLM and render)."""

from unittest.mock import MagicMock

import pytest

from paper2manim.schemas import StoryboardModel


@pytest.fixture
def stub_llm(monkeypatch):
    """Stub get_llm for storyboarder + coder."""
    sb_obj = StoryboardModel(
        title="Test",
        scenes=[{"name": "TestScene", "description": "t", "duration_hint": 5.0}],
    )

    structured = MagicMock()
    structured.invoke.return_value = sb_obj

    coder_msg = MagicMock()
    coder_msg.content = "```python\nfrom manim import *\nclass TestScene(Scene):\n    def construct(self):\n        self.wait(1)\n```"

    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = coder_msg

    monkeypatch.setattr("paper2manim.agents.storyboarder.get_llm", lambda *a, **kw: llm)
    monkeypatch.setattr("paper2manim.agents.coder.get_llm", lambda *a, **kw: llm)
    return llm


def test_mvp1_graph_skip_render(stub_llm):
    from paper2manim.graphs.mvp1 import build_mvp1_graph

    g = build_mvp1_graph()
    state = {
        "run_id": "test-run-mvp1",
        "input_kind": "text",
        "raw_text": "hello",
        "attempts": [],
        "rendered_videos": [],
        "iter_count": 0,
        "current_scene_idx": 0,
        "max_retries": 1,
        "quality": "l",
        "skip_render": True,
    }
    out = g.invoke(state)
    assert out["storyboard"]["title"] == "Test"
    assert out["current_code"].startswith("from manim")
    # render skipped, no fatal_error
    assert not out.get("fatal_error")


def test_mvp1_graph_render_success(stub_llm, monkeypatch):
    """Patch render() to return a fake success result without invoking real manim."""
    from paper2manim.graphs.mvp1 import build_mvp1_graph

    fake_mp4 = "/tmp/fake.mp4"

    def fake_render(code, scene, **kw):
        # write a fake mp4 so copy_final_video has something
        from pathlib import Path

        Path(fake_mp4).write_bytes(b"\x00")
        return {
            "status": "success",
            "category": None,
            "exit_code": 0,
            "scene": scene,
            "video_path": fake_mp4,
            "workdir": kw.get("workdir", "/tmp"),
        }

    monkeypatch.setattr("paper2manim.graphs.mvp1.render", fake_render)
    g = build_mvp1_graph()
    state = {
        "run_id": "test-run-mvp1-ok",
        "input_kind": "text",
        "raw_text": "hello",
        "attempts": [],
        "rendered_videos": [],
        "iter_count": 0,
        "current_scene_idx": 0,
        "max_retries": 1,
        "quality": "l",
        "skip_render": False,
    }
    out = g.invoke(state)
    assert out.get("final_video_path", "").endswith(".mp4")
    assert len(out["attempts"]) == 1
    assert out["attempts"][0]["render_result"]["status"] == "success"
