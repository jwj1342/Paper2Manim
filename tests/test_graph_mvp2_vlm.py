"""VLM Multi-Dim Scoring loop tests on the MVP 2.0 graph.

The graph is stubbed end-to-end:
- parser / summarizer / storyboarder / coder / reviewer LLM calls return canned
  responses (same approach as test_graph_mvp2.py).
- ``sample_frames_montage`` is patched to produce an empty PNG without ffmpeg.
- ``vlm_scene_reviewer.review_scene`` is patched to return a scripted decision
  on first call and ``pass`` on subsequent calls — proving the revise→re-render
  loop fires and then advances.
- ``visual_revision_agent.revise_code`` is patched to return a sentinel string
  so we can assert the revised code reached ``current_code``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper2manim.schemas import StoryboardModel, SummaryModel


def _stub_llm() -> MagicMock:
    summary = SummaryModel(
        title="T", key_contributions=["c"], key_formulas=[],
        main_concepts=["m"], scene_suggestions=["s"],
    )
    sb = StoryboardModel(
        title="T",
        scenes=[{"name": "Scene1", "description": "d", "duration_hint": 4.0}],
    )
    summarizer_struct = MagicMock()
    summarizer_struct.invoke.return_value = summary
    sb_struct = MagicMock()
    sb_struct.invoke.return_value = sb

    coder_msg = MagicMock()
    coder_msg.content = (
        "```python\nfrom manim import *\nclass Scene1(Scene):\n"
        "    def construct(self):\n        self.wait(0.1)\n```"
    )
    reviewer_msg = MagicMock()
    reviewer_msg.content = '{"decision":"retry","hint":""}'

    llm = MagicMock()

    def structured(model_cls, **_kw):
        if model_cls.__name__ == "SummaryModel":
            return summarizer_struct
        if model_cls.__name__ == "StoryboardModel":
            return sb_struct
        return MagicMock()

    llm.with_structured_output.side_effect = structured

    def invoke(messages):
        sys_msg = messages[0][1] if messages else ""
        if "review" in sys_msg.lower() and "build-and-review" in sys_msg.lower():
            return reviewer_msg
        return coder_msg

    llm.invoke.side_effect = invoke
    return llm


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path):
    llm = _stub_llm()
    monkeypatch.setattr("paper2manim.agents.storyboarder.get_llm", lambda *a, **kw: llm)
    monkeypatch.setattr("paper2manim.agents.coder.get_llm", lambda *a, **kw: llm)
    monkeypatch.setattr("paper2manim.agents.summarizer.get_llm", lambda *a, **kw: llm)
    monkeypatch.setattr("paper2manim.agents.reviewer.get_llm", lambda *a, **kw: llm)

    from paper2manim.parsers import ParsedInput
    monkeypatch.setattr(
        "paper2manim.graphs.mvp2.parse_local_pdf",
        lambda p: ParsedInput(text="# T", fmt="markdown", source="pdf:fake"),
    )

    # All renders succeed (visual revisions need a working mp4 to montage from).
    def fake_render(code, scene, **kw):
        workdir = Path(kw.get("workdir", tmp_path))
        workdir.mkdir(parents=True, exist_ok=True)
        mp4 = workdir / "scene.mp4"
        mp4.write_bytes(b"\x00")
        return {
            "status": "success", "category": None, "exit_code": 0,
            "scene": scene, "video_path": str(mp4), "workdir": str(workdir),
        }

    monkeypatch.setattr("paper2manim.graphs.mvp2.render", fake_render)

    # Skip ffmpeg in the sampler — just write a placeholder PNG.
    def fake_sample(video_path, out_png, **kw):
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"\x89PNG\r\n\x1a\n")
        return out_png

    monkeypatch.setattr("paper2manim.graphs.mvp2.sample_frames_montage", fake_sample)

    def fake_concat(paths, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00")
        return out

    monkeypatch.setattr("paper2manim.graphs.mvp2.concat_videos", fake_concat)

    return {"llm": llm}


def _run(initial_state):
    from paper2manim.graphs.mvp2 import build_mvp2_graph
    g = build_mvp2_graph()
    return g.invoke(initial_state, config={"recursion_limit": 80})


def test_vlm_pass_short_circuits_to_advance(stub_pipeline, monkeypatch, tmp_path):
    """decision=pass on first review → no visual_revise, scene advances."""
    review_calls = {"n": 0}

    def review_pass(*a, **kw):
        review_calls["n"] += 1
        return {"scene_id": "Scene1", "decision": "pass", "scores": {},
                "revision_instruction": "", "raw_response": ""}

    monkeypatch.setattr("paper2manim.graphs.mvp2.review_scene", review_pass)
    out = _run({
        "run_id": "vlm-pass", "input_kind": "pdf", "pdf_path": str(tmp_path / "f.pdf"),
        "attempts": [], "rendered_videos": [], "skipped_scenes": [],
        "current_scene_idx": 0, "iter_count": 0, "max_retries": 2,
        "quality": "l", "skip_render": False,
        "vlm_enabled": True, "vlm_revision_count": 0,
        "max_visual_revisions": 2, "visual_revision_decisions": [],
    })
    assert review_calls["n"] == 1
    assert out["visual_revision_decisions"] == ["pass"]
    assert out["last_visual_review"]["decision"] == "pass"
    assert out["final_video_path"].endswith("output.mp4")


def test_vlm_revise_then_pass_advances(stub_pipeline, monkeypatch, tmp_path):
    """First review = revise → visual_revise rewrites code → re-render → second review = pass."""
    review_calls = {"n": 0}

    def review_alternating(*a, **kw):
        review_calls["n"] += 1
        if review_calls["n"] == 1:
            return {"scene_id": "Scene1", "decision": "revise", "scores": {"v": 2},
                    "revision_instruction": "Make text bigger", "raw_response": ""}
        return {"scene_id": "Scene1", "decision": "pass", "scores": {"v": 5},
                "revision_instruction": "", "raw_response": ""}

    revise_calls = {"n": 0}

    def revise_code(*a, **kw):
        revise_calls["n"] += 1
        return "from manim import *\nclass Scene1(Scene):\n    def construct(self):\n        self.wait(0.2)\n"

    monkeypatch.setattr("paper2manim.graphs.mvp2.review_scene", review_alternating)
    monkeypatch.setattr("paper2manim.graphs.mvp2.revise_code", revise_code)

    out = _run({
        "run_id": "vlm-revise", "input_kind": "pdf", "pdf_path": str(tmp_path / "f.pdf"),
        "attempts": [], "rendered_videos": [], "skipped_scenes": [],
        "current_scene_idx": 0, "iter_count": 0, "max_retries": 2,
        "quality": "l", "skip_render": False,
        "vlm_enabled": True, "vlm_revision_count": 0,
        "max_visual_revisions": 2, "visual_revision_decisions": [],
    })

    assert review_calls["n"] == 2
    assert revise_calls["n"] == 1
    assert out["visual_revision_decisions"] == ["revise", "pass"]
    # Render ran twice for the same scene (original + 1 revision).
    assert len(out["attempts"]) == 2
    assert out["final_video_path"].endswith("output.mp4")


def test_vlm_revise_cap_advances(stub_pipeline, monkeypatch, tmp_path):
    """All reviews say revise → loop terminates at max_visual_revisions and advances."""
    review_calls = {"n": 0}

    def always_revise(*a, **kw):
        review_calls["n"] += 1
        return {"scene_id": "Scene1", "decision": "revise", "scores": {},
                "revision_instruction": "again", "raw_response": ""}

    monkeypatch.setattr("paper2manim.graphs.mvp2.review_scene", always_revise)
    monkeypatch.setattr("paper2manim.graphs.mvp2.revise_code", lambda *a, **kw: "from manim import *\nclass Scene1(Scene):\n  def construct(self):\n    self.wait(0.1)\n")

    out = _run({
        "run_id": "vlm-cap", "input_kind": "pdf", "pdf_path": str(tmp_path / "f.pdf"),
        "attempts": [], "rendered_videos": [], "skipped_scenes": [],
        "current_scene_idx": 0, "iter_count": 0, "max_retries": 2,
        "quality": "l", "skip_render": False,
        "vlm_enabled": True, "vlm_revision_count": 0,
        "max_visual_revisions": 2, "visual_revision_decisions": [],
    })
    # cap=2 means 1 original render + 2 visual revisions = 3 renders, then advance
    assert review_calls["n"] == 3  # initial + 2 post-revision reviews
    assert out["visual_revision_decisions"] == ["revise", "revise", "revise"]
    assert len(out["attempts"]) == 3
    assert out["final_video_path"].endswith("output.mp4")


def test_vlm_disabled_skips_loop(stub_pipeline, monkeypatch, tmp_path):
    """When vlm_enabled is False, neither sampler nor reviewer should run."""
    sampler_calls = {"n": 0}
    review_calls = {"n": 0}

    def maybe_sample(*a, **kw):
        sampler_calls["n"] += 1
        return Path(a[1] if len(a) > 1 else kw["out_png"])

    def maybe_review(*a, **kw):
        review_calls["n"] += 1
        return {"decision": "pass"}

    monkeypatch.setattr("paper2manim.graphs.mvp2.sample_frames_montage", maybe_sample)
    monkeypatch.setattr("paper2manim.graphs.mvp2.review_scene", maybe_review)

    out = _run({
        "run_id": "vlm-off", "input_kind": "pdf", "pdf_path": str(tmp_path / "f.pdf"),
        "attempts": [], "rendered_videos": [], "skipped_scenes": [],
        "current_scene_idx": 0, "iter_count": 0, "max_retries": 2,
        "quality": "l", "skip_render": False,
        "vlm_enabled": False, "vlm_revision_count": 0,
        "max_visual_revisions": 2, "visual_revision_decisions": [],
    })
    assert sampler_calls["n"] == 0
    assert review_calls["n"] == 0
    assert out["final_video_path"].endswith("output.mp4")
