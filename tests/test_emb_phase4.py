"""Phase 4 integration tests: EMB nodes wired into the MVP 2.0 graph.

We reuse the stub pattern from ``test_graph_mvp2_vlm.py`` (mocked LLM, render,
sampler, concat) but inject a pre-built in-memory EMB via ``state["emb_instance"]``
so retrieval / consolidation hit a real ``EpisodicMemoryBank`` without touching
disk or downloading sentence-transformers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper2manim.emb import (
    Context,
    FailureBody,
    MemoryRecord,
    Provenance,
    SuccessBody,
)
from paper2manim.emb.manager import build_in_memory_emb
from paper2manim.graphs.scene_graph import _reset_emb_cache
from paper2manim.schemas import StoryboardModel, SummaryModel

# --------------------------------------------------------------------------- #
# Shared stub helpers (lifted / adapted from test_graph_mvp2_vlm.py)
# --------------------------------------------------------------------------- #


def _stub_llm() -> MagicMock:
    summary = SummaryModel(
        title="T",
        key_contributions=["c"],
        key_formulas=[],
        main_concepts=["m"],
        scene_suggestions=["s"],
    )
    sb = StoryboardModel(
        title="T",
        scenes=[
            {"name": "Scene1", "description": "intro scene for testing", "duration_hint": 4.0}
        ],
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

    coder_invocations: list[str] = []

    def invoke(messages):
        sys_msg = messages[0][1] if messages else ""
        if "review" in sys_msg.lower() and "build-and-review" in sys_msg.lower():
            return reviewer_msg
        # Stash the user prompt content so tests can assert on retrieval injection.
        if len(messages) >= 2:
            coder_invocations.append(messages[1][1])
        return coder_msg

    llm.invoke.side_effect = invoke
    llm._coder_invocations = coder_invocations  # type: ignore[attr-defined]
    return llm


@pytest.fixture
def emb_pipeline(monkeypatch, tmp_path):
    """Stub the whole pipeline so we can drive the graph end-to-end."""
    _reset_emb_cache()
    llm = _stub_llm()
    for attr in (
        "paper2manim.agents.storyboarder.get_llm",
        "paper2manim.agents.coder.get_llm",
        "paper2manim.agents.summarizer.get_llm",
        "paper2manim.agents.reviewer.get_llm",
    ):
        monkeypatch.setattr(attr, lambda *a, **kw: llm)

    from paper2manim.parsers import ParsedInput

    monkeypatch.setattr(
        "paper2manim.graphs.mvp2.parse_local_pdf",
        lambda p: ParsedInput(text="# T", fmt="markdown", source="pdf:fake"),
    )

    def fake_render(code, scene, **kw):
        workdir = Path(kw.get("workdir", tmp_path))
        workdir.mkdir(parents=True, exist_ok=True)
        mp4 = workdir / "scene.mp4"
        mp4.write_bytes(b"\x00")
        return {
            "status": "success",
            "category": None,
            "exit_code": 0,
            "scene": scene,
            "video_path": str(mp4),
            "workdir": str(workdir),
        }

    monkeypatch.setattr("paper2manim.graphs.scene_graph.render", fake_render)

    def fake_sample(video_path, out_png, **kw):
        out = Path(out_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x89PNG\r\n\x1a\n")
        return out

    monkeypatch.setattr("paper2manim.graphs.scene_graph.sample_frames_montage", fake_sample)

    def fake_concat(paths, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00")
        return out

    monkeypatch.setattr("paper2manim.graphs.mvp2.concat_videos", fake_concat)

    yield {"llm": llm}
    _reset_emb_cache()


def _run(initial_state):
    from paper2manim.graphs.mvp2 import build_mvp2_graph

    g = build_mvp2_graph()
    return g.invoke(initial_state, config={"recursion_limit": 80})


# --------------------------------------------------------------------------- #
# Disabled EMB: graph behaves like before
# --------------------------------------------------------------------------- #


class TestEMBDisabled:
    def test_disabled_no_retrieval_or_consolidation(self, emb_pipeline, tmp_path):
        out = _run(
            {
                "run_id": "no-emb",
                "input_kind": "pdf",
                "pdf_path": str(tmp_path / "f.pdf"),
                "attempts": [],
                "rendered_videos": [],
                "skipped_scenes": [],
                "current_scene_idx": 0,
                "iter_count": 0,
                "max_retries": 2,
                "quality": "l",
                "skip_render": False,
                "vlm_enabled": False,
                "vlm_revision_count": 0,
                "max_visual_revisions": 2,
                "visual_revision_decisions": [],
                "emb_enabled": False,
            }
        )
        assert out["final_video_path"].endswith("output.mp4")
        # No retrieval blocks injected into Coder prompt
        prompts = emb_pipeline["llm"]._coder_invocations
        assert prompts, "coder should have been called"
        for p in prompts:
            assert "Reference Examples" not in p
            assert "Known Pitfalls" not in p
        # No consolidation writes
        assert not out.get("emb_writes")

    def test_disabled_with_no_store_path(self, emb_pipeline, tmp_path):
        """emb_enabled=True but no store path and no instance → no-op gracefully."""
        out = _run(
            {
                "run_id": "no-store",
                "input_kind": "pdf",
                "pdf_path": str(tmp_path / "f.pdf"),
                "attempts": [],
                "rendered_videos": [],
                "skipped_scenes": [],
                "current_scene_idx": 0,
                "iter_count": 0,
                "max_retries": 2,
                "quality": "l",
                "skip_render": False,
                "vlm_enabled": False,
                "vlm_revision_count": 0,
                "max_visual_revisions": 2,
                "visual_revision_decisions": [],
                "emb_enabled": True,
                "emb_store_path": None,
                "emb_instance": None,
            }
        )
        assert out["final_video_path"].endswith("output.mp4")
        # No instance and no path → emb_for_state returns None → no writes
        assert not out.get("emb_writes")


# --------------------------------------------------------------------------- #
# Empty EMB enabled: still zero-shot, but consolidation runs
# --------------------------------------------------------------------------- #


class TestEMBEnabledEmpty:
    def test_empty_emb_zero_shot_then_consolidates(self, emb_pipeline, tmp_path):
        emb = build_in_memory_emb()
        out = _run(
            {
                "run_id": "emb-empty",
                "input_kind": "pdf",
                "pdf_path": str(tmp_path / "f.pdf"),
                "attempts": [],
                "rendered_videos": [],
                "skipped_scenes": [],
                "current_scene_idx": 0,
                "iter_count": 0,
                "max_retries": 2,
                "quality": "l",
                "skip_render": False,
                "vlm_enabled": False,
                "vlm_revision_count": 0,
                "max_visual_revisions": 2,
                "visual_revision_decisions": [],
                "emb_enabled": True,
                "emb_theta_high": 0.0,  # accept no-VLM scenes
                "emb_use_llm_distillers": False,
                "emb_instance": emb,
            }
        )
        assert out["final_video_path"].endswith("output.mp4")
        # Coder prompt should not contain retrieval blocks (EMB was empty).
        for p in emb_pipeline["llm"]._coder_invocations:
            assert "Reference Examples" not in p
            assert "Known Pitfalls" not in p
        # Consolidation should have written one success record for Scene1
        # (theta_high=0 makes it acceptable even without VLM).
        assert emb.count(polarity="success") == 1
        # emb_writes carries the consolidation report
        writes = out.get("emb_writes") or []
        assert len(writes) == 1
        assert writes[0]["n_success_written"] == 1
        assert writes[0]["n_failure_written"] == 0


# --------------------------------------------------------------------------- #
# Pre-populated EMB: retrieval injects records into Coder prompt
# --------------------------------------------------------------------------- #


class TestEMBPrepopulated:
    def test_prior_success_record_appears_in_coder_prompt(self, emb_pipeline, tmp_path):
        emb = build_in_memory_emb()
        emb.put(
            MemoryRecord(
                polarity="success",
                context=Context(
                    task_text="intro scene for testing",
                    source_paper="arxiv:prior",
                    source_section="Background",
                ),
                body=SuccessBody(
                    rationale="PRIOR_RATIONALE_MARKER",
                    code_full="# PRIOR_SUCCESS_CODE_MARKER",
                ),
                provenance=Provenance(scene_id="PriorIntro", validated=True, vlm_score=4.5),
            )
        )
        out = _run(
            {
                "run_id": "emb-prefilled-s",
                "input_kind": "pdf",
                "pdf_path": str(tmp_path / "f.pdf"),
                "attempts": [],
                "rendered_videos": [],
                "skipped_scenes": [],
                "current_scene_idx": 0,
                "iter_count": 0,
                "max_retries": 2,
                "quality": "l",
                "skip_render": False,
                "vlm_enabled": False,
                "vlm_revision_count": 0,
                "max_visual_revisions": 2,
                "visual_revision_decisions": [],
                "emb_enabled": True,
                "emb_theta_high": 0.0,
                "emb_instance": emb,
            }
        )
        assert out["final_video_path"].endswith("output.mp4")
        # First Coder call should contain Reference Examples with our marker.
        prompts = emb_pipeline["llm"]._coder_invocations
        assert prompts
        assert any("Reference Examples" in p for p in prompts)
        assert any("PRIOR_RATIONALE_MARKER" in p for p in prompts)
        assert any("PRIOR_SUCCESS_CODE_MARKER" in p for p in prompts)

    def test_prior_failure_record_appears_in_coder_prompt(self, emb_pipeline, tmp_path):
        emb = build_in_memory_emb()
        emb.put(
            MemoryRecord(
                polarity="failure",
                context=Context(
                    task_text="intro scene for testing",
                    source_paper="arxiv:prior",
                    source_section="Background",
                ),
                body=FailureBody(
                    trigger_pattern="PRIOR_TRIGGER_MARKER",
                    root_cause="cause",
                    fix_recipe="fix",
                    code_anti_example="PRIOR_ANTI_MARKER",
                    code_good_example="PRIOR_GOOD_MARKER",
                ),
                provenance=Provenance(
                    scene_id="PriorScene",
                    validated=True,
                    before_score=2.0,
                    after_score=4.0,
                    extraction_source="visual_reflection",
                ),
            )
        )
        out = _run(
            {
                "run_id": "emb-prefilled-f",
                "input_kind": "pdf",
                "pdf_path": str(tmp_path / "f.pdf"),
                "attempts": [],
                "rendered_videos": [],
                "skipped_scenes": [],
                "current_scene_idx": 0,
                "iter_count": 0,
                "max_retries": 2,
                "quality": "l",
                "skip_render": False,
                "vlm_enabled": False,
                "vlm_revision_count": 0,
                "max_visual_revisions": 2,
                "visual_revision_decisions": [],
                "emb_enabled": True,
                "emb_theta_high": 0.0,
                "emb_instance": emb,
            }
        )
        assert out["final_video_path"].endswith("output.mp4")
        prompts = emb_pipeline["llm"]._coder_invocations
        assert any("Known Pitfalls" in p for p in prompts)
        assert any("PRIOR_TRIGGER_MARKER" in p for p in prompts)
        assert any("PRIOR_ANTI_MARKER" in p for p in prompts)
        assert any("PRIOR_GOOD_MARKER" in p for p in prompts)


# --------------------------------------------------------------------------- #
# Consolidation: validated transitions become failure records
# --------------------------------------------------------------------------- #


class TestEMBConsolidation:
    """Drive a run with a real visual revision so consolidation has something
    to write. We use the scripted VLM review pattern from test_graph_mvp2_vlm
    (revise → pass) plus a visual_revise stub that returns distinct code."""

    def test_visual_revision_produces_failure_record(self, emb_pipeline, monkeypatch, tmp_path):
        review_calls = {"n": 0}

        def review_alternating(scene, montage, *, summary=None, scene_idx=0):
            review_calls["n"] += 1
            if review_calls["n"] == 1:
                return {
                    "scene_id": "Scene1",
                    "decision": "revise",
                    "scores": {
                        "paper_alignment": 2, "visual_clarity": 2,
                        "readability": 2, "layout_balance": 2,
                        "visual_focus": 2, "animation_perceived": 2,
                    },
                    "revision_instruction": "fix layout",
                    "raw_response": "",
                }
            return {
                "scene_id": "Scene1",
                "decision": "pass",
                "scores": {
                    "paper_alignment": 5, "visual_clarity": 5,
                    "readability": 5, "layout_balance": 5,
                    "visual_focus": 5, "animation_perceived": 5,
                },
                "revision_instruction": "",
                "raw_response": "",
            }

        monkeypatch.setattr("paper2manim.graphs.scene_graph.review_scene", review_alternating)
        monkeypatch.setattr(
            "paper2manim.graphs.scene_graph.revise_code",
            lambda *a, **kw: (
                "from manim import *\nclass Scene1(Scene):\n"
                "    def construct(self):\n        self.wait(0.2)  # revised\n"
            ),
        )

        emb = build_in_memory_emb()
        out = _run(
            {
                "run_id": "emb-consolidate-vlm",
                "input_kind": "pdf",
                "pdf_path": str(tmp_path / "f.pdf"),
                "attempts": [],
                "rendered_videos": [],
                "skipped_scenes": [],
                "current_scene_idx": 0,
                "iter_count": 0,
                "max_retries": 2,
                "quality": "l",
                "skip_render": False,
                "vlm_enabled": True,
                "vlm_revision_count": 0,
                "max_visual_revisions": 2,
                "visual_revision_decisions": [],
                "emb_enabled": True,
                "emb_theta_high": 4.0,
                "emb_use_llm_distillers": False,
                "emb_instance": emb,
            }
        )
        assert out["final_video_path"].endswith("output.mp4")
        # Final high-score scene → 1 success record
        assert emb.count(polarity="success") == 1
        # Validated visual transition (low → high) → 1 failure record
        assert emb.count(polarity="failure") == 1
        fail_rec = emb.all(polarity="failure")[0]
        assert fail_rec.provenance.extraction_source == "visual_reflection"
        assert fail_rec.provenance.before_score == 2.0
        assert fail_rec.provenance.after_score == 5.0


# --------------------------------------------------------------------------- #
# emb_use_llm_distillers wiring
# --------------------------------------------------------------------------- #


class TestEMBLLMDistillers:
    """Verify the LLM distillers are invoked when --emb-llm-distill is set."""

    def test_llm_distillers_called_when_flag_on(self, emb_pipeline, monkeypatch, tmp_path):
        rationale_calls: list[str] = []
        lesson_calls: list[str] = []

        def fake_rationale_writer(scored, desc):
            rationale_calls.append(scored.name)
            return "LLM_RATIONALE_OUTPUT"

        # FailureBody for the stub
        from paper2manim.emb.schema import FailureBody as _FB

        def fake_lesson_distiller(transition, desc):
            lesson_calls.append(transition.scene)
            return _FB(
                trigger_pattern="t",
                root_cause="r",
                fix_recipe="f",
            )

        monkeypatch.setattr(
            "paper2manim.agents.rationale_writer.write_rationale_llm",
            fake_rationale_writer,
        )
        monkeypatch.setattr(
            "paper2manim.agents.lesson_distiller.distill_lesson_llm",
            fake_lesson_distiller,
        )

        emb = build_in_memory_emb()
        _run(
            {
                "run_id": "emb-llm-on",
                "input_kind": "pdf",
                "pdf_path": str(tmp_path / "f.pdf"),
                "attempts": [],
                "rendered_videos": [],
                "skipped_scenes": [],
                "current_scene_idx": 0,
                "iter_count": 0,
                "max_retries": 2,
                "quality": "l",
                "skip_render": False,
                "vlm_enabled": False,
                "vlm_revision_count": 0,
                "max_visual_revisions": 2,
                "visual_revision_decisions": [],
                "emb_enabled": True,
                "emb_theta_high": 0.0,  # accept the no-VLM scene as success
                "emb_use_llm_distillers": True,
                "emb_instance": emb,
            }
        )
        assert rationale_calls == ["Scene1"]
        # No visual / text transitions in this run → no lesson distill calls.
        # We only check the rationale path because lesson_distiller fires only
        # on validated transitions, which require multiple renders.
        assert emb.count(polarity="success") == 1
        rec = emb.all(polarity="success")[0]
        assert rec.body.rationale == "LLM_RATIONALE_OUTPUT"
