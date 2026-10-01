"""Regressions for the paper's generation and memory rules."""

import hashlib
from unittest.mock import MagicMock

import pytest

from paper2manim.agents.reviewer import reviewer_node
from paper2manim.agents.vlm_scene_reviewer import parse_vlm_response
from paper2manim.artifacts import append_trace, save_attempt_code, save_attempt_result
from paper2manim.emb import Context, MemoryRecord, Provenance, SuccessBody
from paper2manim.emb.distill import distill_success_records, find_text_transitions, parse_trace
from paper2manim.emb.manager import build_in_memory_emb
from paper2manim.graphs.generation import run_scene_node
from paper2manim.graphs.scene_graph import emb_retrieve_node
from paper2manim.quality.manim_static_checker import validate_manim_code


def test_repeated_category_stops_without_another_model_call(monkeypatch):
    llm = MagicMock()
    monkeypatch.setattr("paper2manim.agents.reviewer.get_llm", llm)
    attempts = [
        {"render_result": {"status": "error", "category": "latex", "error_message": message}}
        for message in ("undefined control", "missing package")
    ]
    out = reviewer_node({"attempts": attempts, "iter_count": 1, "max_retries": 2})
    assert out["error_feedback"]["decision"] == "give_up"
    llm.assert_not_called()


def test_high_scores_override_fail_verdict():
    review = parse_vlm_response(
        '{"decision":"fail","scores":{"logic_flow":90,"layout_occlusion":90,"accuracy":90}}', "S"
    )
    assert review["decision"] == "pass"


def test_readonly_retrieval_does_not_change_hit_counters():
    emb = build_in_memory_emb()
    rec_id = emb.put(
        MemoryRecord(
            polarity="success",
            context=Context(task_text="Section", scene_role="METHOD"),
            body=SuccessBody(rationale="r", code_full="code"),
            provenance=Provenance(run_id="r", scene_id="S"),
        )
    )
    out = emb_retrieve_node(
        {
            "emb_enabled": True,
            "emb_instance": emb,
            "emb_readonly": True,
            "task_text": "Section",
            "scene_role": "METHOD",
            "scene": {"name": "S"},
            "run_id": "readonly",
        }
    )
    assert len(out["retrieved_success"]) == 1
    assert emb.get(rec_id).provenance.hit_count == 0


def test_best_candidate_survives_later_render_failure(monkeypatch):
    graph = MagicMock()
    graph.invoke.return_value = {
        "attempts": [
            {"render_result": {"status": "success", "video_path": "best.mp4"}},
            {"render_result": {"status": "error"}},
        ],
        "scene_renditions": [{"scene": "S", "v_rev": 0, "avg_score": 86, "video_path": "best.mp4"}],
        "text_retry_count": 1,
        "vlm_revision_count": 1,
    }
    monkeypatch.setattr("paper2manim.graphs.generation.get_compiled_scene_graph", lambda: graph)
    out = run_scene_node({"run_id": "best", "scene": {"name": "S"}})
    assert out["rendered_videos"] == ["best.mp4"]
    assert out["scene_reports"][0]["reflection_rounds"] == 2
    assert not out.get("skipped_scenes")


def test_text_repair_of_visual_revision_uses_its_own_sidecars():
    run_id = "visual-crash"
    save_attempt_code(run_id, "S_v1", 0, "bad")
    save_attempt_code(run_id, "S_v1", 1, "fixed")
    save_attempt_result(run_id, "S_v1", 0, {"error_message": "bad call"})
    append_trace(
        run_id,
        "render",
        {"scene": "S", "iter": 0, "v_rev": 1, "status": "error", "category": "python"},
    )
    append_trace(run_id, "render", {"scene": "S", "iter": 1, "v_rev": 1, "status": "success"})
    transitions = find_text_transitions(run_id, parse_trace(run_id).values())
    assert len(transitions) == 1
    assert transitions[0].before_code == "bad"
    assert transitions[0].after_code == "fixed"
    assert transitions[0].error_message == "bad call"


def test_failed_distillation_skips_record_and_preserves_source_context(tmp_path, monkeypatch):
    from paper2manim.emb.distill import ScoredScene

    montage = tmp_path / "frames.png"
    montage.write_bytes(b"selected frames")
    scored = ScoredScene("S", 1, 92, "selected code", montage, None, True)
    monkeypatch.setattr("paper2manim.emb.distill.find_scored_scenes", lambda *args: [scored])

    def broken(*args):
        raise RuntimeError("model unavailable")

    assert distill_success_records("r", rationale_writer=broken) == []
    records = distill_success_records(
        "r",
        state={"task_text": "Source section", "scene_role": "METHOD"},
        rationale_writer=lambda *args: "r" * 500,
    )
    assert records[0].context.task_text == "Source section"
    assert records[0].context.scene_role == "METHOD"
    assert len(records[0].body.rationale) == 400
    assert records[0].body.frame_hash == hashlib.sha256(montage.read_bytes()).hexdigest()


def test_requested_arxiv_section_never_falls_back_to_full_pdf(monkeypatch):
    from paper2manim.parsers import parse_arxiv
    from paper2manim.parsers.arxiv_source import SourceUnavailable

    def unavailable(*args, **kwargs):
        raise SourceUnavailable("No source")

    download = MagicMock()
    monkeypatch.setattr("paper2manim.parsers.arxiv_source.fetch_arxiv_source", unavailable)
    monkeypatch.setattr("paper2manim.parsers.arxiv_source.download_pdf", download)
    with pytest.raises(SourceUnavailable, match="--input"):
        parse_arxiv("1706.03762", section="Method")
    download.assert_not_called()


@pytest.mark.parametrize(
    "code",
    [
        "from manim import *\nclass Wrong(Scene):\n def construct(self):\n  self.play(Create(Circle()))\n  self.play(FadeOut(Circle()))\n",
        "from manim import *\nclass Right(Scene):\n def construct(self):\n  # self.play(Create(Circle()))\n  # self.play(FadeOut(Circle()))\n  self.wait(1)\n",
    ],
)
def test_renderer_rejects_wrong_name_and_comment_only_animations(code):
    with pytest.raises(ValueError):
        validate_manim_code(code, scene_name="Right")


def test_scene_spec_preserves_explicit_scientific_content():
    from paper2manim.agents.vlm_scene_reviewer import _build_scene_spec_payload

    spec = _build_scene_spec_payload(
        {
            "name": "S",
            "description": "Animation beats.",
            "paper_claim": "Source claim",
            "paper_evidence": "Source evidence",
            "final_takeaway": "Source takeaway",
            "paper_role": "METHOD",
        },
        summary=None,
        scene_idx=0,
    )
    assert spec["paper_claim"] == "Source claim"
    assert spec["paper_evidence"] == ["Source evidence"]
    assert spec["final_takeaway"] == "Source takeaway"
    assert spec["paper_role"] == "METHOD"


def test_duplicate_scene_names_are_rejected():
    from pydantic import ValidationError

    from paper2manim.schemas import StoryboardModel

    with pytest.raises(ValidationError, match="unique"):
        StoryboardModel(
            title="T",
            scenes=[{"name": "S", "description": "first"}, {"name": "S", "description": "second"}],
        )
