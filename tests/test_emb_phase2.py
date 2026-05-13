"""Phase 2 tests: trace parsing, transition detection, distillation, consolidation."""

from __future__ import annotations

import json

from paper2manim.emb.distill import (
    ConsolidationReport,
    ScoredScene,
    TextTransition,
    VisualTransition,
    consolidate_run,
    default_lesson_distiller,
    default_rationale_writer,
    distill_failure_records,
    distill_success_records,
    find_scored_scenes,
    find_text_transitions,
    find_visual_transitions,
    infer_source_metadata,
    parse_trace,
)
from paper2manim.emb.manager import build_in_memory_emb
from paper2manim.emb.schema import FailureBody

# --------------------------------------------------------------------------- #
# Helpers: synthesize a fake run on disk
# --------------------------------------------------------------------------- #


def _fake_state(*scene_names: str) -> dict:
    return {
        "storyboard": {
            "title": "Test",
            "scenes": [
                {"name": n, "description": f"description for {n}", "duration_hint": 5.0}
                for n in scene_names
            ],
        }
    }


def _write_trace(run_id: str, events: list[dict]) -> None:
    from paper2manim.artifacts import run_dir

    p = run_dir(run_id) / "trace.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _write_attempt_code(run_id: str, scene: str, iter_idx: int, v_rev: int, code: str) -> None:
    from paper2manim.artifacts import run_dir

    name = scene if v_rev == 0 else f"{scene}_v{v_rev}"
    p = run_dir(run_id) / "attempts" / f"{iter_idx:02d}_{name}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code, encoding="utf-8")


def _write_render_result(run_id: str, scene: str, iter_idx: int, result: dict) -> None:
    from paper2manim.artifacts import run_dir

    p = run_dir(run_id) / "attempts" / f"{iter_idx:02d}_{scene}.render.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result), encoding="utf-8")


def _write_montage(run_id: str, scene: str, v_rev: int) -> None:
    from paper2manim.artifacts import run_dir

    p = run_dir(run_id) / "vlm_frames" / f"{scene}_v{v_rev}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG fake")


def _high_scores() -> dict:
    """3-dim 0-100 schema (proposal §4.2). Average = 90.0."""
    return {k: 90 for k in ("logic_flow", "layout_occlusion", "accuracy")}


def _low_scores() -> dict:
    """3-dim 0-100 schema. Average = 30.0."""
    return {k: 30 for k in ("logic_flow", "layout_occlusion", "accuracy")}


def _mid_scores() -> dict:
    """3-dim 0-100 schema. Average = 60.0."""
    return {k: 60 for k in ("logic_flow", "layout_occlusion", "accuracy")}


def _legacy_6dim_scores() -> dict:
    """Pre-PR-#15 schema; kept here so the regression test for schema drift
    can confirm the new ``_avg_score`` returns 0.0 on these payloads."""
    return {k: 5 for k in (
        "paper_alignment", "visual_clarity", "readability",
        "layout_balance", "visual_focus", "animation_perceived",
    )}


# --------------------------------------------------------------------------- #
# Trace parsing
# --------------------------------------------------------------------------- #


class TestParseTrace:
    def test_groups_renders_and_reviews_by_scene(self):
        run_id = "rid_a"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "pass",
                 "scores": _high_scores()},
                {"node": "render", "scene": "S2", "iter": 0, "v_rev": 0, "status": "error",
                 "category": "python"},
                {"node": "render", "scene": "S2", "iter": 1, "v_rev": 0, "status": "success"},
            ],
        )
        out = parse_trace(run_id)
        assert set(out.keys()) == {"S1", "S2"}
        assert len(out["S1"].renders) == 1
        assert len(out["S1"].vlm_reviews) == 1
        assert out["S1"].vlm_reviews[0].avg_score == 90.0
        assert len(out["S2"].renders) == 2
        assert out["S2"].vlm_reviews == []

    def test_handles_missing_trace_file(self):
        out = parse_trace("rid_does_not_exist_anywhere")
        assert out == {}

    def test_tolerates_malformed_lines(self):
        run_id = "rid_b"
        from paper2manim.artifacts import run_dir

        p = run_dir(run_id) / "trace.jsonl"
        p.write_text(
            '{"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"}\n'
            "this is not json\n"
            '{"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "pass", "scores": {}}\n',
            encoding="utf-8",
        )
        out = parse_trace(run_id)
        assert "S1" in out
        assert len(out["S1"].renders) == 1
        assert len(out["S1"].vlm_reviews) == 1


# --------------------------------------------------------------------------- #
# Transition finders
# --------------------------------------------------------------------------- #


class TestTextTransitions:
    def test_error_then_success_emits_transition(self):
        run_id = "rid_text_1"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "error",
                 "category": "python"},
                {"node": "render", "scene": "S1", "iter": 1, "v_rev": 0, "status": "success"},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "bad code")
        _write_attempt_code(run_id, "S1", 1, 0, "good code")
        _write_render_result(
            run_id, "S1", 0,
            {
                "status": "error",
                "category": "python",
                "error_message": "NameError: x",
                "traceback_tail": "File ..., line 3: x",
            },
        )
        scenes = parse_trace(run_id)
        out = find_text_transitions(run_id, scenes.values())
        assert len(out) == 1
        t = out[0]
        assert t.scene == "S1"
        assert t.before_iter == 0 and t.after_iter == 1
        assert t.before_code == "bad code"
        assert t.after_code == "good code"
        assert "NameError" in t.error_message

    def test_skips_when_code_files_missing(self):
        run_id = "rid_text_2"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "error",
                 "category": "python"},
                {"node": "render", "scene": "S1", "iter": 1, "v_rev": 0, "status": "success"},
            ],
        )
        # No attempt code files written
        scenes = parse_trace(run_id)
        assert find_text_transitions(run_id, scenes.values()) == []

    def test_no_transition_when_no_success(self):
        run_id = "rid_text_3"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "error",
                 "category": "python"},
                {"node": "render", "scene": "S1", "iter": 1, "v_rev": 0, "status": "error",
                 "category": "python"},
            ],
        )
        scenes = parse_trace(run_id)
        assert find_text_transitions(run_id, scenes.values()) == []


class TestVisualTransitions:
    def test_emits_when_score_strictly_increases(self):
        run_id = "rid_vis_1"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": _low_scores(), "revision_instruction": "move things"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
                 "scores": _high_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "v0 code")
        _write_attempt_code(run_id, "S1", 0, 1, "v1 code")
        scenes = parse_trace(run_id)
        out = find_visual_transitions(run_id, scenes.values())
        assert len(out) == 1
        t = out[0]
        assert t.before_score == 30.0
        assert t.after_score == 90.0
        assert t.before_code == "v0 code"
        assert t.after_code == "v1 code"

    def test_rejects_when_score_did_not_increase(self):
        run_id = "rid_vis_2"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": _mid_scores()},
                {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "revise",
                 "scores": _low_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "v0")
        _write_attempt_code(run_id, "S1", 0, 1, "v1")
        scenes = parse_trace(run_id)
        assert find_visual_transitions(run_id, scenes.values()) == []

    def test_respects_min_margin(self):
        # before=60.0, after=62.67 (one dim bumped 60→68 on the 0-100 schema).
        # Strict-positive but below the default 5.0 margin, so should be filtered.
        run_id = "rid_vis_margin"
        scores_lo = dict(_mid_scores())
        scores_hi = dict(scores_lo)
        scores_hi["logic_flow"] = 68  # bumps avg by 8/3 ≈ 2.67
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": scores_lo},
                {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
                 "scores": scores_hi},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "v0")
        _write_attempt_code(run_id, "S1", 0, 1, "v1")
        scenes = parse_trace(run_id)
        # Default margin 5.0: rejected.
        assert find_visual_transitions(run_id, scenes.values()) == []
        # Lower margin 1.0: accepted.
        out = find_visual_transitions(run_id, scenes.values(), min_margin=1.0)
        assert len(out) == 1

    def test_rejects_equal_score(self):
        run_id = "rid_vis_3"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": _mid_scores()},
                {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "revise",
                 "scores": _mid_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "v0")
        _write_attempt_code(run_id, "S1", 0, 1, "v1")
        scenes = parse_trace(run_id)
        assert find_visual_transitions(run_id, scenes.values()) == []

    def test_non_adjacent_reviews_ignored(self):
        run_id = "rid_vis_4"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": _low_scores()},
                # v_rev=2 jump without v_rev=1 in between
                {"node": "vlm_review", "scene": "S1", "v_rev": 2, "decision": "pass",
                 "scores": _high_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "v0")
        _write_attempt_code(run_id, "S1", 0, 2, "v2")
        scenes = parse_trace(run_id)
        assert find_visual_transitions(run_id, scenes.values()) == []


class TestScoredScenes:
    def test_picks_highest_score_version(self):
        run_id = "rid_sc_1"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": _low_scores()},
                {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
                 "scores": _high_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "v0 code")
        _write_attempt_code(run_id, "S1", 0, 1, "v1 code")
        _write_render_result(run_id, "S1", 0, {"video_path": "/tmp/x.mp4"})
        scenes = parse_trace(run_id)
        out = find_scored_scenes(run_id, scenes.values())
        assert len(out) == 1
        sc = out[0]
        assert sc.final_v_rev == 1
        assert sc.final_score == 90.0
        assert sc.final_code == "v1 code"
        assert sc.had_vlm_review is True
        assert str(sc.final_video_path).endswith("x.mp4")

    def test_no_vlm_means_score_zero(self):
        run_id = "rid_sc_2"
        _write_trace(
            run_id,
            [{"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"}],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "code")
        scenes = parse_trace(run_id)
        out = find_scored_scenes(run_id, scenes.values())
        assert len(out) == 1
        assert out[0].final_score == 0.0
        assert out[0].had_vlm_review is False


# --------------------------------------------------------------------------- #
# Default writers
# --------------------------------------------------------------------------- #


class TestDefaultWriters:
    def test_default_rationale_contains_score_and_name(self):
        sc = ScoredScene(
            name="S1", final_v_rev=2, final_score=87.50,
            final_code="from manim import *", final_montage_path=None,
            final_video_path=None, had_vlm_review=True,
        )
        out = default_rationale_writer(sc, "a description")
        assert "S1" in out
        assert "87.50" in out
        assert len(out) <= 400

    def test_default_lesson_distiller_visual_transition(self):
        vt = VisualTransition(
            scene="S1", before_v_rev=0, after_v_rev=1,
            before_code="x = 1\ny = 2", after_code="x = 1\ny = 3",
            before_score=30.0, after_score=60.0,
            revision_instruction="adjust value",
        )
        body = default_lesson_distiller(vt, "scene desc")
        assert isinstance(body, FailureBody)
        assert "low layout" in body.trigger_pattern.lower() or "low" in body.trigger_pattern.lower()
        assert body.code_anti_example != body.code_good_example

    def test_default_lesson_distiller_text_transition(self):
        tt = TextTransition(
            scene="S1", before_iter=0, after_iter=1,
            before_code="raise X", after_code="ok",
            error_category="python",
            error_message="NameError: X",
            traceback_tail="trace",
        )
        body = default_lesson_distiller(tt, "scene desc")
        assert "render fails" in body.trigger_pattern.lower() or "render" in body.trigger_pattern.lower()
        assert "NameError" in body.root_cause


# --------------------------------------------------------------------------- #
# distill_success_records / distill_failure_records
# --------------------------------------------------------------------------- #


class TestDistillSuccess:
    def test_emits_one_per_high_score_scene(self):
        run_id = "rid_ds_1"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "pass",
                 "scores": _high_scores()},
                {"node": "render", "scene": "S2", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S2", "v_rev": 0, "decision": "revise",
                 "scores": _low_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "S1 code")
        _write_attempt_code(run_id, "S2", 0, 0, "S2 code")
        out = distill_success_records(
            run_id,
            state=_fake_state("S1", "S2"),
            theta_high=85.0,
            source_paper="arxiv:test",
            source_section="Background",
        )
        assert len(out) == 1
        rec = out[0]
        assert rec.polarity == "success"
        assert rec.provenance.scene_id == "S1"
        assert rec.body.code_full == "S1 code"
        assert rec.context.source_paper == "arxiv:test"
        assert rec.context.source_section == "Background"

    def test_theta_zero_admits_no_vlm_scenes(self):
        run_id = "rid_ds_2"
        _write_trace(
            run_id,
            [{"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"}],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "code")
        out = distill_success_records(
            run_id, state=_fake_state("S1"), theta_high=0.0,
        )
        assert len(out) == 1

    def test_custom_rationale_writer_used(self):
        run_id = "rid_ds_3"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "pass",
                 "scores": _high_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "code")
        out = distill_success_records(
            run_id, state=_fake_state("S1"), theta_high=4.0,
            rationale_writer=lambda sc, desc: "CUSTOM RATIONALE",
        )
        assert len(out) == 1
        assert out[0].body.rationale == "CUSTOM RATIONALE"


class TestDistillFailure:
    def test_visual_transition_validated_only(self):
        run_id = "rid_df_1"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": _low_scores(), "revision_instruction": "fix layout"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
                 "scores": _high_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "old code")
        _write_attempt_code(run_id, "S1", 0, 1, "new code")
        out = distill_failure_records(
            run_id,
            state=_fake_state("S1"),
            source_paper="arxiv:test",
            source_section="Background",
        )
        assert len(out) == 1
        rec = out[0]
        assert rec.polarity == "failure"
        assert rec.provenance.validated is True
        assert rec.provenance.before_score == 30.0
        assert rec.provenance.after_score == 90.0
        assert rec.context.source_paper == "arxiv:test"

    def test_text_transition_emitted(self):
        run_id = "rid_df_2"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "error",
                 "category": "latex"},
                {"node": "render", "scene": "S1", "iter": 1, "v_rev": 0, "status": "success"},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "bad")
        _write_attempt_code(run_id, "S1", 1, 0, "good")
        _write_render_result(run_id, "S1", 0, {
            "status": "error", "category": "latex",
            "error_message": "tex error", "traceback_tail": "tail",
        })
        out = distill_failure_records(run_id, state=_fake_state("S1"))
        assert len(out) == 1
        assert out[0].provenance.extraction_source == "text_reflection"

    def test_no_validated_transition_means_no_failure_record(self):
        run_id = "rid_df_3"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": _mid_scores()},
                # v1 strictly worse — not validated
                {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "revise",
                 "scores": _low_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "v0")
        _write_attempt_code(run_id, "S1", 0, 1, "v1")
        out = distill_failure_records(run_id, state=_fake_state("S1"))
        assert out == []

    def test_custom_lesson_distiller_used(self):
        run_id = "rid_df_4"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": _low_scores()},
                {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
                 "scores": _high_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "v0")
        _write_attempt_code(run_id, "S1", 0, 1, "v1")

        calls = []

        def fake_distiller(transition, desc):
            calls.append((transition.scene, desc))
            return FailureBody(
                trigger_pattern="custom trigger",
                root_cause="custom cause",
                fix_recipe="custom fix",
            )

        out = distill_failure_records(
            run_id, state=_fake_state("S1"),
            lesson_distiller=fake_distiller,
        )
        assert len(out) == 1
        assert out[0].body.trigger_pattern == "custom trigger"
        assert calls and calls[0][0] == "S1"


# --------------------------------------------------------------------------- #
# consolidate_run
# --------------------------------------------------------------------------- #


class TestConsolidateRun:
    def test_writes_records_to_emb_and_returns_report(self):
        run_id = "rid_cr_1"
        _write_trace(
            run_id,
            [
                {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
                 "scores": _low_scores(), "revision_instruction": "fix layout"},
                {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
                 "scores": _high_scores()},
            ],
        )
        _write_attempt_code(run_id, "S1", 0, 0, "v0")
        _write_attempt_code(run_id, "S1", 0, 1, "v1")
        emb = build_in_memory_emb()
        report = consolidate_run(
            run_id, emb,
            state=_fake_state("S1"),
            theta_high=85.0,
            source_paper="arxiv:test",
            source_section="Background",
        )
        assert isinstance(report, ConsolidationReport)
        assert report.n_success_written == 1
        assert report.n_failure_written == 1
        assert emb.count(polarity="success") == 1
        assert emb.count(polarity="failure") == 1

    def test_empty_run_writes_nothing(self):
        run_id = "rid_cr_2"
        # no trace.jsonl at all
        emb = build_in_memory_emb()
        report = consolidate_run(run_id, emb, state=_fake_state())
        # parse_trace returns {} so nothing to distill
        assert report.n_success_written == 0
        assert report.n_failure_written == 0


# --------------------------------------------------------------------------- #
# infer_source_metadata
# --------------------------------------------------------------------------- #


class TestInferSourceMetadata:
    def test_arxiv_id(self):
        state = {"input_kind": "arxiv", "arxiv_spec": "1706.03762", "arxiv_section": "Background"}
        p, s = infer_source_metadata(state)
        assert p == "arxiv:1706.03762"
        assert s == "Background"

    def test_arxiv_url(self):
        state = {
            "input_kind": "arxiv",
            "arxiv_spec": "https://arxiv.org/abs/1706.03762v2",
            "arxiv_section": "Method",
        }
        p, s = infer_source_metadata(state)
        assert p == "arxiv:1706.03762"
        assert s == "Method"

    def test_pdf(self):
        state = {"input_kind": "pdf", "pdf_path": "/x/y/paper.pdf"}
        p, s = infer_source_metadata(state)
        assert p == "pdf:paper.pdf"
        assert s == ""

    def test_empty(self):
        assert infer_source_metadata(None) == ("", "")
        assert infer_source_metadata({}) == ("", "")
