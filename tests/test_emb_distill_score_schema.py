"""B1 regression tests: EMB distill is locked to the proposal §4.2 3-dim 0-100 schema.

Pre-fix bug: ``_SCORE_KEYS`` carried the old 6-dim 1-5 names while
``vlm_scene_reviewer`` already emitted the new 3-dim 0-100 keys, so
``_avg_score`` returned ``0.0`` for every real run, and consolidate_run wrote
zero records into the EMB on production traces.

These tests pin down:

1. New 3-dim 0-100 trace produces a non-zero average and lands records.
2. End-to-end ``consolidate_run`` writes ≥1 success + ≥1 failure record on a
   minimal trace exercising both polarities.
3. Legacy 6-dim 1-5 trace (unmigrated old runs) returns ``0.0`` from
   ``_avg_score`` — this is intentional: the old keys must NOT silently
   contribute to the new average.
"""

from __future__ import annotations

import json
import logging

import pytest

from paper2manim.artifacts import run_dir
from paper2manim.emb.distill import (
    _SCORE_KEYS,
    _avg_score,
    consolidate_run,
    find_visual_transitions,
    parse_trace,
)
from paper2manim.emb.manager import build_in_memory_emb
from paper2manim.emb.schema import (
    Context,
    MemoryRecord,
    Provenance,
    SuccessBody,
)


# --------------------------------------------------------------------------- #
# Helpers (mirror tests/test_emb_phase2.py for cohesion)
# --------------------------------------------------------------------------- #


def _write_trace(run_id: str, events: list[dict]) -> None:
    p = run_dir(run_id) / "trace.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _write_attempt_code(run_id: str, scene: str, iter_idx: int, v_rev: int, code: str) -> None:
    name = scene if v_rev == 0 else f"{scene}_v{v_rev}"
    p = run_dir(run_id) / "attempts" / f"{iter_idx:02d}_{name}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code, encoding="utf-8")


def _fake_state(*scene_names: str) -> dict:
    return {
        "storyboard": {
            "title": "T",
            "scenes": [
                {"name": n, "description": f"desc {n}", "duration_hint": 5.0}
                for n in scene_names
            ],
        }
    }


# --------------------------------------------------------------------------- #
# 1. Score keys are the canonical proposal §4.2 schema
# --------------------------------------------------------------------------- #


def test_score_keys_are_three_dim_canonical():
    assert _SCORE_KEYS == ("logic_flow", "layout_occlusion", "accuracy"), (
        "EMB distill must mirror agents.vlm_scene_reviewer._SCORE_KEYS to keep "
        "_avg_score in sync with the VLM output schema."
    )


# --------------------------------------------------------------------------- #
# 2. _avg_score under the new schema
# --------------------------------------------------------------------------- #


def test_avg_score_three_dim_full():
    """All three dims present: simple mean."""
    s = {"logic_flow": 90, "layout_occlusion": 80, "accuracy": 70}
    assert _avg_score(s) == pytest.approx(80.0)


def test_avg_score_three_dim_partial_missing_dim_treated_as_skip():
    """One dim missing (None): mean over the present two, not zero-padded."""
    s = {"logic_flow": 90, "layout_occlusion": None, "accuracy": 70}
    assert _avg_score(s) == pytest.approx(80.0)


def test_avg_score_legacy_six_dim_returns_zero():
    """Pre-PR-#15 6-dim 1-5 payload: must return 0.0, not silently average.

    This is the regression bullseye: prior bug averaged old keys against the
    new ``_SCORE_KEYS`` filter and got 0.0 by accident; we now want 0.0 *by
    construction* so that an unmigrated old store doesn't pollute the gate.
    """
    legacy = {
        k: 5 for k in (
            "paper_alignment", "visual_clarity", "readability",
            "layout_balance", "visual_focus", "animation_perceived",
        )
    }
    assert _avg_score(legacy) == 0.0


def test_avg_score_empty_or_none():
    assert _avg_score({}) == 0.0
    assert _avg_score(None) == 0.0


def test_avg_score_clamps_garbage_values():
    """Non-numeric values should be skipped, not crash."""
    s = {"logic_flow": "high", "layout_occlusion": 80, "accuracy": None}
    assert _avg_score(s) == pytest.approx(80.0)


# --------------------------------------------------------------------------- #
# 3. find_visual_transitions on the new schema
# --------------------------------------------------------------------------- #


def test_visual_transition_writes_with_new_schema():
    run_id = "rid_b1_vt_new"
    _write_trace(
        run_id,
        [
            {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
            {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
             "scores": {"logic_flow": 30, "layout_occlusion": 30, "accuracy": 30},
             "revision_instruction": "fix layout"},
            {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
             "scores": {"logic_flow": 90, "layout_occlusion": 90, "accuracy": 90}},
        ],
    )
    _write_attempt_code(run_id, "S1", 0, 0, "v0")
    _write_attempt_code(run_id, "S1", 0, 1, "v1")
    scenes = parse_trace(run_id)
    out = find_visual_transitions(run_id, scenes.values())  # default min_margin=5.0
    assert len(out) == 1
    t = out[0]
    assert t.before_score == pytest.approx(30.0)
    assert t.after_score == pytest.approx(90.0)


def test_visual_transition_legacy_six_dim_silently_skipped():
    """A legacy 6-dim trace produces avg 0.0 → 0 transitions (the expected
    failure mode for unmigrated data, with the warning surfaced via
    EpisodicMemoryBank's drift scan, not here)."""
    legacy = {k: 5 for k in (
        "paper_alignment", "visual_clarity", "readability",
        "layout_balance", "visual_focus", "animation_perceived",
    )}
    run_id = "rid_b1_vt_legacy"
    _write_trace(
        run_id,
        [
            {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
            {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
             "scores": legacy, "revision_instruction": "fix layout"},
            {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
             "scores": legacy},
        ],
    )
    _write_attempt_code(run_id, "S1", 0, 0, "v0")
    _write_attempt_code(run_id, "S1", 0, 1, "v1")
    scenes = parse_trace(run_id)
    out = find_visual_transitions(run_id, scenes.values())
    assert out == []


# --------------------------------------------------------------------------- #
# 4. End-to-end consolidate_run on the new schema
# --------------------------------------------------------------------------- #


def test_consolidate_run_writes_success_and_failure_on_new_schema():
    """Full pipeline: a single scene that revises 30 → 90 produces both a
    success record (avg 90 ≥ default theta_high=85) AND a failure record
    (margin 60 ≥ default failure_min_margin=5)."""
    run_id = "rid_b1_consolidate"
    _write_trace(
        run_id,
        [
            {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
            {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
             "scores": {"logic_flow": 30, "layout_occlusion": 30, "accuracy": 30},
             "revision_instruction": "tighten layout"},
            {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
             "scores": {"logic_flow": 90, "layout_occlusion": 90, "accuracy": 90}},
        ],
    )
    _write_attempt_code(run_id, "S1", 0, 0, "v0 code")
    _write_attempt_code(run_id, "S1", 0, 1, "v1 code")

    emb = build_in_memory_emb()
    report = consolidate_run(
        run_id, emb,
        state=_fake_state("S1"),
        source_paper="arxiv:test",
        source_section="Background",
    )
    # The bug this PR fixes used to write 0 + 0 here.
    assert report.n_success_written == 1, "0-100 schema should land a success record"
    assert report.n_failure_written == 1, "0-100 schema should land a failure record"
    assert emb.count(polarity="success") == 1
    assert emb.count(polarity="failure") == 1


def test_consolidate_run_writes_nothing_on_legacy_schema():
    """Symmetric: a legacy 6-dim trace consolidates to zero records; this
    documents the *current* behavior on unmigrated data so future schema
    migrations have a baseline to flip."""
    legacy = {k: 5 for k in (
        "paper_alignment", "visual_clarity", "readability",
        "layout_balance", "visual_focus", "animation_perceived",
    )}
    run_id = "rid_b1_consolidate_legacy"
    _write_trace(
        run_id,
        [
            {"node": "render", "scene": "S1", "iter": 0, "v_rev": 0, "status": "success"},
            {"node": "vlm_review", "scene": "S1", "v_rev": 0, "decision": "revise",
             "scores": legacy},
            {"node": "vlm_review", "scene": "S1", "v_rev": 1, "decision": "pass",
             "scores": legacy},
        ],
    )
    _write_attempt_code(run_id, "S1", 0, 0, "v0")
    _write_attempt_code(run_id, "S1", 0, 1, "v1")
    emb = build_in_memory_emb()
    report = consolidate_run(run_id, emb, state=_fake_state("S1"))
    assert report.n_success_written == 0
    assert report.n_failure_written == 0


# --------------------------------------------------------------------------- #
# 5. EpisodicMemoryBank drift warning
# --------------------------------------------------------------------------- #


def test_emb_warns_on_legacy_score_schema_drift(caplog):
    """A success record with vlm_score in the old 1-5 range triggers a single
    WARNING log so unmigrated stores are observable on next process startup."""
    emb = build_in_memory_emb()
    legacy_rec = MemoryRecord(
        polarity="success",
        context=Context(task_text="legacy scene", task_embedding=[]),
        body=SuccessBody(rationale="legacy", code_full="pass"),
        provenance=Provenance(
            run_id="legacy", scene_id="S1",
            extraction_source="high_score_scene",
            validated=True, vlm_score=4.5,  # old 1-5 schema
        ),
    )
    emb.put(legacy_rec)

    # Re-instantiate to trigger the constructor's drift scan.
    with caplog.at_level(logging.WARNING, logger="paper2manim.emb.manager"):
        # Reuse the same store so the scan sees the legacy record.
        from paper2manim.emb.manager import EpisodicMemoryBank
        EpisodicMemoryBank(
            store=emb._store,
            embedder=emb._embedder,
        )
    drift_msgs = [r.message for r in caplog.records if "score schema drift" in r.message]
    assert drift_msgs, "expected a 'score schema drift' WARNING"


def test_emb_no_warning_on_clean_zero_to_one_hundred_store(caplog):
    """A success record at vlm_score=88 (new schema) must NOT trigger the
    drift warning; we only flag values in the (0, 5) suspicious range."""
    emb = build_in_memory_emb()
    new_rec = MemoryRecord(
        polarity="success",
        context=Context(task_text="modern scene", task_embedding=[]),
        body=SuccessBody(rationale="modern", code_full="pass"),
        provenance=Provenance(
            run_id="modern", scene_id="S1",
            extraction_source="high_score_scene",
            validated=True, vlm_score=88.0,
        ),
    )
    emb.put(new_rec)
    with caplog.at_level(logging.WARNING, logger="paper2manim.emb.manager"):
        from paper2manim.emb.manager import EpisodicMemoryBank
        EpisodicMemoryBank(store=emb._store, embedder=emb._embedder)
    drift_msgs = [r.message for r in caplog.records if "score schema drift" in r.message]
    assert not drift_msgs
