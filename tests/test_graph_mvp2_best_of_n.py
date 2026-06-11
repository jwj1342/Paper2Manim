"""B5 (v2): visual best-of-N on the fan-out topology.

The visual-revise loop sometimes produces a strictly worse v(N+1) (see the
v1=85 → v2=60 regression case in ``docs/vlm_experiment.md``). Pre-B5 the
parent-side ``run_scene_node`` blindly took the most recent attempt's
``video_path`` for ``rendered_videos`` — meaning the worse video shipped to
the user *and* the worse code wound up in EMB.success (because
``emb/distill.py:find_scored_scenes`` already picks the highest-VLM-avg
``v_rev`` from disk).

Post-B5 the per-scene subgraph appends a ``rendition`` per VLM review
(``{scene, v_rev, video_path, code, avg_score, decision}``), and
``run_scene_node`` calls ``_pick_best_rendition`` to ship the highest-scoring
version. This re-aligns the final video with the EMB record's chosen v_rev.

Tests:
1. ``_pick_best_rendition`` unit — happy / tie-break / ineligible inputs.
2. End-to-end via mocked VLM scoring 70 → 85 → 60 — assert ``rendered_videos``
   is the v1 (85-scored) mp4, not v2 (60).
3. End-to-end with VLM off (no renditions emitted) — falls back to the
   pre-B5 "last attempt" path so legacy callers are unaffected.
4. Trace audit — when best != last, an ``advance`` event records both.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper2manim.schemas import StoryboardModel, SummaryModel

# --------------------------------------------------------------------------- #
# _pick_best_rendition unit
# --------------------------------------------------------------------------- #


class TestPickBestRendition:
    def _r(self, *, v_rev, score, video="mp4", scene="S1"):
        return {
            "scene": scene, "v_rev": v_rev,
            "video_path": video, "code": "# c",
            "avg_score": score, "decision": "pass",
        }

    def test_highest_score_wins(self):
        from paper2manim.graphs.mvp2 import _pick_best_rendition

        rs = [
            self._r(v_rev=0, score=70.0, video="v0.mp4"),
            self._r(v_rev=1, score=85.0, video="v1.mp4"),
            self._r(v_rev=2, score=60.0, video="v2.mp4"),
        ]
        chosen = _pick_best_rendition(rs, "S1")
        assert chosen is not None
        assert chosen["v_rev"] == 1
        assert chosen["video_path"] == "v1.mp4"

    def test_filters_by_scene_name(self):
        from paper2manim.graphs.mvp2 import _pick_best_rendition

        rs = [
            self._r(v_rev=0, score=99.0, video="other.mp4", scene="OTHER"),
            self._r(v_rev=0, score=70.0, video="s1.mp4", scene="S1"),
        ]
        chosen = _pick_best_rendition(rs, "S1")
        assert chosen is not None
        assert chosen["video_path"] == "s1.mp4"

    def test_skips_ineligible_no_score(self):
        from paper2manim.graphs.mvp2 import _pick_best_rendition

        rs = [
            self._r(v_rev=0, score=None, video="v0.mp4"),  # auto-pass / no review
            self._r(v_rev=1, score=70.0, video="v1.mp4"),
        ]
        chosen = _pick_best_rendition(rs, "S1")
        assert chosen is not None and chosen["v_rev"] == 1

    def test_skips_ineligible_no_video(self):
        from paper2manim.graphs.mvp2 import _pick_best_rendition

        rs = [
            self._r(v_rev=0, score=85.0, video=None),  # frame_sampler failed
            self._r(v_rev=1, score=70.0, video="v1.mp4"),
        ]
        chosen = _pick_best_rendition(rs, "S1")
        assert chosen is not None and chosen["v_rev"] == 1

    def test_returns_none_when_all_ineligible(self):
        """All scores None → no comparable basis. Caller falls back to last
        attempt; must not crash, must not arbitrarily pick one."""
        from paper2manim.graphs.mvp2 import _pick_best_rendition

        rs = [
            self._r(v_rev=0, score=None, video="v0.mp4"),
            self._r(v_rev=1, score=None, video="v1.mp4"),
        ]
        assert _pick_best_rendition(rs, "S1") is None

    def test_returns_none_for_empty(self):
        from paper2manim.graphs.mvp2 import _pick_best_rendition

        assert _pick_best_rendition([], "S1") is None

    def test_tie_breaks_to_earlier_v_rev(self):
        """Same score, earlier v_rev wins — don't reward a tie that includes
        a regression-then-recovery cycle (v0=85, v1=80, v2=85 → still v0)."""
        from paper2manim.graphs.mvp2 import _pick_best_rendition

        rs = [
            self._r(v_rev=0, score=85.0, video="v0.mp4"),
            self._r(v_rev=1, score=80.0, video="v1.mp4"),
            self._r(v_rev=2, score=85.0, video="v2.mp4"),
        ]
        chosen = _pick_best_rendition(rs, "S1")
        assert chosen is not None and chosen["v_rev"] == 0


# --------------------------------------------------------------------------- #
# End-to-end: mock VLM 70 → 85 → 60, assert v1 ships
# --------------------------------------------------------------------------- #


def _stub_llm(scene_name: str = "Scene1") -> MagicMock:
    summary = SummaryModel(
        title="T",
        key_contributions=["c"],
        key_formulas=[],
        main_concepts=["m"],
        scene_suggestions=["s"],
    )
    sb = StoryboardModel(
        title="T",
        scenes=[{"name": scene_name, "description": "d", "duration_hint": 4.0}],
    )
    summarizer_struct = MagicMock()
    summarizer_struct.invoke.return_value = summary
    sb_struct = MagicMock()
    sb_struct.invoke.return_value = sb
    coder_msg = MagicMock()
    coder_msg.content = (
        "```python\nfrom manim import *\nclass " + scene_name + "(Scene):\n"
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
def best_of_n_pipeline(monkeypatch, tmp_path):
    """Stub graph end-to-end: parser, coder, reviewer LLMs, render, sampler,
    concat. Renders return a deterministic per-(iter, v_rev) mp4 path so the
    test can match it back to the chosen rendition."""
    llm = _stub_llm()
    for mod in (
        "paper2manim.agents.storyboarder",
        "paper2manim.agents.coder",
        "paper2manim.agents.summarizer",
        "paper2manim.agents.reviewer",
    ):
        monkeypatch.setattr(f"{mod}.get_llm", lambda *a, **kw: llm)

    from paper2manim.parsers import ParsedInput

    monkeypatch.setattr(
        "paper2manim.graphs.mvp2.parse_local_pdf",
        lambda p: ParsedInput(text="# T", fmt="markdown", source="pdf:fake"),
    )

    # Renders: produce a unique mp4 per workdir so v0/v1/v2 are distinguishable.
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
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"\x89PNG\r\n\x1a\n")
        return out_png

    monkeypatch.setattr("paper2manim.graphs.scene_graph.sample_frames_montage", fake_sample)

    def fake_assemble_voiceover(**kwargs):
        from paper2manim.voiceover.assembly import VoiceoverAssemblyResult

        run_id = kwargs.get("run_id", "test")
        out = Path("runs") / run_id / "final" / "output.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00")
        return VoiceoverAssemblyResult(
            final_video_path=str(out),
            silent_video_path=str(out),
        )

    monkeypatch.setattr("paper2manim.graphs.mvp2.assemble_voiceover", fake_assemble_voiceover)

    def fake_revise(*a, **kw):
        return (
            "from manim import *\nclass Scene1(Scene):\n"
            "    def construct(self):\n        self.wait(0.2)\n"
        )

    monkeypatch.setattr("paper2manim.graphs.scene_graph.revise_code", fake_revise)
    return {"llm": llm, "tmp_path": tmp_path}


def _run_graph(initial_state):
    from paper2manim.graphs.mvp2 import build_mvp2_graph

    g = build_mvp2_graph()
    return g.invoke(initial_state, config={"recursion_limit": 80})


def _initial_state(run_id: str, tmp_path: Path) -> dict:
    return {
        "run_id": run_id,
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
    }


def test_best_of_n_ships_v1_not_regressed_v2(best_of_n_pipeline, monkeypatch, tmp_path):
    """The decisive regression test:

    - v0 review:  score=70, decision=revise → v1 produced
    - v1 review:  score=85, decision=revise → v2 produced (worse than v1)
    - v2 review:  score=60, decision=pass   → loop ends

    Pre-B5: ``rendered_videos[0]`` is v2's mp4 (the worst!). Post-B5: v1's.
    """
    scripted = [
        {"average_score": 70.0, "decision": "revise"},
        {"average_score": 85.0, "decision": "revise"},
        {"average_score": 60.0, "decision": "pass"},
    ]
    call_idx = {"i": 0}

    def scripted_review(scene, montage, *a, **kw):
        i = call_idx["i"]
        call_idx["i"] += 1
        rec = scripted[min(i, len(scripted) - 1)]
        return {
            "scene_id": scene.get("name", "Scene1"),
            "decision": rec["decision"],
            "scores": {
                "logic_flow": int(rec["average_score"]),
                "layout_occlusion": int(rec["average_score"]),
                "accuracy": int(rec["average_score"]),
            },
            "average_score": rec["average_score"],
            "revision_instruction": "more detail" if rec["decision"] == "revise" else "",
            "raw_response": "",
        }

    monkeypatch.setattr(
        "paper2manim.graphs.scene_graph.review_scene", scripted_review
    )

    out = _run_graph(_initial_state("best-of-n-regress", tmp_path))

    assert call_idx["i"] == 3, f"expected 3 VLM reviews, got {call_idx['i']}"
    assert len(out["rendered_videos"]) == 1
    chosen_video = out["rendered_videos"][0]

    # The chosen mp4 must be v1's workdir, not v2's. workdir naming is
    # ``work_<scene>_<iter:02d>[_v<vrev>]``; v1 lives in ``..._00_v1``.
    assert "_00_v1/" in chosen_video, (
        f"expected v1 workdir in path, got {chosen_video!r}"
    )
    assert "_00_v2/" not in chosen_video


def test_legacy_no_vlm_falls_back_to_last_attempt(best_of_n_pipeline, monkeypatch, tmp_path):
    """When VLM is off, no rendition is emitted — must keep pre-B5 behavior
    so MVP-2-style runs (no VLM judge) are unaffected."""
    state = _initial_state("no-vlm", tmp_path)
    state["vlm_enabled"] = False

    out = _run_graph(state)
    assert len(out["rendered_videos"]) == 1
    # iter=0, v_rev=0 → workdir ends in ``_00`` (no v_ suffix)
    chosen_video = out["rendered_videos"][0]
    assert chosen_video.endswith("_00/scene.mp4"), (
        f"expected initial-pass workdir, got {chosen_video!r}"
    )


def test_advance_trace_records_skipped_last_when_best_differs(
    best_of_n_pipeline, monkeypatch, tmp_path
):
    """Audit: when the picker overrides the last attempt, an ``advance``
    trace event must record both choices so downstream audits (and the EMB
    consolidate path) can see when best-of-N intervened."""
    scripted = [
        {"average_score": 70.0, "decision": "revise"},
        {"average_score": 85.0, "decision": "revise"},
        {"average_score": 60.0, "decision": "pass"},
    ]
    i = {"n": 0}

    def scripted_review(scene, montage, *a, **kw):
        n = i["n"]
        i["n"] += 1
        rec = scripted[min(n, len(scripted) - 1)]
        return {
            "scene_id": scene.get("name", "Scene1"),
            "decision": rec["decision"],
            "scores": {"logic_flow": int(rec["average_score"])},
            "average_score": rec["average_score"],
            "revision_instruction": "x" if rec["decision"] == "revise" else "",
            "raw_response": "",
        }

    monkeypatch.setattr(
        "paper2manim.graphs.scene_graph.review_scene", scripted_review
    )

    state = _initial_state("trace-audit", tmp_path)
    _run_graph(state)

    # Read the trace and look for the ``advance`` event.
    from paper2manim.artifacts import run_dir

    trace_path = run_dir("trace-audit") / "trace.jsonl"
    assert trace_path.exists(), trace_path
    import json
    advance_events = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("node") == "advance":
            advance_events.append(ev)
    assert len(advance_events) == 1, advance_events
    ev = advance_events[0]
    assert ev["scene"] == "Scene1"
    assert ev["chose"]["v_rev"] == 1
    assert ev["chose"]["avg_score"] == 85.0
    assert ev["skipped_last"]["v_rev"] == 2
    assert ev["chose"]["video_path"] != ev["skipped_last"]["video_path"]


def test_advance_trace_quiet_when_last_is_best(
    best_of_n_pipeline, monkeypatch, tmp_path
):
    """If the latest attempt IS the highest-scoring (the common case), no
    ``advance`` event needs to fire — keep the trace clean."""

    def review_pass(scene, montage, *a, **kw):
        return {
            "scene_id": scene.get("name", "Scene1"),
            "decision": "pass",
            "scores": {"logic_flow": 90},
            "average_score": 90.0,
            "revision_instruction": "",
            "raw_response": "",
        }

    monkeypatch.setattr(
        "paper2manim.graphs.scene_graph.review_scene", review_pass
    )
    _run_graph(_initial_state("trace-quiet", tmp_path))

    from paper2manim.artifacts import run_dir

    trace_path = run_dir("trace-quiet") / "trace.jsonl"
    import json
    nodes = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        nodes.append(json.loads(line).get("node"))
    assert "advance" not in nodes, "no override happened — advance event should be silent"
