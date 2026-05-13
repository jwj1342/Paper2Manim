"""Parallel scene fan-out: equivalence + failure isolation + concurrency wiring.

These tests drive the MVP 2.0 graph end-to-end with mocked LLM / render / VLM
so the only variable is the new ``run_scene`` Send fan-out plus the
:mod:`paper2manim.concurrency` throttles. They complement the existing
``test_graph_mvp2_vlm.py`` suite by stressing multi-scene paths and
failure-isolation guarantees that didn't exist in the serial pipeline.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper2manim import concurrency
from paper2manim.graphs.scene_graph import _reset_emb_cache
from paper2manim.schemas import StoryboardModel, SummaryModel

# --------------------------------------------------------------------------- #
# Reusable pipeline stub (mirrors test_graph_mvp2_vlm.py / test_emb_phase4.py)
# --------------------------------------------------------------------------- #


def _build_scene(name: str) -> dict:
    return {"name": name, "description": f"desc for {name}", "duration_hint": 4.0}


def _make_llm_stub(scene_names: list[str]) -> MagicMock:
    summary = SummaryModel(
        title="T",
        key_contributions=["c"],
        key_formulas=[],
        main_concepts=["m"],
        scene_suggestions=["s"],
    )
    sb = StoryboardModel(title="T", scenes=[_build_scene(n) for n in scene_names])
    summarizer_struct = MagicMock()
    summarizer_struct.invoke.return_value = summary
    sb_struct = MagicMock()
    sb_struct.invoke.return_value = sb

    llm = MagicMock()

    def structured(model_cls, **_kw):
        if model_cls.__name__ == "SummaryModel":
            return summarizer_struct
        if model_cls.__name__ == "StoryboardModel":
            return sb_struct
        return MagicMock()

    llm.with_structured_output.side_effect = structured

    coder_msg = MagicMock()
    coder_msg.content = (
        "```python\nfrom manim import *\nclass S(Scene):\n"
        "    def construct(self):\n        self.wait(0.1)\n```"
    )
    reviewer_msg = MagicMock()
    reviewer_msg.content = '{"decision":"retry","hint":""}'

    def invoke(messages):
        sys_msg = messages[0][1] if messages else ""
        if "review" in sys_msg.lower() and "build-and-review" in sys_msg.lower():
            return reviewer_msg
        return coder_msg

    llm.invoke.side_effect = invoke
    return llm


@pytest.fixture
def parallel_pipeline(monkeypatch, tmp_path):
    """Stub everything so we can fan out N scenes without touching the network or Manim."""
    _reset_emb_cache()
    concurrency.reset()

    scenes_per_test: dict[str, list[str]] = {"names": []}

    def make_llm():
        return _make_llm_stub(scenes_per_test["names"])

    def fake_get_llm(*_a, **_kw):
        return make_llm()

    for attr in (
        "paper2manim.agents.storyboarder.get_llm",
        "paper2manim.agents.coder.get_llm",
        "paper2manim.agents.summarizer.get_llm",
        "paper2manim.agents.reviewer.get_llm",
    ):
        monkeypatch.setattr(attr, fake_get_llm)

    from paper2manim.parsers import ParsedInput

    monkeypatch.setattr(
        "paper2manim.graphs.mvp2.parse_local_pdf",
        lambda p: ParsedInput(text="# T", fmt="markdown", source="pdf:fake"),
    )

    render_events: list[tuple[str, float, float]] = []
    render_lock = threading.Lock()

    def fake_render(code, scene, **kw):
        # Sleep briefly so concurrent renders show overlap.
        t0 = time.monotonic()
        time.sleep(0.05)
        t1 = time.monotonic()
        workdir = Path(kw.get("workdir", tmp_path))
        workdir.mkdir(parents=True, exist_ok=True)
        mp4 = workdir / "scene.mp4"
        mp4.write_bytes(b"\x00")
        with render_lock:
            render_events.append((scene, t0, t1))
        return {
            "status": "success",
            "category": None,
            "exit_code": 0,
            "scene": scene,
            "video_path": str(mp4),
            "workdir": str(workdir),
        }

    monkeypatch.setattr("paper2manim.graphs.scene_graph.render", fake_render)

    def fake_concat(paths, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00")
        return out

    monkeypatch.setattr("paper2manim.graphs.mvp2.concat_videos", fake_concat)

    yield {"render_events": render_events, "scenes_per_test": scenes_per_test}
    _reset_emb_cache()
    concurrency.reset()


def _initial_state(*, scene_names: list[str], **overrides) -> dict:
    base = {
        "input_kind": "pdf",
        "pdf_path": "/fake.pdf",
        "attempts": [],
        "rendered_videos": [],
        "skipped_scenes": [],
        "scene_reports": [],
        "max_retries": 2,
        "quality": "l",
        "skip_render": False,
        "vlm_enabled": False,
        "max_visual_revisions": 2,
        "visual_revision_decisions": [],
    }
    base.update(overrides)
    return base


def _run(state: dict) -> dict:
    from paper2manim.graphs.mvp2 import build_mvp2_graph

    g = build_mvp2_graph()
    return g.invoke(state, config={"recursion_limit": 200})


# --------------------------------------------------------------------------- #
# Equivalence: parallel result == serial result (modulo order)
# --------------------------------------------------------------------------- #


class TestParallelEquivalence:
    def test_single_scene_parallel_equals_serial(self, parallel_pipeline):
        parallel_pipeline["scenes_per_test"]["names"] = ["SceneA"]
        serial = _run(_initial_state(scene_names=["SceneA"], run_id="serial-1"))
        parallel_pipeline["scenes_per_test"]["names"] = ["SceneA"]
        concurrency.configure(scene_parallelism=4)
        parallel = _run(_initial_state(scene_names=["SceneA"], run_id="parallel-1"))
        assert {r["scene"] for r in serial["scene_reports"]} == {
            r["scene"] for r in parallel["scene_reports"]
        }
        assert serial["scene_reports"][0]["final_status"] == "success"
        assert parallel["scene_reports"][0]["final_status"] == "success"

    def test_four_scenes_parallel_produces_same_scene_set(self, parallel_pipeline):
        names = ["SceneA", "SceneB", "SceneC", "SceneD"]
        parallel_pipeline["scenes_per_test"]["names"] = names
        concurrency.configure(scene_parallelism=4)
        out = _run(_initial_state(scene_names=names, run_id="parallel-4"))
        reported = {r["scene"] for r in out["scene_reports"]}
        assert reported == set(names)
        # Every scene's render succeeded and the final video was concatenated.
        assert len(out["rendered_videos"]) == 4
        assert out["final_video_path"].endswith("output.mp4")

    def test_four_scenes_renders_overlap_when_no_semaphore(self, parallel_pipeline):
        names = ["SceneA", "SceneB", "SceneC", "SceneD"]
        parallel_pipeline["scenes_per_test"]["names"] = names
        concurrency.configure(scene_parallelism=4, render_concurrency=None)
        _run(_initial_state(scene_names=names, run_id="parallel-overlap"))
        # With no render semaphore, at least two renders should overlap.
        events = parallel_pipeline["render_events"]
        assert len(events) == 4
        # Sort by start time; check that some later render started before the
        # earlier render finished.
        events.sort(key=lambda x: x[1])
        overlapped = any(events[i + 1][1] < events[i][2] for i in range(len(events) - 1))
        assert overlapped, "expected at least one render overlap with no semaphore"

    def test_render_semaphore_one_serializes_manim(
        self, parallel_pipeline, monkeypatch, tmp_path
    ):
        """End-to-end check: semaphore=1 + 3-scene fan-out → renders serialize.

        The test fixture's default ``fake_render`` doesn't go through
        ``concurrency.render_slot`` (it bypasses ``sandbox.render.render``);
        we install a semaphore-aware fake here so the full integration path
        is exercised.
        """
        names = ["SceneA", "SceneB", "SceneC"]
        parallel_pipeline["scenes_per_test"]["names"] = names
        concurrency.configure(scene_parallelism=4, render_concurrency=1)

        events: list[tuple[str, float, float]] = []
        lock = threading.Lock()

        def fake_render_with_sem(code, scene, **kw):
            with concurrency.render_slot():
                t0 = time.monotonic()
                time.sleep(0.05)
                t1 = time.monotonic()
                workdir = Path(kw.get("workdir", tmp_path))
                workdir.mkdir(parents=True, exist_ok=True)
                mp4 = workdir / "scene.mp4"
                mp4.write_bytes(b"\x00")
                with lock:
                    events.append((scene, t0, t1))
                return {
                    "status": "success",
                    "category": None,
                    "exit_code": 0,
                    "scene": scene,
                    "video_path": str(mp4),
                    "workdir": str(workdir),
                }

        monkeypatch.setattr("paper2manim.graphs.scene_graph.render", fake_render_with_sem)
        _run(_initial_state(scene_names=names, run_id="parallel-sem"))
        assert len(events) == 3
        events.sort(key=lambda x: x[1])
        # No two renders overlap; each next start >= prev finish.
        for prev, nxt in zip(events, events[1:], strict=False):
            assert nxt[1] >= prev[2] - 0.01, "renders overlapped under semaphore=1"


# --------------------------------------------------------------------------- #
# Failure isolation
# --------------------------------------------------------------------------- #


class TestSceneFailureIsolation:
    def test_one_scene_blows_up_others_still_complete(self, parallel_pipeline, monkeypatch, tmp_path):
        names = ["GoodA", "BadCrash", "GoodC", "GoodD"]
        parallel_pipeline["scenes_per_test"]["names"] = names
        concurrency.configure(scene_parallelism=4)

        def maybe_render(code, scene, **kw):
            if scene == "BadCrash":
                raise RuntimeError("simulated scene crash")
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

        monkeypatch.setattr("paper2manim.graphs.scene_graph.render", maybe_render)
        out = _run(_initial_state(scene_names=names, run_id="failure-isolated"))
        # The 3 good scenes still produced videos and reached concat.
        assert len(out["rendered_videos"]) == 3
        assert out["final_video_path"].endswith("output.mp4")
        # The bad scene is recorded in skipped_scenes.
        assert "BadCrash" in out["skipped_scenes"]
        # Every scene shows up in scene_reports.
        scenes_reported = {r["scene"] for r in out["scene_reports"]}
        assert scenes_reported == set(names)
        bad = next(r for r in out["scene_reports"] if r["scene"] == "BadCrash")
        assert bad["status"] == "fatal"


# --------------------------------------------------------------------------- #
# CLI / concurrency configuration wiring
# --------------------------------------------------------------------------- #


class TestConfigureWiring:
    def test_configure_serial_keeps_throttles_off(self):
        concurrency.configure(scene_parallelism=1)
        assert concurrency.RENDER_SEMAPHORE is None
        assert concurrency.LLM_BUCKET is None

    def test_configure_parallel_with_throttles_sets_globals(self):
        concurrency.configure(scene_parallelism=4, render_concurrency=2, llm_rps=3.0)
        assert concurrency.RENDER_SEMAPHORE is not None
        assert concurrency.LLM_BUCKET is not None
