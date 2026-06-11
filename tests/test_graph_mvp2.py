"""Reflection-loop tests for the MVP 2.0 graph (mocks LLM + render)."""

from unittest.mock import MagicMock

import pytest

from paper2manim.schemas import StoryboardModel, SummaryModel


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub all LLM calls + parser + render so the graph is purely deterministic."""
    summary_obj = SummaryModel(
        title="Test Paper",
        key_contributions=["A", "B"],
        key_formulas=[],
        main_concepts=["concept"],
        scene_suggestions=["show it"],
    )
    sb_obj = StoryboardModel(
        title="Test Paper",
        scenes=[{"name": "OneScene", "description": "x", "duration_hint": 4.0}],
    )

    summarizer_struct = MagicMock()
    summarizer_struct.invoke.return_value = summary_obj
    sb_struct = MagicMock()
    sb_struct.invoke.return_value = sb_obj

    coder_msg = MagicMock()
    coder_msg.content = "```python\nfrom manim import *\nclass OneScene(Scene):\n    def construct(self):\n        self.wait(0.1)\n```"

    reviewer_msg = MagicMock()
    reviewer_msg.content = '{"decision":"retry","hint":"replace \\\\mathbbb with \\\\mathbb"}'

    llm = MagicMock()

    def with_structured_output(model_cls, **_kwargs):
        if model_cls.__name__ == "SummaryModel":
            return summarizer_struct
        if model_cls.__name__ == "StoryboardModel":
            return sb_struct
        return MagicMock()

    llm.with_structured_output.side_effect = with_structured_output

    def fake_invoke(messages):
        # storyboarder/summarizer go through with_structured_output;
        # coder & reviewer call llm.invoke directly
        # Discriminate by message content (system prompt for reviewer mentions "review agent")
        sys_msg = messages[0][1] if messages else ""
        if "review" in sys_msg.lower() and "build-and-review" in sys_msg.lower():
            return reviewer_msg
        return coder_msg

    llm.invoke.side_effect = fake_invoke

    monkeypatch.setattr("paper2manim.agents.storyboarder.get_llm", lambda *a, **kw: llm)
    monkeypatch.setattr("paper2manim.agents.coder.get_llm", lambda *a, **kw: llm)
    monkeypatch.setattr("paper2manim.agents.summarizer.get_llm", lambda *a, **kw: llm)
    monkeypatch.setattr("paper2manim.agents.reviewer.get_llm", lambda *a, **kw: llm)
    from paper2manim.parsers import ParsedInput

    monkeypatch.setattr(
        "paper2manim.graphs.mvp2.parse_local_pdf",
        lambda p: ParsedInput(text="# Test paper\nbody", fmt="markdown", source="pdf:fake.pdf"),
    )

    # First two render calls fail with latex error; third succeeds.
    render_calls = {"n": 0}

    def fake_render(code, scene, **kw):
        render_calls["n"] += 1
        if render_calls["n"] <= 2:
            return {
                "status": "error",
                "category": "latex",
                "exit_code": 1,
                "scene": scene,
                "video_path": None,
                "error_type": "LatexError",
                "error_message": "! Undefined control sequence \\mathbbb",
                "traceback_tail": "LatexError",
                "source_excerpt": [{"line": 1, "code": "x"}],
                "tex_log_excerpt": "! Undefined control sequence \\mathbbb",
                "workdir": str(kw.get("workdir", "/tmp")),
            }
        from pathlib import Path

        fake_mp4 = Path(kw.get("workdir", "/tmp")) / "fake.mp4"
        fake_mp4.parent.mkdir(parents=True, exist_ok=True)
        fake_mp4.write_bytes(b"\x00")
        return {
            "status": "success",
            "category": None,
            "exit_code": 0,
            "scene": scene,
            "video_path": str(fake_mp4),
            "workdir": str(kw.get("workdir", "/tmp")),
        }

    monkeypatch.setattr("paper2manim.graphs.scene_graph.render", fake_render)

    # Stub assemble_voiceover: return a placeholder result with a fake mp4.
    def fake_assemble_voiceover(**kwargs):
        from pathlib import Path

        from paper2manim.voiceover.assembly import VoiceoverAssemblyResult

        run_id = kwargs.get("run_id", "mvp2-test")
        silent = Path("runs") / run_id / "final" / "silent.mp4"
        silent.parent.mkdir(parents=True, exist_ok=True)
        silent.write_bytes(b"\x00")
        return VoiceoverAssemblyResult(
            final_video_path=str(silent),
            silent_video_path=str(silent),
        )

    monkeypatch.setattr("paper2manim.graphs.mvp2.assemble_voiceover", fake_assemble_voiceover)
    return {"render_calls": render_calls, "llm": llm}


def test_mvp2_reflection_succeeds_after_two_retries(stub_pipeline, tmp_path):
    from paper2manim.graphs.mvp2 import build_mvp2_graph

    g = build_mvp2_graph()
    state = {
        "run_id": "mvp2-test",
        "input_kind": "pdf",
        "pdf_path": str(tmp_path / "fake.pdf"),
        "attempts": [],
        "rendered_videos": [],
        "current_scene_idx": 0,
        "iter_count": 0,
        "max_retries": 4,
        "quality": "l",
        "skip_render": False,
    }
    out = g.invoke(state, config={"recursion_limit": 80})

    # 2 latex failures + 1 success = 3 attempts
    assert stub_pipeline["render_calls"]["n"] == 3
    assert len(out["attempts"]) == 3
    assert out["attempts"][-1]["render_result"]["status"] == "success"
    assert out.get("final_video_path", "").endswith(".mp4")


def test_mvp2_reflection_gives_up_at_cap(stub_pipeline, tmp_path, monkeypatch):
    """If max_retries=1, the loop should give up after one retry."""
    from paper2manim.graphs.mvp2 import build_mvp2_graph

    # Force render to always fail
    def always_fail(code, scene, **kw):
        return {
            "status": "error",
            "category": "latex",
            "exit_code": 1,
            "scene": scene,
            "video_path": None,
            "error_type": "LatexError",
            "error_message": "still broken",
            "traceback_tail": "...",
            "source_excerpt": None,
            "tex_log_excerpt": None,
            "workdir": str(kw.get("workdir", "/tmp")),
        }

    monkeypatch.setattr("paper2manim.graphs.scene_graph.render", always_fail)

    g = build_mvp2_graph()
    state = {
        "run_id": "mvp2-give-up",
        "input_kind": "pdf",
        "pdf_path": str(tmp_path / "fake.pdf"),
        "attempts": [],
        "rendered_videos": [],
        "skipped_scenes": [],
        "current_scene_idx": 0,
        "iter_count": 0,
        "max_retries": 1,
        "quality": "l",
        "skip_render": False,
    }
    out = g.invoke(state, config={"recursion_limit": 60})
    # No successful video; concat should have raised but our fake handles empty
    assert all(a["render_result"]["status"] == "error" for a in out["attempts"])
    # Either fatal_error from concat, or final_video_path missing
    assert not out.get("rendered_videos")
    # C2 regression: gave-up scene must be recorded in skipped_scenes (not fatal_error)
    assert out.get("skipped_scenes") == ["OneScene"]


def test_mvp2_early_exit_on_parser_fatal(monkeypatch):
    """C2 regression: a fatal_error set by parser must short-circuit to END,
    not cascade through summarizer/storyboarder/coder/render.

    Note: nodes are bound into the graph at build time, so we must monkeypatch
    BEFORE calling build_mvp2_graph().
    """
    from paper2manim.graphs import mvp2 as mvp2_mod

    # 1) Parser sets fatal_error
    def parser_sets_fatal(state):
        return {"fatal_error": "parser: pdf missing"}

    monkeypatch.setattr(mvp2_mod, "parser_node", parser_sets_fatal)

    # 2) Sentinels for the remaining top-level nodes. The per-scene loop has
    # moved into a sub-graph (scene_graph), so individual reflection nodes
    # like ``coder`` / ``render`` / ``reviewer`` are no longer top-level
    # nodes here. ``run_scene`` is the single Send target.
    called = {
        "summarizer": False,
        "storyboarder": False,
        "narrator": False,
        "run_scene": False,
        "assemble_av": False,
        "emb_consolidate": False,
    }

    def make_sentinel(name):
        def fn(state):
            called[name] = True
            return {}

        return fn

    for name in called:
        monkeypatch.setattr(mvp2_mod, f"{name}_node", make_sentinel(name))

    # 3) Build graph AFTER patching, so the patched functions are bound
    g = mvp2_mod.build_mvp2_graph()
    out = g.invoke(
        {
            "run_id": "early-exit",
            "input_kind": "pdf",
            "pdf_path": "/does/not/exist.pdf",
            "attempts": [],
            "rendered_videos": [],
            "skipped_scenes": [],
            "current_scene_idx": 0,
            "iter_count": 0,
            "max_retries": 3,
            "quality": "l",
            "skip_render": False,
        },
        config={"recursion_limit": 30},
    )

    assert out.get("fatal_error") == "parser: pdf missing"
    assert not any(called.values()), f"early-exit broken; called: {called}"


def test_render_node_handles_degenerate_inputs(monkeypatch, tmp_path):
    """C1 regression: scene_graph.render_node must degrade gracefully.

    Post-refactor the per-scene render_node lives in scene_graph and reads
    ``state["scene"]`` directly (never a full storyboard). Out-of-range / missing
    cases are guarded by fan_out_scenes before Send, so render_node only has to
    handle: skip_render, no-scene, and empty current_code (synthetic error
    attempt for reviewer to act on).
    """
    from paper2manim.graphs.scene_graph import render_node

    # skip_render honored, no-op
    out = render_node({"skip_render": True, "scene": {"name": "S"}})
    assert out == {}

    # No scene attached → no-op (defensive)
    out = render_node({"run_id": "r", "current_code": "x"})
    assert out == {}

    # Empty code with a real scene → synthetic error attempt, NOT a fatal error
    out = render_node(
        {
            "run_id": "r",
            "scene": {"name": "S", "description": "d", "duration_hint": 5},
            "current_code": "",
        }
    )
    attempts = out.get("attempts") or []
    assert len(attempts) == 1
    assert attempts[0]["render_result"]["status"] == "error"
    assert "current_code" in attempts[0]["render_result"]["error_message"]
