"""Graph-level tests for voiceover narration + TTS in MVP1 and MVP2.

These tests use mock TTS (provider=mock) so they don't require real API keys.
ffmpeg/ffprobe must be on PATH for AV assembly steps.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper2manim.config.model_config import TTSConfig
from paper2manim.graphs.mvp1 import assemble_av_node as mvp1_assemble_av
from paper2manim.graphs.mvp2 import assemble_av_node as mvp2_assemble_av
from paper2manim.schemas.narration import NarrationPlanModel
from paper2manim.state import PaperState


def _create_test_mp4(path: Path, duration_s: float = 2.0) -> Path:
    """Minimal black video via ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s=160x120:d={duration_s}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(path),
        ],
        capture_output=True, check=True,
    )
    return path


def _mock_tts_config(monkeypatch):
    """Steer tts_config() to return a mock provider."""
    cfg = TTSConfig(
        provider="mock", model="mock", api_key="fake",
        base_url="", voice="alloy",
    )
    monkeypatch.setattr(
        "paper2manim.graphs.mvp2.get_tts_config", lambda: cfg
    )
    monkeypatch.setattr(
        "paper2manim.graphs.mvp1.get_tts_config", lambda: cfg
    )


def _mock_run_dir(monkeypatch, tmp_path):
    """Redirect run_dir to tmp_path for the graph module under test."""
    fake_run_dir = tmp_path / "runs" / "test-run"
    fake_run_dir.mkdir(parents=True)
    (fake_run_dir / "final").mkdir(exist_ok=True)
    monkeypatch.setattr(
        "paper2manim.graphs.mvp1.run_dir", lambda rid: fake_run_dir
    )
    monkeypatch.setattr(
        "paper2manim.graphs.mvp2.run_dir", lambda rid: fake_run_dir
    )
    return fake_run_dir


# --------------------------------------------------------------------------- #
# Mod 1: Voiceover OFF skips narrator
# --------------------------------------------------------------------------- #


class TestVoiceoverOffSkipsNarrator:
    """fix-plan §Mod 1: --no-voiceover must not call narrator."""

    def test_mvp2_voiceover_off_routes_directly_to_fanout(self, monkeypatch):
        """MVP2: _post_storyboarder returns Send list when voiceover off."""
        from paper2manim.graphs.mvp2 import _post_storyboarder

        state: PaperState = {
            "voiceover_enabled": False,
            "storyboard": {
                "title": "T", "scenes": [{"name": "S", "description": "d", "duration_hint": 5.0}]
            },
            "attempts": [],
        }
        result = _post_storyboarder(state)
        # Should return a list of Send, not "narrator".
        assert isinstance(result, list), f"Expected Send list, got {result!r}"
        assert len(result) == 1

    def test_mvp2_voiceover_on_routes_to_narrator(self, monkeypatch):
        """MVP2: _post_storyboarder returns 'narrator' when voiceover on."""
        from paper2manim.graphs.mvp2 import _post_storyboarder

        state: PaperState = {
            "voiceover_enabled": True,
            "storyboard": {
                "title": "T", "scenes": [{"name": "S", "description": "d", "duration_hint": 5.0}]
            },
            "attempts": [],
        }
        result = _post_storyboarder(state)
        assert result == "narrator"

    def test_mvp1_voiceover_off_routes_to_coder(self):
        """MVP1: _post_storyboarder returns 'coder' when voiceover off."""
        from paper2manim.graphs.mvp1 import _post_storyboarder

        state: PaperState = {
            "voiceover_enabled": False,
            "storyboard": {
                "title": "T", "scenes": [{"name": "S", "description": "d", "duration_hint": 5.0}]
            },
            "attempts": [],
        }
        result = _post_storyboarder(state)
        assert result == "coder"


# --------------------------------------------------------------------------- #
# MVP1 voiceover assembly
# --------------------------------------------------------------------------- #


class TestMVP1Voiceover:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        _mock_tts_config(monkeypatch)

    def test_assemble_av_passthrough_when_voiceover_off(self, monkeypatch, tmp_path):
        run_dir = _mock_run_dir(monkeypatch, tmp_path)
        # Create video OUTSIDE the final/ dir so concat doesn't clash.
        video = _create_test_mp4(run_dir / "scene_s.mp4", duration_s=2.0)

        state: PaperState = {
            "run_id": "test-run",
            "voiceover_enabled": False,
            "rendered_videos": [str(video)],
            "rendered_scene_videos": [
                {"scene": "S", "video_path": str(video), "duration_s": None}
            ],
            "attempts": [],
        }
        out = mvp1_assemble_av(state)
        assert "fatal_error" not in out

    def test_assemble_av_single_scene_narrated(self, monkeypatch, tmp_path):
        run_dir = _mock_run_dir(monkeypatch, tmp_path)
        video = _create_test_mp4(run_dir / "scene.mp4", duration_s=2.0)

        narration = NarrationPlanModel(
            title="Test",
            scenes=[{"scene": "TestScene", "text": "Hello world.", "target_duration_s": 2.0, "language": "en"}],
        )

        state: PaperState = {
            "run_id": "test-run",
            "voiceover_enabled": True,
            "voiceover_strict": True,
            "voiceover_language": "en",
            "narration_plan": narration.model_dump(),
            "rendered_videos": [str(video)],
            "rendered_scene_videos": [
                {"scene": "TestScene", "video_path": str(video), "duration_s": None}
            ],
            "attempts": [],
        }
        out = mvp1_assemble_av(state)
        assert "fatal_error" not in out, out.get("fatal_error")
        assert out.get("narrated_video_path") is not None

        nj = run_dir / "final" / "narration.json"
        assert nj.exists()
        manifest = json.loads(nj.read_text())
        assert manifest["language"] == "en"
        # Per-scene should include voice and speed.
        assert len(manifest["scenes"]) == 1
        assert "voice" in manifest["scenes"][0]

    def test_assemble_av_missing_narration_plan(self, monkeypatch, tmp_path):
        run_dir = _mock_run_dir(monkeypatch, tmp_path)
        video = _create_test_mp4(run_dir / "scene.mp4", duration_s=2.0)

        state: PaperState = {
            "run_id": "test-run",
            "voiceover_enabled": True,
            "voiceover_strict": True,
            "narration_plan": None,
            "rendered_videos": [str(video)],
            "rendered_scene_videos": [
                {"scene": "S", "video_path": str(video), "duration_s": None}
            ],
            "attempts": [],
        }
        out = mvp1_assemble_av(state)
        assert "fatal_error" not in out, out.get("fatal_error")
        # narration_plan=None → voiceover off path → just concat.
        assert out.get("final_video_path") is not None


# --------------------------------------------------------------------------- #
# MVP2 voiceover assembly
# --------------------------------------------------------------------------- #


class TestMVP2Voiceover:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        _mock_tts_config(monkeypatch)

    def test_assemble_av_passthrough_voiceover_off(self, monkeypatch, tmp_path):
        run_dir = _mock_run_dir(monkeypatch, tmp_path)
        video = _create_test_mp4(run_dir / "scene.mp4", duration_s=2.0)

        state: PaperState = {
            "run_id": "test-run",
            "voiceover_enabled": False,
            "rendered_videos": [str(video)],
            "rendered_scene_videos": [
                {"scene": "S", "video_path": str(video), "duration_s": None}
            ],
            "attempts": [],
        }
        out = mvp2_assemble_av(state)
        assert "fatal_error" not in out
        assert out.get("final_video_path") is not None

    def test_multi_scene_narrated(self, monkeypatch, tmp_path):
        run_dir = _mock_run_dir(monkeypatch, tmp_path)
        a = _create_test_mp4(run_dir / "a.mp4", duration_s=2.0)
        b = _create_test_mp4(run_dir / "b.mp4", duration_s=2.0)

        narration = NarrationPlanModel(
            title="Test",
            scenes=[
                {"scene": "Intro", "text": "Welcome.", "target_duration_s": 2.0, "language": "en"},
                {"scene": "Main", "text": "Key idea.", "target_duration_s": 2.0, "language": "en"},
            ],
        )

        state: PaperState = {
            "run_id": "test-run",
            "voiceover_enabled": True,
            "voiceover_strict": True,
            "voiceover_language": "en",
            "narration_plan": narration.model_dump(),
            "rendered_videos": [str(a), str(b)],
            "rendered_scene_videos": [
                {"scene": "Intro", "video_path": str(a), "duration_s": None},
                {"scene": "Main", "video_path": str(b), "duration_s": None},
            ],
            "attempts": [],
        }
        out = mvp2_assemble_av(state)
        assert "fatal_error" not in out, out.get("fatal_error")
        assert out.get("narrated_video_path") is not None

        nj = run_dir / "final" / "narration.json"
        assert nj.exists()
        manifest = json.loads(nj.read_text())
        assert len(manifest["scenes"]) == 2
        for s in manifest["scenes"]:
            assert "voice" in s

    def test_missing_narration_best_effort(self, monkeypatch, tmp_path):
        run_dir = _mock_run_dir(monkeypatch, tmp_path)
        video = _create_test_mp4(run_dir / "v.mp4", duration_s=2.0)

        narration = NarrationPlanModel(
            title="Test",
            scenes=[{"scene": "Other", "text": "x", "target_duration_s": 2.0, "language": "en"}],
        )

        state: PaperState = {
            "run_id": "test-run",
            "voiceover_enabled": True,
            "voiceover_strict": False,
            "narration_plan": narration.model_dump(),
            "rendered_videos": [str(video)],
            "rendered_scene_videos": [
                {"scene": "Missing", "video_path": str(video), "duration_s": None}
            ],
            "attempts": [],
        }
        out = mvp2_assemble_av(state)
        assert "fatal_error" not in out
        # Should have warnings about missing narration.
        assert out.get("voiceover_warnings")


# --------------------------------------------------------------------------- #
# Narrator node tests
# --------------------------------------------------------------------------- #


class TestNarratorNode:
    def test_produces_narration_plan(self, monkeypatch, fake_storyboard):
        from paper2manim.agents.narrator import narrator_node

        narration = NarrationPlanModel(
            title="Test",
            scenes=[
                {"scene": "PythagorasIntro", "text": "Let me explain.", "target_duration_s": 12.0, "language": "en"}
            ],
        )
        structured = MagicMock()
        structured.invoke.return_value = narration
        mock = MagicMock()
        mock.with_structured_output.return_value = structured
        monkeypatch.setattr("paper2manim.agents.narrator.get_llm", lambda *a, **kw: mock)

        state: PaperState = {
            "run_id": "test-run",
            "summary": {"title": "Test Paper"},
            "storyboard": fake_storyboard,
            "voiceover_language": "en",
            "attempts": [],
        }
        out = narrator_node(state)
        assert "narration_plan" in out
        plan = out["narration_plan"]
        assert plan["title"] == "Test"
        assert len(plan["scenes"]) == 1

    def test_fatal_when_no_storyboard(self, monkeypatch):
        from paper2manim.agents.narrator import narrator_node

        state: PaperState = {
            "run_id": "test-run",
            "summary": {"title": "Test"},
            "voiceover_language": "en",
            "attempts": [],
        }
        out = narrator_node(state)
        assert "fatal_error" in out
        assert "storyboard" in out["fatal_error"]


# --------------------------------------------------------------------------- #
# Mod 3: Audio alignment guarantees (trim fallback)
# --------------------------------------------------------------------------- #


class TestAudioAlignment:
    """fix-plan §Mod 3: aligned audio must match target duration exactly."""

    def test_best_effort_trims_overspeed_audio(self, tmp_path):
        """When raw audio exceeds max_speed_factor × target, best-effort
        speeds to max then trims to exact target."""
        from paper2manim.voiceover.assembly import _align_audio, _write_silence_wav

        raw = tmp_path / "raw.wav"
        # Create 10-second silence WAV.
        _write_silence_wav(raw, 10.0)
        out = tmp_path / "aligned.wav"

        # Target 2s, raw 10s → needed factor 5x, exceeds max 1.15.
        aligned, action = _align_audio(raw, 10.0, 2.0, out, strict=False)

        assert "trim" in action, f"Expected trim action, got {action}"
        assert aligned == out
        assert out.exists()

        # Verify aligned duration ≈ target.
        from paper2manim.sandbox.av import probe_duration
        dur = probe_duration(out)
        assert abs(dur - 2.0) <= 0.1, f"Aligned duration {dur:.2f}s, target 2.0s"

    def test_strict_mode_fatal_on_overspeed(self, tmp_path):
        """Strict mode: overspeed audio raises AVError."""
        from paper2manim.voiceover.assembly import _align_audio, _write_silence_wav
        from paper2manim.sandbox.av import AVError

        raw = tmp_path / "raw.wav"
        _write_silence_wav(raw, 10.0)
        out = tmp_path / "aligned.wav"

        with pytest.raises(AVError, match="too long"):
            _align_audio(raw, 10.0, 2.0, out, strict=True)

    def test_pad_silence_for_short_audio(self, tmp_path):
        """Short audio gets padded to target."""
        from paper2manim.voiceover.assembly import _align_audio, _write_silence_wav
        from paper2manim.sandbox.av import probe_duration

        raw = tmp_path / "raw.wav"
        _write_silence_wav(raw, 1.0)
        out = tmp_path / "aligned.wav"

        aligned, action = _align_audio(raw, 1.0, 3.0, out, strict=True)
        assert "pad" in action
        dur = probe_duration(out)
        assert abs(dur - 3.0) <= 0.1

    def test_speed_within_range(self, tmp_path):
        """Audio within 1.15x range gets sped up to exact target."""
        from paper2manim.voiceover.assembly import _align_audio, _write_silence_wav
        from paper2manim.sandbox.av import probe_duration

        raw = tmp_path / "raw.wav"
        _write_silence_wav(raw, 2.3)  # 2.3 / 2.0 = 1.15 → at boundary
        out = tmp_path / "aligned.wav"

        aligned, action = _align_audio(raw, 2.3, 2.0, out, strict=True)
        assert "speed" in action
        dur = probe_duration(out)
        assert abs(dur - 2.0) <= 0.3  # speed has some tolerance
