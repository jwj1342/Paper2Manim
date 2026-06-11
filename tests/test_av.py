"""Tests for paper2manim.sandbox.av — ffmpeg/ffprobe utilities."""

import struct
import wave
from pathlib import Path

import pytest

from paper2manim.sandbox.av import (
    AVError,
    concat_audios,
    mux_audio_video,
    pad_audio,
    probe_duration,
    speed_audio,
)


def _create_silence_wav(path: Path, duration_s: float) -> Path:
    """Write a minimal 16-bit mono PCM WAV at 24kHz."""
    sample_rate = 24000
    num_samples = int(sample_rate * duration_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    return path


def _create_test_mp4(path: Path, duration_s: float = 2.0) -> Path:
    """Create a minimal test MP4 using ffmpeg with blank video."""
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s=320x240:d={duration_s}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return path


class TestProbeDuration:
    def test_returns_duration(self, tmp_path):
        mp4 = _create_test_mp4(tmp_path / "test.mp4", duration_s=2.0)
        dur = probe_duration(mp4)
        assert 1.5 <= dur <= 2.5  # Allow small ffmpeg variance

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AVError, match="not found"):
            probe_duration(tmp_path / "nonexistent.mp4")


class TestPadAudio:
    def test_pads_to_target(self, tmp_path):
        src = _create_silence_wav(tmp_path / "src.wav", 1.0)
        out = tmp_path / "padded.wav"
        result = pad_audio(src, 3.0, out)
        assert result == out
        padded_dur = probe_duration(out)
        assert 2.8 <= padded_dur <= 3.2

    def test_already_at_target_copies(self, tmp_path):
        src = _create_silence_wav(tmp_path / "src.wav", 2.0)
        out = tmp_path / "copied.wav"
        result = pad_audio(src, 1.0, out)
        # Already above target — should copy as-is.
        assert result == out
        assert out.exists()


class TestSpeedAudio:
    def test_speed_up_shortens(self, tmp_path):
        src = _create_silence_wav(tmp_path / "src.wav", 4.0)
        out = tmp_path / "fast.wav"
        result = speed_audio(src, 2.0, out)
        fast_dur = probe_duration(result)
        assert 1.7 <= fast_dur <= 2.3  # ~2x faster → half duration

    def test_slow_down_lengthens(self, tmp_path):
        src = _create_silence_wav(tmp_path / "src.wav", 2.0)
        out = tmp_path / "slow.wav"
        result = speed_audio(src, 0.5, out)
        slow_dur = probe_duration(result)
        assert 3.5 <= slow_dur <= 4.5  # ~0.5x speed → double duration

    def test_factor_one_copies(self, tmp_path):
        src = _create_silence_wav(tmp_path / "src.wav", 1.0)
        out = tmp_path / "same.wav"
        result = speed_audio(src, 1.0, out)
        assert result == out
        assert out.exists()

    def test_factor_zero_raises(self, tmp_path):
        src = _create_silence_wav(tmp_path / "src.wav", 1.0)
        with pytest.raises(AVError, match="positive"):
            speed_audio(src, 0.0, tmp_path / "bad.wav")


class TestConcatAudios:
    def test_single_file_copies(self, tmp_path):
        src = _create_silence_wav(tmp_path / "a.wav", 1.0)
        out = tmp_path / "out.wav"
        result = concat_audios([str(src)], out)
        assert result == out
        assert out.exists()

    def test_two_files_concatenates(self, tmp_path):
        a = _create_silence_wav(tmp_path / "a.wav", 1.0)
        b = _create_silence_wav(tmp_path / "b.wav", 2.0)
        out = tmp_path / "concat.wav"
        result = concat_audios([str(a), str(b)], out)
        dur = probe_duration(result)
        assert 2.7 <= dur <= 3.3  # ~1 + 2 = 3s

    def test_empty_list_raises(self, tmp_path):
        with pytest.raises(AVError, match="empty"):
            concat_audios([], tmp_path / "out.wav")


class TestMuxAudioVideo:
    def test_mux_produces_valid_mp4(self, tmp_path):
        video = _create_test_mp4(tmp_path / "vid.mp4", duration_s=2.0)
        audio = _create_silence_wav(tmp_path / "aud.wav", 2.0)
        out = tmp_path / "muxed.mp4"
        result = mux_audio_video(video, audio, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_mux_missing_video_raises(self, tmp_path):
        audio = _create_silence_wav(tmp_path / "aud.wav", 1.0)
        with pytest.raises(AVError, match="video file not found"):
            mux_audio_video(tmp_path / "nope.mp4", audio, tmp_path / "out.mp4")

    def test_mux_missing_audio_raises(self, tmp_path):
        video = _create_test_mp4(tmp_path / "vid.mp4", duration_s=1.0)
        with pytest.raises(AVError, match="audio file not found"):
            mux_audio_video(video, tmp_path / "nope.wav", tmp_path / "out.mp4")
