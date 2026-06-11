"""Tests for TTS infrastructure: MockTTSClient, factory, and TTSConfig."""

import wave
from pathlib import Path

import pytest

from paper2manim.config.model_config import TTSConfig
from paper2manim.infrastructure.tts.client import TTSClient
from paper2manim.infrastructure.tts.factory import build_tts_client
from paper2manim.infrastructure.tts.mock_tts_client import MockTTSClient


class TestMockTTSClient:
    def test_synthesize_writes_valid_wav(self, tmp_path):
        client = MockTTSClient(default_duration_s=1.5)
        out = tmp_path / "test.wav"
        result = client.synthesize("hello", out, voice="alloy")
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

        # Verify it's a valid WAV.
        with wave.open(str(out), "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000

    def test_synthesize_records_calls(self, tmp_path):
        client = MockTTSClient()
        out = tmp_path / "a.wav"
        client.synthesize("text a", out, voice="echo")
        assert len(client.calls) == 1
        assert client.calls[0]["text"] == "text a"
        assert client.calls[0]["voice"] == "echo"
        assert client.calls[0]["speed"] == 1.0

    def test_speed_affects_duration(self, tmp_path):
        client = MockTTSClient(default_duration_s=2.0)
        out_fast = tmp_path / "fast.wav"
        client.synthesize("x", out_fast, voice="a", speed=2.0)
        with wave.open(str(out_fast), "r") as wf:
            n_frames_fast = wf.getnframes()

        out_slow = tmp_path / "slow.wav"
        client.synthesize("x", out_slow, voice="a", speed=0.5)
        with wave.open(str(out_slow), "r") as wf:
            n_frames_slow = wf.getnframes()

        # fast (2x) should have fewer frames than slow (0.5x).
        assert n_frames_fast < n_frames_slow

    def test_rejects_non_wav_format(self, tmp_path):
        client = MockTTSClient()
        with pytest.raises(ValueError, match="only supports 'wav'"):
            client.synthesize("test", tmp_path / "x.mp3", voice="a", audio_format="mp3")


class TestTTSFactory:
    def test_mock_provider_returns_mock_client(self):
        cfg = TTSConfig(
            provider="mock", model="mock", api_key="fake-key",
            base_url="", voice="a",
        )
        client = build_tts_client(cfg)
        assert isinstance(client, MockTTSClient)

    def test_openai_provider_smoke(self):
        """Verifies factory routing; does not call the real API."""
        cfg = TTSConfig(
            provider="openai", model="tts-1", api_key="sk-fake",
            base_url="https://api.openai.com/v1", voice="alloy",
        )
        client = build_tts_client(cfg)
        assert isinstance(client, TTSClient)

    def test_edge_provider_returns_edge_client(self):
        """edge provider is recognized by the factory."""
        cfg = TTSConfig(
            provider="edge", model="edge-tts", api_key="",
            base_url="", voice="en-US-AriaNeural",
        )
        client = build_tts_client(cfg)
        assert isinstance(client, TTSClient)

    def test_unsupported_provider_raises(self):
        cfg = TTSConfig(
            provider="elevenlabs", model="x", api_key="k",
            base_url="", voice="a",
        )
        with pytest.raises(RuntimeError, match="Unsupported TTS provider"):
            build_tts_client(cfg)


class TestTTSConfig:
    def test_from_dict_minimal(self):
        d = {"model": "gpt-4o-mini-tts", "api_key": "sk-abc"}
        cfg = TTSConfig.from_dict(d)
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o-mini-tts"
        assert cfg.voice == "alloy"
        assert cfg.audio_format == "wav"
        assert cfg.speed == 1.0

    def test_from_dict_custom(self):
        d = {
            "provider": "openai",
            "model": "tts-1-hd",
            "api_key": "sk-xyz",
            "voice": "onyx",
            "speed": 1.25,
            "audio_format": "mp3",
        }
        cfg = TTSConfig.from_dict(d)
        assert cfg.voice == "onyx"
        assert cfg.speed == 1.25
        assert cfg.audio_format == "mp3"

    def test_from_dict_missing_model_raises(self):
        with pytest.raises(KeyError, match="model"):
            TTSConfig.from_dict({"api_key": "k"})

    def test_openai_missing_api_key_raises(self):
        """openai provider requires api_key (fix-plan §Mod 5)."""
        with pytest.raises(KeyError, match="api_key"):
            TTSConfig.from_dict({"provider": "openai", "model": "tts-1"})

    def test_edge_missing_api_key_ok(self):
        """edge provider does not require api_key (fix-plan §Mod 5)."""
        cfg = TTSConfig.from_dict({"provider": "edge", "model": "edge-tts"})
        assert cfg.provider == "edge"
        assert cfg.api_key == ""

    def test_mock_missing_api_key_ok(self):
        """mock provider does not require api_key (fix-plan §Mod 5)."""
        cfg = TTSConfig.from_dict({"provider": "mock", "model": "mock"})
        assert cfg.provider == "mock"
        assert cfg.api_key == ""
