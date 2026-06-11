"""Shared voiceover assembly — TTS synthesis, audio alignment, concat, mux.

Used by both MVP1 and MVP2 ``assemble_av_node`` so the two graphs don't
duplicate the same ~200 lines of TTS/alignment/mux logic.

Per the fix-plan §Mod 3, aligned audio is **guaranteed** to match the target
video duration: when raw audio exceeds ``max_speed_factor × target`` in
best-effort mode, we speed to the cap and then **trim** to the exact target.
This prevents cross-scene audio drift that previously relied on ``-shortest``.
"""

from __future__ import annotations

import logging
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper2manim.artifacts import append_trace, run_dir, save_narration_json
from paper2manim.config.model_config import TTSConfig
from paper2manim.infrastructure.tts.client import TTSClient
from paper2manim.infrastructure.tts.factory import build_tts_client
from paper2manim.sandbox.av import (
    AVError,
    concat_audios,
    mux_audio_video,
    pad_audio,
    probe_duration,
    speed_audio,
    trim_audio,
)
from paper2manim.sandbox.concat import concat_videos

log = logging.getLogger(__name__)

# Maximum speed-up factor in best-effort mode before falling back to trim.
_MAX_SPEED_FACTOR = 1.15


@dataclass
class VoiceoverAssemblyResult:
    """Structured result from :func:`assemble_voiceover`.

    Callers map these fields onto ``PaperState`` returns.
    """

    final_video_path: str  # narrated mp4 (or silent mp4 when voiceover is off)
    silent_video_path: str
    narrated_video_path: str | None = None
    final_audio_path: str | None = None
    tts_audio_paths: list[str] = field(default_factory=list)
    narration_manifest: dict[str, Any] | None = None
    warnings: list[dict] = field(default_factory=list)
    fatal_error: str | None = None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def assemble_voiceover(
    *,
    run_id: str,
    scene_videos: list[dict],
    narration_plan: dict | None,
    tts_cfg: TTSConfig | None,
    voice_override: str | None,
    speed_override: float | None,
    language: str,
    strict: bool,
) -> VoiceoverAssemblyResult:
    """Produce a narrated mp4 from rendered scene videos + narration plan.

    When ``narration_plan`` is ``None`` or ``tts_cfg`` is ``None`` (voiceover
    disabled), this is a thin wrapper around ``concat_videos`` — identical to
    the pre-voiceover behaviour.

    When voiceover is enabled, this runs the full pipeline:
    per-scene TTS → audio alignment → audio concat → mux → narration.json.
    """
    videos = [sv["video_path"] for sv in scene_videos if sv.get("video_path")]
    if not videos:
        return VoiceoverAssemblyResult(
            final_video_path="",
            silent_video_path="",
            fatal_error="assemble_voiceover: no successful scenes to concatenate",
        )

    out_path = run_dir(run_id) / "final" / "output.mp4"
    silent = concat_videos(videos, out_path)
    append_trace(run_id, "concat", {"n_videos": len(videos), "final": str(silent)})

    # Voiceover OFF → pass-through.
    if narration_plan is None or tts_cfg is None:
        return VoiceoverAssemblyResult(
            final_video_path=str(silent),
            silent_video_path=str(silent),
        )

    # ---- Voiceover ON ----
    tts = build_tts_client(tts_cfg)
    final_audio_dir = run_dir(run_id) / "final" / "audio"
    final_audio_dir.mkdir(parents=True, exist_ok=True)

    # Build narration lookup by scene name.
    narration_by_scene: dict[str, dict] = {}
    for entry in narration_plan.get("scenes", []):
        narration_by_scene[entry.get("scene", "")] = entry

    default_voice = voice_override or tts_cfg.voice
    default_speed = speed_override or tts_cfg.speed

    warnings: list[dict] = []
    narration_entries: list[dict] = []
    aligned_audios: list[str] = []

    for sv in scene_videos:
        scene_name = str(sv.get("scene", ""))
        video_path = str(sv.get("video_path", ""))
        if not video_path or not Path(video_path).exists():
            warnings.append(
                {"scene": scene_name, "warning": "video_path missing; skipping"}
            )
            continue

        try:
            video_dur = probe_duration(Path(video_path))
        except AVError as exc:
            if strict:
                return VoiceoverAssemblyResult(
                    final_video_path=str(silent),
                    silent_video_path=str(silent),
                    fatal_error=f"assemble_voiceover: ffprobe failed for {scene_name}: {exc}",
                )
            warnings.append({"scene": scene_name, "warning": f"ffprobe failed: {exc}"})
            continue

        narration = narration_by_scene.get(scene_name)
        if narration is None:
            entry, aligned = _missing_narration(
                scene_name, video_path, video_dur,
                final_audio_dir, strict, warnings,
            )
            if aligned is not None:
                aligned_audios.append(str(aligned))
            narration_entries.append(entry)
            if strict and aligned is None:
                return VoiceoverAssemblyResult(
                    final_video_path=str(silent),
                    silent_video_path=str(silent),
                    fatal_error=(
                        f"assemble_voiceover: scene {scene_name!r} has no narration "
                        f"entry (strict mode)"
                    ),
                )
            continue

        narration_text = str(narration.get("text", ""))
        if not narration_text.strip():
            entry, aligned = _empty_narration(
                scene_name, video_path, video_dur,
                final_audio_dir, strict, warnings,
            )
            if aligned is not None:
                aligned_audios.append(str(aligned))
            narration_entries.append(entry)
            if strict and aligned is None:
                return VoiceoverAssemblyResult(
                    final_video_path=str(silent),
                    silent_video_path=str(silent),
                    fatal_error=(
                        f"assemble_voiceover: scene {scene_name!r} has empty narration "
                        f"text (strict mode)"
                    ),
                )
            continue

        # Synthesize + align this scene.
        scene_voice = narration.get("voice") or default_voice
        scene_speed = float(narration.get("speed") or default_speed)
        entry, aligned, scene_warnings = _synthesize_and_align(
            scene_name=scene_name,
            video_path=video_path,
            video_dur=video_dur,
            narration_text=narration_text,
            voice=scene_voice,
            speed=scene_speed,
            tts=tts,
            tts_cfg=tts_cfg,
            final_audio_dir=final_audio_dir,
            strict=strict,
        )
        if aligned is not None:
            aligned_audios.append(str(aligned))
        narration_entries.append(entry)
        warnings.extend(scene_warnings)
        if strict and aligned is None:
            return VoiceoverAssemblyResult(
                final_video_path=str(silent),
                silent_video_path=str(silent),
                fatal_error=(
                    f"assemble_voiceover: TTS or alignment failed for {scene_name}"
                ),
            )

    # Concat audios + mux.
    voiceover_wav = run_dir(run_id) / "final" / "voiceover.wav"
    narrated_out = run_dir(run_id) / "final" / "output_narrated.mp4"

    try:
        concat_audios(aligned_audios, voiceover_wav)
        mux_audio_video(silent, voiceover_wav, narrated_out)
    except AVError as exc:
        return VoiceoverAssemblyResult(
            final_video_path=str(silent),
            silent_video_path=str(silent),
            fatal_error=f"assemble_voiceover: mux failed: {exc}",
        )

    # Build manifest.
    overrides: dict[str, Any] = {}
    if voice_override:
        overrides["voice"] = voice_override
    if speed_override is not None:
        overrides["speed"] = speed_override

    narration_manifest = {
        "title": narration_plan.get("title", ""),
        "language": language,
        "tts": {
            "provider": tts_cfg.provider,
            "model": tts_cfg.model,
            "default_voice": tts_cfg.voice,
            "default_speed": tts_cfg.speed,
            "audio_format": tts_cfg.audio_format,
        },
        "scenes": narration_entries,
    }
    if overrides:
        narration_manifest["overrides"] = overrides

    try:
        save_narration_json(run_id, narration_manifest)
    except OSError as exc:
        log.warning("[assemble_voiceover] failed to save narration.json: %s", exc)

    append_trace(
        run_id,
        "assemble_av",
        {
            "n_scenes": len(narration_entries),
            "n_warnings": len(warnings),
            "silent_video": str(silent),
            "narrated_video": str(narrated_out),
            "voiceover_wav": str(voiceover_wav),
        },
    )

    return VoiceoverAssemblyResult(
        final_video_path=str(narrated_out),
        silent_video_path=str(silent),
        narrated_video_path=str(narrated_out),
        final_audio_path=str(voiceover_wav),
        tts_audio_paths=aligned_audios,
        narration_manifest=narration_manifest,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Per-scene helpers
# --------------------------------------------------------------------------- #


def _missing_narration(
    scene_name: str,
    video_path: str,
    video_dur: float,
    audio_dir: Path,
    strict: bool,
    warnings: list[dict],
) -> tuple[dict, Path | None]:
    """Handle a scene with no matching narration entry."""
    w = {"scene": scene_name, "warning": "no narration entry; inserting silence"}
    warnings.append(w)
    silence = audio_dir / f"{scene_name}.silence.wav"
    _write_silence_wav(silence, video_dur)
    entry = {
        "scene": scene_name,
        "text": "",
        "video_path": video_path,
        "video_duration_s": round(video_dur, 3),
        "raw_audio_path": None,
        "raw_audio_duration_s": None,
        "aligned_audio_path": str(silence),
        "aligned_audio_duration_s": round(video_dur, 3),
        "alignment_action": "inserted_silence",
    }
    return entry, silence


def _empty_narration(
    scene_name: str,
    video_path: str,
    video_dur: float,
    audio_dir: Path,
    strict: bool,
    warnings: list[dict],
) -> tuple[dict, Path | None]:
    """Handle a scene with empty narration text."""
    w = {"scene": scene_name, "warning": "empty narration text; inserting silence"}
    warnings.append(w)
    silence = audio_dir / f"{scene_name}.silence.wav"
    _write_silence_wav(silence, video_dur)
    entry = {
        "scene": scene_name,
        "text": "",
        "video_path": video_path,
        "video_duration_s": round(video_dur, 3),
        "raw_audio_path": None,
        "raw_audio_duration_s": None,
        "aligned_audio_path": str(silence),
        "aligned_audio_duration_s": round(video_dur, 3),
        "alignment_action": "inserted_silence",
    }
    return entry, silence


def _synthesize_and_align(
    *,
    scene_name: str,
    video_path: str,
    video_dur: float,
    narration_text: str,
    voice: str,
    speed: float,
    tts: TTSClient,
    tts_cfg: TTSConfig,
    final_audio_dir: Path,
    strict: bool,
) -> tuple[dict, Path | None, list[dict]]:
    """Synthesize one scene's narration and align it to video duration.

    Returns ``(manifest_entry, aligned_path | None, warnings)``.
    ``aligned_path`` is ``None`` when a fatal error occurs in strict mode.
    """
    warnings: list[dict] = []
    raw_path = final_audio_dir / f"{scene_name}.raw.wav"

    # TTS synthesis.
    try:
        tts.synthesize(
            narration_text,
            raw_path,
            voice=voice,
            speed=speed,
            audio_format=tts_cfg.audio_format,
        )
    except Exception as exc:  # noqa: BLE001
        if strict:
            return (
                {"scene": scene_name, "text": narration_text},
                None,
                [],
            )
        w = {"scene": scene_name, "warning": f"TTS failed ({type(exc).__name__})"}
        warnings.append(w)
        silence = final_audio_dir / f"{scene_name}.silence.wav"
        _write_silence_wav(silence, video_dur)
        return (
            {
                "scene": scene_name,
                "text": narration_text,
                "video_path": video_path,
                "video_duration_s": round(video_dur, 3),
                "raw_audio_path": None,
                "raw_audio_duration_s": None,
                "aligned_audio_path": str(silence),
                "aligned_audio_duration_s": round(video_dur, 3),
                "alignment_action": "tts_failed_silence",
                "voice": voice,
                "speed": speed,
            },
            silence,
            warnings,
        )

    # Probe raw duration.
    try:
        raw_dur = probe_duration(raw_path)
    except AVError:
        raw_dur = video_dur

    # Align.
    aligned_path = final_audio_dir / f"{scene_name}.aligned.wav"
    try:
        aligned, action = _align_audio(
            raw_path, raw_dur, video_dur, aligned_path, strict=strict
        )
    except AVError as exc:
        if strict:
            return (
                {"scene": scene_name, "text": narration_text},
                None,
                [],
            )
        w = {
            "scene": scene_name,
            "warning": f"audio alignment failed ({exc}); keeping raw",
        }
        warnings.append(w)
        aligned = raw_path
        action = "alignment_failed_kept_raw"

    try:
        aligned_dur = probe_duration(aligned)
    except AVError:
        aligned_dur = None

    return (
        {
            "scene": scene_name,
            "text": narration_text,
            "video_path": video_path,
            "video_duration_s": round(video_dur, 3),
            "raw_audio_path": str(raw_path),
            "raw_audio_duration_s": round(raw_dur, 3),
            "aligned_audio_path": str(aligned),
            "aligned_audio_duration_s": round(aligned_dur, 3) if aligned_dur else None,
            "alignment_action": action,
            "voice": voice,
            "speed": speed,
        },
        aligned,
        warnings,
    )


def _align_audio(
    raw_path: Path,
    raw_dur: float,
    target_dur: float,
    output_path: Path,
    *,
    strict: bool,
) -> tuple[Path, str]:
    """Align ``raw_path`` to exactly ``target_dur`` seconds.

    Rules:
    - raw ≤ target: pad silence to target.
    - target < raw ≤ target × max_speed: speed up by raw/target.
    - raw > target × max_speed:
        strict  → raise AVError (caller converts to fatal).
        best-effort → speed to max_speed, then **trim** to exact target.

    The trim step is the key fix from the fix-plan §Mod 3: without it,
    overspeed audio still exceeds the target and causes cross-scene drift.
    """
    if raw_dur <= target_dur:
        return pad_audio(raw_path, target_dur, output_path), "pad_silence"

    needed_factor = raw_dur / target_dur

    if needed_factor <= _MAX_SPEED_FACTOR:
        return (
            speed_audio(raw_path, needed_factor, output_path),
            f"speed_{needed_factor:.3f}x",
        )

    # Audio is too long even at max speed.
    if strict:
        raise AVError(
            f"narration for {raw_path.name} is too long "
            f"(raw={raw_dur:.1f}s, target={target_dur:.1f}s, "
            f"needed_factor={needed_factor:.2f} > max={_MAX_SPEED_FACTOR})"
        )

    # Best-effort: speed to max, then trim to exact target.
    _ = speed_audio(raw_path, _MAX_SPEED_FACTOR, output_path)
    trimmed = trim_audio(output_path, target_dur, output_path)
    action = f"speed_{_MAX_SPEED_FACTOR}x_then_trim"
    log.warning(
        "[voiceover] %s: raw=%.1fs target=%.1fs → speed_%.2fx_then_trim",
        raw_path.name, raw_dur, target_dur, _MAX_SPEED_FACTOR,
    )
    return trimmed, action


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _write_silence_wav(path: Path, duration_s: float) -> None:
    """Write a minimal 16-bit mono PCM WAV with silence."""
    sample_rate = 24000
    num_samples = int(sample_rate * max(duration_s, 0.1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
