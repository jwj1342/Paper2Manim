"""Audio-video assembly utilities — ffmpeg/ffprobe wrappers.

All ffmpeg/ffprobe calls go through this module so graph nodes never
inline subprocess commands. Reuses ``paper2manim.sandbox.concat.concat_videos``
for the video concatenation path.

Functions:
    probe_duration: ffprobe-based media duration (float seconds)
    concat_audios: ffmpeg concat demuxer for WAV files
    pad_audio: append trailing silence to reach target duration
    speed_audio: apply atempo filter (speed up / slow down)
    mux_audio_video: combine video + audio track into one mp4
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class AVError(RuntimeError):
    """Raised when ffmpeg/ffprobe fails or returns unexpected output."""


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise AVError(f"Required tool '{tool}' not found on PATH.")
    return path


def probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe.

    Raises :class:`AVError` if the file is missing, ffprobe fails, or the
    output is not parseable.
    """
    if not path.exists():
        raise AVError(f"Media file not found: {path}")
    ffprobe = _require("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise AVError(
            f"ffprobe failed for {path}: {result.stderr.strip()}"
        )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AVError(
            f"ffprobe returned non-numeric duration for {path}: {result.stdout!r}"
        ) from exc


def concat_audios(
    audio_paths: list[str], output_path: Path
) -> Path:
    """Concatenate WAV files using ffmpeg concat demuxer.

    When there is only one input, copies it directly to ``output_path``.
    """
    if not audio_paths:
        raise AVError("audio_paths is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(audio_paths) == 1:
        shutil.copy2(audio_paths[0], output_path)
        return output_path

    list_file = output_path.with_suffix(".concat.txt")
    list_file.write_text(
        "".join(f"file '{Path(p).resolve()}'\n" for p in audio_paths),
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    log.info("ffmpeg concat audios: %d files -> %s", len(audio_paths), output_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AVError(f"ffmpeg audio concat failed: {proc.stderr[-500:]}")
    return output_path


def pad_audio(
    audio_path: Path, target_duration_s: float, output_path: Path
) -> Path:
    """Pad ``audio_path`` with trailing silence to reach ``target_duration_s``.

    Uses the ``apad`` filter. If the audio is already at or above the target,
    copies the file as-is (no truncation).
    """
    current = probe_duration(audio_path)
    if current >= target_duration_s:
        log.info(
            "[av.pad] audio=%s already %.2fs >= target %.2fs; copying as-is",
            audio_path.name, current, target_duration_s,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, output_path)
        return output_path

    pad_dur = target_duration_s - current
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-af", f"apad=pad_dur={pad_dur:.3f}",
        str(output_path),
    ]
    log.info(
        "[av.pad] %s +%.2fs silence -> %s",
        audio_path.name, pad_dur, output_path.name,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AVError(f"ffmpeg pad_audio failed: {proc.stderr[-500:]}")
    return output_path


def speed_audio(
    audio_path: Path, factor: float, output_path: Path
) -> Path:
    """Change audio playback speed by ``factor`` using the ``atempo`` filter.

    ``factor`` > 1.0 = faster; < 1.0 = slower. ffmpeg's ``atempo`` operates
    in [0.5, 2.0] per invocation; we chain multiple atempo for values outside
    that range. A factor of exactly 1.0 copies the file.
    """
    if factor <= 0:
        raise AVError(f"speed factor must be positive, got {factor}")
    if abs(factor - 1.0) < 0.001:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg atempo range: [0.5, 2.0]. Chain as needed.
    remaining = factor
    filters: list[str] = []
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.4f}")

    af = ",".join(filters)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-af", af,
        str(output_path),
    ]
    log.info(
        "[av.speed] %s ×%.3f (%s) -> %s",
        audio_path.name, factor, af, output_path.name,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AVError(f"ffmpeg speed_audio failed: {proc.stderr[-500:]}")
    return output_path


def trim_audio(
    audio_path: Path, target_duration_s: float, output_path: Path
) -> Path:
    """Trim ``audio_path`` to exactly ``target_duration_s`` using ffmpeg.

    If the audio is already at or below the target, copies as-is (no padding).
    This is the complement to :func:`pad_audio`: together they guarantee
    that aligned audio matches the target duration exactly.

    Handles in-place trimming (input == output) by writing to a temp file first.
    """
    current = probe_duration(audio_path)
    if current <= target_duration_s:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if audio_path.resolve() != output_path.resolve():
            import shutil
            shutil.copy2(audio_path, output_path)
        return output_path

    # ffmpeg cannot overwrite input in-place; use a temp path when needed.
    in_place = audio_path.resolve() == output_path.resolve()
    actual_out = output_path.parent / f".trim_tmp_{output_path.name}" if in_place else output_path
    actual_out.parent.mkdir(parents=True, exist_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-t", f"{target_duration_s:.3f}",
        "-c", "copy",
        str(actual_out),
    ]
    log.info(
        "[av.trim] %s %.2fs → %.2fs",
        audio_path.name, current, target_duration_s,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        # Stream-copy trim may fail on some codecs; fall back to re-encode.
        log.warning("[av.trim] stream-copy failed; re-encoding: %s", proc.stderr[-200:])
        cmd2 = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-t", f"{target_duration_s:.3f}",
            "-acodec", "pcm_s16le",
            str(actual_out),
        ]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)
        if proc2.returncode != 0:
            if in_place:
                actual_out.unlink(missing_ok=True)
            raise AVError(f"ffmpeg trim failed: {proc2.stderr[-500:]}")

    if in_place:
        actual_out.replace(output_path)
    return output_path


def mux_audio_video(
    video_path: Path, audio_path: Path, output_path: Path
) -> Path:
    """Mux a video track and an audio track into a single mp4.

    Tries stream-copy first (``-c:v copy -c:a aac``); falls back to full
    video re-encode on failure. Uses ``-shortest`` so the output duration
    is the shorter of the two inputs.
    """
    if not video_path.exists():
        raise AVError(f"video file not found: {video_path}")
    if not audio_path.exists():
        raise AVError(f"audio file not found: {audio_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Primary: stream-copy video, encode audio to AAC.
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    log.info("[av.mux] %s + %s -> %s", video_path.name, audio_path.name, output_path.name)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0 and output_path.exists():
        return output_path

    # Fallback: re-encode video.
    log.warning("[av.mux] stream-copy failed; re-encoding video: %s", proc.stderr[-500:])
    cmd2 = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    proc2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)
    if proc2.returncode != 0:
        raise AVError(f"ffmpeg mux failed (both paths): {proc2.stderr[-500:]}")
    return output_path
