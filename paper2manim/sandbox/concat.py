"""Concatenate multiple scene mp4s into a single output mp4 via ffmpeg concat demuxer."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def concat_videos(mp4_paths: list[str], output_path: Path) -> Path:
    """Concatenate using ffmpeg concat demuxer (lossless if codecs match).

    Manim outputs all scenes with the same encoding params (cairo + libx264 + same fps),
    so concat demuxer works without re-encoding.
    """
    if not mp4_paths:
        raise ValueError("mp4_paths is empty")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(mp4_paths) == 1:
        # Just copy
        import shutil

        shutil.copy2(mp4_paths[0], output_path)
        return output_path

    list_file = output_path.with_suffix(".concat.txt")
    list_file.write_text(
        "".join(f"file '{Path(p).resolve()}'\n" for p in mp4_paths), encoding="utf-8"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_path),
    ]
    log.info("ffmpeg concat: %s -> %s", mp4_paths, output_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        # Fallback: re-encode (slower but more permissive)
        log.warning("concat copy failed, re-encoding: %s", proc.stderr[-500:])
        cmd2 = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            str(output_path),
        ]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)
        if proc2.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {proc2.stderr[-500:]}")
    return output_path
