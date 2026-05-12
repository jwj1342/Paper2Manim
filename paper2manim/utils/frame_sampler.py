"""Sample N frames evenly from a Manim-rendered mp4 and assemble a montage PNG.

The output is one wide image with all frames laid side-by-side; the VLM gets a
single image and can describe motion / consistency across the scene without us
having to send N separate uploads.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class FrameSamplerError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or returns non-zero for a video."""


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise FrameSamplerError(f"Required tool '{tool}' not found on PATH.")
    return path


def _video_duration_seconds(video_path: Path) -> float:
    ffprobe = _require("ffprobe")
    out = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        raise FrameSamplerError(f"ffprobe failed for {video_path}: {out.stderr.strip()}")
    try:
        return float(out.stdout.strip())
    except ValueError as exc:
        raise FrameSamplerError(f"ffprobe returned non-numeric duration: {out.stdout!r}") from exc


def sample_frames_montage(
    video_path: Path | str,
    out_png: Path | str,
    *,
    n_frames: int = 4,
    frame_height: int = 240,
) -> Path:
    """Extract ``n_frames`` evenly spaced frames and concat them horizontally.

    Parameters
    ----------
    video_path
        Input mp4 (any container that ffmpeg can read).
    out_png
        Destination PNG path.
    n_frames
        Number of frames to sample. Must be >= 2.
    frame_height
        Resize each extracted frame to this height in pixels (aspect preserved).
        Keeps the montage compact for VLM context budgets.

    Returns
    -------
    pathlib.Path
        Path to ``out_png`` (also returned for convenient chaining).
    """
    video = Path(video_path)
    if not video.exists():
        raise FrameSamplerError(f"Video does not exist: {video}")
    if n_frames < 2:
        raise ValueError("n_frames must be >= 2.")

    ffmpeg = _require("ffmpeg")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    duration = _video_duration_seconds(video)
    if duration <= 0:
        raise FrameSamplerError(f"Video has non-positive duration: {duration}s")

    # Pick the timestamp at the midpoint of each evenly-divided segment so we
    # avoid the title fade-in at t=0 and the fade-out tail at t=duration.
    segment = duration / n_frames
    timestamps = [segment * (i + 0.5) for i in range(n_frames)]

    tmpdir = out_png.with_suffix("")
    tmpdir = tmpdir.parent / (tmpdir.name + "_frames")
    tmpdir.mkdir(exist_ok=True)

    frame_paths: list[Path] = []
    try:
        for i, t in enumerate(timestamps):
            frame_path = tmpdir / f"frame_{i:02d}.png"
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner", "-loglevel", "error",
                    "-ss", f"{t:.3f}",
                    "-i", str(video),
                    "-frames:v", "1",
                    "-vf", f"scale=-2:{frame_height}",
                    "-y",
                    str(frame_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0 or not frame_path.exists():
                raise FrameSamplerError(
                    f"ffmpeg failed to extract frame {i} at t={t:.2f}s: "
                    f"{result.stderr.strip()}"
                )
            frame_paths.append(frame_path)

        hstack_filter = (
            "".join(f"[{i}:v]" for i in range(len(frame_paths))) + f"hstack=inputs={len(frame_paths)}"
        )
        inputs: list[str] = []
        for fp in frame_paths:
            inputs.extend(["-i", str(fp)])
        merge = subprocess.run(
            [
                ffmpeg,
                "-hide_banner", "-loglevel", "error",
                *inputs,
                "-filter_complex", hstack_filter,
                "-y",
                str(out_png),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if merge.returncode != 0 or not out_png.exists():
            raise FrameSamplerError(f"ffmpeg hstack failed: {merge.stderr.strip()}")
        log.info("[frame_sampler] %s -> %s (n=%d)", video.name, out_png, n_frames)
        return out_png
    finally:
        for fp in frame_paths:
            try:
                fp.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass
