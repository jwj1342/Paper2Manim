from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderResult:
    attempted: bool
    success: bool
    command: list[str]
    log_path: Path | None
    returncode: int | None
    video_path: Path | None = None
    stdout: str | None = None
    stderr: str | None = None
    error_message: str | None = None
    duration_seconds: float | None = None
    attempt: int = 1
    scene_id: str | None = None


def render_scene(
    scene_path: Path,
    output_dir: Path,
    quality: str = "ql",
    *,
    scene_class_name: str = "Paper2ManimScene",
    output_filename: str = "paper2manim_scene.mp4",
    log_filename: str = "render.log",
    timeout: int = 180,
    attempt: int = 1,
    scene_id: str | None = None,
) -> RenderResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_filename
    command = [
        sys.executable,
        "-m",
        "manim",
        f"-{quality}",
        str(scene_path.resolve()),
        scene_class_name,
        "-o",
        output_filename,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=output_dir.resolve(),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        log_path.write_text(stdout + "\n\n--- STDERR ---\n" + stderr, encoding="utf-8")
        return RenderResult(
            attempted=True,
            success=False,
            command=command,
            log_path=log_path,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            error_message=f"Manim render timed out after {timeout} seconds.",
            attempt=attempt,
            scene_id=scene_id,
        )
    log_path.write_text(
        completed.stdout + "\n\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    video_path = _find_rendered_video(output_dir, output_filename)
    if completed.returncode == 0:
        _cleanup_manim_cache(output_dir)

    return RenderResult(
        attempted=True,
        success=completed.returncode == 0,
        command=command,
        log_path=log_path,
        returncode=completed.returncode,
        video_path=video_path,
        stdout=completed.stdout,
        stderr=completed.stderr,
        error_message=None if completed.returncode == 0 else completed.stderr.strip(),
        attempt=attempt,
        scene_id=scene_id,
    )


def _find_rendered_video(output_dir: Path, filename: str) -> Path | None:
    matches = sorted((output_dir / "media").rglob(filename))
    return matches[0] if matches else None


def _cleanup_manim_cache(output_dir: Path) -> None:
    media_dir = output_dir / "media"
    if not media_dir.exists():
        return

    for cache_dir in ("Tex", "texts", "images"):
        shutil.rmtree(media_dir / cache_dir, ignore_errors=True)

    for partial_dir in media_dir.rglob("partial_movie_files"):
        shutil.rmtree(partial_dir, ignore_errors=True)

    directories = sorted(
        media_dir.rglob("*"),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass
