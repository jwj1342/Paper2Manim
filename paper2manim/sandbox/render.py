"""Run Manim code in a subprocess sandbox with rlimits and capture structured errors."""

from __future__ import annotations

import logging
import os
import resource
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from paper2manim import concurrency
from paper2manim.quality.manim_static_checker import validate_manim_code
from paper2manim.sandbox.classify import (
    classify_error,
    excerpt_source,
    extract_tex_log_errors,
    extract_traceback_tail,
    find_error_line,
    first_error_line,
)

log = logging.getLogger(__name__)


def render(
    code: str,
    scene_name: str,
    *,
    quality: str = "l",
    wall_timeout: int = 180,
    cpu_seconds: int = 120,
    mem_mb: int = 4096,
    fsize_mb: int = 512,
    workdir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render a Manim scene in a sandboxed subprocess.

    Returns a dict matching state.RenderResult shape.
    """
    if quality not in {"l", "m", "h"}:
        raise ValueError(f"quality must be 'l'/'m'/'h', got {quality!r}")

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="manim_"))
    work.mkdir(parents=True, exist_ok=True)

    # Pre-flight static check (issue #1 D4): AST blacklist + structural validation.
    # If the LLM produced obviously dangerous or malformed Manim code, fail fast
    # without spinning up a subprocess.
    try:
        validate_manim_code(code)
    except (ValueError, SyntaxError) as exc:
        return {
            "status": "error",
            "category": "python",
            "exit_code": -1,
            "scene": scene_name,
            "video_path": None,
            "error_type": "StaticCheckError",
            "error_message": f"static check failed: {exc}",
            "traceback_tail": str(exc),
            "source_excerpt": None,
            "tex_log_excerpt": None,
            "workdir": str(work),
        }

    script = work / "scene.py"
    script.write_text(code, encoding="utf-8")
    out_dir = work / "out"
    out_dir.mkdir(exist_ok=True)

    cmd = [
        "manim",
        "render",
        f"-q{quality}",
        "--disable_caching",
        "--renderer=cairo",  # HPC headless; OpenGL would crash with no display
        "--media_dir",
        str(out_dir),
        str(script),
        scene_name,
    ]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # Avoid pulling user's local manim cache between runs
    env.setdefault("MANIM_DISABLE_CACHING", "1")

    def _limits() -> None:
        # CPU time (seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        # Address space (bytes); RLIMIT_AS may be too aggressive on some libs (LaTeX)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
        except (OSError, ValueError):
            pass
        # Max single-file size
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb * 1024 * 1024,) * 2)
        except (OSError, ValueError):
            pass

    timed_out = False
    # Acquire a render slot. Under --scene-parallelism > 1, this caps the
    # number of concurrent Manim subprocesses; in serial mode it's a no-op.
    with concurrency.render_slot():
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=wall_timeout,
                preexec_fn=_limits,
                cwd=work,
                env=env,
                check=False,
            )
            stderr = proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            stderr = stderr + "\n[TIMEOUT]"
            returncode = -1

    if not timed_out and returncode == 0:
        mp4 = _find_output_mp4(out_dir, scene_name)
        return {
            "status": "success",
            "category": None,
            "exit_code": 0,
            "scene": scene_name,
            "video_path": str(mp4) if mp4 else None,
            "error_type": None,
            "error_message": None,
            "traceback_tail": None,
            "source_excerpt": None,
            "tex_log_excerpt": None,
            "workdir": str(work),
        }

    cat = classify_error(stderr, returncode, timed_out=timed_out)
    line = find_error_line(stderr) if cat in {"python", "manim_runtime"} else None
    return {
        "status": "error",
        "category": cat,
        "exit_code": returncode,
        "scene": scene_name,
        "video_path": None,
        "error_type": _guess_error_type(stderr, cat),
        "error_message": first_error_line(stderr),
        "traceback_tail": extract_traceback_tail(stderr, n_lines=30),
        "source_excerpt": excerpt_source(code, line),
        "tex_log_excerpt": extract_tex_log_errors(work) if cat == "latex" else None,
        "workdir": str(work),
    }


def _find_output_mp4(out_dir: Path, scene_name: str) -> Path | None:
    """Manim writes to out_dir/videos/<scene_stem>/<quality>p<fps>/<SceneName>.mp4"""
    for p in out_dir.rglob(f"{scene_name}.mp4"):
        return p
    # fallback: any mp4
    for p in out_dir.rglob("*.mp4"):
        return p
    return None


def _guess_error_type(stderr: str, category: str) -> str | None:
    if not stderr:
        return None
    if category == "timeout":
        return "TimeoutExpired"
    if category == "latex":
        return "LatexError"
    # try to extract Python exception name from last line
    for ln in reversed(stderr.splitlines()):
        ln = ln.strip()
        if ":" in ln:
            head = ln.split(":", 1)[0].strip()
            if head and head[0].isupper() and head.replace("_", "").isalnum():
                return head
    return None
