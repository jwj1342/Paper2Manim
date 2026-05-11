"""Classify Manim render errors and extract structured snippets for the LLM.

Three useful categories (per调研结论):
    python        — Python syntax/runtime error (SyntaxError, NameError, ImportError, ...)
    latex         — LatexError / "LaTeX compilation failed"
    manim_runtime — Mobject / VMobject / Renderer / OpenGL / Manim-specific
    timeout       — wall_timeout exceeded
    unknown       — fallback
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

ErrorCategory = Literal["python", "latex", "manim_runtime", "timeout", "unknown"]

_PYTHON_HINTS = (
    "SyntaxError",
    "IndentationError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "TypeError",
    "ValueError",
    "AttributeError",
    "KeyError",
    "ZeroDivisionError",
    "FileNotFoundError",
)
_LATEX_HINTS = ("LatexError", "LaTeX compilation failed", "latex.tex", "latex error")
_MANIM_HINTS = (
    "Mobject",
    "VMobject",
    "OpenGLError",
    "manim.utils",
    "manim.scene",
    "Scene.construct",
    "Animation",
    "Renderer",
    "no scene named",
    "is not in the scene",
)


def classify_error(stderr: str, returncode: int, *, timed_out: bool = False) -> ErrorCategory:
    if timed_out or returncode == -9 or "[TIMEOUT]" in (stderr or ""):
        return "timeout"
    s = stderr or ""
    if any(h in s for h in _LATEX_HINTS):
        return "latex"
    # Manim hints take precedence over generic Python hints when both fire
    # (e.g., AttributeError on VMobject is a manim runtime error, not a stray Python bug).
    if any(h in s for h in _MANIM_HINTS):
        return "manim_runtime"
    if any(h in s for h in _PYTHON_HINTS):
        return "python"
    if "Traceback" in s:
        return "python"
    return "unknown"


def extract_traceback_tail(stderr: str, n_lines: int = 30) -> str:
    """Return the last n non-empty lines from stderr (typical traceback location)."""
    if not stderr:
        return ""
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    return "\n".join(lines[-n_lines:])


_LINE_RE = re.compile(r'File ".*scene\.py", line (\d+)')


def find_error_line(stderr: str) -> int | None:
    """Try to find the source line in scene.py mentioned by the traceback."""
    if not stderr:
        return None
    matches = _LINE_RE.findall(stderr)
    if matches:
        return int(matches[-1])
    return None


def excerpt_source(code: str, line: int | None, *, context: int = 3) -> list[dict]:
    """Return code lines around `line` with 1-based line numbers."""
    if line is None or not code:
        return []
    lines = code.splitlines()
    lo = max(1, line - context)
    hi = min(len(lines), line + context)
    return [{"line": i, "code": lines[i - 1]} for i in range(lo, hi + 1)]


_TEX_BANG_RE = re.compile(r"^!.*$", re.MULTILINE)


def extract_tex_log_errors(workdir: Path, max_lines: int = 20) -> str:
    """Find all *.log under workdir/out/Tex/, grep '! ' lines, return joined."""
    out: list[str] = []
    tex_dirs = list(Path(workdir).glob("**/Tex"))
    for tex_dir in tex_dirs:
        for log in tex_dir.glob("*.log"):
            try:
                text = log.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in _TEX_BANG_RE.findall(text):
                out.append(m.strip())
                if len(out) >= max_lines:
                    break
            if len(out) >= max_lines:
                break
        if len(out) >= max_lines:
            break
    return "\n".join(out[:max_lines])


def first_error_line(stderr: str) -> str | None:
    """Find the most informative single error line, preferring the line starting with the last
    Python exception name, otherwise the first 'Error:' line."""
    if not stderr:
        return None
    lines = stderr.splitlines()
    for ln in reversed(lines):
        ln = ln.strip()
        if not ln:
            continue
        for h in _PYTHON_HINTS:
            if ln.startswith(h):
                return ln
    for ln in lines:
        if "Error" in ln or "error" in ln:
            return ln.strip()
    return None
