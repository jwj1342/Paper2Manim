"""MVP 1.0: trivial text parser — accept either a literal string or a path to a .txt file."""

from __future__ import annotations

from pathlib import Path


def load_text(input_arg: str) -> str:
    """If `input_arg` looks like an existing file path, read it; otherwise treat as literal text."""
    p = Path(input_arg)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return input_arg
