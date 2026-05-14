"""Shared CSV helpers for task-pool files (proposal §8.1).

Task CSVs in ``examples/datasets/*.csv`` ship with leading ``#`` schema
notes — both the validator and the experiment runner need to strip those
before handing the file to :class:`csv.DictReader` (which otherwise turns
the first ``#``-line into a single garbled fieldname).
"""

from __future__ import annotations

from pathlib import Path


def strip_comment_lines(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(raw)
    return out


__all__ = ["strip_comment_lines"]
