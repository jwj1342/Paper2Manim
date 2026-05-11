"""Input parsers: dispatch to arXiv source / Marker PDF / plain text."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Literal, NamedTuple

log = logging.getLogger(__name__)


class ParsedInput(NamedTuple):
    text: str
    fmt: Literal["latex", "markdown"]
    source: str  # human-readable origin tag for trace.jsonl


def parse_arxiv(
    spec: str, *, section: str | None = None, allow_pdf_fallback: bool = True
) -> ParsedInput:
    """arXiv-first: try LaTeX source; if author only uploaded PDF, fall back to Marker.

    ``section`` is a case-insensitive substring of a ``\\section{...}`` title; if set
    and the source path is used, only that section is returned.
    """
    from paper2manim.parsers.arxiv_source import (
        SourceUnavailable,
        download_pdf,
        fetch_arxiv_source,
        parse_arxiv_id,
    )

    arxiv_id = parse_arxiv_id(spec)
    try:
        src = fetch_arxiv_source(spec, section=section)
        return ParsedInput(text=src.tex, fmt="latex", source=f"arxiv-src:{src.arxiv_id}")
    except SourceUnavailable as exc:
        log.warning("[parser] arXiv %s has no LaTeX source: %s", arxiv_id, exc)
        if not allow_pdf_fallback:
            raise
        with tempfile.TemporaryDirectory(prefix=f"arxiv_pdf_{arxiv_id.replace('/', '_')}_") as td:
            pdf_path = Path(td) / f"{arxiv_id.replace('/', '_')}.pdf"
            download_pdf(arxiv_id, pdf_path)
            md = _parse_pdf_with_marker(pdf_path)
            return ParsedInput(text=md, fmt="markdown", source=f"arxiv-pdf:{arxiv_id}")


def parse_local_pdf(pdf_path: str | Path) -> ParsedInput:
    md = _parse_pdf_with_marker(pdf_path)
    return ParsedInput(text=md, fmt="markdown", source=f"pdf:{Path(pdf_path).name}")


def _parse_pdf_with_marker(pdf_path: str | Path) -> str:
    # Local import: marker-pdf is an optional dep ([mvp2] extras).
    from paper2manim.parsers.marker import parse_pdf

    return parse_pdf(pdf_path)
