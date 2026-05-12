"""Input parsers: dispatch to arXiv source / Marker PDF / plain text."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, NamedTuple

log = logging.getLogger(__name__)


class ParsedInput(NamedTuple):
    text: str
    fmt: Literal["latex", "markdown"]
    source: str  # human-readable origin tag for trace.jsonl
    # Phase 0 multimodal assets — set-once by the dispatch fn, persisted by parser_node.
    # ``figures``: list of {source_name: str, pil_image: PIL.Image} pre-persist.
    # ``tables``:  list of {raw_md: str, fmt: "markdown"|"latex"}.
    # Required (no default) to avoid the NamedTuple mutable-default shared-state footgun;
    # callers pass `[]` explicitly when there are no assets.
    figures: list[dict]
    tables: list[dict]


# Markdown table: one or more consecutive lines that start AND end with `|`.
# Captures the header + separator + body as a single block.
_MD_TABLE_RE = re.compile(r"(?:^\|[^\n]*\|\s*\n){2,}", re.MULTILINE)
# LaTeX tabular env. DOTALL so multi-line bodies are captured; non-greedy.
_LATEX_TABLE_RE = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)
# Matches the column-spec block in `\begin{tabular}{lcc}` so we can strip it.
_LATEX_TABULAR_HEAD_RE = re.compile(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\s*\{[^}]*\}")
# LaTeX bookkeeping commands we drop inside tabular bodies (best-effort).
_LATEX_NOISE_RE = re.compile(r"\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\}|rule)\b")


_ESCAPED_PIPE = "\x00ESC_PIPE\x00"  # sentinel that won't appear in real markdown


def _split_md_row(line: str) -> list[str]:
    """Split a markdown table row by `|`, dropping the empty edge cells.

    Handles escaped pipes (``\\|``) by replacing them with a sentinel before the
    split and restoring them to literal ``|`` in each cell. NB: cells containing
    raw ``|`` inside inline code spans (e.g. ``a | b``) are NOT correctly handled
    — that would require a full markdown parser; in that case the caller should
    fall back to ``raw_md``.
    """
    safe = line.replace(r"\|", _ESCAPED_PIPE)
    parts = [c.strip().replace(_ESCAPED_PIPE, "|") for c in safe.split("|")]
    # `| a | b |` splits to ['', 'a', 'b', ''] — drop leading/trailing empties only
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def _is_md_separator(cells: list[str]) -> bool:
    """A markdown table separator row looks like `|---|---|` — cells are all dashes (and optional :)."""
    return bool(cells) and all(re.fullmatch(r":?-+:?", c) for c in cells)


def _parse_md_table_structure(raw: str) -> tuple[list[str] | None, list[list[str]] | None]:
    """Best-effort: return (header, rows) for a markdown table block, or (None, None)."""
    lines = [ln for ln in raw.splitlines() if ln.strip().startswith("|") and ln.strip().endswith("|")]
    if len(lines) < 2:
        return None, None
    header = _split_md_row(lines[0])
    body_start = 1
    if _is_md_separator(_split_md_row(lines[1])):
        body_start = 2
    rows = [_split_md_row(ln) for ln in lines[body_start:]]
    return header or None, rows or []


def _parse_latex_table_structure(raw: str) -> tuple[list[str] | None, list[list[str]] | None]:
    """Best-effort: return (header, rows) for a LaTeX tabular block, or (None, None)."""
    # Strip the env wrappers and column-spec
    body = _LATEX_TABULAR_HEAD_RE.sub("", raw)
    body = body.replace(r"\end{tabular}", "")
    body = _LATEX_NOISE_RE.sub("", body)
    # Split rows on `\\` (which in raw text is the two-char sequence backslash-backslash)
    row_strings = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
    if not row_strings:
        return None, None
    parsed = [[cell.strip() for cell in r.split("&")] for r in row_strings]
    header = parsed[0] if parsed else None
    rows = parsed[1:]
    return header or None, rows


def _extract_md_tables(text: str) -> list[dict]:
    out: list[dict] = []
    for m in _MD_TABLE_RE.finditer(text):
        raw = m.group(0)
        header, rows = _parse_md_table_structure(raw)
        out.append({"raw_md": raw, "fmt": "markdown", "header": header, "rows": rows})
    return out


def _extract_latex_tables(tex: str) -> list[dict]:
    out: list[dict] = []
    for m in _LATEX_TABLE_RE.finditer(tex):
        raw = m.group(0)
        header, rows = _parse_latex_table_structure(raw)
        out.append({"raw_md": raw, "fmt": "latex", "header": header, "rows": rows})
    return out


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
        # arXiv tarball path: tables come from \begin{tabular}; figures stay empty
        # for Phase 0 — \includegraphics{...} resolution against the tarball is
        # a separate follow-up.
        return ParsedInput(
            text=src.tex,
            fmt="latex",
            source=f"arxiv-src:{src.arxiv_id}",
            figures=[],
            tables=_extract_latex_tables(src.tex),
        )
    except SourceUnavailable as exc:
        log.warning("[parser] arXiv %s has no LaTeX source: %s", arxiv_id, exc)
        if not allow_pdf_fallback:
            raise
        with tempfile.TemporaryDirectory(prefix=f"arxiv_pdf_{arxiv_id.replace('/', '_')}_") as td:
            pdf_path = Path(td) / f"{arxiv_id.replace('/', '_')}.pdf"
            download_pdf(arxiv_id, pdf_path)
            md, images = _parse_pdf_with_marker(pdf_path)
            return ParsedInput(
                text=md,
                fmt="markdown",
                source=f"arxiv-pdf:{arxiv_id}",
                figures=_pack_marker_images(images),
                tables=_extract_md_tables(md),
            )


def parse_local_pdf(pdf_path: str | Path) -> ParsedInput:
    md, images = _parse_pdf_with_marker(pdf_path)
    return ParsedInput(
        text=md,
        fmt="markdown",
        source=f"pdf:{Path(pdf_path).name}",
        figures=_pack_marker_images(images),
        tables=_extract_md_tables(md),
    )


def _parse_pdf_with_marker(pdf_path: str | Path) -> tuple[str, dict[str, Any]]:
    # Local import: marker-pdf is an optional dep ([mvp2] extras).
    from paper2manim.parsers.marker import parse_pdf

    return parse_pdf(pdf_path)


def _pack_marker_images(images: dict[str, Any]) -> list[dict]:
    """Convert marker's ``{source_name: PIL.Image}`` into the ParsedInput.figures shape."""
    return [{"source_name": name, "pil_image": img} for name, img in images.items()]
