"""arXiv source-tarball parser for MVP 2.0.

Strategy: pull the author-uploaded LaTeX source from arxiv.org/e-print/<id>
(usually a gzipped tarball; sometimes a single .tex or a single .pdf when the
author only uploaded a PDF). Flatten `\\input` / `\\include`, strip preamble
clutter, optionally slice out one `\\section{...}`, and return a single string
suitable for the Summarizer.

Falls back via raising ``SourceUnavailable`` so the dispatcher can route to
Marker on the corresponding PDF.
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import NamedTuple

import requests

log = logging.getLogger(__name__)

_ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{id}"
_ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}.pdf"
_DEFAULT_TIMEOUT = 60

# Accept: "1706.03762", "1706.03762v2", "arxiv.org/abs/1706.03762",
#         "https://arxiv.org/abs/1706.03762v2", "arXiv:1706.03762"
_ID_PATTERNS = [
    re.compile(r"arxiv\.org/(?:abs|pdf|e-print)/(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)", re.I),
    re.compile(r"arxiv[:\s]*?(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)", re.I),
    re.compile(r"^(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)$"),
    # pre-2007 style: hep-th/0701001 etc. — accept letters/dot/slash
    re.compile(r"^(?P<id>[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$", re.I),
]


class SourceUnavailable(RuntimeError):
    """Author did not upload LaTeX source (only a PDF), or arXiv returned non-source bytes."""


class ArxivSource(NamedTuple):
    arxiv_id: str
    tex: str
    main_tex_name: str


def parse_arxiv_id(spec: str) -> str:
    """Normalize various user inputs to a bare arXiv id (e.g. '1706.03762v2')."""
    spec = spec.strip()
    for pat in _ID_PATTERNS:
        m = pat.search(spec)
        if m:
            return m.group("id")
    raise ValueError(f"Could not parse arXiv id from: {spec!r}")


def download_eprint(arxiv_id: str, *, timeout: int = _DEFAULT_TIMEOUT) -> bytes:
    url = _ARXIV_EPRINT_URL.format(id=arxiv_id)
    log.info("[arxiv] GET %s", url)
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "paper2manim/0.1"})
    r.raise_for_status()
    return r.content


def download_pdf(arxiv_id: str, dest: Path, *, timeout: int = _DEFAULT_TIMEOUT) -> Path:
    url = _ARXIV_PDF_URL.format(id=arxiv_id)
    log.info("[arxiv] GET %s", url)
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "paper2manim/0.1"})
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _looks_like_pdf(blob: bytes) -> bool:
    return blob[:5] == b"%PDF-"


def extract_to_tmp(blob: bytes, dest: Path) -> Path:
    """Extract arXiv e-print bytes into ``dest`` directory. Returns the directory.

    Handles three observed shapes:
    - gzipped tarball (most common multi-file source)
    - gzipped single .tex (no tar)
    - gzipped pdf (author uploaded only PDF) -> SourceUnavailable
    """
    dest.mkdir(parents=True, exist_ok=True)
    if _looks_like_pdf(blob):
        raise SourceUnavailable("arxiv returned a PDF directly — no LaTeX source")
    # Try as tar.gz first
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            tar.extractall(dest)
            return dest
    except tarfile.TarError:
        pass
    # Try plain gz (single-file source)
    try:
        decompressed = gzip.decompress(blob)
    except OSError as e:
        raise SourceUnavailable(f"could not decompress arxiv blob: {e}") from e
    if _looks_like_pdf(decompressed):
        raise SourceUnavailable("arxiv source decompressed to a PDF — no LaTeX")
    # Heuristic: single .tex stream
    (dest / "main.tex").write_bytes(decompressed)
    return dest


def find_main_tex(root: Path) -> Path:
    """Pick the most plausible top-level .tex file.

    Strategy: prefer a file containing ``\\documentclass`` (and not a `.sty`/`.cls`).
    Fall back to any *.tex; raise if none.
    """
    candidates = [p for p in root.rglob("*.tex") if not p.name.endswith((".sty", ".cls"))]
    if not candidates:
        raise SourceUnavailable(f"no .tex files in extracted source ({root})")
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if r"\documentclass" in text:
            return p
    # No \documentclass found anywhere — pick the largest .tex as a last resort
    return max(candidates, key=lambda p: p.stat().st_size)


_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def flatten_tex(main: Path, *, max_depth: int = 6) -> str:
    """Recursively inline ``\\input{...}`` and ``\\include{...}``. Comments stripped only
    on the very outermost pass to avoid removing in-content `%` references unfairly."""
    root = main.parent

    def _resolve(name: str, base: Path) -> Path | None:
        name = name.strip()
        cands = [
            base / name,
            base / f"{name}.tex",
            root / name,
            root / f"{name}.tex",
        ]
        for c in cands:
            if c.is_file():
                return c
        return None

    def _expand(path: Path, depth: int) -> str:
        if depth > max_depth:
            return ""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.warning("[arxiv] could not read %s: %s", path, e)
            return ""

        def sub(match: re.Match) -> str:
            target = _resolve(match.group(1), path.parent)
            if target is None:
                log.debug("[arxiv] could not resolve \\input{%s} from %s", match.group(1), path)
                return ""
            return "\n" + _expand(target, depth + 1) + "\n"

        return _INPUT_RE.sub(sub, text)

    return _expand(main, 0)


# Strip LaTeX comments (a single `%` not preceded by backslash). Keeps `\%` literal.
_COMMENT_RE = re.compile(r"(?<!\\)%.*?$", re.MULTILINE)


def strip_comments(tex: str) -> str:
    return _COMMENT_RE.sub("", tex)


# Match `\section{...}` or `\section*{...}`; group 1 is the title text
_SECTION_HEAD_RE = re.compile(r"\\section\*?\s*\{([^}]+)\}")


def extract_section(tex: str, name: str) -> str:
    """Return the text from the first ``\\section{...}`` whose title contains ``name``
    (case-insensitive substring) up to the next ``\\section``. If not found, returns ''."""
    target = name.strip().lower()
    matches = list(_SECTION_HEAD_RE.finditer(tex))
    for i, m in enumerate(matches):
        if target in m.group(1).lower():
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(tex)
            return tex[start:end]
    return ""


def fetch_arxiv_source(
    spec: str,
    *,
    workdir: Path | None = None,
    section: str | None = None,
) -> ArxivSource:
    """High-level: fetch source for ``spec``, flatten, optionally slice a section.

    Raises ``SourceUnavailable`` if the author only uploaded a PDF (caller should
    fall back to ``parsers.marker.parse_pdf`` on the PDF URL).
    """
    arxiv_id = parse_arxiv_id(spec)
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix=f"arxiv_{arxiv_id.replace('/', '_')}_"))
    blob = download_eprint(arxiv_id)
    extract_to_tmp(blob, workdir)
    main = find_main_tex(workdir)
    flat = flatten_tex(main)
    flat = strip_comments(flat)
    if section:
        sliced = extract_section(flat, section)
        if not sliced:
            log.warning(
                "[arxiv] section %r not found in %s — keeping full document", section, arxiv_id
            )
        else:
            flat = sliced
    log.info("[arxiv] %s flattened to %d chars", arxiv_id, len(flat))
    return ArxivSource(arxiv_id=arxiv_id, tex=flat, main_tex_name=main.name)


def cleanup_workdir(p: Path) -> None:
    """Best-effort cleanup for caller-managed scratch dirs."""
    try:
        shutil.rmtree(p)
    except OSError:
        pass
