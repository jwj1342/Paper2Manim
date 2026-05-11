"""Unit tests for paper2manim.parsers.arxiv_source.

Network is mocked — no real arXiv hits in this file.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from paper2manim.parsers.arxiv_source import (
    SourceUnavailable,
    extract_section,
    extract_to_tmp,
    find_main_tex,
    flatten_tex,
    parse_arxiv_id,
    strip_comments,
)

# ---- parse_arxiv_id ---------------------------------------------------------

@pytest.mark.parametrize(
    "spec, expected",
    [
        ("1706.03762", "1706.03762"),
        ("1706.03762v2", "1706.03762v2"),
        ("arXiv:1706.03762", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762v3", "1706.03762v3"),
        ("https://arxiv.org/pdf/2401.12345.pdf", "2401.12345"),
        ("hep-th/9901001", "hep-th/9901001"),
    ],
)
def test_parse_arxiv_id_variants(spec, expected):
    assert parse_arxiv_id(spec) == expected


def test_parse_arxiv_id_rejects_garbage():
    with pytest.raises(ValueError):
        parse_arxiv_id("not-an-arxiv-id")


# ---- flatten_tex ------------------------------------------------------------

def _write(p: Path, txt: str) -> Path:
    p.write_text(txt, encoding="utf-8")
    return p


def test_flatten_inlines_input_and_include(tmp_path):
    _write(tmp_path / "sec1.tex", "Hello from section 1.")
    _write(tmp_path / "sub" / "sec2.tex" if False else tmp_path / "sec2.tex", "Section 2 body.")
    main = _write(
        tmp_path / "main.tex",
        r"\documentclass{article}\begin{document}\input{sec1}\include{sec2}\end{document}",
    )
    out = flatten_tex(main)
    assert "Hello from section 1." in out
    assert "Section 2 body." in out
    assert r"\input{" not in out
    assert r"\include{" not in out


def test_flatten_handles_missing_include_gracefully(tmp_path):
    main = _write(tmp_path / "main.tex", r"\documentclass{article}\input{nope}rest")
    out = flatten_tex(main)
    assert "rest" in out  # don't crash; just drop the missing input


# ---- strip_comments + extract_section --------------------------------------

def test_strip_comments_keeps_escaped_percent():
    src = "real % comment to drop\nkeep \\% literal\n"
    out = strip_comments(src)
    assert "comment to drop" not in out
    assert "\\% literal" in out


def test_extract_section_picks_target_and_stops_at_next():
    src = (
        r"\section{Introduction}" "intro body. "
        r"\section{Method}" "method body. "
        r"\section{Experiments}" "exp body."
    )
    out = extract_section(src, "Method")
    assert "method body" in out
    assert "exp body" not in out
    assert "intro body" not in out


def test_extract_section_case_insensitive_substring():
    src = r"\section{Scaled Dot-Product Attention}" "core. " r"\section{Conclusion}" "end."
    out = extract_section(src, "dot-product")
    assert "core" in out and "end" not in out


def test_extract_section_returns_empty_when_missing():
    src = r"\section{Intro}" "x. " r"\section{Refs}" "y."
    assert extract_section(src, "Methodology") == ""


# ---- extract_to_tmp ---------------------------------------------------------

def _build_tar_gz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_extract_to_tmp_unpacks_tarball(tmp_path):
    blob = _build_tar_gz({
        "main.tex": b"\\documentclass{article}\\begin{document}body\\end{document}",
        "sec1.tex": b"section content",
    })
    extract_to_tmp(blob, tmp_path)
    assert (tmp_path / "main.tex").exists()
    assert (tmp_path / "sec1.tex").exists()


def test_extract_to_tmp_rejects_pdf_blob(tmp_path):
    pdf_blob = b"%PDF-1.4\n...binary..."
    with pytest.raises(SourceUnavailable, match="PDF directly"):
        extract_to_tmp(pdf_blob, tmp_path)


def test_find_main_tex_prefers_documentclass(tmp_path):
    (tmp_path / "appendix.tex").write_text("just appendix text, no documentclass.")
    (tmp_path / "main.tex").write_text(r"\documentclass{article}\begin{document}body\end{document}")
    assert find_main_tex(tmp_path).name == "main.tex"


def test_find_main_tex_raises_when_no_tex(tmp_path):
    with pytest.raises(SourceUnavailable):
        find_main_tex(tmp_path)
