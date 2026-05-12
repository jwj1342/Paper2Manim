"""Unit tests for table extraction helpers in paper2manim.parsers.

No external deps (no marker, no network) — exercises only the regex helpers.
"""

from __future__ import annotations

from paper2manim.parsers import (
    _extract_latex_tables,
    _extract_md_tables,
    _parse_latex_table_structure,
    _parse_md_table_structure,
)


# ---- markdown tables --------------------------------------------------------

def test_md_table_extraction_basic():
    text = """Some intro prose.

| col1 | col2 |
|------|------|
| a    | b    |
| c    | d    |

Trailing prose.
"""
    tables = _extract_md_tables(text)
    assert len(tables) == 1
    assert tables[0]["fmt"] == "markdown"
    assert "| col1 | col2 |" in tables[0]["raw_md"]
    assert "| c    | d    |" in tables[0]["raw_md"]


def test_md_table_extraction_multiple():
    text = """First:

| a | b |
|---|---|
| 1 | 2 |

Some prose between tables.

| x | y | z |
|---|---|---|
| 9 | 8 | 7 |
| 6 | 5 | 4 |
"""
    tables = _extract_md_tables(text)
    assert len(tables) == 2
    assert all(t["fmt"] == "markdown" for t in tables)
    assert "| a | b |" in tables[0]["raw_md"]
    assert "| x | y | z |" in tables[1]["raw_md"]


def test_md_table_extraction_no_table():
    text = "Just some prose with a stray | pipe | inside one line.\n\nAnd another paragraph.\n"
    assert _extract_md_tables(text) == []


# ---- latex tables -----------------------------------------------------------

def test_latex_table_extraction_basic():
    tex = r"""
\section{Results}
We report numbers below.

\begin{table}[h]
\centering
\caption{Main results}
\begin{tabular}{lcc}
\hline
Method & Acc & F1 \\
\hline
A & 0.9 & 0.88 \\
B & 0.7 & 0.71 \\
\hline
\end{tabular}
\end{table}

More prose.
"""
    tables = _extract_latex_tables(tex)
    assert len(tables) == 1
    assert tables[0]["fmt"] == "latex"
    raw = tables[0]["raw_md"]
    assert raw.startswith(r"\begin{tabular}")
    assert raw.endswith(r"\end{tabular}")
    assert "Method & Acc & F1" in raw


def test_latex_table_extraction_nested_environments():
    """An equation env before/after must not be swallowed into the tabular match."""
    tex = r"""
\begin{equation}
E = mc^2
\end{equation}

\begin{tabular}{ll}
key & value \\
\end{tabular}

\begin{equation}
F = ma
\end{equation}
"""
    tables = _extract_latex_tables(tex)
    assert len(tables) == 1
    raw = tables[0]["raw_md"]
    assert raw.startswith(r"\begin{tabular}")
    assert raw.endswith(r"\end{tabular}")
    # Critical: the regex must NOT extend into the second equation env
    assert "E = mc^2" not in raw
    assert "F = ma" not in raw


# ---- structured parsing (Phase 1) -------------------------------------------

def test_md_table_structure_basic():
    raw = "| col1 | col2 |\n|------|------|\n| a    | b    |\n| c    | d    |\n"
    header, rows = _parse_md_table_structure(raw)
    assert header == ["col1", "col2"]
    assert rows == [["a", "b"], ["c", "d"]]


def test_md_table_structure_no_separator():
    """Some tables in the wild skip the separator row — we still want a header + rows."""
    raw = "| a | b |\n| 1 | 2 |\n"
    header, rows = _parse_md_table_structure(raw)
    assert header == ["a", "b"]
    assert rows == [["1", "2"]]


def test_md_table_structure_with_alignment_separator():
    raw = "| name | score |\n|:-----|------:|\n| foo  | 1     |\n"
    header, rows = _parse_md_table_structure(raw)
    assert header == ["name", "score"]
    assert rows == [["foo", "1"]]


def test_latex_table_structure_basic():
    raw = r"""\begin{tabular}{lcc}
\hline
Method & Acc & F1 \\
\hline
A & 0.9 & 0.88 \\
B & 0.7 & 0.71 \\
\hline
\end{tabular}"""
    header, rows = _parse_latex_table_structure(raw)
    assert header == ["Method", "Acc", "F1"]
    assert rows == [["A", "0.9", "0.88"], ["B", "0.7", "0.71"]]


def test_latex_table_structure_no_hline():
    raw = r"""\begin{tabular}{ll}
key & value \\
foo & 1 \\
\end{tabular}"""
    header, rows = _parse_latex_table_structure(raw)
    assert header == ["key", "value"]
    assert rows == [["foo", "1"]]


def test_extract_md_tables_includes_structure():
    text = "intro\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nouttro\n"
    tables = _extract_md_tables(text)
    assert len(tables) == 1
    assert tables[0]["header"] == ["a", "b"]
    assert tables[0]["rows"] == [["1", "2"]]


def test_extract_latex_tables_includes_structure():
    tex = r"\begin{tabular}{ll}" + "\n" + r"k & v \\" + "\n" + r"x & y \\" + "\n" + r"\end{tabular}"
    tables = _extract_latex_tables(tex)
    assert len(tables) == 1
    assert tables[0]["header"] == ["k", "v"]
    assert tables[0]["rows"] == [["x", "y"]]
