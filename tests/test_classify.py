"""Tests for sandbox.classify error categorization."""

from paper2manim.sandbox.classify import (
    classify_error,
    excerpt_source,
    extract_traceback_tail,
    find_error_line,
    first_error_line,
)


def test_classify_python():
    err = 'Traceback (most recent call last):\n  File "scene.py", line 5\n    x =\nSyntaxError: invalid syntax'
    assert classify_error(err, returncode=1) == "python"


def test_classify_latex():
    err = "manim.utils.tex_file_writing.TeX compilation failed\nLatexError: ! Undefined control sequence \\mathbbb"
    assert classify_error(err, returncode=1) == "latex"


def test_classify_manim_runtime():
    err = "AttributeError: 'VMobject' object has no attribute 'center'"
    assert classify_error(err, returncode=1) == "manim_runtime"


def test_classify_timeout_marker():
    assert classify_error("anything\n[TIMEOUT]", returncode=-1, timed_out=True) == "timeout"


def test_classify_unknown_when_clean():
    assert classify_error("just some random output", returncode=2) == "unknown"


def test_traceback_tail_trims():
    err = "\n".join(f"line {i}" for i in range(100))
    tail = extract_traceback_tail(err, n_lines=5)
    assert tail.splitlines() == [f"line {i}" for i in range(95, 100)]


def test_find_error_line_picks_last_in_scene_py():
    err = '  File "scene.py", line 12, in construct\n  File "scene.py", line 33, in helper'
    assert find_error_line(err) == 33


def test_excerpt_source_window():
    code = "\n".join(f"line{i}" for i in range(1, 11))
    out = excerpt_source(code, line=5, context=2)
    nums = [item["line"] for item in out]
    assert nums == [3, 4, 5, 6, 7]


def test_first_error_line_picks_python_exception():
    err = "Traceback (most recent call last):\n  ...\nNameError: name 'foo' is not defined"
    assert first_error_line(err) == "NameError: name 'foo' is not defined"


# ---- Static checker integration (issue #1 D4) ----

def test_render_blocks_forbidden_import():
    """Pre-flight static check rejects code that imports `os`."""
    from paper2manim.sandbox.render import render

    bad = (
        "from manim import *\n"
        "import os\n"
        "class S(Scene):\n"
        "    def construct(self):\n"
        "        os.system('rm -rf /')\n"
        "        self.play(Create(Circle()))\n"
        "        self.play(FadeOut(Circle()))\n"
    )
    result = render(bad, "S")
    assert result["status"] == "error"
    assert result["error_type"] == "StaticCheckError"
    assert "static check" in (result["error_message"] or "").lower()


def test_render_allows_tex_and_mathtex():
    """Per issue #1 D1: Tex / MathTex must NOT be blocked."""
    from paper2manim.quality.manim_static_checker import validate_manim_code

    code = (
        "from manim import *\n"
        "class S(Scene):\n"
        "    def construct(self):\n"
        "        eq = MathTex(r'a^2 + b^2 = c^2')\n"
        "        self.play(Write(eq))\n"
        "        self.play(FadeOut(eq))\n"
    )
    validate_manim_code(code)  # should not raise


def test_render_blocks_paragraph():
    """Per issue #1 D1: Paragraph stays blocked (wall-of-text scenes)."""
    import pytest

    from paper2manim.quality.manim_static_checker import validate_manim_code

    code = (
        "from manim import *\n"
        "class S(Scene):\n"
        "    def construct(self):\n"
        "        p = Paragraph('line1', 'line2')\n"
        "        self.play(Write(p))\n"
        "        self.play(FadeOut(p))\n"
    )
    with pytest.raises(ValueError, match="Paragraph"):
        validate_manim_code(code)
