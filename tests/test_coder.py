"""Tests for the coder agent and its python-block extraction."""

from unittest.mock import MagicMock

from paper2manim.agents.coder import coder_node, extract_python_block


def test_extract_python_block_with_fence():
    raw = "Here is the code:\n```python\nfrom manim import *\nclass S(Scene):\n    pass\n```\nDone."
    code = extract_python_block(raw)
    assert code.startswith("from manim import *")
    assert "class S(Scene):" in code
    assert "Done." not in code


def test_extract_python_block_no_fence():
    raw = "from manim import *\nclass S(Scene):\n    pass\n"
    code = extract_python_block(raw)
    assert code.strip().startswith("from manim import *")


def test_coder_node_uses_storyboard(mock_llm, fake_storyboard, tmp_path):
    msg = MagicMock()
    msg.content = "```python\nfrom manim import *\nclass PythagorasIntro(Scene):\n    def construct(self):\n        self.wait(1)\n```"
    mock_llm.invoke.return_value = msg

    state = {
        "run_id": "test-run",
        "storyboard": fake_storyboard,
        "current_scene_idx": 0,
        "iter_count": 0,
    }
    out = coder_node(state)
    assert "current_code" in out
    assert "class PythagorasIntro(Scene):" in out["current_code"]


def test_coder_node_includes_error_feedback(mock_llm, fake_storyboard):
    captured: dict = {}

    def fake_invoke(messages):
        captured["msgs"] = messages
        m = MagicMock()
        m.content = "```python\nclass PythagorasIntro(Scene):\n    pass\n```"
        return m

    mock_llm.invoke.side_effect = fake_invoke
    state = {
        "run_id": "test-run",
        "storyboard": fake_storyboard,
        "current_scene_idx": 0,
        "iter_count": 1,
        "current_code": "broken",
        "error_feedback": {
            "decision": "retry",
            "hint": "Replace \\mathbbb with \\mathbb",
            "render_result": {
                "category": "latex",
                "error_message": "Undefined control sequence \\mathbbb",
                "scene": "PythagorasIntro",
                "traceback_tail": "LatexError",
                "source_excerpt": [{"line": 4, "code": "MathTex(r'\\mathbbb{R}')"}],
            },
        },
    }
    coder_node(state)
    user_msg = captured["msgs"][1][1]
    assert "Replace \\mathbbb" in user_msg
    assert "Previous code (failed)" in user_msg
