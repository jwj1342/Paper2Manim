"""Unit tests for vlm_scene_reviewer.parse_vlm_response — the JSON extractor.

The original implementation used a greedy ``r"\\{.*\\}"`` regex which, when a
chatty VLM emitted two top-level JSON objects (e.g. a debug trace followed by
the real answer), matched from the first ``{`` of the first object to the last
``}`` of the second and handed json.loads malformed input. The balanced-brace
scanner introduced in :func:`_extract_first_json_object` should pick the first
valid object only.
"""

from __future__ import annotations

from paper2manim.agents.vlm_scene_reviewer import (
    _extract_first_json_object,
    parse_vlm_response,
)


def test_extract_first_of_two_objects():
    raw = '{"a": 1}\n\n{"b": 2}'
    blob = _extract_first_json_object(raw)
    assert blob == '{"a": 1}'


def test_extract_handles_nested_braces():
    raw = 'noise {"outer": {"inner": 1}, "k": "v"} tail'
    blob = _extract_first_json_object(raw)
    assert blob == '{"outer": {"inner": 1}, "k": "v"}'


def test_extract_ignores_braces_inside_strings():
    raw = '{"msg": "see { here", "n": 1}'
    blob = _extract_first_json_object(raw)
    assert blob == raw  # The whole thing is one valid object.


def test_extract_returns_none_when_unbalanced():
    assert _extract_first_json_object("{ no closer") is None
    assert _extract_first_json_object("") is None
    assert _extract_first_json_object("no braces at all") is None


def test_parse_vlm_response_picks_first_object_when_model_chats_twice():
    """End-to-end: two top-level JSON objects → parser uses the first one."""
    raw = (
        'Reasoning: the scene looks fine.\n'
        '{"scene_id": "S1", "decision": "pass", "scores": '
        '{"paper_alignment": 5, "visual_clarity": 5, "readability": 5, '
        '"layout_balance": 5, "visual_focus": 5, "animation_perceived": 5}}\n'
        '{"unrelated": "trailing object the greedy regex would have swallowed"}'
    )
    result = parse_vlm_response(raw, "S1")
    assert result["decision"] == "pass"
    assert result["scores"]["paper_alignment"] == 5
