"""Unit tests for vlm_scene_reviewer.parse_vlm_response — the JSON extractor.

The original implementation used a greedy ``r"\\{.*\\}"`` regex which, when a
chatty VLM emitted two top-level JSON objects (e.g. a debug trace followed by
the real answer), matched from the first ``{`` of the first object to the last
``}`` of the second and handed json.loads malformed input. The balanced-brace
scanner introduced in :func:`_extract_first_json_object` should pick the first
valid object only.

These tests cover the proposal §4.2 canonical schema: 3 dimensions
(``logic_flow`` / ``layout_occlusion`` / ``accuracy``) on a 0–100 scale plus
the ≥ 90 average → auto-pass bypass.
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
        "Reasoning: the scene looks fine.\n"
        '{"scene_id": "S1", "decision": "pass", "scores": '
        '{"logic_flow": 85, "layout_occlusion": 80, "accuracy": 90}}\n'
        '{"unrelated": "trailing object the greedy regex would have swallowed"}'
    )
    result = parse_vlm_response(raw, "S1")
    assert result["decision"] == "pass"
    assert result["scores"]["logic_flow"] == 85
    assert result["scores"]["layout_occlusion"] == 80
    assert result["scores"]["accuracy"] == 90


def test_missing_score_dimensions_record_none_not_zero():
    """A dimension the VLM forgot must record None, not bottom-out at 0."""
    raw = '{"scene_id": "S1", "decision": "revise", "scores": {"logic_flow": 70}}'
    result = parse_vlm_response(raw, "S1")
    assert result["scores"]["logic_flow"] == 70
    for missing in ("layout_occlusion", "accuracy"):
        assert result["scores"][missing] is None, f"{missing} should be None when VLM omitted it"


def test_scores_clamped_to_0_100():
    """Out-of-range scores must be clamped, not propagated as-is."""
    raw = (
        '{"scene_id": "S1", "decision": "revise", '
        '"scores": {"logic_flow": 120, "layout_occlusion": -5, "accuracy": 50}}'
    )
    result = parse_vlm_response(raw, "S1")
    assert result["scores"]["logic_flow"] == 100
    assert result["scores"]["layout_occlusion"] == 0
    assert result["scores"]["accuracy"] == 50


def test_auto_pass_when_avg_at_or_above_threshold():
    """A revise decision with avg ≥ 90 is auto-upgraded to pass; raw kept for trace."""
    raw = (
        '{"scene_id": "S1", "decision": "revise", '
        '"scores": {"logic_flow": 92, "layout_occlusion": 90, "accuracy": 95}}'
    )
    result = parse_vlm_response(raw, "S1")
    assert result["decision"] == "pass"
    assert result["raw_decision"] == "revise"
    assert result["average_score"] > 90


def test_no_auto_pass_when_avg_below_threshold():
    """Revise stays revise when the average is below the bypass threshold."""
    raw = (
        '{"scene_id": "S1", "decision": "revise", '
        '"scores": {"logic_flow": 70, "layout_occlusion": 80, "accuracy": 85}}'
    )
    result = parse_vlm_response(raw, "S1")
    assert result["decision"] == "revise"
    assert result["raw_decision"] == "revise"


def test_no_auto_pass_when_scores_missing():
    """If the VLM omitted every score, we can't bypass — keep revise."""
    raw = '{"scene_id": "S1", "decision": "revise", "scores": {}}'
    result = parse_vlm_response(raw, "S1")
    assert result["decision"] == "revise"
    assert result["average_score"] is None
