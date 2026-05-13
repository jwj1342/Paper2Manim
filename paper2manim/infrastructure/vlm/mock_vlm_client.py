"""Mock VLM client for unit tests — returns deterministic 3-dim JSON."""

from __future__ import annotations

import json
from pathlib import Path


class MockVLMClient:
    """Returns a canned 3-dim review on the 0–100 scale.

    ``scripted`` overrides specific scene IDs with their own decision/scores.
    Anything not in ``scripted`` falls back to ``default_decision`` /
    ``default_score`` for all three dimensions.
    """

    def __init__(
        self,
        *,
        default_decision: str = "pass",
        default_score: int = 80,
        scripted: dict[str, dict] | None = None,
    ) -> None:
        self.default_decision = default_decision
        self.default_score = default_score
        self.scripted = scripted or {}
        self.calls: list[tuple[str, str]] = []

    def review_scene(self, prompt: str, image_path: str | Path) -> str:
        self.calls.append((prompt[:80], str(image_path)))
        scene_id = _extract_scene_id(prompt)
        scripted = self.scripted.get(scene_id, {})
        decision = scripted.get("decision", self.default_decision)
        score = scripted.get("score", self.default_score)
        return json.dumps(
            {
                "scene_id": scene_id,
                "decision": decision,
                "scores": {
                    "logic_flow": score,
                    "layout_occlusion": score,
                    "accuracy": score,
                },
                "issues": scripted.get("issues", []),
                "paper_alignment_notes": scripted.get("notes", "mock"),
                "revision_instruction": scripted.get(
                    "revision_instruction",
                    "" if decision == "pass" else "Increase font size and reduce overlap.",
                ),
                "requires_replanning": False,
            }
        )


def _extract_scene_id(prompt: str) -> str:
    """Best-effort scrape of ``"scene_id": "..."`` from the inlined SceneSpec JSON."""
    needle = '"scene_id"'
    idx = prompt.find(needle)
    if idx < 0:
        return "unknown"
    rest = prompt[idx + len(needle) :]
    q1 = rest.find('"')
    if q1 < 0:
        return "unknown"
    q2 = rest.find('"', q1 + 1)
    if q2 < 0:
        return "unknown"
    return rest[q1 + 1 : q2]
