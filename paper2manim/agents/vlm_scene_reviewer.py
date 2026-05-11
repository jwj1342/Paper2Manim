from __future__ import annotations

import json
from pathlib import Path

from paper2manim.domain.models import FrameSampleResult, SceneSpec, VisualReviewResult
from paper2manim.infrastructure.vlm.client import VLMClient
from paper2manim.utils.prompt_loader import load_prompt
from paper2manim.utils.text_utils import extract_json_object

VLM_REVIEWER_PROMPT = "vlm_scene_reviewer.md"


class VLMSceneReviewerAgent:
    def __init__(self, vlm_client: VLMClient) -> None:
        self.vlm_client = vlm_client

    def run(self, scene_spec: SceneSpec, montage_path: Path) -> VisualReviewResult:
        result, _ = self.run_with_raw(scene_spec, montage_path, attempt=1)
        return result

    def review_scene(
        self,
        scene_spec: SceneSpec,
        frame_sample: FrameSampleResult,
    ) -> VisualReviewResult:
        result, _ = self.run_with_raw(
            scene_spec,
            Path(frame_sample.montage_path),
            attempt=frame_sample.attempt,
        )
        return result

    def run_with_raw(
        self, scene_spec: SceneSpec, montage_path: Path, attempt: int = 1
    ) -> tuple[VisualReviewResult, str]:
        prompt = load_prompt(VLM_REVIEWER_PROMPT)
        payload = (
            f"SceneSpec JSON:\n{json.dumps(scene_spec.to_dict(), ensure_ascii=False, indent=2)}"
        )
        response = self.vlm_client.review_scene(
            f"{prompt}\n\n{payload}",
            montage_path,
        )
        try:
            result = VisualReviewResult.from_dict(extract_json_object(response))
            result = VisualReviewResult.from_dict(
                {**result.to_dict(), "attempt": attempt, "raw_response": response}
            )
        except Exception as exc:
            result = VisualReviewResult.from_dict(
                {
                    "scene_id": scene_spec.scene_id,
                    "attempt": attempt,
                    "decision": "revise",
                    "scores": {
                        "paper_alignment": 1,
                        "visual_clarity": 1,
                        "readability": 1,
                        "layout_balance": 1,
                        "visual_focus": 1,
                        "animation_perceived": 1,
                    },
                    "issues": [
                        {
                            "type": "other",
                            "severity": "high",
                            "evidence": f"Could not parse VLM JSON: {exc}",
                            "suggestion": "Retry visual review or inspect montage manually.",
                        }
                    ],
                    "paper_alignment_notes": "VLM response could not be parsed.",
                    "revision_instruction": "Make conservative layout fixes: reduce text density, enlarge main visual, and increase spacing.",
                    "requires_replanning": False,
                    "raw_response": response,
                }
            )
        return result, response
