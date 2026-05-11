from __future__ import annotations

import json

from paper2manim.domain.models import ManimSceneCode, SceneSpec, VisualReviewResult
from paper2manim.infrastructure.llm.client import ChatMessage, LLMClient
from paper2manim.utils.prompt_loader import load_agent_prompt
from paper2manim.utils.text_utils import extract_python_code

VISUAL_REVISION_PROMPT = "visual_revision_agent.md"


class VisualRevisionAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def revise_scene_code(
        self,
        scene_spec: SceneSpec,
        current_code: str,
        visual_review: VisualReviewResult,
        attempt: int,
    ) -> ManimSceneCode:
        response = self.llm_client.complete(
            [
                ChatMessage(role="system", content=load_agent_prompt(VISUAL_REVISION_PROMPT)),
                ChatMessage(
                    role="user",
                    content="Revise this Manim scene based on visual review feedback.\n\n"
                    f"Required scene class name: {_scene_class_name(scene_spec)}\n\n"
                    "Scene spec:\n"
                    f"{json.dumps(scene_spec.to_dict(), ensure_ascii=False, indent=2)}\n\n"
                    "Visual review:\n"
                    f"{json.dumps(visual_review.to_dict(), ensure_ascii=False, indent=2)}\n\n"
                    f"Current code:\n{current_code[-9000:]}",
                ),
            ],
            temperature=0.1,
            max_tokens=5200,
        )
        scene_class_name = _scene_class_name(scene_spec)
        code = extract_python_code(response)
        if scene_class_name not in code and "Paper2ManimScene" in code:
            code = code.replace("Paper2ManimScene", scene_class_name)
        return ManimSceneCode(
            scene_id=scene_spec.scene_id,
            scene_class_name=scene_class_name,
            code=code,
            attempt=attempt,
            metadata={"agent": "VisualRevisionAgent"},
        )


def _scene_class_name(scene_spec: SceneSpec) -> str:
    return f"Scene{max(scene_spec.order, 1):03d}"
