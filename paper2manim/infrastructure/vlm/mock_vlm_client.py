from __future__ import annotations

from pathlib import Path

from paper2manim.infrastructure.vlm.client import VLMClient


class MockVLMClient(VLMClient):
    def __init__(self, result: str | None = None) -> None:
        self.result = result or _default_pass()

    def review_images(
        self,
        prompt: str,
        image_paths: list[Path],
        *,
        response_format: str = "json",
    ) -> str:
        return self.result

    def review_scene(self, prompt: str, image_path: str | Path) -> str:
        return self.result


def _default_pass() -> str:
    return (
        "{\n"
        "  \"scene_id\": \"scene_mock\",\n"
        "  \"decision\": \"pass\",\n"
        "  \"scores\": {\n"
        "    \"paper_alignment\": 4,\n"
        "    \"visual_clarity\": 4,\n"
        "    \"readability\": 4,\n"
        "    \"layout_balance\": 4,\n"
        "    \"visual_focus\": 4,\n"
        "    \"animation_perceived\": 4\n"
        "  },\n"
        "  \"issues\": [],\n"
        "  \"paper_alignment_notes\": \"Looks aligned.\",\n"
        "  \"revision_instruction\": \"\",\n"
        "  \"requires_replanning\": false\n"
        "}"
    )
