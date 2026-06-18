"""Narration data models — scene-level spoken commentary for voiceover.

The narrator agent produces a :class:`NarrationPlanModel` containing one
:class:`SceneNarrationModel` per storyboard scene. These models are consumed
by the TTS layer (text → audio) and the AV assembly layer (alignment + mux).

Separate from the Manim code path: Manim scripts never read or write audio.
All audio work happens post-render in the ``assemble_av`` graph node.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SceneNarrationModel(BaseModel):
    """One scene's spoken narration.

    ``scene`` must match ``Scene.name`` from the storyboard exactly.
    ``text`` is spoken prose, not a visual description.
    ``target_duration_s`` is the initial duration estimate from the storyboard;
    the actual alignment step uses the real video duration from ffprobe.
    """

    scene: str = Field(..., description="Scene name matching storyboard Scene.name")
    text: str = Field(
        ..., min_length=1, max_length=2000, description="Spoken narration text"
    )
    target_duration_s: float = Field(
        ..., ge=1.0, le=120.0, description="Target speaking duration in seconds"
    )
    language: str = Field(
        ..., min_length=2, max_length=10, description="ISO language code (e.g. 'en', 'zh')"
    )
    voice: str | None = Field(
        default=None, description="TTS voice override; empty = use configured default"
    )


class NarrationPlanModel(BaseModel):
    """Complete narration plan for a video.

    One SceneNarrationModel per storyboard scene. Scene order is the
    order they appear in the ``scenes`` list.
    """

    title: str = Field(..., description="Narration title (typically matches the paper title)")
    scenes: list[SceneNarrationModel] = Field(
        ..., min_length=1, max_length=10, description="Per-scene narration entries"
    )
