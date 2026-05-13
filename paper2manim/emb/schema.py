"""Pydantic models for EMB records.

A :class:`MemoryRecord` is a polarity-tagged envelope around two pieces:

* ``context``: shared retrieval head — task text + embedding + scene metadata.
* ``body``: polarity-specific payload — ``SuccessBody`` for §4.4a entries,
  ``FailureBody`` for §4.4b entries (a validated reflection Lesson).

Both polarities share the same ``Provenance`` schema so trace / hit-count /
freshness queries are uniform.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Polarity = Literal["success", "failure"]


# ---- Context (shared by both polarities) ----


class Context(BaseModel):
    """Retrieval head — same shape across success and failure records.

    ``task_embedding`` is materialized by the embedder at write time; readers
    should treat it as opaque vector data. ``source_paper`` / ``source_section``
    let us trace each record back to its arXiv origin and ablate by domain.
    """

    model_config = ConfigDict(extra="forbid")

    task_text: str = Field(..., description="Scene description / SceneSpec text")
    task_embedding: list[float] = Field(default_factory=list, description="Dense vector")
    scene_role: str = Field(
        default="unknown",
        description="Coarse scene category: background / method / experiment / conclusion / unknown",
    )
    domain_tags: list[str] = Field(default_factory=list)
    source_paper: str = Field(default="", description="e.g. 'arxiv:1706.03762'")
    source_section: str = Field(default="", description="e.g. 'Background'")


# ---- Body variants ----


class SuccessBody(BaseModel):
    """§4.4a positive consolidation — Rationale + full code + frame hash."""

    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(..., description="High-Score Rationale (LLM/VLM-written)")
    code_full: str = Field(..., description="Final Manim source that scored ≥ θ_high")
    code_keypoints: list[str] = Field(
        default_factory=list,
        description="Optional bullet-list of reusable patterns extracted from code_full",
    )
    frame_hash: str = Field(default="", description="Perceptual / SHA-256 hash of final montage")


class FailureBody(BaseModel):
    """§4.4b negative consolidation — a validated Lesson.

    A ``FailureBody`` is only created from transitions where ``after_score >
    before_score`` (see :class:`Provenance`), so anti / good examples are
    grounded in an actually-observed improvement.
    """

    model_config = ConfigDict(extra="forbid")

    trigger_pattern: str = Field(..., description="When this lesson applies (natural-language)")
    root_cause: str = Field(..., description="What was wrong")
    fix_recipe: str = Field(..., description="How to fix (natural-language)")
    code_anti_example: str = Field(default="", description="Minimal failing fragment")
    code_good_example: str = Field(default="", description="Minimal repaired fragment")
    vlm_diagnostic: str = Field(
        default="", description="Original VLM diagnostic or traceback signature"
    )


# ---- Provenance ----


ExtractionSource = Literal[
    "text_reflection",
    "visual_reflection",
    "high_score_scene",
    "manual",
]


class Provenance(BaseModel):
    """Where this record came from and how it's been used since.

    The ``validated`` flag is the EMB-quality gate for ``failure`` records:
    set to ``True`` only after the writer confirms ``after_score >
    before_score``. For ``success`` records we set it to ``True`` once the
    final scene clears ``θ_high``.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default="", description="paper2manim run that produced this record")
    scene_id: str = Field(default="", description="Scene name within the run")
    extraction_source: ExtractionSource = Field(default="manual")
    transition_ordinal: int = Field(
        default=0,
        ge=0,
        description=(
            "Within-scene ordering for failure transitions so that v0→v1 and "
            "v1→v2 of the same scene don't collide on the dedup key. "
            "Visual: before_v_rev. Text: after_iter. Success records: 0."
        ),
    )
    validated: bool = Field(default=False)
    before_score: float | None = Field(default=None, description="Failure: low-score version")
    after_score: float | None = Field(default=None, description="Failure: high-score version")
    vlm_score: float | None = Field(default=None, description="Success: final VLM score")
    hit_count: int = Field(default=0, ge=0)
    first_seen: float = Field(default_factory=time.time)
    last_used: float | None = Field(default=None)


# ---- Top-level record ----


def _make_record_id() -> str:
    return uuid.uuid4().hex


class MemoryRecord(BaseModel):
    """Polarity-tagged record stored in the EMB."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_make_record_id)
    polarity: Polarity
    context: Context
    body: SuccessBody | FailureBody
    provenance: Provenance = Field(default_factory=Provenance)

    def model_post_init(self, __ctx) -> None:  # noqa: D401
        # Body / polarity coherence check at construction.
        if self.polarity == "success" and not isinstance(self.body, SuccessBody):
            raise ValueError("polarity=success requires body=SuccessBody")
        if self.polarity == "failure" and not isinstance(self.body, FailureBody):
            raise ValueError("polarity=failure requires body=FailureBody")
