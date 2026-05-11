from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper2manim.infrastructure.rendering.manim_renderer import RenderResult


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    text: str
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "char_count": self.char_count,
        }


@dataclass(frozen=True)
class DocumentTable:
    table_id: str
    page_number: int | None = None
    markdown: str | None = None
    html: str | None = None
    csv_path: str | None = None
    caption: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "markdown": self.markdown,
            "html": self.html,
            "csv_path": self.csv_path,
            "caption": self.caption,
        }


@dataclass(frozen=True)
class DocumentFigure:
    figure_id: str
    page_number: int | None = None
    image_path: str | None = None
    caption: str | None = None
    nearby_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "page_number": self.page_number,
            "image_path": self.image_path,
            "caption": self.caption,
            "nearby_text": self.nearby_text,
        }


@dataclass(frozen=True)
class PaperDocument:
    source_path: Path | str
    source_type: str = "pdf"
    markdown_text: str = ""
    plain_text: str = ""
    json_doc: dict[str, Any] | None = None
    pages: list[DocumentPage] = field(default_factory=list)
    tables: list[DocumentTable] = field(default_factory=list)
    figures: list[DocumentFigure] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_type": self.source_type,
            "markdown_text": self.markdown_text,
            "plain_text": self.plain_text,
            "json_doc": self.json_doc,
            "pages": [page.to_dict() for page in self.pages],
            "tables": [table.to_dict() for table in self.tables],
            "figures": [figure.to_dict() for figure in self.figures],
            "metadata": self.metadata,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class PaperContextPackage:
    user_prompt: str
    source_path: str
    title: str | None
    context_markdown: str
    main_text: str
    tables_text: str
    figures_text: str
    metadata_text: str
    warnings_text: str
    token_estimate: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_prompt": self.user_prompt,
            "source_path": self.source_path,
            "title": self.title,
            "context_markdown": self.context_markdown,
            "main_text": self.main_text,
            "tables_text": self.tables_text,
            "figures_text": self.figures_text,
            "metadata_text": self.metadata_text,
            "warnings_text": self.warnings_text,
            "token_estimate": self.token_estimate,
        }


@dataclass(frozen=True)
class PaperArgumentStep:
    role: str
    claim: str
    evidence: str | None = None
    source_section: str | None = None
    why_it_matters: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PaperArgumentStep:
        return cls(
            role=str(value.get("role") or "").strip(),
            claim=str(value.get("claim") or "").strip(),
            evidence=_optional_str(value.get("evidence")),
            source_section=_optional_str(value.get("source_section")),
            why_it_matters=str(value.get("why_it_matters") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class KeyMechanism:
    name: str
    description: str
    paper_evidence: str | None = None
    source_section: str | None = None
    visual_potential: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> KeyMechanism:
        if isinstance(value, str):
            return cls(name=value.strip(), description=value.strip())
        if not isinstance(value, dict):
            return cls(name="", description="")
        name = str(value.get("name") or value.get("mechanism") or value.get("concept") or "").strip()
        return cls(
            name=name,
            description=str(value.get("description") or value.get("summary") or name).strip(),
            paper_evidence=_optional_str(value.get("paper_evidence") or value.get("evidence")),
            source_section=_optional_str(value.get("source_section")),
            visual_potential=_optional_str(value.get("visual_potential")),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class KeyExperiment:
    name: str
    purpose: str
    setup: str | None = None
    result: str | None = None
    source_section: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> KeyExperiment:
        if isinstance(value, str):
            return cls(name=value.strip(), purpose=value.strip())
        if not isinstance(value, dict):
            return cls(name="", purpose="")
        name = str(value.get("name") or value.get("experiment") or "").strip()
        return cls(
            name=name,
            purpose=str(value.get("purpose") or value.get("description") or name).strip(),
            setup=_optional_str(value.get("setup")),
            result=_optional_str(value.get("result")),
            source_section=_optional_str(value.get("source_section")),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class KeyResult:
    metric: str
    value: str
    comparison: str | None = None
    significance: str = ""
    source_section: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> KeyResult:
        if isinstance(value, str):
            return cls(metric="reported_result", value=value.strip(), significance=value.strip())
        if not isinstance(value, dict):
            return cls(metric="", value="", significance="")
        return cls(
            metric=str(value.get("metric") or value.get("name") or "reported_result").strip(),
            value=str(value.get("value") or value.get("result") or "").strip(),
            comparison=_optional_str(value.get("comparison")),
            significance=str(value.get("significance") or value.get("why_it_matters") or "").strip(),
            source_section=_optional_str(value.get("source_section")),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class VisualCandidate:
    candidate_id: str
    source_claim: str
    source_evidence: str | None
    paper_role: str
    visualization_type: str
    visual_idea: str
    why_visualize: str
    priority: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VisualCandidate:
        source_claim = str(
            value.get("source_claim")
            or value.get("claim")
            or value.get("concept")
            or value.get("item")
            or ""
        ).strip()
        return cls(
            candidate_id=str(value.get("candidate_id") or value.get("id") or "").strip(),
            source_claim=source_claim,
            source_evidence=_optional_str(value.get("source_evidence") or value.get("evidence")),
            paper_role=str(value.get("paper_role") or "").strip(),
            visualization_type=str(
                value.get("visualization_type")
                or value.get("suggested_visual_form")
                or "mechanism_animation"
            ).strip(),
            visual_idea=str(value.get("visual_idea") or value.get("visualization_potential") or "").strip(),
            why_visualize=str(value.get("why_visualize") or value.get("visualization_potential") or "").strip(),
            priority=_int(value.get("priority"), 5),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GlobalBrief:
    title: str | None
    topic: str
    problem: str
    motivation: str
    main_claim: str
    contributions: list[str]
    method_overview: str
    key_mechanisms: list[KeyMechanism]
    key_experiments: list[KeyExperiment]
    key_results: list[KeyResult]
    paper_argument: list[PaperArgumentStep]
    visual_candidates: list[VisualCandidate]
    source_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GlobalBrief:
        title = _optional_str(value.get("title") or value.get("topic"))
        return cls(
            title=title,
            topic=str(value.get("topic") or title or "Untitled paper").strip(),
            problem=str(value.get("problem") or "").strip(),
            motivation=str(value.get("motivation") or "").strip(),
            main_claim=str(value.get("main_claim") or "").strip(),
            contributions=_string_list(value.get("contributions")),
            method_overview=str(value.get("method_overview") or "").strip(),
            key_mechanisms=[
                KeyMechanism.from_dict(item)
                for item in _list_value(value.get("key_mechanisms"))
                if KeyMechanism.from_dict(item).name or KeyMechanism.from_dict(item).description
            ],
            key_experiments=[
                KeyExperiment.from_dict(item)
                for item in _list_value(value.get("key_experiments"))
                if KeyExperiment.from_dict(item).name or KeyExperiment.from_dict(item).purpose
            ],
            key_results=[
                KeyResult.from_dict(item)
                for item in _list_value(value.get("key_results"))
                if KeyResult.from_dict(item).value or KeyResult.from_dict(item).significance
            ],
            paper_argument=[
                PaperArgumentStep.from_dict(item)
                for item in value.get("paper_argument", [])
                if isinstance(item, dict)
            ],
            visual_candidates=[
                VisualCandidate.from_dict(item)
                for item in value.get("visual_candidates", [])
                if isinstance(item, dict)
            ],
            source_notes=_string_list(value.get("source_notes")),
            warnings=_string_list(value.get("warnings")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "topic": self.topic,
            "problem": self.problem,
            "motivation": self.motivation,
            "main_claim": self.main_claim,
            "contributions": self.contributions,
            "method_overview": self.method_overview,
            "key_mechanisms": [item.to_dict() for item in self.key_mechanisms],
            "key_experiments": [item.to_dict() for item in self.key_experiments],
            "key_results": [item.to_dict() for item in self.key_results],
            "paper_argument": [step.to_dict() for step in self.paper_argument],
            "visual_candidates": [candidate.to_dict() for candidate in self.visual_candidates],
            "source_notes": self.source_notes,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class NarrativeArc:
    hook: str = ""
    problem: str = ""
    method: str = ""
    mechanisms: list[str] = field(default_factory=list)
    evidence: str = ""
    result: str = ""
    takeaway: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NarrativeArc:
        return cls(
            hook=str(value.get("hook") or "").strip(),
            problem=str(value.get("problem") or "").strip(),
            method=str(value.get("method") or "").strip(),
            mechanisms=_string_list(value.get("mechanisms")),
            evidence=_string_or_join(value.get("evidence")),
            result=str(value.get("result") or "").strip(),
            takeaway=str(value.get("takeaway") or value.get("final_takeaway") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook": self.hook,
            "problem": self.problem,
            "method": self.method,
            "mechanisms": self.mechanisms,
            "evidence": self.evidence,
            "result": self.result,
            "takeaway": self.takeaway,
        }


@dataclass(frozen=True)
class VideoChapter:
    chapter_id: str
    title: str
    purpose: str
    target_duration_seconds: int | None = None
    scene_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, index: int, value: dict[str, Any]) -> VideoChapter:
        return cls(
            chapter_id=str(value.get("chapter_id") or value.get("id") or f"chapter_{index + 1:02d}").strip(),
            title=str(value.get("title") or f"Chapter {index + 1}").strip(),
            purpose=str(value.get("purpose") or "").strip(),
            target_duration_seconds=(
                _int(value.get("target_duration_seconds"), 0)
                if value.get("target_duration_seconds") is not None
                else None
            ),
            scene_ids=_string_list(value.get("scene_ids")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "purpose": self.purpose,
            "target_duration_seconds": self.target_duration_seconds,
            "scene_ids": self.scene_ids,
        }


@dataclass(frozen=True)
class AnimationBeat:
    beat_id: str
    duration_seconds: int | None
    visual_action: str
    narration_intent: str
    on_screen_text: list[str]

    @classmethod
    def from_dict(cls, index: int, value: Any) -> AnimationBeat:
        if isinstance(value, str):
            return cls(
                beat_id=f"beat_{index + 1:02d}",
                duration_seconds=None,
                visual_action=value.strip(),
                narration_intent=value.strip(),
                on_screen_text=[],
            )
        if not isinstance(value, dict):
            return cls(f"beat_{index + 1:02d}", None, "", "", [])
        return cls(
            beat_id=str(value.get("beat_id") or value.get("id") or f"beat_{index + 1:02d}").strip(),
            duration_seconds=(
                _int(value.get("duration_seconds"), 0)
                if value.get("duration_seconds") is not None
                else None
            ),
            visual_action=str(value.get("visual_action") or value.get("action") or "").strip(),
            narration_intent=str(value.get("narration_intent") or value.get("intent") or "").strip(),
            on_screen_text=_string_list(value.get("on_screen_text")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "beat_id": self.beat_id,
            "duration_seconds": self.duration_seconds,
            "visual_action": self.visual_action,
            "narration_intent": self.narration_intent,
            "on_screen_text": self.on_screen_text,
        }


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    title: str
    order: int
    chapter_id: str | None = None
    duration_seconds: int | None = None
    paper_role: str = ""
    paper_claim: str = ""
    paper_evidence: str | None = None
    source_section: str | None = None
    why_this_scene_matters: str = ""
    narrative_role: str = ""
    previous_scene_link: str | None = None
    next_scene_link: str | None = None
    visualization_type: str = ""
    visual_mapping: str = ""
    main_visual_object: str = ""
    supporting_visual_objects: list[str] = field(default_factory=list)
    mechanism: str | None = None
    cause: str | None = None
    visual_change: str | None = None
    effect: str | None = None
    final_takeaway: str = ""
    animation_beats: list[AnimationBeat] = field(default_factory=list)
    text_labels: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, index: int, value: dict[str, Any]) -> SceneSpec:
        scene_id = str(value.get("scene_id") or f"scene_{index + 1:02d}").strip()
        title = str(value.get("title") or f"Scene {index + 1}").strip()
        return cls(
            scene_id=scene_id,
            title=title,
            order=_int(value.get("order"), index + 1),
            chapter_id=_optional_str(value.get("chapter_id")),
            duration_seconds=(
                _int(value.get("duration_seconds"), 0)
                if value.get("duration_seconds") is not None
                else None
            ),
            paper_role=str(value.get("paper_role") or value.get("narrative_role") or "").strip(),
            paper_claim=str(value.get("paper_claim") or "").strip(),
            paper_evidence=_optional_str(value.get("paper_evidence")),
            source_section=_optional_str(value.get("source_section")),
            why_this_scene_matters=str(value.get("why_this_scene_matters") or "").strip(),
            narrative_role=str(value.get("narrative_role") or "").strip(),
            previous_scene_link=_optional_str(value.get("previous_scene_link")),
            next_scene_link=_optional_str(value.get("next_scene_link")),
            visualization_type=str(value.get("visualization_type") or "").strip(),
            visual_mapping=str(value.get("visual_mapping") or "").strip(),
            main_visual_object=str(value.get("main_visual_object") or "").strip(),
            supporting_visual_objects=_string_list(value.get("supporting_visual_objects")),
            mechanism=_optional_str(value.get("mechanism")),
            cause=_optional_str(value.get("cause")),
            visual_change=_optional_str(value.get("visual_change")),
            effect=_optional_str(value.get("effect")),
            final_takeaway=str(value.get("final_takeaway") or "").strip(),
            animation_beats=[
                AnimationBeat.from_dict(beat_index, beat)
                for beat_index, beat in enumerate(_list_value(value.get("animation_beats")))
            ],
            text_labels=_string_list(value.get("text_labels")),
            forbidden_patterns=_string_list(value.get("forbidden_patterns")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "order": self.order,
            "chapter_id": self.chapter_id,
            "duration_seconds": self.duration_seconds,
            "paper_role": self.paper_role,
            "paper_claim": self.paper_claim,
            "paper_evidence": self.paper_evidence,
            "source_section": self.source_section,
            "why_this_scene_matters": self.why_this_scene_matters,
            "narrative_role": self.narrative_role,
            "previous_scene_link": self.previous_scene_link,
            "next_scene_link": self.next_scene_link,
            "visualization_type": self.visualization_type,
            "visual_mapping": self.visual_mapping,
            "main_visual_object": self.main_visual_object,
            "supporting_visual_objects": self.supporting_visual_objects,
            "mechanism": self.mechanism,
            "cause": self.cause,
            "visual_change": self.visual_change,
            "effect": self.effect,
            "final_takeaway": self.final_takeaway,
            "animation_beats": [beat.to_dict() for beat in self.animation_beats],
            "text_labels": self.text_labels,
            "forbidden_patterns": self.forbidden_patterns,
        }


@dataclass(frozen=True)
class PaperVideoPlan:
    title: str
    user_prompt: str | None
    duration_mode: str | None
    target_duration_seconds: int | None
    narrative_arc: NarrativeArc | None
    chapters: list[VideoChapter]
    scenes: list[SceneSpec]
    plan_warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PaperVideoPlan:
        raw_arc = value.get("narrative_arc")
        raw_scenes = value.get("scenes")
        if not isinstance(raw_scenes, list):
            raise ValueError("PaperVideoPlan requires a scenes array.")
        return cls(
            title=str(value.get("title") or "Paper video plan").strip(),
            user_prompt=_optional_str(value.get("user_prompt")),
            duration_mode=_optional_str(value.get("duration_mode")),
            target_duration_seconds=(
                _int(value.get("target_duration_seconds") or value.get("expected_duration_sec"), 0)
                if value.get("target_duration_seconds") is not None or value.get("expected_duration_sec") is not None
                else None
            ),
            narrative_arc=NarrativeArc.from_dict(raw_arc) if isinstance(raw_arc, dict) else None,
            chapters=[
                VideoChapter.from_dict(index, item)
                for index, item in enumerate(value.get("chapters", []))
                if isinstance(item, dict)
            ],
            scenes=[
                SceneSpec.from_dict(index, item)
                for index, item in enumerate(raw_scenes)
                if isinstance(item, dict)
            ],
            plan_warnings=_string_list(value.get("plan_warnings") or value.get("warnings")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "user_prompt": self.user_prompt,
            "duration_mode": self.duration_mode,
            "target_duration_seconds": self.target_duration_seconds,
            "narrative_arc": self.narrative_arc.to_dict() if self.narrative_arc else None,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "scenes": [scene.to_dict() for scene in self.scenes],
            "plan_warnings": self.plan_warnings,
        }


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    code: str
    message: str
    target: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "target": self.target,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class QualityReport:
    passed: bool
    issues: list[QualityIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class FinalVideoResult:
    run_id: str
    output_path: str
    clip_paths: list[str]
    scene_ids: list[str]
    success: bool
    duration_seconds: float | None = None
    ffmpeg_stdout_path: str | None = None
    ffmpeg_stderr_path: str | None = None
    concat_list_path: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_path": self.output_path,
            "clip_paths": self.clip_paths,
            "scene_ids": self.scene_ids,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "ffmpeg_stdout_path": self.ffmpeg_stdout_path,
            "ffmpeg_stderr_path": self.ffmpeg_stderr_path,
            "concat_list_path": self.concat_list_path,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SceneSummary:
    scene_id: str
    title: str | None
    status: str
    accepted_clip_path: str | None
    render_attempt_count: int
    visual_revision_attempt_count: int
    visual_review_decisions: list[str]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "status": self.status,
            "accepted_clip_path": self.accepted_clip_path,
            "render_attempt_count": self.render_attempt_count,
            "visual_revision_attempt_count": self.visual_revision_attempt_count,
            "visual_review_decisions": self.visual_review_decisions,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    source_pdf: str
    user_prompt: str | None
    final_video_path: str | None
    artifacts_root: str
    total_scenes: int
    accepted_scenes: int
    failed_scenes: int
    skipped_scenes: int
    visual_review_enabled: bool
    visual_review_pass_count: int
    visual_review_revise_count: int
    visual_review_fail_count: int
    render_failure_count: int
    render_fix_count: int
    warnings: list[str]
    errors: list[str]
    scene_summaries: list[SceneSummary]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_pdf": self.source_pdf,
            "user_prompt": self.user_prompt,
            "final_video_path": self.final_video_path,
            "artifacts_root": self.artifacts_root,
            "total_scenes": self.total_scenes,
            "accepted_scenes": self.accepted_scenes,
            "failed_scenes": self.failed_scenes,
            "skipped_scenes": self.skipped_scenes,
            "visual_review_enabled": self.visual_review_enabled,
            "visual_review_pass_count": self.visual_review_pass_count,
            "visual_review_revise_count": self.visual_review_revise_count,
            "visual_review_fail_count": self.visual_review_fail_count,
            "render_failure_count": self.render_failure_count,
            "render_fix_count": self.render_fix_count,
            "warnings": self.warnings,
            "errors": self.errors,
            "scene_summaries": [scene.to_dict() for scene in self.scene_summaries],
            "success": self.success,
        }


@dataclass(frozen=True)
class VisualReviewScores:
    paper_alignment: int
    visual_clarity: int
    readability: int
    layout_balance: int
    visual_focus: int
    animation_perceived: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VisualReviewScores:
        return cls(
            paper_alignment=_int(value.get("paper_alignment"), 1),
            visual_clarity=_int(value.get("visual_clarity"), 1),
            readability=_int(value.get("readability"), 1),
            layout_balance=_int(value.get("layout_balance"), 1),
            visual_focus=_int(value.get("visual_focus"), 1),
            animation_perceived=_int(value.get("animation_perceived"), 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_alignment": self.paper_alignment,
            "visual_clarity": self.visual_clarity,
            "readability": self.readability,
            "layout_balance": self.layout_balance,
            "visual_focus": self.visual_focus,
            "animation_perceived": self.animation_perceived,
        }


@dataclass(frozen=True)
class VisualIssue:
    type: str
    severity: str
    evidence: str
    suggestion: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VisualIssue:
        return cls(
            type=str(value.get("type") or "other").strip(),
            severity=str(value.get("severity") or "low").strip(),
            evidence=str(value.get("evidence") or "").strip(),
            suggestion=str(value.get("suggestion") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class VisualReviewResult:
    scene_id: str
    decision: str
    scores: VisualReviewScores
    issues: list[VisualIssue]
    paper_alignment_notes: str
    revision_instruction: str
    attempt: int = 1
    requires_replanning: bool = False
    raw_response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VisualReviewResult:
        return cls(
            scene_id=str(value.get("scene_id") or "").strip(),
            attempt=_int(value.get("attempt"), 1),
            decision=str(value.get("decision") or "pass").strip(),
            scores=VisualReviewScores.from_dict(value.get("scores", {})),
            issues=[
                VisualIssue.from_dict(item)
                for item in value.get("issues", [])
                if isinstance(item, dict)
            ],
            paper_alignment_notes=str(value.get("paper_alignment_notes") or "").strip(),
            revision_instruction=str(value.get("revision_instruction") or "").strip(),
            requires_replanning=bool(value.get("requires_replanning", False)),
            raw_response=_optional_str(value.get("raw_response")),
            metadata=value.get("metadata") if isinstance(value.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "attempt": self.attempt,
            "decision": self.decision,
            "scores": self.scores.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "paper_alignment_notes": self.paper_alignment_notes,
            "revision_instruction": self.revision_instruction,
            "requires_replanning": self.requires_replanning,
            "raw_response": self.raw_response,
            "metadata": self.metadata,
        }


@dataclass
class ManimSceneCode:
    scene_id: str
    scene_class_name: str
    code: str
    file_path: str | None = None
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_class_name": self.scene_class_name,
            "code": self.code,
            "file_path": self.file_path,
            "attempt": self.attempt,
            "metadata": self.metadata,
        }


@dataclass
class SceneRenderResult:
    scene_id: str
    attempt: int
    success: bool
    clip_path: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    error_message: str | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "attempt": self.attempt,
            "success": self.success,
            "clip_path": self.clip_path,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class FrameSampleResult:
    scene_id: str
    attempt: int
    frame_paths: list[str]
    montage_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "attempt": self.attempt,
            "frame_paths": self.frame_paths,
            "montage_path": self.montage_path,
            "metadata": self.metadata,
        }


@dataclass
class SceneRunState:
    scene_id: str
    spec: SceneSpec
    code_attempts: list[ManimSceneCode] = field(default_factory=list)
    render_attempts: list[SceneRenderResult] = field(default_factory=list)
    frame_samples: list[FrameSampleResult] = field(default_factory=list)
    visual_reviews: list[VisualReviewResult] = field(default_factory=list)
    accepted_clip_path: str | None = None
    status: str = "planned"
    render_attempt_count: int = 0
    visual_revision_attempt_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "spec": self.spec.to_dict(),
            "code_attempts": [item.to_dict() for item in self.code_attempts],
            "render_attempts": [item.to_dict() for item in self.render_attempts],
            "frame_samples": [item.to_dict() for item in self.frame_samples],
            "visual_reviews": [item.to_dict() for item in self.visual_reviews],
            "accepted_clip_path": self.accepted_clip_path,
            "status": self.status,
            "render_attempt_count": self.render_attempt_count,
            "visual_revision_attempt_count": self.visual_revision_attempt_count,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class ManimScript:
    scene_id: str
    code: str
    path: Path


@dataclass
class SceneArtifact:
    scene_id: str
    title: str
    directory: Path
    spec_path: Path
    code_path: Path
    log_path: Path
    render: RenderResult | None = None
    validation_error: str | None = None
    repair_attempted: bool = False
    fallback_used: bool = False

    @property
    def success(self) -> bool:
        if self.validation_error:
            return False
        if self.render is None:
            return True
        return self.render.success


@dataclass
class ProjectState:
    paper_document: PaperDocument
    paper_context: PaperContextPackage
    output_dir: Path
    render_enabled: bool
    provider: str
    model: str
    duration_mode: str = "medium"
    target_duration_seconds: int | None = None
    max_failed_scenes: int = 0
    continue_on_scene_failure: bool = False
    visual_review_enabled: bool = False
    visual_review_results: list[VisualReviewResult] = field(default_factory=list)
    visual_review_warnings: list[str] = field(default_factory=list)
    global_brief: GlobalBrief | None = None
    video_plan: PaperVideoPlan | None = None
    quality_report: QualityReport | None = None
    scene_states: list[SceneRunState] = field(default_factory=list)
    scene_artifacts: list[SceneArtifact] = field(default_factory=list)
    final_video_path: Path | None = None
    final_video_result: FinalVideoResult | None = None
    run_summary: RunSummary | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _string_or_join(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
