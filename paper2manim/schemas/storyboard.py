import keyword

from pydantic import BaseModel, Field, field_validator, model_validator


class SceneModel(BaseModel):
    name: str = Field(..., description="PascalCase Scene class name (used as manim render arg)")
    description: str = Field(
        ..., description="Natural-language description of what the scene shows"
    )
    paper_claim: str = Field(
        "", description="Scientific claim to preserve through coding and revision"
    )
    paper_evidence: str = Field("", description="Source evidence supporting the claim")
    final_takeaway: str = Field("", description="Final pedagogical takeaway to preserve")
    duration_hint: float = Field(8.0, ge=1.0, le=60.0, description="Expected duration in seconds")

    @field_validator("name")
    @classmethod
    def _pascal_case(cls, v: str) -> str:
        v = v.strip()
        if not v or not v[0].isalpha():
            raise ValueError("Scene name must start with a letter")
        if not v.replace("_", "").isalnum():
            raise ValueError("Scene name must be alphanumeric (PascalCase)")
        name = v[0].upper() + v[1:]
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError("Scene name must be a valid Python class name")
        return name


class StoryboardModel(BaseModel):
    title: str = Field(..., description="Title of the overall video")
    scenes: list[SceneModel] = Field(..., min_length=1, max_length=10)

    @model_validator(mode="after")
    def _unique_scene_names(self):
        names = [scene.name for scene in self.scenes]
        if len(names) != len(set(names)):
            raise ValueError("Scene names must be unique")
        return self
