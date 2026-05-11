from pydantic import BaseModel, Field, field_validator


class SceneModel(BaseModel):
    name: str = Field(..., description="PascalCase Scene class name (used as manim render arg)")
    description: str = Field(..., description="Natural-language description of what the scene shows")
    duration_hint: float = Field(8.0, ge=1.0, le=60.0, description="Expected duration in seconds")

    @field_validator("name")
    @classmethod
    def _pascal_case(cls, v: str) -> str:
        v = v.strip()
        if not v or not v[0].isalpha():
            raise ValueError("Scene name must start with a letter")
        if not v.replace("_", "").isalnum():
            raise ValueError("Scene name must be alphanumeric (PascalCase)")
        return v[0].upper() + v[1:]


class StoryboardModel(BaseModel):
    title: str = Field(..., description="Title of the overall video")
    scenes: list[SceneModel] = Field(..., min_length=1, max_length=10)
