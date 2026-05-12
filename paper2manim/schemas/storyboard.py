from pydantic import BaseModel, Field, field_validator


class SceneModel(BaseModel):
    name: str = Field(..., description="PascalCase Scene class name (used as manim render arg)")
    description: str = Field(..., description="Natural-language description of what the scene shows")
    duration_hint: float = Field(8.0, ge=1.0, le=60.0, description="Expected duration in seconds")
    referenced_tables: list[str] = Field(
        default_factory=list,
        description="Optional list of tab_id strings (e.g. 'tab_001') that this scene should render. "
        "Only set when state.tables has matching entries; leave empty otherwise.",
    )
    referenced_figures: list[str] = Field(
        default_factory=list,
        description="Optional list of fig_id strings (e.g. 'fig_001') that this scene should show. "
        "Coder reproduces redrawable figures via FigureRecipe; embeds non-redrawable ones via "
        "ImageMobject. Only set when state.figures has matching entries.",
    )

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
