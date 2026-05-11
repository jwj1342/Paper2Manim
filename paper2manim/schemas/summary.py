from pydantic import BaseModel, Field


class FormulaItem(BaseModel):
    latex: str = Field(..., description="LaTeX source")
    explanation: str = Field(..., description="Plain-language explanation")


class SummaryModel(BaseModel):
    title: str
    key_contributions: list[str] = Field(..., min_length=1, max_length=8)
    key_formulas: list[FormulaItem] = Field(default_factory=list, max_length=10)
    main_concepts: list[str] = Field(default_factory=list, max_length=10)
    scene_suggestions: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Hints for the storyboarder agent on what to visualize",
    )
