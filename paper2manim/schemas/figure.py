"""VLM-produced semantic labels + (optional) reconstruction recipe for paper figures.

Populated by ``paper2manim/agents/figure_understander.py`` during MVP 2.x runs.
Stored back into ``state.figures[i].semantics`` and ``state.figures[i].recipe``
for downstream consumers (summarizer, storyboarder, coder).

The wrapper model ``FigureUnderstanding`` is what the VLM emits in one shot:
classification (always) + recipe (only when redrawable=True).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


FigType = Literal[
    "schematic",  # architecture / pipeline / flow diagram
    "chart",  # plot with axes (loss curve, bar chart, scaling law)
    "heatmap",  # 2D color-coded matrix (attention map, confusion matrix)
    "photo",  # photograph, microscopy, dataset sample, model output sample
    "equation_figure",  # formula illustration with arrows/labels
    "algorithm_box",  # pseudocode in a bordered block
    "unknown",  # VLM could not classify
]


class FigureSemantics(BaseModel):
    fig_type: FigType = Field(..., description="High-level category of the figure")
    redrawable: bool = Field(
        ...,
        description=(
            "True when Manim can reconstruct the figure from structure "
            "(schematic / chart / equation_figure / algorithm_box); "
            "False for empirical artifacts (heatmap / photo) that must be embedded as-is."
        ),
    )
    salience: Literal["high", "medium", "low"] = Field(
        ..., description="How central this figure is to the paper's main result"
    )
    one_line_summary: str = Field(
        ..., max_length=300, description="One-sentence semantic summary in plain English"
    )
    key_elements: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="2–6 short noun phrases naming the components shown",
    )


# ---- Recipe (Phase 3) -------------------------------------------------------

RecipeTemplate = Literal[
    "vertical_stack",  # boxes stacked top-to-bottom (e.g. encoder/decoder stack)
    "horizontal_flow",  # left-to-right pipeline (input → processing → output)
    "branching_flow",  # split-then-merge (multi-head attention)
    "comparison_grid",  # 2D grid of comparison cells
    "plot_with_axes",  # an Axes + plotted curve (loss curve, scaling law)
    "module_diagram",  # arbitrary node-and-edge graph
    "freeform",  # fallback when no template fits
]


class RecipeNode(BaseModel):
    id: str = Field(..., description="Unique identifier referenced by edges (e.g. 'n1')")
    label: str = Field(..., description="Visible text label for the node")
    shape: Literal["rect", "circle", "diamond", "text"] = Field(
        default="rect", description="Manim shape used to draw the node"
    )
    group: str | None = Field(
        default=None,
        description="Optional grouping tag (e.g. 'encoder_block') for stacked layouts",
    )


class RecipeEdge(BaseModel):
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    label: str | None = Field(default=None, description="Optional edge label")
    style: Literal["solid", "dashed", "skip"] = Field(
        default="solid", description="Solid for data flow; dashed for residual/aux; skip for skip-connections"
    )


class FigureRecipe(BaseModel):
    """A structured re-draw recipe for figures with redrawable=True.

    The VLM produces this only when classification picks a redrawable fig_type;
    coder consumes it together with manim_skill_rules to emit Manim code.
    """

    template: RecipeTemplate = Field(
        ..., description="Layout template; constrains how nodes are arranged"
    )
    nodes: list[RecipeNode] = Field(
        ..., min_length=1, max_length=20, description="Components shown in the figure"
    )
    edges: list[RecipeEdge] = Field(
        default_factory=list, max_length=40, description="Connections between nodes"
    )
    annotations: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Free-text labels NOT attached to a specific node (e.g. 'Encoder ×6')",
    )
    animation_hint: str = Field(
        default="",
        max_length=300,
        description="One sentence telling the coder how to animate the reveal",
    )


# ---- Combined VLM output ----------------------------------------------------


class FigureUnderstanding(BaseModel):
    """One-shot VLM output: classification + optional recipe.

    The VLM emits this when reading a paper figure. ``recipe`` MUST be ``None``
    when ``semantics.redrawable`` is False; conversely, ``recipe`` SHOULD be
    present when ``redrawable`` is True (but coder degrades gracefully if not).
    """

    semantics: FigureSemantics
    recipe: FigureRecipe | None = None
