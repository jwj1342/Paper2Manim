# Manim CE 0.20 quick API reference (paper2manim project conventions)

## Core skeleton
```python
from manim import *

class MyScene(Scene):
    def construct(self):
        title = Text("Hello").to_edge(UP)
        self.play(Write(title))
        self.wait(1)
```

## Mobjects (visible objects)
- **Text**: `Text("hello", font_size=36, color=WHITE)`
- **LaTeX prose**: `Tex(r"This is $x^2$")`
- **Math equation**: `MathTex(r"\int_0^1 x^2\, dx = \tfrac{1}{3}")`
- **Shapes**: `Circle(radius=1, color=BLUE)`, `Square(side_length=2)`, `Line(LEFT, RIGHT)`, `Arrow(LEFT, RIGHT, buff=0.1)`, `Dot(point=ORIGIN)`
- **Axes / graph**: `Axes(x_range=[-3, 3], y_range=[-1, 5])`, `axes.plot(lambda x: x**2, color=RED)`
- **Group**: `VGroup(a, b, c).arrange(RIGHT, buff=0.4)`

## Animations (use inside `self.play(...)`)
- `Write(mobj)` — for text/equations (draws stroke-by-stroke)
- `Create(mobj)` — for shapes/curves
- `FadeIn(mobj)`, `FadeOut(mobj)`
- `Transform(a, b)` — morphs `a` into `b` (mutates `a`)
- `ReplacementTransform(a, b)` — like Transform but cleaner removes/adds
- `Indicate(mobj)` — pulse highlight
- `MoveToTarget(mobj)` after setting `mobj.target = mobj.copy().shift(...)`

## Positioning
- `.to_edge(UP | DOWN | LEFT | RIGHT, buff=0.5)`
- `.move_to(ORIGIN)` / `.shift(2*RIGHT + UP)`
- `.next_to(other, DOWN, buff=0.3)`
- `.scale(1.5)`, `.rotate(PI/4)`

## Constants
- Directions: `UP DOWN LEFT RIGHT ORIGIN UR UL DR DL`
- Colors: `WHITE BLACK BLUE RED GREEN YELLOW PURPLE ORANGE PINK GREY GREY_A..GREY_E`
- Math: `PI TAU DEGREES`

## Timing rules
- `self.play(anim_a, anim_b, run_time=2)` — default 1s
- `self.wait(t)` — hold for t seconds
- Always end with `self.wait(...)` so the last frame doesn't snap away.

## Pitfalls to avoid
- **Always raw-string LaTeX**: `MathTex(r"\frac{1}{2}")`, never `"\\frac{1}{2}"` in normal strings.
- **Don't `from manim.opengl import *`**: this project uses cairo.
- **Don't load arbitrary images**: `ImageMobject(...)` is ONLY allowed when the scene prompt includes a `## Figures to show` block AND the figure's `recipe` field is null (embed-mode for non-redrawable figures). Otherwise stick to Manim-native shapes.
- **Don't reference unknown LaTeX macros**: stick to `amsmath`/`amssymb`. `\R` is not standard; use `\mathbb{R}`.
- **Class name must match the requested scene.name** so `manim render scene.py SceneName` finds it.

## Rendering tables (used when the scene prompt includes a `## Tables to render` block)

You'll be given JSON like:
```json
{"tab_id": "tab_001", "header": ["Method", "Acc", "F1"], "rows": [["A", "0.9", "0.88"], ["B", "0.7", "0.71"]]}
```

Render it with Manim's `Table` mobject. Minimum viable pattern:

```python
from manim import Table, FadeIn, Write

tbl = Table(
    [["A", "0.9", "0.88"], ["B", "0.7", "0.71"]],
    col_labels=[Text("Method"), Text("Acc"), Text("F1")],
    include_outer_lines=True,
).scale(0.6).to_edge(UP)

self.play(FadeIn(tbl.get_horizontal_lines()), Write(tbl.get_labels()))
self.play(*[FadeIn(row) for row in tbl.get_rows()[1:]], run_time=1.5)
self.wait(1)
```

Rules:
- Use `Text(...)` for `col_labels`, not raw strings — strings will be inferred as `MathTex` and break for non-LaTeX cells.
- If a header cell contains LaTeX (e.g. `"$F_1$"`), wrap it with `MathTex(...)` instead.
- Use `.scale(0.6)` or smaller — full-size tables overflow the frame.
- Animate header first, then reveal rows progressively for narrative pacing.
- Use `header` and `rows` from the JSON **verbatim**. Do not paraphrase, reorder, or drop cells.
- If `header` / `rows` are `null`, fall back to `raw_md` — render it as `Text(raw_md, font_size=18)` in a `Code`-style block.

## Embedding paper figures (used when the scene's figure JSON has `recipe: null`)

When the prompt JSON looks like:
```json
{"fig_id": "fig_003", "path": "/runs/.../assets/figures/fig_003.png", "fig_type": "heatmap",
 "redrawable": false, "one_line_summary": "Attention pattern showing source-target alignment",
 "recipe": null}
```

The figure is an empirical artifact (heatmap / photo / sample output) that can't be reconstructed from structure. Embed the original PNG and narrate around it:

```python
from manim import ImageMobject, FadeIn, Write, SurroundingRectangle, YELLOW

img = ImageMobject(payload["path"]).scale_to_fit_height(4.5).shift(LEFT*1.5)
caption = Text(payload["one_line_summary"], font_size=24).next_to(img, DOWN, buff=0.3)
self.play(FadeIn(img), run_time=1.5)
self.play(Write(caption))
# Optional: highlight a region you want to draw attention to
hl = SurroundingRectangle(img, color=YELLOW, buff=0.1)
self.play(Create(hl))
self.wait(2)
```

Rules:
- Use `ImageMobject(payload["path"])` — the path is the absolute path the parser saved.
- Always pair the image with a `Text` caption from `one_line_summary` so viewers know what they're looking at.
- `.scale_to_fit_height(4.5)` or smaller keeps the frame from clipping. Never `.scale_to_fit_height(8)` — text and image will collide.
- For multi-figure scenes, place figures side-by-side with `Group(img1, img2).arrange(RIGHT, buff=0.4)`.

## Reproducing figures via FigureRecipe (used when the scene's figure JSON has `recipe: { ... }`)

When the prompt JSON's `recipe` field is populated, the figure is redrawable — synthesize it in Manim from the structured nodes/edges. The `template` field tells you the layout strategy:

### `vertical_stack` — boxes stacked top→bottom
```python
boxes = VGroup(*[
    VGroup(
        Rectangle(width=3, height=0.8),
        Text(n["label"], font_size=24),
    ).set_z_index(0)  # text on rect
    for n in recipe["nodes"]
]).arrange(DOWN, buff=0.3)
self.play(*[FadeIn(b, shift=DOWN*0.2) for b in boxes], run_time=2)
# edges: vertical arrows between consecutive boxes
for e in recipe["edges"]:
    src = boxes[next(i for i,n in enumerate(recipe["nodes"]) if n["id"]==e["source"])]
    tgt = boxes[next(i for i,n in enumerate(recipe["nodes"]) if n["id"]==e["target"])]
    self.play(GrowArrow(Arrow(src.get_bottom(), tgt.get_top(), buff=0.05)))
```

### `horizontal_flow` — left→right pipeline
Identical to vertical_stack but with `.arrange(RIGHT, buff=0.5)` and arrows from `get_right()` to `get_left()`.

### `branching_flow` — split-then-merge
Stack the `group` tag for parallel branches. Example for multi-head attention (group="head"):
```python
heads = VGroup(*[Rectangle(width=1, height=0.6) for _ in range(8)]).arrange(RIGHT, buff=0.15)
```

### `plot_with_axes` — axes + curves
```python
ax = Axes(x_range=[0, 10], y_range=[0, 1])
for node in recipe["nodes"]:  # each node describes a curve
    curve = ax.plot(lambda x, n=node: ... , color=BLUE)
    self.play(Create(curve), run_time=2)
self.play(Write(Text(node["label"]).next_to(curve, UP)))
```
For `plot_with_axes`, the recipe `nodes` describe curves rather than boxes; `edges` are typically empty.

### `comparison_grid` — 2D grid of cells
Use `VGroup(...).arrange_in_grid(rows=R, cols=C)`. Each cell can be a `VGroup(Rectangle, Text)`.

### `module_diagram` — arbitrary node-and-edge graph
Fallback: place nodes with `.arrange_in_grid()` or manual `.move_to(coords)`. Use `Arrow(src.get_center(), tgt.get_center(), buff=0.4)` for edges. Style: `style="dashed"` → `Arrow(...).set_stroke(dash_length=0.1)`; `style="skip"` → curved arrow via `CurvedArrow(src.get_top(), tgt.get_top(), angle=-PI/4)`.

### `freeform` — fallback
No template fits cleanly. Best-effort: lay out nodes in a sensible flow based on the labels, connect per edges, use `animation_hint` for reveal order.

Rules (apply to all templates):
- Use `node.label` verbatim — do not paraphrase.
- Edge styles map to: `solid` → `Arrow`, `dashed` → `DashedLine` + small arrow tip, `skip` → `CurvedArrow`.
- Annotations from `recipe["annotations"]` go OUTSIDE the diagram (`to_edge(RIGHT)` or `.next_to(whole_diagram, UP)`).
- Use `animation_hint` to decide reveal pacing (left-to-right, bottom-to-top, etc.).
- If `recipe` looks malformed or you can't render it, fall back to embedding via `ImageMobject(payload["path"])` with a caption noting "(reproduced from paper)".
