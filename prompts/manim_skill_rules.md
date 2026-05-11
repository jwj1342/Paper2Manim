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
- **Don't load images**: no `ImageMobject(...)` for MVP 1.0/2.0.
- **Don't reference unknown LaTeX macros**: stick to `amsmath`/`amssymb`. `\R` is not standard; use `\mathbb{R}`.
- **Class name must match the requested scene.name** so `manim render scene.py SceneName` finds it.
