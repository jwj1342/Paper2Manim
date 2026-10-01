# Contributing to ManimAgent

The research system is ManimAgent; the Python package and CLI are named `paper2manim`. Start with [Getting started](docs/getting-started.md) for local setup and [Architecture](docs/architecture.md) for code ownership.

## Development setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check paper2manim tests scripts
pytest -m "not slow"
```

The fast tests mock external model calls and rendering. Mark tests that need live APIs or a real Manim render with `@pytest.mark.slow`. Run the relevant focused tests when changing an agent, graph, parser, or EMB component.

## Where to work

| Change | Main location |
|---|---|
| Agent behavior and prompts | `paper2manim/agents/`, `prompts/` |
| Paper-level and per-scene flow | `paper2manim/graphs/generation.py`, `paper2manim/graphs/scene_graph.py` |
| CLI and model configuration | `paper2manim/cli.py`, `paper2manim/config/` |
| Rendering and video assembly | `paper2manim/sandbox/`, `paper2manim/voiceover/` |
| Episodic memory | `paper2manim/emb/` |
| Setup and batch utilities | `scripts/` |

Keep generated videos, run directories, model caches, and API keys out of commits. `.env` and `runs/` are ignored by Git. Do not put secrets in examples or test fixtures.
