from __future__ import annotations

from pathlib import Path

# Top-level repo prompts/ directory (avoids conflict with main's
# paper2manim/prompts.py module). prompt_loader.py lives at
# paper2manim/utils/prompt_loader.py, so parents[2] is the repo root.
PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str, *, prompt_dir: Path = PROMPT_DIR) -> str:
    if Path(name).name != name:
        raise ValueError(f"Prompt name must be a file name, got: {name!r}")
    path = prompt_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {path}")
    return content


def load_agent_prompt(name: str) -> str:
    return load_prompt("global_system.md") + "\n\n" + load_prompt(name)
