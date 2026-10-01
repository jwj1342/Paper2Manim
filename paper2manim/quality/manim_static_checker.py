from __future__ import annotations

import ast
import re

# Adjusted per issue #1 consensus (D1 + D4):
# - FORBIDDEN_MODULES: re-allowed `pathlib` and `sys` (the original list was too
#   strict for legitimate use); still blocking filesystem write/delete +
#   process + network + dynamic exec.
# - FORBIDDEN_ATTRS: added Path.write_text / write_bytes since pathlib is now
#   allowed (raw read access through Path is fine, mutation is not).
# - FORBIDDEN_NAMES: re-allowed `Tex` / `MathTex` per D1 (academic videos need
#   formulas); only `Paragraph` stays banned to discourage wall-of-text scenes.
FORBIDDEN_MODULES = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "shutil",
}

FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__"}
FORBIDDEN_ATTRS = {
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "rmtree",
    "write_text",
    "write_bytes",
}
FORBIDDEN_NAMES = {"Paragraph"}


def validate_manim_code(code: str, scene_name: str | None = None) -> None:
    if "from manim import *" not in code:
        raise ValueError("Generated code must include `from manim import *`.")
    if not re.search(r"class\s+\w+\s*\(\s*Scene\s*\)", code):
        raise ValueError("Generated code must define a Scene subclass.")
    tree = ast.parse(code)
    scene_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "Scene" for base in node.bases)
    ]
    if len(scene_classes) != 1 or (scene_name and scene_classes[0].name != scene_name):
        raise ValueError(
            f"Generated code must define exactly one Scene subclass named {scene_name!r}."
        )
    plays = [
        node
        for node in ast.walk(scene_classes[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "play"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ]
    if len(plays) < 2:
        raise ValueError("Generated code must contain multiple animation steps.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    raise ValueError(f"Forbidden import in generated code: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in FORBIDDEN_MODULES:
                raise ValueError(f"Forbidden import in generated code: {node.module}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"Forbidden call in generated code: {node.func.id}")
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
                raise ValueError(f"LaTeX-backed object is not allowed: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRS:
                raise ValueError(f"Forbidden call in generated code: {node.func.attr}")
