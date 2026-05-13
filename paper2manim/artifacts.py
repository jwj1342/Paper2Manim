"""Run artifact management: runs/<run_id>/ directory + trace.jsonl."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from paper2manim.config.env import settings

# Serializes concurrent ``append_trace`` writers. ``--scene-parallelism > 1``
# fans out scenes onto LangGraph's thread pool, all of which share one
# ``trace.jsonl`` per run. The OS-level append is atomic for individual
# writes, but interleaved partial-line writes from multiple threads can still
# corrupt the file under load — the lock is the simplest correctness fix.
_TRACE_LOCK = threading.Lock()


def new_run_id() -> str:
    """Timestamp-prefixed UUID short id, e.g., '20260510-073142-a1b2c3'."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}-{short}"


def run_dir(run_id: str) -> Path:
    d = Path(settings.PAPER2MANIM_RUNS_DIR) / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "attempts").mkdir(exist_ok=True)
    (d / "final").mkdir(exist_ok=True)
    return d


def save_input(run_id: str, *, raw_text: str | None = None, pdf_path: str | None = None) -> None:
    d = run_dir(run_id)
    if raw_text is not None:
        (d / "input.txt").write_text(raw_text, encoding="utf-8")
    if pdf_path is not None:
        # don't copy large PDFs; just record the absolute path
        (d / "input.pdf.path").write_text(str(Path(pdf_path).resolve()), encoding="utf-8")


def save_json(run_id: str, name: str, data: Any) -> Path:
    """Save data as <run_dir>/<name>.json (utf-8, indent=2)."""
    p = run_dir(run_id) / f"{name}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def save_attempt_code(run_id: str, scene: str, iter_idx: int, code: str) -> Path:
    p = run_dir(run_id) / "attempts" / f"{iter_idx:02d}_{scene}.py"
    p.write_text(code, encoding="utf-8")
    return p


def save_attempt_result(run_id: str, scene: str, iter_idx: int, result: dict) -> Path:
    p = run_dir(run_id) / "attempts" / f"{iter_idx:02d}_{scene}.render.json"
    p.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return p


def append_trace(run_id: str, node: str, payload: dict) -> None:
    """Append a JSONL line to runs/<run_id>/trace.jsonl.

    Thread-safe under scene-parallel fan-out via :data:`_TRACE_LOCK`.
    """
    p = run_dir(run_id) / "trace.jsonl"
    rec = {"ts": time.time(), "node": node, **payload}
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    with _TRACE_LOCK, p.open("a", encoding="utf-8") as f:
        f.write(line)


def copy_final_video(run_id: str, src: str, name: str) -> Path:
    """Copy a rendered mp4 into runs/<run_id>/final/<name>.mp4."""
    import shutil

    dst = run_dir(run_id) / "final" / name
    if not dst.suffix:
        dst = dst.with_suffix(".mp4")
    shutil.copy2(src, dst)
    return dst
