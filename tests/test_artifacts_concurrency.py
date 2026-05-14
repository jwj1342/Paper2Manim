"""Stress-test ``artifacts.append_trace`` under concurrent threads.

Confirms the trace.jsonl lock prevents interleaved writes. Without it, multiple
threads racing into ``open("a") + write`` can produce torn lines and corrupt
the JSON parser at consolidation time.
"""

from __future__ import annotations

import json
import threading
import uuid

from paper2manim.artifacts import append_trace, run_dir


def test_append_trace_no_interleave_under_concurrency():
    # Unique run_id per invocation — the env-based settings module caches
    # PAPER2MANIM_RUNS_DIR at import time, so the autouse tmp-path fixture
    # cannot redirect ``run_dir`` for this code path. A unique id keeps the
    # file fresh across repeated test runs without depending on cleanup.
    run_id = f"rid_trace_{uuid.uuid4().hex}"
    # Touch the run dir so trace.jsonl path is stable from thread 0.
    trace_path = run_dir(run_id) / "trace.jsonl"
    # Ensure clean slate even on the (very unlikely) uuid collision.
    if trace_path.exists():
        trace_path.unlink()

    N_THREADS = 8
    LINES_PER_THREAD = 50
    barrier = threading.Barrier(N_THREADS)

    def worker(tid: int):
        barrier.wait()
        for i in range(LINES_PER_THREAD):
            append_trace(
                run_id,
                "stress",
                {"thread_id": tid, "seq": i, "filler": "x" * 200},
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    # Validate: every line is a parseable JSON object with the expected keys.
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == N_THREADS * LINES_PER_THREAD

    seen: set[tuple[int, int]] = set()
    for ln in lines:
        obj = json.loads(ln)  # would raise if torn
        seen.add((obj["thread_id"], obj["seq"]))
    # All (thread, seq) pairs accounted for; no dupes, no drops.
    expected = {(t, i) for t in range(N_THREADS) for i in range(LINES_PER_THREAD)}
    assert seen == expected
