#!/usr/bin/env python3
"""
Complete ManimAgent experiment pipeline — maximal concurrency + full checkpoint/resume.

Covers every experiment in the EMNLP paper:
  RQ1 (Table 4):  A, B, RAG×3, Random_EMB, EMB@0/50/100/200/400
  RQ2 (Table 6):  C_no_vlm, C_no_{success,failure}_channel, C_rationale_only,
                   C_no_failure_gate, C_top1
  Cross-domain (Table 13): A/B/C/RAG_cross

Strategy (sequential phases to avoid OOM on memory-constrained machines):
  T=0   Phase 1: all memory_build tasks fire in parallel → independent temp EMBs
  T≈15m Phase 1 finishes → 1-shot merge → freeze snapshots
  T≈15m Phase 2: ALL conditions launch together (no overlap with Phase 1)

Resume guarantee: kill -9 at any point, re-run with --resume, and every
already-completed (condition, seed, task_idx) triple is skipped via on-disk
checkpoints.  No in-memory state is authoritative.

Usage:
    python scripts/run_all_experiments.py \\
        --out-dir runs/exp_$(date +%Y%m%d-%H%M%S) \\
        --seeds 1,2,3 --phase1-workers 25 --workers 10
"""

from __future__ import annotations

import argparse, json, os, re, shutil, sqlite3, subprocess, sys, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from paper2manim.ablations import resolve as resolve_preset

SNAPSHOT_SIZES = [0, 50, 100, 200, 400]

# ── All paper conditions ────────────────────────────────────────────────────
# (cond_key, preset, emb_kind, subdir)
# emb_kind: "none" | "emb0" | "snapshot" | "random_snapshot" | "fresh" | "cross_snapshot"

CONDITIONS: list[dict] = [
    # ── Table 4 baselines (no EMB wait needed) ──
    {"key": "A",                "preset": "A",               "emb": "none",             "subdir": "probe"},
    {"key": "B",                "preset": "B",               "emb": "none",             "subdir": "probe"},
    {"key": "RAG",              "preset": "RAG",             "emb": "none",             "subdir": "probe"},
    {"key": "RAG_docs_only",    "preset": "RAG_docs_only",   "emb": "none",             "subdir": "probe"},
    {"key": "RAG_docs_api",     "preset": "RAG_docs_api",    "emb": "none",             "subdir": "probe"},
    {"key": "EMB@0",            "preset": "C",               "emb": "snapshot",         "subdir": "probe"},
    {"key": "Random_EMB@400",   "preset": "Random_EMB",      "emb": "random_snapshot",  "subdir": "probe"},
    # ── EMB@K snapshots (need merge done) ──
    {"key": "EMB@50",           "preset": "C",               "emb": "snapshot",         "subdir": "probe"},
    {"key": "EMB@100",          "preset": "C",               "emb": "snapshot",         "subdir": "probe"},
    {"key": "EMB@200",          "preset": "C",               "emb": "snapshot",         "subdir": "probe"},
    {"key": "EMB@400",          "preset": "C",               "emb": "snapshot",         "subdir": "probe"},
    # ── Table 6 ablations (need populated EMB) ──
    {"key": "C_no_vlm",              "preset": "C_no_vlm",              "emb": "snapshot", "subdir": "ablation"},
    {"key": "C_no_success_channel",  "preset": "C_no_success_channel",  "emb": "snapshot", "subdir": "ablation"},
    {"key": "C_no_failure_channel",  "preset": "C_no_failure_channel",  "emb": "snapshot", "subdir": "ablation"},
    {"key": "C_rationale_only",      "preset": "C_rationale_only",      "emb": "snapshot", "subdir": "ablation"},
    {"key": "C_no_failure_gate",     "preset": "C_no_failure_gate",     "emb": "snapshot", "subdir": "ablation"},
    {"key": "C_top1",                "preset": "C_top1",                "emb": "snapshot", "subdir": "ablation"},
    # ── Cross-domain (no EMB wait needed) ──
    {"key": "A_cross",          "preset": "A",               "emb": "none",             "subdir": "cross"},
    {"key": "B_cross",          "preset": "B",               "emb": "none",             "subdir": "cross"},
    {"key": "C_cross",          "preset": "C",               "emb": "cross_snapshot",   "subdir": "cross"},
    {"key": "RAG_cross",        "preset": "RAG",             "emb": "none",             "subdir": "cross"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset_tasks(split: str) -> list[dict]:
    root = PROJECT_ROOT / "data" / "p2m_bench_v2"
    index = json.loads((root / "dataset_index.json").read_text())
    payload = json.loads((root / index["single_json_path"]).read_text())
    return [t for t in payload["tasks"] if t.get("split") == split]


def task_to_spec(task: dict, idx: int) -> dict:
    mi = task.get("model_input", {})
    return {
        "arxiv_id": mi.get("arxiv_id", task.get("arxiv_id", "")),
        "section": mi.get("section", task.get("section", "")),
        "domain": task.get("domain", mi.get("domain", "")),
        "task_id": task.get("task_id", f"task_{idx:04d}"),
        "idx": idx,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EMB helpers
# ═══════════════════════════════════════════════════════════════════════════════

def count_emb_records(emb_path: Path) -> int:
    db = emb_path / "memory.db"
    if not db.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        c = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
        conn.close()
        return c
    except Exception:
        return 0


def create_empty_emb(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    db = path / "memory.db"
    if db.exists():
        return
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE IF NOT EXISTS memory_records (
        id TEXT PRIMARY KEY, polarity TEXT NOT NULL, run_id TEXT NOT NULL DEFAULT '',
        scene_id TEXT NOT NULL DEFAULT '', extraction_source TEXT NOT NULL DEFAULT '',
        transition_ordinal INTEGER NOT NULL DEFAULT 0, domain TEXT NOT NULL DEFAULT '',
        context_json TEXT NOT NULL, body_json TEXT NOT NULL,
        provenance_json TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)""")
    conn.commit()
    conn.close()


def copy_emb_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def merge_emb_dirs(src_dirs: list[Path], dst: Path) -> int:
    """Merge EMBs. Idempotent — deduplicates by record ID. Returns total records."""
    from paper2manim.emb.store import SQLiteMemoryStore
    from paper2manim.emb.schema import MemoryRecord

    dst.mkdir(parents=True, exist_ok=True)
    create_empty_emb(dst)

    store = SQLiteMemoryStore(dst / "memory.db")
    seen_ids: set[str] = {r.id for r in store.all()}

    total_added = 0
    for src in src_dirs:
        db = src / "memory.db"
        if not db.exists():
            continue
        src_store = SQLiteMemoryStore(db)
        for rec in src_store.all():
            if rec.id in seen_ids:
                continue
            if not rec.context.task_embedding:
                continue
            store.put(rec)
            seen_ids.add(rec.id)
            total_added += 1

    return store.count()


# ═══════════════════════════════════════════════════════════════════════════════
# Subprocess runner
# ═══════════════════════════════════════════════════════════════════════════════

_RUN_ID_RE = re.compile(r"RUN_ID=(\S+)")


def build_cli(preset: str, task_spec: dict, **kw) -> list[str]:
    flags = list(resolve_preset(preset))
    cmd = [
        sys.executable, "-m", "paper2manim", "mvp2",
        "--arxiv", task_spec["arxiv_id"],
        "--quality", kw.get("quality", "l"),
        "--max-retries", str(kw.get("max_retries", 3)),
        "--max-visual-revisions", str(kw.get("max_visual_revisions", 2)),
        "--scene-parallelism", "1",
    ]
    if task_spec["section"]:
        cmd.extend(["--section", task_spec["section"]])
    if task_spec["domain"]:
        cmd.extend(["--dataset-domain", task_spec["domain"]])
    if kw.get("no_render"):
        cmd.append("--no-render")
    if kw.get("allow_render_on_login", True):
        cmd.append("--allow-render-on-login")
    cmd.extend(flags)
    if "--emb" in flags and kw.get("emb_store_path"):
        cmd.extend(["--emb-store-path", kw["emb_store_path"]])
    if "--emb" in flags:
        cmd.append("--emb-fake-embedder")  # skip sentence-transformers download
    if "--emb" in flags and kw.get("emb_readonly"):
        cmd.append("--emb-readonly")
    if "--emb" in flags and kw.get("emb_theta_high") is not None:
        cmd.extend(["--emb-theta-high", str(kw["emb_theta_high"])])
    if "--emb" in flags and kw.get("emb_failure_min_margin") is not None:
        cmd.extend(["--emb-failure-min-margin", str(kw["emb_failure_min_margin"])])
    return cmd


def run_subprocess(cmd: list[str], timeout_s: float, cwd: str,
                   log_dir: Path | None = None) -> dict:
    env = os.environ.copy()
    env.setdefault("PAPER2MANIM_RUNS_DIR", str(PROJECT_ROOT / "runs"))
    # Ensure manim CLI is findable (conda bin may not be in nohup PATH)
    path = env.get("PATH", "")
    for d in ["/root/miniconda3/bin", "/usr/local/bin"]:
        if d not in path:
            env["PATH"] = f"{d}:{path}"
    # Redirect to files to avoid pipe-buffer deadlock when subprocess
    # output exceeds the OS pipe size (common with Manim render logs).
    stdout_path = None
    stderr_path = None
    try:
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / "stdout.log"
            stderr_path = log_dir / "stderr.log"
            stdout_fh = open(stdout_path, "w")
            stderr_fh = open(stderr_path, "w")
            proc = subprocess.run(cmd, stdout=stdout_fh, stderr=stderr_fh,
                                  text=True, timeout=timeout_s, cwd=cwd, env=env)
            stdout_fh.close()
            stderr_fh.close()
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_s, cwd=cwd, env=env)
        ec = proc.returncode
    except subprocess.TimeoutExpired:
        ec = 124
    except Exception:
        ec = 2
    run_id = None
    if stdout_path and stdout_path.exists():
        for line in stdout_path.read_text().splitlines():
            m = _RUN_ID_RE.search(line)
            if m:
                run_id = m.group(1)
                break
    elif not stdout_path:
        # Fallback: read from captured stdout (no log_dir case)
        pass
    return {"exit_code": ec, "run_id": run_id}


# ═══════════════════════════════════════════════════════════════════════════════
# Logger
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint primitives — the only source of truth for resume
# ═══════════════════════════════════════════════════════════════════════════════

class Checkpoint:
    """Granular on-disk checkpointing so any interruption is recoverable."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.state_dir = out_dir / ".checkpoints"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- Phase 1 per-task status -----------------------------------------------

    def phase1_task_status(self, idx: int) -> str:
        """Return 'done', 'failed', or 'pending'."""
        state_file = self.state_dir / "phase1_state.json"
        with self._lock:
            if state_file.exists():
                states = json.loads(state_file.read_text())
                return states.get(f"task_{idx}", "pending")
        return "pending"

    def phase1_mark_done(self, idx: int) -> None:
        self._phase1_update(idx, "done")

    def phase1_mark_failed(self, idx: int) -> None:
        self._phase1_update(idx, "failed")

    def _phase1_update(self, idx: int, status: str) -> None:
        state_file = self.state_dir / "phase1_state.json"
        with self._lock:
            states = {}
            if state_file.exists():
                states = json.loads(state_file.read_text())
            states[f"task_{idx}"] = status
            tmp = state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(states))
            tmp.replace(state_file)

    def phase1_pending_tasks(self, total: int) -> list[int]:
        """Return indices of tasks that still need to run."""
        pending = []
        for i in range(total):
            status = self.phase1_task_status(i)
            if status != "done":
                pending.append(i)
        return pending

    # -- Phase 2 manifest (streaming, per-condition×seed) ---------------------

    def phase2_completed_tasks(self, cond_key: str, seed: int, subdir: str) -> set[int]:
        """Return set of task_idx that have exit_code==0 in existing manifest."""
        mp = self.out_dir / subdir / f"{cond_key}_seed{seed}" / "manifest.json"
        if not mp.exists():
            return set()
        try:
            entries = json.loads(mp.read_text())
        except Exception:
            return set()
        return {e["_task_idx"] for e in entries if e.get("exit_code") == 0}

    def phase2_write_result(self, cond_key: str, seed: int, subdir: str, result: dict) -> None:
        """Atomically append/update one result to the condition's manifest."""
        cond_dir = self.out_dir / subdir / f"{cond_key}_seed{seed}"
        cond_dir.mkdir(parents=True, exist_ok=True)
        mp = cond_dir / "manifest.json"
        # Per-manifest lock
        lock = self._manifest_lock(str(mp))
        with lock:
            entries: list[dict] = []
            if mp.exists():
                try:
                    entries = json.loads(mp.read_text())
                except Exception:
                    entries = []
            # Replace or append by task_idx
            existing = {e["_task_idx"]: i for i, e in enumerate(entries)}
            tidx = result.get("_task_idx", -1)
            if tidx in existing:
                entries[existing[tidx]] = result
            else:
                entries.append(result)
            tmp = mp.with_suffix(".tmp")
            tmp.write_text(json.dumps(entries, indent=2, default=str))
            tmp.replace(mp)

    # -- Merge state -----------------------------------------------------------

    @property
    def merge_done(self) -> bool:
        return (self.state_dir / "merge_done").exists()

    def mark_merge_done(self) -> None:
        (self.state_dir / "merge_done").touch()

    # -- Snapshot state --------------------------------------------------------

    def snapshot_exists(self, k: int) -> bool:
        snap = self.out_dir / "snapshots" / f"EMB@{k}"
        return snap.exists() and count_emb_records(snap) >= k

    # -- Internal --------------------------------------------------------------

    _manifest_locks: dict[str, threading.Lock] = {}

    @classmethod
    def _manifest_lock(cls, path: str) -> threading.Lock:
        if path not in cls._manifest_locks:
            cls._manifest_locks[path] = threading.Lock()
        return cls._manifest_locks[path]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Fully parallel memory build → independent temp EMBs
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(args, ck: Checkpoint, memory_tasks: list[dict]) -> int:
    """Launch all memory_build tasks in parallel to independent temp EMBs.

    Returns total records across all completed temp EMBs.
    On resume, skips tasks whose temp EMB already has records.
    """
    total = len(memory_tasks)
    pending = ck.phase1_pending_tasks(total)

    # Also skip tasks whose temp dir already has a non-empty memory.db.
    # (create_empty_emb writes a schema-only db upfront, so existence alone is
    #  not enough to decide the task completed successfully.)
    actually_pending = []
    for idx in pending:
        temp_dir = args.out_dir / ".phase1_tmp" / f"task_{idx:04d}"
        if count_emb_records(temp_dir) > 0:
            ck.phase1_mark_done(idx)
            continue
        actually_pending.append(idx)

    if not actually_pending:
        # All done — count total records
        total_records = 0
        for idx in range(total):
            temp_dir = args.out_dir / ".phase1_tmp" / f"task_{idx:04d}"
            total_records += count_emb_records(temp_dir)
        log(f"[Phase1] All {total} tasks already done ({total_records} records)")
        return total_records

    log(f"[Phase1] {len(actually_pending)}/{total} pending, launching in parallel")

    # Build payloads for pending tasks
    payloads = []
    for idx in actually_pending:
        task = memory_tasks[idx]
        spec = task_to_spec(task, idx)
        temp_dir = args.out_dir / ".phase1_tmp" / f"task_{idx:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        create_empty_emb(temp_dir)

        cmd = build_cli(
            "C", spec,
            quality=args.quality,
            max_retries=args.max_retries,
            max_visual_revisions=args.max_visual_revisions,
            no_render=args.no_render,
            allow_render_on_login=args.allow_render_on_login,
            emb_store_path=str(temp_dir),
            emb_theta_high=args.emb_theta_high,
            emb_failure_min_margin=args.emb_failure_min_margin,
            paper_dir=args.paper_dir,
        )
        payloads.append({
            "cmd": cmd, "timeout_s": args.per_task_timeout,
            "cwd": str(PROJECT_ROOT), "task_idx": idx,
            "arxiv_id": spec["arxiv_id"],
            "section": spec["section"], "domain": spec["domain"],
            "task_id": spec["task_id"],
        })

    t0 = time.time()
    workers = min(len(payloads), args.phase1_workers)

    def _worker(p: dict) -> dict:
        log_dir = temp_dir = args.out_dir / ".phase1_tmp" / f"task_{p['task_idx']:04d}"
        r = run_subprocess(p["cmd"], timeout_s=p["timeout_s"], cwd=p["cwd"],
                           log_dir=log_dir)
        r["_task_idx"] = p["task_idx"]; r["_arxiv_id"] = p["arxiv_id"]
        r["_section"] = p["section"]; r["_domain"] = p["domain"]
        r["_task_id"] = p["task_id"]
        return r

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, p): p for p in payloads}
        for future in as_completed(futures):
            r = future.result()
            idx = r["_task_idx"]
            n_records = count_emb_records(args.out_dir / ".phase1_tmp" / f"task_{idx:04d}")
            completed += 1
            if r["exit_code"] == 0:
                ck.phase1_mark_done(idx)
            else:
                ck.phase1_mark_failed(idx)
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            log(f"[Phase1] {completed}/{len(actually_pending)} "
                f"task {idx}: exit={r['exit_code']} records={n_records} "
                f"rate={rate:.1f}/s elapsed={elapsed:.0f}s")

    # Count total
    total_records = 0
    for idx in range(total):
        temp_dir = args.out_dir / ".phase1_tmp" / f"task_{idx:04d}"
        total_records += count_emb_records(temp_dir)

    log(f"[Phase1] Done: {total_records} total records across {total} tasks "
        f"in {time.time() - t0:.0f}s")
    return total_records


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Probe evaluations — all conditions at max concurrency
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase2(args, ck: Checkpoint, probe_tasks: list[dict], cross_tasks: list[dict],
               conditions: list[dict]) -> None:
    """Launch all condition×seed evaluations.

    All conditions compete for the shared thread pool. Called after Phase 1
    merge + snapshot freeze, so EMB snapshots are always available.
    """
    total_launched = 0

    # Collect all (cond, seed, task) payloads, skipping already-completed
    payloads = []
    for g in conditions:
        cond_key = g["key"]
        preset = g["preset"]
        subdir = g["subdir"]
        cond_tasks = cross_tasks if subdir == "cross" else probe_tasks
        done = ck.phase2_completed_tasks(cond_key, g.get("_seed", 1), subdir)
        for i, task in enumerate(cond_tasks):
            if i in done:
                continue
            spec = task_to_spec(task, i)
            emb_store, emb_readonly = _resolve_emb(args, ck, g, cond_key, g.get("_seed", 1))
            cmd = build_cli(
                preset, spec,
                quality=args.quality,
                max_retries=args.max_retries,
                max_visual_revisions=args.max_visual_revisions,
                no_render=args.no_render,
                allow_render_on_login=args.allow_render_on_login,
                emb_store_path=emb_store,
                emb_readonly=emb_readonly,
                emb_theta_high=args.emb_theta_high,
                emb_failure_min_margin=args.emb_failure_min_margin,
                paper_dir=args.paper_dir,
            )
            payloads.append({
                "cmd": cmd, "timeout_s": args.per_task_timeout,
                "cwd": str(PROJECT_ROOT),
                "task_idx": i, "config": cond_key, "seed": g.get("_seed", 1),
                "arxiv_id": spec["arxiv_id"],
                "section": spec["section"], "domain": spec["domain"],
                "task_id": spec["task_id"], "preset": preset,
                "_subdir": subdir,
            })

    if not payloads:
        log(f"[Phase2] All {len(conditions)} conditions fully cached")
        return

    total_launched = len(payloads)
    log(f"[Phase2] Launching {len(payloads)} payloads for {len(conditions)} conditions")

    def _worker(p: dict) -> dict:
        log_dir = args.out_dir / ".p2_logs" / f"{p['config']}_seed{p['seed']}" / f"task_{p['task_idx']:04d}"
        r = run_subprocess(p["cmd"], timeout_s=p["timeout_s"], cwd=p["cwd"],
                           log_dir=log_dir)
        r["_task_idx"] = p["task_idx"]; r["_config"] = p["config"]
        r["_seed"] = p["seed"]; r["_arxiv_id"] = p["arxiv_id"]
        r["_section"] = p["section"]; r["_domain"] = p["domain"]
        r["_task_id"] = p["task_id"]; r["_preset"] = p["preset"]
        # Write to checkpoint IMMEDIATELY
        ck.phase2_write_result(p["config"], p["seed"], p["_subdir"], r)
        return r

    completed = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, p): p for p in payloads}
        for future in as_completed(futures):
            r = future.result()
            completed += 1
            elapsed = time.time() - t0
            if completed % max(1, len(payloads) // 10) == 0 or completed == len(payloads):
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(payloads) - completed) / rate / 60 if rate > 0 else 0
                log(f"[Phase2] {completed}/{len(payloads)} rate={rate:.1f}/s "
                    f"elapsed={elapsed/60:.0f}m eta={eta:.0f}m")

    log(f"[Phase2] All done ({total_launched} runs)")


# ═══════════════════════════════════════════════════════════════════════════════
# Snapshots (post-merge)
# ═══════════════════════════════════════════════════════════════════════════════

def freeze_snapshots(args, ck: Checkpoint, total_records: int) -> None:
    """Freeze EMB snapshots at {0, 50, 100, 200, 400}.

    Idempotent: skips snapshots that already exist.
    """
    main_emb = args.out_dir / "emb" / "C_seed_1"
    snaps_dir = args.out_dir / "snapshots"

    # EMB@0
    emb0 = snaps_dir / "EMB@0"
    if not ck.snapshot_exists(0):
        create_empty_emb(emb0)
        log("[Snap] EMB@0 created")

    # EMB@K
    for k in SNAPSHOT_SIZES:
        if k == 0:
            continue
        snap = snaps_dir / f"EMB@{k}"
        if ck.snapshot_exists(k):
            log(f"[Snap] EMB@{k} already exists ({count_emb_records(snap)} records)")
            continue
        if total_records >= k:
            copy_emb_dir(main_emb, snap)
            log(f"[Snap] EMB@{k} frozen ({count_emb_records(snap)} records)")
        else:
            log(f"[Snap] EMB@{k} UNREACHABLE — only {total_records} records (need {k})")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_emb(args, ck: Checkpoint, cond: dict, cond_key: str, seed: int) -> tuple[str | None, bool]:
    """Resolve (emb_store_path, emb_readonly) for a condition."""
    emb_kind = cond.get("emb", "none")
    if emb_kind == "none":
        return None, False
    if emb_kind == "fresh":
        path = args.out_dir / "emb_fresh" / f"{cond_key}_seed{seed}"
        path.mkdir(parents=True, exist_ok=True)
        create_empty_emb(path)
        return str(path), False
    if emb_kind == "snapshot":
        # Match the snapshot size from the key name, default to 400
        for k in sorted(SNAPSHOT_SIZES, reverse=True):
            if f"@{k}" in cond_key:
                snap = args.out_dir / "snapshots" / f"EMB@{k}"
                if snap.exists():
                    return str(snap), True
                break
        # fallback
        snap = args.out_dir / "snapshots" / "EMB@400"
        if snap.exists():
            return str(snap), True
        return None, True
    if emb_kind == "random_snapshot":
        snap = args.out_dir / "snapshots" / "EMB@400"
        if snap.exists():
            return str(snap), True
        return None, True
    if emb_kind == "cross_snapshot":
        snap = args.out_dir / "snapshots" / "EMB@400"
        if snap.exists():
            return str(snap), True
        # Fallback: use merged EMB
        main_emb = args.out_dir / "emb" / "C_seed_1"
        if main_emb.exists() and count_emb_records(main_emb) > 0:
            return str(main_emb), True
        return None, True
    return None, False


def _generate_report(args, ck: Checkpoint, probe_tasks: list[dict], cross_tasks: list[dict]) -> None:
    """Aggregate per-condition manifest stats."""
    from collections import Counter

    by_cond: dict[str, Counter] = {}
    for cond in CONDITIONS:
        key = cond["key"]
        subdir = cond["subdir"]
        for seed in args.seeds:
            mp = args.out_dir / subdir / f"{key}_seed{seed}" / "manifest.json"
            if not mp.exists():
                continue
            try:
                entries = json.loads(mp.read_text())
            except Exception:
                continue
            for e in entries:
                by_cond.setdefault(key, Counter())
                by_cond[key]["total"] += 1
                if e.get("exit_code") == 0:
                    by_cond[key]["ok"] += 1
                else:
                    by_cond[key]["err"] += 1

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"{'Condition':<28} {'Total':>6} {'OK':>6} {'Err':>6} {'OK%':>7}")
    print("-" * 60)

    for cond in CONDITIONS:
        c = by_cond.get(cond["key"])
        if not c:
            print(f"{cond['key']:<28} {'N/A':>6}")
            continue
        total = c["total"]
        ok = c.get("ok", 0)
        print(f"{cond['key']:<28} {total:>6} {ok:>6} {c.get('err', 0):>6} {100*ok/max(total,1):>6.1f}%")

    report = {
        "experiment": "ManimAgent EMNLP — full suite (max concurrency + checkpoint/resume)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "quality": args.quality,
            "seeds": args.seeds,
            "workers": args.workers,
            "phase1_workers": args.phase1_workers,
            "snapshot_sizes": SNAPSHOT_SIZES,
        },
        "results": {k: dict(v) for k, v in by_cond.items()},
    }
    report_path = args.out_dir / "results.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.seeds = [int(s.strip()) for s in args.seeds.split(",")]
    ck = Checkpoint(args.out_dir)

    log(f"Experiment: {args.out_dir}")
    log(f"Config: seeds={args.seeds} workers={args.workers} "
        f"phase1_workers={args.phase1_workers}")

    memory_tasks = load_dataset_tasks("memory_build")
    probe_tasks = load_dataset_tasks("fixed_probe")
    cross_tasks = load_dataset_tasks("cross_test")

    if args.quick_test:
        memory_tasks = memory_tasks[:2]
        probe_tasks = probe_tasks[:2]
        cross_tasks = cross_tasks[:2]

    log(f"Tasks: memory_build={len(memory_tasks)} fixed_probe={len(probe_tasks)} "
        f"cross_test={len(cross_tasks)}")

    # ═══════════════════════════════════════════════════════════════
    # Expand conditions with seeds (all run after Phase 1 + merge)
    # ═══════════════════════════════════════════════════════════════
    all_conditions: list[dict] = []

    # --conditions filter: only run specified keys
    cond_filter = set(args.conditions.split(",")) if args.conditions else None

    for cond in CONDITIONS:
        if cond_filter and cond["key"] not in cond_filter:
            continue
        for seed in args.seeds:
            c = dict(cond)
            c["_seed"] = seed
            all_conditions.append(c)

    log(f"Total conditions: {len(all_conditions)} condition×seed pairs")

    embed_dir = args.out_dir / "emb" / "C_seed_1"
    embed_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = args.out_dir / "snapshots"

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: memory build (no Phase 2 overlap → can use high concurrency)
    # ═══════════════════════════════════════════════════════════════
    if args.skip_phase1:
        log("[Phase1] SKIPPED (--skip-phase1)")
        total_records = 0
    else:
        total_records = run_phase1(args, ck, memory_tasks)
        log(f"[Phase1] Done: {total_records} total records")

    # ═══════════════════════════════════════════════════════════════
    # Merge + freeze snapshots
    # ═══════════════════════════════════════════════════════════════

    if total_records > 0 and not ck.merge_done:
        log("\n=== Merging EMBs ===")
        temp_dirs = sorted(
            (args.out_dir / ".phase1_tmp").glob("task_*"),
            key=lambda p: p.name,
        )
        valid_dirs = [d for d in temp_dirs if count_emb_records(d) > 0]
        log(f"Merging {len(valid_dirs)} temp EMBs into {embed_dir}...")
        merged_records = merge_emb_dirs(valid_dirs, embed_dir)
        ck.mark_merge_done()
        log(f"Merge done: {merged_records} records")
        total_records = merged_records
    elif ck.merge_done:
        total_records = count_emb_records(embed_dir)
        log(f"Merge already done: {total_records} records")

    # Idempotent snapshot freeze
    freeze_snapshots(args, ck, total_records)

    if args.phase1_only:
        log(f"\n=== Phase 1 complete (--phase1-only): {total_records} records, "
            f"snapshots frozen at {SNAPSHOT_SIZES} ===")
        log(f"Output ready for transfer: {args.out_dir}")
        _generate_report(args, ck, probe_tasks, cross_tasks)
        return

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: all conditions (merge + snapshots already done)
    # ═══════════════════════════════════════════════════════════════

    if all_conditions:
        log(f"\n=== Phase 2: {len(all_conditions)} conditions ===")
        run_phase2(args, ck, probe_tasks, cross_tasks, all_conditions)

    # ═══════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════

    _generate_report(args, ck, probe_tasks, cross_tasks)
    log(f"\nAll done. Output: {args.out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="ManimAgent EMNLP experiment suite")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--workers", type=int, default=10,
                   help="Max concurrent Phase 2 subprocesses. Default 10 (safe: ~2.5G parents + ~2.5G render children ≈ 5G peak).")
    p.add_argument("--phase1-workers", type=int, default=4,
                   help="Max parallel memory_build tasks. Keep low on CPU-only: marker-pdf OCR model I/O is heavy.")
    p.add_argument("--quality", type=str, default="l", choices=["l", "m", "h"])
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-visual-revisions", type=int, default=2)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--allow-render-on-login", action="store_true", default=True)
    p.add_argument("--per-task-timeout", type=float, default=3600)
    p.add_argument("--emb-theta-high", type=float, default=None)
    p.add_argument("--emb-failure-min-margin", type=float, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint (auto-detected, this is a no-op safety flag)")
    p.add_argument("--paper-dir", type=Path, default=None,
                   help="Directory with pre-downloaded arxiv .tar.gz files (avoids arxiv.org rate-limit).")
    p.add_argument("--phase1-only", action="store_true",
                   help="Stop after Phase 1 merge + snapshot freeze (skip Phase 2).")
    p.add_argument("--skip-phase1", action="store_true",
                   help="Skip Phase 1 entirely. Use for non-EMB conditions (A, B, RAG variants) "
                   "that don't need an EMB snapshot.")
    p.add_argument("--conditions", type=str, default="",
                   help="Comma-separated condition keys to run (e.g. 'A,B,RAG'). "
                   "Runs all conditions when empty.")
    p.add_argument("--quick-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
