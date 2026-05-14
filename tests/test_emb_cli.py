"""B4 tests for ``paper2manim emb`` subcommand group.

Cover:

* ``HitStats.of`` quantile math on small / empty / uniform inputs.
* ``collect_stats`` over an empty store and a populated store.
* ``select_cold_records`` filter (hit_count + age + polarity).
* CliRunner end-to-end on stats / list / show / prune (dry-run + apply).
"""

from __future__ import annotations

import json
import time

import pytest
from click.testing import CliRunner

from paper2manim.cli import cli
from paper2manim.cli_emb import (
    HitStats,
    collect_stats,
    select_cold_records,
)
from paper2manim.emb.manager import build_default_emb
from paper2manim.emb.schema import (
    Context,
    FailureBody,
    MemoryRecord,
    Provenance,
    SuccessBody,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def empty_store(tmp_path):
    """A fresh on-disk EMB at tmp_path/emb_store, no records."""
    p = tmp_path / "emb_store"
    p.mkdir()
    # Build once just to lay down memory.db + indices files.
    build_default_emb(str(p), use_real_embedder=False)
    return p


@pytest.fixture
def populated_store(tmp_path):
    """EMB with 3 success + 2 failure records spanning various hit_count and
    last_used values to exercise stats / prune filters."""
    p = tmp_path / "emb_store"
    p.mkdir()
    emb = build_default_emb(str(p), use_real_embedder=False)
    now = time.time()
    # Two cold success records (hit=0, old) + one warm (hit=10, recent)
    emb.put(MemoryRecord(
        polarity="success",
        context=Context(task_text="cold scene 1", source_paper="arxiv:cold1"),
        body=SuccessBody(rationale="r1", code_full="# c1"),
        provenance=Provenance(
            run_id="run-cold-1", scene_id="S1",
            extraction_source="high_score_scene",
            validated=True, vlm_score=88.0,
            hit_count=0, last_used=now - 60 * 86400,  # 60 days ago
        ),
    ))
    emb.put(MemoryRecord(
        polarity="success",
        context=Context(task_text="cold scene 2", source_paper="arxiv:cold2"),
        body=SuccessBody(rationale="r2", code_full="# c2"),
        provenance=Provenance(
            run_id="run-cold-2", scene_id="S2",
            extraction_source="high_score_scene",
            validated=True, vlm_score=82.0,
            hit_count=0, last_used=now - 45 * 86400,
        ),
    ))
    emb.put(MemoryRecord(
        polarity="success",
        context=Context(task_text="warm scene", source_paper="arxiv:warm"),
        body=SuccessBody(rationale="r3", code_full="# c3"),
        provenance=Provenance(
            run_id="run-warm", scene_id="S3",
            extraction_source="high_score_scene",
            validated=True, vlm_score=92.0,
            hit_count=10, last_used=now - 1 * 86400,
        ),
    ))
    # Two failure records: one cold, one moderately used
    emb.put(MemoryRecord(
        polarity="failure",
        context=Context(task_text="failure 1", source_paper="arxiv:f1"),
        body=FailureBody(
            trigger_pattern="t1", root_cause="c1", fix_recipe="f1",
        ),
        provenance=Provenance(
            run_id="run-f1", scene_id="F1",
            extraction_source="visual_reflection",
            validated=True, before_score=30.0, after_score=80.0,
            hit_count=0, last_used=now - 90 * 86400,
        ),
    ))
    emb.put(MemoryRecord(
        polarity="failure",
        context=Context(task_text="failure 2", source_paper="arxiv:f2"),
        body=FailureBody(
            trigger_pattern="t2", root_cause="c2", fix_recipe="f2",
        ),
        provenance=Provenance(
            run_id="run-f2", scene_id="F2",
            extraction_source="text_reflection",
            validated=True, before_score=0.0, after_score=1.0,
            hit_count=3, last_used=now - 5 * 86400,
        ),
    ))
    emb.save_indices()
    return p


@pytest.fixture
def runner():
    return CliRunner()


# --------------------------------------------------------------------------- #
# HitStats
# --------------------------------------------------------------------------- #


class TestHitStats:
    def test_empty(self):
        s = HitStats.of([])
        assert s.n == 0 and s.median == 0 and s.maximum == 0

    def test_singleton(self):
        s = HitStats.of([5])
        assert s.n == 1
        assert s.minimum == s.median == s.maximum == 5
        assert s.n_zero == 0

    def test_quantiles_match_numpy_linear(self):
        # Hand-checked against numpy.percentile([0,1,2,3,4,5,6,7,8,9], [25,50,75])
        # = [2.25, 4.5, 6.75] → rounded to int with banker's-style tiebreak.
        s = HitStats.of([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert s.minimum == 0
        assert s.maximum == 9
        assert s.median == 4 or s.median == 5  # 4.5 rounds either way
        assert s.n_zero == 1

    def test_n_zero_counts_zeros(self):
        s = HitStats.of([0, 0, 0, 5, 10])
        assert s.n_zero == 3


# --------------------------------------------------------------------------- #
# collect_stats
# --------------------------------------------------------------------------- #


class TestCollectStats:
    def test_empty_store(self, empty_store):
        emb = build_default_emb(str(empty_store), use_real_embedder=False)
        s = collect_stats(emb)
        assert s["total"] == 0
        assert s["success"] == 0
        assert s["failure"] == 0
        # No records → no hit_stats entries beyond zeroed defaults
        assert s["success_hit_stats"]["n"] == 0
        assert s["failure_hit_stats"]["n"] == 0

    def test_populated_store(self, populated_store):
        emb = build_default_emb(str(populated_store), use_real_embedder=False)
        s = collect_stats(emb)
        assert s["total"] == 5
        assert s["success"] == 3
        assert s["failure"] == 2
        # success hits = [0, 0, 10] → n=3 zero=2 max=10
        assert s["success_hit_stats"]["n"] == 3
        assert s["success_hit_stats"]["n_zero"] == 2
        assert s["success_hit_stats"]["max"] == 10
        assert s["success_last_used_age_days"]["n_with_last_used"] == 3


# --------------------------------------------------------------------------- #
# select_cold_records
# --------------------------------------------------------------------------- #


class TestSelectColdRecords:
    def test_default_thresholds_pick_cold_zero_hit_old_records(self, populated_store):
        emb = build_default_emb(str(populated_store), use_real_embedder=False)
        cold = select_cold_records(
            emb, populated_store, cold_hit_threshold=0, max_age_days=30,
        )
        # Cold success 1 (60d), cold success 2 (45d), failure 1 (90d) — all
        # have hit=0 AND age > 30d. Warm success (1d) and failure 2 (hit=3) excluded.
        assert len(cold) == 3, f"expected 3, got {[(r.id[:8], r.polarity) for r in cold]}"

    def test_polarity_filter(self, populated_store):
        emb = build_default_emb(str(populated_store), use_real_embedder=False)
        cold_succ = select_cold_records(
            emb, populated_store,
            cold_hit_threshold=0, max_age_days=30, polarity="success",
        )
        for r in cold_succ:
            assert r.polarity == "success"
        assert len(cold_succ) == 2

    def test_max_age_too_long_returns_empty(self, populated_store):
        emb = build_default_emb(str(populated_store), use_real_embedder=False)
        cold = select_cold_records(
            emb, populated_store, cold_hit_threshold=0, max_age_days=365,
        )
        assert cold == []

    def test_higher_hit_threshold_admits_more_records(self, populated_store):
        emb = build_default_emb(str(populated_store), use_real_embedder=False)
        # Threshold 5 → still excludes warm success (10) but includes failure 2 (3)
        cold = select_cold_records(
            emb, populated_store, cold_hit_threshold=5, max_age_days=1,
        )
        # Failure 2 has age 5 days > 1 day cutoff and hit=3 ≤ 5 → included
        assert len(cold) == 4, f"got {[(r.id[:8], r.polarity, r.provenance.hit_count) for r in cold]}"


# --------------------------------------------------------------------------- #
# CLI commands
# --------------------------------------------------------------------------- #


class TestEmbCli:
    def test_stats_default_table(self, runner, populated_store):
        result = runner.invoke(cli, ["emb", "stats", "--store-path", str(populated_store)])
        assert result.exit_code == 0, result.output
        assert "total" in result.output
        assert "success" in result.output
        assert "failure" in result.output

    def test_stats_json_emits_parseable_payload(self, runner, populated_store):
        result = runner.invoke(
            cli, ["emb", "stats", "--store-path", str(populated_store), "--json"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["total"] == 5

    def test_list_default_shows_records(self, runner, populated_store):
        result = runner.invoke(cli, ["emb", "list", "--store-path", str(populated_store)])
        assert result.exit_code == 0, result.output
        assert "arxiv:warm" in result.output or "arxiv:cold1" in result.output

    def test_list_polarity_filter(self, runner, populated_store):
        result = runner.invoke(cli, [
            "emb", "list", "--store-path", str(populated_store),
            "--polarity", "failure",
        ])
        assert result.exit_code == 0, result.output
        # Should show failure-related sources but not success ones
        assert "arxiv:f1" in result.output or "arxiv:f2" in result.output
        assert "arxiv:warm" not in result.output

    def test_list_source_paper_substring_filter(self, runner, populated_store):
        result = runner.invoke(cli, [
            "emb", "list", "--store-path", str(populated_store),
            "--source-paper", "warm",
        ])
        assert result.exit_code == 0
        assert "arxiv:warm" in result.output
        assert "arxiv:cold1" not in result.output

    def test_show_by_id_prefix(self, runner, populated_store):
        # Find any record id from the store
        emb = build_default_emb(str(populated_store), use_real_embedder=False)
        recs = emb.all()
        rid = recs[0].id
        result = runner.invoke(cli, [
            "emb", "show", "--store-path", str(populated_store), rid[:6],
        ])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["id"] == rid

    def test_show_unknown_id_errors(self, runner, populated_store):
        result = runner.invoke(cli, [
            "emb", "show", "--store-path", str(populated_store), "deadbeef",
        ])
        assert result.exit_code != 0
        assert "no record matches" in result.output

    def test_prune_dry_run_does_not_delete(self, runner, populated_store):
        emb_pre = build_default_emb(str(populated_store), use_real_embedder=False)
        n_before = emb_pre.count()
        result = runner.invoke(cli, [
            "emb", "prune", "--store-path", str(populated_store),
            "--cold-hit-threshold", "0", "--max-age-days", "30",
        ])
        assert result.exit_code == 0, result.output
        assert "[dry-run]" in result.output
        emb_post = build_default_emb(str(populated_store), use_real_embedder=False)
        assert emb_post.count() == n_before

    def test_prune_apply_actually_deletes(self, runner, populated_store):
        emb_pre = build_default_emb(str(populated_store), use_real_embedder=False)
        n_before = emb_pre.count()
        result = runner.invoke(cli, [
            "emb", "prune", "--store-path", str(populated_store),
            "--cold-hit-threshold", "0", "--max-age-days", "30",
            "--apply",
        ])
        assert result.exit_code == 0, result.output
        assert "deleted" in result.output
        emb_post = build_default_emb(str(populated_store), use_real_embedder=False)
        assert emb_post.count() < n_before

    def test_stats_on_missing_store_fails_loudly(self, runner, tmp_path):
        result = runner.invoke(cli, [
            "emb", "stats", "--store-path", str(tmp_path / "does_not_exist"),
        ])
        assert result.exit_code != 0
        assert "EMB store not found" in result.output
