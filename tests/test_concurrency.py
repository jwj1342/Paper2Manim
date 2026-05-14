"""Tests for paper2manim.concurrency: token bucket + render semaphore + configure."""

from __future__ import annotations

import threading
import time

import pytest

from paper2manim import concurrency


@pytest.fixture(autouse=True)
def _reset_concurrency():
    """Wipe module-level throttle state around every test."""
    concurrency.reset()
    yield
    concurrency.reset()


class TestTokenBucket:
    def test_starts_full(self):
        bucket = concurrency.TokenBucket(rate=10.0, burst=5)
        start = time.monotonic()
        for _ in range(5):
            bucket.acquire()
        elapsed = time.monotonic() - start
        # 5 starting tokens should be consumed without waiting.
        assert elapsed < 0.1

    def test_blocks_when_empty(self):
        bucket = concurrency.TokenBucket(rate=5.0, burst=1)
        bucket.acquire()  # drain the initial token
        start = time.monotonic()
        bucket.acquire()  # must wait ~0.2s for next refill
        elapsed = time.monotonic() - start
        assert 0.15 < elapsed < 0.45

    def test_sustained_rate_enforced(self):
        # 5 rps, burst=1: first call free, next 4 each ~0.2s -> ~0.8s
        bucket = concurrency.TokenBucket(rate=5.0, burst=1)
        start = time.monotonic()
        for _ in range(5):
            bucket.acquire()
        elapsed = time.monotonic() - start
        assert 0.6 < elapsed < 1.2

    def test_rejects_non_positive_rate(self):
        with pytest.raises(ValueError):
            concurrency.TokenBucket(rate=0.0)
        with pytest.raises(ValueError):
            concurrency.TokenBucket(rate=-1.0)

    def test_rejects_non_positive_burst(self):
        with pytest.raises(ValueError):
            concurrency.TokenBucket(rate=1.0, burst=0)

    def test_thread_safe(self):
        # Two threads compete for tokens after the initial token is drained.
        bucket = concurrency.TokenBucket(rate=10.0, burst=1)
        bucket.acquire()
        results: list[float] = []
        lock = threading.Lock()

        def worker():
            t0 = time.monotonic()
            bucket.acquire()
            with lock:
                results.append(time.monotonic() - t0)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        assert len(results) == 2
        # Both eventually acquired a token; at least one waited.
        assert max(results) >= 0.05


class TestConfigure:
    def test_defaults_off(self):
        concurrency.configure(scene_parallelism=1)
        assert concurrency.RENDER_SEMAPHORE is None
        assert concurrency.LLM_BUCKET is None

    def test_render_concurrency_sets_semaphore(self):
        concurrency.configure(scene_parallelism=4, render_concurrency=2)
        assert concurrency.RENDER_SEMAPHORE is not None

    def test_llm_rps_sets_bucket(self):
        concurrency.configure(scene_parallelism=4, llm_rps=3.0)
        assert concurrency.LLM_BUCKET is not None
        assert concurrency.LLM_BUCKET._rate == 3.0  # noqa: SLF001

    def test_reconfigure_replaces(self):
        concurrency.configure(scene_parallelism=4, render_concurrency=2, llm_rps=3.0)
        sem1 = concurrency.RENDER_SEMAPHORE
        bucket1 = concurrency.LLM_BUCKET
        concurrency.configure(scene_parallelism=4, render_concurrency=4, llm_rps=5.0)
        assert concurrency.RENDER_SEMAPHORE is not sem1
        assert concurrency.LLM_BUCKET is not bucket1

    def test_zero_or_none_disables(self):
        concurrency.configure(scene_parallelism=4, render_concurrency=0, llm_rps=0.0)
        assert concurrency.RENDER_SEMAPHORE is None
        assert concurrency.LLM_BUCKET is None


class TestRenderSlot:
    def test_passthrough_when_unconfigured(self):
        with concurrency.render_slot():
            pass  # must not block or raise

    def test_serializes_at_capacity_one(self):
        concurrency.configure(scene_parallelism=4, render_concurrency=1)
        peak = 0
        current = 0
        lock = threading.Lock()

        def worker():
            nonlocal peak, current
            with concurrency.render_slot():
                with lock:
                    current += 1
                    peak = max(peak, current)
                time.sleep(0.05)
                with lock:
                    current -= 1

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert peak == 1

    def test_capacity_two_allows_two_in_flight(self):
        concurrency.configure(scene_parallelism=4, render_concurrency=2)
        peak = 0
        current = 0
        lock = threading.Lock()

        def worker():
            nonlocal peak, current
            with concurrency.render_slot():
                with lock:
                    current += 1
                    peak = max(peak, current)
                time.sleep(0.1)
                with lock:
                    current -= 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert peak == 2


class TestLLMAcquire:
    def test_passthrough_when_unconfigured(self):
        start = time.monotonic()
        for _ in range(100):
            concurrency.llm_acquire()
        elapsed = time.monotonic() - start
        # 100 no-op calls should be effectively free.
        assert elapsed < 0.01

    def test_gated_under_bucket(self):
        concurrency.configure(scene_parallelism=4, llm_rps=5.0, llm_burst=1)
        start = time.monotonic()
        for _ in range(5):
            concurrency.llm_acquire()
        elapsed = time.monotonic() - start
        # 5 acquires at 5 rps with burst 1: 1 free + 4×0.2s ≈ 0.8s
        assert 0.6 < elapsed < 1.2


class TestRenderSemaphoreIntegration:
    """End-to-end: actual render() respects RENDER_SEMAPHORE via render_slot()."""

    def _mock_render_call(self, monkeypatch, peak_tracker):
        """Patch subprocess.run + static checker so render() returns success fast.

        ``peak_tracker`` is a 2-list [peak, current] mutated via a lock.
        Note: render.py imports both ``validate_manim_code`` and ``subprocess``
        by name, so we patch the names bound *in render's module*.
        """
        from paper2manim.sandbox import render as render_mod

        monkeypatch.setattr(render_mod, "validate_manim_code", lambda code: None)

        track_lock = threading.Lock()

        class _FakeProc:
            stderr = ""
            returncode = 0

        def fake_run(*args, **kwargs):
            with track_lock:
                peak_tracker[1] += 1
                peak_tracker[0] = max(peak_tracker[0], peak_tracker[1])
            time.sleep(0.05)
            with track_lock:
                peak_tracker[1] -= 1
            return _FakeProc()

        monkeypatch.setattr(render_mod.subprocess, "run", fake_run)

    def test_passthrough_when_no_semaphore(self, monkeypatch, tmp_path):
        from paper2manim.sandbox.render import render

        peak_tracker = [0, 0]
        self._mock_render_call(monkeypatch, peak_tracker)

        def worker(i):
            render("code", "S", workdir=tmp_path / f"w{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        # No semaphore → expect overlap (peak >= 2).
        assert peak_tracker[0] >= 2

    def test_serialized_at_capacity_one(self, monkeypatch, tmp_path):
        from paper2manim.sandbox.render import render

        concurrency.configure(scene_parallelism=3, render_concurrency=1)
        peak_tracker = [0, 0]
        self._mock_render_call(monkeypatch, peak_tracker)

        def worker(i):
            render("code", "S", workdir=tmp_path / f"w{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        # Semaphore=1 forces serial Manim execution.
        assert peak_tracker[0] == 1
