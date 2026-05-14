"""Tests for the ``RateLimitedLLM`` proxy in ``paper2manim.llm``.

Confirms the wrapper:
  - delegates attribute access transparently
  - calls ``concurrency.llm_acquire`` before every network method
  - keeps the throttle on ``with_structured_output(...).invoke(...)`` chains
  - composes with ``|`` and ``bind_tools`` without losing the throttle
  - is a near-zero-overhead pass-through when no bucket is configured
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from paper2manim import concurrency
from paper2manim.llm import RateLimitedLLM, _ThrottledRunnable


@pytest.fixture(autouse=True)
def _reset_concurrency():
    concurrency.reset()
    yield
    concurrency.reset()


# --------------------------------------------------------------------------- #
# Basic delegation
# --------------------------------------------------------------------------- #


class TestDelegation:
    def test_invoke_delegates(self):
        inner = MagicMock()
        inner.invoke.return_value = "ok"
        proxy = RateLimitedLLM(inner)
        out = proxy.invoke([("user", "hello")])
        assert out == "ok"
        inner.invoke.assert_called_once_with([("user", "hello")], None)

    def test_attribute_delegated(self):
        inner = MagicMock()
        inner.model_name = "fake-model"
        proxy = RateLimitedLLM(inner)
        assert proxy.model_name == "fake-model"

    def test_batch_delegates_per_item_acquire(self):
        concurrency.configure(scene_parallelism=2, llm_rps=10.0, llm_burst=1)
        inner = MagicMock()
        inner.batch.return_value = ["a", "b", "c"]
        proxy = RateLimitedLLM(inner)
        # Drain the initial token first so wall time reflects 3 acquires
        concurrency.llm_acquire()
        start = time.monotonic()
        result = proxy.batch([1, 2, 3])
        elapsed = time.monotonic() - start
        assert result == ["a", "b", "c"]
        # 3 acquires at 10rps burst=1 (already drained): ~0.2s
        assert elapsed >= 0.15


# --------------------------------------------------------------------------- #
# Throttling
# --------------------------------------------------------------------------- #


class TestThrottle:
    def test_invoke_acquires_before_delegating(self):
        order: list[str] = []

        class _RecordingBucket:
            def acquire(self):
                order.append("acquire")

        concurrency.LLM_BUCKET = _RecordingBucket()  # type: ignore[assignment]
        inner = MagicMock()

        def _inv(*_a, **_kw):
            order.append("invoke")
            return "result"

        inner.invoke.side_effect = _inv
        proxy = RateLimitedLLM(inner)
        proxy.invoke("hello")
        assert order == ["acquire", "invoke"]

    def test_invoke_throttled_at_rps_1(self):
        concurrency.configure(scene_parallelism=4, llm_rps=5.0, llm_burst=1)
        inner = MagicMock()
        inner.invoke.return_value = "ok"
        proxy = RateLimitedLLM(inner)
        start = time.monotonic()
        for _ in range(5):
            proxy.invoke("hello")
        elapsed = time.monotonic() - start
        # 5 invokes at 5rps burst=1: 1 free + 4×0.2s ≈ 0.8s
        assert 0.6 < elapsed < 1.2

    def test_passthrough_overhead_when_no_bucket(self):
        inner = MagicMock()
        inner.invoke.return_value = "ok"
        proxy = RateLimitedLLM(inner)
        start = time.monotonic()
        for _ in range(1000):
            proxy.invoke("x")
        elapsed = time.monotonic() - start
        # 1000 wrapped invokes should be much faster than 0.5s on any modern box.
        assert elapsed < 0.5


# --------------------------------------------------------------------------- #
# Chain integrity (structured output, __or__, bind_tools)
# --------------------------------------------------------------------------- #


class TestChainPropagation:
    def test_with_structured_output_returns_throttled_runnable(self):
        inner = MagicMock()
        inner_runnable = MagicMock()
        inner_runnable.invoke.return_value = "structured-out"
        inner.with_structured_output.return_value = inner_runnable
        proxy = RateLimitedLLM(inner)
        structured = proxy.with_structured_output(MagicMock())
        assert isinstance(structured, _ThrottledRunnable)
        # Calling .invoke on the structured runnable should hit our throttle.
        order: list[str] = []

        class _RecordingBucket:
            def acquire(self):
                order.append("acquire")

        concurrency.LLM_BUCKET = _RecordingBucket()  # type: ignore[assignment]

        def _inv(*_a, **_kw):
            order.append("invoke")
            return "structured-out"

        inner_runnable.invoke.side_effect = _inv
        out = structured.invoke([("user", "hello")])
        assert out == "structured-out"
        assert order == ["acquire", "invoke"]

    def test_or_composition_keeps_throttle(self):
        inner = MagicMock()
        composed_inner = MagicMock()
        composed_inner.invoke.return_value = "composed-out"
        inner.__or__.return_value = composed_inner
        proxy = RateLimitedLLM(inner)
        seq = proxy | "fake_parser"
        assert isinstance(seq, _ThrottledRunnable)
        order: list[str] = []

        class _RecordingBucket:
            def acquire(self):
                order.append("acquire")

        concurrency.LLM_BUCKET = _RecordingBucket()  # type: ignore[assignment]

        def _inv(*_a, **_kw):
            order.append("invoke")
            return "composed-out"

        composed_inner.invoke.side_effect = _inv
        out = seq.invoke("hi")
        assert out == "composed-out"
        assert order == ["acquire", "invoke"]

    def test_bind_tools_returns_wrapped(self):
        inner = MagicMock()
        bound = MagicMock()
        inner.bind_tools.return_value = bound
        proxy = RateLimitedLLM(inner)
        wrapped = proxy.bind_tools([{"name": "tool"}])
        assert isinstance(wrapped, RateLimitedLLM)
        assert wrapped._llm is bound  # noqa: SLF001

    def test_bind_returns_wrapped_and_invoke_throttled(self):
        """``llm.bind(stop=...)`` must keep the throttle on its .invoke()."""
        inner = MagicMock()
        bound = MagicMock()
        bound.invoke.return_value = "bound-out"
        inner.bind.return_value = bound
        proxy = RateLimitedLLM(inner)
        wrapped = proxy.bind(stop=["x"])
        assert isinstance(wrapped, RateLimitedLLM)
        assert wrapped._llm is bound  # noqa: SLF001

        order: list[str] = []

        class _RecordingBucket:
            def acquire(self):
                order.append("acquire")

        concurrency.LLM_BUCKET = _RecordingBucket()  # type: ignore[assignment]
        bound.invoke.side_effect = lambda *_a, **_kw: (order.append("invoke") or "bound-out")
        wrapped.invoke("hello")
        assert order == ["acquire", "invoke"]

    def test_with_config_returns_wrapped_and_invoke_throttled(self):
        """``llm.with_config({"tags": [...]})`` must keep the throttle."""
        inner = MagicMock()
        configured = MagicMock()
        configured.invoke.return_value = "cfg-out"
        inner.with_config.return_value = configured
        proxy = RateLimitedLLM(inner)
        wrapped = proxy.with_config({"tags": ["agent_x"]})
        assert isinstance(wrapped, RateLimitedLLM)
        assert wrapped._llm is configured  # noqa: SLF001

        order: list[str] = []

        class _RecordingBucket:
            def acquire(self):
                order.append("acquire")

        concurrency.LLM_BUCKET = _RecordingBucket()  # type: ignore[assignment]
        configured.invoke.side_effect = lambda *_a, **_kw: (
            order.append("invoke") or "cfg-out"
        )
        wrapped.invoke("hello")
        assert order == ["acquire", "invoke"]

    def test_with_retry_returns_throttled_wrapper(self):
        """``llm.with_retry(...).invoke(x)`` must hit the bucket on each
        outer invocation.

        Caveat: internal retry attempts inside ``RunnableRetry.invoke`` still
        call the bare ``_llm.invoke`` (langchain's internals), so a single
        outer call that retries N times consumes one token, not N. This
        matches the cost model of ``with_structured_output`` (one acquire
        per outer chain invoke, regardless of how many inner network calls
        the chain makes). The fix here closes the obvious gap — pre-fix,
        the outer wrapper itself was missing, so zero tokens were
        consumed."""
        inner = MagicMock()
        retried = MagicMock()
        retried.invoke.return_value = "retry-success"
        inner.with_retry.return_value = retried
        proxy = RateLimitedLLM(inner)
        wrapped = proxy.with_retry(stop_after_attempt=3)
        assert isinstance(wrapped, _ThrottledRunnable)
        assert wrapped._runnable is retried  # noqa: SLF001

        order: list[str] = []

        class _RecordingBucket:
            def acquire(self):
                order.append("acquire")

        concurrency.LLM_BUCKET = _RecordingBucket()  # type: ignore[assignment]
        retried.invoke.side_effect = lambda *_a, **_kw: (
            order.append("invoke") or "retry-success"
        )
        wrapped.invoke("hi")
        assert order == ["acquire", "invoke"]

    def test_with_fallbacks_returns_throttled_wrapper(self):
        """``llm.with_fallbacks([fb1, fb2]).invoke(...)`` must hit the bucket."""
        inner = MagicMock()
        fb_chain = MagicMock()
        fb_chain.invoke.return_value = "fb-out"
        inner.with_fallbacks.return_value = fb_chain
        proxy = RateLimitedLLM(inner)
        wrapped = proxy.with_fallbacks([MagicMock(), MagicMock()])
        assert isinstance(wrapped, _ThrottledRunnable)
        assert wrapped._runnable is fb_chain  # noqa: SLF001

        order: list[str] = []

        class _RecordingBucket:
            def acquire(self):
                order.append("acquire")

        concurrency.LLM_BUCKET = _RecordingBucket()  # type: ignore[assignment]
        fb_chain.invoke.side_effect = lambda *_a, **_kw: (order.append("invoke") or "fb-out")
        wrapped.invoke("hi")
        assert order == ["acquire", "invoke"]
