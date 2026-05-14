"""Process-global throttles for parallel scene execution.

When ``--scene-parallelism > 1`` the MVP 2.0 graph fans out scenes via
``langgraph.types.Send``; LangGraph runs the branches on its internal thread
pool. Two resources need explicit bounding:

1. **Manim rendering** — CPU/GPU heavyweight. Bounded by
   :data:`RENDER_SEMAPHORE`. Used by ``sandbox.render.render``.
2. **LLM API calls** — provider-side rate limited. Bounded by
   :data:`LLM_BUCKET`. Acquired by the ``RateLimitedLLM`` proxy in ``llm.py``.

Both globals default to ``None`` (no throttling), so single-threaded serial
runs pay zero overhead. The CLI calls :func:`configure` exactly once before
invoking the graph; tests use :func:`reset` to wipe state between cases.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator

log = logging.getLogger(__name__)


# Module-level state. Read by render() + the LLM proxy; written by configure().
RENDER_SEMAPHORE: threading.BoundedSemaphore | None = None
LLM_BUCKET: TokenBucket | None = None


class TokenBucket:
    """Thread-safe token bucket with lazy refill.

    Tokens accumulate continuously at ``rate`` tokens/sec, capped at ``burst``
    (default ``max(1.0, rate)``). :meth:`acquire` blocks until at least one
    token is available, then consumes exactly one. The bucket starts full so
    the first ``burst`` calls don't wait — matches the intuition that an idle
    process should be able to fire a small batch immediately.
    """

    __slots__ = ("_rate", "_burst", "_tokens", "_last", "_lock")

    def __init__(self, rate: float, burst: int | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"TokenBucket rate must be positive; got {rate}")
        self._rate = float(rate)
        self._burst = float(burst) if burst is not None else max(1.0, float(rate))
        if self._burst <= 0:
            raise ValueError(f"TokenBucket burst must be positive; got {burst}")
        self._tokens = self._burst
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Compute wait time *inside* the lock so the snapshot is consistent;
                # do the actual sleep outside so other threads can refill too.
                sleep_for = (1.0 - self._tokens) / self._rate
            time.sleep(sleep_for)


@contextlib.contextmanager
def render_slot() -> Iterator[None]:
    """Acquire a render slot for the duration of the ``with`` block.

    Passthrough when no semaphore is configured. Used by ``sandbox.render``.
    """
    sem = RENDER_SEMAPHORE
    if sem is None:
        yield
        return
    with sem:
        yield


def llm_acquire() -> None:
    """Acquire one LLM call token (no-op when no bucket configured)."""
    bucket = LLM_BUCKET
    if bucket is not None:
        bucket.acquire()


def configure(
    *,
    scene_parallelism: int = 1,
    render_concurrency: int | None = None,
    llm_rps: float | None = None,
    llm_burst: int | None = None,
) -> None:
    """One-shot configuration of the process-global throttles.

    Called from the CLI before ``graph.invoke()``. Re-calling resets the
    semaphore and bucket — tests depend on this for isolation.

    Args:
        scene_parallelism: Logged for diagnostics. The actual fan-out
            concurrency is bounded by LangGraph's step-level executor; this
            value is informational here.
        render_concurrency: Max concurrent Manim subprocesses; ``None`` = off.
        llm_rps: Max LLM calls/sec across all threads; ``None`` = off.
        llm_burst: Bucket burst capacity; defaults to ``max(1, llm_rps)``.
    """
    global RENDER_SEMAPHORE, LLM_BUCKET
    if render_concurrency is not None and render_concurrency > 0:
        RENDER_SEMAPHORE = threading.BoundedSemaphore(int(render_concurrency))
    else:
        RENDER_SEMAPHORE = None
    if llm_rps is not None and llm_rps > 0:
        LLM_BUCKET = TokenBucket(float(llm_rps), burst=llm_burst)
    else:
        LLM_BUCKET = None
    log.info(
        "[concurrency] scene_parallelism=%d render_concurrency=%s llm_rps=%s",
        scene_parallelism,
        render_concurrency,
        llm_rps,
    )


def reset() -> None:
    """Wipe throttle state (test helper)."""
    global RENDER_SEMAPHORE, LLM_BUCKET
    RENDER_SEMAPHORE = None
    LLM_BUCKET = None


__all__ = [
    "LLM_BUCKET",
    "RENDER_SEMAPHORE",
    "TokenBucket",
    "configure",
    "llm_acquire",
    "render_slot",
    "reset",
]
