"""Bounded concurrency for blocking model inference, with backpressure.

The load test found the failure mode this exists for. Every request handed its embedding or
reranking to a thread, so under load dozens of ONNX inferences ran at once. Each allocates its own
working memory, and ONNX Runtime keeps what it has allocated, so the API pods went from ~350 MiB
to their 1.5 GiB limit, were OOM-killed and crash-looped: 74% errors at just 50 users. A bigger
limit would only have moved that cliff.

``InferenceGate`` caps how many inferences run at once per model (which bounds memory, and cuts
the CPU contention that made each one slower) and how many may wait. Past that, it sheds: callers
get ``OverloadedError`` at once instead of queueing into a timeout, and each caller decides how to
degrade (skip the cache, skip reranking) or returns 503 with Retry-After.
"""

import asyncio
from collections.abc import Callable
from typing import TypeVar

from app.observability import metrics

T = TypeVar("T")


class OverloadedError(Exception):
    """A bounded resource is saturated; the request should degrade or be shed."""


class InferenceGate:
    def __init__(self, name: str, *, max_concurrency: int, max_queue: int):
        if max_concurrency < 1 or max_queue < 0:
            raise ValueError("need max_concurrency >= 1 and max_queue >= 0")
        self.name = name
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self._slots = asyncio.Semaphore(max_concurrency)
        self._waiting = 0

    @property
    def waiting(self) -> int:
        return self._waiting

    async def run(self, fn: Callable[[], T], *, max_wait: float | None = None) -> T:
        """Run ``fn`` in a worker thread once a slot is free.

        Sheds (``OverloadedError``) if too many callers are already waiting, or, with ``max_wait``,
        if no slot frees up in time. A wait budget bounds tail latency for optional work: a
        rerank that can't start within its budget is better skipped than waited for.
        """
        if self._slots.locked() and self._waiting >= self.max_queue:
            metrics.INFERENCE_SHED.labels(self.name).inc()
            raise OverloadedError(f"{self.name} is saturated; try again shortly")
        self._waiting += 1
        metrics.INFERENCE_WAITING.labels(self.name).set(self._waiting)
        try:
            if max_wait is None:
                await self._slots.acquire()
            else:
                await asyncio.wait_for(self._slots.acquire(), max_wait)
        except TimeoutError:
            metrics.INFERENCE_SHED.labels(self.name).inc()
            raise OverloadedError(
                f"{self.name} had no free slot within {max_wait * 1000:.0f} ms"
            ) from None
        finally:
            self._waiting -= 1
            metrics.INFERENCE_WAITING.labels(self.name).set(self._waiting)
        try:
            return await asyncio.to_thread(fn)
        finally:
            self._slots.release()
