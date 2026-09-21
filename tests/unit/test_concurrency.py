"""The inference gate: bounded concurrency, bounded waiting, and shedding past that."""

import asyncio
import threading
import time

import pytest

from app.core.concurrency import InferenceGate, OverloadedError
from app.observability import metrics


class Tracker:
    """A blocking 'inference' that records how many copies run at the same time."""

    def __init__(self, seconds: float = 0.05):
        self.seconds = seconds
        self.running = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        time.sleep(self.seconds)
        with self._lock:
            self.running -= 1
        return "ok"


async def test_never_runs_more_than_the_limit_at_once():
    gate = InferenceGate("test", max_concurrency=2, max_queue=100)
    work = Tracker()
    results = await asyncio.gather(*(gate.run(work) for _ in range(10)))
    assert results == ["ok"] * 10
    assert work.peak == 2  # 10 callers, never more than 2 inferences in flight
    assert gate.waiting == 0


async def test_sheds_once_the_queue_is_full():
    gate = InferenceGate("shed-test", max_concurrency=1, max_queue=2)
    work = Tracker(seconds=0.2)
    before = metrics.INFERENCE_SHED.labels("shed-test")._value.get()

    # 1 runs, 2 wait; the 4th and 5th are refused at once instead of queueing into a timeout.
    outcomes = await asyncio.gather(*(gate.run(work) for _ in range(5)), return_exceptions=True)

    shed = [o for o in outcomes if isinstance(o, OverloadedError)]
    assert outcomes.count("ok") == 3 and len(shed) == 2
    assert "saturated" in str(shed[0])
    assert metrics.INFERENCE_SHED.labels("shed-test")._value.get() == before + 2


async def test_a_zero_queue_sheds_whenever_every_slot_is_busy():
    gate = InferenceGate("zero-queue", max_concurrency=1, max_queue=0)
    busy = asyncio.create_task(gate.run(Tracker(seconds=0.1)))
    await asyncio.sleep(0.02)  # let it take the only slot
    with pytest.raises(OverloadedError):
        await gate.run(Tracker())
    assert await busy == "ok"


async def test_a_wait_budget_sheds_instead_of_waiting():
    gate = InferenceGate("budget", max_concurrency=1, max_queue=10)
    busy = asyncio.create_task(gate.run(Tracker(seconds=0.3)))
    await asyncio.sleep(0.02)

    started = time.perf_counter()
    with pytest.raises(OverloadedError, match="no free slot within 50 ms"):
        await gate.run(Tracker(), max_wait=0.05)
    assert time.perf_counter() - started < 0.2  # gave up at the budget, not after the 0.3 s job
    assert await busy == "ok"


async def test_a_timed_out_wait_does_not_leak_a_slot():
    gate = InferenceGate("no-leak", max_concurrency=1, max_queue=10)
    busy = asyncio.create_task(gate.run(Tracker(seconds=0.1)))
    await asyncio.sleep(0.02)
    for _ in range(5):
        with pytest.raises(OverloadedError):
            await gate.run(Tracker(), max_wait=0.01)
    await busy
    work = Tracker()
    await asyncio.gather(*(gate.run(work) for _ in range(4)))
    assert work.peak == 1 and gate.waiting == 0  # still exactly one slot, nobody stuck waiting


async def test_a_wait_within_budget_runs_normally():
    gate = InferenceGate("fits", max_concurrency=1, max_queue=10)
    busy = asyncio.create_task(gate.run(Tracker(seconds=0.05)))
    await asyncio.sleep(0.01)
    assert await gate.run(Tracker(), max_wait=1.0) == "ok"
    await busy


async def test_a_failing_inference_releases_its_slot():
    gate = InferenceGate("failing", max_concurrency=1, max_queue=5)

    def boom():
        raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError):
        await gate.run(boom)
    assert await gate.run(lambda: "still usable") == "still usable"


@pytest.mark.parametrize(("concurrency", "queue"), [(0, 1), (1, -1)])
def test_invalid_limits_are_rejected(concurrency, queue):
    with pytest.raises(ValueError):
        InferenceGate("x", max_concurrency=concurrency, max_queue=queue)
