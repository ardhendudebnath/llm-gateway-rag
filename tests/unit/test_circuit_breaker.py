import pytest

from app.gateway.circuit_breaker import BreakerState, CircuitBreaker


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def breaker(clock):
    return CircuitBreaker(failure_threshold=3, cooldown_seconds=30, clock=clock)


def test_starts_closed_and_allows_traffic(breaker):
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allow_request()


def test_opens_after_threshold_consecutive_failures(breaker):
    assert breaker.record_failure() is False
    assert breaker.record_failure() is False
    assert breaker.record_failure() is True  # the transition is reported exactly once
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allow_request()


def test_success_resets_the_consecutive_count(breaker):
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED


def test_half_open_after_cooldown_allows_single_probe(breaker, clock):
    for _ in range(3):
        breaker.record_failure()
    clock.now += 29.9
    assert not breaker.allow_request()
    clock.now += 0.1
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow_request()  # the probe
    assert not breaker.allow_request()  # everyone else waits for the probe's verdict


def test_successful_probe_closes_circuit(breaker, clock):
    for _ in range(3):
        breaker.record_failure()
    clock.now += 30
    assert breaker.allow_request()
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.consecutive_failures == 0


def test_failed_probe_reopens_for_a_fresh_cooldown(breaker, clock):
    for _ in range(3):
        breaker.record_failure()
    clock.now += 30
    assert breaker.allow_request()
    assert breaker.record_failure() is True
    assert breaker.state is BreakerState.OPEN
    clock.now += 29
    assert not breaker.allow_request()
    clock.now += 1
    assert breaker.allow_request()


def test_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0, cooldown_seconds=1)
