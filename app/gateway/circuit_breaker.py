"""Per-deployment circuit breaker.

CLOSED    -> traffic flows; consecutive failures are counted.
OPEN      -> after ``failure_threshold`` consecutive failures; traffic is skipped for ``cooldown``.
HALF_OPEN -> cooldown elapsed; exactly one probe request is let through. Success closes the
             circuit, failure re-opens it for another cooldown.

State is in-process (one breaker set per API replica). That is deliberate: each replica learns a
provider is down within ``failure_threshold`` requests, which is cheaper than a Redis round-trip on
every call. Sharing state across replicas is a listed stretch goal.
"""

import time
from collections.abc import Callable
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self.cooldown_seconds
        ):
            self._state = BreakerState.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def allow_request(self) -> bool:
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._probe_in_flight = False

    def record_failure(self) -> bool:
        """Record a failure. Returns True if this call transitioned the breaker to OPEN."""
        self._consecutive_failures += 1
        if self.state is BreakerState.HALF_OPEN or (
            self._state is BreakerState.CLOSED
            and self._consecutive_failures >= self.failure_threshold
        ):
            self._state = BreakerState.OPEN
            self._opened_at = self._clock()
            self._probe_in_flight = False
            return True
        return False
