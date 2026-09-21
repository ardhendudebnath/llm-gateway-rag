"""LLM router: walks a route's fallback chain with retries, timeouts and circuit breaking."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from app.gateway.circuit_breaker import BreakerState, CircuitBreaker
from app.gateway.faults import FaultInjector
from app.gateway.pricing import compute_cost
from app.gateway.providers import Provider, ProviderError
from app.gateway.routing_config import Deployment, RoutingConfig
from app.gateway.schemas import Attempt, ChatRequest, ProviderResponse
from app.observability import metrics

log = logging.getLogger(__name__)

RETRY_BASE_DELAY_SECONDS = 0.2


class UnknownModelError(Exception):
    pass


class AllProvidersFailedError(Exception):
    def __init__(self, route: str, attempts: list[Attempt]):
        super().__init__(f"all deployments for route '{route}' failed")
        self.route = route
        self.attempts = attempts


class ClientRequestError(Exception):
    """The provider rejected the request itself (HTTP 400) — not worth falling back."""

    def __init__(self, message: str, attempts: list[Attempt]):
        super().__init__(message)
        self.attempts = attempts


@dataclass
class RoutedResult:
    response: ProviderResponse
    deployment: Deployment
    attempts: list[Attempt]
    cost_usd: float


class LLMRouter:
    def __init__(
        self,
        config: RoutingConfig,
        providers: Mapping[str, Provider],
        *,
        default_timeout: float,
        breaker_factory: Callable[[], CircuitBreaker],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        faults: FaultInjector | None = None,
    ):
        unknown = {d.provider for d in config.deployments.values()} - set(providers)
        if unknown:
            raise ValueError(f"routes reference unregistered providers: {sorted(unknown)}")
        self.config = config
        self._providers = providers
        self._default_timeout = default_timeout
        self._sleep = sleep
        self.faults = faults
        self.breakers = {name: breaker_factory() for name in config.deployments}

    @property
    def routes(self) -> list[str]:
        return list(self.config.routes)

    def breaker_states(self) -> dict[str, dict]:
        return {
            name: {"state": b.state.value, "consecutive_failures": b.consecutive_failures}
            for name, b in self.breakers.items()
        }

    async def complete(self, request: ChatRequest) -> RoutedResult:
        chain = self.config.routes.get(request.model)
        if chain is None:
            raise UnknownModelError(request.model)

        attempts: list[Attempt] = []
        for position, dep in enumerate(chain):
            breaker = self.breakers[dep.name]
            if not breaker.allow_request():
                attempts.append(Attempt(deployment=dep.name, outcome="skipped_circuit_open"))
                metrics.LLM_CALLS.labels(dep.name, "skipped_circuit_open").inc()
                continue

            response = await self._call_with_retries(dep, request, breaker, attempts)
            if response is None:
                continue  # exhausted this deployment's retries; fall back to the next

            cost = compute_cost(dep, response.usage)
            metrics.LLM_TOKENS.labels(dep.name, "prompt").inc(response.usage.prompt_tokens)
            metrics.LLM_TOKENS.labels(dep.name, "completion").inc(response.usage.completion_tokens)
            metrics.LLM_COST.labels(dep.name).inc(cost)
            if position > 0:
                metrics.LLM_FALLBACKS.labels(request.model).inc()
                log.info(
                    "served by fallback deployment",
                    extra={"route": request.model, "deployment": dep.name, "position": position},
                )
            return RoutedResult(response=response, deployment=dep, attempts=attempts, cost_usd=cost)

        metrics.LLM_ALL_FAILED.labels(request.model).inc()
        log.error(
            "all deployments failed",
            extra={"route": request.model, "attempts": [a.model_dump() for a in attempts]},
        )
        raise AllProvidersFailedError(request.model, attempts)

    async def _call_with_retries(
        self,
        dep: Deployment,
        request: ChatRequest,
        breaker: CircuitBreaker,
        attempts: list[Attempt],
    ) -> ProviderResponse | None:
        provider = self._providers[dep.provider]
        timeout = dep.timeout_seconds or self._default_timeout

        for retry in range(dep.max_retries + 1):
            if retry > 0:
                await self._sleep(RETRY_BASE_DELAY_SECONDS * 2 ** (retry - 1))
            start = time.perf_counter()
            try:
                if self.faults is not None:  # chaos testing: fail exactly where a provider would
                    await self.faults.maybe_fail(dep.name)
                response = await asyncio.wait_for(provider.complete(dep, request), timeout)
            except TimeoutError:
                outcome, error = "timeout", f"no response within {timeout}s"
            except ProviderError as e:
                if not e.retryable:
                    # The provider is healthy; the request is bad. Don't penalise the breaker.
                    breaker.record_success()
                    attempts.append(
                        Attempt(
                            deployment=dep.name,
                            outcome="client_error",
                            latency_ms=_ms_since(start),
                            error=str(e),
                        )
                    )
                    metrics.LLM_CALLS.labels(dep.name, "client_error").inc()
                    raise ClientRequestError(str(e), attempts) from e
                outcome, error = "error", str(e)
            except Exception as e:  # an adapter bug must not take the gateway down
                log.exception("unexpected provider exception", extra={"deployment": dep.name})
                outcome, error = "error", f"{type(e).__name__}: {e}"
            else:
                latency = time.perf_counter() - start
                breaker.record_success()
                metrics.BREAKER_OPEN.labels(dep.name).set(0)
                metrics.LLM_CALLS.labels(dep.name, "success").inc()
                metrics.LLM_LATENCY.labels(dep.name).observe(latency)
                attempts.append(
                    Attempt(deployment=dep.name, outcome="success", latency_ms=latency * 1000)
                )
                return response

            attempts.append(
                Attempt(
                    deployment=dep.name, outcome=outcome, latency_ms=_ms_since(start), error=error
                )
            )
            metrics.LLM_CALLS.labels(dep.name, outcome).inc()
            log.warning(
                "provider call failed",
                extra={"deployment": dep.name, "outcome": outcome, "error": error, "retry": retry},
            )
            if breaker.record_failure():
                metrics.BREAKER_OPEN.labels(dep.name).set(1)
                log.error(
                    "circuit opened",
                    extra={"deployment": dep.name, "cooldown_s": breaker.cooldown_seconds},
                )
            if breaker.state is not BreakerState.CLOSED:
                return None  # breaker tripped mid-retry loop: stop hammering this deployment
        return None


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000
