"""LLM router: walks a route's fallback chain with retries, timeouts and circuit breaking."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from app.gateway.breaker_cluster import ClusterBreakers
from app.gateway.circuit_breaker import BreakerState, CircuitBreaker
from app.gateway.faults import FaultInjector
from app.gateway.pricing import compute_cost
from app.gateway.providers import Provider, ProviderError
from app.gateway.routing_config import Deployment, RoutingConfig
from app.gateway.schemas import Attempt, ChatRequest, ProviderResponse, StreamDelta, Usage
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


class StreamInterruptedError(Exception):
    """A stream failed after its first token. Falling back now would mean sending the client a
    second answer on top of a partial one, so the request ends here."""

    def __init__(self, message: str, deployment: str, delivered: str):
        super().__init__(message)
        self.deployment = deployment
        self.delivered = delivered  # what the client already has


@dataclass
class RoutedResult:
    response: ProviderResponse
    deployment: Deployment
    attempts: list[Attempt]
    cost_usd: float


@dataclass
class OpenStream:
    """A stream that has already produced its first token.

    `open_stream` returns only once some deployment has answered, so every fallback decision is
    made before the caller can send anything. After that the choice is locked in: an error out of
    `rest` is a `StreamInterruptedError`, never another attempt.

    The last four fields are filled in when `rest` is exhausted, because providers report usage
    only at the end of a stream.
    """

    deployment: Deployment
    model: str
    attempts: list[Attempt]
    first: StreamDelta
    rest: AsyncIterator[StreamDelta]
    ttft_ms: float
    prompt_estimate: int = 0  # used while, or if, the provider reports no usage
    synthesized: bool = False
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    cost_usd: float = 0.0
    usage_estimated: bool = False
    reported_usage: bool = False  # the provider sent real token counts
    settled: bool = False  # priced and counted; done once, however the stream ended


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
        cluster: ClusterBreakers | None = None,
    ):
        unknown = {d.provider for d in config.deployments.values()} - set(providers)
        if unknown:
            raise ValueError(f"routes reference unregistered providers: {sorted(unknown)}")
        self.config = config
        self._providers = providers
        self._default_timeout = default_timeout
        self._sleep = sleep
        self.faults = faults
        self.cluster = cluster
        self._breaker_factory = breaker_factory
        self.breakers = {name: breaker_factory() for name in config.deployments}

    async def _prepare(self, chain: list[Deployment]) -> None:
        """Before walking a chain, learn what other replicas already know about these deployments.

        Syncing here rather than at start-up means a deployment a canary route table introduced is
        included too, and it costs one pipelined Redis read per refresh window, not per request.
        """
        if self.cluster is None:
            return
        for dep in chain:
            self._breaker(dep.name)  # make sure lazily-created breakers take part
        await self.cluster.sync(self.breakers, [dep.name for dep in chain])

    async def sync_breakers(self) -> None:
        """Bring every known breaker up to date, for readers rather than for a request.

        Without this a pod that has not served a request lately reports its own stale view: the
        admin page showed a closed circuit next to shared state saying it was open.
        """
        if self.cluster is not None:
            await self.cluster.sync(self.breakers, list(self.breakers))

    async def _failed(self, name: str, breaker: CircuitBreaker) -> bool:
        """Record a failure, and report whether the circuit is now open — here or cluster-wide."""
        if self.cluster is not None:
            return await self.cluster.record_failure(name, breaker)
        return breaker.record_failure()

    async def _succeeded(self, name: str, breaker: CircuitBreaker) -> None:
        if self.cluster is not None:
            await self.cluster.record_success(name, breaker)
        else:
            breaker.record_success()

    def _breaker(self, deployment: str) -> CircuitBreaker:
        """A canary route table can name deployments the start-up table never had; they get a
        breaker on first use, and keep it if the canary is promoted."""
        if deployment not in self.breakers:
            self.breakers[deployment] = self._breaker_factory()
        return self.breakers[deployment]

    @property
    def routes(self) -> list[str]:
        return list(self.config.routes)

    def breaker_states(self) -> dict[str, dict]:
        return {
            name: {"state": b.state.value, "consecutive_failures": b.consecutive_failures}
            for name, b in self.breakers.items()
        }

    async def complete(
        self, request: ChatRequest, config: RoutingConfig | None = None
    ) -> RoutedResult:
        """`config` overrides the start-up route table for this request: that is how a canary
        rollout (app/gateway/rollout.py) serves a slice of traffic from a different table."""
        chain = (config or self.config).routes.get(request.model)
        if chain is None:
            raise UnknownModelError(request.model)
        await self._prepare(chain)

        attempts: list[Attempt] = []
        for position, dep in enumerate(chain):
            breaker = self._breaker(dep.name)
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

    async def open_stream(
        self, request: ChatRequest, config: RoutingConfig | None = None
    ) -> OpenStream:
        """Walk the same fallback chain, but stop as soon as a deployment produces a first token.

        Everything a non-streamed call does — retries, timeouts, circuit breaking, chaos faults —
        happens here, while nothing has reached the client yet. That is the whole design: the
        timeout that matters for a stream is time-to-first-token, not time-to-last, and the last
        moment a fallback is honest is just before the first token goes out.
        """
        chain = (config or self.config).routes.get(request.model)
        if chain is None:
            raise UnknownModelError(request.model)
        await self._prepare(chain)

        attempts: list[Attempt] = []
        for position, dep in enumerate(chain):
            breaker = self._breaker(dep.name)
            if not breaker.allow_request():
                attempts.append(Attempt(deployment=dep.name, outcome="skipped_circuit_open"))
                metrics.LLM_CALLS.labels(dep.name, "skipped_circuit_open").inc()
                continue

            opened = await self._open_with_retries(dep, request, breaker, attempts)
            if opened is None:
                continue  # never produced a token; the next deployment still can
            if position > 0:
                metrics.LLM_FALLBACKS.labels(request.model).inc()
                log.info(
                    "stream served by fallback deployment",
                    extra={"route": request.model, "deployment": dep.name, "position": position},
                )
            metrics.STREAM_TTFT.labels(dep.name).observe(opened.ttft_ms / 1000)
            return opened

        metrics.LLM_ALL_FAILED.labels(request.model).inc()
        log.error(
            "all deployments failed to start a stream",
            extra={"route": request.model, "attempts": [a.model_dump() for a in attempts]},
        )
        raise AllProvidersFailedError(request.model, attempts)

    async def _open_with_retries(
        self,
        dep: Deployment,
        request: ChatRequest,
        breaker: CircuitBreaker,
        attempts: list[Attempt],
    ) -> OpenStream | None:
        provider = self._providers.get(dep.provider)
        if provider is None:
            raise UnknownModelError(f"no provider adapter '{dep.provider}' for {dep.name}")
        timeout = dep.timeout_seconds or self._default_timeout

        for retry in range(dep.max_retries + 1):
            if retry > 0:
                await self._sleep(RETRY_BASE_DELAY_SECONDS * 2 ** (retry - 1))
            start = time.perf_counter()
            iterator: AsyncIterator[StreamDelta] | None = None
            try:
                if self.faults is not None:
                    await self.faults.maybe_fail(dep.name)
                if not hasattr(provider, "stream"):
                    # An adapter that cannot stream still answers: its completion is chunked here
                    # rather than refused, and the client is told the stream was synthesized.
                    response = await asyncio.wait_for(provider.complete(dep, request), timeout)
                    first, iterator = _chunk_completion(response)
                    synthesized = True
                else:
                    iterator = provider.stream(dep, request)
                    first = await asyncio.wait_for(anext(iterator), timeout)
                    synthesized = False
            except (TimeoutError, StopAsyncIteration) as e:
                outcome = "timeout" if isinstance(e, TimeoutError) else "error"
                error = (
                    f"no first token within {timeout}s"
                    if isinstance(e, TimeoutError)
                    else "stream ended before any token"
                )
                await _aclose(iterator)
            except ProviderError as e:
                await _aclose(iterator)
                if not e.retryable:
                    # The provider is fine; the request is not. Clear it for every replica.
                    await self._succeeded(dep.name, breaker)
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
            except Exception as e:
                log.exception("unexpected provider exception", extra={"deployment": dep.name})
                await _aclose(iterator)
                outcome, error = "error", f"{type(e).__name__}: {e}"
            else:
                ttft = _ms_since(start)
                # Note what answered, but do not tell the breaker this succeeded yet. A deployment
                # that always produces one token and then dies is broken, and clearing its failure
                # count here would hide exactly that: the breaker is settled when the stream ends.
                metrics.LLM_CALLS.labels(dep.name, "success").inc()
                metrics.LLM_LATENCY.labels(dep.name).observe(ttft / 1000)
                attempts.append(Attempt(deployment=dep.name, outcome="success", latency_ms=ttft))
                opened = OpenStream(
                    deployment=dep,
                    model=first.model or dep.model,
                    attempts=attempts,
                    first=first,
                    rest=iterator,
                    ttft_ms=ttft,
                    prompt_estimate=sum(len(m.content.split()) for m in request.messages),
                    synthesized=synthesized,
                )
                # The relay needs the OpenStream it is filling in, so it replaces `rest` here.
                opened.rest = self._relay(opened, iterator, breaker)
                return opened

            attempts.append(
                Attempt(
                    deployment=dep.name, outcome=outcome, latency_ms=_ms_since(start), error=error
                )
            )
            metrics.LLM_CALLS.labels(dep.name, outcome).inc()
            log.warning(
                "could not start a stream",
                extra={"deployment": dep.name, "outcome": outcome, "error": error, "retry": retry},
            )
            if await self._failed(dep.name, breaker):
                metrics.BREAKER_OPEN.labels(dep.name).set(1)
                log.error(
                    "circuit opened",
                    extra={"deployment": dep.name, "cooldown_s": breaker.cooldown_seconds},
                )
            if breaker.state is not BreakerState.CLOSED:
                return None
        return None

    async def _relay(
        self, opened: OpenStream, iterator: AsyncIterator[StreamDelta], breaker: CircuitBreaker
    ) -> AsyncIterator[StreamDelta]:
        """Pass the rest of the stream through, then settle up: usage, cost and metrics.

        A stalled provider is as bad as a dead one, so each chunk gets the same timeout the first
        token did. Whatever ends the stream, the provider iterator is closed — a client that hangs
        up must not leave a provider connection open.
        """
        dep = opened.deployment
        timeout = dep.timeout_seconds or self._default_timeout
        delivered = [opened.first.content]
        _absorb(opened, opened.first)
        # Keep a running estimate as text goes out, so the cost of a stream is known at every
        # moment and not only at the end. A client that hangs up mid-answer is billed from this.
        words = opened.first.content.count(" ") + bool(opened.first.content)
        self._estimate(opened, words)
        try:
            while True:
                try:
                    delta = await asyncio.wait_for(anext(iterator), timeout)
                except StopAsyncIteration:
                    break
                except Exception as e:
                    error = (
                        f"stalled for more than {timeout}s"
                        if isinstance(e, TimeoutError)
                        else f"{type(e).__name__}: {e}"
                    )
                    text = "".join(delivered)
                    # Those tokens were generated and will be billed by the provider, so they are
                    # priced here too — from the partial text, since no usage report is coming.
                    self._settle(opened, text)
                    await self._record_interruption(opened, breaker, error)
                    raise StreamInterruptedError(error, dep.name, text) from e
                _absorb(opened, delta)
                if delta.content:
                    delivered.append(delta.content)
                    words += delta.content.count(" ")
                    self._estimate(opened, words)
                    yield delta
            # It answered in full: only now is the deployment known to be healthy.
            await self._succeeded(dep.name, breaker)
            metrics.BREAKER_OPEN.labels(dep.name).set(0)
            self._settle(opened, "".join(delivered))
            metrics.STREAMS.labels("completed").inc()
        finally:
            # A client that walked away leaves the stream unsettled: price it on the way out, so
            # the tokens it caused are counted once, here, and not lost.
            self._settle(opened, "".join(delivered))
            await _aclose(iterator)

    async def _record_interruption(
        self, opened: OpenStream, breaker: CircuitBreaker, error: str
    ) -> None:
        dep = opened.deployment
        opened.attempts.append(
            Attempt(deployment=dep.name, outcome="stream_interrupted", error=error)
        )
        metrics.LLM_CALLS.labels(dep.name, "stream_interrupted").inc()
        metrics.STREAMS.labels("interrupted").inc()
        log.error(
            "stream interrupted after its first token; no fallback is possible",
            extra={"deployment": dep.name, "error": error},
        )
        if await self._failed(dep.name, breaker):  # it did fail, even though it answered first
            metrics.BREAKER_OPEN.labels(dep.name).set(1)

    def _estimate(self, opened: OpenStream, words: int) -> None:
        """Price what has gone out so far, until the provider says what it actually charged."""
        if opened.reported_usage:
            return
        opened.usage = Usage(prompt_tokens=opened.prompt_estimate, completion_tokens=words)
        opened.usage_estimated = True
        opened.cost_usd = compute_cost(opened.deployment, opened.usage)

    def _settle(self, opened: OpenStream, text: str) -> None:
        """Price the stream and count its tokens, once, however it ended.

        A provider that reported usage is billed on its own numbers; one that did not is billed on
        the text it produced, and the client is told so rather than being handed a silent guess.
        """
        if opened.settled:
            return
        opened.settled = True
        if opened.reported_usage:
            opened.usage_estimated = False
        else:
            opened.usage = Usage(
                prompt_tokens=opened.prompt_estimate, completion_tokens=len(text.split())
            )
            opened.usage_estimated = True
        opened.cost_usd = compute_cost(opened.deployment, opened.usage)
        name = opened.deployment.name
        metrics.LLM_TOKENS.labels(name, "prompt").inc(opened.usage.prompt_tokens)
        metrics.LLM_TOKENS.labels(name, "completion").inc(opened.usage.completion_tokens)
        metrics.LLM_COST.labels(name).inc(opened.cost_usd)

    async def _call_with_retries(
        self,
        dep: Deployment,
        request: ChatRequest,
        breaker: CircuitBreaker,
        attempts: list[Attempt],
    ) -> ProviderResponse | None:
        provider = self._providers.get(dep.provider)
        if provider is None:  # only reachable if a canary table names an unregistered adapter
            raise UnknownModelError(f"no provider adapter '{dep.provider}' for {dep.name}")
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
                    await self._succeeded(dep.name, breaker)
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
                await self._succeeded(dep.name, breaker)
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
            if await self._failed(dep.name, breaker):
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


def _absorb(opened: OpenStream, delta: StreamDelta) -> None:
    """Carry across what a delta reports about the stream as a whole."""
    if delta.usage is not None:
        opened.usage = delta.usage
        opened.reported_usage = True
    if delta.finish_reason is not None:
        opened.finish_reason = delta.finish_reason
    if delta.model:
        opened.model = delta.model


def _chunk_completion(
    response: ProviderResponse, size: int = 24
) -> tuple[StreamDelta, AsyncIterator[StreamDelta]]:
    """Turn a whole answer into a stream, for adapters that cannot produce one."""
    pieces = [response.content[i : i + size] for i in range(0, len(response.content), size)] or [""]

    async def rest() -> AsyncIterator[StreamDelta]:
        for piece in pieces[1:]:
            yield StreamDelta(content=piece, model=response.model)
        yield StreamDelta(
            finish_reason=response.finish_reason, usage=response.usage, model=response.model
        )

    return StreamDelta(content=pieces[0], model=response.model), rest()


async def _aclose(iterator: AsyncIterator | None) -> None:
    """Release the provider's connection, whatever ended the stream."""
    if iterator is not None and hasattr(iterator, "aclose"):
        try:
            await iterator.aclose()
        except Exception:  # a broken stream failing to close is not worth another failure
            log.debug("closing a provider stream failed", exc_info=True)
