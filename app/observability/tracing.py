"""LLM tracing.

Every chat completion becomes one Langfuse *generation*: the prompt, the response, the model and
deployment that answered, token counts, cost, latency, and whether it was served from the semantic
cache. Traces carry the tenant as ``user_id`` and the request id as ``session_id``, so a line in the
JSON logs and a trace in Langfuse can be matched up.

Two rules shape this module:

* **Optional.** Without Langfuse credentials the gateway uses ``NoopTracer`` and behaves exactly as
  before. Nothing else in the code knows the difference.
* **Never fatal.** Every call into the SDK is guarded: if Langfuse is misconfigured, slow to start
  or down, the request still gets answered. Observability must not be able to take the service
  down.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.logging import request_id_var

log = logging.getLogger(__name__)


@dataclass
class ChatTrace:
    """Filled in by the chat service as the request progresses; read by the tracer at the end."""

    route: str
    output: str | None = None
    model: str | None = None
    deployment: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    cache_similarity: float | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "deployment": self.deployment,
            "cached": self.cached,
            "cache_similarity": self.cache_similarity,
            "attempts": self.attempts,
            "request_id": request_id_var.get(),
        }


class Tracer(Protocol):
    enabled: bool

    def chat(
        self,
        *,
        tenant_id: str,
        key_id: str,
        route: str,
        messages: Sequence[dict[str, Any]],
        model_parameters: dict[str, Any] | None = None,
    ) -> Iterator[ChatTrace]:
        """Context manager around one completion, yielding the trace to fill in."""
        ...

    def span(self, name: str, **metadata: Any) -> Iterator[None]:
        """Context manager for a non-LLM step (retrieval, reranking) that nests child spans."""
        ...

    def shutdown(self) -> None: ...


class NoopTracer:
    enabled = False

    @contextmanager
    def chat(self, *, tenant_id, key_id, route, messages, model_parameters=None):
        yield ChatTrace(route=route)

    @contextmanager
    def span(self, name: str, **metadata: Any):
        yield

    def shutdown(self) -> None:
        return None


class LangfuseTracer:
    """Langfuse v3/v4 SDK. ``propagate_attributes`` is injected so tests don't need the real one."""

    enabled = True

    def __init__(self, client: Any, propagate: Any):
        self._client = client
        self._propagate = propagate

    @contextmanager
    def chat(self, *, tenant_id, key_id, route, messages, model_parameters=None):
        trace = ChatTrace(route=route)
        with ExitStack() as stack:
            generation = None
            try:
                stack.enter_context(
                    self._propagate(
                        user_id=tenant_id,
                        session_id=request_id_var.get(),
                        tags=[f"route:{route}"],
                        metadata={"key_id": key_id},
                    )
                )
                generation = stack.enter_context(
                    self._client.start_as_current_observation(
                        name="chat.completion",
                        as_type="generation",
                        input=list(messages),
                        model_parameters=model_parameters or {},
                    )
                )
            except Exception:
                log.exception("could not start a trace; serving the request untraced")
                yield trace
                return
            try:
                yield trace
            except Exception as e:
                trace.error = f"{type(e).__name__}: {e}"
                raise
            finally:
                self._record(generation, trace)

    def _record(self, generation: Any, trace: ChatTrace) -> None:
        try:
            generation.update(
                output=trace.output,
                model=trace.model,
                usage_details={
                    "input": trace.prompt_tokens,
                    "output": trace.completion_tokens,
                    "total": trace.prompt_tokens + trace.completion_tokens,
                },
                cost_details={"total": trace.cost_usd},
                metadata=trace.metadata(),
                status_message=trace.error,
            )
        except Exception:
            log.exception("could not record a trace")

    @contextmanager
    def span(self, name: str, **metadata: Any):
        # Enter inside the try: a @contextmanager function only raises on __enter__, so calling it
        # outside would let a Langfuse failure escape into the request.
        with ExitStack() as stack:
            try:
                stack.enter_context(
                    self._client.start_as_current_observation(
                        name=name, as_type="span", metadata=metadata or None
                    )
                )
            except Exception:
                log.exception("could not start a span; continuing untraced")
            yield

    def shutdown(self) -> None:
        try:
            self._client.shutdown()  # flushes anything still buffered
        except Exception:
            log.exception("could not shut the tracer down cleanly")
