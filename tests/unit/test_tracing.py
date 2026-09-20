"""The Langfuse tracer: what it records, and that it can never break a request."""

from contextlib import contextmanager

import pytest

from app.core.config import Settings
from app.core.container import build_tracer
from app.observability.tracing import ChatTrace, LangfuseTracer, NoopTracer


class FakeGeneration:
    def __init__(self, fail_update: bool = False):
        self.updates: list[dict] = []
        self._fail_update = fail_update

    def update(self, **kwargs):
        if self._fail_update:
            raise RuntimeError("langfuse is down")
        self.updates.append(kwargs)


class FakeClient:
    """Mimics the bits of the Langfuse client the tracer touches."""

    def __init__(self, *, fail_start: bool = False, fail_update: bool = False):
        self.observations: list[dict] = []
        self.generation = FakeGeneration(fail_update)
        self.shutdowns = 0
        self._fail_start = fail_start

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        if self._fail_start:
            raise RuntimeError("no connection to langfuse")
        self.observations.append(kwargs)
        yield self.generation

    def shutdown(self):
        self.shutdowns += 1


@contextmanager
def fake_propagate(**kwargs):
    fake_propagate.calls.append(kwargs)
    yield


fake_propagate.calls = []


@pytest.fixture(autouse=True)
def _reset():
    fake_propagate.calls = []


def tracer(**kwargs) -> tuple[LangfuseTracer, FakeClient]:
    client = FakeClient(**kwargs)
    return LangfuseTracer(client, fake_propagate), client


def chat_kwargs(**overrides) -> dict:
    return {
        "tenant_id": "acme",
        "key_id": "k1",
        "route": "default",
        "messages": [{"role": "user", "content": "hi"}],
        "model_parameters": {"temperature": 0.2},
        **overrides,
    }


def test_a_completion_is_recorded_as_a_generation():
    t, client = tracer()
    with t.chat(**chat_kwargs()) as trace:
        trace.output = "hello"
        trace.model = "fake/primary"
        trace.deployment = "primary"
        trace.prompt_tokens, trace.completion_tokens = 10, 4
        trace.cost_usd = 0.002

    started = client.observations[0]
    assert started["name"] == "chat.completion" and started["as_type"] == "generation"
    assert started["input"] == [{"role": "user", "content": "hi"}]
    assert started["model_parameters"] == {"temperature": 0.2}

    update = client.generation.updates[0]
    assert update["output"] == "hello" and update["model"] == "fake/primary"
    assert update["usage_details"] == {"input": 10, "output": 4, "total": 14}
    assert update["cost_details"] == {"total": 0.002}
    assert update["metadata"]["deployment"] == "primary"
    assert update["status_message"] is None


def test_the_trace_is_attributed_to_the_tenant():
    t, _ = tracer()
    with t.chat(**chat_kwargs()):
        pass
    assert fake_propagate.calls[0]["user_id"] == "acme"
    assert fake_propagate.calls[0]["tags"] == ["route:default"]
    assert fake_propagate.calls[0]["metadata"] == {"key_id": "k1"}


def test_a_cache_hit_is_traced_as_a_zero_cost_generation():
    t, client = tracer()
    with t.chat(**chat_kwargs()) as trace:
        trace.output = "cached answer"
        trace.cached = True
        trace.cache_similarity = 0.97

    update = client.generation.updates[0]
    assert update["cost_details"] == {"total": 0.0}
    assert update["metadata"]["cached"] is True
    assert update["metadata"]["cache_similarity"] == 0.97


def test_an_error_is_recorded_and_re_raised():
    t, client = tracer()
    with pytest.raises(RuntimeError, match="providers failed"), t.chat(**chat_kwargs()):
        raise RuntimeError("all providers failed")
    assert "all providers failed" in client.generation.updates[0]["status_message"]


def test_a_tracer_that_cannot_start_still_serves_the_request():
    t, client = tracer(fail_start=True)
    with t.chat(**chat_kwargs()) as trace:
        trace.output = "served anyway"
    assert isinstance(trace, ChatTrace)
    assert client.generation.updates == []


def test_a_tracer_that_cannot_record_still_serves_the_request():
    t, _ = tracer(fail_update=True)
    with t.chat(**chat_kwargs()) as trace:  # the failing update must not escape
        trace.output = "served anyway"


def test_spans_nest_and_survive_failures():
    t, client = tracer()
    with t.span("rag.retrieve", top_k=5):
        pass
    assert client.observations[0]["name"] == "rag.retrieve"
    assert client.observations[0]["metadata"] == {"top_k": 5}

    broken, _ = tracer(fail_start=True)
    with broken.span("rag.retrieve"):
        pass  # no exception


def test_shutdown_flushes():
    t, client = tracer()
    t.shutdown()
    assert client.shutdowns == 1


def test_noop_tracer_yields_a_usable_trace():
    t = NoopTracer()
    assert t.enabled is False
    with t.chat(**chat_kwargs()) as trace:
        trace.output = "ignored"
    with t.span("x"):
        pass
    t.shutdown()


def test_tracing_is_off_without_credentials():
    assert isinstance(build_tracer(Settings(env="test")), NoopTracer)
    partial = Settings(env="test", langfuse_public_key="pk-only")
    assert isinstance(build_tracer(partial), NoopTracer)
