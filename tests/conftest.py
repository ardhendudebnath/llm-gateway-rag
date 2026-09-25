"""Shared fixtures. Nothing here touches the network or a real Redis."""

from collections import defaultdict, deque
from contextlib import contextmanager

import fakeredis
import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.container import build_services
from app.core.embeddings import HashingEmbedder
from app.gateway.providers import ProviderError
from app.gateway.routing_config import Deployment, RoutingConfig
from app.gateway.schemas import ChatRequest, ProviderResponse, Usage
from app.main import create_app
from app.observability.tracing import ChatTrace

ADMIN_TOKEN = "test-admin-token"


class ScriptedProvider:
    """Deterministic fake provider.

    ``script(name, *outcomes)`` queues outcomes for a deployment: an Exception instance is raised,
    anything else is ignored and a normal response is returned. With an empty queue every call
    succeeds. ``calls`` records deployment names in call order.
    """

    def __init__(self):
        self._scripts: dict[str, deque] = defaultdict(deque)
        self.calls: list[str] = []

    def script(self, deployment: str, *outcomes) -> None:
        self._scripts[deployment].extend(outcomes)

    def always_fail(self, deployment: str, n: int = 100, retryable: bool = True) -> None:
        self.script(
            deployment,
            *[ProviderError(f"{deployment} down", retryable=retryable, status_code=503)] * n,
        )

    async def complete(self, deployment: Deployment, request: ChatRequest) -> ProviderResponse:
        self.calls.append(deployment.name)
        queue = self._scripts[deployment.name]
        outcome = queue.popleft() if queue else None
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            await outcome()
        return ProviderResponse(
            content=f"{deployment.name}: {request.messages[-1].content}",
            usage=Usage(prompt_tokens=100, completion_tokens=50),
            model=deployment.model,
        )


def make_routing() -> RoutingConfig:
    primary = Deployment(
        name="primary",
        provider="fake",
        model="fake/primary",
        pricing={"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    )
    secondary = Deployment(
        name="secondary",
        provider="fake",
        model="fake/secondary",
        pricing={"input_per_mtok": 0.5, "output_per_mtok": 1.5},
    )
    # Priced at zero, and flagged: metering reports self-hosted capacity apart from paid calls.
    local = Deployment(
        name="local",
        provider="fake",
        model="fake/local",
        self_hosted=True,
        pricing={"input_per_mtok": 0.0, "output_per_mtok": 0.0},
    )
    return RoutingConfig(
        routes={"default": [primary, secondary], "single": [primary], "selfhosted": [local]}
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        env="test",
        admin_token=SecretStr(ADMIN_TOKEN),
        jwt_secret=SecretStr("test-jwt-secret-that-is-long-enough-32b"),
        rate_limit_capacity=1000,
        rate_limit_refill_per_sec=100.0,
        cache_similarity_threshold=0.9,
        breaker_failure_threshold=3,
        breaker_cooldown_seconds=30.0,
        provider_timeout_seconds=2.0,
        fault_injection_enabled=True,
        log_level="WARNING",
    )


@pytest.fixture
async def redis_pair():
    server = fakeredis.FakeServer()
    text = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    raw = fakeredis.FakeAsyncRedis(server=server, decode_responses=False)
    yield text, raw
    await text.aclose()
    await raw.aclose()


class RecordingTracer:
    """Stands in for the Langfuse tracer: keeps what would have been sent."""

    enabled = True

    def __init__(self):
        self.chats: list[dict] = []
        self.spans: list[tuple[str, dict]] = []
        self.shutdown_calls = 0

    @contextmanager
    def chat(self, *, tenant_id, key_id, route, messages, model_parameters=None):
        trace = ChatTrace(route=route)
        self.chats.append(
            {
                "tenant_id": tenant_id,
                "key_id": key_id,
                "messages": list(messages),
                "model_parameters": model_parameters,
                "trace": trace,
            }
        )
        try:
            yield trace
        except Exception as e:  # mirrors LangfuseTracer, which records then re-raises
            trace.error = f"{type(e).__name__}: {e}"
            raise

    @contextmanager
    def span(self, name: str, **metadata):
        self.spans.append((name, metadata))
        yield

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def provider() -> ScriptedProvider:
    return ScriptedProvider()


@pytest.fixture
def tracer() -> RecordingTracer:
    return RecordingTracer()


@pytest.fixture
async def services(settings, redis_pair, provider, tracer):
    text, raw = redis_pair
    return await build_services(
        settings,
        redis=text,
        cache_redis=raw,
        providers={"fake": provider},
        routing=make_routing(),
        embedder=HashingEmbedder(),
        tracer=tracer,
    )


@pytest.fixture
async def client(settings, services):
    app = create_app(settings, services)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def admin_headers() -> dict:
    return {"X-Admin-Token": ADMIN_TOKEN}


@pytest.fixture
async def api_key(client, admin_headers) -> str:
    resp = await client.post(
        "/v1/admin/keys", json={"tenant_id": "acme", "name": "ci"}, headers=admin_headers
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["api_key"]


def chat_body(content: str = "What is the capital of France?", **extra) -> dict:
    return {"model": "default", "messages": [{"role": "user", "content": content}], **extra}
