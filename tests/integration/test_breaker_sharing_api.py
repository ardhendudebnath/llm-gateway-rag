"""Shared breaker state through the API: what one replica learns, the next pod already knows."""

import httpx

from app.core.container import build_services
from app.core.embeddings import HashingEmbedder
from app.main import create_app
from tests.conftest import ScriptedProvider, chat_body, make_routing
from tests.integration.test_rag_api import auth


class replica:
    """One more API replica over the same Redis, with its own provider and its own breakers."""

    def __init__(self, settings, redis_pair, provider: ScriptedProvider):
        self._settings = settings
        self._redis_pair = redis_pair
        self._provider = provider

    async def __aenter__(self) -> httpx.AsyncClient:
        text, raw = self._redis_pair
        services = await build_services(
            self._settings,
            redis=text,
            cache_redis=raw,
            providers={"fake": self._provider},
            routing=make_routing(),
            embedder=HashingEmbedder(),
        )
        app = create_app(self._settings, services)
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://replica"
        )
        return await self._client.__aenter__()

    async def __aexit__(self, *exc) -> None:
        await self._client.__aexit__(*exc)


async def chat(client, key: str, content: str = "hello"):
    return await client.post(
        "/v1/chat/completions", json=chat_body(content, cache=False), headers=auth(key)
    )


async def test_the_admin_view_shows_the_cluster_state_beside_the_local_one(
    client, api_key, admin_headers, provider
):
    provider.always_fail("primary")
    for _ in range(3):  # breaker_failure_threshold in the test settings
        await chat(client, api_key)

    body = (await client.get("/v1/admin/providers", headers=admin_headers)).json()

    assert body["breakers"]["primary"]["state"] == "open", "this replica"
    shared = body["shared"]["primary"]
    assert shared["open"] is True, "and every replica"
    assert 0 < shared["cooldown_remaining_s"] <= 30
    assert body["shared"]["secondary"]["open"] is False


async def test_a_second_replica_skips_a_deployment_it_never_called(settings, redis_pair, api_key):
    """A pod joining a cluster with an open circuit must not have to learn it the hard way."""
    first = ScriptedProvider()
    first.always_fail("primary")
    async with replica(settings, redis_pair, first) as one:
        for _ in range(3):
            await chat(one, api_key)
        assert first.calls.count("primary") == 3

    second = ScriptedProvider()  # a pod that has never seen primary fail
    second.always_fail("primary")  # it is still down, if anyone asks
    async with replica(settings, redis_pair, second) as two:
        answered = await chat(two, api_key)

    assert answered.status_code == 200
    meta = answered.json()["nexusgate"]
    assert meta["deployment"] == "secondary"
    assert meta["attempts"][0]["outcome"] == "skipped_circuit_open"
    assert second.calls == ["secondary"], "it never called the failing deployment at all"


async def test_the_admin_view_is_not_stale_on_a_replica_that_has_been_idle(
    settings, redis_pair, api_key, admin_headers
):
    """Found on the cluster: a pod that had served no requests reported a closed circuit next to
    shared state saying it was open. The two columns must agree."""
    first = ScriptedProvider()
    first.always_fail("primary")
    async with replica(settings, redis_pair, first) as one:
        for _ in range(3):
            await chat(one, api_key)

    idle = ScriptedProvider()  # this replica never serves a chat request at all
    async with replica(settings, redis_pair, idle) as two:
        body = (await two.get("/v1/admin/providers", headers=admin_headers)).json()

    assert body["shared"]["primary"]["open"] is True
    assert body["breakers"]["primary"]["state"] == "open", (
        "the local view was refreshed for the read"
    )
    assert idle.calls == []


async def test_without_sharing_every_replica_learns_the_hard_way(settings, redis_pair, api_key):
    alone = settings.model_copy(update={"breaker_shared": False})
    first = ScriptedProvider()
    first.always_fail("primary")
    async with replica(alone, redis_pair, first) as one:
        for _ in range(3):
            await chat(one, api_key)

    second = ScriptedProvider()
    second.always_fail("primary")
    async with replica(alone, redis_pair, second) as two:
        await chat(two, api_key)

    # The only difference from the test above is the setting: here the user's request pays for a
    # call to a deployment another replica already knows is down.
    assert second.calls == ["primary", "secondary"], "it had to find out for itself"


async def test_the_admin_view_says_nothing_is_shared_when_sharing_is_off(
    settings, redis_pair, admin_headers
):
    alone = settings.model_copy(update={"breaker_shared": False})

    async with replica(alone, redis_pair, ScriptedProvider()) as one:
        body = (await one.get("/v1/admin/providers", headers=admin_headers)).json()

    assert body["shared"] is None
    assert body["breakers"]["primary"]["state"] == "closed"
