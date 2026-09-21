"""Public demo mode: landing page with a working key, handbook pre-loaded, uploads closed."""

import httpx
import pytest

from app.demo import DEMO_TENANT, prepare_demo
from app.main import create_app
from tests.conftest import ADMIN_TOKEN


@pytest.fixture
async def demo_client(settings, services):
    settings = settings.model_copy(update={"demo_mode": True, "demo_rate_limit_capacity": 50})
    services.settings = settings
    app = create_app(settings, services)
    # httpx's ASGI transport doesn't run the lifespan, so do what start-up would.
    app.state.demo_key = await prepare_demo(services)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://demo.test"
    ) as client:
        yield client, app.state.demo_key


async def test_landing_page_shows_a_key_that_works(demo_client):
    client, key = demo_client
    page = await client.get("/")
    assert page.status_code == 200
    assert key in page.text and "http://demo.test/v1/chat/completions" in page.text

    # Tests use the lexical hashing embedder, so the query shares words with the passage.
    resp = await client.post(
        "/v1/rag/search",
        headers={"Authorization": f"Bearer {key}"},
        json={"query": "postmortem draft due within three business days"},
    )
    assert resp.status_code == 200
    titles = [h["title"] for h in resp.json()["hits"]]
    assert "Incident response" in titles  # the handbook was pre-loaded


async def test_the_demo_key_is_rate_limited_and_scoped_to_the_demo_tenant(demo_client, services):
    _, key = demo_client
    record = await services.keys.verify(key)
    assert record.tenant_id == DEMO_TENANT
    assert record.rate_limit_capacity == 50  # never the unlimited defaults


async def test_uploads_are_disabled_in_the_demo(demo_client):
    client, key = demo_client
    resp = await client.post(
        "/v1/rag/documents",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": ("x.md", b"# Anything", "text/markdown")},
    )
    assert resp.status_code == 403
    assert "disabled in the public demo" in resp.json()["error"]["message"]


async def test_the_demo_key_cannot_use_admin_endpoints(demo_client):
    client, key = demo_client
    resp = await client.get("/v1/admin/providers", headers={"X-Admin-Token": key})
    assert resp.status_code == 403
    ok = await client.get("/v1/admin/providers", headers={"X-Admin-Token": ADMIN_TOKEN})
    assert ok.status_code == 200  # the real (secret) admin token still works


async def test_the_api_docs_offer_an_authorize_button(demo_client):
    # The landing page sends visitors to /docs to paste the key, so the scheme must be declared.
    client, _ = demo_client
    spec = (await client.get("/openapi.json")).json()
    assert spec["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    assert {"HTTPBearer": []} in spec["paths"]["/v1/chat/completions"]["post"]["security"]
    assert "security" not in spec["paths"]["/healthz"]["get"]


async def test_there_is_no_landing_page_outside_demo_mode(client):
    assert (await client.get("/")).status_code == 404


def test_the_demo_routes_are_offline_only():
    from pathlib import Path

    from app.gateway.routing_config import RoutingConfig

    root = Path(__file__).resolve().parents[2]
    config = RoutingConfig.from_yaml(root / "config" / "routes.demo.yaml")
    assert {d.provider for d in config.deployments.values()} == {"mock"}
