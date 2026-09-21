"""Fault injection through the admin API, exercising the real fallback and breaker paths."""

from tests.conftest import chat_body
from tests.integration.test_rag_api import auth


async def test_an_injected_fault_is_absorbed_by_fallback(client, api_key, admin_headers, provider):
    put = await client.put(
        "/v1/admin/faults/primary", headers=admin_headers, json={"failure_rate": 1.0}
    )
    assert put.json() == {"deployment": "primary", "failure_rate": 1.0}

    resp = await client.post(
        "/v1/chat/completions", json=chat_body(cache=False), headers=auth(api_key)
    )
    assert resp.status_code == 200
    meta = resp.json()["nexusgate"]
    assert meta["deployment"] == "secondary"
    assert meta["attempts"][0] == {
        **meta["attempts"][0],
        "deployment": "primary",
        "outcome": "error",
    }
    assert "injected fault" in meta["attempts"][0]["error"]
    assert provider.calls == ["secondary"]  # the fault fired before primary was ever called


async def test_repeated_injected_faults_open_the_breaker(client, api_key, admin_headers):
    await client.put("/v1/admin/faults/primary", headers=admin_headers, json={"failure_rate": 1.0})
    for _ in range(3):  # breaker_failure_threshold in the test settings
        await client.post(
            "/v1/chat/completions", json=chat_body(cache=False), headers=auth(api_key)
        )

    breakers = (await client.get("/v1/admin/providers", headers=admin_headers)).json()["breakers"]
    assert breakers["primary"]["state"] == "open"

    resp = await client.post(
        "/v1/chat/completions", json=chat_body(cache=False), headers=auth(api_key)
    )
    assert resp.json()["nexusgate"]["attempts"][0]["outcome"] == "skipped_circuit_open"


async def test_clearing_a_fault_restores_the_primary(client, api_key, admin_headers):
    await client.put("/v1/admin/faults/primary", headers=admin_headers, json={"failure_rate": 1.0})
    assert (await client.get("/v1/admin/faults", headers=admin_headers)).json() == {
        "faults": {"primary": 1.0}
    }
    gone = await client.delete("/v1/admin/faults/primary", headers=admin_headers)
    assert gone.status_code == 204

    resp = await client.post(
        "/v1/chat/completions", json=chat_body(cache=False), headers=auth(api_key)
    )
    assert resp.json()["nexusgate"]["deployment"] == "primary"


async def test_fault_endpoints_validate_input(client, admin_headers):
    unknown = await client.put(
        "/v1/admin/faults/nope", headers=admin_headers, json={"failure_rate": 1.0}
    )
    assert unknown.status_code == 404
    bad_rate = await client.put(
        "/v1/admin/faults/primary", headers=admin_headers, json={"failure_rate": 1.5}
    )
    assert bad_rate.status_code == 422
    assert (
        await client.delete("/v1/admin/faults/primary", headers=admin_headers)
    ).status_code == 404


async def test_fault_endpoints_are_admin_only(client, api_key):
    resp = await client.put(
        "/v1/admin/faults/primary", headers=auth(api_key), json={"failure_rate": 1.0}
    )
    assert resp.status_code == 403


async def test_fault_injection_is_refused_when_disabled(client, admin_headers, services):
    services.router.faults = None  # the production default
    resp = await client.put(
        "/v1/admin/faults/primary", headers=admin_headers, json={"failure_rate": 1.0}
    )
    assert resp.status_code == 409
    assert "disabled" in resp.json()["error"]["message"]
