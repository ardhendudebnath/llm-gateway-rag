"""Rolling a route change out over HTTP: publish, watch, promote — or let it withdraw itself."""

import pytest

from tests.conftest import chat_body

CANDIDATE = """
routes:
  default:
    - name: candidate
      provider: fake
      model: fake/candidate
      pricing: { input_per_mtok: 0.5, output_per_mtok: 1.5 }
"""


def bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


async def chat(client, key, content: str):
    return await client.post(
        "/v1/chat/completions", json=chat_body(content, cache=False), headers=bearer(key)
    )


async def publish(client, admin_headers, weight: int = 100, config: str = CANDIDATE):
    return await client.put(
        "/v1/admin/routes/canary",
        json={"config": config, "weight": weight, "note": "cheaper model"},
        headers=admin_headers,
    )


async def test_a_published_canary_serves_traffic_and_says_so(client, api_key, admin_headers):
    resp = await publish(client, admin_headers, weight=100)
    assert resp.status_code == 200, resp.text
    canary = resp.json()["canary"]
    assert canary["weight"] == 100 and canary["active"] and canary["note"] == "cheaper model"
    assert canary["routes"] == {"default": ["candidate"]}

    answered = (await chat(client, api_key, "hello")).json()["nexusgate"]

    assert answered["deployment"] == "candidate"  # the candidate table, not the start-up one
    assert answered["route_variant"] == "canary"


async def test_the_stable_table_still_serves_when_the_weight_is_zero(
    client, api_key, admin_headers
):
    await publish(client, admin_headers, weight=0)

    answered = (await chat(client, api_key, "hello")).json()["nexusgate"]

    assert answered["deployment"] == "primary" and answered["route_variant"] == "stable"


async def test_the_admin_view_reports_how_the_canary_is_doing(client, api_key, admin_headers):
    await publish(client, admin_headers, weight=100)
    for i in range(3):
        await chat(client, api_key, f"question {i}")

    view = (await client.get("/v1/admin/routes", headers=admin_headers)).json()

    assert view["canary"]["stats"] == {"requests": 3, "errors": 0, "error_rate": 0.0}
    assert view["canary"]["rolls_back_above"] == 0.1
    assert view["stable"]["routes"]["default"] == ["primary", "secondary"]


async def test_a_malformed_route_table_is_refused(client, admin_headers):
    resp = await publish(client, admin_headers, config="routes: {default: []}")

    assert resp.status_code == 400
    assert "invalid route table" in resp.json()["error"]["message"]
    assert (await client.get("/v1/admin/routes", headers=admin_headers)).json()["canary"] is None


async def test_a_canary_that_fails_withdraws_itself_and_traffic_returns_to_stable(
    client, api_key, admin_headers, provider, services
):
    services.rollout.settings.min_requests = 3  # the default 20 would need a long test
    services.rollout.settings.max_error_rate = 0.5
    provider.always_fail("candidate")
    await publish(client, admin_headers, weight=100)

    failures = [(await chat(client, api_key, f"q{i}")).status_code for i in range(3)]

    assert failures == [503, 503, 503]  # the canary's only deployment is down
    view = (await client.get("/v1/admin/routes", headers=admin_headers)).json()
    assert view["canary"]["active"] is False and view["canary"]["weight"] == 0
    assert "error rate 100%" in view["canary"]["rollback_reason"]

    recovered = (await chat(client, api_key, "after")).json()["nexusgate"]
    assert recovered["deployment"] == "primary" and recovered["route_variant"] == "stable"


async def test_promoting_leaves_one_table_and_no_canary(client, api_key, admin_headers):
    await publish(client, admin_headers, weight=10)

    promoted = (await client.post("/v1/admin/routes/canary/promote", headers=admin_headers)).json()

    assert promoted["canary"] is None
    assert promoted["stable"]["routes"] == {"default": ["candidate"]}
    answered = (await chat(client, api_key, "hello")).json()["nexusgate"]
    assert answered["deployment"] == "candidate" and answered["route_variant"] == "stable"


async def test_the_weight_can_be_turned_down_without_discarding_the_canary(
    client, api_key, admin_headers
):
    await publish(client, admin_headers, weight=100)

    view = (
        await client.post(
            "/v1/admin/routes/canary/weight", json={"weight": 0}, headers=admin_headers
        )
    ).json()

    assert view["canary"]["weight"] == 0 and view["canary"]["version"]
    assert (await chat(client, api_key, "x")).json()["nexusgate"]["deployment"] == "primary"


async def test_discarding_removes_it(client, api_key, admin_headers):
    await publish(client, admin_headers, weight=100)

    assert (
        await client.delete("/v1/admin/routes/canary", headers=admin_headers)
    ).status_code == 200

    assert (await chat(client, api_key, "x")).json()["nexusgate"]["route_variant"] == "stable"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/v1/admin/routes/canary/promote"),
        ("delete", "/v1/admin/routes/canary"),
    ],
)
async def test_acting_on_a_canary_that_does_not_exist_is_a_404(client, admin_headers, method, path):
    resp = await getattr(client, method)(path, headers=admin_headers)

    assert resp.status_code == 404


async def test_rollout_endpoints_need_the_admin_token(client, api_key):
    resp = await client.get("/v1/admin/routes", headers=bearer(api_key))

    assert resp.status_code == 403
