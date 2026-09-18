from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.core.container import check_production_secrets
from app.core.metering import UsageMeter
from app.gateway.pricing import compute_cost
from app.gateway.routing_config import Deployment, RoutingConfig
from app.gateway.schemas import Usage

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shipped_routes_yaml_is_valid():
    config = RoutingConfig.from_yaml(REPO_ROOT / "config" / "routes.yaml")
    assert {"default", "mock", "chaos"} <= set(config.routes)
    assert {d.provider for d in config.deployments.values()} <= {"litellm", "mock"}


def test_empty_route_rejected():
    with pytest.raises(ValidationError, match="no deployments"):
        RoutingConfig(routes={"default": []})


def test_conflicting_deployment_names_rejected():
    a = Deployment(name="x", provider="mock", model="a")
    b = Deployment(name="x", provider="mock", model="b")
    with pytest.raises(ValidationError, match="defined twice"):
        RoutingConfig(routes={"r1": [a], "r2": [b]})


def test_same_deployment_shared_across_routes_is_fine():
    a = Deployment(name="x", provider="mock", model="a")
    assert RoutingConfig(routes={"r1": [a], "r2": [a]}).deployments == {"x": a}


def test_explicit_pricing():
    dep = Deployment(
        name="d", provider="mock", model="m", pricing={"input_per_mtok": 2, "output_per_mtok": 8}
    )
    cost = compute_cost(dep, Usage(prompt_tokens=1_000_000, completion_tokens=500_000))
    assert cost == pytest.approx(6.0)


def test_unknown_model_without_pricing_costs_zero():
    dep = Deployment(name="d", provider="mock", model="definitely-not-a-real-model-xyz")
    assert compute_cost(dep, Usage(prompt_tokens=10, completion_tokens=10)) == 0.0


def test_production_refuses_default_secrets():
    with pytest.raises(RuntimeError, match="ADMIN_TOKEN"):
        check_production_secrets(Settings(env="prod"))
    check_production_secrets(
        Settings(env="prod", admin_token=SecretStr("a" * 32), jwt_secret=SecretStr("b" * 32))
    )


async def test_metering_buckets_by_day_and_separates_cache_hits(redis_pair):
    meter = UsageMeter(redis_pair[0], retention_days=90)
    day1 = datetime(2026, 9, 17, 12, tzinfo=UTC)
    day2 = datetime(2026, 9, 18, 12, tzinfo=UTC)
    await meter.record(
        "k", prompt_tokens=100, completion_tokens=50, cost_usd=0.01, cached=False, now=day1
    )
    await meter.record(
        "k", prompt_tokens=100, completion_tokens=50, cost_usd=0.01, cached=False, now=day2
    )
    await meter.record(
        "k",
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=0,
        cached=True,
        cost_saved_usd=0.01,
        now=day2,
    )

    rows = await meter.daily("k", days=3, today=date(2026, 9, 18))
    assert [r.date.day for r in rows] == [16, 17, 18]
    assert rows[0].requests == 0
    assert (rows[2].requests, rows[2].cache_hits, rows[2].prompt_tokens) == (2, 1, 100)
    assert rows[2].cost_usd == pytest.approx(0.01)
    assert rows[2].cost_saved_usd == pytest.approx(0.01)
