"""End-to-end smoke test against a running NexusGate stack (the kind cluster, or any deployment).

    python scripts/smoke_test.py --admin-token <token> [--prometheus-url http://localhost:9090]

Goes through the real network path: key issuance, a cache miss then a hit, fallback on the `chaos`
route, usage metering, /metrics, and (optionally) Prometheus scraping every API replica. It only
uses the offline mock routes, so it needs no provider API keys and spends nothing. It cleans up the
key and cache entries it creates, and exits non-zero on the first failed check.
"""

import argparse
import os
import sys
import time
import uuid

import httpx


class CheckFailedError(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailedError(message)


def ok(message: str) -> None:
    print(f"  ok  {message}")


def wait_ready(api: httpx.Client, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if api.get("/readyz").status_code == 200:
                return
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            raise CheckFailedError(f"/readyz did not return 200 within {timeout_s:.0f}s")
        time.sleep(2)


def chat(api: httpx.Client, key: str, model: str, prompt: str, *, cache: bool) -> httpx.Response:
    return api.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "cache": cache, "messages": [{"role": "user", "content": prompt}]},
    )


def check_gateway(api: httpx.Client, admin_token: str) -> None:
    admin = {"X-Admin-Token": admin_token}
    # A fresh tenant per run: the cache namespace includes the tenant, so a near-identical prompt
    # from a previous run can never turn this run's expected miss into a hit.
    run_id = uuid.uuid4().hex[:10]
    r = api.post(
        "/v1/admin/keys", headers=admin, json={"tenant_id": f"smoke-{run_id}", "name": "smoke"}
    )
    check(r.status_code == 201, f"issue API key: expected 201, got {r.status_code} {r.text}")
    key, key_id = r.json()["api_key"], r.json()["record"]["key_id"]
    auth = {"Authorization": f"Bearer {key}"}
    ok("issued an API key via the admin API")

    try:
        prompt = "What does a circuit breaker do in a distributed system?"
        first = chat(api, key, "mock", prompt, cache=True)
        check(first.status_code == 200, f"chat: expected 200, got {first.status_code} {first.text}")
        check(first.headers.get("x-nexusgate-cache") == "miss", "first call should be a cache miss")
        second = chat(api, key, "mock", prompt, cache=True)
        check(second.headers.get("x-nexusgate-cache") == "hit", "repeat call should be a cache hit")
        check(second.json()["nexusgate"]["cost_usd"] == 0, "a cache hit should cost $0")
        ok("mock route: miss, then a $0 cache hit on the repeat")

        r = chat(api, key, "chaos", "ping", cache=False)
        check(
            r.status_code == 200, f"chaos: expected 200 via fallback, got {r.status_code} {r.text}"
        )
        meta = r.json()["nexusgate"]
        check(meta["deployment"] == "mock-primary", f"chaos served by {meta['deployment']}")
        broken = [a["outcome"] for a in meta["attempts"] if a["deployment"] == "mock-broken"]
        check(
            broken and broken[0] in {"error", "skipped_circuit_open"},
            f"attempts: {meta['attempts']}",
        )
        ok(f"chaos route: primary {broken[0]}, fallback answered")

        totals = api.get("/v1/usage", params={"days": 1}, headers=auth).json()["totals"]
        check(totals["requests"] == 3 and totals["cache_hits"] == 1, f"usage totals: {totals}")
        ok("usage metering: 3 requests, 1 cache hit")

        metrics = api.get("/metrics").text
        check("nexusgate_http_requests_total" in metrics, "/metrics lacks nexusgate_ metrics")
        ok("/metrics exports nexusgate_* series")
    finally:
        api.delete("/v1/cache", headers=auth)
        api.delete(f"/v1/admin/keys/{key_id}", headers=admin)


def check_prometheus(url: str, expected_targets: int, timeout_s: float) -> None:
    # DNS service discovery refreshes every 15s, so a fresh rollout may take a moment to appear.
    deadline = time.monotonic() + timeout_s
    with httpx.Client(base_url=url, timeout=10) as prom:
        while True:
            result = prom.get("/api/v1/query", params={"query": 'up{job="nexusgate-api"}'}).json()
            series = result["data"]["result"]
            healthy = [s for s in series if s["value"][1] == "1"]
            if len(series) == expected_targets and len(healthy) == expected_targets:
                ok(f"Prometheus scrapes {expected_targets} API pods, all up")
                return
            if time.monotonic() > deadline:
                raise CheckFailedError(
                    f"Prometheus: expected {expected_targets} healthy API targets, "
                    f"got {len(healthy)} of {len(series)}"
                )
            time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--admin-token", default=os.environ.get("NEXUSGATE_ADMIN_TOKEN"))
    parser.add_argument("--prometheus-url", help="also check that Prometheus scrapes every pod")
    parser.add_argument("--api-replicas", type=int, default=2)
    parser.add_argument("--wait", type=float, default=120, help="seconds to wait for readiness")
    args = parser.parse_args()
    if not args.admin_token:
        parser.error("--admin-token (or NEXUSGATE_ADMIN_TOKEN) is required")

    print(f"Smoke test against {args.base_url}")
    try:
        with httpx.Client(base_url=args.base_url, timeout=30) as api:
            wait_ready(api, args.wait)
            ok("/readyz: API is ready and Redis reachable")
            check_gateway(api, args.admin_token)
        if args.prometheus_url:
            check_prometheus(args.prometheus_url, args.api_replicas, args.wait)
    except (CheckFailedError, httpx.HTTPError) as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
