"""Chaos test: break the primary provider under steady traffic, then heal it.

    python loadtest/in_cluster.py --script chaos.py --label chaos

Timeline (defaults): 15 s healthy, then a fault injected on `mock-primary` for 45 s, then the
fault cleared and 45 s of recovery. Traffic is a fixed 20 requests/s to the `mock` route with the
cache off, so every request reaches the router. The fault is injected through the admin API, which
fails the deployment exactly where a real provider error would, on every replica.

Reported per phase: success rate, share served by the primary vs the fallback, p50/p95 latency,
and cost per request (the fallback is priced 5x the primary). Plus the reliability numbers:

* user-visible failures (the goal is zero)
* primary failures absorbed by fallback, split into the ones before the breakers opened and the
  deliberate half-open probes the breakers send each cooldown while the primary stays down
* seconds from the fault until the breakers opened
* seconds from clearing the fault until the primary served again (up to one cooldown, since the
  breakers only re-check the primary once per cooldown)

Uses only `requests` and threads, so it runs in the stock Locust image.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://api:8000")
    parser.add_argument("--admin-token", default=os.environ.get("NEXUSGATE_ADMIN_TOKEN"))
    parser.add_argument("--rate", type=float, default=20, help="requests per second")
    parser.add_argument("--healthy", type=float, default=15)
    parser.add_argument("--fault", type=float, default=45)
    parser.add_argument("--recovery", type=float, default=45)
    parser.add_argument("--deployment", default="mock-primary")
    parser.add_argument(
        "--breaker-cooldown", type=float, default=30, help="the gateway's breaker cooldown (s)"
    )
    parser.add_argument("--label", default="chaos")
    parser.add_argument("--corpus", type=Path, help="ignored; accepted for in_cluster.py")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    admin = {"X-Admin-Token": args.admin_token}

    created = requests.post(
        f"{base}/v1/admin/keys",
        headers=admin,
        json={
            "tenant_id": "chaos",
            "name": f"chaos-{int(time.time())}",
            "rate_limit_capacity": 1_000_000,
            "rate_limit_refill_per_sec": 1_000_000,
        },
        timeout=30,
    ).json()
    auth = {"Authorization": f"Bearer {created['api_key']}"}
    fault_url = f"{base}/v1/admin/faults/{args.deployment}"
    requests.delete(fault_url, headers=admin, timeout=30)  # start clean

    records: list[dict] = []
    lock = threading.Lock()
    start = time.monotonic()

    def send() -> None:
        sent = time.monotonic() - start
        t0 = time.perf_counter()
        try:
            resp = session().post(
                f"{base}/v1/chat/completions",
                headers=auth,
                json={
                    "model": "mock",
                    "cache": False,
                    "messages": [{"role": "user", "content": "status check"}],
                },
                timeout=30,
            )
            body = resp.json() if resp.status_code == 200 else {}
            meta = body.get("nexusgate", {})
            record = {
                "t": sent,
                "status": resp.status_code,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "deployment": meta.get("deployment"),
                "cost_usd": meta.get("cost_usd", 0.0),
                "primary_failed": any(
                    a["deployment"] == args.deployment and a["outcome"] in {"error", "timeout"}
                    for a in meta.get("attempts", [])
                ),
            }
        except requests.RequestException as e:
            record = {"t": sent, "status": 0, "latency_ms": 0.0, "deployment": None,
                      "cost_usd": 0.0, "primary_failed": False, "error": str(e)}  # fmt: skip
        with lock:
            records.append(record)

    fault_at = args.healthy
    clear_at = args.healthy + args.fault
    total = clear_at + args.recovery
    injected = cleared = False
    print(f"chaos: {args.rate} req/s for {total:.0f}s; fault on {args.deployment} "
          f"at {fault_at:.0f}s, cleared at {clear_at:.0f}s", flush=True)  # fmt: skip
    with ThreadPoolExecutor(max_workers=64) as pool:
        for i in range(int(total * args.rate)):
            target = start + i / args.rate
            time.sleep(max(0.0, target - time.monotonic()))
            now = time.monotonic() - start
            if not injected and now >= fault_at:
                requests.put(fault_url, headers=admin, json={"failure_rate": 1.0}, timeout=30)
                injected = True
                print(f"  {now:5.1f}s fault injected", flush=True)
            if not cleared and now >= clear_at:
                requests.delete(fault_url, headers=admin, timeout=30)
                cleared = True
                print(f"  {now:5.1f}s fault cleared", flush=True)
            pool.submit(send)

    requests.delete(fault_url, headers=admin, timeout=30)
    requests.delete(
        f"{base}/v1/admin/keys/{created['record']['key_id']}", headers=admin, timeout=30
    )

    def phase_of(t: float) -> str:
        return "healthy" if t < fault_at else "fault" if t < clear_at else "recovery"

    phases = {}
    for name in ("healthy", "fault", "recovery"):
        rows = [r for r in records if phase_of(r["t"]) == name]
        ok = [r for r in rows if r["status"] == 200]
        latencies = [r["latency_ms"] for r in ok]
        phases[name] = {
            "requests": len(rows),
            "success_pct": round(100 * len(ok) / len(rows), 2) if rows else 0.0,
            "served_by_primary_pct": round(
                100 * sum(r["deployment"] == args.deployment for r in ok) / max(1, len(ok)), 1
            ),
            "p50_ms": round(percentile(latencies, 0.50), 1),
            "p95_ms": round(percentile(latencies, 0.95), 1),
            "cost_per_1k_requests_usd": round(
                1000 * sum(r["cost_usd"] for r in ok) / max(1, len(ok)), 6
            ),
        }

    # Failures before the first cooldown elapses are the breakers learning the primary is down.
    # Later ones are deliberate half-open probes: one request per replica per cooldown, sent to
    # check whether the primary has recovered. Counting those as "still not bypassed" would
    # misreport the breaker working as designed.
    failure_times = sorted(
        round(r["t"] - fault_at, 2) for r in records if r["primary_failed"] and r["status"] == 200
    )
    initial = [t for t in failure_times if t < args.breaker_cooldown]
    probes = [t for t in failure_times if t >= args.breaker_cooldown]
    back = [
        r["t"] for r in records
        if phase_of(r["t"]) == "recovery" and r["deployment"] == args.deployment
    ]  # fmt: skip
    result = {
        "label": args.label,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rate_rps": args.rate,
        "breaker_cooldown_s": args.breaker_cooldown,
        "timeline_s": {"fault_at": fault_at, "cleared_at": clear_at, "end": total},
        "phases": phases,
        "user_visible_failures": sum(r["status"] != 200 for r in records),
        "primary_failures_absorbed_by_fallback": len(failure_times),
        "failures_before_breakers_opened": len(initial),
        "seconds_until_breakers_opened": max(initial) if initial else None,
        "failed_half_open_probes": len(probes),
        "primary_failure_times_after_fault_s": failure_times,
        "seconds_until_primary_served_again": round(min(back) - clear_at, 2) if back else None,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / f"{args.label}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    for name, p in phases.items():
        print(
            f"  {name:>8}: {p['requests']} req, {p['success_pct']}% ok, "
            f"{p['served_by_primary_pct']}% by primary, p50 {p['p50_ms']} ms, "
            f"p95 {p['p95_ms']} ms, ${p['cost_per_1k_requests_usd']}/1k"
        )
    print(
        f"  user-visible failures: {result['user_visible_failures']}; primary failures "
        f"absorbed by fallback: {result['primary_failures_absorbed_by_fallback']}"
    )
    print(
        f"  breakers opened {result['seconds_until_breakers_opened']}s after the fault, after "
        f"{result['failures_before_breakers_opened']} failures; "
        f"{result['failed_half_open_probes']} failed probes while it stayed down; primary serving "
        f"again {result['seconds_until_primary_served_again']}s after the fix"
    )
    print(f"  primary failures at (s after fault): {failure_times}")
    print("RESULT_JSON:" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
