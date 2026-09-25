"""Stepped load test: latency percentiles per load level, and the throughput ceiling.

    python loadtest/in_cluster.py --label baseline    # recommended: runs this inside the cluster
    python loadtest/run_load.py --admin-token <t> --label x --base-url http://localhost:8000

Run it from inside the cluster (loadtest/in_cluster.py does that). Driven from the host on
Windows, requests also cross Podman's port forwarder, which added ~2 s to about 5% of requests in
testing: the numbers then describe the laptop, not the service.

For each level it runs Locust headless for a fixed time with every user spawned at once, and
`--reset-stats` so the ramp-up isn't counted. It records requests/s and p50/p95/p99 per request
kind, then reports the ceiling: the highest throughput whose p95 stays under the SLO with under 1%
errors. The result is written to <results-dir>/<label>.json and printed on a `RESULT_JSON:` line.

Setup, once per run: an API key with a very high rate limit (otherwise the gateway's own limiter
is what "breaks"), the eval corpus ingested for RAG traffic, and the popular questions answered
once so they are cache hits during the test. Uses `requests`, which ships with Locust, so it runs
in the stock Locust image.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from traffic import POPULAR  # noqa: E402  (not locustfile: see traffic.py)


class Api:
    def __init__(self, base: str, headers: dict | None = None):
        self.base = base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(headers or {})

    def call(self, method: str, path: str, **kwargs) -> requests.Response:
        resp = self.session.request(method, self.base + path, timeout=120, **kwargs)
        if resp.status_code >= 400:
            raise SystemExit(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp


def setup(base: str, admin_token: str, corpus: Path) -> tuple[str, str]:
    admin = Api(base, {"X-Admin-Token": admin_token})
    created = admin.call(
        "POST",
        "/v1/admin/keys",
        json={
            "tenant_id": "loadtest",
            "name": f"locust-{int(time.time())}",
            "rate_limit_capacity": 1_000_000,
            "rate_limit_refill_per_sec": 1_000_000,
        },
    ).json()
    key, key_id = created["api_key"], created["record"]["key_id"]
    api = Api(base, {"Authorization": f"Bearer {key}"})

    job_ids = [
        api.call(
            "POST",
            "/v1/rag/documents",
            files={"file": (p.name, p.read_bytes(), "text/markdown")},
        ).json()["job_id"]
        for p in sorted(corpus.glob("*.md"))
    ]
    deadline = time.monotonic() + 180
    pending = set(job_ids)
    while pending and time.monotonic() < deadline:
        for job_id in list(pending):
            if api.call("GET", f"/v1/rag/jobs/{job_id}").json()["status"] == "done":
                pending.discard(job_id)
        time.sleep(1)
    if pending:
        raise SystemExit(f"{len(pending)} ingestion jobs did not finish")

    for question in POPULAR:  # warm the semantic cache
        api.call(
            "POST",
            "/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": question}]},
        )
    print(f"setup: key {key_id}, {len(job_ids)} documents ingested, cache warmed", flush=True)
    return key, key_id


def run_level(base: str, key: str, users: int, duration: int, prefix: Path) -> dict:
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(HERE / "locustfile.py"),
        "--headless", "--only-summary", "--reset-stats",
        "-u", str(users), "-r", str(users), "-t", f"{duration}s",
        "--host", base, "--csv", str(prefix), "--stop-timeout", "10",
    ]  # fmt: skip
    subprocess.run(
        cmd,
        env={**os.environ, "NEXUSGATE_LOADTEST_KEY": key},
        capture_output=True,
        check=False,
    )
    with open(f"{prefix}_stats.csv", newline="", encoding="utf-8") as f:
        rows = {r["Name"]: r for r in csv.DictReader(f)}

    def summarise(r: dict) -> dict:
        count = int(r["Request Count"])
        return {
            "requests": count,
            "rps": round(float(r["Requests/s"]), 1),
            "p50_ms": float(r["50%"]),
            "p95_ms": float(r["95%"]),
            "p99_ms": float(r["99%"]),
            "fail_pct": round(100 * int(r["Failure Count"]) / count, 2) if count else 0.0,
        }

    for suffix in ("_stats.csv", "_stats_history.csv", "_failures.csv", "_exceptions.csv"):
        Path(f"{prefix}{suffix}").unlink(missing_ok=True)
    return {
        "users": users,
        **summarise(rows["Aggregated"]),
        "by_kind": {name: summarise(r) for name, r in rows.items() if name != "Aggregated"},
    }


def usage(base: str, key: str) -> dict:
    api = Api(base, {"Authorization": f"Bearer {key}"})
    body = api.call("GET", "/v1/usage", params={"days": 1}).json()
    totals = body["totals"]
    requests_ = totals["requests"] or 1
    return {
        "requests": totals["requests"],
        "cache_hits": totals["cache_hits"],
        "cache_hit_rate": round(totals["cache_hits"] / requests_, 3),
        "cost_usd": totals["cost_usd"],
        "cost_saved_usd": totals["cost_saved_usd"],
        "cost_per_1k_requests_usd": round(1000 * totals["cost_usd"] / requests_, 6),
        "saved_per_1k_requests_usd": round(1000 * totals["cost_saved_usd"] / requests_, 6),
        # Per request kind, straight from the gateway's own metering: the blended figure above
        # hides whether the cache or cheaper capacity moved it.
        "cost_per_1k_by_kind": body.get("cost_per_1k_usd"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://api:8000")
    parser.add_argument("--admin-token", default=os.environ.get("NEXUSGATE_ADMIN_TOKEN"))
    parser.add_argument("--levels", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    parser.add_argument("--duration", type=int, default=45, help="seconds per level")
    parser.add_argument("--slo-ms", type=float, default=500, help="p95 latency objective")
    parser.add_argument("--label", default="run")
    parser.add_argument("--corpus", type=Path, default=HERE.parent / "eval" / "corpus")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if not args.admin_token:
        parser.error("--admin-token (or NEXUSGATE_ADMIN_TOKEN) is required")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    key, key_id = setup(args.base_url, args.admin_token, args.corpus)
    levels = []
    try:
        for users in args.levels:
            level = run_level(
                args.base_url,
                key,
                users,
                args.duration,
                args.results_dir / f"{args.label}-{users}u",
            )
            levels.append(level)
            print(
                f"{users:>4} users: {level['rps']:>6} req/s  p50 {level['p50_ms']:>5.0f} ms  "
                f"p95 {level['p95_ms']:>5.0f} ms  p99 {level['p99_ms']:>5.0f} ms  "
                f"errors {level['fail_pct']}%",
                flush=True,
            )
        costs = usage(args.base_url, key)
    finally:
        Api(args.base_url, {"X-Admin-Token": args.admin_token}).session.delete(
            f"{args.base_url}/v1/admin/keys/{key_id}", timeout=30
        )

    within_slo = [lv for lv in levels if lv["p95_ms"] <= args.slo_ms and lv["fail_pct"] < 1]
    ceiling = max(within_slo, key=lambda lv: lv["rps"]) if within_slo else None
    result = {
        "label": args.label,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "slo_p95_ms": args.slo_ms,
        "duration_s": args.duration,
        "levels": levels,
        "ceiling": {"users": ceiling["users"], "rps": ceiling["rps"]} if ceiling else None,
        "usage": costs,
    }
    (args.results_dir / f"{args.label}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    reached = f"{ceiling['rps']} req/s at {ceiling['users']} users" if ceiling else "none"
    print(f"\nceiling under p95 <= {args.slo_ms:.0f} ms: {reached}")
    print(
        f"cache hit rate {costs['cache_hit_rate']:.1%}, "
        f"cost/1k requests ${costs['cost_per_1k_requests_usd']}, "
        f"saved/1k ${costs['saved_per_1k_requests_usd']}"
    )
    print("RESULT_JSON:" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
