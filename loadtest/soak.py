"""Soak test: hold a modest load for an hour and watch for drift the burst tests cannot see.

    python loadtest/in_cluster.py --script soak.py --label soak            # one hour
    python loadtest/in_cluster.py --script soak.py --label soak -- --minutes 20

The stepped load test answers "how much traffic before it misses its objective". It says nothing
about what happens over time at a load the gateway handles comfortably, which is the regime a real
deployment lives in. Three things in this gateway are worth an hour of suspicion:

* **Memory.** Bounded inference was added because unbounded ONNX work OOM-killed the API at 50
  users (loadtest/README.md). That fixed a cliff; a slow leak would look nothing like it.
* **Background work.** A streamed request whose client disconnects hands its metering to an
  asyncio task, and the task set that keeps those alive is module-level. A leak there would grow
  resident memory without any request being slow.
* **Redis keys.** Canary rollouts count every request, and pooled circuit state writes per
  deployment. Both are meant to expire or be bounded; neither is checked by any other test.

It samples Prometheus rather than measuring from the client, because the question is what happens
*inside* the pods. Latency is compared between the first and last quarter of the run: a soak that
only reports averages hides a slope.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_load import Api, setup  # noqa: E402  (same in-cluster setup as the stepped test)

# What is sampled, and what each series would show if something were wrong.
SERIES = {
    "api_rss_bytes": 'process_resident_memory_bytes{job="nexusgate-api"}',
    "api_open_fds": 'process_open_fds{job="nexusgate-api"}',
    "gc_objects": 'python_gc_objects_collected_total{job="nexusgate-api"}',
    "requests_total": 'sum(nexusgate_http_requests_total{job="nexusgate-api"})',
    "errors_total": 'sum(nexusgate_http_requests_total{job="nexusgate-api",status=~"5.."})',
    "circuits_open": 'sum(nexusgate_circuit_open{job="nexusgate-api"})',
    "degraded_total": 'sum(nexusgate_degraded_total{job="nexusgate-api"})',
    "shed_total": 'sum(nexusgate_inference_shed_total{job="nexusgate-api"})',
    "inference_waiting": 'sum(nexusgate_inference_waiting{job="nexusgate-api"})',
    "latency_p95": (
        "histogram_quantile(0.95, sum by (le) "
        '(rate(nexusgate_http_request_duration_seconds_bucket{job="nexusgate-api"}[2m])))'
    ),
}


def query(prometheus: str, expr: str) -> list[float]:
    """Every value for an instant query, one per series (so per pod where it is per-pod)."""
    resp = requests.get(
        f"{prometheus.rstrip('/')}/api/v1/query", params={"query": expr}, timeout=15
    )
    resp.raise_for_status()
    values = []
    for item in resp.json()["data"]["result"]:
        try:
            value = float(item["value"][1])
        except (TypeError, ValueError):
            continue
        if value == value:  # drop NaN, which histogram_quantile returns with no traffic
            values.append(value)
    return values


def sample(prometheus: str) -> dict:
    row = {"t": time.time()}
    for name, expr in SERIES.items():
        try:
            row[name] = query(prometheus, expr)
        except Exception as e:
            row[name] = []
            row.setdefault("errors", []).append(f"{name}: {type(e).__name__}")
    return row


def drift(samples: list[dict], name: str) -> dict | None:
    """First-quarter against last-quarter, which is what a slope looks like in a summary."""
    series = [sum(s[name]) for s in samples if s.get(name)]
    if len(series) < 4:
        return None
    quarter = max(len(series) // 4, 1)
    early, late = series[:quarter], series[-quarter:]
    first, last = statistics.fmean(early), statistics.fmean(late)
    return {
        "first_quarter": round(first, 3),
        "last_quarter": round(last, 3),
        "change_pct": round(100 * (last - first) / first, 1) if first else None,
        "peak": round(max(series), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://api:8000")
    parser.add_argument("--prometheus-url", default="http://prometheus:9090")
    parser.add_argument("--admin-token", default=None)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--users", type=int, default=40, help="a load the gateway handles easily")
    parser.add_argument("--sample-seconds", type=float, default=30)
    parser.add_argument("--stream-weight", type=int, default=4, help="include streamed requests")
    parser.add_argument("--rss-growth-limit-pct", type=float, default=15.0)
    parser.add_argument("--label", default="soak")
    parser.add_argument("--corpus", type=Path, default=HERE.parent / "eval" / "corpus")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    import os

    args.admin_token = args.admin_token or os.environ.get("NEXUSGATE_ADMIN_TOKEN")
    if not args.admin_token:
        parser.error("--admin-token (or NEXUSGATE_ADMIN_TOKEN) is required")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    key, key_id = setup(args.base_url, args.admin_token, args.corpus, args.label)

    locust = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(HERE / "locustfile.py"),
            "--headless",
            "--only-summary",
            "-u",
            str(args.users),
            "-r",
            str(args.users),
            "-t",
            f"{int(args.minutes * 60) + 30}s",
            "--host",
            args.base_url,
            "--stop-timeout",
            "10",
        ],
        env={
            **os.environ,
            "NEXUSGATE_LOADTEST_KEY": key,
            "NEXUSGATE_LOADTEST_STREAM_WEIGHT": str(args.stream_weight),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(
        f"soaking {args.minutes:.0f} min at {args.users} users "
        f"(stream weight {args.stream_weight}), sampling every {args.sample_seconds:.0f}s",
        flush=True,
    )

    samples: list[dict] = []
    deadline = time.monotonic() + args.minutes * 60
    try:
        # One sample before the load ramps, as the baseline the rest is compared against.
        time.sleep(min(args.sample_seconds, 20))
        while time.monotonic() < deadline:
            row = sample(args.prometheus_url)
            samples.append(row)
            rss = sum(row.get("api_rss_bytes") or [0]) / 1e6
            p95 = (row.get("latency_p95") or [0])[0] * 1000
            elapsed = (len(samples) * args.sample_seconds) / 60
            print(
                f"  {elapsed:>5.1f} min  rss {rss:>7.1f} MB  p95 {p95:>6.0f} ms  "
                f"errors {sum(row.get('errors_total') or [0]):>3.0f}  "
                f"degraded {sum(row.get('degraded_total') or [0]):>4.0f}",
                flush=True,
            )
            time.sleep(args.sample_seconds)
    finally:
        locust.terminate()
        try:
            locust.wait(timeout=30)
        except subprocess.TimeoutExpired:
            locust.kill()
        Api(args.base_url, {"X-Admin-Token": args.admin_token}).session.delete(
            f"{args.base_url}/v1/admin/keys/{key_id}", timeout=30
        )

    rss = drift(samples, "api_rss_bytes")
    latency = drift(samples, "latency_p95")
    fds = drift(samples, "api_open_fds")
    totals = {
        name: (sum(samples[-1][name]) if samples[-1].get(name) else 0)
        for name in ("requests_total", "errors_total", "degraded_total", "shed_total")
    }
    verdict = []
    if rss and rss["change_pct"] is not None and rss["change_pct"] > args.rss_growth_limit_pct:
        verdict.append(
            f"resident memory grew {rss['change_pct']}% (limit {args.rss_growth_limit_pct}%)"
        )
    if totals["errors_total"]:
        verdict.append(f"{totals['errors_total']:.0f} server errors")
    if fds and fds["change_pct"] is not None and fds["change_pct"] > 25:
        verdict.append(f"open file descriptors grew {fds['change_pct']}%")

    result = {
        "label": args.label,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "minutes": args.minutes,
        "users": args.users,
        "stream_weight": args.stream_weight,
        "samples": len(samples),
        "api_rss_bytes": rss,
        "latency_p95_seconds": latency,
        "api_open_fds": fds,
        "totals": totals,
        "passed": not verdict,
        "problems": verdict,
        "series": samples,
    }
    (args.results_dir / f"{args.label}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + ("PASS" if result["passed"] else "FAIL: " + "; ".join(verdict)))
    if rss:
        print(
            f"resident memory {rss['first_quarter'] / 1e6:.0f} MB -> "
            f"{rss['last_quarter'] / 1e6:.0f} MB ({rss['change_pct']:+.1f}%), "
            f"peak {rss['peak'] / 1e6:.0f} MB"
        )
    if latency:
        print(
            f"p95 {latency['first_quarter'] * 1000:.0f} ms -> "
            f"{latency['last_quarter'] * 1000:.0f} ms ({latency['change_pct']:+.1f}%)"
        )
    print("RESULT_JSON:" + json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
