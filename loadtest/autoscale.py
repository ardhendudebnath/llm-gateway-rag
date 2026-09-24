"""Autoscaling test: bury the ingestion workers in jobs, and watch the HPA react.

    python loadtest/autoscale.py --admin-token <token>          # 60 uploads, then watch
    python loadtest/autoscale.py --admin-token <token> --documents 120 --concurrency 12

Unlike the load and chaos tests, this one runs on the host: it has to read Kubernetes objects
(`kubectl get hpa`, the worker's replica count), not just HTTP. The uploads themselves are cheap
to send, so the Windows port forward does not distort what is being measured here — the numbers
that matter are queue depth and replicas over time, both read from inside the cluster.

What it shows: queue depth crossing the HPA's target (5 waiting jobs per worker) makes the worker
Deployment scale out, the backlog drains faster than one worker could manage, and the replica
count then stays put for the scale-down stabilisation window rather than flapping.

Writes loadtest/results/autoscale.json and prints a timeline.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "eval" / "corpus"
RESULTS = HERE / "results"


def kubectl(*args: str, context: str, namespace: str) -> str:
    result = subprocess.run(
        ["kubectl", "--context", context, "-n", namespace, *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def worker_replicas(context: str, namespace: str) -> tuple[int, int]:
    """(desired, ready) for the worker Deployment."""
    spec = json.loads(
        kubectl("get", "deploy/worker", "-o", "json", context=context, namespace=namespace)
    )
    return spec["spec"]["replicas"], spec.get("status", {}).get("readyReplicas", 0)


def hpa_metrics(context: str, namespace: str) -> tuple[float | None, float | None]:
    """What the worker HPA currently reads: (queue depth per pod, CPU % of request).

    Both, because the worker scales on either. Recording only the queue would make a scale-out
    that CPU actually drove look like proof that queue-depth scaling works.
    """
    hpa = json.loads(
        kubectl("get", "hpa/worker", "-o", "json", context=context, namespace=namespace)
    )
    queue = cpu = None
    for metric in hpa.get("status", {}).get("currentMetrics") or []:
        if metric.get("type") == "External":
            value = metric["external"]["current"].get("averageValue")
            if value is not None:
                queue = float(str(value).rstrip("m")) / (1000 if str(value).endswith("m") else 1)
        elif metric.get("type") == "Resource":
            cpu = metric["resource"]["current"].get("averageUtilization")
    return queue, cpu


def queue_depth(prometheus_url: str) -> float:
    query = 'max(nexusgate_rag_queue_depth{queue="ingest"})'
    with httpx.Client(base_url=prometheus_url, timeout=10) as prom:
        result = prom.get("/api/v1/query", params={"query": query}).json()["data"]["result"]
    return float(result[0]["value"][1]) if result else 0.0


def upload_burst(
    api: httpx.Client, auth: dict, documents: int, concurrency: int, repeat: int
) -> int:
    """Queue `documents` ingestion jobs as fast as the API will take them.

    `repeat` pads each document, because the handbook files are ~2 KB and a warm worker ingests
    one in about 100 ms: with documents that small the workers keep up with the uploads and no
    queue ever forms to scale on. Padding makes each job a second or two of real work.
    """
    corpus = sorted(CORPUS.glob("*.md"))

    def one(n: int) -> bool:
        source = corpus[n % len(corpus)]
        body = source.read_bytes()
        # A unique line per upload: document ids are content hashes, so identical bytes would
        # collapse into one document and a much shorter queue than asked for.
        data = b"\n\n".join([body] * repeat) + f"\n\nBurst upload {n}.\n".encode()
        resp = api.post(
            "/v1/rag/documents",
            headers=auth,
            files={"file": (f"burst-{n}-{source.name}", data, "text/markdown")},
        )
        return resp.status_code == 202

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return sum(pool.map(one, range(documents)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--admin-token", default=os.environ.get("NEXUSGATE_ADMIN_TOKEN"))
    parser.add_argument("--documents", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--repeat", type=int, default=12, help="repeat each document's text, to make a heavier job"
    )
    parser.add_argument("--context", default="kind-nexusgate")
    parser.add_argument("--namespace", default="nexusgate")
    parser.add_argument("--sample-seconds", type=float, default=5)
    parser.add_argument("--watch-seconds", type=float, default=420)
    parser.add_argument("--label", default="autoscale")
    args = parser.parse_args()
    if not args.admin_token:
        parser.error("--admin-token (or NEXUSGATE_ADMIN_TOKEN) is required")

    api = httpx.Client(base_url=args.base_url, timeout=60)
    created = api.post(
        "/v1/admin/keys",
        headers={"X-Admin-Token": args.admin_token},
        json={"tenant_id": f"autoscale-{int(time.time())}", "name": "autoscale"},
    ).json()
    key, key_id = created["api_key"], created["record"]["key_id"]
    auth = {"Authorization": f"Bearer {key}"}

    started = time.monotonic()
    samples: list[dict] = []

    def sample(note: str = "") -> dict:
        desired, ready = worker_replicas(args.context, args.namespace)
        per_pod, cpu = hpa_metrics(args.context, args.namespace)
        row = {
            "t": round(time.monotonic() - started, 1),
            "queue_depth": queue_depth(args.prometheus_url),
            "hpa_per_pod": per_pod,
            "hpa_cpu_percent": cpu,
            "workers_desired": desired,
            "workers_ready": ready,
            "note": note,
        }
        samples.append(row)
        shown = "-" if per_pod is None else f"{per_pod:.1f}"
        print(
            f"  t={row['t']:>6.1f}s  queue={row['queue_depth']:>5.0f}  per-pod={shown:>5}"
            f"  cpu={'-' if cpu is None else str(cpu) + '%':>5}"
            f"  workers={desired} (ready {ready})  {note}"
        )
        return row

    try:
        print(f"Autoscaling test against {args.base_url}")
        sample("before")
        print(f"\nQueueing {args.documents} uploads with {args.concurrency} connections...")
        accepted = upload_burst(api, auth, args.documents, args.concurrency, args.repeat)
        print(f"  {accepted}/{args.documents} accepted (202)\n")

        start_workers = samples[0]["workers_desired"]
        peak_queue, peak_workers, scaled_at, drained_at = 0.0, start_workers, None, None
        queue_at_scale_out = None
        deadline = time.monotonic() + args.watch_seconds
        while time.monotonic() < deadline:
            row = sample()
            peak_queue = max(peak_queue, row["queue_depth"])
            if row["workers_desired"] > peak_workers:
                peak_workers = row["workers_desired"]
                if scaled_at is None:
                    scaled_at, queue_at_scale_out = row["t"], row["queue_depth"]
            if row["queue_depth"] == 0 and drained_at is None and peak_queue > 0:
                drained_at = row["t"]
                sample("queue drained")
                break
            time.sleep(args.sample_seconds)

        summary = {
            "label": args.label,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "documents": args.documents,
            "repeat": args.repeat,
            "accepted": accepted,
            "start_workers": start_workers,
            "peak_queue_depth": peak_queue,
            "peak_workers": peak_workers,
            "seconds_to_scale_out": scaled_at,
            "queue_depth_at_scale_out": queue_at_scale_out,
            "seconds_to_drain": drained_at,
            "samples": samples,
        }
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{args.label}.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        print(f"\nPeak queue depth      {peak_queue:.0f} jobs")
        print(f"Worker replicas       {start_workers} -> {peak_workers}")
        print(f"Scaled out after      {scaled_at}s" if scaled_at else "Never scaled out")
        print(f"Backlog drained after {drained_at}s" if drained_at else "Backlog did not drain")
        print(f"Saved {path}")
        return 0
    finally:
        api.delete(f"/v1/admin/keys/{key_id}", headers={"X-Admin-Token": args.admin_token})
        api.close()


if __name__ == "__main__":
    sys.exit(main())
