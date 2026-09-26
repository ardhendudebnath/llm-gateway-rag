"""End-to-end smoke test against a running NexusGate stack (the kind cluster, or any deployment).

    python scripts/smoke_test.py --admin-token <token> [--prometheus-url http://localhost:9090]

Goes through the real network path: key issuance, a cache miss then a hit, fallback on the `chaos`
route, usage metering, /metrics, RAG (upload queued to a worker, search, a cited answer), an agent
run through its state graph, and (optionally) Prometheus scraping every API replica. It only uses
the offline mock routes, so it needs no provider API keys and spends nothing. It cleans up the key,
cache entries and document it creates, and exits non-zero on the first failed check.
"""

import argparse
import json
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


def check_gateway(api: httpx.Client, admin_token: str, job_timeout: float = 60) -> None:
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

        check_streaming(api, key)
        check_canary(api, admin, key)
        check_rag(api, auth, job_timeout)
    finally:
        api.delete("/v1/cache", headers=auth)
        api.delete(f"/v1/admin/keys/{key_id}", headers=admin)


def check_streaming(api: httpx.Client, key: str) -> None:
    """Stream a completion end to end, and read the footer the last chunk carries."""
    chunks = []
    with api.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "mock",
            "stream": True,
            "cache": False,
            "messages": [{"role": "user", "content": "stream me a sentence"}],
        },
    ) as resp:
        check(resp.status_code == 200, f"stream: {resp.status_code}")
        check(
            resp.headers["content-type"].startswith("text/event-stream"),
            f"stream content-type: {resp.headers.get('content-type')}",
        )
        done = False
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                done = True
                break
            chunks.append(json.loads(payload))
    check(done, "stream ended without [DONE]")
    check(len(chunks) > 2, f"expected several chunks, got {len(chunks)}")
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    check("stream me a sentence" in text, f"streamed text looks wrong: {text!r}")
    meta = chunks[-1].get("nexusgate") or {}
    check(meta.get("ttft_ms", 0) > 0, f"no time-to-first-token in the footer: {meta}")
    check(chunks[-1].get("usage", {}).get("total_tokens", 0) > 0, "no usage on the last chunk")
    ok(f"streaming: {len(chunks)} chunks, ttft {meta['ttft_ms']:.0f} ms, usage reported")


# A candidate route table: one deployment, a mock priced differently from the live `mock` route.
CANARY_ROUTES = """
routes:
  mock:
    - name: canary-mock
      provider: mock
      model: mock/canary
      options: { latency_ms: 20 }
      pricing: { input_per_mtok: 0.10, output_per_mtok: 0.40 }
"""


def check_canary(api: httpx.Client, admin: dict, key: str) -> None:
    """Roll a route change out to all traffic, confirm it serves, then withdraw it."""
    r = api.put(
        "/v1/admin/routes/canary",
        headers=admin,
        json={"config": CANARY_ROUTES, "weight": 100, "note": "smoke test"},
    )
    check(r.status_code == 200, f"publish canary: {r.status_code} {r.text}")
    version = r.json()["canary"]["version"]
    try:
        served = chat(api, key, "mock", "which table served this?", cache=False)
        check(served.status_code == 200, f"canary chat: {served.status_code} {served.text}")
        meta = served.json()["nexusgate"]
        check(meta["route_variant"] == "canary", f"served by {meta['route_variant']}")
        check(meta["deployment"] == "canary-mock", f"served by {meta['deployment']}")
        view = api.get("/v1/admin/routes", headers=admin).json()
        check(view["canary"]["stats"]["requests"] >= 1, f"canary stats: {view['canary']['stats']}")
        ok(f"canary {version[:8]} took 100% of traffic and reported its own stats")
    finally:
        api.delete("/v1/admin/routes/canary", headers=admin)
    back = chat(api, key, "mock", "and after withdrawing?", cache=False)
    check(back.json()["nexusgate"]["route_variant"] == "stable", "traffic did not return to stable")
    ok("canary withdrawn; traffic back on the stable route table")


RUNBOOK = b"""# Billing worker runbook

## Restarting the worker
Drain the queue first, then wait 90 seconds before restarting the billing worker.

## Scaling
The billing worker scales between two and eight replicas based on queue depth.
"""


def wait_for_job(api: httpx.Client, auth: dict, job_id: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        job = api.get(f"/v1/rag/jobs/{job_id}", headers=auth).json()
        if job["status"] in {"done", "failed"}:
            return job
        if time.monotonic() > deadline:
            raise CheckFailedError(f"ingestion job stuck in '{job['status']}' after {timeout_s}s")
        time.sleep(1)


def check_rag(api: httpx.Client, auth: dict, job_timeout: float) -> None:
    r = api.post(
        "/v1/rag/documents",
        headers=auth,
        files={"file": ("runbook.md", RUNBOOK, "text/markdown")},
    )
    check(r.status_code == 202, f"RAG upload: expected 202, got {r.status_code} {r.text}")
    queued = r.json()
    check(queued["doc_id"] is None, "a queued job should not have a document yet")
    ok(f"RAG: upload accepted as job {queued['job_id'][:8]} ({queued['status']})")

    job = wait_for_job(api, auth, queued["job_id"], job_timeout)
    check(job["status"] == "done", f"ingestion job failed: {job.get('error')}")
    doc = {"doc_id": job["doc_id"]}
    try:
        ok(f"RAG: worker ingested it as {job['chunks']} chunks in {job['attempts']} attempt(s)")

        r = api.post(
            "/v1/rag/search",
            headers=auth,
            json={"query": "how long to wait before bouncing the billing worker", "top_k": 2},
        )
        check(r.status_code == 200, f"RAG search: {r.status_code} {r.text}")
        body = r.json()
        check(body["hits"] and "90 seconds" in body["hits"][0]["text"], f"search: {body}")
        ok(f"RAG search: right passage ranked first (reranked={body['reranked']})")

        r = api.post(
            "/v1/rag/answer",
            headers=auth,
            json={"question": "How do I restart the billing worker?", "model": "mock"},
        )
        check(r.status_code == 200, f"RAG answer: {r.status_code} {r.text}")
        answer = r.json()
        check(answer["citations"] and answer["nexusgate"], f"answer: {answer}")
        ok(f"RAG answer via the mock route with {len(answer['citations'])} numbered passages")

        r = api.post(
            "/v1/agents/research",
            headers=auth,
            json={
                "question": "How do I restart the billing worker?",
                "model": "mock",
                "max_revisions": 1,
            },
        )
        check(r.status_code == 200, f"agent run: {r.status_code} {r.text}")
        run = r.json()
        path = [s["node"] for s in run["steps"]]
        check(path[:3] == ["plan", "retrieve", "draft"], f"agent path: {path}")
        check(run["citations"] and run["llm_calls"] >= 3, f"agent run: {run}")
        steps = " -> ".join(path)
        ok(f"agent: {steps} in {run['llm_calls']} calls, {run['revisions']} revision(s)")
    finally:
        api.delete(f"/v1/rag/documents/{doc['doc_id']}", headers=auth)


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
    parser.add_argument(
        "--job-timeout", type=float, default=60, help="seconds to wait for a worker to ingest"
    )
    args = parser.parse_args()
    if not args.admin_token:
        parser.error("--admin-token (or NEXUSGATE_ADMIN_TOKEN) is required")

    print(f"Smoke test against {args.base_url}")
    try:
        with httpx.Client(base_url=args.base_url, timeout=30) as api:
            wait_ready(api, args.wait)
            ok("/readyz: API is ready and Redis reachable")
            check_gateway(api, args.admin_token, args.job_timeout)
        if args.prometheus_url:
            check_prometheus(args.prometheus_url, args.api_replicas, args.wait)
    except (CheckFailedError, httpx.HTTPError) as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
