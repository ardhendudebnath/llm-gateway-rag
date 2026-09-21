"""A narrated walkthrough of NexusGate in one command, paced for a screen recording.

    python scripts/demo.py --base-url http://localhost:7860 --key ng_...   # the demo container
    python scripts/demo.py --admin-token <token>                           # local cluster

With an admin token it mints its own key (and revokes it afterwards); with --key it uses the one
shown on the public demo's landing page. It only uses the offline `mock` and `chaos` routes, so it
costs nothing. Open the Grafana dashboard next to it to watch the panels move.
"""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

BOLD, DIM, GREEN, CYAN, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[32m",
    "\033[36m",
    "\033[33m",
    "\033[0m",
)


def step(n: int, title: str, pace: float) -> None:
    time.sleep(pace)
    print(f"\n{BOLD}{CYAN}{n}. {title}{RESET}")


def show(label: str, value: object, colour: str = "") -> None:
    print(f"   {DIM}{label:<18}{RESET}{colour}{value}{RESET}")


def chat(api: httpx.Client, auth: dict, model: str, content: str, cache: bool = True) -> dict:
    resp = api.post(
        "/v1/chat/completions",
        headers=auth,
        json={"model": model, "cache": cache, "messages": [{"role": "user", "content": content}]},
    )
    resp.raise_for_status()
    return resp.json()


def upload_document(api: httpx.Client, auth: dict) -> None:
    doc = ROOT / "eval" / "corpus" / "incident-response.md"
    resp = api.post(
        "/v1/rag/documents",
        headers=auth,
        files={"file": (doc.name, doc.read_bytes(), "text/markdown")},
    )
    if resp.status_code == 403:  # the public demo: uploads are off, the handbook is pre-loaded
        show("upload", "disabled in the public demo; using the pre-loaded handbook", YELLOW)
        return
    resp.raise_for_status()
    job = resp.json()
    show("HTTP", f"{resp.status_code} (accepted, not yet processed)")
    show("job", f"{job['job_id'][:8]} {job['status']}")
    deadline = time.monotonic() + 60
    while job["status"] not in {"done", "failed"} and time.monotonic() < deadline:
        time.sleep(0.5)
        job = api.get(f"/v1/rag/jobs/{job['job_id']}", headers=auth).json()
    colour = GREEN if job["status"] == "done" else YELLOW
    show("job", f"{job['job_id'][:8]} {job['status']}, {job['chunks']} chunks", colour)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--key", help="an existing API key (e.g. the public demo's)")
    parser.add_argument("--admin-token", default=os.environ.get("NEXUSGATE_ADMIN_TOKEN"))
    parser.add_argument("--pace", type=float, default=1.5, help="seconds between steps")
    args = parser.parse_args()
    if not (args.key or args.admin_token):
        parser.error("pass --key, or --admin-token to mint one")
    if os.name == "nt":
        os.system("")  # enable ANSI colours in the Windows console

    api = httpx.Client(base_url=args.base_url, timeout=30)
    minted = None
    if args.key:
        key = args.key
    else:
        created = api.post(
            "/v1/admin/keys",
            headers={"X-Admin-Token": args.admin_token},
            json={"tenant_id": f"demo-{uuid.uuid4().hex[:6]}", "name": "walkthrough"},
        ).json()
        key, minted = created["api_key"], created["record"]["key_id"]
    auth = {"Authorization": f"Bearer {key}"}
    question = f"What does a circuit breaker do? (demo {uuid.uuid4().hex[:4]})"

    try:
        print(f"{BOLD}NexusGate{RESET} {DIM}at {args.base_url}{RESET}")
        step(1, "Ready?", 0)
        show("readiness", api.get("/readyz").json()["checks"], GREEN)

        step(2, "A chat request goes to a provider", args.pace)
        first = chat(api, auth, "mock", question)
        meta = first["nexusgate"]
        show("answered by", meta["deployment"])
        show("latency", f"{meta['latency_ms']:.0f} ms")
        show("cost", f"${meta['cost_usd']:.6f}")

        step(3, "Asking again is a semantic-cache hit", args.pace)
        again = chat(api, auth, "mock", question)["nexusgate"]
        show("cached", again["cached"], GREEN)
        show("similarity", again["cache_similarity"])
        show("latency", f"{again['latency_ms']:.0f} ms", GREEN)
        show("cost", f"${again['cost_usd']:.6f}", GREEN)

        step(4, "The primary provider fails; fallback answers", args.pace)
        chaos = chat(api, auth, "chaos", "ping", cache=False)["nexusgate"]
        for attempt in chaos["attempts"]:
            colour = GREEN if attempt["outcome"] == "success" else YELLOW
            show(attempt["deployment"], attempt["outcome"], colour)
        show("answered by", chaos["deployment"], GREEN)

        step(5, "Upload a document: queued, then ingested by a worker", args.pace)
        upload_document(api, auth)

        step(6, "Ask it: retrieval, rerank, and a cited answer", args.pace)
        answer = api.post(
            "/v1/rag/answer",
            headers=auth,
            json={"question": "When is a postmortem due?", "model": "mock", "cache": False},
        ).json()
        for c in answer["citations"][:3]:
            lines = c["text"].split("\n")
            body = lines[1] if len(lines) > 1 else lines[0]  # line 0 is "title > heading"
            where = (c["heading"] or c["title"]).split(" > ")[-1]
            show(f"[{c['n']}] {where[:13]}", body if len(body) <= 62 else f"{body[:59]}...")
        if not answer["citations"]:
            show("answer", answer["answer"], YELLOW)

        step(7, "Usage and spend, including what the cache saved", args.pace)
        totals = api.get("/v1/usage", params={"days": 1}, headers=auth).json()["totals"]
        show("requests", totals["requests"])
        show("cache hits", totals["cache_hits"], GREEN)
        show("spent", f"${totals['cost_usd']:.6f}")
        show("saved by cache", f"${totals['cost_saved_usd']:.6f}", GREEN)
        print(f"\n{DIM}API docs: {args.base_url}/docs · metrics: {args.base_url}/metrics{RESET}")
    finally:
        if minted:
            api.delete(f"/v1/admin/keys/{minted}", headers={"X-Admin-Token": args.admin_token})
        api.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
