"""How fast is the self-hosted model, through the gateway, on this hardware?

    python loadtest/self_hosted.py --label laptop

Streams completions from the `local` route (Qwen2.5-1.5B-Instruct, 4-bit, llama.cpp on CPU) at
1, 2 and 4 concurrent requests, and once with a RAG-sized prompt, and reports per request:

- time to first token, which for a short prompt is mostly queueing and for a long one is mostly
  prompt processing;
- decode speed, tokens per second after the first, which is what a reader watches;
- aggregate tokens per second across all streams, which is what capacity planning needs.

It runs *inside* an API pod (kubectl exec), so the numbers include the gateway but not Podman's
port forwarder, which adds ~2 s to some requests on Windows (README "From inside the cluster").
Results go to results/self-hosted-<label>.json.
"""

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
CONTEXT, NAMESPACE = "kind-nexusgate", "nexusgate"

# Runs in the API pod: httpx is installed there, and the admin token is in its environment.
IN_POD = r"""
import asyncio, json, os, statistics, time, uuid
import httpx

BASE = "http://localhost:8000"
QUESTIONS = [
    "Explain what a circuit breaker does in a distributed system.",
    "What is the difference between a process and a thread?",
    "Why do databases use write-ahead logs?",
    "What does a load balancer do?",
    "Explain eventual consistency in two sentences.",
    "What is backpressure in a streaming system?",
    "Why cache the results of an expensive query?",
    "What is a dead-letter queue for?",
]
# A RAG-sized prompt: about as many tokens as an answer request carries with its passages.
PASSAGE = ("The billing worker drains its queue before restarting, then waits ninety seconds so "
           "in-flight invoices settle. It scales between two and eight replicas on queue depth. ")
LONG = "Passages:\n" + "\n".join(f"[{i}] " + PASSAGE * 4 for i in range(1, 9)) + (
    "\n\nUsing only the passages, how long does the billing worker wait before restarting?")


async def one(client, key, prompt, max_tokens):
    started = time.perf_counter()
    first = None
    usage = {}
    body = {"model": "local", "stream": True, "cache": False, "temperature": 0,
            "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    async with client.stream("POST", "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if first is None and chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                first = time.perf_counter()
            usage = chunk.get("usage") or usage
    total = time.perf_counter() - started
    tokens = usage.get("completion_tokens", 0)
    ttft = (first or started) - started
    decode = (tokens - 1) / (total - ttft) if tokens > 1 and total > ttft else 0.0
    return {"ttft_s": ttft, "total_s": total, "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": tokens, "decode_tps": decode}


def unique(prompt):
    # llama.cpp reuses the cached prefix of a prompt it has seen. A tag at the very start means no
    # two requests share more than the chat template, so prompt processing is measured, not skipped.
    return f"Request {uuid.uuid4().hex[:8]}. {prompt}"


async def level(client, key, name, concurrency, prompts, max_tokens):
    prompts = [unique(p) for p in prompts]
    await one(client, key, unique(prompts[0]), 8)  # warm-up, discarded
    started, started_at = time.perf_counter(), time.time()
    sem = asyncio.Semaphore(concurrency)
    async def gated(p):
        async with sem:
            return await one(client, key, p, max_tokens)
    runs = await asyncio.gather(*(gated(p) for p in prompts))
    wall = time.perf_counter() - started
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, round(p * (len(xs) - 1)))]
    ttfts, decodes = [r["ttft_s"] for r in runs], [r["decode_tps"] for r in runs]
    # Wall-clock bounds, so the host's CPU samples can be matched to the level that ran.
    return {"level": name, "concurrency": concurrency, "requests": len(runs), "wall_s": wall,
            "started_at": started_at, "ended_at": time.time(),
            "ttft_p50_s": statistics.median(ttfts), "ttft_p95_s": q(ttfts, 0.95),
            "decode_tps_p50": statistics.median(decodes), "decode_tps_min": min(decodes),
            "aggregate_tps": sum(r["completion_tokens"] for r in runs) / wall,
            "prompt_tokens_p50": statistics.median(r["prompt_tokens"] for r in runs),
            "runs": runs}


async def main():
    admin = {"X-Admin-Token": os.environ["NEXUSGATE_ADMIN_TOKEN"]}
    async with httpx.AsyncClient(base_url=BASE, timeout=600) as client:
        r = await client.post("/v1/admin/keys", headers=admin,
                              json={"tenant_id": f"bench-{uuid.uuid4().hex[:8]}", "name": "bench"})
        r.raise_for_status()
        key, key_id = r.json()["api_key"], r.json()["record"]["key_id"]
        try:
            # Twice through: one pass is one sample, and two that disagree say the conditions moved.
            passes = []
            for _ in range(2):
                passes.append([
                    await level(client, key, "short x1", 1, QUESTIONS, 128),
                    await level(client, key, "short x2", 2, QUESTIONS, 128),
                    await level(client, key, "short x4", 4, QUESTIONS, 128),
                    await level(client, key, "rag-sized x1", 1, [LONG] * 4, 64),
                ])
        finally:
            await client.delete(f"/v1/admin/keys/{key_id}", headers=admin)
    print(json.dumps(passes))

asyncio.run(main())
"""


def kubectl(*args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["kubectl", "--context", CONTEXT, "-n", NAMESPACE, *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"kubectl {args[0]} failed:\n{result.stderr.strip()}")
    return result.stdout


def power_plan() -> str | None:
    """The laptop's power plan moved latency by double digits in the soak test, so record it."""
    if platform.system() != "Windows":
        return None
    out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout
    return out.split("(")[-1].rstrip(")\n ") if "(" in out else out.strip() or None


def namespace_cpu_m() -> int:
    """Millicores used across the namespace right now (metrics-server's view)."""
    total = 0
    for line in kubectl("top", "pod", "--no-headers").splitlines():
        cpu = line.split()[1]
        total += int(cpu[:-1]) if cpu.endswith("m") else int(float(cpu) * 1000)
    return total


def _cpu_times() -> tuple[int, int] | None:
    """(idle, total) CPU time for the whole host, or None where this can't be read."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
        ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(t) for t in (idle, kernel, user)))
        ticks = [(t.dwHighDateTime << 32) | t.dwLowDateTime for t in (idle, kernel, user)]
        return ticks[0], ticks[1] + ticks[2]  # kernel time includes idle time
    if Path("/proc/stat").exists():
        fields = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
        return fields[3] + fields[4], sum(fields)
    return None


class HostCPU:
    """Samples how busy the whole host is, once a second, in a background thread.

    The cluster being quiet is not enough. On this laptop the kind node shares one WSL2 VM with
    every other WSL distro, and that VM shares the CPU with Windows. One run measured single-stream
    decode at 2.9 tokens/s with the host 98% busy, rising to 25 as the host's load fell to 24%,
    while the namespace stayed idle throughout. So the host's load is recorded next to every
    number, and a level that ran on a busy host is marked as such."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        last = _cpu_times()
        while last and not self._stop.wait(1.0):
            now = _cpu_times()
            idle, total = now[0] - last[0], now[1] - last[1]
            if total > 0:
                self.samples.append((time.time(), 100.0 * (1 - idle / total)))
            last = now

    def __enter__(self) -> "HostCPU":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def mean(self, start: float, end: float) -> float | None:
        window = [busy for t, busy in self.samples if start <= t <= end]
        return round(sum(window) / len(window), 1) if window else None


def settle(host: HostCPU, quiet_m: int = 300, quiet_host: float = 15.0, timeout_s: float = 900):
    """Wait until both the namespace and the host are quiet, sampled twice 15 s apart: the first
    run of this benchmark began straight after a rollout, with four API pods loading their models.
    Gives up after `timeout_s` and records what it saw rather than refusing to run."""
    deadline, quiet = time.monotonic() + timeout_s, 0
    while True:
        started = time.time()
        time.sleep(15)
        used, busy = namespace_cpu_m(), host.mean(started, time.time())
        calm = used < quiet_m and (busy is None or busy < quiet_host)
        quiet = quiet + 1 if calm else 0
        if quiet == 2 or time.monotonic() > deadline:
            return used, busy


# The model itself keeps 4 of the host's cores busy while it serves, about 17% of this laptop;
# well beyond that, something else was running.
BUSY_HOST = 35.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    llm = json.loads(kubectl("get", "deployment", "llm", "-o", "json"))
    container = llm["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in container.get("env", [])}
    with HostCPU() as host:
        print("Waiting for the namespace and the host to go quiet")
        idle_m, idle_host = settle(host)
        api_pods = json.loads(kubectl("get", "deployment", "api", "-o", "json"))["status"]
        print(
            f"  namespace {idle_m}m CPU, host {idle_host}% busy, "
            f"{api_pods.get('readyReplicas')} API pods"
        )
        print(
            "Streaming from the `local` route inside an API pod, twice through; about ten minutes"
        )
        passes = json.loads(kubectl("exec", "-i", "deploy/api", "--", "python", "-", stdin=IN_POD))
    for levels in passes:
        for lv in levels:
            lv["host_busy_pct"] = host.mean(lv["started_at"], lv["ended_at"])

    record = {
        "label": args.label,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_image": container["image"],
        "threads": env.get("LLAMA_ARG_THREADS"),
        "parallel_slots": env.get("LLAMA_ARG_N_PARALLEL"),
        "cpu_limit": container["resources"]["limits"].get("cpu"),
        "power_plan": power_plan(),
        "namespace_cpu_before_m": idle_m,
        "host_busy_before_pct": idle_host,
        "api_pods": api_pods.get("readyReplicas"),
        "passes": passes,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"self-hosted-{args.label}.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(
        f"\n{'pass':4} {'level':14} {'req':>4} {'ttft p50':>9} {'ttft p95':>9} {'decode p50':>11} "
        f"{'decode min':>11} {'aggregate':>10} {'host':>6}"
    )
    busy_levels = 0
    for n, levels in enumerate(passes, 1):
        for lv in levels:
            busy = lv["host_busy_pct"]
            flag = "*" if busy is not None and busy > BUSY_HOST else " "
            busy_levels += flag == "*"
            print(
                f"{n:<4} {lv['level']:14} {lv['requests']:>4} "
                f"{lv['ttft_p50_s']:>8.2f}s {lv['ttft_p95_s']:>8.2f}s "
                f"{lv['decode_tps_p50']:>7.1f} t/s {lv['decode_tps_min']:>7.1f} t/s "
                f"{lv['aggregate_tps']:>6.1f} t/s {busy if busy is not None else '-':>5}%{flag}"
            )
    if busy_levels:
        print(
            f"\n* {busy_levels} level(s) ran with the host over {BUSY_HOST:.0f}% busy: "
            "those numbers "
            "describe what else the laptop was doing, not the model"
        )
    print(
        f"\n{record['threads']} threads, {record['parallel_slots']} slots, "
        f"cpu limit {record['cpu_limit']}, power plan {record['power_plan']} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
