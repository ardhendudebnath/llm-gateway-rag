# The self-hosted model, measured

How fast is Qwen2.5-1.5B-Instruct (4-bit, Q4_K_M) on llama.cpp, on CPU, behind the gateway? It
serves the `local` route and is the last fallback of `default`
([`Containerfile.llm`](../Containerfile.llm), [`infra/k8s/base/llm.yaml`](../infra/k8s/base/llm.yaml)).
Raw data: [`results/self-hosted-laptop.json`](results/self-hosted-laptop.json).

```bash
python loadtest/self_hosted.py --label laptop      # kind stack up; ~15 minutes
```

## Method

**Streams through the gateway, from inside the cluster.** [`self_hosted.py`](self_hosted.py) runs its
requests in an API pod (`kubectl exec`), so they cross the gateway's auth, routing, metering and
SSE path but not Podman's port forwarder (see [README](README.md#method)). Every request streams
from the `local` route with `temperature: 0`, and is reported as:

| Metric | What it is |
|---|---|
| time to first token | queueing plus prompt processing |
| decode | tokens per second after the first, per stream: what a reader watches |
| aggregate | completion tokens across all streams over the level's wall time: capacity |

**Levels.** 8 short questions (~40 prompt tokens, up to 128 out) one at a time, 2 at a time and 4
at a time; then 4 RAG-sized prompts (~1,040 tokens of passages, up to 64 out) one at a time. Each
level starts with a discarded warm-up request, and the whole sequence runs twice.

**Every prompt is unique.** llama.cpp reuses the cached prefix of a prompt it has seen. The first
version sent the same RAG prompt four times and measured 0.06 s to first token for 1,039 tokens: a
cache hit, not prompt processing. A tag at the start of each prompt now means requests share only
the chat template.

**The host is measured, not just the cluster.** This is the lesson that cost the most runs:

- The first run started straight after a rollout. The autoscaler had four API pods loading their
  models, and single-stream decode measured 0.9-3.7 tokens/s. The benchmark now waits for the
  namespace to go quiet.
- The second run started with the namespace at 176 millicores, and was *still* erratic: 1.4
  tokens/s in one pass, 23.5 in the other, and 72 s to first token at 4 concurrent. CPU throttling
  of the pod was ruled out: its cgroup had been throttled for 8.4 s in total across both runs.
  Then 15 sequential requests were sampled against Windows' counters. The CPU's clock stayed at
  158-195% of base throughout, so it was not slowing down; what moved with decode speed was the
  load on **the whole laptop**: decode rose from 2.9 to 25 tokens/s as the host's load fell from
  98% to 24%. The kind node runs in WSL2, whose one VM every WSL distro shares, and that VM
  shares the CPU with Windows.

So `self_hosted.py` samples the host's CPU once a second throughout, waits for the host as well as
the namespace to be quiet before starting, stores the host's load beside every level, and marks
any level that ran with the host over 35% busy (the model's own 4 of 24 cores are ~17%). None of
the results below is marked.

## Results

4 threads, 2 parallel slots, CPU limit 4, on a 24-core Core Ultra 9 275HX laptop (power plan
"Silent"). Host 6.8% busy before the run, 24-33% during it.

| Level | Pass | First token p50 | p95 | Decode p50 | Decode min | Aggregate |
|---|---|---|---|---|---|---|
| short, 1 at a time | 1 | 0.17 s | 0.19 s | 27.6 t/s | 25.7 t/s | 26.2 t/s |
| | 2 | 0.18 s | 0.21 s | 25.0 t/s | 23.7 t/s | 24.3 t/s |
| short, 2 at a time | 1 | 0.22 s | 0.40 s | 21.5 t/s | 20.4 t/s | 39.5 t/s |
| | 2 | 0.25 s | 0.38 s | 18.8 t/s | 16.6 t/s | 34.3 t/s |
| short, 4 at a time | 1 | 5.86 s | 6.78 s | 20.8 t/s | 18.8 t/s | 37.6 t/s |
| | 2 | 5.85 s | 7.65 s | 18.8 t/s | 16.1 t/s | 33.9 t/s |
| RAG-sized, 1 at a time | 1 | 7.27 s | 7.48 s | 24.6 t/s | 23.8 t/s | — |
| | 2 | 8.61 s | 10.21 s | 18.6 t/s | 16.5 t/s | — |

The passes agree within ~10-15%, and the second was the slower one each time, with the host a few
points busier (29-33% against 24-27%). The RAG-sized level has no aggregate figure: four
sequential requests that are mostly prompt processing say nothing about capacity.

## Findings

**1. On a CPU, reading the prompt is what makes RAG slow, not writing the answer.** Prompt
processing runs at ~120-150 tokens/s, so a RAG prompt of ~1,040 tokens costs 7-9 s before the
first word, while the answer streams at ~20-25 tokens/s. A short question gets its first token in
under 0.2 s. The lever for RAG latency on this hardware is prompt size: fewer or shorter passages
for the self-hosted route, not a faster decoder.

**2. One pod serves ~35-40 tokens/s, and reaches it at 2 concurrent requests.** Two slots share the
4 threads, so each stream slows from ~26 to ~20 tokens/s while the total rises ~1.5x. A third and
fourth request don't add throughput: they wait for a slot, 5.9 s to first token at 4 concurrent.
Capacity is added with replicas (or cores), not with more slots on the same 4 threads.

**3. No cold-start penalty on a restart.** Restarted alone on a quiet host, the pod was Ready in
7 s, and its first request took 2.59 s against 2.43 and 2.49 s for the next two, at the same
27-28 tokens/s. The only extra work was the chat template's built-in system prompt, which llama.cpp
then caches: 46 prompt tokens on the first request, 21 after. The 19.4 s first request seen right
after the rollout was the contention described above. One caveat: the model is memory-mapped,
and the new pod ran on the same node, whose page cache may still have held the file. A first
start on a fresh node was not measured on its own.

**4. A 1.5B model answers correctly but does not cite.** Asked in two smoke runs how long to wait
before restarting the billing worker, it said "90 seconds" both times, which is right and comes
from the passage, but without the `[1]` the prompt asks for. The smoke test reports inline citations rather than requiring them;
how often the model cites, and whether its answers stay within the passages, is what a
groundedness eval should measure next.

## Limitations

- **One laptop, one power plan.** These numbers were taken under the "Silent" plan; a performance
  plan or a server CPU would likely do better, and a GPU far better. The method carries over; the
  figures don't.
- **Short runs.** 8 requests per level, twice. Enough to see the shape and that the passes agree,
  not to quote a p99.
- **Not in the Helm chart yet.** The model runs in the kind stack only; on a Helm install the
  `local` route has nothing behind it.
