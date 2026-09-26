# Load and chaos testing

How much traffic can NexusGate take before it misses its latency objective, what breaks first,
and what happens when a provider dies mid-traffic. Numbers: [`RESULTS.md`](RESULTS.md); raw data:
`results/*.json`.

```bash
python loadtest/in_cluster.py --label baseline                 # stepped load test, 5 levels x 45 s
python loadtest/in_cluster.py --script chaos.py --label chaos  # kill a provider under traffic
python loadtest/report.py --runs baseline after-fix --chaos chaos
```

## Method

**Traffic.** [`locustfile.py`](locustfile.py) mixes four request kinds, each measured separately:
cache hits (a warm pool of popular questions), provider calls (cache off, through the router to the
offline `mock` provider, 80 ms simulated latency), RAG search (embed, vector search, cross-encoder
rerank) and RAG answers (retrieval plus an LLM call). Misses are forced with `cache: false`: with a
real embedding model, "unique" templated prompts can land inside the similarity threshold and
silently become hits.

**Levels.** 25, 50, 100, 200 and 400 simulated users with 0.5–1.5 s think time, 45 s each, all
users spawned at once and `--reset-stats` so the ramp isn't counted. A dedicated API key with a
very high rate limit keeps the gateway's own limiter from being what "breaks".

**From inside the cluster.** [`in_cluster.py`](in_cluster.py) runs Locust as a Kubernetes Job next
to the API. The first run went from the Windows host instead, and Podman's port forwarder added
about 2 s to ~5% of requests: the server measured chat completions at p95 96 ms, the client
2,100 ms. Those numbers describe the laptop, not the service, so they were discarded.

**Objective.** p95 under 500 ms with under 1% errors over the whole mix. RAG can't meet that on
this hardware even unloaded (below), so results are also reported per request class: chat
p95 < 500 ms, RAG p95 < 1,000 ms. The per-class objectives were added after the first runs.

## Findings

**1. The breaking point was a cliff, not a slope: out-of-memory at 50 users.** At 25 users the
gateway served 20.6 req/s cleanly. At 50, 74% of requests failed with a 2 ms median. Both API pods
had been OOM-killed (exit 137, 4 restarts each) and were crash-looping, so requests hit pods that
weren't there. Every request handed its embedding or rerank to a thread with nothing limiting how
many ran at once. Each ONNX inference allocates working memory and ONNX Runtime keeps what it has
allocated, so pods went from ~350 MiB idle past their 1.5 GiB limit.

**2. Fix: bounded concurrency with backpressure** ([`app/core/concurrency.py`](../app/core/concurrency.py)).
At most 2 inferences per model run at once, a bounded queue waits, and anything beyond is refused
immediately. Each caller degrades before failing: a saturated reranker returns vector order, a
saturated embedder makes chat skip the cache, and only a request with no fallback gets a fast 503
with `Retry-After`. Result: **zero errors up to 200 users, 6× the throughput (127.8 req/s), memory
peaking at ~630 MiB, no restarts.** A bigger memory limit would only have moved the cliff.

**3. That fix regressed RAG, and the next one repaired it.** With a queue in front of the
reranker, RAG requests *waited*: at 25 users RAG search p95 went from 920 ms before the fix to
2,000 ms after it, and reached 4,100 ms at 100 users.
Reranking is optional — unreranked order scored recall@5 0.971 in the [retrieval
eval](../eval/README.md) at the time, and 1.000 now that retrieval is hybrid — so it got a **250 ms
wait budget**: a rerank that can't start in time is skipped. p95 fell about 4× at 50–100 users, and
the per-class ceilings became:

| Class (p95 objective) | Before | Bounded concurrency | + wait budget |
|---|---:|---:|---:|
| chat (< 500 ms) | 20.6 req/s | 68.1 req/s | **153.7 req/s** |
| RAG (< 1,000 ms) | 20.6 req/s | none | **79.0 req/s** |

**4. What limits it now is CPU, and specifically the reranker.** The kind node has 12 vCPUs for
everything. At 200 users each API pod used ~9.5 cores and the cross-encoder was the largest
consumer. On one laptop, more replicas don't add CPU. On a real cluster the API scales
horizontally, since it is stateless (Redis and Qdrant hold the state).

**5. RAG has a latency floor on CPU.** Unloaded, a RAG search takes ~400 ms at the median, most of
it the cross-encoder scoring 20 passages. That is why no load level meets 500 ms over the whole
mix. The measured options: rerank 10 candidates instead of 20 (the eval measured 107 vs 231 ms), a
smaller reranker, or a GPU. Hybrid retrieval has since made a fourth option real — skipping the
reranker entirely now costs ordering rather than recall (hit@1 0.922 → 0.794, recall@5 unchanged at
1.000), which these numbers predate and a re-run would price properly.

**6. Losing the primary provider cost nothing visible to users.** Under a steady 20 req/s, the
primary was made to fail every call for 45 s (through `PUT /v1/admin/faults/{deployment}`, which
fails the deployment on every replica, exactly where a real provider error would):

- **0 user-visible failures** out of 2,100 requests.
- Breakers opened **0.9 s** after the fault, after 7 failed calls, all absorbed by fallback.
- While it stayed down, the breakers probed it once per replica per 30 s cooldown.
- The primary was serving again **16 s** after the fault was cleared (up to one cooldown).
- Failing over has a price: the fallback is slower (p50 83 → 153 ms) and priced 5×, so cost per
  1,000 requests went from $0.027 to $0.134 during the fault. That is worth knowing before
  choosing a fallback order.

## Autoscaling

Separate question, separate test: when a backlog builds, does the worker Deployment grow, and is
the cluster better off for it? [`autoscale.py`](autoscale.py) queues a burst of uploads and then
samples queue depth, what the HPA reads, and the replica count every 5 s until the backlog clears.

```bash
python loadtest/autoscale.py --admin-token <token>                        # burst, then watch
python loadtest/autoscale.py --admin-token <token> --documents 150 --repeat 80 --concurrency 32
```

It runs on the host, not in the cluster, because it reads Kubernetes objects rather than just
HTTP. `--repeat` pads each document: the handbook files are ~2 KB and a warm worker ingests one in
about 100 ms, so unpadded uploads never form a queue to scale on.

**Three things the test settled**, each of which changed the manifests:

1. **CPU is the wrong signal for the workers.** With a CPU metric on the worker HPA, replicas grew
   2 → 4 while the queue was **empty**: one ingest job drives a worker to 3355% of its CPU request,
   because embedding is what it does. A HPA takes whichever metric asks for the most replicas, so
   CPU always won and queue depth never got a say. The worker now scales on queue depth alone.
2. **A start-up spike is not load.** The API HPA scaled 2 → 4 on an idle cluster: a fresh pod loads
   the embedding and reranking models and pegs its CPU for about a minute, and every pod it adds
   does the same. Scale-up now ignores anything shorter than two minutes.
3. **Extra workers only help if a backlog outruns one worker.** The first A/B, 300 documents of
   24 KB, showed no improvement at all — uploads arrived at about the rate a single worker cleared
   them, so the queue sat flat at 58 and the extra pods had nothing to do. Making each job heavy
   enough to saturate a worker is what produced the numbers in [`AUTOSCALING.md`](AUTOSCALING.md).

**And then it found two bugs**, by making the jobs heavy enough (150 documents of 160 KB):

4. **A big document could kill the worker that ingested it.** Every chunk of a document went into
   the embedding model in one call, so peak memory scaled with the file: ~660 chunks exceeded the
   pod's 2 GiB limit and the kernel killed it (`OOMKilled`, exit 137, 15 s after start). The file
   was well inside the 10 MB upload limit, so nothing rejected it. Fixed by embedding in batches
   of `NEXUSGATE_EMBED_BATCH_SIZE` (32), which bounds memory by the batch rather than the file.
5. **That job then became a poison pill.** `acks_late` redelivers a job whose worker died — which
   is right — but the attempt number was passed in by the caller, and Celery resets it on
   redelivery, so the job was attempt 1 every time. It killed each worker in turn and all four
   crash-looped on it for **70 minutes**. Attempts are now counted per job in Redis, where they
   survive the worker, and a job delivered more times than `job_max_attempts` is dead-lettered
   with "it may be killing its worker" instead of taking the pipeline down.

## Limitations

- **One laptop.** The load generator, the gateway, Redis, Qdrant and the monitoring stack share
  12 vCPUs, so absolute throughput is a floor, not a capacity plan. The comparison between runs is
  the meaningful part.
- **Autoscaling was measured on a single node**, so "more replicas" means more processes on the
  same 12 vCPUs, not more hardware. It shows the control loop works and what it costs; on a real
  cluster the gain would be larger.
- **A mock provider with fixed latency.** This measures the gateway (auth, rate limiting, cache,
  routing, retrieval), not real LLM latency, which would dominate end-to-end time.
- **45 s per level.** Long enough for stable percentiles at these rates, too short to show slow
  leaks; a soak test is future work.
