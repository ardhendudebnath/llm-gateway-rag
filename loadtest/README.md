# Load and chaos testing

How much traffic can NexusGate take before it misses its latency objective, what breaks first,
and what happens when a provider dies mid-traffic. Numbers: [`RESULTS.md`](RESULTS.md); raw data:
`results/*.json`.

```bash
python loadtest/in_cluster.py --label baseline                 # stepped load test, 5 levels x 45 s
python loadtest/in_cluster.py --label streaming -- --stream-weight 4   # with streamed requests
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

**Streamed requests are a fifth kind, off by default** (`--stream-weight`, default 0). Turning them
on changes the mix, which would make a run incomparable with the ones already in `results/` — so a
streaming run is a separate, labelled one, and every result file records the weight it ran with.

A streamed request is reported as two figures, because one would hide what matters: a slow
generation with a fast first token is a good experience.

| Metric | What it is |
|---|---|
| `chat: stream first token` | What Locust times for a streamed request: time to the response headers. For this gateway that *is* time to first token — the endpoint pulls the first chunk before the response exists, so the headers cannot precede it |
| `chat: stream complete` | The whole stream, timed in the task and reported as its own metric, because Locust stops timing at the headers |

The task consumes the entire body rather than abandoning it after the first chunk: the gateway
meters a stream when it ends, and a client that walks away mid-stream is a different measurement.

**Levels.** 25, 50, 100, 200 and 400 simulated users with 0.5–1.5 s think time, 45 s each, all
users spawned at once and `--reset-stats` so the ramp isn't counted. A dedicated API key with a
very high rate limit keeps the gateway's own limiter from being what "breaks".

**From inside the cluster.** [`in_cluster.py`](in_cluster.py) runs Locust as a Kubernetes Job next
to the API. The first run went from the Windows host instead, and Podman's port forwarder added
about 2 s to ~5% of requests: the server measured chat completions at p95 96 ms, the client
2,100 ms. Those numbers describe the laptop, not the service, so they were discarded.

**A fixed two API replicas, with the autoscaler suspended.** Re-running the test months later gave
89.8 req/s at 200 users against the 153.7 on record — with no change that could explain it. The
cause was the environment, not the code: the HPA had scaled the API to **six** pods, and six copies
of the embedding and reranking models contend for one laptop's cores. Two back-to-back levels in
that state agreed with each other (119.4 and 118.7 req/s), so the measurement was stable and the
*conditions* were not.

Comparable runs therefore delete the `api` HPA and scale the deployment to 2, then restore it
afterwards; autoscaling is measured separately by [`autoscale.py`](autoscale.py)
([AUTOSCALING.md](AUTOSCALING.md)). Every level now records the number of API pods Prometheus could
see while it ran, so a result that was taken under different conditions says so itself. The runs
from before that existed do not carry the field.

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

**6. The first level of a run measures start-up, not the service.** Re-running the test after
months of changes appeared to show a 42% throughput regression at 200 users (89.8 req/s against
153.7). Repeating the same level three times, interleaved, settled it: the first pass is the worst
at *every* load — 78.8 / 111.7 / 149.6 req/s at 100 / 150 / 200 users — and the second and third
agree within about 1% at 84.8 / 123.3 / 156.8. The pods had restarted shortly before each run, so
the first measured level was paying for cold ONNX sessions and empty connection pools. Runs now
start with a discarded warm-up level, levels are 60 s rather than 45, and the harness can repeat a
level so the spread is reported instead of assumed:

| Users | runs | median | range | spread |
|---:|---:|---:|---:|---:|
| 100 | 3 | 84.7 req/s | 78.8 to 84.8 | 7% |
| 150 | 3 | 123.3 req/s | 111.7 to 123.5 | 10% |
| 200 | 3 | 154.6 req/s | 149.6 to 156.8 | 5% |

Warm, the current code matches the number on record: **156.8 req/s at 200 users against 153.7**, with
streaming, hybrid retrieval, canary route selection and pooled circuit state all in the request path.
Three measurement artifacts had stacked up to hide that — an autoscaler that had taken the API to six
pods, a shared tenant that accumulated every corpus ever ingested, and cold pods inside a 45 s window.

**7. A level between 100 and 200 users was worth measuring.** The old levels jumped from 100 to 200
and no run ever met the whole-mix objective. At 150 users it does: **123.5 req/s with p95 460 ms**
over the whole mix, which is the honest headline for this stack rather than a per-class figure.

**8. RAG latency did regress, and the reranker's input is why.** RAG search p95 at 100 users is now
1,700–2,000 ms against 960 ms on record, reproducibly across all three warm passes, at unchanged
throughput (14.6 req/s against 13.7). Measured by condition at 100 users: hybrid with 9 documents
1,300 ms, dense with 9 documents 1,500 ms, dense with 13 documents 1,800 ms. So **hybrid retrieval
costs nothing measurable here** — the fused search is nominally faster than the dense one — while the
corpus growing from 9 documents to 13 costs 300–500 ms, because the four documents added for the
retrieval eval are dense tables and structure-aware chunking keeps their rows whole, handing the
cross-encoder more text per candidate. A residual of roughly half a second is not attributed; the
historical 960 ms is a single 45 s sample whose own spread was never measured, and the evidence above
is that single samples here can be off by a factor of two. RAG no longer meets its 1 s objective at
any level, and the summary tables say so.

The fix came from the same reasoning: if the cross-encoder's input is the cost, hand it less of it.
The retrieval eval priced two ways of doing that on the large corpus. Capping each candidate's text
is a bad trade — 90 words halves the per-query cost but takes identifier-only recall from 1.000 to
0.833, because a table's answer is a row and a cap cuts rows off. Scoring **12 candidates instead of
20** is free: recall@5 0.980, identifier-only 1.000 and hit@1 0.922 are unchanged, for 353 ms a
query against 722. Hybrid retrieval is what made it free, by putting the answer near the top of the
candidate list. That is now the default.

**10. Measuring that fix needed an A/B in one sitting, and taught the resolution of this bench.**
The first attempt compared a 12-candidate run against the 20-candidate run from the week before and
appeared to show the fix making everything *worse* — 95.1 req/s at 200 users against 154.6. Re-running
the 20-candidate configuration on the same day gave 89.0–100.7 req/s at 150 users against 123.3 the
week before: the same configuration, 20% apart, so the week-over-week comparison was worthless
again. Adjacent repetitions within one warm run also differ by about 12%, which is this bench's
resolution: it cannot see an effect smaller than roughly 15%.

Run as two arms back to back in one session, on the same pods, the effect is above that floor and
consistent across both levels and both repetitions:

| 150 users | 20 candidates | 12 candidates |
|---|---:|---:|
| throughput | 100.7, 89.0 req/s | **108.9, 112.9 req/s** |
| whole-mix p95 | 1,200, 1,800 ms | **990, 740 ms** |
| rag: search p95 | 2,000 ms | **1,600 ms** |
| rag: answer p95 | 2,800 ms | **1,800 ms** |
| chat: cache hit p95 | 1,100 ms | **710 ms** |

Chat improves too, which is the giveaway for the mechanism: the reranker and the embedder compete
for the same cores, so taking work away from the reranker hands it back to the embedding behind
every cache lookup. RAG search p95 at 100 users falls from 2,200 ms to 1,400 ms. Against the 960 ms
on record that still leaves a few hundred milliseconds, which is about what the larger corpus costs
(finding 8) — so the regression is now explained by the fixture rather than outstanding.

**11. An hour at steady load: no leak, no errors, and a latency drift mostly from the laptop.**
[`soak.py`](soak.py) held 40 users — streamed requests included — for an hour, sampling the pods
every 30 s. It was written for the state recent work added: asyncio tasks that meter streams whose
clients hang up, a Redis counter per request for canary rollouts, and pooled breaker keys. None of
it leaks: per-pod resident memory went 823 → 828 MB (+0.5%) across 123,157 requests and about
30,000 streams, with zero server errors and ~10,000 requests degraded gracefully rather than failed.

The first version of the script then printed PASS while p95 rose 65%, because it only had thresholds
for memory, errors and file descriptors. A soak that ignores latency drift is not a soak, so it now
fails on drift past 25%, and the stored result was re-derived from the same samples and records that
it was.

The drift rose for ~20 minutes then plateaued, at flat throughput, with a queue forming at the
inference gate — what a capped CPU looks like, and the laptop was on its "Silent" power plan. So the
same load ran again on "Performance", compared over windows fixed before the second run:

| p95 | Silent | Performance |
|---|---:|---:|
| minutes 0-17 | 772 ms | 898 ms |
| minutes 20-30 | 1,222 ms | 1,048 ms |
| drift | **+58%** | **+17%** |

The power plan accounts for most of it. The residual 17% is not distinguishable from noise: the two
runs' *starting* windows already differ by 16%, the same size as the residual and inside this bench's
~12-15% resolution (finding 10). What is consistent across both is the inference queue growing from
under one waiting request to about one, which is the thing to watch on dedicated hardware, where a
power plan cannot be the explanation.

**12. Losing the primary provider cost nothing visible to users.** Under a steady 20 req/s, the
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
