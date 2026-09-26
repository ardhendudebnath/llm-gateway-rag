# NexusGate

[![CI](https://github.com/ardhendudebnath/llm-gateway-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ardhendudebnath/llm-gateway-rag/actions/workflows/ci.yml)

**NexusGate is a self-hosted LLM gateway and RAG backend: one OpenAI-compatible API in front of several model providers, with automatic fallback, circuit breakers and a tenant-isolated semantic cache.** Documents uploaded to it are ingested by background workers into Qdrant, and questions about them get reranked answers with numbered citations — either in one retrieval, or from a multi-step agent that plans its own searches and reviews its draft before answering. Every request is authenticated, rate-limited, metered in tokens and dollars, traced and exported to Prometheus, and the whole stack runs on Kubernetes, load-tested and chaos-tested.

**Try it in one container** (offline mock providers, no API keys, nothing to pay for):

```bash
podman build -f Containerfile.demo -t nexusgate-demo . && podman run --rm -p 7860:7860 nexusgate-demo
```

Then open http://localhost:7860. The page gives you an API key and curl commands to paste; `docker` works in place of `podman`.

<!-- Live demo: add the public URL here once it is deployed (see "Deploying the demo"). -->

![NexusGate walkthrough: a provider call, a cache hit, fallback, a queued upload, a cited RAG answer, spend](docs/demo.gif)

<sub>A real run of <a href="scripts/demo.py"><code>scripts/demo.py</code></a> against the Kubernetes stack, replayed at recorded speed.</sub>

```mermaid
flowchart LR
    C[Client / OpenAI SDK] --> G[API Gateway<br/>FastAPI]
    G -->|auth + token bucket| R[(Redis)]
    G --> SC{Semantic cache<br/>Redis vectors}
    SC -- hit --> C
    SC -- miss --> RT[LLM Router<br/>fallback + circuit breaker]
    RT --> A[Anthropic]
    RT --> O[OpenAI]
    RT --> V[vLLM self-hosted]
    G -->|/v1/rag/answer| RAG[RAG orchestrator<br/>embed · Qdrant · rerank]
    RAG --> QD[(Qdrant)]
    RAG -->|augmented prompt| SC
    G -->|/v1/agents/research| AG[Research agent<br/>plan · retrieve · draft<br/>critique · revise]
    AG --> RAG
    U[Document upload] -->|202 + job id| Q[[Celery queue<br/>Redis]]
    Q --> W[Ingestion workers<br/>parse · chunk · embed]
    W --> QD
    W -. failed after retries .-> DLQ[[Dead-letter queue]]
    G --> P[Prometheus → Grafana]
    P --> AM[Alertmanager] -->|webhook| G
    G -. optional .-> LF[Langfuse traces]
```
<sub>Dashed arrows are optional (tracing) or failure-only (dead letters) paths.</sub>

## Headline numbers

Measured on one 12-vCPU laptop shared by every component, with Locust running inside the cluster. Method, raw results and limitations: [loadtest/README.md](loadtest/README.md) and [eval/README.md](eval/README.md).

| What | Result |
|---|---|
| Chat throughput, p95 under 500 ms, <1% errors | **153.7 req/s**, up from 20.6 before the load-test fixes |
| RAG throughput, p95 under 1 s, <1% errors | **79 req/s**, up from 20.6 |
| Primary provider killed under traffic | **0 user-visible failures** in 2,100 requests; breakers opened 0.9 s after the fault |
| Retrieval quality, 48 labelled questions | **recall@5 1.000, MRR 0.990** |
| Tests | **349** (unit, integration, provider contracts), **97% coverage**, no network access, no API spend |

The load test found a real breaking point. Unbounded model inference OOM-killed the API at 50 users, with 74% errors. Bounding it with backpressure and graceful degradation took it to 0 errors at 200 users and 6× the throughput (see [Design decisions](#design-decisions)).

## Quickstart

There are three ways to run it, from quickest to most complete.

### 1. The one-container demo

This is the command at the top: the gateway, Redis, an in-process Qdrant and the real embedding and reranking models in one image ([`Containerfile.demo`](Containerfile.demo)). A fictional engineering handbook is pre-loaded for RAG. It is locked down so it can be put on the internet as is: mock providers only, one shared rate-limited key, uploads off (see [the design notes](#a-public-demo-thats-safe-to-expose)).

### 2. The full stack on Kubernetes

This runs 2 API replicas, a Celery worker, Redis, Qdrant, Prometheus, Alertmanager and Grafana on a local Kubernetes cluster: [kind](https://kind.sigs.k8s.io/) on [Podman](https://podman.io/). No Docker needed.

**Prerequisites:** Podman, kind and kubectl.
- On Windows: `winget install RedHat.Podman Kubernetes.kind Kubernetes.kubectl` (Podman uses WSL2).
- If PowerShell blocks the script, run it as `powershell -ExecutionPolicy Bypass -File .\scripts\cluster-up.ps1`.

```bash
cp .env.example .env              # optional: ANTHROPIC_API_KEY / OPENAI_API_KEY for real providers
./scripts/cluster-up.sh           # Windows: .\scripts\cluster-up.ps1
```

That one command does the following:
- Creates a rootful Podman machine if needed, then a kind cluster.
- Builds the image from the [`Containerfile`](Containerfile) with Podman and loads it into kind.
- Creates the Secret from `.env`, generating any admin, JWT or Qdrant secrets that `.env` lacks, and prints the admin token at the end.
- Applies the [kustomize overlay](infra/k8s/overlays/kind).

The embedding and reranking models are baked into the image, so pods never download weights. Re-running rebuilds the image and rolls the API. `scripts/cluster-down.sh` (or `.ps1`) deletes the cluster.

Once it is up:
- API docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (the **NexusGate** dashboard is provisioned automatically)

Check the running stack end to end. It uses mock routes only, so there are no API keys and no spend:

```bash
python scripts/smoke_test.py --admin-token "$ADMIN_TOKEN" --prometheus-url http://localhost:9090
```

The `mock` and `chaos` routes work with **no API keys**:

```bash
# 1. Issue an API key (the admin token is printed by cluster-up)
curl -s -X POST localhost:8000/v1/admin/keys \
  -H "X-Admin-Token: $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme", "name": "demo"}'

# 2. Chat. The second identical call is a cache hit: $0, no provider call.
curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ng_..." -H "Content-Type: application/json" \
  -d '{"model": "mock", "messages": [{"role": "user", "content": "What is a circuit breaker?"}]}'

# 3. Watch fallback: the primary of `chaos` always fails, so the secondary answers.
#    After 3 failures the breaker opens and the broken deployment is skipped outright.
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer ng_..." \
  -H "Content-Type: application/json" \
  -d '{"model": "chaos", "cache": false, "messages": [{"role": "user", "content": "ping"}]}'

# 4. Usage and spend, including $ saved by the cache
curl -s localhost:8000/v1/usage -H "Authorization: Bearer ng_..."
```

RAG over your own documents (PDF, Markdown or text):

```bash
# Upload: returns 202 with a job id; a worker parses, chunks and embeds it
curl -s localhost:8000/v1/rag/documents -H "Authorization: Bearer ng_..." \
  -F file=@eval/corpus/incident-response.md

# Poll the job: queued -> processing -> done (or failed, with the reason)
curl -s localhost:8000/v1/rag/jobs/<job_id> -H "Authorization: Bearer ng_..."

# Search: vector top-20, reranked by a cross-encoder, top 5 returned
curl -s localhost:8000/v1/rag/search -H "Authorization: Bearer ng_..." \
  -H "Content-Type: application/json" -d '{"query": "when is a postmortem due?"}'

# Answer with numbered citations, through the normal gateway path (fallback, cache, metering)
curl -s localhost:8000/v1/rag/answer -H "Authorization: Bearer ng_..." \
  -H "Content-Type: application/json" \
  -d '{"question": "When is a postmortem due?", "model": "default"}'

# Or let the agent plan the searches, draft, criticise itself and revise.
# The response lists every transition it took, and what the run cost.
curl -s localhost:8000/v1/agents/research -H "Authorization: Bearer ng_..." \
  -H "Content-Type: application/json" \
  -d '{"question": "How do we handle a SEV1?", "model": "default"}'
```

Because the API is OpenAI-compatible, existing SDKs work unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="ng_...")
client.chat.completions.create(model="mock", messages=[{"role": "user", "content": "hi"}])

for chunk in client.chat.completions.create(  # streaming, same as any OpenAI endpoint
    model="mock", messages=[{"role": "user", "content": "hi"}], stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

### 3. Local development (no cluster)

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Windows: .venv\Scripts\pip
pytest --cov
uvicorn app.main:create_app --factory --reload               # needs a Redis on localhost:6379
```

## Demo walkthrough

[`scripts/demo.py`](scripts/demo.py) tells the whole story in eight narrated steps, paced for a screen recording:
1. Readiness.
2. A chat request goes to a provider.
3. The same question again is a semantic-cache hit: $0 and a few milliseconds.
4. A failing provider, and the fallback that answers.
5. An upload that is queued and then ingested by a worker.
6. A cited RAG answer.
7. An agent run: the path it took through the graph, and what it cost.
8. Spend and cache savings.

It uses only the offline routes, so it costs nothing.

```bash
# Against the one-container demo: paste the key from the landing page
python scripts/demo.py --base-url http://localhost:7860 --key ng_...

# Against the Kubernetes stack: it mints a throwaway key and revokes it afterwards
python scripts/demo.py --admin-token "$ADMIN_TOKEN"
```

In the public demo, step 5 reports that uploads are disabled and moves on to the pre-loaded handbook. CI runs this same walkthrough against the demo image on every push.

### Recording the demo GIF

The GIF at the top is generated, not screen-captured. [`scripts/record_gif.py`](scripts/record_gif.py) runs the walkthrough, timestamps every line it prints, and replays them as terminal frames at the speed they appeared. Nothing is staged, and credentials never appear in the frames. Everything after `--` is passed to `demo.py`:

```bash
NEXUSGATE_ADMIN_TOKEN=... python scripts/record_gif.py -- --pace 1.2    # writes docs/demo.gif
```

### Deploying the demo

The demo image is self-contained, and the host needs to provide very little:
- **No volumes and no secrets.** Admin and JWT secrets are generated at start-up and never shown.
- **About 0.5 GB of memory** under light traffic (both ONNX models plus Redis).
- **A port.** It listens on `$PORT`, 7860 by default, which is what Hugging Face Spaces (Docker SDK) expects.

State lives in memory, so a restart resets the handbook, the usage counters and the shared key. For a demo, that is intended.

To put it on a free [Hugging Face Space](https://huggingface.co/docs/hub/spaces-sdks-docker):

```bash
hf auth login                                          # once; the token stays in the HF CLI's store
python scripts/deploy_space.py <hf-user>/nexusgate      # add --private to try it privately first
```

[`scripts/deploy_space.py`](scripts/deploy_space.py) stages exactly what `Containerfile.demo` needs, under the names a Space expects (a `Dockerfile` and a README with a config header). It uploads them, waits for Hugging Face to build and start the image, then checks `/readyz` on the public URL. Re-running it deploys the current code, and `--dry-run --stage-dir <dir>` shows what would be uploaded.

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/chat/completions` | key / JWT | OpenAI-shaped chat; response adds a `nexusgate` block (deployment, attempts, cost, cache hit) |
| ↳ with `"stream": true` | key / JWT | Server-sent events of `chat.completion.chunk`, ending in `data: [DONE]`; the last chunk carries usage and the `nexusgate` footer (time to first token, cost, attempts, and `interrupted` if the stream broke) |
| `GET /v1/models` | key / JWT | Route aliases usable as `model` |
| `POST /v1/auth/token` | API key | Exchange a key for a short-lived JWT |
| `GET /v1/usage?days=7` | key / JWT | Daily requests, tokens, $ spent, $ saved by cache, and cost per 1,000 split by cache hit / paid provider / self-hosted |
| `DELETE /v1/cache` | key / JWT | Purge the caller's tenant cache |
| `POST /v1/rag/documents` | key / JWT | Queue a PDF / Markdown / text file (multipart); **202** with a job; 413 over 10 MB, 415 unsupported or binary; 403 in the public demo |
| `GET /v1/rag/jobs[/{id}]` | key / JWT | Ingestion jobs: queued / processing / done / failed, with attempts and the failure reason |
| `GET /v1/rag/documents[/{id}]` | key / JWT | Your documents, newest first; one document's record |
| `DELETE /v1/rag/documents/{id}` | key / JWT | Delete a document and its chunks |
| `POST /v1/rag/search` | key / JWT | Top-k passages with vector and rerank scores |
| `POST /v1/rag/answer` | key / JWT | Grounded answer with numbered citations; no LLM call when nothing matches |
| `POST /v1/agents/research` | key / JWT | Multi-step agent: plans searches, drafts a cited answer, critiques and revises it; reports every transition and what the run cost |
| `GET /v1/agents/graph` | key / JWT | The agent's nodes, allowed transitions and a Mermaid diagram |
| `POST/GET/DELETE /v1/admin/keys` | admin token | Issue, list, revoke API keys |
| `GET /v1/admin/providers` | admin token | Route table + live circuit-breaker states |
| `GET /v1/admin/routes` | admin token | The live route table, plus any canary and how it is faring |
| `PUT /v1/admin/routes/canary` | admin token | Send a share of traffic to a candidate route table |
| `POST /v1/admin/routes/canary/weight` | admin token | Turn the canary's share up or down |
| `POST /v1/admin/routes/canary/promote` | admin token | Make the canary the table for all traffic |
| `DELETE /v1/admin/routes/canary` | admin token | Withdraw it; everything returns to stable |
| `GET/DELETE /v1/admin/dead-letters` | admin token | Jobs that failed for good; inspect or clear |
| `GET /v1/admin/alerts` | admin token | Alerts Alertmanager has delivered, newest first |
| `GET/PUT/DELETE /v1/admin/faults[/{deployment}]` | admin token | Chaos testing: make a deployment fail on every replica (off unless enabled) |
| `POST /v1/alerts/webhook` | basic auth | Alertmanager receiver (admin token as the password) |
| `DELETE /v1/admin/cache` | admin token | Purge the whole cache |
| `GET /healthz`, `/readyz`, `/metrics` | none | Liveness, readiness (Redis + Qdrant), Prometheus |
| `GET /` | none | Demo mode only: landing page with the shared key and examples |

Every error uses one envelope: `{"error": {"type": ..., "message": ...}}`. A 429 carries `Retry-After`. A 503 from total provider failure includes each attempt.

## Design decisions

**Fallback lives in our router, not in LiteLLM.** LiteLLM is used only as a provider adapter, with `num_retries=0`. Retry, fallback and circuit breaking are ~150 lines in [`app/gateway/router.py`](app/gateway/router.py), so the behaviour is explicit and unit-tested, and every attempt shows up in the response and in metrics.

**Not every error triggers fallback.** Timeouts, 429s, 5xx and auth failures move on to the next deployment. A provider-side **400** fails fast: another vendor would reject the same malformed request, so the chain isn't burned. It also doesn't count against the breaker, because a bad request isn't an outage.

**Circuit breaker.** A deployment opens after N consecutive failures. After the cooldown it goes half-open and lets **exactly one** probe through; the probe's result either closes the breaker or reopens it for a fresh cooldown. State is kept per replica on purpose. Each replica learns about an outage within N requests, which is cheaper than a Redis round-trip on every call.

**The cache matches only the last user message semantically.** Tenant, route, temperature, max_tokens, system prompt and earlier turns are all hashed into an exact-match namespace. That way "answer in French" and "answer in English" can never share a cached answer, however close their embeddings are, and tenant A can never be served tenant B's response. Entries have a TTL, each namespace has a size cap, and there are purge endpoints. If the cache fails, requests are served uncached instead of erroring.

**Two vector-index backends behind one storage layout.** `redisearch` (an HNSW index in redis-stack) is what the Kubernetes stack runs. `bruteforce` (a numpy dot product) needs no extra infrastructure; the tests and the one-container demo use it.

**Rate limiting is one Lua script.** The token-bucket refill-and-take runs atomically inside Redis, so replicas sharing a Redis never overspend a bucket. A test fires 50 concurrent requests at a 10-token bucket and asserts that exactly 10 get through.

**Cost is metered per kind, not just blended.** `GET /v1/usage` reports cost per 1,000 requests for cache hits, paid provider calls and self-hosted deployments separately, because the blended number hides which lever moved it: a better hit rate and a shift onto your own GPU both push it down and cost completely different things to arrange. Self-hosted is a flag on the deployment ([`routes.yaml`](config/routes.yaml)), not an inference from a zero price, which a free API tier would also have.

**Only a SHA-256 of each API key is stored.** A Redis dump contains no usable credentials. JWTs are checked against the key record on every request, so revoking a key immediately kills its outstanding tokens. In `prod` the app refuses to start with the default secrets.

**RAG on Qdrant directly, no framework.** The pipeline is ~800 lines, docstrings included, in [`app/rag/`](app/rag), using `qdrant-client` and `fastembed` without LlamaIndex or LangChain. Every stage is small, injectable and unit-tested, and the retrieval eval runs the exact production code.
- **Chunking was chosen by measurement.** [Three strategies](app/rag/chunking.py) were compared on 48 labelled questions: fixed windows, sentence packing, and structure-aware chunks that start at every heading and carry a "title > heading path" prefix. Structure-aware won (MRR 0.924 vs 0.856 without a reranker). A cross-encoder rerank of the top 20 then reaches **recall@5 1.000, MRR 0.990** at ~230 ms per query. Full method, numbers and limitations: [eval/README.md](eval/README.md).
- **Multi-tenancy is enforced in the store.** It is one collection with a `tenant_id` payload index (`is_tenant=True`), and every query and delete carries a tenant filter. There is no code path that searches across tenants. A test proves tenant B can't retrieve or delete tenant A's documents.
- **Ingestion is idempotent.** Document ids are content hashes and point ids are UUIDv5 of (tenant, document, chunk), so re-uploads and retried jobs overwrite rather than duplicate.
- **Answers go through the normal chat path**, so they get fallback, circuit breaking, metering and the semantic cache. Retrieved passages sit in the system message, which is part of the cache namespace, so an answer is only reused when the *same* passages were retrieved. If nothing matches, the API says so without calling an LLM, so there is no cost and no invented answer.
- **Retrieved text is untrusted.** The prompt tells the model to treat passages as data and ignore instructions inside them. Uploads are type-checked, size-capped, and rejected if they contain NUL bytes (binaries disguised as text).
- **A 4× latency fix came from profiling.** Reranking first measured ~1 s per query: ONNX Runtime gave each model a 24-thread pool and the two pools fought over the CPU. Capping threads per model (`NEXUSGATE_MODEL_THREADS=4`) brought it to ~230 ms with identical scores.

**The agent is an explicit state graph, not a loop with a prompt.** `POST /v1/agents/research` plans its own searches, merges the results, drafts a cited answer, then critiques that draft against the passages and revises it. The [engine](app/agents/graph.py) is ~90 lines, with no agent framework, for the same reason RAG has none: every transition is inspectable and unit-tested.
- **The graph is data, and it is published.** `GET /v1/agents/graph` returns the nodes, the allowed transitions and a Mermaid diagram — all generated from the definition the service actually runs, so a picture of the agent can't drift from its behaviour.
- **A run is bounded in two ways.** A node may only move to a target its edges declare, so no prompt (and no model inventing a step name) can steer the run off the graph; and a step budget ends a critique/revise cycle that won't settle. A budgeted-out run still returns its last draft and its path, rather than failing.
- **Degrade per step, except when the failure dooms the run.** A failed plan falls back to searching for the question as asked, and a failed reviewer keeps the draft. But an unknown route, a bad request, shedding or every provider being down [fails immediately](app/agents/nodes.py), because paying for retrieval and a draft before hitting the same error is worse than failing now.
- **Every step goes through the gateway**, so the agent inherits fallback, circuit breaking, the cache, metering and tracing. Multi-step means multi-cost, so each response reports its own `llm_calls` and `cost_usd`, and each run is one trace with a completion per step.
- **Retrieved text stays untrusted at every step.** Each prompt that shows passages repeats that they are data, so a document can't steer the agent's later steps.

**Degrade before failing, and shed before falling over.** The load test showed that unbounded model inference takes the API down: dozens of simultaneous ONNX runs pushed the pods past their memory limit. Each model now sits behind an [`InferenceGate`](app/core/concurrency.py): a fixed number of inferences run at once, a bounded number wait, and the rest are refused immediately.
- **Callers degrade first.** A saturated reranker means vector order is returned; a saturated embedder means chat skips the cache and still answers.
- **Only requests with no fallback are shed**, with a 503 and `Retry-After`.
- **Reranking has a 250 ms wait budget**, because it is optional work: waiting for it cost more than skipping it.
- **All of it is counted and alerted on:** `nexusgate_degraded_total`, `nexusgate_inference_shed_total`, and the `NexusGateSheddingLoad` alert.

**Streaming makes fallback a deadline, not a policy.** With `stream: true` the gateway sends
server-sent events in OpenAI's format ([`app/api/chat.py`](app/api/chat.py)), and that changes what
the router can promise. You cannot un-send a token, so every fallback decision has to be made before
the first one goes out — [`open_stream`](app/gateway/router.py) returns only once some deployment has
actually produced a token, and after that the choice is locked in.
- **The timeout that matters is time to first token.** A generation may legitimately run for a
  minute; waiting for it to finish before deciding whether the provider is alive would be useless.
  The deployment timeout applies to the first token, and again between chunks, so a provider that
  goes quiet mid-answer is cut off instead of hanging the client.
- **A break after the first token is reported, not retried.** Falling back then would append a
  second answer to a partial one. The partial text stands, the last chunk says
  `nexusgate.interrupted`, and the stream still ends with `data: [DONE]` so clients terminate
  cleanly. It is also billed — for the words that were actually sent, from a running estimate.
- **A first token is not a success.** The breaker is settled when the stream *completes*: a
  deployment that reliably answers one token and then dies would otherwise reset its own failure
  count on every attempt and never trip.
- **Failures before anything is sent keep their HTTP status.** The endpoint pulls the first chunk
  before the response exists, so a dead chain is still a 503 with `Retry-After` and a rejected
  prompt is still a 400 — not a 200 carrying an error in the body.
- **Cache hits stream too**, chunked at the gateway, so a client cannot tell a replay from a
  generation apart from `cached: true` and the cost it did not pay. Partial answers are never cached.
- **A provider that cannot stream still serves streaming clients**: its answer is chunked here and
  flagged `synthesized`, rather than the request being refused.

**A route change is a deployment, so it rolls out like one.** Swapping a provider, reordering a fallback chain or putting a cheaper model in front is a production change that happens to be configuration. [`app/gateway/rollout.py`](app/gateway/rollout.py) ships one to a percentage of traffic, with no restart: `PUT /v1/admin/routes/canary` publishes a candidate table at a weight, `GET /v1/admin/routes` shows how it is doing, and it is then promoted or withdrawn.
- **It withdraws itself.** Once the canary has served `NEXUSGATE_CANARY_MIN_REQUESTS` (20), an error rate above `NEXUSGATE_CANARY_MAX_ERROR_RATE` (10%) drops its weight to 0 and records why. A canary that needs someone watching a dashboard is just a slower outage.
- **The counters are pooled in Redis**, not per replica: with 2 pods each judging the canary on its own handful of requests, neither would reach a sample worth acting on. The same read gives every replica the current table within a second.
- **Every response says which table served it** (`nexusgate.route_variant`), and so does the log line, so a difference in behaviour can be traced to the version that caused it.
- **A bad table is rejected at publish time**, not at request time: it is parsed and validated before it is stored, so a typo is a 400 for the admin rather than a 500 for a user.
- **Nothing here can fail a request.** If Redis is unreadable the gateway serves the table it started with — exactly what it would have done without any of this.

**Chaos is a first-class API.** `PUT /v1/admin/faults/{deployment}` makes a deployment fail on demand, on every replica. The table lives in Redis, so it isn't one pod's memory, and it is cached for a second so it costs nothing per request. The failure is raised exactly where a real provider error would be, so it exercises the real retry, fallback and breaker paths. It is off unless `NEXUSGATE_FAULT_INJECTION_ENABLED` is set, which only the local cluster overlay does.

**Observability that can't take the service down.** Every completion produces a Langfuse generation (prompt, response, model, tokens, cost, cache hit) *and* a structured log line, both carrying the request id.
- **Tracing is optional and never fatal.** Without Langfuse credentials the gateway uses a no-op tracer. Every call into the SDK is guarded, and tests cover the cases where Langfuse can't be reached or fails mid-request: the request is still served.
- **One request id, end to end.** It is generated (or accepted) at the edge, carried in a `ContextVar` through gateway, router and provider, stored on the ingestion job, and restored inside the worker, so an upload and the job that fulfils it share one id across processes.
- **Alerts go somewhere by default.** Prometheus [rules](infra/k8s/base/config/alert-rules.yml) cover error rate, exhausted providers, open breakers, p95 latency, queue backlog, dead letters and a missing target. Alertmanager posts them to the gateway's own webhook, which logs and counts them, so the whole path works out of the box; switching to Slack or Discord is a receiver change, not a code change.
- **The dashboard ships with the stack.** [One JSON file](infra/k8s/base/config/grafana-dashboard.json) provisioned into Grafana: request rate, error rate, latency percentiles, provider outcomes and latency, cost per hour against cache savings, queue and dead-letter depth, and retrieval latency by stage.

**Ingestion runs in workers, not in the request.** Uploading returns **202** with a job id; a Celery worker does the parsing, chunking and embedding. A 10 MB PDF no longer holds an API worker for seconds, and ingestion capacity scales by scaling the worker Deployment alone.
- **The file doesn't travel through the broker.** Bytes are parked in Redis under a TTL and the queued message carries only a pointer. Celery messages stay small, and an upload whose job never runs expires by itself instead of leaking.
- **Two failure classes, opposite handling.** A bad upload (unsupported, no text, expired payload) fails immediately: retrying it would fail identically. An infrastructure error (Qdrant down, Redis blip) is retried with exponential backoff. Either way the final state lands in the [dead-letter queue](app/workers/jobs.py) with the reason, visible through the admin API.
- **Retry policy sits in the service, not the task**, so all of it is unit-tested without a broker: permanent failure, retry-then-succeed, and exhausting the last attempt.
- **Redelivery is safe.** `acks_late` plus `reject_on_worker_lost` mean a job whose worker is killed is redelivered rather than lost, and ingestion is idempotent, so re-running it overwrites rather than duplicates.
- **The same code path in tests.** An in-process queue runs jobs without a broker, so tests and `uvicorn` on a laptop behave like production: still 202, still polled.
- **A big document can't take a worker down.** Embedding runs in batches of 32 chunks, so peak memory follows the batch, not the file. It didn't, once: a 160 KB upload put ~660 chunks into one call and the worker was OOM-killed.
- **A job that keeps killing its worker gets dead-lettered.** Attempts are counted per job in Redis, not passed in by the caller, because a redelivery after a worker dies resets the caller's count. Without that, one bad document crash-looped four workers for 70 minutes ([how it was found](loadtest/README.md#autoscaling)).
- **Job counters live in Redis**, because the worker has no `/metrics` endpoint. The API publishes queue depth, dead-letter depth and per-outcome totals when Prometheus scrapes it, whichever process ran the job.

**Kubernetes, built with Podman.** The image is an OCI image built from a `Containerfile`, and the stack is plain kustomize: a cluster-agnostic [base](infra/k8s/base) plus a [kind overlay](infra/k8s/overlays/kind) that adds NodePorts and the locally built image.
- **Replicas:** the API runs 2 replicas. Rate limits, the cache and metering live in Redis, so they hold across replicas.
- **Per-pod scraping:** Prometheus discovers every API pod through a headless Service's DNS records. Each replica keeps its own counters, and scraping the load-balanced Service would sample a random pod each time.
- **Hardened pods:** they run as non-root with a read-only root filesystem and all capabilities dropped. An init container waits for Redis and Qdrant instead of letting the API crash-loop. Qdrant runs its unprivileged image and requires an API key.
- **Models in the image:** the embedding and reranking weights are downloaded at build time, and pods run with `HF_HUB_OFFLINE=1`. Startup is fast and needs no internet access.
- **Autoscaling on the signal that matters per workload** ([autoscaling.yaml](infra/k8s/base/autoscaling.yaml)). The API scales on CPU, because the load test showed CPU is its ceiling once inference is bounded. The workers scale on **ingestion queue depth**, served to the HPA by a [prometheus-adapter](infra/k8s/base/prometheus-adapter.yaml) reading the gauge the API publishes from Redis — and on nothing else, because a CPU target scaled them out while the queue was *empty*: one ingest job pegs a worker's CPU, which says it is busy, never that work is piling up.
  - **Measured, including the part that doesn't flatter it.** The loop works — a backlog of 51 jobs per worker against a target of 5 added pods and drained. But on this one-node cluster the same backlog cleared in **312 s with one worker and 324 s with four**: a single worker already uses ~8 cores, so extra pods competed for busy CPUs. The kind overlay therefore caps workers at 2, while the base keeps 4 for a cluster with room. [Numbers](loadtest/AUTOSCALING.md).
  - **A start-up spike is not load.** The API HPA scaled 2 → 4 on an idle cluster, because a fresh pod loads both models and pegs its CPU for a minute — and each pod it added did the same. Scale-up now ignores anything shorter than two minutes.
- **Secrets:** they never enter git. The cluster-up script creates the Secret from `.env`. The pods run with `NEXUSGATE_ENV=prod`, which refuses default secrets.

### A public demo that's safe to expose

`NEXUSGATE_DEMO_MODE=true` ([`app/demo.py`](app/demo.py)) turns the same application into something a stranger can use:
- **Offline routes only** ([`config/routes.demo.yaml`](config/routes.demo.yaml)). The image holds no provider keys, so a visitor can't spend anything, and a test asserts the demo routes contain nothing but mock providers.
- **One shared key for a `demo` tenant, tightly rate-limited** (a burst of 20, then one request every 2 s). It is minted at start-up and shown on the landing page, and it changes on every restart.
- **Uploads are disabled.** Everyone shares the demo tenant, so accepting files would let one visitor serve content to the next. The handbook is ingested at start-up instead, before the server accepts traffic, so RAG works from the first request.
- **Admin stays closed** behind a random token that is never displayed, and a test checks that the public key can't open it.
- **It is the same code, not a fork.** Only configuration differs, and CI boots the image and runs the full walkthrough against it, so the demo can't drift from what is tested.

## Repository layout

```
app/
  api/            FastAPI routers: chat, rag, agents, account, admin, alerts, health
  core/           config, auth (API keys + JWT), rate limiting, metering, embeddings,
                  bounded inference, logging, DI
  gateway/        provider adapters (whole + streaming), routing config, canary rollouts,
                  circuit breaker, router, pricing, fault injection, chat service
  cache/          semantic cache (bruteforce / RediSearch indexes)
  rag/            parsing, chunking, Qdrant store, reranking, retrieval, ingestion, grounded answers
  agents/         the state-graph engine, the research agent's nodes, prompts and run state
  workers/        job records + DLQ, upload payloads, queues (celery / inline), Celery entry point
  observability/  Prometheus metrics, request-ID middleware, Langfuse tracing
  demo.py         public demo mode: shared key, pre-loaded handbook, landing page
config/
  routes.yaml       route aliases → ordered fallback chains, with per-deployment pricing
  routes.demo.yaml  the public demo's routes: mock providers only
tests/unit, tests/integration
infra/k8s/base            kustomize base: API, worker, Redis, Qdrant, Prometheus,
                          Alertmanager, Grafana + their config (rules, dashboard)
infra/k8s/overlays/kind   local kind cluster: NodePorts, Podman-built image, cluster config
scripts/          cluster-up / cluster-down (PowerShell + bash), smoke test, demo walkthrough,
                  GIF recorder, Hugging Face Space deploy
Containerfile       API and worker image for Kubernetes, built with Podman
Containerfile.demo  the one-container public demo
docs/demo.gif       the README walkthrough, generated by scripts/record_gif.py
eval/             retrieval eval: fictional corpus, 48 labelled questions, harness, results
loadtest/         Locust traffic, in-cluster runner, chaos test, autoscaling test, results
```

## How it was built

Eight weekly milestones, following the project spec:

| Week | Milestone | Outcome |
|---|---|---|
| 1 | Skeleton + gateway: FastAPI, Podman image + Kubernetes manifests, health checks, multi-provider fallback | ✅ |
| 2 | Semantic cache, API keys + JWT, per-key token-bucket rate limiting, cost metering | ✅ |
| 3–4 | RAG: ingestion → chunking → embedding → Qdrant → rerank; labelled eval set | ✅ recall@5 **1.000**, MRR **0.990** ([eval](eval/README.md)) |
| 5 | Celery workers: ingestion off the request path, job status, DLQ + backoff | ✅ |
| 6 | Langfuse tracing, Grafana dashboard, alerting | ✅ |
| 7 | Locust load test, breaking point, chaos test | ✅ breaking point found and fixed; 0 failures when a provider dies ([results](loadtest/README.md)) |
| 8 | README, demo GIF, one-container demo, live deployment | ✅ except the live URL: `scripts/deploy_space.py` is ready and waits on a Hugging Face login |
| — | Beyond the roadmap: the agent layer the spec lists as a stretch feature (§4.7) | ✅ [`app/agents`](app/agents): an explicit state graph, published as a diagram |
| — | Beyond the roadmap: autoscaling tied to queue depth (§10) | ✅ [HPAs](infra/k8s/base/autoscaling.yaml) on CPU (API) and ingestion queue depth (workers) |
| — | Beyond the roadmap: canary rollout of provider changes (§10) | ✅ [`app/gateway/rollout.py`](app/gateway/rollout.py): weighted traffic split with automatic rollback |
| — | Beyond the roadmap: streaming responses | ✅ SSE with fallback decided before the first token, and a breaker that waits for the whole stream |

## What I'd do with more time

- Shared circuit-breaker state across replicas.
- Load-test the streaming path: the numbers above are for whole responses, and time to first token
  under load is the figure a streaming client actually feels.
- Hybrid retrieval (BM25 + dense), and an LLM-judged eval of answer faithfulness.
- An eval for the agent: whether planned searches and self-critique actually beat one-shot `/v1/rag/answer` on the labelled set, and what the extra LLM calls buy. It needs a real provider, so the offline test suite can't answer it.
- A model-comparison harness that feeds back into route ordering.
- GPU inference or a smaller reranker, to lower the RAG latency floor.
- A soak test.
- A Helm chart for real clusters.
