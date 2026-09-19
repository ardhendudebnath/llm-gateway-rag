# NexusGate

*A self-hosted gateway that turns "call an LLM" into a production system — provider fallback, semantic caching, retrieval, and cost/latency observability, all behind one API.*

**A production-grade LLM gateway and RAG backend.** NexusGate exposes one OpenAI-compatible API in front of several model providers. It falls back automatically when a provider fails and trips a circuit breaker when one keeps failing. Near-duplicate prompts are answered from a tenant-isolated semantic cache, so they cost nothing. Every request is authenticated, rate-limited, metered for tokens and dollars, and exported as Prometheus metrics.

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
    RT -.-> RAG[RAG orchestrator<br/>Qdrant · rerank]:::todo
    U[Document upload] -.-> Q[Celery workers]:::todo -.-> RAG
    G --> P[Prometheus → Grafana]
    classDef todo stroke-dasharray: 4 4,opacity:0.6
```
<sub>Dashed components are on the roadmap below.</sub>

## Status

| Week | Milestone | State |
|---|---|---|
| 1 | Skeleton + gateway: FastAPI, Podman image + Kubernetes manifests, health checks, multi-provider fallback | ✅ done |
| 2 | Semantic cache, API keys + JWT, per-key token-bucket rate limiting, cost metering | ✅ done |
| 3–4 | RAG: ingestion → chunking → embedding → Qdrant → rerank; P@5/R@5 eval set | ⏳ next |
| 5 | Celery workers: ingestion off the request path, job status, DLQ + backoff | ⏳ |
| 6 | Langfuse tracing, Grafana dashboard, error-budget alert | 🟡 Prometheus metrics + JSON logs done |
| 7 | Locust load test, breaking point, chaos test | 🟡 `chaos` route + fallback tests done |
| 8 | README polish, demo GIF, live deployment | ⏳ |

**Quality:** 98 tests (unit + integration + provider contract tests), **97% coverage**, ruff-clean. No test touches the network or spends API credits. The Kubernetes stack is checked end to end by [`scripts/smoke_test.py`](scripts/smoke_test.py).

## Quickstart

The whole stack (2 API replicas, Redis, Prometheus, Grafana) runs on a local Kubernetes cluster: [kind](https://kind.sigs.k8s.io/) on [Podman](https://podman.io/). No Docker needed.

**Prerequisites:** Podman, kind and kubectl. On Windows: `winget install RedHat.Podman Kubernetes.kind Kubernetes.kubectl` (Podman uses WSL2). If PowerShell blocks the script, run it as `powershell -ExecutionPolicy Bypass -File .\scripts\cluster-up.ps1`.

```bash
cp .env.example .env              # optional: ANTHROPIC_API_KEY / OPENAI_API_KEY for real providers
./scripts/cluster-up.sh           # Windows: .\scripts\cluster-up.ps1
```

That one command creates a rootful Podman machine if needed, then a kind cluster. It builds the image from the [`Containerfile`](Containerfile) with Podman, loads it into kind, creates the Secret from `.env`, and applies the [kustomize overlay](infra/k8s/overlays/kind). Admin and JWT secrets missing from `.env` are generated, and the admin token is printed at the end. Re-running rebuilds the image and rolls the API. `scripts/cluster-down.sh` (or `.ps1`) deletes the cluster.

API docs: http://localhost:8000/docs · Prometheus: http://localhost:9090 · Grafana: http://localhost:3000

Verify the running stack end to end (mock routes only, no API keys, no spend):

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

Because the API is OpenAI-compatible, existing SDKs work unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="ng_...")
client.chat.completions.create(model="mock", messages=[{"role": "user", "content": "hi"}])
```

### Local development (no cluster)

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Windows: .venv\Scripts\pip
pytest --cov
uvicorn app.main:create_app --factory --reload               # needs a Redis on localhost:6379
```

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/chat/completions` | key / JWT | OpenAI-shaped chat; response adds a `nexusgate` block (deployment, attempts, cost, cache hit) |
| `GET /v1/models` | key / JWT | Route aliases usable as `model` |
| `POST /v1/auth/token` | API key | Exchange a key for a short-lived JWT |
| `GET /v1/usage?days=7` | key / JWT | Daily requests, tokens, $ spent, $ saved by cache |
| `DELETE /v1/cache` | key / JWT | Purge the caller's tenant cache |
| `POST/GET/DELETE /v1/admin/keys` | admin token | Issue, list, revoke API keys |
| `GET /v1/admin/providers` | admin token | Route table + live circuit-breaker states |
| `DELETE /v1/admin/cache` | admin token | Purge the whole cache |
| `GET /healthz`, `/readyz`, `/metrics` | none | Liveness, readiness (Redis ping), Prometheus |

Every error uses one envelope: `{"error": {"type": ..., "message": ...}}`. A 429 carries `Retry-After`. A 503 from total provider failure includes each attempt.

## Design decisions

**Fallback lives in our router, not in LiteLLM.** LiteLLM is used only as a provider adapter, with `num_retries=0`. Retry, fallback and circuit breaking are ~150 lines in [`app/gateway/router.py`](app/gateway/router.py), so the behaviour is explicit and unit-tested, and every attempt shows up in the response and in metrics.

**Not every error triggers fallback.** Timeouts, 429s, 5xx and auth failures move on to the next deployment. A provider-side **400** fails fast: another vendor would reject the same malformed request, so the chain isn't burned. It also doesn't count against the breaker, because a bad request isn't an outage.

**Circuit breaker.** A deployment opens after N consecutive failures. After the cooldown it goes half-open and lets **exactly one** probe through; the probe's result either closes the breaker or reopens it for a fresh cooldown. State is kept per replica on purpose. Each replica learns about an outage within N requests, which is cheaper than a Redis round-trip on every call.

**The cache matches only the last user message semantically.** Tenant, route, temperature, max_tokens, system prompt and earlier turns are all hashed into an exact-match namespace. That way "answer in French" and "answer in English" can never share a cached answer, however close their embeddings are, and tenant A can never be served tenant B's response. Entries have a TTL, each namespace has a size cap, and there are purge endpoints. If the cache fails, requests are served uncached instead of erroring.

**Two vector-index backends behind one storage layout.** `redisearch` (an HNSW index in redis-stack) is what the Kubernetes stack runs. `bruteforce` (a numpy dot product) needs no extra infrastructure and is what the tests use.

**Rate limiting is one Lua script.** The token-bucket refill-and-take runs atomically inside Redis, so replicas sharing a Redis never overspend a bucket. A test fires 50 concurrent requests at a 10-token bucket and asserts that exactly 10 get through.

**Only a SHA-256 of each API key is stored.** A Redis dump contains no usable credentials. JWTs are checked against the key record on every request, so revoking a key immediately kills its outstanding tokens. In `prod` the app refuses to start with the default secrets.

**Kubernetes, built with Podman.** The image is an OCI image built from a `Containerfile`, and the stack is plain kustomize: a cluster-agnostic [base](infra/k8s/base) plus a [kind overlay](infra/k8s/overlays/kind) that adds NodePorts and the locally built image.
- **Replicas:** the API runs 2 replicas. Rate limits, the cache and metering live in Redis, so they hold across replicas.
- **Per-pod scraping:** Prometheus discovers every API pod through a headless Service's DNS records. Each replica keeps its own counters, and scraping the load-balanced Service would sample a random pod each time.
- **Hardened pods:** they run as non-root with a read-only root filesystem and all capabilities dropped. An init container waits for Redis instead of letting the API crash-loop.
- **Secrets:** they never enter git. The cluster-up script creates the Secret from `.env`. The pods run with `NEXUSGATE_ENV=prod`, which refuses default secrets.

## Repository layout

```
app/
  api/            FastAPI routers: chat, account, admin, health
  core/           config, auth (API keys + JWT), rate limiting, metering, logging, DI container
  gateway/        provider adapters, routing config, circuit breaker, router, pricing, chat service
  cache/          embedders + semantic cache (bruteforce / RediSearch indexes)
  rag/            (weeks 3–4)
  workers/        (week 5)
  observability/  Prometheus metrics, request-ID middleware
config/routes.yaml  route aliases → ordered fallback chains, with per-deployment pricing
tests/unit, tests/integration
infra/k8s/base            kustomize base: API, Redis, Prometheus, Grafana (+ their config)
infra/k8s/overlays/kind   local kind cluster: NodePorts, Podman-built image, cluster config
scripts/          cluster-up / cluster-down (PowerShell + bash), end-to-end smoke test
Containerfile     API image, built with Podman
eval/             retrieval eval set + script (weeks 3–4)
```

## What I'd do with more time

Shared circuit-breaker state across replicas; streaming (SSE) responses; per-tenant Qdrant collections with a test proving tenant isolation; a model-comparison harness that feeds back into route ordering; a HorizontalPodAutoscaler driven by queue depth; a Helm chart for real clusters.
