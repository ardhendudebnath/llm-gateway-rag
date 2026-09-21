"""Prometheus metrics. Label sets are kept small and bounded (route templates, deployment names)
so cardinality cannot grow with user input."""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "nexusgate_http_requests_total", "HTTP requests handled", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "nexusgate_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

LLM_CALLS = Counter(
    "nexusgate_llm_calls_total", "Provider call attempts by outcome", ["deployment", "outcome"]
)
LLM_LATENCY = Histogram(
    "nexusgate_llm_call_duration_seconds",
    "Provider call latency (successful calls)",
    ["deployment"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
LLM_TOKENS = Counter("nexusgate_llm_tokens_total", "Tokens consumed", ["deployment", "kind"])
LLM_COST = Counter("nexusgate_llm_cost_usd_total", "Provider spend in USD", ["deployment"])
LLM_FALLBACKS = Counter(
    "nexusgate_llm_fallbacks_total", "Requests served by a non-primary deployment", ["route"]
)
LLM_ALL_FAILED = Counter(
    "nexusgate_llm_all_providers_failed_total", "Requests where every deployment failed", ["route"]
)
BREAKER_OPEN = Gauge(
    "nexusgate_circuit_open", "1 if the deployment's circuit breaker is open", ["deployment"]
)

CACHE_LOOKUPS = Counter("nexusgate_cache_lookups_total", "Semantic cache lookups", ["result"])
CACHE_COST_SAVED = Counter(
    "nexusgate_cache_cost_saved_usd_total", "Provider spend avoided by cache hits"
)

RATE_LIMITED = Counter("nexusgate_rate_limited_total", "Requests rejected by the rate limiter")

INFERENCE_WAITING = Gauge(
    "nexusgate_inference_waiting", "Requests waiting for a model inference slot", ["model"]
)
INFERENCE_SHED = Counter(
    "nexusgate_inference_shed_total",
    "Inferences refused because too many were already waiting (backpressure)",
    ["model"],
)
DEGRADED = Counter(
    "nexusgate_degraded_total",
    "Requests served in a degraded mode instead of failing",
    ["mode"],  # cache_skipped | rerank_skipped
)

ALERTS_RECEIVED = Counter(
    "nexusgate_alerts_received_total",
    "Alerts delivered by Alertmanager to the gateway's webhook",
    ["alertname", "status"],
)

RAG_DOCUMENTS_INGESTED = Counter("nexusgate_rag_documents_ingested_total", "Documents ingested")
RAG_CHUNKS_INGESTED = Counter("nexusgate_rag_chunks_ingested_total", "Chunks embedded and stored")
RAG_INGEST_LATENCY = Histogram(
    "nexusgate_rag_ingest_duration_seconds",
    "Parse + chunk + embed + store time per document",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
RAG_STAGE_LATENCY = Histogram(
    "nexusgate_rag_stage_duration_seconds",
    "Retrieval latency by stage",
    ["stage"],  # embed_query | vector_search | rerank
    # Up to 10 s: the first load test's tail sat above the old 2.5 s top bucket, hiding its size.
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
# Job counters live in Redis (the worker process has no /metrics endpoint of its own) and are
# published as gauges when the API is scraped, so they are correct whichever process ran the job.
RAG_QUEUE_DEPTH = Gauge(
    "nexusgate_rag_queue_depth", "Ingestion jobs waiting in the queue", ["queue"]
)
RAG_DLQ_DEPTH = Gauge("nexusgate_rag_dead_letter_depth", "Jobs in the dead-letter queue")
RAG_JOB_TOTALS = Gauge(
    "nexusgate_rag_job_totals", "Ingestion jobs by outcome, cumulative in Redis", ["outcome"]
)
RAG_JOB_SECONDS = Gauge(
    "nexusgate_rag_job_processing_seconds_total", "Total time spent running ingestion jobs"
)

RAG_EMPTY_RETRIEVALS = Counter(
    "nexusgate_rag_empty_retrievals_total", "Answer requests where no passage matched (no LLM call)"
)
