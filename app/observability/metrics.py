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
