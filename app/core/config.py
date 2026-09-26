"""Runtime configuration, loaded from environment variables (prefix ``NEXUSGATE_``) or ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEXUSGATE_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    redis_url: str = "redis://localhost:6379/0"
    routes_file: Path = Path("config/routes.yaml")

    # --- auth ---
    admin_token: SecretStr = SecretStr("change-me-admin-token")
    jwt_secret: SecretStr = SecretStr("change-me-jwt-secret-at-least-32-bytes")
    jwt_algorithm: str = "HS256"
    jwt_ttl_seconds: int = 3600

    # --- rate limiting (token bucket defaults; overridable per API key) ---
    rate_limit_capacity: int = 60
    rate_limit_refill_per_sec: float = 1.0

    # --- semantic cache ---
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 86_400
    cache_index_backend: Literal["bruteforce", "redisearch"] = "bruteforce"
    cache_max_entries_per_namespace: int = 1_000
    # --- embeddings (shared by the semantic cache and RAG) ---
    embedding_backend: Literal["hash", "fastembed"] = "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: str | None = None  # pre-downloaded fastembed weights (baked into the image)
    # ONNX Runtime sizes each session's thread pool to every core, and the embedder's and
    # reranker's spinning pools then fight over the CPU. Measured on a 24-core host: reranking 20
    # chunks took 929 ms with default pools and 242 ms with 4 threads per model.
    model_threads: int | None = Field(default=4, gt=0)
    # Bounded inference (app/core/concurrency.py): how many run at once per model, and how many
    # may wait before requests are shed. Unbounded, the load test OOM-killed the API at 50 users.
    embed_max_concurrency: int = Field(default=2, gt=0)
    embed_max_queue: int = Field(default=64, ge=0)
    rerank_max_concurrency: int = Field(default=2, gt=0)
    rerank_max_queue: int = Field(default=8, ge=0)
    # Reranking is skipped if it can't start within this budget (0 = wait as long as the queue
    # allows). Bounds RAG tail latency under load; vector order is the fallback.
    rerank_max_wait_ms: float = Field(default=250, ge=0)

    # --- RAG ---
    qdrant_url: str = ":memory:"  # in-process Qdrant for dev/tests; http://qdrant:6333 in k8s
    qdrant_api_key: SecretStr | None = None
    rag_collection: str = "nexusgate_chunks"
    rag_chunker: Literal["fixed", "sentence", "structured"] = "structured"
    rag_chunk_max_words: int = Field(default=180, gt=0)
    rag_chunk_overlap_words: int = Field(default=40, ge=0)
    rag_reranker: Literal["none", "cross-encoder"] = "none"
    rag_reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rag_rerank_candidates: int = Field(default=20, gt=0)
    rag_max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)

    # --- background jobs ---
    worker_mode: Literal["inline", "celery"] = "inline"  # inline runs jobs in the API process
    celery_broker_url: str | None = None  # defaults to redis_url, so queue depth is readable
    ingest_queue: str = "ingest"
    job_max_attempts: int = Field(default=3, gt=0)
    job_retry_backoff_seconds: float = Field(default=2.0, gt=0)
    job_retention_seconds: int = Field(default=86_400, gt=0)
    upload_payload_ttl_seconds: int = Field(default=3_600, gt=0)
    job_visibility_timeout_seconds: int = Field(default=3_600, gt=0)

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    # Chunks per embedding batch. Bounds the memory one document can demand: a worker was
    # OOM-killed embedding a large file's chunks in a single call (see loadtest/RESULTS.md).
    embed_batch_size: int = Field(default=32, gt=0)

    # --- agent (app/agents): the step budget is the graph's safety net against a cycle ---
    agent_max_steps: int = Field(default=12, gt=0)

    # --- canary rollouts of the route table (app/gateway/rollout.py) ---
    rollout_refresh_seconds: float = Field(default=1.0, gt=0)  # how fast a replica sees a change
    canary_min_requests: int = Field(default=20, gt=0)  # before its error rate means anything
    canary_max_error_rate: float = Field(default=0.10, gt=0, le=1)  # above this it withdraws itself

    # --- gateway resilience ---
    provider_timeout_seconds: float = 30.0
    breaker_failure_threshold: int = 3
    breaker_cooldown_seconds: float = 30.0
    # Circuit state shared across replicas (app/gateway/breaker_cluster.py). Off means each replica
    # decides alone, which needs failure_threshold failures *per replica* to stop a dead provider.
    breaker_shared: bool = True
    breaker_refresh_seconds: float = Field(default=1.0, gt=0)  # how fast a replica adopts the state
    breaker_failure_window_seconds: float = Field(default=60.0, gt=0)  # pooled failures expire
    breaker_probe_seconds: float = Field(default=5.0, gt=0)  # one replica's claim on the probe
    # Chaos testing: lets an admin make a deployment fail on demand. Off unless asked for.
    fault_injection_enabled: bool = False

    # --- metering ---
    usage_retention_days: int = 90

    # --- tracing (optional; without Langfuse keys the gateway traces nothing) ---
    tracing_enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_environment: str | None = None  # defaults to `env`

    # --- public demo (app/demo.py): shared rate-limited key, offline routes, uploads disabled ---
    demo_mode: bool = False
    demo_rate_limit_capacity: int = Field(default=20, gt=0)
    demo_rate_limit_refill_per_sec: float = Field(default=0.5, gt=0)
    demo_corpus_dir: Path = Path("eval/corpus")

    @property
    def tracing_configured(self) -> bool:
        return bool(self.tracing_enabled and self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()
