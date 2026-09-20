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

    # --- gateway resilience ---
    provider_timeout_seconds: float = 30.0
    breaker_failure_threshold: int = 3
    breaker_cooldown_seconds: float = 30.0

    # --- metering ---
    usage_retention_days: int = 90

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()
