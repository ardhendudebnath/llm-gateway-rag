"""Runtime configuration, loaded from environment variables (prefix ``NEXUSGATE_``) or ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
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
    embedding_backend: Literal["hash", "fastembed"] = "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

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
