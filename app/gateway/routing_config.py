"""Route table: a model alias maps to an ordered fallback chain of deployments."""

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Pricing(BaseModel):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class Deployment(BaseModel):
    name: str
    provider: str = Field(description="Adapter key: 'litellm' or 'mock'.")
    model: str = Field(description="Provider-specific model id, e.g. 'anthropic/claude-haiku-4-5'.")
    api_base: str | None = None
    timeout_seconds: float | None = None
    max_retries: int = Field(default=0, ge=0, le=5)
    pricing: Pricing | None = None
    self_hosted: bool = Field(
        default=False,
        description="You pay for the hardware, not per token. Metered separately, because a "
        "zero price is otherwise indistinguishable from a free API.",
    )
    # Free-form adapter options (the mock provider reads latency/failure settings from here).
    options: dict = Field(default_factory=dict)


class RoutingConfig(BaseModel):
    routes: dict[str, list[Deployment]]

    @model_validator(mode="after")
    def _validate(self) -> "RoutingConfig":
        seen: dict[str, Deployment] = {}
        for alias, chain in self.routes.items():
            if not chain:
                raise ValueError(f"route '{alias}' has no deployments")
            for dep in chain:
                # Breakers are keyed by deployment name, so a name must mean one thing.
                if seen.setdefault(dep.name, dep) != dep:
                    raise ValueError(f"deployment '{dep.name}' defined twice with different config")
        return self

    @property
    def deployments(self) -> dict[str, Deployment]:
        return {d.name: d for chain in self.routes.values() for d in chain}

    @classmethod
    def from_yaml(cls, path: Path) -> "RoutingConfig":
        with path.open(encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    @classmethod
    def from_text(cls, text: str) -> "RoutingConfig":
        """Parse a route table posted to the admin API, rather than read from disk."""
        return cls.model_validate(yaml.safe_load(text))

    def fingerprint(self) -> str:
        """Short stable id for a route table, so two versions can be told apart in logs."""
        canonical = self.model_dump_json(exclude_defaults=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]
