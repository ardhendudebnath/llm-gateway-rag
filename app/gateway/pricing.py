"""Per-request cost calculation.

Prices declared in routes.yaml win: they are explicit, reviewable, and deterministic in tests.
Otherwise fall back to LiteLLM's community-maintained price map; unknown models cost 0 and log once.
"""

import logging
from functools import cache

from app.gateway.routing_config import Deployment
from app.gateway.schemas import Usage

log = logging.getLogger(__name__)


def compute_cost(deployment: Deployment, usage: Usage) -> float:
    if deployment.pricing is not None:
        p = deployment.pricing
        return (
            usage.prompt_tokens * p.input_per_mtok + usage.completion_tokens * p.output_per_mtok
        ) / 1_000_000
    return _litellm_cost(deployment.model, usage.prompt_tokens, usage.completion_tokens)


def _litellm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        return float(prompt_cost + completion_cost)
    except Exception:
        _warn_unpriced(model)
        return 0.0


@cache
def _warn_unpriced(model: str) -> None:
    log.warning("no pricing for model; cost recorded as 0", extra={"model": model})
