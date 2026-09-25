"""Caller-scoped endpoints: token exchange, usage, cache purge for the caller's tenant."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import authenticate_api_key, get_services, rate_limited
from app.core.container import Services
from app.core.metering import DailyUsage
from app.core.security import Principal

router = APIRouter(prefix="/v1", tags=["account"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class CostPer1k(BaseModel):
    """Spend per 1,000 requests of each kind, and blended across all of them.

    The blended figure alone hides which lever moved it: a better cache hit rate and a shift onto
    self-hosted capacity both lower it, and they cost very different things to arrange.
    """

    cache_hit: float = Field(default=0.0, description="Always 0: a hit makes no provider call.")
    provider_call: float = Field(description="Paid providers only.")
    self_hosted: float = Field(description="Usually 0: the GPU is paid for by the hour, not here.")
    blended: float = Field(description="Total spend over total requests, x1000.")
    requests: dict[str, int] = Field(description="How many requests of each kind it averages over.")


class UsageResponse(BaseModel):
    key_id: str
    tenant_id: str
    days: list[DailyUsage]
    totals: DailyUsage
    cost_per_1k_usd: CostPer1k


@router.post("/auth/token", response_model=TokenResponse, summary="Exchange an API key for a JWT")
async def issue_token(
    principal: Principal = Depends(authenticate_api_key),
    services: Services = Depends(get_services),
) -> TokenResponse:
    record = await services.keys.get(principal.key_id)
    return TokenResponse(
        access_token=services.tokens.issue(record), expires_in=services.tokens.ttl_seconds
    )


@router.get("/usage", response_model=UsageResponse, summary="Requests, tokens and spend per day")
async def usage(
    days: int = Query(default=7, ge=1, le=90),
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> UsageResponse:
    rows = await services.meter.daily(principal.key_id, days)
    totals = DailyUsage(
        date=rows[-1].date,
        requests=sum(r.requests for r in rows),
        cache_hits=sum(r.cache_hits for r in rows),
        prompt_tokens=sum(r.prompt_tokens for r in rows),
        completion_tokens=sum(r.completion_tokens for r in rows),
        cost_usd=round(sum(r.cost_usd for r in rows), 8),
        cost_saved_usd=round(sum(r.cost_saved_usd for r in rows), 8),
        self_hosted_requests=sum(r.self_hosted_requests for r in rows),
        self_hosted_cost_usd=round(sum(r.self_hosted_cost_usd for r in rows), 8),
    )
    return UsageResponse(
        key_id=principal.key_id,
        tenant_id=principal.tenant_id,
        days=rows,
        totals=totals,
        cost_per_1k_usd=_cost_per_1k(totals),
    )


def _per_1k(cost: float, requests: int) -> float:
    return round(cost / requests * 1000, 6) if requests else 0.0


def _cost_per_1k(totals: DailyUsage) -> CostPer1k:
    return CostPer1k(
        cache_hit=0.0,
        provider_call=_per_1k(totals.provider_cost_usd, totals.provider_requests),
        self_hosted=_per_1k(totals.self_hosted_cost_usd, totals.self_hosted_requests),
        blended=_per_1k(totals.cost_usd, totals.requests),
        requests={
            "cache_hit": totals.cache_hits,
            "provider_call": totals.provider_requests,
            "self_hosted": totals.self_hosted_requests,
        },
    )


@router.delete("/cache", summary="Purge the semantic cache for the caller's tenant")
async def purge_tenant_cache(
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> dict:
    if services.cache is None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="semantic cache is disabled")
    deleted = await services.cache.purge_tenant(principal.tenant_id)
    return {"tenant_id": principal.tenant_id, "entries_deleted": deleted}
