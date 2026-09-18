"""Caller-scoped endpoints: token exchange, usage, cache purge for the caller's tenant."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.api.deps import authenticate_api_key, get_services, rate_limited
from app.core.container import Services
from app.core.metering import DailyUsage
from app.core.security import Principal

router = APIRouter(prefix="/v1", tags=["account"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UsageResponse(BaseModel):
    key_id: str
    tenant_id: str
    days: list[DailyUsage]
    totals: DailyUsage


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
    )
    return UsageResponse(
        key_id=principal.key_id, tenant_id=principal.tenant_id, days=rows, totals=totals
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
