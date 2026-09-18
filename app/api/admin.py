"""Operator endpoints, guarded by ``X-Admin-Token``."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import get_services, require_admin
from app.core.container import Services
from app.core.security import ApiKeyRecord

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Depends(require_admin)])


class CreateKeyRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    name: str = Field(min_length=1, max_length=128)
    rate_limit_capacity: int | None = Field(default=None, gt=0)
    rate_limit_refill_per_sec: float | None = Field(default=None, gt=0)


class CreateKeyResponse(BaseModel):
    api_key: str = Field(description="Shown once. Store it now; only a hash is kept.")
    record: ApiKeyRecord


@router.post("/keys", response_model=CreateKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_key(body: CreateKeyRequest, services: Services = Depends(get_services)):
    raw_key, record = await services.keys.create(
        body.tenant_id, body.name, body.rate_limit_capacity, body.rate_limit_refill_per_sec
    )
    return CreateKeyResponse(api_key=raw_key, record=record)


@router.get("/keys", response_model=list[ApiKeyRecord])
async def list_keys(tenant_id: str = Query(...), services: Services = Depends(get_services)):
    return await services.keys.list_for_tenant(tenant_id)


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(key_id: str, services: Services = Depends(get_services)) -> None:
    if not await services.keys.revoke(key_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such key")


@router.get("/providers", summary="Circuit-breaker state per deployment")
async def providers(services: Services = Depends(get_services)) -> dict:
    return {
        "routes": {
            alias: [d.name for d in chain] for alias, chain in services.router.config.routes.items()
        },
        "breakers": services.router.breaker_states(),
    }


@router.delete("/cache", summary="Purge the entire semantic cache")
async def purge_all_cache(services: Services = Depends(get_services)) -> dict:
    if services.cache is None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="semantic cache is disabled")
    return {"entries_deleted": await services.cache.purge_all()}
