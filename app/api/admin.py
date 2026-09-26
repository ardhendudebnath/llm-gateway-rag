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
    """`breakers` is this replica's view; `shared` is what every replica sees, read from Redis —
    including how much cooldown an open circuit has left."""
    router_ = services.router
    cluster = router_.cluster
    return {
        "routes": {
            alias: [d.name for d in chain] for alias, chain in router_.config.routes.items()
        },
        "breakers": router_.breaker_states(),
        "shared": await cluster.states(router_.breakers) if cluster else None,
    }


class CanaryRequest(BaseModel):
    config: str = Field(
        min_length=1, description="A complete route table, YAML or JSON, same shape as routes.yaml."
    )
    weight: int = Field(default=10, ge=0, le=100, description="Percent of traffic to send to it.")
    note: str | None = Field(default=None, max_length=200, description="Why this is going out.")


class WeightRequest(BaseModel):
    weight: int = Field(ge=0, le=100)


async def _rollout_view(services: Services) -> dict:
    rollout = services.rollout
    stable_version, stable_config = await rollout.stable()
    canary = await rollout.canary()
    view = {
        "stable": {
            "version": stable_version,
            "routes": {a: [d.name for d in c] for a, c in stable_config.routes.items()},
            "stats": await rollout.stats(stable_version),
        },
        "canary": None,
    }
    if canary is not None:
        state, config = canary
        stats = await rollout.stats(state.version)
        errors = stats["errors"] / stats["requests"] if stats["requests"] else 0.0
        view["canary"] = {
            **vars(state),
            "active": state.active,
            "routes": {a: [d.name for d in c] for a, c in config.routes.items()},
            "stats": {**stats, "error_rate": round(errors, 4)},
            "rolls_back_above": rollout.settings.max_error_rate,
            "after_requests": rollout.settings.min_requests,
        }
    return view


@router.get("/routes", summary="The live route table, and any canary being rolled out")
async def routes(services: Services = Depends(get_services)) -> dict:
    return await _rollout_view(services)


@router.put("/routes/canary", summary="Send a share of traffic to a candidate route table")
async def publish_canary(body: CanaryRequest, services: Services = Depends(get_services)) -> dict:
    """The config is parsed and validated before anything is stored, so a malformed table is a
    400 here rather than a failed request later. Replicas pick it up within a second."""
    try:
        await services.rollout.publish(body.config, body.weight, body.note)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"invalid route table: {e}") from e
    return await _rollout_view(services)


@router.post("/routes/canary/weight", summary="Change how much traffic the canary takes")
async def set_canary_weight(
    body: WeightRequest, services: Services = Depends(get_services)
) -> dict:
    if await services.rollout.set_weight(body.weight) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no canary is published")
    return await _rollout_view(services)


@router.post("/routes/canary/promote", summary="Make the canary the route table for all traffic")
async def promote_canary(services: Services = Depends(get_services)) -> dict:
    if await services.rollout.promote() is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no canary is published")
    return await _rollout_view(services)


@router.delete("/routes/canary", summary="Withdraw the canary; all traffic returns to stable")
async def discard_canary(services: Services = Depends(get_services)) -> dict:
    if not await services.rollout.discard():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no canary is published")
    return await _rollout_view(services)


class FaultRequest(BaseModel):
    failure_rate: float = Field(ge=0, le=1, description="Share of calls to fail, 0 to 1.")


def _faults(services: Services):
    if services.router.faults is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="fault injection is disabled (set NEXUSGATE_FAULT_INJECTION_ENABLED=true)",
        )
    return services.router.faults


@router.get("/faults", summary="Faults currently injected, per deployment")
async def list_faults(services: Services = Depends(get_services)) -> dict:
    return {"faults": await _faults(services).rates()}


@router.put("/faults/{deployment}", summary="Make a deployment fail (chaos testing)")
async def inject_fault(
    deployment: str, body: FaultRequest, services: Services = Depends(get_services)
) -> dict:
    faults = _faults(services)
    if deployment not in services.router.config.deployments:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such deployment")
    await faults.set(deployment, body.failure_rate)
    return {"deployment": deployment, "failure_rate": body.failure_rate}


@router.delete(
    "/faults/{deployment}", status_code=status.HTTP_204_NO_CONTENT, summary="Stop a fault"
)
async def clear_fault(deployment: str, services: Services = Depends(get_services)) -> None:
    if not await _faults(services).clear(deployment):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no fault on that deployment")


@router.get("/dead-letters", summary="Ingestion jobs that failed for good")
async def dead_letters(
    limit: int = Query(default=50, ge=1, le=500), services: Services = Depends(get_services)
) -> dict:
    return {
        "depth": await services.rag.dead_letters.depth(),
        "jobs": await services.rag.dead_letters.list(limit),
    }


@router.delete("/dead-letters", summary="Clear the dead-letter queue")
async def purge_dead_letters(services: Services = Depends(get_services)) -> dict:
    return {"entries_deleted": await services.rag.dead_letters.purge()}


@router.delete("/cache", summary="Purge the entire semantic cache")
async def purge_all_cache(services: Services = Depends(get_services)) -> dict:
    if services.cache is None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="semantic cache is disabled")
    return {"entries_deleted": await services.cache.purge_all()}
