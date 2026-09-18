from fastapi import APIRouter, Depends, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.deps import get_services
from app.core.container import Services

router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="Liveness: the process is up")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness: dependencies are reachable")
async def readyz(response: Response, services: Services = Depends(get_services)) -> dict:
    checks: dict[str, str] = {}
    try:
        await services.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {type(e).__name__}"
    ready = all(v == "ok" for v in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ready else "not_ready", "checks": checks}


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
