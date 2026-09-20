import logging

from fastapi import APIRouter, Depends, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.deps import get_services
from app.core.container import Services

log = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="Liveness: the process is up")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness: dependencies are reachable")
async def readyz(response: Response, services: Services = Depends(get_services)) -> dict:
    checks: dict[str, str] = {}
    for name, probe in (("redis", services.redis.ping), ("qdrant", services.rag.store.ping)):
        try:
            await probe()
            checks[name] = "ok"
        except Exception as e:
            checks[name] = f"error: {type(e).__name__}"
    ready = all(v == "ok" for v in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ready else "not_ready", "checks": checks}


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(services: Services = Depends(get_services)) -> Response:
    try:
        # Queue depth and job counters live in Redis; read them at scrape time.
        await services.rag.jobs.refresh_metrics(services.settings.ingest_queue)
    except Exception:
        log.exception("could not refresh job metrics; serving the rest")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
