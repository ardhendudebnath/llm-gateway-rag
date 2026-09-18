from fastapi import APIRouter, Depends, Response

from app.api.deps import get_services, rate_limited
from app.core.container import Services
from app.core.security import Principal
from app.gateway.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest,
    response: Response,
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> ChatResponse:
    result = await services.chat.complete(principal, body)
    response.headers["X-NexusGate-Cache"] = "hit" if result.nexusgate.cached else "miss"
    if result.nexusgate.deployment:
        response.headers["X-NexusGate-Deployment"] = result.nexusgate.deployment
    return result


@router.get("/models", summary="Route aliases usable as `model`")
async def list_models(
    _: Principal = Depends(rate_limited), services: Services = Depends(get_services)
) -> dict:
    return {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "owned_by": "nexusgate"}
            for alias in services.router.routes
        ],
    }
