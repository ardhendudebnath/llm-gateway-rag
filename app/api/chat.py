from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from app.api.deps import get_services, rate_limited
from app.core.container import Services
from app.core.security import Principal
from app.gateway.schemas import ChatChunk, ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # tell nginx not to buffer, or chunks arrive in one lump
}


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest,
    response: Response,
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
):
    if body.stream:
        return await _stream_completions(principal, body, services)
    result = await services.chat.complete(principal, body)
    response.headers["X-NexusGate-Cache"] = "hit" if result.nexusgate.cached else "miss"
    if result.nexusgate.deployment:
        response.headers["X-NexusGate-Deployment"] = result.nexusgate.deployment
    return result


async def _stream_completions(
    principal: Principal, body: ChatRequest, services: Services
) -> StreamingResponse:
    """Server-sent events, in OpenAI's format.

    The first chunk is pulled here, before the response exists. That is deliberate: routing,
    retries and fallback all happen while producing it, so a total failure is still an ordinary
    exception and the handlers turn it into a 503 or a 400. Once the response has started, the
    status line is already on the wire and the only honest place left to report a failure is
    inside the stream (the last chunk's `nexusgate.interrupted`).
    """
    chunks = services.chat.stream(principal, body).__aiter__()
    first = await anext(chunks)  # may raise: that is the point

    async def events() -> AsyncIterator[str]:
        yield _sse(first)
        async for chunk in chunks:
            yield _sse(chunk)
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


def _sse(chunk: ChatChunk) -> str:
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


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
