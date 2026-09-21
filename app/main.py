"""Application factory. Run with: ``uvicorn app.main:create_app --factory``."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api import account, admin, alerts, chat, health, rag
from app.core.concurrency import OverloadedError
from app.core.config import Settings, get_settings
from app.core.container import Services, build_services
from app.core.logging import configure_logging
from app.gateway.router import AllProvidersFailedError, ClientRequestError, UnknownModelError
from app.observability.middleware import RequestContextMiddleware


def _error(status_code: int, type_: str, message: str, headers=None, **extra) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": type_, "message": message, **extra}},
        status_code=status_code,
        headers=headers,
    )


def _install_error_handlers(app: FastAPI) -> None:
    """Every error uses one OpenAI-style envelope: {"error": {"type", "message", ...}}."""

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return _error(422, "invalid_request", "request validation failed", details=exc.errors())

    @app.exception_handler(UnknownModelError)
    async def _unknown_model(_: Request, exc: UnknownModelError):
        return _error(404, "model_not_found", f"no route named '{exc}'")

    @app.exception_handler(ClientRequestError)
    async def _client(_: Request, exc: ClientRequestError):
        return _error(400, "provider_rejected_request", str(exc))

    @app.exception_handler(OverloadedError)
    async def _overloaded(_: Request, exc: OverloadedError):
        # Backpressure: a fast, retryable refusal instead of queueing into a timeout.
        return _error(503, "overloaded", str(exc), headers={"Retry-After": "1"})

    @app.exception_handler(AllProvidersFailedError)
    async def _all_failed(_: Request, exc: AllProvidersFailedError):
        return _error(
            503,
            "all_providers_failed",
            str(exc),
            headers={"Retry-After": "5"},
            attempts=[a.model_dump() for a in exc.attempts],
        )


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = getattr(app.state, "services", None) is None
        if owned:
            app.state.services = await build_services(settings)
        try:
            yield
        finally:
            if owned:
                await app.state.services.aclose()

    app = FastAPI(
        title="NexusGate",
        version=__version__,
        description="LLM gateway and RAG backend: multi-provider routing with fallback, semantic "
        "caching, per-key rate limiting, cost metering, and document retrieval with reranking.",
        lifespan=lifespan,
    )
    if services is not None:
        app.state.services = services

    app.add_middleware(RequestContextMiddleware)
    _install_error_handlers(app)
    for module in (health, chat, rag, account, admin, alerts):
        app.include_router(module.router)
    return app
