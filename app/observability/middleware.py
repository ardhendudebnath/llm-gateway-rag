"""ASGI middleware: request-ID propagation, access logging and HTTP metrics."""

import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_var
from app.observability import metrics

log = logging.getLogger("nexusgate.access")

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_QUIET_PATHS = {"/metrics", "/healthz", "/readyz"}


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        # Honour a caller-supplied ID (for cross-service tracing) only if it is safe to log.
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        status = 500
        start = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            elapsed = time.perf_counter() - start
            route = scope.get("route")
            # Route *template* ("/v1/admin/keys/{key_id}"), never the raw path: bounded cardinality.
            route_label = getattr(route, "path", "unmatched")
            metrics.HTTP_REQUESTS.labels(scope["method"], route_label, str(status)).inc()
            metrics.HTTP_LATENCY.labels(scope["method"], route_label).observe(elapsed)
            if scope["path"] not in _QUIET_PATHS:
                log.info(
                    "request",
                    extra={
                        "method": scope["method"],
                        "path": scope["path"],
                        "status": status,
                        "duration_ms": round(elapsed * 1000, 2),
                    },
                )
            request_id_var.reset(token)
