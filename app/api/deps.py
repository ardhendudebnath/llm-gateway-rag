"""FastAPI dependencies: service access, authentication, rate limiting, admin guard."""

from fastapi import Depends, Header, HTTPException, Request, Response, status

from app.core.container import Services
from app.core.security import (
    KEY_PREFIX,
    InvalidCredentials,
    Principal,
    constant_time_equals,
    principal_from,
)
from app.observability import metrics


def get_services(request: Request) -> Services:
    return request.app.state.services


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail=message, headers={"WWW-Authenticate": "Bearer"}
    )


def _credentials(request: Request) -> tuple[str | None, str | None]:
    """Return (api_key, jwt) — exactly one is set if credentials were supplied."""
    if api_key := request.headers.get("x-api-key"):
        return api_key, None
    auth = request.headers.get("authorization", "")
    scheme, _, value = auth.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None, None
    value = value.strip()
    # OpenAI SDKs send the API key as a bearer token; accept it so they work unmodified.
    return (value, None) if value.startswith(KEY_PREFIX) else (None, value)


async def authenticate(request: Request, services: Services = Depends(get_services)) -> Principal:
    api_key, token = _credentials(request)
    try:
        if api_key is not None:
            record = await services.keys.verify(api_key)
        elif token is not None:
            claims = services.tokens.decode(token)
            record = await services.keys.get(claims["sub"])
            if record is None:
                raise InvalidCredentials("API key behind this token was revoked")
        else:
            raise _unauthorized("missing credentials: send X-API-Key or Authorization: Bearer")
    except InvalidCredentials as e:
        raise _unauthorized(str(e)) from e
    return principal_from(record)


async def authenticate_api_key(
    request: Request, services: Services = Depends(get_services)
) -> Principal:
    """Like ``authenticate`` but rejects JWTs — used by the token-exchange endpoint."""
    api_key, _ = _credentials(request)
    if api_key is None:
        raise _unauthorized("an API key is required to obtain a token")
    try:
        return principal_from(await services.keys.verify(api_key))
    except InvalidCredentials as e:
        raise _unauthorized(str(e)) from e


async def rate_limited(
    response: Response,
    principal: Principal = Depends(authenticate),
    services: Services = Depends(get_services),
) -> Principal:
    settings = services.settings
    decision = await services.limiter.hit(
        principal.key_id,
        capacity=principal.rate_limit_capacity or settings.rate_limit_capacity,
        refill_per_sec=principal.rate_limit_refill_per_sec or settings.rate_limit_refill_per_sec,
    )
    headers = {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
    }
    if not decision.allowed:
        metrics.RATE_LIMITED.inc()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={**headers, "Retry-After": str(max(1, decision.retry_after_seconds))},
        )
    response.headers.update(headers)
    return principal


async def require_admin(
    x_admin_token: str | None = Header(default=None),
    services: Services = Depends(get_services),
) -> None:
    expected = services.settings.admin_token.get_secret_value()
    if x_admin_token is None or not constant_time_equals(x_admin_token, expected):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin token required")
