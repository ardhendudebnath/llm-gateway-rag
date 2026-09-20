"""Alertmanager webhook receiver.

Alertmanager posts firing and resolved alerts here; each one is logged as structured JSON and
counted, and the last few are kept in Redis for `GET /v1/admin/alerts`. Pointing alerts at Slack,
Discord or PagerDuty instead is a receiver change in Alertmanager's config, not a code change, but
shipping a working end-to-end path means the alert rules can be demonstrated without an external
account.

Authentication is HTTP Basic with the admin token as the password (Alertmanager reads it from a
mounted file), because Alertmanager cannot send our `X-Admin-Token` header.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from app.api.deps import get_services, require_admin
from app.core.container import Services
from app.core.security import constant_time_equals
from app.observability import metrics

log = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])
basic = HTTPBasic(auto_error=False)

RECENT_KEY = "alerts:recent"
RECENT_MAX = 100


class Alert(BaseModel):
    status: str = "firing"
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    startsAt: datetime | None = None
    endsAt: datetime | None = None
    fingerprint: str | None = None

    @property
    def name(self) -> str:
        return self.labels.get("alertname", "unknown")

    @property
    def severity(self) -> str:
        return self.labels.get("severity", "none")


class AlertmanagerPayload(BaseModel):
    status: str = "firing"
    receiver: str | None = None
    alerts: list[Alert] = Field(default_factory=list)


async def _authorise(
    credentials: HTTPBasicCredentials | None = Depends(basic),
    services: Services = Depends(get_services),
) -> None:
    expected = services.settings.admin_token.get_secret_value()
    if credentials is None or not constant_time_equals(credentials.password, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="alert webhook requires the admin token as the basic-auth password",
            headers={"WWW-Authenticate": "Basic"},
        )


@router.post(
    "/v1/alerts/webhook",
    summary="Alertmanager webhook receiver",
    dependencies=[Depends(_authorise)],
)
async def receive_alerts(
    payload: AlertmanagerPayload, services: Services = Depends(get_services)
) -> dict:
    for alert in payload.alerts:
        metrics.ALERTS_RECEIVED.labels(alert.name, alert.status).inc()
        log.log(
            logging.ERROR if alert.status == "firing" else logging.INFO,
            f"alert {alert.status}: {alert.name}",
            extra={
                "alertname": alert.name,
                "severity": alert.severity,
                "alert_status": alert.status,
                "summary": alert.annotations.get("summary"),
                "description": alert.annotations.get("description"),
            },
        )
    try:
        await _remember(services, payload)
    except Exception:
        log.exception("could not store received alerts")
    return {"received": len(payload.alerts)}


async def _remember(services: Services, payload: AlertmanagerPayload) -> None:
    pipe = services.redis.pipeline(transaction=True)
    for alert in payload.alerts:
        pipe.lpush(RECENT_KEY, alert.model_dump_json())
    pipe.ltrim(RECENT_KEY, 0, RECENT_MAX - 1)
    await pipe.execute()


@router.get(
    "/v1/admin/alerts",
    summary="Alerts this gateway has received, newest first",
    dependencies=[Depends(require_admin)],
)
async def recent_alerts(
    limit: int = Query(default=20, ge=1, le=RECENT_MAX),
    services: Services = Depends(get_services),
) -> dict:
    raw = await services.redis.lrange(RECENT_KEY, 0, limit - 1)
    return {"alerts": [Alert.model_validate_json(r) for r in raw]}
