# Observability standards

Every production service must be observable before it goes live. This document sets the minimum
standard for service level objectives, alerting, dashboards and logging.

## Service level objectives

The public API has an availability objective of 99.9% per calendar month, which is an error budget
of about 43 minutes of downtime per month. Read endpoints must keep p95 latency under 300 ms and
write endpoints under 800 ms.

## Error budget policy

If a service burns more than half of its monthly error budget within a single week, the team
pauses feature work and spends the next sprint on reliability. Once the error budget is exhausted,
only bug fixes and security patches may be deployed until the budget recovers.

## Alerting

Page a human only for symptoms that customers can feel, such as errors or slow responses, never for
causes such as high CPU. Burn-rate alerts page when the error budget burns 14.4 times faster than
sustainable over one hour, or 6 times faster over six hours. Everything that is not urgent creates
a ticket in the team's queue instead of a page.

## Dashboards

Each service has a dashboard showing the four golden signals: latency, traffic, errors and
saturation. The dashboard is linked from the service's runbook and from every alert it sends.

## Logging

Logs are structured JSON, and every line includes the request_id so that a single request can be
followed across services. Personal data must never be written to logs. Production services log at
INFO level; DEBUG logging may be switched on for at most 24 hours while investigating an issue.
