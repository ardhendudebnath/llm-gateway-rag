# Service catalogue

Every deployed service is listed here with its owning team and its tier. Tier 1 means a page at any
hour; tier 2 means a page during business hours; tier 3 means a ticket the owning team picks up in
its next planning slot.

| Service | Owner | Tier | Availability target |
|---|---|---|---|
| api-gateway | Platform | 1 | 99.95% |
| auth-service | Identity | 1 | 99.95% |
| billing-reconciler | Revenue | 2 | 99.5% |
| invoice-mailer | Revenue | 3 | 99.0% |
| usage-aggregator | Revenue | 2 | 99.5% |
| document-ingester | Knowledge | 2 | 99.5% |
| embedding-worker | Knowledge | 2 | 99.5% |
| search-api | Knowledge | 1 | 99.9% |
| notification-fanout | Growth | 3 | 99.0% |
| onboarding-wizard | Growth | 3 | 99.0% |
| audit-log-writer | Compliance | 1 | 99.99% |
| export-builder | Compliance | 3 | 98.0% |
| feature-flag-service | Platform | 1 | 99.99% |
| schema-registry | Platform | 2 | 99.5% |
| webhook-dispatcher | Growth | 2 | 99.5% |

## Ownership rules

A service without a named owning team cannot be deployed to production: the deploy pipeline checks
this catalogue and fails the build. When a team is dissolved, its services are reassigned within one
sprint, and until then the Platform team holds them.

## Retiring a service

Announce the retirement four weeks ahead in `#engineering`, move its traffic to the replacement,
keep it running with no traffic for one more week, then delete it and remove its row from this
table. The audit-log-writer is the one service that may never be retired without written approval
from the Compliance team.
