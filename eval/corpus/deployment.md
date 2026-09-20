# Deployment process

All production changes go through the shipit pipeline. This document describes when you may
deploy, how a release is rolled out, and how to undo it.

## Deploy windows

Production deploys are allowed Monday to Thursday between 09:00 and 16:00 CET. On Fridays deploys
are allowed only until 12:00. Deploying outside these windows requires approval from the on-call
lead of the affected service, recorded in the deploy ticket.

## Change freeze

A company-wide change freeze runs from December 18 to January 3. A shorter freeze applies for the
48 hours before a major marketing launch. During a freeze only security fixes and fixes for active
incidents may be deployed, and each one needs sign-off from a director.

## Canary releases

Every release starts as a canary. The new version first receives 5% of traffic for 30 minutes,
then 25% of traffic for another 30 minutes, and only then 100%. The pipeline rolls back
automatically if the canary's error rate exceeds 2% or if its p95 latency is more than 20% worse
than the stable version.

## Rollbacks

Any engineer may roll back a release without asking for approval. Run shipit rollback with the
service name; a rollback must complete within 10 minutes. Roll back first and investigate
afterwards: a rollback is never the wrong call during an incident.

## Database migrations

Schema migrations must be backward compatible with the currently deployed code. We use the
expand-and-contract pattern: first add the new column or table, then migrate the code, and only
then remove the old structure. Destructive migrations, such as dropping a column, run as a
separate change at least one week after the code stopped using that column.

## Feature flags

User-facing changes ship behind a feature flag so they can be turned off without a deploy. A flag
must be removed from the code within 60 days of the feature reaching 100% of users.
