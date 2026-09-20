# Incident response

An incident is any unplanned event that degrades a customer-facing service. Anyone at Acme Cloud
may declare an incident; it is always better to declare early and downgrade later.

## Severity levels

| Severity | Meaning | Acknowledge within | First status page update |
|---|---|---|---|
| SEV1 | Full outage, data loss, or a confirmed security breach | 5 minutes | within 15 minutes |
| SEV2 | A major feature is degraded for many customers | 15 minutes | within 30 minutes |
| SEV3 | A minor feature is degraded and a workaround exists | next business day | not required |
| SEV4 | Cosmetic issues with no customer impact | backlog | not required |

A SEV1 always gets a dedicated incident commander. A SEV2 gets one if it lasts longer than an hour.

## Declaring an incident

Declare an incident by running the incident bot in the #incidents channel. The bot opens a new
channel named after the date and a short description, for example #inc-20260312-login-errors, and
pages the on-call engineer of the affected service.

## Roles

The incident commander coordinates the response and makes decisions; the incident commander does
not debug. The communications lead posts status page updates every 30 minutes during a SEV1 and
keeps customer support informed. The scribe records a timeline of actions and findings in the
incident channel, which becomes the backbone of the postmortem.

## Postmortems

Every SEV1 and SEV2 incident needs a postmortem. The draft is due within three business days of
the incident being resolved, and the review meeting must happen within five business days.
Postmortems are blameless: they describe what happened and why the system allowed it, never who
is at fault.

Every action item gets an owner and a due date and is tracked in the reliability backlog. SEV1
postmortems are shared with the whole company in the monthly engineering review.
