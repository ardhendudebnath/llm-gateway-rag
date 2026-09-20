# On-call policy

Acme Cloud runs a follow-the-sun support desk, but engineering on-call is owned by each product
team. This policy applies to every team that runs a production service.

## Rotation

Each team runs a weekly rotation with a primary and a secondary on-call engineer. The handover
happens every Monday at 10:00 CET, in a short call where the outgoing primary walks the incoming
one through the handover checklist. A team needs at least six engineers before it may run its own
rotation; smaller teams share a rotation with a sibling team.

New hires shadow two full rotations before they are scheduled as primary. Nobody should be on call
for more than two consecutive weeks, and swaps must be recorded in the scheduling tool so that
pages are routed to the right person.

## Paging and escalation

Pages are sent through Beacon, our alerting tool, by push notification and phone call.

- The primary must acknowledge a page within 5 minutes.
- If the primary does not acknowledge within 5 minutes, the secondary is paged.
- After 15 minutes without acknowledgement, the engineering manager is paged.
- After 30 minutes, the VP of Engineering is paged.

Acknowledging a page means you are looking at it, not that it is fixed. While on call you must be
able to reach a laptop with a working internet connection within 15 minutes of being paged.

## Compensation

On-call engineers receive a flat stipend of 300 euros for each week on call, whether or not they
are paged. Any page outside business hours that takes more than 30 minutes to resolve earns two
hours of time off in lieu. Incident work on weekends and public holidays is paid at 1.5 times the
normal hourly rate.

## Handover checklist

At every handover the outgoing primary reviews:

- open incidents and their current owners,
- alerts that are silenced, and when each silence expires,
- deploys or migrations that are scheduled during the coming week,
- anything unusual about the week, such as a customer launch or a planned vendor maintenance.
