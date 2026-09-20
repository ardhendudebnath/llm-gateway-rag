# Public API guidelines

These guidelines keep the Acme Cloud public API consistent across teams. Every new endpoint is
reviewed against them before it is released.

## Versioning

The major version is part of the URL, for example /v2/projects. Breaking changes are only allowed
in a new major version. When a new major version becomes generally available, the previous one
stays supported for 12 months. Customers are told about a deprecation at least six months before
an endpoint is removed, and deprecated endpoints return a Sunset header with the removal date.

## Pagination

List endpoints use cursor-based pagination. The default page size is 50 items and the maximum page
size is 200 items. Offset-based pagination is not allowed for new endpoints because it becomes slow
and inconsistent on large collections.

## Rate limits

By default each API key may make 100 requests per second, with bursts of up to 200 requests.
Requests over the limit receive HTTP 429 with a Retry-After header. Enterprise customers can ask
for a higher limit of up to 1,000 requests per second.

## Errors

Errors use a single JSON envelope with the fields code, message and request_id. Error codes are
lowercase snake_case strings such as invalid_argument or not_found, and they never change once
published. Include the request_id when contacting support.

## Idempotency

POST endpoints that create resources accept an Idempotency-Key header. The server remembers each
key for 24 hours and returns the original response if the same key is sent again, so clients can
safely retry after a timeout.

## Naming and formats

Field names use snake_case. Timestamps are strings in RFC 3339 format, always in UTC. Amounts of
money are integers in the smallest currency unit, never floating point numbers.
