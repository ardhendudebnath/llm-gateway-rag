# API error codes

Every error the platform API returns carries a stable code in the `error.code` field. Clients should
branch on the code, never on the message text, which we reword freely.

## Client errors

| Code | Meaning | What the caller should do |
|---|---|---|
| NG-1001 | The request body is not valid JSON | Fix the payload; retrying will not help |
| NG-1004 | A required field is missing from the request body | Add the field named in `error.field` |
| NG-1007 | The `model` alias does not exist in the route table | Call `GET /v1/models` and use one of those aliases |
| NG-1012 | The API key has been revoked | Issue a new key; revocation is permanent |
| NG-1015 | The API key is valid but belongs to another tenant | Use a key issued for the tenant in the path |
| NG-1017 | The tenant's monthly token budget is exhausted | Raise the budget or wait for the next billing period |
| NG-1021 | The request exceeds the per-key rate limit | Back off for the seconds given in `Retry-After` |
| NG-1023 | The prompt is longer than the model's context window | Shorten the prompt or choose a larger model |
| NG-1026 | The uploaded document is larger than 10 MB | Split the document and upload the parts |
| NG-1029 | The uploaded document's type is not supported | Convert it to PDF, Markdown or plain text |
| NG-1031 | The document is a binary file disguised as text | Upload the original document instead |
| NG-1034 | The requested document belongs to another tenant | Nothing; this is never permitted |

## Server and upstream errors

| Code | Meaning | What the caller should do |
|---|---|---|
| NG-2002 | Every deployment in the route's chain failed | Retry after the `Retry-After` interval |
| NG-2005 | The upstream provider timed out before any token arrived | Retry; the gateway has already tried the chain |
| NG-2008 | A stream broke after it had begun | Treat the partial answer as incomplete and retry |
| NG-2011 | The vector store is unreachable | Retry; retrieval is degraded, chat is not |
| NG-2014 | The embedding queue is saturated | Retry with backoff; the gateway is shedding load |
| NG-2017 | The semantic cache is unavailable | Nothing; requests are served uncached |
| NG-2021 | An ingestion job exhausted its retries and was dead-lettered | Inspect `GET /v1/rag/jobs/{id}` and re-upload |

## Deprecated codes

`NG-1002` and `NG-1003` were merged into `NG-1001` in the March release. `NG-2001` was split into
`NG-2002` and `NG-2005` so that callers can tell a dead chain from a slow provider. Deprecated codes
are never reused for a different meaning.
