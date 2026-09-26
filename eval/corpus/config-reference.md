# Configuration reference

Every setting is read from the environment with the `NEXUSGATE_` prefix. Defaults are what the
production cluster runs unless the value is overridden in the Helm values or the kustomize overlay.

## Gateway

| Setting | Default | Effect |
|---|---|---|
| PROVIDER_TIMEOUT_SECONDS | 30 | How long one provider call may take before it counts as a timeout |
| BREAKER_FAILURE_THRESHOLD | 3 | Consecutive failures before a deployment's circuit opens |
| BREAKER_COOLDOWN_SECONDS | 30 | How long an open circuit stays open before one probe is allowed |
| BREAKER_SHARED | true | Whether replicas pool their circuit state through Redis |
| RATE_LIMIT_CAPACITY | 60 | Requests a single key may burst to |
| RATE_LIMIT_REFILL_PER_SEC | 1 | Tokens added back to a key's bucket every second |
| USAGE_RETENTION_DAYS | 90 | How long per-key usage records are kept |

## Retrieval and caching

| Setting | Default | Effect |
|---|---|---|
| CACHE_SIMILARITY_THRESHOLD | 0.93 | Cosine similarity a cached answer must reach to be reused |
| CACHE_TTL_SECONDS | 86400 | How long a cached answer may be served |
| CACHE_MAX_ENTRIES | 10000 | Cached answers kept per tenant namespace before eviction |
| RETRIEVAL_CANDIDATES | 20 | Passages fetched from the vector store before reranking |
| RETRIEVAL_TOP_K | 5 | Passages kept after reranking and shown as citations |
| CHUNK_MAX_WORDS | 180 | Longest chunk the chunker will emit |
| CHUNK_OVERLAP_WORDS | 40 | Words repeated between neighbouring chunks |
| EMBED_MAX_CONCURRENCY | 2 | Embedding calls allowed to run at once |
| EMBED_MAX_QUEUE | 64 | Embedding calls allowed to wait before requests are shed |
| EMBED_BATCH_SIZE | 32 | Chunks embedded in one call, which bounds worker memory |

## Ingestion

| Setting | Default | Effect |
|---|---|---|
| MAX_UPLOAD_BYTES | 10485760 | Largest document accepted, in bytes |
| INGEST_MAX_ATTEMPTS | 3 | Attempts before an ingestion job is dead-lettered |
| INGEST_RETRY_BACKOFF_SECONDS | 5 | Base delay between ingestion attempts, doubled each time |

## Precedence

Environment variables win over the `.env` file, which wins over the defaults in this table. A
setting that is misspelled is ignored silently, which is why the deploy pipeline diffs the rendered
environment against this reference and fails on an unknown `NEXUSGATE_` key.
