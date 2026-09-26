# Alert thresholds

Every alert below is defined in Prometheus and routed by Alertmanager. An alert without a runbook is
not allowed to page anyone, so each row names one.

| Alert | Fires when | For | Severity | Runbook |
|---|---|---|---|---|
| HighErrorRate | 5xx responses exceed 5% of requests | 5 minutes | critical | RB-101 |
| ElevatedErrorRate | 5xx responses exceed 1% of requests | 15 minutes | warning | RB-101 |
| SlowResponses | p95 latency exceeds 2 seconds | 10 minutes | warning | RB-104 |
| VerySlowResponses | p95 latency exceeds 5 seconds | 5 minutes | critical | RB-104 |
| ProviderCircuitOpen | any deployment's circuit is open | 2 minutes | warning | RB-108 |
| AllProvidersFailing | every deployment in a chain fails | 1 minute | critical | RB-109 |
| IngestionBacklog | the ingestion queue exceeds 500 jobs | 10 minutes | warning | RB-112 |
| IngestionStalled | the ingestion queue has not shrunk | 30 minutes | critical | RB-113 |
| DeadLettersGrowing | dead letters increase by more than 10 | 15 minutes | warning | RB-114 |
| CacheHitRateCollapsed | the cache hit rate falls below 10% | 30 minutes | warning | RB-118 |
| TokenSpendSpike | hourly spend is triple the weekly average | 15 minutes | warning | RB-121 |
| VectorStoreUnreachable | the vector store rejects connections | 2 minutes | critical | RB-124 |
| RedisMemoryHigh | Redis uses more than 85% of its limit | 10 minutes | warning | RB-127 |
| WorkerPodsCrashLooping | a worker restarts more than 3 times | 15 minutes | critical | RB-131 |

## Severity means a response, not a feeling

A critical alert pages the on-call engineer immediately, at any hour. A warning is delivered to
`#alerts` and is picked up during business hours. Anything that is neither of those is a dashboard
panel, not an alert.

## Silences

A silence longer than 24 hours needs a linked issue explaining the fix and a named owner. Silences
are reviewed every Monday at the on-call handover, and an expired silence that is renewed twice is
escalated to the engineering manager.
