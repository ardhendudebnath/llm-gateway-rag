"""Request content shared by the Locust file and the runner.

Kept free of Locust imports on purpose: importing Locust monkey-patches `ssl` via gevent, and doing
that after `requests` has already imported `ssl` (as the runner does) recurses forever on
Python 3.13.
"""

# A small pool, so after warm-up these are cache hits, like FAQ-style traffic.
POPULAR = [
    "What is a circuit breaker in a distributed system?",
    "How does token bucket rate limiting work?",
    "Explain semantic caching for LLM responses.",
    "What is retrieval-augmented generation?",
    "When should a service fall back to another provider?",
    "What does p95 latency mean?",
    "How do I rotate an API key safely?",
    "What is an error budget?",
]

# Questions about the eval corpus, which the runner ingests before the test.
RAG_QUESTIONS = [
    "How quickly must a SEV1 be acknowledged?",
    "When is the end-of-year change freeze?",
    "How long are audit logs retained?",
    "What is the default rate limit for an API key?",
    "How many approvals does a change to billing code need?",
    "What is the hotel limit for a trip to Paris?",
    "When must a feature flag be removed?",
    "Who gets paged after 15 minutes without acknowledgement?",
]
