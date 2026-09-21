"""Mixed, realistic traffic for NexusGate. Driven by loadtest/run_load.py, or on its own:

    NEXUSGATE_LOADTEST_KEY=ng_... locust -f loadtest/locustfile.py --host http://localhost:8000

Uses only the offline `mock` route, so it needs no provider keys and spends nothing. The traffic
is split into request kinds that exercise different paths, each reported separately:

    chat: cache hit       a question from a small warm pool -> answered from the semantic cache
    chat: provider call   cache disabled -> the router and the (mock) provider, 80 ms simulated
    rag: search           embed the query, vector search, cross-encoder rerank
    rag: answer           retrieval + an LLM call through the chat path

Misses are forced with `cache: false` rather than with "unique" prompts: with a real embedding
model, two templated prompts can land inside the similarity threshold and silently become hits.
"""

import os
import random
import sys
from pathlib import Path

from locust import FastHttpUser, between, task

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traffic import POPULAR, RAG_QUESTIONS

KEY = os.environ.get("NEXUSGATE_LOADTEST_KEY", "")


class GatewayUser(FastHttpUser):
    wait_time = between(0.5, 1.5)  # think time between a user's requests

    def on_start(self) -> None:
        self.auth = {"Authorization": f"Bearer {KEY}"}

    def _chat(self, content: str, *, cache: bool, name: str) -> None:
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "mock",
                "cache": cache,
                "messages": [{"role": "user", "content": content}],
            },
            headers=self.auth,
            name=name,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    @task(4)
    def chat_cache_hit(self) -> None:
        self._chat(random.choice(POPULAR), cache=True, name="chat: cache hit")

    @task(4)
    def chat_provider_call(self) -> None:
        self._chat(random.choice(POPULAR), cache=False, name="chat: provider call")

    @task(2)
    def rag_search(self) -> None:
        self.client.post(
            "/v1/rag/search",
            json={"query": random.choice(RAG_QUESTIONS), "top_k": 5},
            headers=self.auth,
            name="rag: search",
        )

    @task(1)
    def rag_answer(self) -> None:
        self.client.post(
            "/v1/rag/answer",
            json={"question": random.choice(RAG_QUESTIONS), "model": "mock", "cache": False},
            headers=self.auth,
            name="rag: answer",
        )
