"""Mixed, realistic traffic for NexusGate. Driven by loadtest/run_load.py, or on its own:

    NEXUSGATE_LOADTEST_KEY=ng_... locust -f loadtest/locustfile.py --host http://localhost:8000

Uses only the offline `mock` route, so it needs no provider keys and spends nothing. The traffic
is split into request kinds that exercise different paths, each reported separately:

    chat: cache hit       a question from a small warm pool -> answered from the semantic cache
    chat: provider call   cache disabled -> the router and the (mock) provider, 80 ms simulated
    rag: search           embed the query, vector search, cross-encoder rerank
    rag: answer           retrieval + an LLM call through the chat path
    chat: stream          server-sent events; reported as two figures (below), off by default

Misses are forced with `cache: false` rather than with "unique" prompts: with a real embedding
model, two templated prompts can land inside the similarity threshold and silently become hits.

**Streaming traffic is opt-in** (`NEXUSGATE_LOADTEST_STREAM_WEIGHT`, default 0) so that the default
mix stays identical to the runs already in `results/`. Changing the mix would make a new run
incomparable with the old numbers, and the first question to answer is whether the features added
since then cost any throughput at all. A run with streaming is a separate, labelled run.

When it is on, each streamed request is reported as two figures, because one would hide the thing
that matters: a slow generation with a fast first token is a good experience.

    chat: stream first token   what Locust times for a streamed request, which is time to the
                              response headers. For this gateway that *is* time to first token:
                              the endpoint pulls the first chunk before the response exists (so a
                              failed chain is still a 503), and headers cannot be sent before it.
    chat: stream complete      the whole stream, timed here and fired as its own metric, because
                              Locust stops timing at the headers.
"""

import os
import random
import sys
import time
from pathlib import Path

from locust import FastHttpUser, between, task

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traffic import POPULAR, RAG_QUESTIONS

KEY = os.environ.get("NEXUSGATE_LOADTEST_KEY", "")
STREAM_WEIGHT = int(os.environ.get("NEXUSGATE_LOADTEST_STREAM_WEIGHT", "0"))
TTFT_NAME = "chat: stream first token"


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

    @task(STREAM_WEIGHT)
    def chat_stream(self) -> None:
        start = time.perf_counter()
        received = 0
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "mock",
                "cache": False,
                "stream": True,
                "messages": [{"role": "user", "content": random.choice(POPULAR)}],
            },
            headers=self.auth,
            name=TTFT_NAME,  # Locust times this to the headers: see the module docstring
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            # The whole body is consumed, not abandoned after the first chunk: the gateway meters a
            # stream when it ends, and a client that walks away is a different measurement.
            # `carry` holds the tail of the previous chunk, because a marker can straddle a chunk
            # boundary — without it, one stream in six was reported as missing its terminator.
            carry, saw_event, done = b"", False, False
            for chunk in resp.iter_content(64):
                if not chunk:
                    continue
                received += len(chunk)
                window = carry + chunk
                saw_event = saw_event or b"data:" in window
                done = done or b"[DONE]" in window
                carry = chunk[-16:]
            if not saw_event:
                resp.failure("stream produced no events")
                return
            if not done:
                resp.failure("stream ended without [DONE]")
                return
        self.environment.events.request.fire(
            request_type="POST",
            name="chat: stream complete",
            response_time=(time.perf_counter() - start) * 1000,
            response_length=received,
            exception=None,
            context={},
        )

    @task(1)
    def rag_answer(self) -> None:
        self.client.post(
            "/v1/rag/answer",
            json={"question": random.choice(RAG_QUESTIONS), "model": "mock", "cache": False},
            headers=self.auth,
            name="rag: answer",
        )
