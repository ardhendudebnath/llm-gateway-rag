"""Public demo mode (``NEXUSGATE_DEMO_MODE=true``): one container that anyone can try.

What makes it safe to put on the internet:

* **Offline routes only** (``config/routes.demo.yaml``): the mock providers, so the image holds no
  provider keys and a visitor can't spend anything.
* **One shared, tightly rate-limited API key** for the ``demo`` tenant, shown on the landing page.
  It is minted at startup and changes on every restart.
* **Uploads disabled.** Everyone shares the demo tenant, so accepting files would let one visitor
  serve content to the next. The eval handbook is ingested at startup instead, so RAG search and
  answers work immediately.
* **Admin endpoints stay closed**, behind a random admin token that is never displayed.
"""

import asyncio
import html
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.core.container import Services
from app.rag.ingestion import DocumentUpload

log = logging.getLogger(__name__)
router = APIRouter(include_in_schema=False)

DEMO_TENANT = "demo"


async def prepare_demo(services: Services) -> str:
    """Mint the public key and ingest the demo corpus. Returns the raw key."""
    settings = services.settings
    raw_key, record = await services.keys.create(
        DEMO_TENANT,
        "public-demo",
        rate_limit_capacity=settings.demo_rate_limit_capacity,
        rate_limit_refill_per_sec=settings.demo_rate_limit_refill_per_sec,
    )
    corpus = await asyncio.to_thread(_read_corpus, Path(settings.demo_corpus_dir))
    for name, data in corpus:
        # Directly, not through the job queue: the demo must be ready before it takes traffic.
        await services.rag.ingestion.ingest(
            DEMO_TENANT, DocumentUpload(name, "text/markdown", data)
        )
    log.info("demo ready", extra={"key_id": record.key_id, "documents": len(corpus)})
    return raw_key


def _read_corpus(directory: Path) -> list[tuple[str, bytes]]:
    return [(p.name, p.read_bytes()) for p in sorted(directory.glob("*.md"))]


@router.get("/", response_class=HTMLResponse)
async def landing(request: Request) -> HTMLResponse:
    key = html.escape(request.app.state.demo_key)
    base = html.escape(str(request.base_url).rstrip("/"))
    return HTMLResponse(PAGE.format(key=key, base=base))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NexusGate demo</title>
<style>
  :root {{ --bg:#0f1115; --fg:#e7e9ee; --muted:#9aa3b2; --card:#171a21; --accent:#7aa2f7; }}
  body {{ margin:0; font:16px/1.55 system-ui, sans-serif;
          background:var(--bg); color:var(--fg); }}
  main {{ max-width:860px; margin:0 auto; padding:32px 16px 64px; }}
  h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin-top:32px; font-size:19px; }}
  p.lead {{ color:var(--muted); margin-top:0; }}
  code, pre {{ font:14px/1.5 ui-monospace, Consolas, monospace; }}
  pre {{ background:var(--card); padding:14px 16px; border-radius:8px; overflow-x:auto; }}
  .key {{ background:var(--card); padding:10px 14px; border-radius:8px;
          word-break:break-all; }}
  a {{ color:var(--accent); }} ul {{ padding-left:20px; }}
</style></head><body><main>
<h1>NexusGate</h1>
<p class="lead">An LLM gateway and RAG backend: provider fallback with circuit breakers, a semantic
cache, per-key rate limiting and cost metering, and retrieval with reranking. This public demo runs
on offline mock providers, so nothing here costs anything.</p>

<h2>Your demo API key</h2>
<div class="key"><code>{key}</code></div>
<p>Shared by everyone and rate-limited. It changes when the demo restarts. Try the endpoints in
the <a href="/docs">interactive API docs</a> (click <em>Authorize</em> and paste the key), or with
curl:</p>

<h2>1. Chat, then hit the semantic cache</h2>
<pre>curl -s {base}/v1/chat/completions \\
  -H "Authorization: Bearer {key}" \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "mock",
       "messages": [{{"role": "user", "content": "What is a circuit breaker?"}}]}}'</pre>
<p>Run it twice: the second response has <code>"cached": true</code> and
<code>"cost_usd": 0</code>.</p>

<h2>2. Watch fallback</h2>
<pre>curl -s {base}/v1/chat/completions \\
  -H "Authorization: Bearer {key}" \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "chaos", "cache": false,
       "messages": [{{"role": "user", "content": "ping"}}]}}'</pre>
<p>The <code>chaos</code> route's primary always fails; <code>nexusgate.attempts</code> shows the
failed attempt and the fallback that answered.</p>

<h2>3. Ask the handbook (RAG)</h2>
<pre>curl -s {base}/v1/rag/answer \\
  -H "Authorization: Bearer {key}" \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "When is a postmortem due?", "model": "mock"}}'</pre>
<p>A fictional engineering handbook is pre-loaded. Answers come with numbered citations;
<code>/v1/rag/search</code> shows the reranked passages.</p>

<h2>4. Let the agent work</h2>
<pre>curl -s {base}/v1/agents/research \\
  -H "Authorization: Bearer {key}" \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "How do we handle a SEV1?", "model": "mock"}}'</pre>
<p>It plans its own searches, merges the results, drafts a cited answer, then criticises and
revises it. The response lists every transition, and what the run cost.
<a href="/v1/agents/graph">The graph itself</a> is an endpoint too.</p>

<h2>More</h2>
<ul>
  <li><a href="/docs">API docs</a> · <a href="/metrics">Prometheus metrics</a> ·
      <code>GET /v1/usage</code> for your spend and cache savings</li>
  <li>Source, design notes, the retrieval eval and the load tests:
      <a href="https://github.com/ardhendudebnath/llm-gateway-rag">GitHub</a></li>
</ul>
</main></body></html>
"""
