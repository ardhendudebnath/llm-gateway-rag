"""The agent's five nodes: plan, retrieve, draft, critique, revise.

Each one is a plain async function over ``ResearchState``, returning the node to run next. They are
unit-tested with a scripted chat service, without a graph, a provider or a vector store.

Every LLM call goes through ``ChatService``, so the agent inherits provider fallback, circuit
breaking, the semantic cache, metering and tracing instead of re-implementing any of it.
"""

import logging
import re

from app.agents import prompts
from app.agents.graph import END, Next
from app.agents.state import ResearchState
from app.core.concurrency import OverloadedError
from app.gateway.router import AllProvidersFailedError, ClientRequestError, UnknownModelError
from app.gateway.schemas import ChatMessage, ChatRequest
from app.gateway.service import ChatService
from app.rag.retrieval import RetrievedChunk, Retriever

log = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "I couldn't find anything relevant to that question in your documents."
MAX_QUERY_CHARS = 200
# Models number or bullet their lists however they like. Only a leading marker is removed: the
# obvious alternative, stripping "-*0123456789." from both ends, turns "SEV1" into "SEV".
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

# A step may degrade when *it* fails, but not when the failure dooms the whole run: an unknown
# route or a malformed request is the caller's, and shedding or every provider being down would
# hit the next step too. Failing now beats paying for retrieval and failing later anyway.
FATAL = (UnknownModelError, ClientRequestError, AllProvidersFailedError, OverloadedError)


def format_context(passages: list[RetrievedChunk]) -> str:
    def label(hit: RetrievedChunk) -> str:
        parts = [hit.chunk.title]
        if hit.chunk.page is not None:
            parts.append(f"p. {hit.chunk.page}")
        return ", ".join(parts)

    return "\n\n".join(
        f"[{n}] ({label(h)})\n{h.chunk.text}" for n, h in enumerate(passages, start=1)
    )


class AgentNodes:
    def __init__(self, retriever: Retriever, chat: ChatService):
        self._retriever = retriever
        self._chat = chat

    async def _ask(self, state: ResearchState, system: str, user: str) -> str:
        """One LLM call, with its cost and tokens added to the run's totals."""
        response = await self._chat.complete(
            state.principal,
            ChatRequest(
                model=state.model,
                messages=[
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ],
                cache=state.cache,
            ),
        )
        state.llm_calls += 1
        if response.usage is not None:
            state.prompt_tokens += response.usage.prompt_tokens
            state.completion_tokens += response.usage.completion_tokens
        if response.nexusgate is not None:
            state.cost_usd += response.nexusgate.cost_usd
        return response.choices[0].message.content.strip()

    async def plan(self, state: ResearchState) -> Next:
        """Turn the question into search queries. A useless plan costs recall, not correctness,
        so anything unparseable falls back to searching for the question itself."""
        try:
            reply = await self._ask(
                state,
                prompts.PLAN_SYSTEM.format(max_searches=state.max_searches),
                state.question,
            )
        except FATAL:
            raise
        except Exception:
            log.exception("agent: planning failed; searching for the question as asked")
            reply = ""
        queries = [
            _LIST_MARKER.sub("", line).strip()[:MAX_QUERY_CHARS]
            for line in reply.splitlines()
            if line.strip()
        ]
        queries = [q for q in dict.fromkeys(queries) if q][: state.max_searches]
        state.searches = queries or [state.question]
        return Next("retrieve", note=f"{len(state.searches)} search(es)")

    async def retrieve(self, state: ResearchState) -> Next:
        """Run every planned search and merge the hits, best score first, without duplicates.

        Sequentially, on purpose: embedding and reranking run behind bounded gates (the load test
        showed why), so firing the searches at once would mostly queue them, and could get the
        rerank shed. A few hundred milliseconds is the right price here.
        """
        best: dict[tuple[str, int], RetrievedChunk] = {}
        for query in state.searches:
            for hit in await self._retriever.search(
                state.principal.tenant_id, query, state.top_k, rerank=True
            ):
                key = (hit.chunk.doc_id, hit.chunk.chunk_index)
                if key not in best or _score(hit) > _score(best[key]):
                    best[key] = hit
        state.passages = sorted(best.values(), key=_score, reverse=True)[: state.top_k]
        if not state.passages:
            state.draft = NO_CONTEXT_ANSWER
            return Next(END, note="nothing matched")
        return Next("draft", note=f"{len(state.passages)} passage(s)")

    async def draft(self, state: ResearchState) -> Next:
        context = format_context(state.passages)
        state.draft = await self._ask(
            state, prompts.DRAFT_SYSTEM.format(context=context), state.question
        )
        return Next("critique", note=f"{len(state.draft)} chars")

    async def critique(self, state: ResearchState) -> Next:
        """Check the draft against the passages. The reviewer is advisory: if it fails, or the
        revision budget is spent, the run ends with the draft it already has."""
        if state.revisions >= state.max_revisions:
            return Next(END, note="revision budget spent")
        context = format_context(state.passages)
        try:
            verdict = await self._ask(
                state,
                prompts.CRITIQUE_SYSTEM.format(context=context),
                prompts.CRITIQUE_USER.format(question=state.question, draft=state.draft),
            )
        except FATAL:
            raise
        except Exception:
            log.exception("agent: critique failed; keeping the draft")
            return Next(END, note="critique unavailable")
        state.critique = verdict
        state.issues = [line.strip(" -*") for line in verdict.splitlines() if line.strip(" -*")]
        if _approves(verdict):
            state.issues = []
            return Next(END, note="approved")
        return Next("revise", note=f"{len(state.issues)} issue(s)")

    async def revise(self, state: ResearchState) -> Next:
        context = format_context(state.passages)
        state.draft = await self._ask(
            state,
            prompts.REVISE_SYSTEM.format(context=context),
            prompts.REVISE_USER.format(
                question=state.question,
                draft=state.draft,
                issues="\n".join(f"- {i}" for i in state.issues),
            ),
        )
        state.revisions += 1
        return Next("critique", note=f"revision {state.revisions}")


def _score(hit: RetrievedChunk) -> float:
    return hit.rerank_score if hit.rerank_score is not None else hit.vector_score


def _approves(verdict: str) -> bool:
    """Only a clear "OK" (or an empty reply) counts as approval; anything else is treated as
    problems to fix. Erring this way costs at most one extra revision, which the budget caps,
    whereas the opposite reading would silently skip the review the agent exists to do."""
    head = verdict.strip().strip(".").upper()
    return head.startswith("OK") or not verdict.strip()
