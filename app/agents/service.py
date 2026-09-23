"""The research agent: plan -> retrieve -> draft -> critique -> revise, as a state graph.

    start -> plan -> retrieve -> draft -> critique -> done
                        |                    |
                        |                    +--> revise -> critique (bounded by max_revisions)
                        +--> done (nothing matched: no draft, no LLM call)

What this adds over ``POST /v1/rag/answer``, which is one retrieval and one completion: the agent
decides what to search for (several queries, merged), and reviews its own draft against the
passages before answering. It costs more LLM calls, which is why every run reports its own
``llm_calls`` and ``cost_usd``.
"""

import logging
import time

from app.agents.graph import END, Graph, Step, StepBudgetExceeded
from app.agents.nodes import AgentNodes
from app.agents.schemas import AgentStep, GraphResponse, ResearchRequest, ResearchResponse
from app.agents.state import ResearchState
from app.core.security import Principal
from app.gateway.schemas import UsageOut
from app.gateway.service import ChatService
from app.observability import metrics
from app.observability.tracing import NoopTracer, Tracer
from app.rag.retrieval import Retriever
from app.rag.schemas import Citation
from app.rag.service import cited_numbers

log = logging.getLogger(__name__)


def build_graph(nodes: AgentNodes, *, max_steps: int) -> Graph[ResearchState]:
    return Graph(
        name="research",
        entry="plan",
        nodes={
            "plan": nodes.plan,
            "retrieve": nodes.retrieve,
            "draft": nodes.draft,
            "critique": nodes.critique,
            "revise": nodes.revise,
        },
        edges={
            "plan": ("retrieve",),
            "retrieve": ("draft", END),
            "draft": ("critique",),
            "critique": ("revise", END),
            "revise": ("critique",),
        },
        max_steps=max_steps,
    )


class ResearchAgent:
    def __init__(
        self,
        retriever: Retriever,
        chat: ChatService,
        tracer: Tracer | None = None,
        *,
        max_steps: int = 12,
    ):
        self._tracer = tracer or NoopTracer()
        self.graph = build_graph(AgentNodes(retriever, chat), max_steps=max_steps)

    async def research(self, principal: Principal, request: ResearchRequest) -> ResearchResponse:
        state = ResearchState(
            question=request.question,
            principal=principal,
            model=request.model,
            top_k=request.top_k,
            max_searches=request.max_searches,
            max_revisions=request.max_revisions,
            cache=request.cache,
        )
        start = time.perf_counter()
        # Collected as they happen, so a run stopped by the budget still reports its path.
        steps: list[Step] = []

        def record(step: Step) -> None:
            steps.append(step)
            metrics.AGENT_STEPS.labels(step.node).inc()

        with self._tracer.span("agent.research", question=request.question, model=request.model):
            try:
                await self.graph.run(state, on_step=record)
                outcome = "answered" if state.passages else "no_context"
            except StepBudgetExceeded:
                # The graph stopped a run that wouldn't settle. The last draft still stands, so
                # return it rather than failing the request, and say so in the metrics.
                log.warning("agent: step budget exceeded", extra={"question": request.question})
                outcome = "budget_exceeded"
        metrics.AGENT_RUNS.labels(outcome).inc()
        metrics.AGENT_DURATION.observe(time.perf_counter() - start)
        metrics.AGENT_REVISIONS.observe(state.revisions)
        _log_run(principal, state, outcome, steps)
        return _response(state, steps)

    def describe(self) -> GraphResponse:
        return GraphResponse(
            name=self.graph.name,
            entry=self.graph.entry,
            nodes=list(self.graph.nodes),
            edges={node: list(targets) for node, targets in self.graph.edges.items()},
            mermaid=self.graph.to_mermaid(),
        )


def _response(state: ResearchState, steps: list[Step]) -> ResearchResponse:
    cited = cited_numbers(state.draft)
    usage = (
        UsageOut(
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            total_tokens=state.prompt_tokens + state.completion_tokens,
        )
        if state.llm_calls
        else None
    )
    return ResearchResponse(
        answer=state.draft,
        citations=[
            Citation(
                n=n,
                doc_id=h.chunk.doc_id,
                title=h.chunk.title,
                chunk_index=h.chunk.chunk_index,
                page=h.chunk.page,
                heading=h.chunk.heading,
                text=h.chunk.text,
                cited=n in cited,
            )
            for n, h in enumerate(state.passages, start=1)
        ],
        searches=state.searches,
        steps=[AgentStep(**vars(s)) for s in steps],
        revisions=state.revisions,
        critique=state.critique,
        llm_calls=state.llm_calls,
        cost_usd=round(state.cost_usd, 8),
        usage=usage,
    )


def _log_run(principal: Principal, state: ResearchState, outcome: str, steps: list[Step]) -> None:
    log.info(
        "agent run",
        extra={
            "tenant_id": principal.tenant_id,
            "key_id": principal.key_id,
            "outcome": outcome,
            "path": " -> ".join([s.node for s in steps] + ["end"]) if steps else None,
            "searches": len(state.searches),
            "passages": len(state.passages),
            "revisions": state.revisions,
            "llm_calls": state.llm_calls,
            "cost_usd": round(state.cost_usd, 8),
        },
    )
