"""Agent endpoints: a multi-step research run, and the graph it runs, scoped to the tenant."""

from fastapi import APIRouter, Depends

from app.agents.schemas import GraphResponse, ResearchRequest, ResearchResponse
from app.api.deps import get_services, rate_limited
from app.core.container import Services
from app.core.security import Principal

router = APIRouter(prefix="/v1/agents", tags=["agents"])


@router.post(
    "/research",
    response_model=ResearchResponse,
    summary="Plan searches, retrieve, draft a cited answer, then self-critique and revise",
)
async def research(
    body: ResearchRequest,
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> ResearchResponse:
    """Several LLM calls per request, unlike `/v1/rag/answer`. The response reports every
    transition the graph took, plus the run's own `llm_calls` and `cost_usd`."""
    return await services.agent.research(principal, body)


@router.get(
    "/graph",
    response_model=GraphResponse,
    summary="The agent's state graph: nodes, allowed transitions, and a Mermaid diagram",
)
async def graph(
    principal: Principal = Depends(rate_limited),
    services: Services = Depends(get_services),
) -> GraphResponse:
    return services.agent.describe()
