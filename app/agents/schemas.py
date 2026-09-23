"""Public request/response models for the /v1/agents endpoints."""

from pydantic import BaseModel, Field

from app.gateway.schemas import UsageOut
from app.rag.schemas import Citation


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: str = Field(default="default", description="Route alias used for every step.")
    top_k: int = Field(default=5, ge=1, le=10, description="Passages kept after merging searches.")
    max_searches: int = Field(default=3, ge=1, le=5)
    max_revisions: int = Field(
        default=1, ge=0, le=3, description="0 skips the self-critique entirely."
    )
    cache: bool = Field(default=True, description="Allow semantic-cache hits for each step.")


class AgentStep(BaseModel):
    node: str
    next: str
    duration_ms: float
    note: str | None = None


class ResearchResponse(BaseModel):
    answer: str
    citations: list[Citation]
    searches: list[str] = Field(description="The queries the agent decided to run.")
    steps: list[AgentStep] = Field(description="Every transition the graph took, in order.")
    revisions: int
    critique: str | None = Field(
        default=None, description="The reviewer's verdict on the last draft it saw."
    )
    llm_calls: int
    cost_usd: float
    usage: UsageOut | None = None


class GraphResponse(BaseModel):
    name: str
    entry: str
    nodes: list[str]
    edges: dict[str, list[str]]
    mermaid: str = Field(description="The same graph as a Mermaid flowchart.")
