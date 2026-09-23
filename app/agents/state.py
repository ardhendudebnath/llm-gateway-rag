"""Everything one agent run carries. Nodes read and mutate this; nothing else is shared."""

from dataclasses import dataclass, field

from app.core.security import Principal
from app.rag.retrieval import RetrievedChunk


@dataclass
class ResearchState:
    question: str
    principal: Principal
    model: str
    top_k: int
    max_searches: int
    max_revisions: int
    cache: bool

    searches: list[str] = field(default_factory=list)
    passages: list[RetrievedChunk] = field(default_factory=list)
    draft: str = ""
    critique: str | None = None
    issues: list[str] = field(default_factory=list)
    revisions: int = 0

    # Totals across every LLM call the run made, so a caller sees what the whole run cost.
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
