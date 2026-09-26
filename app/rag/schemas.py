"""Public request/response models for the /v1/rag endpoints."""

from typing import Literal

from pydantic import BaseModel, Field

from app.gateway.schemas import GatewayMeta, UsageOut


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(default=True, description="Apply the cross-encoder, if one is configured.")


class SearchHit(BaseModel):
    doc_id: str
    title: str
    chunk_index: int
    page: int | None
    heading: str | None
    text: str
    vector_score: float
    rerank_score: float | None


class SearchResponse(BaseModel):
    query: str
    reranked: bool
    retrieval: Literal["dense", "hybrid"] = Field(
        default="dense",
        description="'hybrid' fuses BM25 with the vector ranking, which is what finds exact "
        "identifiers; 'dense' means vectors only, either by configuration or because this "
        "collection has no lexical vector.",
    )
    hits: list[SearchHit]


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: str = Field(default="default", description="Route alias used to generate the answer.")
    top_k: int = Field(default=5, ge=1, le=10)
    rerank: bool = True
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    cache: bool = Field(default=True, description="Allow a semantic-cache hit for the answer.")


class Citation(BaseModel):
    n: int = Field(description="The number the answer uses to cite this passage, e.g. [2].")
    doc_id: str
    title: str
    chunk_index: int
    page: int | None
    heading: str | None
    text: str
    cited: bool = Field(description="Whether the answer actually cites [n].")


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    usage: UsageOut | None = None
    nexusgate: GatewayMeta | None = Field(
        default=None, description="Gateway metadata; null when no passage matched and no LLM ran."
    )
