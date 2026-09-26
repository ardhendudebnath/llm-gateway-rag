"""Request/response models. The public API is OpenAI-compatible so existing SDKs work unchanged."""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="default", description="Route alias, e.g. 'default' or 'mock'.")
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = Field(default=False, description="Server-sent events, as the OpenAI API does.")
    cache: bool = Field(default=True, description="NexusGate extension: set false to bypass.")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ProviderResponse(BaseModel):
    """What a provider adapter returns — normalized across vendors."""

    content: str
    usage: Usage
    model: str
    finish_reason: str = "stop"


class StreamDelta(BaseModel):
    """One piece of a streamed completion, as provider adapters yield them.

    Providers report usage only at the end of a stream, so `usage` and `finish_reason` arrive on
    the last delta — which is why the router cannot know the cost of a stream until it finishes.
    """

    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    model: str | None = None


class Attempt(BaseModel):
    deployment: str
    outcome: Literal[
        "success",
        "error",
        "timeout",
        "skipped_circuit_open",
        "client_error",
        "stream_interrupted",  # it answered, then broke: too late to fall back
    ]
    latency_ms: float = 0.0
    error: str | None = None


class GatewayMeta(BaseModel):
    deployment: str | None
    cached: bool
    route_variant: str | None = Field(
        default=None, description="Which route table served it: 'stable' or 'canary'."
    )
    cache_similarity: float | None = None
    cost_usd: float
    latency_ms: float
    attempts: list[Attempt] = []


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageOut
    nexusgate: GatewayMeta


# --- streaming wire format: OpenAI's `chat.completion.chunk`, so SDKs work unchanged ---


class ChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: str | None = None


class StreamMeta(GatewayMeta):
    """The gateway's own footer on the last chunk: what a non-streamed response returns in
    `nexusgate`, plus the two things only a stream has."""

    ttft_ms: float | None = Field(default=None, description="Time to the first token.")
    synthesized: bool = Field(
        default=False, description="The deployment cannot stream; its answer was chunked here."
    )
    usage_estimated: bool = Field(
        default=False, description="The provider reported no usage; tokens are counted from text."
    )
    interrupted: str | None = Field(
        default=None, description="The stream broke after it began; no fallback was possible."
    )


class ChatChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]
    usage: UsageOut | None = None
    nexusgate: StreamMeta | None = None
