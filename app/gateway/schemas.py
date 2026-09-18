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


class Attempt(BaseModel):
    deployment: str
    outcome: Literal["success", "error", "timeout", "skipped_circuit_open", "client_error"]
    latency_ms: float = 0.0
    error: str | None = None


class GatewayMeta(BaseModel):
    deployment: str | None
    cached: bool
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
