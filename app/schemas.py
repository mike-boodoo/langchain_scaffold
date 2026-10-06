from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceKind(StrEnum):
    DOCUMENT = "document"
    MCP = "mcp"
    MODEL = "model"


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    excerpt: str = Field(min_length=1, max_length=2_000)
    score: float | None = Field(default=None, ge=0)
    kind: SourceKind = SourceKind.DOCUMENT
    uri: str | None = None


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=4_000)
    top_k: int = Field(default=6, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0)
    filters: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    text: str = Field(min_length=1)
    title: str | None = None
    score: float | None = Field(default=None, ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    chunks: list[RetrievedChunk]

    @property
    def context(self) -> str:
        return "\n\n".join(
            f"[{i + 1}] {chunk.text}"
            for i, chunk in enumerate(self.chunks)
        )


class ToolInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    purpose: str
    risk: Literal["low", "medium", "high"] = "low"


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=12_000)
    confidence: float = Field(ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list, max_length=20)
    tool_trace: list[ToolInvocation] = Field(default_factory=list, max_length=50)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=10)


class SearchBooksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2)
    limit: int = Field(default=5, ge=1, le=20)


class Book(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    author: str
    year: int = Field(ge=1450, le=2100)
    summary: str


class Health(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str
    details: dict[str, str]


class SearchAction(BaseModel):
    kind: Literal["search"]
    query: str = Field(min_length=2)


class LookupAction(BaseModel):
    kind: Literal["lookup"]
    id: str = Field(min_length=1)


Action = Annotated[
    SearchAction | LookupAction,
    Field(discriminator="kind"),
]
