from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str


class Source(BaseModel):
    id: int
    url: str
    title: str
    snippet: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Finding(BaseModel):
    content: str
    source_ids: list[int]
    confidence: Literal["high", "medium", "low"]


class SubQuestion(BaseModel):
    id: str
    question: str
    status: Literal["pending", "investigating", "answered", "unanswerable"] = "pending"
    findings: list[Finding] = []


class ResearchPlan(BaseModel):
    query: str
    sub_questions: list[SubQuestion]
    iteration: int = 0
    max_iterations: int = 10


class ReportSection(BaseModel):
    title: str
    content: str
    source_ids: list[int] = []


class ReportMetadata(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str = ""
    total_sources: int = 0
    total_iterations: int = 0
    tokens_used: int = 0


class Report(BaseModel):
    title: str
    executive_summary: str
    sections: list[ReportSection]
    contradictions: list[str] = []
    sources: list[Source] = []
    metadata: ReportMetadata = Field(default_factory=ReportMetadata)


class TokenBudget(BaseModel):
    max_tokens: int = 500_000
    used_tokens: int = 0
    warn_threshold: float = 0.8

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def is_exceeded(self) -> bool:
        return self.used_tokens >= self.max_tokens

    @property
    def should_warn(self) -> bool:
        return self.max_tokens > 0 and self.used_tokens / self.max_tokens >= self.warn_threshold

    def add(self, tokens: int) -> None:
        self.used_tokens += tokens


class StoppingEvaluation(BaseModel):
    should_stop: bool
    reasoning: str
    coverage_score: float = 0.0
    saturation_detected: bool = False
    unanswered_questions: list[str] = []


class SessionState(BaseModel):
    session_id: str
    plan: ResearchPlan
    sources: list[Source] = []
    findings: list[Finding] = []
    budget: TokenBudget = Field(default_factory=TokenBudget)
    status: Literal["planning", "investigating", "synthesizing", "complete", "failed"] = "planning"
    report: Report | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
