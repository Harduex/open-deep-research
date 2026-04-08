from __future__ import annotations

from pydantic import BaseModel

from open_deep_research.llm.client import LLMClient
from open_deep_research.models import Finding, ResearchPlan, Source, SubQuestion

PLANNING_PROMPT = """You are a research planner. Given a user's research query, decompose it into 3-7 specific sub-questions that, when answered comprehensively, would produce a thorough research report.

Each sub-question should:
- Be specific and searchable
- Cover a distinct aspect of the topic
- Together provide comprehensive coverage

User query: {query}"""


PLAN_UPDATE_PROMPT = """You are a research planner reviewing progress on a research investigation.

Original query: {query}

Current sub-questions and their status:
{sub_questions_summary}

Findings so far:
{findings_summary}

Number of sources consulted: {source_count}

Based on the findings so far:
1. Update the status of each sub-question: "answered" if adequately covered, "unanswerable" if sources found nothing, or "investigating" if more work needed
2. Identify 0-3 NEW sub-questions that have emerged from the findings

Important: Only include sub-questions that are truly new and distinct from existing ones."""


class _SubQuestionDraft(BaseModel):
    question: str


class _PlanResponse(BaseModel):
    sub_questions: list[_SubQuestionDraft]


class _StatusUpdate(BaseModel):
    id: str
    status: str


class _PlanUpdateResponse(BaseModel):
    status_updates: list[_StatusUpdate]
    new_questions: list[str] = []


class Planner:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def create_plan(self, query: str, max_iterations: int = 10) -> ResearchPlan:
        prompt = PLANNING_PROMPT.format(query=query)
        response = await self._client.complete(prompt, _PlanResponse)

        sub_questions = [
            SubQuestion(id=f"sq_{i}", question=sq.question)
            for i, sq in enumerate(response.sub_questions)
        ]
        return ResearchPlan(query=query, sub_questions=sub_questions, max_iterations=max_iterations)

    async def update_plan(self, plan: ResearchPlan, new_findings: list[Finding], sources: list[Source]) -> ResearchPlan:
        sq_summary = "\n".join(
            f"- [{sq.id}] ({sq.status}) {sq.question}" for sq in plan.sub_questions
        )
        findings_summary = "\n".join(
            f"- {f.content[:200]}..." if len(f.content) > 200 else f"- {f.content}"
            for f in new_findings
        ) or "No new findings yet."

        prompt = PLAN_UPDATE_PROMPT.format(
            query=plan.query,
            sub_questions_summary=sq_summary,
            findings_summary=findings_summary,
            source_count=len(sources),
        )
        response = await self._client.complete(prompt, _PlanUpdateResponse)

        # Apply status updates
        status_map = {u.id: u.status for u in response.status_updates}
        for sq in plan.sub_questions:
            if sq.id in status_map and status_map[sq.id] in ("answered", "unanswerable", "investigating", "pending"):
                sq.status = status_map[sq.id]  # type: ignore[assignment]

        # Add new sub-questions
        next_id = len(plan.sub_questions)
        for q in response.new_questions:
            plan.sub_questions.append(SubQuestion(id=f"sq_{next_id}", question=q))
            next_id += 1

        plan.iteration += 1
        return plan
