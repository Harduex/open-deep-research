from __future__ import annotations

from open_deep_research.llm.client import LLMClient
from open_deep_research.models import Finding, IterationMetrics, ResearchPlan, Source, StoppingEvaluation, TokenBudget

STOPPING_EVALUATION_PROMPT = """You are evaluating whether a research investigation has gathered sufficient information to produce a comprehensive report.

Original query: {query}

Sub-questions and their status:
{sub_questions_summary}

Total findings: {finding_count}
Total sources: {source_count}
Current iteration: {iteration} of {max_iterations}
Token budget used: {budget_used}/{budget_max}

{metrics_section}
Consider:
1. Are the key sub-questions adequately answered?
2. Are new searches yielding diminishing returns (saturation)?
3. Is there enough material for a comprehensive report?
4. Are there critical gaps that must be filled?
5. If iteration metrics show declining new findings or increasing deduplication, saturation is likely."""


class Evaluator:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def evaluate_stopping(
        self, plan: ResearchPlan, findings: list[Finding], sources: list[Source], budget: TokenBudget,
        iteration_metrics: list[IterationMetrics] | None = None,
    ) -> StoppingEvaluation:
        # Hard ceilings
        if plan.iteration >= plan.max_iterations:
            return StoppingEvaluation(
                should_stop=True,
                reasoning="Hard iteration ceiling reached",
                coverage_score=self._estimate_coverage(plan),
                unanswered_questions=[sq.question for sq in plan.sub_questions if sq.status == "pending"],
            )

        if budget.is_exceeded:
            return StoppingEvaluation(
                should_stop=True,
                reasoning="Token budget exceeded",
                coverage_score=self._estimate_coverage(plan),
                unanswered_questions=[sq.question for sq in plan.sub_questions if sq.status == "pending"],
            )

        # Algorithmic saturation: 3 consecutive iterations with 0 new findings
        if iteration_metrics and len(iteration_metrics) >= 3:
            last_three = iteration_metrics[-3:]
            if all(m.new_findings_count == 0 for m in last_three):
                return StoppingEvaluation(
                    should_stop=True,
                    reasoning="Saturation detected: 3 consecutive iterations with no new findings",
                    coverage_score=self._estimate_coverage(plan),
                    saturation_detected=True,
                    unanswered_questions=[sq.question for sq in plan.sub_questions if sq.status == "pending"],
                )

        sq_summary = "\n".join(
            f"- [{sq.id}] ({sq.status}) {sq.question} — {len(sq.findings)} findings"
            for sq in plan.sub_questions
        )

        # Build metrics section for the prompt
        metrics_section = ""
        if iteration_metrics:
            metrics_lines = ["Iteration metrics (recent):"]
            for m in iteration_metrics[-5:]:
                metrics_lines.append(
                    f"  Iteration {m.iteration}: {m.new_findings_count} new findings, "
                    f"{m.new_sources_count} new sources, {m.dedup_removed_count} duplicates removed"
                )
            metrics_section = "\n".join(metrics_lines)

        prompt = STOPPING_EVALUATION_PROMPT.format(
            query=plan.query,
            sub_questions_summary=sq_summary,
            finding_count=len(findings),
            source_count=len(sources),
            iteration=plan.iteration,
            max_iterations=plan.max_iterations,
            budget_used=budget.used_tokens,
            budget_max=budget.max_tokens,
            metrics_section=metrics_section,
        )
        return await self._client.complete(prompt, StoppingEvaluation)

    @staticmethod
    def _estimate_coverage(plan: ResearchPlan) -> float:
        if not plan.sub_questions:
            return 0.0
        answered = sum(1 for sq in plan.sub_questions if sq.status == "answered")
        return answered / len(plan.sub_questions)
