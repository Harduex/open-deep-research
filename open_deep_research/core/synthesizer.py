from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from open_deep_research.config import OutputConfig
from open_deep_research.core.verifier import Verifier
from open_deep_research.llm.client import LLMClient
from open_deep_research.models import (
    Finding,
    Report,
    ReportMetadata,
    ReportSection,
    ResearchPlan,
    Source,
    TokenBudget,
)

SECTION_GENERATION_PROMPT = """Write a detailed section for a research report addressing the following question.

Question: {question}

Findings:
{findings_with_sources}

Source Reference Map:
{source_map}

Instructions:
- Write in clear, professional prose
- Use numbered inline citations [1], [2], etc. matching the source reference map
- Synthesize information across sources rather than summarizing each individually
- Note any caveats or limitations in the evidence
- Do not include a section title (it will be added separately)"""


CONTRADICTION_DETECTION_PROMPT = """Review the following research findings and identify any direct contradictions between them.

Findings:
{all_findings_with_sources}

For each contradiction found, describe:
- What the conflicting claims are
- Which sources make each claim (by source ID)
- The nature of the disagreement

If no contradictions are found, return an empty list."""


EXECUTIVE_SUMMARY_PROMPT = """Write a 2-4 sentence executive summary for a research report on the following query.

Query: {query}

Report sections:
{sections_summary}

Write a concise overview capturing the most important findings. Do not use citations in the summary."""


class _ContradictionResponse(BaseModel):
    contradictions: list[str] = []


class Synthesizer:
    def __init__(self, client: LLMClient, config: OutputConfig, model_name: str = "") -> None:
        self._client = client
        self._config = config
        self._model_name = model_name
        self._verifier = Verifier(client)

    async def synthesize(
        self, plan: ResearchPlan, findings: list[Finding], sources: list[Source], budget: TokenBudget,
    ) -> Report:
        source_map = self._build_source_map(sources)

        # Generate sections for answered sub-questions
        sections: list[ReportSection] = []
        for sq in plan.sub_questions:
            if sq.status != "answered" or not sq.findings:
                continue
            section = await self._generate_section(sq.question, sq.findings, sources, source_map)
            sections.append(section)

        # Executive summary
        exec_summary = await self._generate_executive_summary(sections, plan.query)

        # Contradiction detection
        contradictions: list[str] = []
        if self._config.include_contradictions and findings:
            contradictions = await self._detect_contradictions(findings, sources)

        # Title
        title = await self._client.complete_text(
            f'Generate a concise, descriptive title (5-10 words) for a research report about: {plan.query}\n\nRespond with ONLY the title, no quotes or punctuation around it.',
            stage="title_generation",
        )

        return Report(
            title=title.strip().strip('"').strip("'"),
            executive_summary=exec_summary,
            sections=sections,
            contradictions=contradictions,
            sources=sources,
            metadata=ReportMetadata(
                generated_at=datetime.now(timezone.utc),
                model=self._model_name,
                total_sources=len(sources),
                total_iterations=plan.iteration,
                tokens_used=budget.used_tokens,
            ),
        )

    async def _generate_section(
        self, question: str, section_findings: list[Finding],
        sources: list[Source], source_map: str,
        max_regenerations: int = 1,
    ) -> ReportSection:
        findings_text = "\n".join(
            f"- (confidence: {f.confidence}, sources: {f.source_ids}) {f.content}"
            for f in section_findings
        )

        for attempt in range(1 + max_regenerations):
            prompt = SECTION_GENERATION_PROMPT.format(
                question=question,
                findings_with_sources=findings_text,
                source_map=source_map,
            )
            content = await self._client.complete_text(prompt, stage="section_generation")

            # Self-verification pass
            content, was_flawed = await self._verifier.verify_and_revise(content, section_findings, sources)
            if not was_flawed or attempt == max_regenerations:
                break
            # Fundamentally flawed — regenerate from scratch

        all_source_ids = []
        for f in section_findings:
            all_source_ids.extend(f.source_ids)

        return ReportSection(title=question, content=content, source_ids=list(set(all_source_ids)))

    async def _generate_executive_summary(self, sections: list[ReportSection], query: str) -> str:
        sections_summary = "\n".join(f"## {s.title}\n{s.content[:300]}..." for s in sections)
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(query=query, sections_summary=sections_summary)
        return await self._client.complete_text(prompt, stage="executive_summary")

    async def _detect_contradictions(self, findings: list[Finding], sources: list[Source]) -> list[str]:
        findings_text = "\n".join(
            f"- Finding (sources {f.source_ids}): {f.content}" for f in findings
        )
        prompt = CONTRADICTION_DETECTION_PROMPT.format(all_findings_with_sources=findings_text)
        try:
            response = await self._client.complete(prompt, _ContradictionResponse, stage="contradiction_detection")
            return response.contradictions
        except Exception:
            return []

    @staticmethod
    def _build_source_map(sources: list[Source]) -> str:
        return "\n".join(f"[{s.id}] {s.title} - {s.url}" for s in sources)


def format_report_markdown(report: Report) -> str:
    lines = [f"# {report.title}", "", "## Executive Summary", "", report.executive_summary, ""]

    for section in report.sections:
        lines.extend([f"## {section.title}", "", section.content, ""])

    if report.contradictions:
        lines.extend(["## Contradictions / Uncertainties", ""])
        for c in report.contradictions:
            lines.append(f"- {c}")
        lines.append("")

    lines.extend(["## Sources", ""])
    for s in report.sources:
        lines.append(f"{s.id}. [{s.title}]({s.url})")
    lines.append("")

    m = report.metadata
    lines.extend([
        "---",
        f"*Generated: {m.generated_at.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Model: {m.model} | Sources: {m.total_sources} | "
        f"Iterations: {m.total_iterations} | Tokens: {m.tokens_used:,}*",
    ])

    return "\n".join(lines)
