from __future__ import annotations

from pydantic import BaseModel

from open_deep_research.llm.client import LLMClient
from open_deep_research.models import Finding, Source


VERIFICATION_PROMPT = """You are a research report verifier. Review the following draft section against the source material and assess its accuracy.

Draft section:
{draft}

Source material used:
{source_material}

Evaluate:
1. Are all claims supported by the sources?
2. Are citations used correctly (matching the right source)?
3. Are there any factual errors or misrepresentations?
4. Is anything important from the sources missing?

Return your verdict and specific issues found."""


REVISION_PROMPT = """Revise the following draft section based on the verification feedback.

Original draft:
{draft}

Verification feedback:
{feedback}

Source material:
{source_material}

Instructions:
- Fix all identified issues
- Keep the same general structure and tone
- Ensure all claims have correct citations
- Do not add information not present in the sources
- Return ONLY the revised text, no preamble"""


class VerificationResult(BaseModel):
    verdict: str  # "correct", "minor_fixes", "fundamentally_flawed"
    issues: list[str] = []
    feedback: str = ""


class Verifier:
    def __init__(self, client: LLMClient, max_revisions: int = 2) -> None:
        self._client = client
        self._max_revisions = max_revisions

    async def verify_and_revise(
        self, draft: str, findings: list[Finding], sources: list[Source],
    ) -> tuple[str, bool]:
        """Verify and revise a draft. Returns (text, was_fundamentally_flawed)."""
        source_material = self._format_sources(findings, sources)

        current_draft = draft
        for _ in range(self._max_revisions):
            result = await self._verify(current_draft, source_material)

            if result.verdict == "correct":
                return current_draft, False

            if result.verdict == "fundamentally_flawed":
                # Signal caller to regenerate from scratch
                return current_draft, True

            if result.verdict == "minor_fixes":
                current_draft = await self._revise(current_draft, result.feedback, source_material)
                continue

            # Unknown verdict, return as-is
            return current_draft, False

        return current_draft, False

    async def _verify(self, draft: str, source_material: str) -> VerificationResult:
        prompt = VERIFICATION_PROMPT.format(draft=draft, source_material=source_material)
        try:
            return await self._client.complete(prompt, VerificationResult)
        except Exception:
            return VerificationResult(verdict="correct")

    async def _revise(self, draft: str, feedback: str, source_material: str) -> str:
        prompt = REVISION_PROMPT.format(
            draft=draft, feedback=feedback, source_material=source_material,
        )
        return await self._client.complete_text(prompt)

    @staticmethod
    def _format_sources(findings: list[Finding], sources: list[Source]) -> str:
        source_map = {s.id: s for s in sources}
        lines = []
        for f in findings:
            src_refs = ", ".join(
                f"[{sid}] {source_map[sid].title}" for sid in f.source_ids if sid in source_map
            )
            lines.append(f"- ({f.confidence}) {f.content}\n  Sources: {src_refs}")
        return "\n".join(lines)
