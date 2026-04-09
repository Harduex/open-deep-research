from __future__ import annotations

import asyncio

from pydantic import BaseModel

from open_deep_research.core.reader import Reader
from open_deep_research.llm.client import LLMClient
from open_deep_research.models import Finding, SearchResult, Source, SubQuestion
from open_deep_research.providers.base import SearchProvider

QUERY_GENERATION_PROMPT = """Generate 2-3 effective web search queries to investigate this research question.

Question: {question}

{existing_context}

The queries should:
- Use different angles/phrasings to maximize coverage
- Be concise and search-engine-friendly (use keywords, not full sentences)
- Avoid redundancy with previous searches if context is provided"""


FINDING_EXTRACTION_PROMPT = """You are analyzing a source to extract findings relevant to a research question.

Research question: {question}

Source [{source_id}] - {title}:
{snippet}

Extract the key finding from this source that is relevant to the research question. Assess your confidence in this finding.

If the source contains no relevant information, set is_relevant to false."""


class _QueryResponse(BaseModel):
    queries: list[str]


class _FindingResponse(BaseModel):
    content: str
    confidence: str = "medium"
    is_relevant: bool = True


class Searcher:
    def __init__(self, provider: SearchProvider, reader: Reader, client: LLMClient, max_sources: int = 30) -> None:
        self._provider = provider
        self._reader = reader
        self._client = client
        self._max_sources = max_sources

    async def search_sub_question(
        self, sq: SubQuestion, existing_sources: list[Source],
    ) -> tuple[list[Source], list[Finding]]:
        queries = await self._generate_queries(sq)

        # Search concurrently
        search_tasks = [self._provider.search(q) for q in queries]
        all_results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Flatten and deduplicate by URL
        seen_urls = {s.url for s in existing_sources}
        unique_results: list[SearchResult] = []
        for results in all_results_lists:
            if isinstance(results, Exception):
                continue
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)

        # Cap by remaining source budget
        remaining = self._max_sources - len(existing_sources)
        unique_results = unique_results[:max(0, remaining)]

        # Read concurrently
        next_id = max((s.id for s in existing_sources), default=0) + 1
        read_tasks = [
            self._reader.read(r, source_id=next_id + i, query_context=sq.question)
            for i, r in enumerate(unique_results)
        ]
        read_results = await asyncio.gather(*read_tasks, return_exceptions=True)

        new_sources: list[Source] = []
        for result in read_results:
            if isinstance(result, Source):
                new_sources.append(result)

        # Extract findings concurrently
        extract_tasks = [self._extract_finding(sq.question, source) for source in new_sources]
        extract_results = await asyncio.gather(*extract_tasks, return_exceptions=True)
        new_findings: list[Finding] = [r for r in extract_results if isinstance(r, Finding)]

        return new_sources, new_findings

    async def _generate_queries(self, sq: SubQuestion) -> list[str]:
        existing_ctx = ""
        if sq.findings:
            existing_ctx = "Previous findings:\n" + "\n".join(
                f"- {f.content[:150]}" for f in sq.findings[:5]
            )

        prompt = QUERY_GENERATION_PROMPT.format(
            question=sq.question,
            existing_context=existing_ctx,
        )
        try:
            response = await self._client.complete(prompt, _QueryResponse)
            return response.queries[:3]
        except Exception:
            # Fallback: use the question itself as a search query
            return [sq.question]

    async def _extract_finding(self, question: str, source: Source) -> Finding | None:
        prompt = FINDING_EXTRACTION_PROMPT.format(
            question=question,
            source_id=source.id,
            title=source.title,
            snippet=source.snippet[:2000],
        )
        try:
            response = await self._client.complete(prompt, _FindingResponse)
            if not response.is_relevant:
                return None
            confidence = response.confidence if response.confidence in ("high", "medium", "low") else "medium"
            return Finding(
                content=response.content,
                source_ids=[source.id],
                confidence=confidence,  # type: ignore[arg-type]
            )
        except Exception:
            return None
