from __future__ import annotations

import asyncio

from ddgs import DDGS

from open_deep_research.models import SearchResult
from open_deep_research.providers.base import SearchProvider


class DuckDuckGoProvider(SearchProvider):
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        try:
            raw = await asyncio.to_thread(
                lambda: list(DDGS().text(query, max_results=num_results))
            )
        except Exception:
            return []

        return [
            SearchResult(url=r.get("href", ""), title=r.get("title", ""), snippet=r.get("body", ""))
            for r in raw
        ]
