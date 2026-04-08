from __future__ import annotations

import aiohttp

from open_deep_research.config import SearXNGConfig
from open_deep_research.models import SearchResult
from open_deep_research.providers.base import SearchProvider


class SearXNGProvider(SearchProvider):
    def __init__(self, config: SearXNGConfig) -> None:
        self._base_url = config.base_url.rstrip("/")

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        url = f"{self._base_url}/search"
        params = {"q": query, "format": "json", "pageno": "1"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
        except (aiohttp.ClientError, TimeoutError):
            return []

        results = []
        for r in data.get("results", [])[:num_results]:
            results.append(SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", ""),
            ))
        return results
