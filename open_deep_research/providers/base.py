from __future__ import annotations

from abc import ABC, abstractmethod

from open_deep_research.models import SearchResult


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]: ...
