from __future__ import annotations

from open_deep_research.config import SearchConfig
from open_deep_research.providers.base import SearchProvider
from open_deep_research.providers.duckduckgo import DuckDuckGoProvider
from open_deep_research.providers.searxng import SearXNGProvider


def create_provider(config: SearchConfig) -> SearchProvider:
    if config.provider == "searxng":
        return SearXNGProvider(config.searxng)
    if config.provider == "duckduckgo":
        return DuckDuckGoProvider()
    raise ValueError(f"Unknown search provider: {config.provider}")
