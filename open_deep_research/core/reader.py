from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
import trafilatura
from pydantic import BaseModel

from open_deep_research.llm.client import LLMClient
from open_deep_research.models import SearchResult, Source

SOURCE_SUMMARY_PROMPT = """Summarize the following web page content in approximately {max_tokens} tokens.

Preserve:
- Key facts, statistics, and numerical data
- Names of people, organizations, and places
- Dates and timelines
- Direct quotes that are particularly relevant
- Any conclusions or key arguments

Source URL: {url}
Source Title: {title}

Content:
{content}

Write a dense, factual summary. Do not include preamble or meta-commentary."""

_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
_MAX_EXTRACTED_LINKS = 20


class ReadResult(BaseModel):
    source: Source | None = None
    extracted_urls: list[str] = []


class Reader:
    def __init__(self, client: LLMClient, summary_tokens: int = 500) -> None:
        self._client = client
        self._summary_tokens = summary_tokens

    async def read(self, result: SearchResult, source_id: int, query_context: str = "") -> ReadResult:
        html = await self._fetch_html(result.url)
        if not html:
            return ReadResult()

        extracted_urls = self._extract_links(html, result.url)

        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if not text:
            return ReadResult(extracted_urls=extracted_urls)

        token_estimate = int(len(text.split()) * 1.3)
        if token_estimate > 1000:
            snippet = await self._summarize(text, result.url, result.title)
        else:
            snippet = text

        source = Source(id=source_id, url=result.url, title=result.title, snippet=snippet)
        return ReadResult(source=source, extracted_urls=extracted_urls)

    async def _fetch_html(self, url: str) -> str | None:
        headers = {"User-Agent": "OpenDeepResearch/0.1 (research agent)"}
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        return None
                    # Cap at 5MB
                    content = await resp.text(errors="replace")
                    return content[:5_000_000]
        except (aiohttp.ClientError, TimeoutError, UnicodeDecodeError):
            return None

    async def _summarize(self, text: str, url: str, title: str) -> str:
        # Truncate very long text to avoid blowing context
        words = text.split()
        if len(words) > 5000:
            text = " ".join(words[:5000])

        prompt = SOURCE_SUMMARY_PROMPT.format(
            max_tokens=self._summary_tokens,
            url=url,
            title=title,
            content=text,
        )
        return await self._client.complete_text(prompt)

    @staticmethod
    def _extract_links(html: str, base_url: str) -> list[str]:
        """Extract unique http/https URLs from HTML href attributes."""
        seen: set[str] = set()
        urls: list[str] = []
        for match in _HREF_RE.finditer(html):
            raw = match.group(1).strip()
            resolved = urljoin(base_url, raw)
            parsed = urlparse(resolved)
            if parsed.scheme not in ("http", "https"):
                continue
            # Strip fragment, normalize
            clean = urlunparse(parsed._replace(fragment=""))
            if clean not in seen:
                seen.add(clean)
                urls.append(clean)
            if len(urls) >= _MAX_EXTRACTED_LINKS:
                break
        return urls
