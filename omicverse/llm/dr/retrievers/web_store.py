"""Live web search-backed VectorStore.

This module provides a `WebRetrieverStore` that conforms to the
`VectorStore` protocol used by `ResearchAgent`. It performs a web
search and optional page fetching to return lightweight document
objects with `id`, `text`, and `metadata` fields.

Backends supported:

- "tavily": Uses the Tavily Search API (`TAVILY_API_KEY` required).
- "duckduckgo": Uses the `duckduckgo_search` package if available,
  otherwise falls back to parsing DuckDuckGo HTML results.
- "brave": Uses Brave Search API (`BRAVE_API_KEY` required).

The implementation is dependency-light and degrades gracefully when
optional libraries are not installed. For reliability and quality,
prefer the Tavily backend if you have an API key.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import requests

try:  # optional dependency
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional
    BeautifulSoup = None  # type: ignore


@dataclass
class WebDocument:
    """Lightweight document object returned by the web retriever."""

    id: str
    text: str
    metadata: Dict[str, Any]


class WebRetrieverStore:
    """VectorStore-like interface backed by live web search.

    Parameters
    ----------
    backend: str
        One of {"tavily", "duckduckgo", "brave"}. Defaults to "duckduckgo".
    max_results: int
        Maximum number of search results to use.
    fetch_content: bool
        If True, fetch and extract text from each result URL.
        If False, use snippets from the search engine (when available).
    request_timeout: int
        HTTP timeout in seconds.
    user_agent: str | None
        Custom User-Agent header for page fetches.
    tavily_api_key: str | None
        API key for Tavily. If None, taken from `TAVILY_API_KEY` env.
    brave_api_key: str | None
        API key for Brave. If None, taken from `BRAVE_API_KEY` env.
    cache: bool
        If True and `requests_cache` is installed, cache HTTP responses.
    cache_ttl: int
        Cache expiration in seconds (if cache enabled).
    """

    def __init__(
        self,
        *,
        backend: str = "duckduckgo",
        max_results: int = 5,
        fetch_content: bool = True,
        request_timeout: int = 20,
        user_agent: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        cache: bool = False,
        cache_ttl: int = 600,
    ) -> None:
        self.backend = backend.lower()
        self.max_results = max_results
        self.fetch_content = fetch_content
        self.timeout = request_timeout
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
        )
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        self.cache_enabled = cache or bool(os.getenv("OV_DR_CACHE"))
        self.cache_ttl = cache_ttl

        # Per-instance session (optionally cached)
        try:
            if self.cache_enabled:
                import requests_cache  # type: ignore

                self.session = requests_cache.CachedSession(
                    cache_name="ov_dr_cache",
                    backend="sqlite",
                    expire_after=self.cache_ttl,
                )
            else:  # pragma: no cover - trivial
                self.session = requests.Session()
        except Exception:  # pragma: no cover - optional
            self.session = requests.Session()

        if self.backend not in {"tavily", "duckduckgo", "brave"}:
            raise ValueError("backend must be 'tavily', 'duckduckgo' or 'brave'")

        if self.backend == "tavily" and not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required for Tavily backend")
        if self.backend == "brave" and not self.brave_api_key:
            raise ValueError("BRAVE_API_KEY is required for Brave backend")

    # ------------------------------------------------------------------
    def search(self, query: str) -> Sequence[WebDocument]:
        if self.backend == "tavily":
            results = self._search_tavily(query)
        elif self.backend == "brave":
            results = self._search_brave(query)
        else:
            results = self._search_duckduckgo(query)

        docs: List[WebDocument] = []
        for idx, r in enumerate(results[: self.max_results]):
            url = r.get("url") or r.get("href")
            if not url:
                continue
            title = r.get("title") or r.get("text") or url
            snippet = r.get("snippet") or r.get("body") or ""

            if self.fetch_content:
                text = self._fetch_text(url) or snippet or title
            else:
                text = snippet or title

            docs.append(
                WebDocument(
                    id=f"web:{idx}",
                    text=text,
                    metadata={
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "backend": self.backend,
                    },
                )
            )
        return docs

    # ------------------------------------------------------------------
    def _search_tavily(self, query: str) -> List[Dict[str, Any]]:
        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": self.max_results,
            "include_answer": False,
        }
        resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("results", []))

    def _search_brave(self, query: str) -> List[Dict[str, Any]]:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key or "",
            "User-Agent": self.user_agent,
        }
        params = {"q": query, "count": self.max_results}
        resp = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        results: List[Dict[str, Any]] = []
        for item in (data.get("web", {}) or {}).get("results", [])[: self.max_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("description") or "",
                }
            )
        return results

    def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        # Prefer python package if present for stability
        try:  # pragma: no cover - optional
            from duckduckgo_search import DDGS  # type: ignore

            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=self.max_results))
            # Normalize fields
            results: List[Dict[str, Any]] = []
            for h in hits:
                results.append(
                    {
                        "title": h.get("title"),
                        "url": h.get("href") or h.get("url"),
                        "snippet": h.get("body") or h.get("snippet"),
                    }
                )
            return results
        except Exception:
            pass

        # Fallback: parse HTML results page (brittle, but no extra deps)
        params = {"q": query}
        headers = {"User-Agent": self.user_agent}
        resp = self.session.get("https://duckduckgo.com/html/", params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        html = resp.text
        results: List[Dict[str, Any]] = []
        if BeautifulSoup is None:
            return results
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a.result__a"):
            href = a.get("href")
            title = a.get_text(strip=True)
            snippet_el = a.find_parent("div", class_="result__body")
            snippet = ""
            if snippet_el is not None:
                sn = snippet_el.find("a", class_="result__snippet")
                if sn is not None:
                    snippet = sn.get_text(strip=True)
            results.append({"title": title, "url": href, "snippet": snippet})
            if len(results) >= self.max_results:
                break
        return results

    # ------------------------------------------------------------------
    def _fetch_text(self, url: str) -> str:
        try:
            headers = {"User-Agent": self.user_agent}
            resp = self.session.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return resp.text[:10_000]
            if BeautifulSoup is None:
                return resp.text[:10_000]

            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove scripts and style
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            # Prefer main/article if present
            main = soup.find("main") or soup.find("article") or soup.body
            text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
            return text[:20_000]
        except Exception:
            return ""
