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
- "pubmed": Uses NCBI E-utilities (esearch/esummary) to retrieve
  PubMed records. Recommended to set `NCBI_API_KEY` and optionally
  `NCBI_EMAIL` for higher rate limits and polite usage.

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
        ncbi_api_key: Optional[str] = None,
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
        self.ncbi_api_key = ncbi_api_key or os.getenv("NCBI_API_KEY")
        self.ncbi_email = os.getenv("NCBI_EMAIL") or os.getenv("CROSSREF_MAILTO") or os.getenv("CROSSREF_EMAIL")
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

        if self.backend not in {"tavily", "duckduckgo", "brave", "pubmed"}:
            raise ValueError("backend must be 'tavily', 'duckduckgo', 'brave' or 'pubmed'")

        if self.backend == "tavily" and not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required for Tavily backend")
        if self.backend == "brave" and not self.brave_api_key:
            raise ValueError("BRAVE_API_KEY is required for Brave backend")
        # For PubMed, an API key is optional but recommended

    # ------------------------------------------------------------------
    def search(self, query: str) -> Sequence[WebDocument]:
        if self.backend == "tavily":
            results = self._search_tavily(query)
        elif self.backend == "brave":
            results = self._search_brave(query)
        elif self.backend == "pubmed":
            results = self._search_pubmed(query)
        else:
            results = self._search_duckduckgo(query)

        docs: List[WebDocument] = []
        # Prefetch PubMed abstracts (efetch) when requested
        pubmed_details: Dict[str, Dict[str, Any]] = {}
        if self.backend == "pubmed" and self.fetch_content:
            pmids = [str(r.get("pmid")) for r in results[: self.max_results] if r.get("pmid")]
            if pmids:
                try:
                    pubmed_details = self._efetch_pubmed_details(pmids)
                except Exception:
                    pubmed_details = {}
        for idx, r in enumerate(results[: self.max_results]):
            url = r.get("url") or r.get("href")
            if not url:
                continue
            title = r.get("title") or r.get("text") or url
            snippet = r.get("snippet") or r.get("body") or ""
            pmid = r.get("pmid")
            doi = r.get("doi")
            pmcid = r.get("pmcid")
            meta_extra: Dict[str, Any] = {}

            if self.fetch_content:
                if self.backend == "pubmed" and pmid:
                    det = pubmed_details.get(str(pmid), {})
                    text = det.get("abstract") or snippet or title
                    doi = doi or det.get("doi")
                    pmcid = pmcid or det.get("pmcid")
                    oa_url = self._resolve_open_access(doi=doi, pmcid=pmcid)
                    proxy = self._build_proxy_url(url)
                    meta_extra.update({
                        **({"proxy_url": proxy} if proxy else {}),
                        **({"open_access_url": oa_url} if oa_url else {}),
                        "access_required": False,
                    })
                else:
                    text, acc = self._fetch_text(url)
                    # Build proxy and OA links
                    oa_url = self._resolve_open_access(doi=doi)
                    proxy = self._build_proxy_url(url)
                    meta_extra.update({
                        "access_required": acc.get("access_required", False),
                        "http_status": acc.get("http_status"),
                        **({"login_detected": True} if acc.get("login_detected") else {}),
                        **({"proxy_url": proxy} if proxy else {}),
                        **({"open_access_url": oa_url} if oa_url else {}),
                    })
            else:
                text = snippet or title
                oa_url = self._resolve_open_access(doi=doi, pmcid=pmcid)
                proxy = self._build_proxy_url(url)
                meta_extra.update({
                    **({"proxy_url": proxy} if proxy else {}),
                    **({"open_access_url": oa_url} if oa_url else {}),
                })

            docs.append(
                WebDocument(
                    id=f"web:{idx}",
                    text=text,
                    metadata={
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "backend": self.backend,
                        **({"pmid": pmid} if pmid else {}),
                        **({"doi": doi} if doi else {}),
                        **({"pmcid": pmcid} if pmcid else {}),
                        **meta_extra,
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
    def _fetch_text(self, url: str) -> tuple[str, Dict[str, Any]]:
        try:
            headers = {"User-Agent": self.user_agent}
            resp = self.session.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            access_meta = {"access_required": False, "http_status": resp.status_code}
            if "text/html" not in content_type:
                return resp.text[:10_000], access_meta
            if BeautifulSoup is None:
                return resp.text[:10_000], access_meta

            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove scripts and style
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            # Prefer main/article if present
            main = soup.find("main") or soup.find("article") or soup.body
            text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
            # Heuristic: detect paywall language
            low = (text or "").lower()
            hints = ["purchase", "subscribe", "institutional access", "login", "sign in", "get access", "rent", "buy"]
            if any(h in low for h in hints) and len(text) < 1000:
                access_meta["access_required"] = True
                access_meta["login_detected"] = True
            return text[:20_000], access_meta
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            return "", {"access_required": status in {401, 402, 403}, "http_status": status}
        except Exception:
            return "", {"access_required": False, "http_status": None}

    def _build_proxy_url(self, url: str) -> Optional[str]:
        import urllib.parse as up
        tmpl = os.getenv("INSTITUTION_PROXY_URL")
        if not tmpl:
            return None
        try:
            enc = up.quote(url, safe="")
            if "{url}" in tmpl:
                return tmpl.replace("{url}", enc)
            # Common pattern: login?url=
            if tmpl.endswith("=") or tmpl.endswith("url="):
                return f"{tmpl}{enc}"
            # Fallback: append as query param
            sep = "&" if ("?" in tmpl) else "?"
            return f"{tmpl}{sep}url={enc}"
        except Exception:
            return None

    def _resolve_open_access(self, doi: Optional[str] = None, pmcid: Optional[str] = None) -> Optional[str]:
        # Prefer PMC if available
        if pmcid:
            pmcid = pmcid.strip().upper()
            if not pmcid.startswith("PMC"):
                pmcid = "PMC" + pmcid
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        # Try Unpaywall when configured
        email = os.getenv("UNPAYWALL_EMAIL")
        if doi and email:
            try:  # pragma: no cover - network optional
                url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
                r = self.session.get(url, timeout=min(self.timeout, 15))
                r.raise_for_status()
                data = r.json() or {}
                oa = data.get("best_oa_location") or {}
                # Prefer landing page or URL for PDF
                return oa.get("url_for_landing_page") or oa.get("url")
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    def _search_pubmed(self, query: str) -> List[Dict[str, Any]]:
        """Search PubMed via NCBI E-utilities and return normalized results.

        This uses `esearch` to get PMIDs and `esummary` to fetch metadata.
        We build canonical PubMed URLs and provide a short snippet.
        If `fetch_content` is enabled, caller may later fetch and parse the HTML page.
        """
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        # Parse special tokens like after:YYYY and site:domain from the incoming query
        clean_query, dp_filter = self._parse_pubmed_query(query)
        term = clean_query if not dp_filter else (f"({clean_query}) AND {dp_filter}" if clean_query.strip() else dp_filter)
        params = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": str(self.max_results),
            "term": term,
        }
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        if self.ncbi_email:
            params["email"] = self.ncbi_email
            params["tool"] = "OmicVerse-DR"

        # 1) esearch
        r = self.session.get(f"{base}/esearch.fcgi", params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json() or {}
        idlist = ((data.get("esearchresult") or {}).get("idlist") or [])
        if not idlist:
            return []

        # 2) esummary for metadata
        params2 = {
            "db": "pubmed",
            "retmode": "json",
            "id": ",".join(idlist[: self.max_results]),
        }
        if self.ncbi_api_key:
            params2["api_key"] = self.ncbi_api_key
        if self.ncbi_email:
            params2["email"] = self.ncbi_email
            params2["tool"] = "OmicVerse-DR"
        r2 = self.session.get(f"{base}/esummary.fcgi", params=params2, timeout=self.timeout)
        r2.raise_for_status()
        sdata = r2.json() or {}
        result = sdata.get("result", {})
        uids = result.get("uids", [])

        out: List[Dict[str, Any]] = []
        for uid in uids[: self.max_results]:
            rec = result.get(uid, {}) or {}
            title = rec.get("title") or f"PMID {uid}"
            journal = rec.get("fulljournalname") or rec.get("source") or ""
            pubdate = rec.get("pubdate") or rec.get("epubdate") or ""
            eloc = rec.get("elocationid") or ""
            snippet = "; ".join([p for p in [journal, pubdate, eloc] if p])
            url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
            # try to extract DOI from elocationid if present
            doi = None
            if isinstance(eloc, str) and "10." in eloc.lower():
                # naive DOI extraction
                import re
                m = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", eloc, flags=re.I)
                if m:
                    doi = m.group(0)
            out.append({"title": title, "url": url, "snippet": snippet, "pmid": uid, **({"doi": doi} if doi else {})})
        return out

    def _parse_pubmed_query(self, query: str) -> tuple[str, str]:
        """Transform generic query tokens into PubMed-specific date filters.

        - Strips occurrences of 'site:...'
        - Recognizes 'after:YYYY[-MM[-DD]]' and 'before:YYYY[-MM[-DD]]'
        - Returns (clean_query, dp_filter) where dp_filter is a string like
          '("YYYY/MM/DD"[dp] : "YYYY/MM/DD"[dp])' or empty if none.
        """
        import re
        q = query or ""
        # Remove site:domain tokens
        q = re.sub(r"\bsite:[^\s]+", "", q, flags=re.I)

        def _parse_date(token: str) -> tuple[int, int, int] | None:
            m = re.search(rf"\b{token}:(\d{{4}})(?:-(\d{{2}}))?(?:-(\d{{2}}))?\b", q, flags=re.I)
            if not m:
                return None
            y = int(m.group(1))
            mo = int(m.group(2)) if m.group(2) else (1 if token == "after" else 12)
            dy = int(m.group(3)) if m.group(3) else (1 if token == "after" else 31)
            return y, mo, dy

        after = _parse_date("after")
        before = _parse_date("before")
        # Strip the tokens from the query text
        q = re.sub(r"\bafter:\d{4}(?:-\d{2})?(?:-\d{2})?\b", "", q, flags=re.I)
        q = re.sub(r"\bbefore:\d{4}(?:-\d{2})?(?:-\d{2})?\b", "", q, flags=re.I)
        clean = re.sub(r"\s+", " ", q).strip()

        if not after and not before:
            return clean, ""

        def _fmt(y: int, m: int, d: int) -> str:
            return f"{y:04d}/{m:02d}/{d:02d}"

        start = _fmt(after[0], after[1], after[2]) if after else "1000/01/01"
        end = _fmt(before[0], before[1], before[2]) if before else "3000/12/31"
        dp = f'("{start}"[dp] : "{end}"[dp])'
        return clean, dp

    def _efetch_pubmed_details(self, pmids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch abstracts and DOIs for given PMIDs via efetch (XML)."""
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "id": ",".join(pmids),
        }
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        if self.ncbi_email:
            params["email"] = self.ncbi_email
            params["tool"] = "OmicVerse-DR"
        r = self.session.get(f"{base}/efetch.fcgi", params=params, timeout=self.timeout)
        r.raise_for_status()
        xml = r.text
        import xml.etree.ElementTree as ET
        out: Dict[str, Dict[str, Any]] = {}
        try:
            root = ET.fromstring(xml)
        except Exception:
            return out
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//MedlineCitation/PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
            if not pmid:
                continue
            # gather abstract text
            abs_parts: List[str] = []
            for ab in art.findall(".//Abstract/AbstractText"):
                # join text including nested tags
                txt = "".join(ab.itertext()).strip()
                if txt:
                    abs_parts.append(txt)
            abstract = "\n\n".join(abs_parts)[:20000] if abs_parts else ""
            # try to get DOI/PMCID from ArticleIdList
            doi = None
            pmcid = None
            for aid in art.findall(".//ArticleIdList/ArticleId"):
                idtype = aid.get("IdType", "").lower()
                if idtype == "doi" and aid.text:
                    doi = aid.text.strip()
                if idtype == "pmc" and aid.text:
                    pmcid = aid.text.strip()
            out[pmid] = {"abstract": abstract, **({"doi": doi} if doi else {}), **({"pmcid": pmcid} if pmcid else {})}
        return out
