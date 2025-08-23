"""Embedding-backed web retriever with chunking, re-ranking, and dedup.

This retriever integrates live web search (via ``WebRetrieverStore``) with
optional robust content extraction and in-memory embedding search to select
the most relevant passages for a query.

Dependencies are optional; when unavailable, the retriever degrades
gracefully to the basic web search results without embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import concurrent.futures
import time
import re

import requests

try:  # optional, improves extraction quality
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None  # type: ignore

try:  # optional
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore


@dataclass
class PassageDoc:
    id: str
    text: str
    metadata: Dict[str, Any]


class EmbedWebRetriever:
    """Web retriever combining search + extraction + embedding rerank."""

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
        # Embedding search parameters
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        max_concurrency: int = 4,
    ) -> None:
        from .web_store import WebRetrieverStore

        self.web = WebRetrieverStore(
            backend=backend,
            max_results=max_results,
            fetch_content=fetch_content,
            request_timeout=request_timeout,
            user_agent=user_agent,
            tavily_api_key=tavily_api_key,
            brave_api_key=brave_api_key,
            cache=cache,
        )
        self.chunk_size = max(200, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.top_k = top_k
        self.max_concurrency = max(1, min(16, max_concurrency))
        self.session = getattr(self.web, "session", None)

    # ------------------------------------------------------------------
    def search(self, query: str) -> Sequence[PassageDoc]:
        # 1) Web search
        results = list(self.web.search(query))
        if not results:
            return []

        # 2) Extract text (robust) with simple concurrency
        def extract(url: str, fallback_text: str) -> str:
            return self._robust_fetch_and_extract(url, fallback_text)

        passages: List[PassageDoc] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futs = []
            for idx, r in enumerate(results):
                url = r.metadata.get("url") if hasattr(r, "metadata") else None
                snippet = r.metadata.get("snippet") if hasattr(r, "metadata") else ""
                title = r.metadata.get("title") if hasattr(r, "metadata") else None
                if not url:
                    text = getattr(r, "text", "") or snippet
                    for j, ch in enumerate(self._chunk(text)):
                        passages.append(
                            PassageDoc(
                                id=f"passage:{idx}:{j}",
                                text=ch,
                                metadata={"url": None, "title": title or r.metadata.get("title", ""), "source_index": idx},
                            )
                        )
                    continue
                futs.append((idx, url, title or r.metadata.get("title", ""), ex.submit(extract, url, getattr(r, "text", "") or snippet)))

            for idx, url, title, fut in futs:
                try:
                    text = fut.result(timeout=30)
                except Exception:
                    text = ""
                for j, ch in enumerate(self._chunk(text)):
                    passages.append(
                        PassageDoc(
                            id=f"passage:{idx}:{j}",
                            text=ch,
                            metadata={"url": url, "title": title, "source_index": idx},
                        )
                    )

        # 3) Embed + rerank (optional)
        ranked = self._embed_and_rank(query, passages)

        # 4) Deduplicate by URL, keep top passages
        seen: set = set()
        deduped: List[PassageDoc] = []
        for p in ranked:
            key = p.metadata.get("url") or p.text[:100]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
            if len(deduped) >= self.top_k:
                break
        return deduped

    # ------------------------------------------------------------------
    def _robust_fetch_and_extract(self, url: str, fallback: str) -> str:
        # retries with simple backoff
        headers = {"User-Agent": getattr(self.web, "user_agent", "Mozilla/5.0")}
        for i in range(3):
            try:
                # Prefer shared session if available (enables caching)
                sess = self.session or requests
                resp = sess.get(url, headers=headers, timeout=getattr(self.web, "timeout", 20))
                resp.raise_for_status()
                html = resp.text
                # prefer trafilatura if available
                if trafilatura is not None:
                    extracted = trafilatura.extract(html) or ""
                else:
                    extracted = self._basic_html_extract(html)
                if extracted:
                    return extracted
            except Exception:
                time.sleep(0.5 * (i + 1))
        return fallback

    @staticmethod
    def _basic_html_extract(html: str) -> str:
        if BeautifulSoup is None:
            return html[:20_000]
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.extract()
        main = soup.find("main") or soup.find("article") or soup.body
        text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
        return text[:20_000]

    def _chunk(self, text: str) -> Sequence[str]:
        if not text:
            return []
        # sentence-aware-ish split, then merge to size
        sents = re.split(r"(?<=[.!?])\s+", text)
        out: List[str] = []
        buf = ""
        for s in sents:
            if len(buf) + len(s) + 1 <= self.chunk_size:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    out.append(buf)
                # overlap
                if self.chunk_overlap and out:
                    tail = out[-1][-self.chunk_overlap :]
                else:
                    tail = ""
                buf = (tail + " " + s).strip()
        if buf:
            out.append(buf)
        return out

    def _embed_and_rank(self, query: str, passages: Sequence[PassageDoc]) -> Sequence[PassageDoc]:
        # Try Chroma + GPT4All embeddings; fallback to naive ranking
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions

            ef = embedding_functions.DefaultEmbeddingFunction()
            client = chromadb.Client(Settings())
            coll = client.create_collection(name=f"tmp_web_{int(time.time()*1000)}", embedding_function=ef)
            ids = [p.id for p in passages]
            texts = [p.text for p in passages]
            metas = [p.metadata for p in passages]
            coll.add(ids=ids, documents=texts, metadatas=metas)
            res = coll.query(query_texts=[query], n_results=min(self.top_k * 3, max(5, len(passages))))
            # Map back to PassageDoc in order
            id_to_doc = {p.id: p for p in passages}
            ordered: List[PassageDoc] = []
            for i in res.get("ids", [[]])[0]:
                if i in id_to_doc:
                    ordered.append(id_to_doc[i])
            return ordered or passages
        except Exception:
            # naive scoring by query term frequency
            terms = [t for t in re.split(r"\W+", query.lower()) if t]
            def score(p: PassageDoc) -> int:
                txt = p.text.lower()
                return sum(txt.count(t) for t in terms)
            return sorted(passages, key=score, reverse=True)
