"""Lightweight source verification and quality scoring utilities.

This module provides heuristics to score and optionally verify references.
Network checks are disabled by default; enable with OV_DR_VERIFY_NETWORK=1.
"""

from __future__ import annotations

from typing import Dict, Iterable, List
import os
import re

from ..research.agent import SourceCitation


TRUSTED_DOMAINS = [
    "pubmed", "ncbi", "nih.gov", "arxiv.org", "nature.com", "science.org", "cell.com",
    "springer.com", "sciencedirect.com", "wiley.com", "ieee.org",
]


class SourceVerifier:
    def __init__(self, *, allow_network: bool | None = None, timeout: int = 6, enrich: bool | None = None) -> None:
        self.allow_network = bool(int(os.getenv("OV_DR_VERIFY_NETWORK", "0"))) if allow_network is None else allow_network
        self.timeout = timeout
        # Enable metadata enrichment (Crossref/PubMed) when set or when keys are present
        if enrich is None:
            self.enrich = bool(int(os.getenv("OV_DR_ENRICH", "0"))) or bool(os.getenv("NCBI_API_KEY")) or bool(os.getenv("CROSSREF_MAILTO") or os.getenv("CROSSREF_EMAIL"))
        else:
            self.enrich = enrich
        self._session = None

    def score(self, meta: Dict) -> int:
        url = (meta or {}).get("url", "")
        host = url.split("/")[2] if "://" in url else url
        host = host.lower()
        for dom in TRUSTED_DOMAINS:
            if dom in host:
                return 3
        return 1

    def _has_valid_doi(self, meta: Dict) -> bool:
        doi = (meta or {}).get("doi", "")
        return bool(re.match(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", str(doi), flags=re.I))

    def _network_verify(self, url: str) -> bool:
        if not self.allow_network or not url:
            return False
        try:  # pragma: no cover - network optional
            import requests

            resp = requests.head(url, timeout=self.timeout, allow_redirects=True)
            return resp.status_code < 400
        except Exception:
            return False

    def verify_citation(self, c: SourceCitation) -> SourceCitation:
        meta = dict(c.metadata or {})
        meta["quality_score"] = self.score(meta)
        if self._has_valid_doi(meta):
            meta["verified"] = "doi"
        elif self._network_verify(meta.get("url", "")):
            meta["verified"] = "url"
        else:
            meta["verified"] = "none"
        # Optional enrichment (Crossref/PubMed) if network use is allowed and keys/context available
        try:
            net_ok = self.allow_network or self.enrich
            if net_ok:
                doi = meta.get("doi")
                # Crossref enrichment by DOI
                if doi:
                    cr = self._enrich_from_crossref(str(doi))
                    if cr:
                        meta.update(cr)
                # PubMed enrichment by DOI or PMID (requires NCBI_API_KEY)
                pmid = meta.get("pmid")
                if os.getenv("NCBI_API_KEY") and (doi or pmid):
                    pm = self._enrich_from_pubmed(doi=str(doi) if doi else None, pmid=str(pmid) if pmid else None)
                    if pm:
                        meta.update(pm)
        except Exception:
            pass
        return SourceCitation(source_id=c.source_id, content=c.content, metadata=meta)

    def verify_all(self, citations: Iterable[SourceCitation]) -> List[SourceCitation]:
        return [self.verify_citation(c) for c in citations]

    # -------------------------- Enrichment helpers --------------------------
    def _session_get(self):
        if self._session is None:
            try:
                import requests

                self._session = requests.Session()
            except Exception:  # pragma: no cover
                self._session = None
        return self._session

    def _enrich_from_crossref(self, doi: str) -> Dict:
        try:
            sess = self._session_get()
            if sess is None:
                return {}
            headers = {
                "User-Agent": f"OmicVerse-DR/1.0 (+https://github.com/Starlitnightly/omicverse)"
            }
            mail = os.getenv("CROSSREF_MAILTO") or os.getenv("CROSSREF_EMAIL")
            if mail:
                headers["User-Agent"] += f" mailto:{mail}"
            url = f"https://api.crossref.org/works/{doi}"
            resp = sess.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            msg = (resp.json().get("message") or {})
            title = ", ".join(msg.get("title", [])).strip() if isinstance(msg.get("title"), list) else msg.get("title") or ""
            journal = ""
            if isinstance(msg.get("container-title"), list):
                journal = ", ".join(msg.get("container-title", [])).strip()
            else:
                journal = msg.get("container-title") or ""
            year = None
            issued = msg.get("issued", {}).get("date-parts", [[None]])
            if issued and issued[0] and issued[0][0]:
                year = issued[0][0]
            return {
                "title": title or None,
                "journal": journal or None,
                "year": year,
                "crossref_enriched": True,
            }
        except Exception:
            return {}

    def _enrich_from_pubmed(self, doi: str | None = None, pmid: str | None = None) -> Dict:
        api_key = os.getenv("NCBI_API_KEY")
        if not api_key:
            return {}
        try:  # pragma: no cover - depends on network
            sess = self._session_get()
            if sess is None:
                return {}
            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            the_pmid = pmid
            if not the_pmid and doi:
                # Find PMID via esearch by DOI
                q = f"{doi}".replace(" ", "+")
                esearch = f"{base}/esearch.fcgi?db=pubmed&retmode=json&term={q}&api_key={api_key}"
                r = sess.get(esearch, timeout=self.timeout)
                r.raise_for_status()
                ids = (r.json().get("esearchresult", {}) or {}).get("idlist", [])
                if ids:
                    the_pmid = ids[0]
            if not the_pmid:
                return {}
            # Get summary
            esum = f"{base}/esummary.fcgi?db=pubmed&retmode=json&id={the_pmid}&api_key={api_key}"
            r2 = sess.get(esum, timeout=self.timeout)
            r2.raise_for_status()
            data = r2.json().get("result", {})
            uid = data.get("uids", [None])[0]
            if not uid:
                return {}
            rec = data.get(uid, {})
            title = rec.get("title")
            journal = rec.get("fulljournalname") or rec.get("source")
            pubdate = rec.get("pubdate") or rec.get("epubdate")
            return {
                "pmid": the_pmid,
                "title": title or None,
                "journal": journal or None,
                "date": pubdate or None,
                "pubmed_enriched": True,
            }
        except Exception:
            return {}
