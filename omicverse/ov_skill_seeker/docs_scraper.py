from __future__ import annotations

"""
Simple documentation scraper for building OmicVerse skills from a base URL.

Design goals:
- Minimal dependencies: requests + beautifulsoup4 (optional)
- Conservative: same-domain crawl with a max_pages cap
- Extracts headings, text, and code blocks into lightweight markdown
- Security: validates URLs, sanitizes content, prevents traversal attacks

Note: Network access depends on runtime environment. This module is safe to
import when dependencies are missing; functions will raise informative errors.
"""

from typing import Iterable, List, Set, Tuple
from urllib.parse import urljoin, urlparse
from pathlib import Path

from .security import validate_url, sanitize_filename, sanitize_html_content, SecurityError


def _require_deps():
    try:
        import requests  # noqa: F401
        from bs4 import BeautifulSoup  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional
        raise RuntimeError(
            "Documentation scraping requires 'requests' and 'beautifulsoup4'."
        ) from exc


def _same_domain(url: str, base_netloc: str) -> bool:
    return urlparse(url).netloc == base_netloc


def _normalize_url(url: str) -> str:
    # Drop fragments; keep scheme/netloc/path/query
    p = urlparse(url)
    return p._replace(fragment="").geturl()


def _extract_markdown(html: str, url: str) -> str:
    from bs4 import BeautifulSoup

    # Sanitize HTML before processing
    html = sanitize_html_content(html)

    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title")
    md: List[str] = []
    if title and title.get_text(strip=True):
        md.append(f"# {title.get_text(strip=True)}")
    md.append(f"\n> Source: {url}\n")

    # Headings and paragraphs
    for tag in soup.find_all(["h1", "h2", "h3", "p", "pre", "code"]):
        name = tag.name or ""
        text = tag.get_text("\n", strip=True)
        if not text:
            continue
        if name in {"h1", "h2", "h3"}:
            level = int(name[1])
            md.append("#" * level + " " + text)
        elif name == "pre":
            md.append("\n```\n" + text + "\n```\n")
        elif name == "code":
            # Inline code: render as backticked line
            md.append("`" + text + "`")
        else:
            md.append(text)

    return "\n\n".join(md).strip()


def scrape(base_url: str, max_pages: int = 50) -> List[Tuple[str, str]]:
    """Crawl same-domain pages from base_url up to max_pages.

    Returns a list of (filename, markdown_content) tuples.

    Raises
    ------
    SecurityError
        If the base_url fails security validation
    """
    _require_deps()
    import requests

    # Validate the base URL
    try:
        validate_url(base_url)
    except SecurityError as e:
        raise SecurityError(f"Invalid base URL: {e}")

    base_url = _normalize_url(base_url)
    netloc = urlparse(base_url).netloc

    to_visit: List[str] = [base_url]
    seen: Set[str] = set()
    results: List[Tuple[str, str]] = []

    while to_visit and len(seen) < max_pages:
        url = to_visit.pop(0)
        url = _normalize_url(url)
        if url in seen:
            continue

        # Validate each URL before visiting
        try:
            validate_url(url)
        except SecurityError:
            # Skip invalid URLs silently
            continue

        seen.add(url)
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
        except Exception:
            continue

        md = _extract_markdown(resp.text, url)
        slug = urlparse(url).path.strip("/").replace("/", "-") or "index"
        # Sanitize the filename
        safe_slug = sanitize_filename(slug)
        filename = f"docs-{len(results)+1:03d}-{safe_slug}.md"
        results.append((filename, md))

        # Discover links
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            abs_url = urljoin(url, href)
            abs_url = _normalize_url(abs_url)
            if _same_domain(abs_url, netloc) and abs_url not in seen:
                to_visit.append(abs_url)

    return results

