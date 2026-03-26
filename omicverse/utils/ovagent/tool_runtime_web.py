"""Web tool handlers: fetch, search, and download.

Extracted from ``tool_runtime.py`` during Phase 3 decomposition.
These are pure functions with no ``ToolRuntime`` or ``AgentContext``
dependency — they only use stdlib networking.
"""

from __future__ import annotations

import os
from typing import List, Optional


def handle_web_fetch(
    url: str, prompt: Optional[str] = None, timeout: int = 15
) -> str:
    """Fetch a URL and return readable text content."""
    import urllib.error
    import urllib.request
    from html.parser import HTMLParser

    class _HTMLToText(HTMLParser):
        def __init__(self):
            super().__init__()
            self._pieces: list = []
            self._skip = False
            self._skip_tags = {
                "script", "style", "noscript", "svg", "head",
            }

        def handle_starttag(self, tag, attrs):
            if tag in self._skip_tags:
                self._skip = True
            elif tag in ("br", "hr"):
                self._pieces.append("\n")
            elif tag in (
                "p", "div", "tr", "li",
                "h1", "h2", "h3", "h4", "h5", "h6",
            ):
                self._pieces.append("\n")

        def handle_endtag(self, tag):
            if tag in self._skip_tags:
                self._skip = False
            elif tag in (
                "p", "div", "tr",
                "h1", "h2", "h3", "h4", "h5", "h6",
            ):
                self._pieces.append("\n")

        def handle_data(self, data):
            if not self._skip:
                self._pieces.append(data)

        def get_text(self) -> str:
            raw = "".join(self._pieces)
            lines = []
            for line in raw.splitlines():
                stripped = " ".join(line.split())
                if stripped:
                    lines.append(stripped)
            return "\n".join(lines)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "OmicVerseAgent/1.0 (research bot)",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw_bytes = resp.read(512_000)
            charset = "utf-8"
            if "charset=" in content_type:
                charset = (
                    content_type.split("charset=")[-1]
                    .split(";")[0]
                    .strip()
                )
            body = raw_bytes.decode(charset, errors="replace")

        if "html" in content_type.lower() or body.strip().startswith("<"):
            parser = _HTMLToText()
            parser.feed(body)
            text = parser.get_text()
        else:
            text = body

        max_chars = 4000 if prompt else 6000
        if len(text) > max_chars:
            text = (
                text[:max_chars] + "\n\n... [truncated, page too long]"
            )

        header = f"Content from {url}:\n\n"
        if prompt:
            header = f"Content from {url} (focus: {prompt}):\n\n"
        return header + text

    except urllib.error.HTTPError as e:
        return f"HTTP error fetching {url}: {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"URL error fetching {url}: {e.reason}"
    except Exception as e:
        return f"Error fetching {url}: {type(e).__name__}: {e}"


def handle_web_search(query: str, num_results: int = 5) -> str:
    """Search the web via DuckDuckGo HTML and return results."""
    import re as _re
    import urllib.error
    import urllib.parse
    import urllib.request

    num_results = max(1, min(int(num_results), 10))
    encoded_q = urllib.parse.urlencode({"q": query})
    url = f"https://html.duckduckgo.com/html/?{encoded_q}"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "OmicVerseAgent/1.0 (research bot)",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read(256_000).decode("utf-8", errors="replace")
    except Exception as e:
        return f"Search error: {type(e).__name__}: {e}"

    results: List[str] = []
    result_blocks = _re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        r".*?"
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        body,
        _re.DOTALL,
    )

    if not result_blocks:
        links = _re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            body,
        )
        for href, title in links[:num_results]:
            clean_title = _re.sub(r"<[^>]+>", "", title).strip()
            actual_url = href
            uddg_match = _re.search(r"uddg=([^&]+)", href)
            if uddg_match:
                actual_url = urllib.parse.unquote(uddg_match.group(1))
            results.append(f"- {clean_title}\n  {actual_url}")
    else:
        for href, title, snippet in result_blocks[:num_results]:
            clean_title = _re.sub(r"<[^>]+>", "", title).strip()
            clean_snippet = _re.sub(r"<[^>]+>", "", snippet).strip()
            actual_url = href
            uddg_match = _re.search(r"uddg=([^&]+)", href)
            if uddg_match:
                actual_url = urllib.parse.unquote(uddg_match.group(1))
            results.append(
                f"- {clean_title}\n  {actual_url}\n  {clean_snippet}"
            )

    if not results:
        return f"No results found for '{query}'."

    return (
        f"Search results for '{query}':\n\n" + "\n\n".join(results)
    )


def handle_web_download(
    url: str,
    filename: Optional[str] = None,
    directory: Optional[str] = None,
) -> str:
    """Download a file from a URL to disk."""
    import urllib.error
    import urllib.parse
    import urllib.request

    MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

    if not filename:
        path_part = urllib.parse.urlparse(url).path
        filename = os.path.basename(path_part) or "downloaded_file"

    save_dir = directory or os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "OmicVerseAgent/1.0 (research bot)",
            },
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_SIZE:
                size_gb = int(content_length) / (1024**3)
                return (
                    f"File too large ({size_gb:.1f} GB). "
                    "Max allowed is 2 GB."
                )

            downloaded = 0
            chunk_size = 1024 * 1024
            with open(save_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_SIZE:
                        f.close()
                        os.remove(save_path)
                        return (
                            "Download aborted: exceeded 2GB limit at "
                            f"{downloaded / (1024**3):.1f} GB."
                        )
                    f.write(chunk)

        size_bytes = os.path.getsize(save_path)
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            size_str = f"{size_bytes / (1024**2):.1f} MB"
        else:
            size_str = f"{size_bytes / (1024**3):.2f} GB"

        return (
            f"Downloaded successfully:\n"
            f"  File: {save_path}\n"
            f"  Size: {size_str}\n"
            f"You can now load this file with execute_code, e.g.:\n"
            f"  adata = ov.read('{save_path}')"
        )

    except urllib.error.HTTPError as e:
        return f"HTTP error downloading {url}: {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"URL error downloading {url}: {e.reason}"
    except Exception as e:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass
        return f"Download error: {type(e).__name__}: {e}"
