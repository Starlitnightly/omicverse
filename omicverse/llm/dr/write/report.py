"""Simple utilities for composing research reports with citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..research.agent import SourceCitation


@dataclass
class Section:
    """A single report section.

    Parameters
    ----------
    title:
        Heading for the section.
    text:
        Main body text of the section.
    citations:
        Citations supporting the section's content.
    """

    title: str
    text: str
    citations: Sequence[SourceCitation]


class ReportWriter:
    """Compose sections into a formatted report.

    The writer can render output in either Markdown or HTML. Citations are
    injected as superscripts (HTML) or bracketed numbers (Markdown) and a
    reference list is appended to the end of the document.
    """

    def __init__(self, fmt: str = "markdown") -> None:
        self.fmt = fmt.lower()

    def compose(self, sections: Sequence[Section]) -> str:
        """Render ``sections`` into a single report string."""

        parts: List[str] = []
        references: List[Tuple[int, SourceCitation]] = []
        counter = 1
        for section in sections:
            body, new_refs, counter = self._inject_citations(
                section.text, section.citations, counter
            )
            parts.append(self._format_section(section.title, body))
            references.extend(new_refs)

        if references:
            parts.append(self._format_references(references))
        return "\n\n".join(parts)

    def _inject_citations(
        self,
        text: str,
        sources: Sequence[SourceCitation],
        start: int,
    ) -> Tuple[str, List[Tuple[int, SourceCitation]], int]:
        """Append citation markers to ``text`` and collect references."""

        body = text
        collected: List[Tuple[int, SourceCitation]] = []
        idx = start
        for src in sources:
            marker = self._format_marker(idx)
            body += marker
            collected.append((idx, src))
            idx += 1
        return body, collected, idx

    def _format_section(self, title: str, body: str) -> str:
        """Format a titled section in the target output format."""

        if self.fmt == "html":
            return f"<h2>{title}</h2>\n<p>{body}</p>"
        return f"## {title}\n{body}"

    def _format_marker(self, idx: int) -> str:
        """Return the citation marker for ``idx``."""

        if self.fmt == "html":
            return f"<sup>{idx}</sup>"
        return f"[{idx}]"

    def _format_references(self, citations: Sequence[Tuple[int, SourceCitation]]) -> str:
        """Format the reference list with light normalization and dedup."""

        def normalize(c: SourceCitation) -> str:
            # Try to build a friendly reference when metadata exists
            md = c.metadata or {}
            title = md.get("title") or ""
            url = md.get("url") or ""
            doi = md.get("doi") or ""
            date = md.get("date") or md.get("year") or ""
            core = title.strip() or c.content.strip()
            extra = []
            if date:
                extra.append(str(date))
            if doi:
                extra.append(f"doi:{doi}")
            if url:
                extra.append(url)
            if extra:
                return f"{core} ({', '.join(extra)})"
            return core

        # Deduplicate by DOI or URL or normalized text (case-insensitive)
        seen_keys = set()
        norm_items: List[Tuple[int, str]] = []
        for num, c in citations:
            md = c.metadata or {}
            key = (md.get("doi") or md.get("url") or normalize(c)).strip().lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            norm_items.append((num, normalize(c)))

        if self.fmt == "html":
            items = "\n".join(f"<li>{txt}</li>" for _, txt in norm_items)
            return f"<h2>References</h2>\n<ol>\n{items}\n</ol>"
        items = "\n".join(f"[{num}] {txt}" for num, txt in norm_items)
        return f"## References\n{items}"
