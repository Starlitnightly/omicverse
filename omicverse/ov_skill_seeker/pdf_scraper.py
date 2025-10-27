from __future__ import annotations

"""
Lightweight PDF extractor using PyMuPDF (fitz) if available.

Falls back to a clear error if dependency is missing.
"""

from pathlib import Path
from typing import List, Tuple


def extract(path: Path) -> List[Tuple[str, str]]:
    """Extract text from a PDF file into markdown chunks by page.

    Returns list of (filename, markdown_content).
    """
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - optional
        raise RuntimeError("PDF extraction requires PyMuPDF (fitz)") from exc

    doc = fitz.open(str(path))
    results: List[Tuple[str, str]] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        md = f"# Page {i}\n\n" + (text or "")
        results.append((f"pdf-{i:03d}.md", md))
    doc.close()
    return results

