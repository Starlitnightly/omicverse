from __future__ import annotations

"""
Build a new skill from a single link to cover functionality OmicVerse does not yet provide.

This scaffolds a SKILL.md and scrapes the provided URL (same-domain crawl)
to populate reference markdown files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _slugify(value: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if len(slug) > 64:
        slug = slug[:64].strip("-")
    return slug


def _ensure_unique_dir(base_dir: Path, slug: str) -> Path:
    target = base_dir / slug
    if not target.exists():
        return target
    i = 2
    while True:
        candidate = base_dir / f"{slug}-{i}"
        if not candidate.exists():
            return candidate
        i += 1


def _generate_skill_md(slug: str, title: str, description: str, link: str) -> str:
    fm: Dict[str, object] = {
        "name": slug,
        "title": title,
        "description": description,
        "_source_link": link,
        "_category": "experimental",
    }
    try:
        import yaml  # type: ignore
        fm_text = yaml.safe_dump(fm, sort_keys=False).strip()
    except Exception:
        import json as _json
        fm_text = (
            f"name: {slug}\n"
            f"title: {title}\n"
            f"description: {description}\n"
            f"_source_link: {link}\n"
            f"_category: experimental\n"
        )
    frontmatter = "---\n" + fm_text + "\n---\n\n"

    body = (
        f"# {title}\n\n"
        f"This skill helps prototype and explore a capability not yet built into OmicVerse, using the external reference below.\n\n"
        f"## Source\n- {link}\n\n"
        f"## Guidance\n"
        f"- Use reference snippets to answer questions and outline steps.\n"
        f"- Clearly mark assumptions versus verified information from the source.\n"
        f"- When suggesting code, prefer small, testable examples.\n"
        f"- If an API is unclear, propose experiments to validate behavior.\n\n"
        f"## References\nSee files under references/ for extracted documentation.\n"
    )
    return frontmatter + body


def build_from_link(link: str, output_root: Path, name: Optional[str] = None, description: Optional[str] = None, max_pages: int = 30) -> Path:
    from .docs_scraper import scrape

    # Derive name/slug
    title = name or link
    slug = _slugify(name) if name else _slugify(link)

    # Ensure target dir unique
    skill_dir = _ensure_unique_dir(output_root, slug)
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)

    # Scrape source
    files: List[Tuple[str, str]]
    try:
        files = scrape(link, max_pages=max_pages)
    except Exception as exc:
        files = [("source-error.md", f"Error scraping {link}: {exc}")]

    for fname, content in files:
        (refs_dir / fname).write_text(content, encoding="utf-8")

    # Write SKILL.md
    desc = description or f"Prototype skill for capability not yet in OmicVerse, sourced from {link}"
    skill_md = _generate_skill_md(slug=slug, title=title, description=desc, link=link)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    return skill_dir

