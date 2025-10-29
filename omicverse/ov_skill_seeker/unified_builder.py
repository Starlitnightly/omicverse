from __future__ import annotations

"""
Unified Skill Builder for OmicVerse.

Given a JSON config (validated by config_validator), this module gathers
content from documentation, GitHub, and PDFs into a skill directory with
SKILL.md and reference markdown files.

Security: validates output paths and sanitizes generated filenames.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .config_validator import validate_config
from omicverse.utils.skill_registry import SkillRegistry  # for slug helpers only
from .security import validate_output_path, sanitize_filename, SecurityError


def _slugify(value: str) -> str:
    # Keep in sync with SkillRegistry _slugify behavior
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if len(slug) > 64:
        slug = slug[:64].strip("-")
    return slug


@dataclass
class BuildConfig:
    name: str
    description: str
    sources: List[Dict]

    @classmethod
    def from_json(cls, path: Path) -> "BuildConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_config(data)
        return cls(name=data["name"], description=data["description"], sources=data["sources"])


def _write_files(target_dir: Path, files: List[Tuple[str, str]]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in files:
        # Sanitize filename before writing
        safe_fname = sanitize_filename(fname)
        (target_dir / safe_fname).write_text(content, encoding="utf-8")


def _generate_skill_md(slug: str, title: str, description: str, sources: List[Dict]) -> str:
    # Include source summary in frontmatter for traceability
    fm: Dict[str, object] = {
        "name": slug,  # slug identifier
        "title": title,
        "description": description,
        "_sources": sources,
    }
    try:
        import yaml  # type: ignore
        fm_text = yaml.safe_dump(fm, sort_keys=False).strip()
    except Exception:
        # Minimal manual YAML if PyYAML not available
        import json as _json
        fm_text = (
            f"name: {slug}\n"
            f"title: {title}\n"
            f"description: {description}\n"
            f"_sources: {_json.dumps(sources)}"
        )
    frontmatter = "---\n" + fm_text + "\n---\n\n"
    body = f"# {title}\n\n" \
           f"This skill was built from the configured sources. Review the references/ files for detailed content."\
           "\n\n## How to Use\n- Ask targeted questions; this skill includes documentation excerpts and repository metadata.\n\n"
    return frontmatter + body


def build_from_config(config_path: Path, output_root: Path) -> Path:
    # Validate output_root to prevent directory traversal
    try:
        output_root = validate_output_path(output_root, base_dir=Path.cwd(), create=True)
    except SecurityError as e:
        raise SecurityError(f"Invalid output directory: {e}")

    cfg = BuildConfig.from_json(config_path)
    raw_slug = _slugify(cfg.name)
    # Sanitize the slug to ensure it's a safe directory name
    slug = sanitize_filename(raw_slug)
    skill_dir = output_root / slug

    # Collect references
    references: List[Tuple[str, str]] = []
    for src in cfg.sources:
        t = src.get("type")
        if t == "documentation":
            from .docs_scraper import scrape
            base_url = src["base_url"]
            max_pages = int(src.get("max_pages", 50))
            try:
                references += scrape(base_url, max_pages=max_pages)
            except Exception as exc:
                references.append((f"docs-error.md", f"Error scraping {base_url}: {exc}"))
        elif t == "github":
            from .github_scraper import extract
            repo = src["repo"]
            try:
                references += extract(repo)
            except Exception as exc:
                references.append((f"github-error.md", f"Error extracting {repo}: {exc}"))
        elif t == "pdf":
            from .pdf_scraper import extract as pdf_extract
            # Validate and expand PDF path
            p = Path(src["path"]).expanduser()
            try:
                # Ensure PDF path is within reasonable bounds (not absolute traversal)
                if p.is_absolute():
                    # For absolute paths, just check they exist and are readable
                    if not p.exists() or not p.is_file():
                        raise SecurityError(f"PDF path does not exist or is not a file: {p}")
                references += pdf_extract(p)
            except Exception as exc:
                references.append((f"pdf-error.md", f"Error extracting {p}: {exc}"))

    # Write outputs
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "references").mkdir(exist_ok=True)
    _write_files(skill_dir / "references", references)

    skill_md = _generate_skill_md(slug=slug, title=cfg.name, description=cfg.description, sources=cfg.sources)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    return skill_dir
