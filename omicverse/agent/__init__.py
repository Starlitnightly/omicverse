from __future__ import annotations

"""
OmicVerse Agent utilities

Public API
----------
seeker(links, *, name=None, description=None, max_pages=30, target="skills", out_dir=None, package=False, package_dir=None)
    Create a Claude Agent skill from a link (or list of links) directly from Jupyter.

Examples
--------
>>> import omicverse as ov
>>> ov.agent.seeker("https://example.com/docs/feature", name="New Analysis")
{'slug': 'new-analysis', 'skill_dir': '.../.claude/skills/new-analysis'}

>>> ov.agent.seeker([
...   "https://docs.site-a.com/",
...   "https://docs.site-b.com/guide"
... ], name="multi-source", package=True)
{'slug': 'multi-source', 'skill_dir': '.../output/multi-source', 'zip': '.../output/multi-source.zip'}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


def _slugify(value: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if len(slug) > 64:
        slug = slug[:64].strip("-")
    return slug


def _zip_dir(skill_dir: Path, zip_path: Path) -> Path:
    import zipfile
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(skill_dir.rglob("*")):
            if p.is_file():
                if p.name in {".DS_Store"} or "__pycache__" in p.parts:
                    continue
                zf.write(p, p.relative_to(skill_dir))
    return zip_path


def seeker(
    links: Union[str, List[str]],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    max_pages: int = 30,
    target: str = "skills",
    out_dir: Optional[Union[str, Path]] = None,
    package: bool = False,
    package_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Create a new skill from a link (or links) for quick prototyping.

    .. deprecated::
        Use ``ov.Agent.seeker()`` instead. This function will be removed in a future version.

    Parameters
    ----------
    links : str | list[str]
        A starting URL or a list of URLs to crawl (same-domain for each).
    name : str, optional
        Display title; also used to derive slug. Defaults to the first link.
    description : str, optional
        Frontmatter description. Default mentions the source links.
    max_pages : int, default 30
        Per-source page crawl cap for documentation scraping.
    target : {"skills", "output"}, default "skills"
        Where to write the created skill: `.claude/skills` or `./output`.
    out_dir : str | Path, optional
        Explicit output directory root when target="output".
    package : bool, default False
        If True, also create a `.zip` adjacent to the skill directory.
    package_dir : str | Path, optional
        Directory to place the `.zip`; defaults to `out_dir or ./output`.

    Returns
    -------
    dict
        Keys: slug, skill_dir, optionally zip.
    """
    import warnings
    warnings.warn(
        "ov.agent.seeker is deprecated; use ov.Agent.seeker instead",
        DeprecationWarning,
        stacklevel=2
    )

    # Import and forward to Agent.seeker
    from omicverse.utils.smart_agent import Agent
    return Agent.seeker(
        links,
        name=name,
        description=description,
        max_pages=max_pages,
        target=target,
        out_dir=out_dir,
        package=package,
        package_dir=package_dir,
    )


__all__ = ["seeker"]

