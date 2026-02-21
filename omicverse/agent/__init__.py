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
                if p.name in {".DS_Store"} or "__pycache__" in p.parts or p.suffix == ".zip":
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
    cwd = Path.cwd()
    is_multi = isinstance(links, (list, tuple))
    first_link = links[0] if is_multi and links else links
    title = name or (str(first_link) if isinstance(first_link, str) else "skill")
    slug = _slugify(title)

    # Resolve base output location
    if target == "skills":
        base_out = cwd / ".claude" / "skills"
    else:
        base_out = Path(out_dir) if out_dir else (cwd / "output")
    base_out.mkdir(parents=True, exist_ok=True)

    if not is_multi and isinstance(links, str):
        # Single-link builder
        from omicverse.ov_skill_seeker.link_builder import build_from_link
        skill_dir = build_from_link(
            link=links,
            output_root=base_out,
            name=title,
            description=description,
            max_pages=max_pages,
        )
    else:
        # Multi-link unified build: synthesize a minimal config
        from omicverse.ov_skill_seeker.unified_builder import build_from_config
        cfg = {
            "name": title,
            "description": description
            or "Prototype skill from multiple links for a capability not yet in OmicVerse",
            "sources": [
                {"type": "documentation", "base_url": str(u), "max_pages": max_pages}
                for u in links  # type: ignore[arg-type]
            ],
        }
        tmp_cfg = base_out / f"{slug}.config.json"
        tmp_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            skill_dir = build_from_config(tmp_cfg, base_out)
        finally:
            try:
                tmp_cfg.unlink()
            except Exception:
                pass

    result = {
        "slug": skill_dir.name,
        "skill_dir": str(skill_dir),
    }

    if package:
        zip_root = Path(package_dir) if package_dir else (out_dir if out_dir else (cwd / "output"))
        zip_root = Path(zip_root)
        zip_path = zip_root / f"{skill_dir.name}.zip"
        zip_path = _zip_dir(skill_dir, zip_path)
        result["zip"] = str(zip_path)

    return result


# ---------------------------------------------------------------------------
# MCP public API
# ---------------------------------------------------------------------------

def mcp_connect(
    name: str,
    *,
    url: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Connect to an MCP server and return its tool inventory.

    Parameters
    ----------
    name : str
        Human-readable name for the server.
    url : str, optional
        HTTP(S) endpoint (Streamable HTTP transport).
    command : str, optional
        Executable for stdio transport (e.g. ``"uvx"``).
    args : list[str], optional
        Arguments for the stdio command.

    Returns
    -------
    dict
        ``{"name", "tools", "tool_count", "transport"}``.

    Examples
    --------
    >>> import omicverse as ov
    >>> info = ov.agent.mcp_connect("biocontext",
    ...     url="https://mcp.biocontext.ai/mcp/")
    >>> print(info["tools"][:3])
    """
    from omicverse.utils.mcp_client import MCPClientManager

    mgr = MCPClientManager()
    server = mgr.connect(name, url=url, command=command, args=args)
    return {
        "name": server.name,
        "tools": [t.name for t in server.tools],
        "tool_count": len(server.tools),
        "transport": server.transport,
    }


def biocontext(
    tool_name: str,
    arguments: Optional[Dict[str, object]] = None,
    mode: str = "remote",
) -> object:
    """Quick one-shot query to BioContext MCP.

    Parameters
    ----------
    tool_name : str
        Name of the BioContext MCP tool to call.
    arguments : dict, optional
        Arguments for the tool.
    mode : str
        Connection mode: ``"remote"`` (default), ``"local"``, or ``"auto"``.

    Returns
    -------
    object
        Parsed tool result (dict, list, or string).

    Examples
    --------
    >>> import omicverse as ov
    >>> result = ov.agent.biocontext("string_interaction_partners",
    ...     {"identifiers": "TP53", "species": 9606})
    """
    from omicverse.utils.biocontext_bridge import BioContextBridge

    bridge = BioContextBridge(mode=mode)
    bridge.connect()
    return bridge.query(tool_name, arguments or {})


__all__ = ["seeker", "mcp_connect", "biocontext"]

