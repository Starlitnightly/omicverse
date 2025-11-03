from __future__ import annotations

"""
Config validator for the OmicVerse Skill Seeker unified build flow.

Supports a simplified unified format like:
{
  "name": "omicverse",
  "description": "OmicVerse documentation + GitHub",
  "sources": [
    {"type": "documentation", "base_url": "https://omicverse.readthedocs.io/", "max_pages": 50},
    {"type": "github", "repo": "Starlitnightly/omicverse", "include_code": true},
    {"type": "pdf", "path": "path/to/file.pdf"}
  ]
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List


VALID_SOURCE_TYPES = {"documentation", "github", "pdf"}


def load_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")
    return data


def validate_config(cfg: Dict[str, Any]) -> None:
    # Required top-level fields
    for field in ("name", "description", "sources"):
        if field not in cfg:
            raise ValueError(f"Missing required field: {field}")

    sources = cfg["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("'sources' must be a non-empty list")

    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            raise ValueError(f"Source {i} must be an object")
        t = src.get("type")
        if t not in VALID_SOURCE_TYPES:
            raise ValueError(f"Source {i}: invalid type '{t}'")
        if t == "documentation" and "base_url" not in src:
            raise ValueError("documentation source missing 'base_url'")
        if t == "github" and "repo" not in src:
            raise ValueError("github source missing 'repo'")
        if t == "pdf" and "path" not in src:
            raise ValueError("pdf source missing 'path'")

