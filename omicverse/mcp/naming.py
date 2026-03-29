"""
Canonical MCP tool naming for OmicVerse registry entries.

Generates stable ``ov.<domain>.<name>`` identifiers from registry full_names.
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Domain mapping
# ---------------------------------------------------------------------------

MODULE_TO_DOMAIN: Dict[str, str] = {
    "omicverse.pp": "pp",
    "omicverse.pl": "pl",
    "omicverse.single": "single",
    "omicverse.bulk": "bulk",
    "omicverse.space": "space",
    "omicverse.utils.biocontext": "biocontext",
    "omicverse.utils": "utils",
    "omicverse.external": "external",
    "omicverse.alignment": "alignment",
}

_TOOL_NAME_RE = re.compile(r"^ov\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tool_name(full_name: str, category: Optional[str] = None) -> str:
    """Return canonical MCP tool name ``ov.<domain>.<symbol>``.

    Parameters
    ----------
    full_name : str
        Registry ``full_name``, e.g. ``"omicverse.pp._preprocess.pca"``.
    category : str, optional
        Registry category (unused currently, domain derived from module path).
    """
    from .overrides import get_name_override

    override = get_name_override(full_name)
    if override is not None:
        return override

    domain = _extract_domain(full_name)
    symbol = full_name.rsplit(".", 1)[-1]
    name = normalize_symbol_name(symbol)
    return f"ov.{domain}.{name}"


def normalize_symbol_name(name: str) -> str:
    """Normalize a Python symbol into a stable lowercase tool-name fragment.

    Examples
    --------
    >>> normalize_symbol_name("featureCount")
    'featurecount'
    >>> normalize_symbol_name("Cal_Spatial_Net")
    'cal_spatial_net'
    >>> normalize_symbol_name("anndata_to_GPU")
    'anndata_to_gpu'
    """
    return name.lower()


def resolve_name_collision(
    tool_name: str,
    entry: dict,
    seen: Dict[str, dict],
) -> str:
    """Disambiguate *tool_name* when it already exists in *seen*.

    Appends the penultimate module segment to break ties deterministically.
    """
    parts = entry["full_name"].rsplit(".", 2)
    if len(parts) >= 3:
        mod_segment = parts[-2].lstrip("_").lower()
        candidate = f"{tool_name}_{mod_segment}"
    else:
        candidate = f"{tool_name}_alt"

    # If still colliding, append numeric suffix
    idx = 2
    base = candidate
    while candidate in seen:
        candidate = f"{base}_{idx}"
        idx += 1
    return candidate


def build_search_aliases(entry: dict) -> List[str]:
    """Aggregate all searchable names for an entry."""
    names: set[str] = set()
    names.add(entry.get("short_name", "").lower())
    names.add(entry.get("full_name", "").lower())
    for alias in entry.get("aliases", []):
        names.add(alias.lower())
    names.discard("")
    return sorted(names)


def validate_tool_name(tool_name: str) -> None:
    """Raise ``ValueError`` if *tool_name* violates naming rules."""
    if not _TOOL_NAME_RE.match(tool_name):
        raise ValueError(
            f"Invalid MCP tool name {tool_name!r}: must match ov.<domain>.<name> "
            f"(lowercase alphanumeric + underscores)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_domain(full_name: str) -> str:
    """Derive the MCP domain from a registry full_name."""
    for prefix, domain in MODULE_TO_DOMAIN.items():
        if full_name.startswith(prefix + ".") or full_name.startswith(prefix + "_"):
            return domain
    # Fallback: second segment
    parts = full_name.split(".")
    if len(parts) >= 2:
        return parts[1].lower()
    return "core"
