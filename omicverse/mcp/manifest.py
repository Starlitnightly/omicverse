"""
Manifest generation from the OmicVerse function registry.

The manifest is the single internal representation consumed by the MCP server,
schema generator, executor, and adapters.  It is built entirely from
``_global_registry`` — no hand-maintained parallel tool list.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import warnings
from typing import Any, Dict, List, Optional

from . import naming, availability as avail_mod, overrides as overrides_mod
from .schema import build_parameter_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry hydration
# ---------------------------------------------------------------------------

_HYDRATED = False


def ensure_registry_populated():
    """Import modules containing ``@register_function`` decorators.

    OmicVerse uses lazy imports, so the decorators that populate
    ``_global_registry`` only fire when their host modules are loaded.
    This function imports exactly the modules listed in
    ``PHASE_WHITELIST`` so the registry contains all P0/P0.5 tools.

    Some package ``__init__.py`` files pull in heavy optional deps
    (e.g. ``single/__init__.py`` imports ``_cefcon`` → ``torch_geometric``).
    When the normal import fails, we fall back to loading the leaf file
    directly via ``importlib.util`` to bypass the ``__init__`` chain.

    Safe to call multiple times (idempotent).
    """
    global _HYDRATED
    if _HYDRATED:
        return
    _HYDRATED = True

    from .overrides import PHASE_WHITELIST

    modules: set = set()
    for phase_names in PHASE_WHITELIST.values():
        for full_name in phase_names:
            # "omicverse.pp._preprocess.scale" → "omicverse.pp._preprocess"
            modules.add(full_name.rsplit(".", 1)[0])

    for mod_path in sorted(modules):
        try:
            importlib.import_module(mod_path)
        except Exception as exc:
            # Package __init__ may drag in heavy deps or runtime-only failures.
            # Try loading the leaf file directly.
            if not _try_load_leaf_module(mod_path):
                logger.warning("Could not import %s: %s", mod_path, exc)


def _try_load_leaf_module(mod_path: str) -> bool:
    """Load a leaf module file directly, bypassing its package __init__.

    For example, ``omicverse.single._markers`` is loaded from
    ``omicverse/single/_markers.py`` without executing
    ``omicverse/single/__init__.py`` (which may import heavy deps).
    """
    import importlib.util
    import sys

    if mod_path in sys.modules:
        return True

    # Resolve file path from the parent package that IS importable
    parts = mod_path.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        parent_path = ".".join(parts[:depth])
        try:
            parent = importlib.import_module(parent_path)
        except Exception:
            continue

        # Build the .py path for the remaining segments
        parent_dir = os.path.dirname(getattr(parent, "__file__", "") or "")
        if not parent_dir:
            continue
        leaf_file = os.path.join(parent_dir, *parts[depth:]) + ".py"
        if not os.path.isfile(leaf_file):
            # Might be a package directory
            leaf_file = os.path.join(parent_dir, *parts[depth:], "__init__.py")
        if not os.path.isfile(leaf_file):
            continue

        try:
            spec = importlib.util.spec_from_file_location(mod_path, leaf_file)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_path] = mod
            spec.loader.exec_module(mod)
            return True
        except Exception as exc:
            logger.warning("Direct load of %s failed: %s", mod_path, exc)
            sys.modules.pop(mod_path, None)
            return False

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_registry_manifest(
    registry=None,
    overrides=None,
    phase: Optional[str] = None,
) -> List[dict]:
    """Build the full MCP manifest from the global registry.

    Parameters
    ----------
    registry : FunctionRegistry, optional
        Registry instance.  Defaults to ``_global_registry``.
    overrides : module, optional
        Overrides module.  Defaults to ``mcp.overrides``.
    phase : str, optional
        If given, only return entries enabled in this rollout phase
        (e.g. ``"P0"``).  Pass ``"P0+P0.5"`` for both waves.

    Returns
    -------
    list[dict]
        Manifest entries sorted by tool_name.
    """
    if registry is None:
        ensure_registry_populated()
        from .._registry import _global_registry
        registry = _global_registry
    if overrides is None:
        overrides = overrides_mod

    # Step 1: extract unique entries
    entries = iter_registry_entries(registry)

    # Step 2: dedupe
    entries = dedupe_registry_entries(entries)

    # Step 3: build full manifest entries
    seen_names: Dict[str, dict] = {}
    manifest: List[dict] = []

    for entry in entries:
        item = build_manifest_entry(entry, overrides, seen_names)
        if item is not None:
            manifest.append(item)
            seen_names[item["tool_name"]] = item

    # Step 4: sort for deterministic output
    manifest.sort(key=lambda e: e["tool_name"])

    # Step 5: filter by phase if requested
    if phase is not None:
        manifest = filter_enabled_entries(manifest, phase)

    return manifest


def iter_registry_entries(registry) -> List[dict]:
    """Extract unique entries from the registry.

    The registry maps many keys (aliases, short_name, full_name) to the same
    entry dict.  We dedupe by ``full_name``.
    """
    seen_full_names: set = set()
    unique: List[dict] = []

    for _key, entry in registry._registry.items():
        full_name = entry.get("full_name", "")
        if full_name and full_name not in seen_full_names:
            seen_full_names.add(full_name)
            unique.append(entry)

    return unique


def dedupe_registry_entries(entries: List[dict]) -> List[dict]:
    """Handle duplicate full_names (e.g. ``convert_to_pandas``).

    If two entries share the same ``full_name``, keep the one whose module
    matches ``DUPLICATE_RESOLUTION``, or warn and keep the first.
    """
    by_full_name: Dict[str, List[dict]] = {}
    for e in entries:
        fn = e["full_name"]
        by_full_name.setdefault(fn, []).append(e)

    result: List[dict] = []
    for fn, group in by_full_name.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Check resolution rules
            resolved = _resolve_duplicate(fn, group)
            result.append(resolved)
    return result


def classify_execution_class(entry: dict) -> str:
    """Classify a registry entry as ``stateless``, ``adata``, or ``class``.

    Heuristic:
    1. ``inspect.isclass(func)`` → ``"class"``
    2. First positional param named ``adata`` → ``"adata"``
    3. Otherwise → ``"stateless"``
    """
    func = entry.get("function")

    if func is not None:
        # Unwrap wrapper to get the original function/class
        original = getattr(func, "__wrapped__", func)

        if inspect.isclass(original):
            return "class"

        # Check first parameter name
        try:
            sig = inspect.signature(original)
            params = list(sig.parameters.keys())
            if params and params[0] in ("adata", "data"):
                return "adata"
        except (ValueError, TypeError):
            pass

    # Fallback: check the stored signature string
    sig_str = entry.get("signature", "")
    if sig_str.startswith("(adata") or sig_str.startswith("(data"):
        return "adata"

    return "stateless"


def build_manifest_entry(
    entry: dict,
    overrides,
    seen_names: Dict[str, dict],
) -> Optional[dict]:
    """Build a single manifest entry from a registry entry."""
    if entry.get("virtual_entry"):
        return None

    full_name = entry.get("full_name", "")
    if not full_name:
        return None

    # Classification
    execution_class = classify_execution_class(entry)
    kind = "class" if execution_class == "class" else "function"
    adapter_type = {
        "stateless": "function_adapter",
        "adata": "adata_adapter",
        "class": "class_adapter",
    }[execution_class]

    # For class tools: look up ClassWrapperSpec
    class_spec = None
    if execution_class == "class":
        from .class_specs import get_spec as _get_class_spec, build_class_tool_schema
        class_spec = _get_class_spec(full_name)

    # Tool name
    tool_name = naming.build_tool_name(full_name, entry.get("category"))
    if tool_name in seen_names:
        tool_name = naming.resolve_name_collision(tool_name, entry, seen_names)

    # Source ref
    source_ref = build_source_ref(entry.get("function"))

    # Metadata from registry
    meta = extract_registry_metadata(entry)

    # Schema
    schema_override = overrides.get_schema_override(full_name)
    if schema_override:
        parameter_schema = {
            "type": "object",
            **schema_override,
        }
    elif class_spec is not None:
        # Class tools use the action-based schema from the spec
        parameter_schema = build_class_tool_schema(class_spec)
    else:
        parameter_schema = build_parameter_schema(
            {**entry, "execution_class": execution_class},
            schema_override or None,
        )

    # State contract
    state_contract = _build_state_contract(entry, execution_class)

    # Dependency contract
    dependency_contract = {
        "prerequisites": entry.get("prerequisites", {}),
        "requires": entry.get("requires", {}),
        "produces": entry.get("produces", {}),
    }

    # Return contract
    return_contract = _build_return_contract(entry, execution_class)

    # Availability
    availability = avail_mod.build_availability(
        {"full_name": full_name, "category": meta["category"]},
        class_spec=class_spec,
    )

    # Rollout phase & status
    rollout_phase = overrides.get_rollout_phase(full_name)
    risk_level = _assess_risk(entry, execution_class, availability)
    status = "planned" if rollout_phase != "defer" else "hidden"

    # Build entry
    item = {
        "tool_name": tool_name,
        "full_name": full_name,
        "kind": kind,
        "execution_class": execution_class,
        "adapter_type": adapter_type,
        "source_ref": source_ref,
        "category": meta["category"],
        "aliases": meta["aliases"],
        "description": meta["description"],
        "signature": meta["signature"],
        "parameter_schema": parameter_schema,
        "state_contract": state_contract,
        "dependency_contract": dependency_contract,
        "return_contract": return_contract,
        "availability": availability,
        "risk_level": risk_level,
        "rollout_phase": rollout_phase,
        "status": status,
    }

    # Enrich class tools with action metadata
    if class_spec is not None:
        item["class_actions"] = _build_class_actions_summary(class_spec)

    # Apply manifest overrides (they win)
    manifest_override = overrides.get_manifest_override(full_name)
    if manifest_override:
        for key, value in manifest_override.items():
            item[key] = value

    # Store the function reference for runtime (not serialized)
    item["_function"] = entry.get("function")

    return item


def build_source_ref(func_or_cls) -> str:
    """Generate ``path:line`` source reference."""
    if func_or_cls is None:
        return ""
    try:
        original = getattr(func_or_cls, "__wrapped__", func_or_cls)
        filepath = inspect.getfile(original)
        # Make path relative to omicverse package
        parts = filepath.replace("\\", "/").split("/")
        try:
            idx = parts.index("omicverse")
            # Find the second 'omicverse' (the package, not the repo)
            remaining = parts[idx + 1:]
            if "omicverse" in remaining:
                idx2 = remaining.index("omicverse")
                rel = "/".join(remaining[idx2 + 1:])
            else:
                rel = "/".join(remaining)
        except (ValueError, IndexError):
            rel = os.path.basename(filepath)

        lines = inspect.getsourcelines(original)
        lineno = lines[1]
        return f"{rel}:{lineno}"
    except (TypeError, OSError):
        return ""


def extract_registry_metadata(entry: dict) -> dict:
    """Pull standard metadata fields from a registry entry."""
    return {
        "aliases": entry.get("aliases", []),
        "category": entry.get("category", ""),
        "description": entry.get("description", ""),
        "signature": entry.get("signature", ""),
        "prerequisites": entry.get("prerequisites", {}),
        "requires": entry.get("requires", {}),
        "produces": entry.get("produces", {}),
    }


def filter_enabled_entries(
    entries: List[dict],
    phase: str,
) -> List[dict]:
    """Filter manifest entries by rollout phase.

    If *phase* contains ``+`` (e.g. ``"P0+P0.5"``), include entries from
    all listed phases.
    """
    phases = {p.strip() for p in phase.split("+")}
    return [e for e in entries if e.get("rollout_phase") in phases]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_duplicate(full_name: str, group: List[dict]) -> dict:
    """Pick the canonical entry from a group of duplicates."""
    from .overrides import DUPLICATE_RESOLUTION

    preferred_module = DUPLICATE_RESOLUTION.get(full_name)
    if preferred_module:
        for entry in group:
            if entry.get("module", "").startswith(preferred_module):
                return entry

    warnings.warn(
        f"Duplicate registration for {full_name!r}: keeping first occurrence",
        stacklevel=3,
    )
    return group[0]


def _build_state_contract(entry: dict, execution_class: str) -> dict:
    """Derive state_contract from execution class and metadata."""
    if execution_class == "stateless":
        return {
            "needs_session": False,
            "session_inputs": [],
            "mutates_state": False,
            "returns_updated_adata_ref": False,
        }
    elif execution_class == "adata":
        produces = entry.get("produces", {})
        mutates = bool(produces)
        return {
            "needs_session": True,
            "session_inputs": ["adata_id"],
            "mutates_state": mutates,
            "returns_updated_adata_ref": True,
        }
    else:  # class
        return {
            "needs_session": True,
            "session_inputs": ["instance_id"],
            "mutates_state": True,
            "returns_updated_adata_ref": False,
        }


def _build_return_contract(entry: dict, execution_class: str) -> dict:
    """Derive return_contract from execution class and category."""
    category = entry.get("category", "")

    if execution_class == "stateless":
        return {
            "primary_output": "json",
            "secondary_outputs": [],
            "emits_artifacts": False,
            "artifact_types": [],
        }
    elif execution_class == "adata":
        # Plotting tools return images
        if category in ("pl", "visualization"):
            return {
                "primary_output": "image",
                "secondary_outputs": ["object_ref"],
                "emits_artifacts": True,
                "artifact_types": ["image/png"],
            }
        # Analysis tools that return data (e.g. get_markers)
        produces = entry.get("produces", {})
        if not produces:
            return {
                "primary_output": "json",
                "secondary_outputs": [],
                "emits_artifacts": False,
                "artifact_types": [],
            }
        return {
            "primary_output": "object_ref",
            "secondary_outputs": ["json"],
            "emits_artifacts": False,
            "artifact_types": [],
        }
    else:  # class
        return {
            "primary_output": "object_ref",
            "secondary_outputs": [],
            "emits_artifacts": False,
            "artifact_types": [],
        }


def _build_class_actions_summary(spec) -> list:
    """Build a serializable list of action summaries for a ClassWrapperSpec."""
    actions = []
    for name, action in sorted(spec.actions.items()):
        actions.append({
            "name": name,
            "description": action.description,
            "method": action.method,
            "needs_instance": action.needs_instance,
            "needs_adata": action.needs_adata,
            "returns": action.returns,
            "params": action.params,
            "required_params": action.required_params,
        })
    return actions


def _assess_risk(entry: dict, execution_class: str, availability: dict) -> str:
    """Assign a risk level based on dependencies and mutability."""
    if availability.get("requires_gpu") or availability.get("requires_network"):
        return "high"
    if execution_class == "class":
        return "medium"
    if availability.get("required_binaries"):
        return "high"
    return "low"
