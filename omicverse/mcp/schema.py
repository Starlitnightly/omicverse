"""
JSON Schema generation from Python function signatures.

Converts registry function signatures and type annotations into MCP-compatible
JSON Schema ``inputSchema`` objects.  Handles AnnData/MuData replacement with
``adata_id``, kwargs expansion, and override merging.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Dict, Optional, get_type_hints

# ---------------------------------------------------------------------------
# Type annotation → JSON Schema mapping
# ---------------------------------------------------------------------------

_PRIMITIVE_MAP: Dict[type, dict] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
    type(None): {"type": "null"},
}

# Types that should be replaced with adata_id or excluded entirely
_SESSION_TYPES = {"AnnData", "MuData"}
_EXCLUDED_TYPES = {"Axes", "Figure", "Colormap", "Normalize", "Cycler", "AxesSubplot"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_parameter_schema(entry: dict, overrides: Optional[dict] = None) -> dict:
    """Build an MCP-compatible JSON Schema for the tool's input parameters.

    Parameters
    ----------
    entry : dict
        Registry entry (must contain ``'function'`` key).
    overrides : dict, optional
        Schema override from ``overrides.py``.

    Returns
    -------
    dict
        JSON Schema object with ``type``, ``properties``, ``required``.
    """
    func = entry.get("function")
    if func is None:
        return build_empty_schema()

    schema = signature_to_schema(func)

    # Inject session params (adata_id) for adata/class tools
    execution_class = entry.get("execution_class", "stateless")
    if execution_class in ("adata", "class"):
        schema = inject_session_params(schema, entry)

    # Apply overrides last (they win)
    if overrides:
        schema = apply_schema_overrides(schema, overrides)

    return schema


def signature_to_schema(func) -> dict:
    """Generate a JSON Schema from a Python function's signature."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return build_empty_schema()

    # Try to get type hints, fall back gracefully
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: Dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        # Skip *args, **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Skip self/cls
        if name in ("self", "cls"):
            continue

        annotation = hints.get(name, param.annotation)
        type_name = _get_type_name(annotation)

        # Skip session types (will be injected separately)
        if type_name in _SESSION_TYPES:
            continue

        # Skip matplotlib / internal types
        if type_name in _EXCLUDED_TYPES:
            continue

        prop = normalize_parameter(param, annotation)
        if prop is not None:
            properties[name] = prop

            # Required if no default and not Optional
            if param.default is inspect.Parameter.empty and not _is_optional(annotation):
                required.append(name)

    schema: dict = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    # If function accepts **kwargs, allow additional properties
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_kwargs:
        schema["additionalProperties"] = True

    return schema


def annotation_to_json_schema(annotation) -> dict:
    """Map a single Python type annotation to a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty:
        return {}

    # Handle None / NoneType
    if annotation is type(None):
        return {"type": "null"}

    # Direct primitive match
    if annotation in _PRIMITIVE_MAP:
        return dict(_PRIMITIVE_MAP[annotation])

    # String annotations (unresolved forward refs)
    if isinstance(annotation, str):
        return _string_annotation_to_schema(annotation)

    # Get origin for generic types (Optional, Union, List, etc.)
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    # Optional[X] → nullable X
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X]
            inner = annotation_to_json_schema(non_none[0])
            return inner  # MCP handles nullability via required/not-required
        # Union of multiple types
        schemas = [annotation_to_json_schema(a) for a in non_none if a is not type(None)]
        schemas = [s for s in schemas if s]
        if len(schemas) == 1:
            return schemas[0]
        if schemas:
            return {"anyOf": schemas}
        return {}

    # List[X]
    if origin is list:
        if args:
            items = annotation_to_json_schema(args[0])
            return {"type": "array", "items": items} if items else {"type": "array"}
        return {"type": "array"}

    # Dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Tuple
    if origin is tuple:
        return {"type": "array"}

    # Literal['a', 'b']
    if origin is typing.Literal:
        return {"type": "string", "enum": list(args)}

    # Sequence → array
    seq_name = getattr(origin, "__name__", "") if origin else ""
    if seq_name == "Sequence" or (hasattr(annotation, "__name__") and annotation.__name__ == "Sequence"):
        return {"type": "array"}

    # Mapping → object
    if seq_name == "Mapping" or (hasattr(annotation, "__name__") and annotation.__name__ == "Mapping"):
        return {"type": "object"}

    # Fallback: check class name
    name = _get_type_name(annotation)
    if name in _SESSION_TYPES:
        return {"type": "string", "description": "Session reference ID"}
    if name in _EXCLUDED_TYPES:
        return {}

    # Unknown → string fallback
    if name and name[0].isupper():
        return {"type": "string"}

    return {}


def normalize_parameter(param: inspect.Parameter, annotation=inspect.Parameter.empty) -> Optional[dict]:
    """Convert a single ``inspect.Parameter`` into a JSON Schema property."""
    if annotation is inspect.Parameter.empty:
        annotation = param.annotation

    prop = annotation_to_json_schema(annotation)

    # If we got nothing from annotation, infer from default
    if not prop:
        if param.default is not inspect.Parameter.empty and param.default is not None:
            prop = _infer_from_default(param.default)
        if not prop:
            prop = {}

    # Add default value
    if param.default is not inspect.Parameter.empty and param.default is not None:
        try:
            # Only add JSON-serializable defaults
            import json
            json.dumps(param.default)
            prop["default"] = param.default
        except (TypeError, ValueError):
            pass

    return prop if prop else None


def inject_session_params(schema: dict, entry: dict) -> dict:
    """Replace the ``adata`` parameter with ``adata_id`` for session-aware tools."""
    props = dict(schema.get("properties", {}))
    required = list(schema.get("required", []))

    # Remove adata/data if present
    for key in ("adata", "data"):
        props.pop(key, None)
        if key in required:
            required.remove(key)

    # Add adata_id as first required param
    props["adata_id"] = {
        "type": "string",
        "description": "Session dataset reference",
    }
    if "adata_id" not in required:
        required.insert(0, "adata_id")

    return {
        **schema,
        "properties": props,
        "required": required,
    }


def apply_schema_overrides(schema: dict, override: dict) -> dict:
    """Deep-merge *override* into *schema*.  Override wins on conflict."""
    result = dict(schema)

    if "properties" in override:
        merged_props = dict(result.get("properties", {}))
        merged_props.update(override["properties"])
        result["properties"] = merged_props

    if "required" in override:
        result["required"] = override["required"]

    if "additionalProperties" in override:
        result["additionalProperties"] = override["additionalProperties"]

    return result


def build_empty_schema() -> dict:
    """Fallback schema for functions with no usable signature."""
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_type_name(annotation) -> str:
    """Extract a readable type name from an annotation."""
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    name = getattr(annotation, "__name__", "")
    if name:
        return name
    # For generic aliases like Optional[str]
    name = getattr(annotation, "_name", "")
    if name:
        return name
    return str(annotation)


def _is_optional(annotation) -> bool:
    """Check if annotation is Optional[X] (Union[X, None])."""
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        args = getattr(annotation, "__args__", ())
        return type(None) in args
    return False


def _string_annotation_to_schema(name: str) -> dict:
    """Handle string-form annotations."""
    lower = name.lower()
    if lower in ("str", "string"):
        return {"type": "string"}
    if lower in ("int", "integer"):
        return {"type": "integer"}
    if lower in ("float", "number"):
        return {"type": "number"}
    if lower in ("bool", "boolean"):
        return {"type": "boolean"}
    if "anndata" in lower or "mudata" in lower:
        return {"type": "string", "description": "Session reference ID"}
    return {}


def _infer_from_default(default) -> dict:
    """Infer JSON Schema type from a default value."""
    if isinstance(default, bool):
        return {"type": "boolean"}
    if isinstance(default, int):
        return {"type": "integer"}
    if isinstance(default, float):
        return {"type": "number"}
    if isinstance(default, str):
        return {"type": "string"}
    if isinstance(default, list):
        return {"type": "array"}
    if isinstance(default, dict):
        return {"type": "object"}
    return {}
