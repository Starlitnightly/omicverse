"""
Unified execution coordinator for MCP tool calls.

Handles the full lifecycle: resolve → validate → check prerequisites →
select adapter → invoke → return structured response.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .session_store import SessionStore, SessionError
from .availability import check_tool_availability
from .adapters.base import BaseAdapter
from .adapters.function_adapter import FunctionAdapter
from .adapters.adata_adapter import AdataAdapter
from .adapters.class_adapter import ClassAdapter


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

TOOL_NOT_FOUND = "tool_not_found"
TOOL_UNAVAILABLE = "tool_unavailable"
INVALID_ARGUMENTS = "invalid_arguments"
MISSING_SESSION_OBJECT = "missing_session_object"
MISSING_PREREQUISITES = "missing_prerequisites"
MISSING_DATA_REQUIREMENTS = "missing_data_requirements"
EXECUTION_FAILED = "execution_failed"
CROSS_SESSION_ACCESS = "cross_session_access"
PERSISTENCE_FAILED = "persistence_failed"
UNSUPPORTED_PERSISTENCE = "unsupported_persistence"
QUOTA_EXCEEDED = "quota_exceeded"
SESSION_EXPIRED = "session_expired"
HANDLE_EXPIRED = "handle_expired"


# ---------------------------------------------------------------------------
# McpExecutor
# ---------------------------------------------------------------------------


class McpExecutor:
    """Orchestrates MCP tool execution with validation and error handling."""

    def __init__(
        self,
        manifest: List[dict],
        store: SessionStore,
        adapters: Optional[List[BaseAdapter]] = None,
    ):
        self._manifest = manifest
        self._store = store
        self._adapters = adapters or [
            FunctionAdapter(),
            AdataAdapter(),
            ClassAdapter(),
        ]

        # Build lookup index by tool_name
        self._by_name: Dict[str, dict] = {}
        for entry in manifest:
            self._by_name[entry["tool_name"]] = entry

    @property
    def store(self) -> SessionStore:
        return self._store

    @property
    def manifest(self) -> List[dict]:
        return list(self._manifest)

    # -- Main entry point ----------------------------------------------------

    def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute a tool by canonical name.

        Returns a response envelope (always a dict with ``ok`` key).
        Records ``tool_called`` / ``tool_failed`` events for observability.
        """
        result = self._execute_tool_core(tool_name, args)
        try:
            event_type = "tool_called" if result.get("ok") else "tool_failed"
            self._store.record_event(event_type, {
                "tool_name": tool_name,
                "ok": result.get("ok"),
                "error_code": result.get("error_code"),
            })
        except Exception:
            pass
        return result

    def _execute_tool_core(self, tool_name: str, args: dict) -> dict:
        """Core execution logic (called by ``execute_tool``)."""
        # 1. Resolve
        entry = self.resolve_entry(tool_name)
        if entry is None:
            return self.build_error_response(
                TOOL_NOT_FOUND,
                f"Unknown tool: {tool_name}",
                {"tool_name": tool_name},
                [],
            )

        # 2. Availability
        avail = check_tool_availability(entry)
        if not avail.get("available", True):
            return self.build_error_response(
                TOOL_UNAVAILABLE,
                f"Tool {tool_name} is not available: {avail.get('reason', '')}",
                {"availability": avail},
                [],
            )

        # 3. Validate arguments
        validation_error = self.validate_call_args(entry, args)
        if validation_error:
            return validation_error

        # 4. For adata tools: check session object exists
        execution_class = entry.get("execution_class", "stateless")
        if execution_class in ("adata",) and "adata_id" in args:
            adata_id = args["adata_id"]
            try:
                self._store.get_adata(adata_id)
            except SessionError as exc:
                return self.build_error_response(
                    exc.error_code, str(exc), exc.details, [],
                )
            except KeyError:
                return self.build_error_response(
                    MISSING_SESSION_OBJECT,
                    f"Dataset not found: {adata_id}",
                    {"adata_id": adata_id},
                    ["ov.utils.read"],
                )

        # 4b. For class tools: check instance_id exists (except for create)
        if execution_class == "class" and args.get("action") != "create":
            instance_id = args.get("instance_id")
            if instance_id:
                try:
                    self._store.get_instance(instance_id)
                except SessionError as exc:
                    return self.build_error_response(
                        exc.error_code, str(exc), exc.details, [],
                    )
                except KeyError:
                    return self.build_error_response(
                        MISSING_SESSION_OBJECT,
                        f"Instance not found: {instance_id}",
                        {"instance_id": instance_id},
                        [],
                    )

        # 5. Check prerequisites (data state)
        prereq_error = self.check_prerequisites(entry, args)
        if prereq_error:
            return prereq_error

        # 6. Check data requirements
        data_error = self.check_data_requirements(entry, args)
        if data_error:
            return data_error

        # 7. Select adapter and invoke
        adapter = self.select_adapter(entry)
        if adapter is None:
            return self.build_error_response(
                EXECUTION_FAILED,
                f"No adapter found for {tool_name} (execution_class={execution_class})",
                {},
                [],
            )

        return adapter.invoke(entry, args, self._store)

    # -- Resolution ----------------------------------------------------------

    def resolve_entry(self, tool_name: str) -> Optional[dict]:
        """Look up a manifest entry by canonical tool name."""
        return self._by_name.get(tool_name)

    # -- Validation ----------------------------------------------------------

    def validate_call_args(self, entry: dict, args: dict) -> Optional[dict]:
        """Validate arguments against the parameter schema.

        Returns ``None`` if valid, or an error response dict.
        """
        schema = entry.get("parameter_schema", {})
        required = schema.get("required", [])

        # Check required params
        missing = [r for r in required if r not in args]
        if missing:
            return self.build_error_response(
                INVALID_ARGUMENTS,
                f"Missing required arguments: {', '.join(missing)}",
                {"missing": missing, "schema": schema},
                [],
            )

        # Type checks for known properties (lightweight, no jsonschema dep)
        properties = schema.get("properties", {})
        for key, value in args.items():
            if key not in properties:
                if not schema.get("additionalProperties", False):
                    continue  # silently skip unknown in strict mode
                continue

            prop_schema = properties[key]
            type_error = _check_type(key, value, prop_schema)
            if type_error:
                return self.build_error_response(
                    INVALID_ARGUMENTS,
                    type_error,
                    {"parameter": key, "expected": prop_schema},
                    [],
                )

        return None

    # -- Prerequisite checking -----------------------------------------------

    def check_prerequisites(self, entry: dict, args: dict) -> Optional[dict]:
        """Check function prerequisites against actual AnnData state.

        Only checks ``requires`` — validates that required data structures
        exist on the AnnData object.  Does NOT track call history.
        """
        dep_contract = entry.get("dependency_contract", {})
        requires = dep_contract.get("requires", {})
        if not requires:
            return None

        adata_id = args.get("adata_id")
        if not adata_id:
            return None

        try:
            adata = self._store.get_adata(adata_id)
        except KeyError:
            return None  # Will be caught by session check

        return self._check_requires(entry, adata, requires)

    def check_data_requirements(self, entry: dict, args: dict) -> Optional[dict]:
        """Check data structure requirements.

        This is a separate check from prerequisites for clarity, but in
        practice they overlap.  This method checks ``requires`` keys on
        the actual AnnData.
        """
        # Currently merged with check_prerequisites
        return None

    # -- Adapter selection ---------------------------------------------------

    def select_adapter(self, entry: dict) -> Optional[BaseAdapter]:
        """Pick the adapter that handles this entry's execution class."""
        for adapter in self._adapters:
            if adapter.can_handle(entry):
                return adapter
        return None

    # -- Response builders ---------------------------------------------------

    def build_success_response(
        self,
        tool_name: str,
        summary: str,
        outputs: List[dict],
        state_updates: Optional[dict] = None,
        warnings_list: Optional[List[str]] = None,
    ) -> dict:
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": summary,
            "outputs": outputs,
            "state_updates": state_updates or {},
            "warnings": warnings_list or [],
        }

    def build_error_response(
        self,
        error_code: str,
        message: str,
        details: Optional[dict] = None,
        suggested_next_tools: Optional[List[str]] = None,
    ) -> dict:
        return {
            "ok": False,
            "error_code": error_code,
            "message": message,
            "details": details or {},
            "suggested_next_tools": suggested_next_tools or [],
        }

    def suggest_next_tools(self, entry: dict, error_code: str) -> List[str]:
        """Generate suggested next tools based on error context."""
        suggestions: List[str] = []
        dep_contract = entry.get("dependency_contract", {})

        if error_code == MISSING_SESSION_OBJECT:
            suggestions.append("ov.utils.read")

        elif error_code in (MISSING_PREREQUISITES, MISSING_DATA_REQUIREMENTS):
            prereqs = dep_contract.get("prerequisites", {})
            required_funcs = prereqs.get("functions", [])
            for fn in required_funcs:
                # Try to find the tool name for this function
                for e in self._manifest:
                    short = e.get("full_name", "").rsplit(".", 1)[-1]
                    if short == fn:
                        suggestions.append(e["tool_name"])
                        break

        return suggestions

    # -- Internal helpers ----------------------------------------------------

    def _check_requires(
        self, entry: dict, adata, requires: dict
    ) -> Optional[dict]:
        """Validate that required AnnData keys exist."""
        missing: Dict[str, List[str]] = {}

        for slot, keys in requires.items():
            obj = getattr(adata, slot, None)
            if obj is None:
                missing[slot] = list(keys)
                continue
            try:
                existing = set(obj.keys())
            except (AttributeError, TypeError):
                if hasattr(obj, "columns"):
                    existing = set(obj.columns)
                else:
                    missing[slot] = list(keys)
                    continue
            slot_missing = [k for k in keys if k not in existing]
            if slot_missing:
                missing[slot] = slot_missing

        if not missing:
            return None

        tool_name = entry.get("tool_name", "")
        suggestions = self.suggest_next_tools(entry, MISSING_DATA_REQUIREMENTS)

        return self.build_error_response(
            MISSING_DATA_REQUIREMENTS,
            f"Tool {tool_name} requires missing data: {missing}",
            {
                "missing_structures": missing,
                "required": requires,
            },
            suggestions,
        )


# ---------------------------------------------------------------------------
# Lightweight type checking (no jsonschema dependency)
# ---------------------------------------------------------------------------


def _check_type(key: str, value: Any, prop_schema: dict) -> Optional[str]:
    """Return an error message if *value* doesn't match *prop_schema*, else None."""
    expected_type = prop_schema.get("type")
    if expected_type is None:
        return None

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": (list, tuple),
        "object": dict,
    }

    expected = type_map.get(expected_type)
    if expected is None:
        return None

    if not isinstance(value, expected):
        # Special case: int is also valid for number
        if expected_type == "number" and isinstance(value, (int, float)):
            return None
        return (
            f"Parameter '{key}' expected type {expected_type}, "
            f"got {type(value).__name__}"
        )

    # Check enum
    enum_values = prop_schema.get("enum")
    if enum_values and value not in enum_values:
        return f"Parameter '{key}' must be one of {enum_values}, got {value!r}"

    return None
