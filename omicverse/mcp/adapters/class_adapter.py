"""
Adapter for class-backed tools (Phase 2).

Dispatches to registered ``ClassWrapperSpec`` actions.  Each class tool
exposes a fixed action set (create/run/results/destroy) — no generic
method dispatch is ever allowed.
"""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any, Dict, List, Optional

from .base import BaseAdapter
from ..session_store import SessionStore, SessionError
from ..class_specs import get_spec, ClassWrapperSpec, ActionSpec

logger = logging.getLogger(__name__)


class ClassAdapter(BaseAdapter):
    """Adapter for ``execution_class == "class"`` tools.

    Dispatches to the registered ``ClassWrapperSpec`` for the class.
    Returns ``tool_unavailable`` if no spec is registered or the spec
    is marked ``available=False``.
    """

    def can_handle(self, entry: dict) -> bool:
        return entry.get("execution_class") == "class"

    def invoke(self, entry: dict, args: dict, store: SessionStore) -> dict:
        full_name = entry.get("full_name", "")
        tool_name = entry.get("tool_name", "")

        # Look up spec
        spec = get_spec(full_name)
        if spec is None:
            return self._unavailable(tool_name, full_name, "No ClassWrapperSpec registered")

        if not spec.available:
            return self._unavailable(
                tool_name, full_name,
                f"Class tool is defined but not yet available (rollout_phase={spec.rollout_phase})",
            )

        # Runtime availability probe (catches missing packages)
        from ..availability import check_class_availability
        avail_ok, avail_reason = check_class_availability(spec)
        if not avail_ok:
            return self._unavailable(tool_name, full_name, avail_reason)

        # Validate action
        action_name = args.get("action")
        if not action_name or action_name not in spec.actions:
            return {
                "ok": False,
                "tool_name": tool_name,
                "error_code": "invalid_arguments",
                "message": f"Invalid or missing action. Must be one of: {sorted(spec.actions.keys())}",
                "details": {"available_actions": sorted(spec.actions.keys())},
                "suggested_next_tools": [],
            }

        action = spec.actions[action_name]

        # Validate per-action required params
        missing = [p for p in action.required_params if p not in args]
        if missing:
            return {
                "ok": False,
                "tool_name": tool_name,
                "error_code": "invalid_arguments",
                "message": f"Action '{action_name}' requires: {', '.join(missing)}",
                "details": {"missing": missing, "action": action_name},
                "suggested_next_tools": [],
            }

        # Dispatch
        try:
            if action_name == "create":
                return self._handle_create(entry, args, store, spec, action)
            elif action_name == "destroy":
                return self._handle_destroy(entry, args, store, spec)
            else:
                return self._handle_action(entry, args, store, spec, action)
        except Exception as exc:
            return self.normalize_exception(exc, entry)

    # -- Handlers ------------------------------------------------------------

    def _handle_create(
        self,
        entry: dict,
        args: dict,
        store: SessionStore,
        spec: ClassWrapperSpec,
        action: ActionSpec,
    ) -> dict:
        """Import the class, instantiate with params, store and return instance_id."""
        tool_name = entry.get("tool_name", "")

        # Import the class
        cls = self._import_class(spec.full_name)

        # Build constructor kwargs
        kwargs = self._extract_action_kwargs(action, args)

        # If action needs adata, inject it
        if action.needs_adata:
            adata_id = args.get("adata_id")
            if not adata_id:
                return {
                    "ok": False,
                    "tool_name": tool_name,
                    "error_code": "missing_session_object",
                    "message": "adata_id is required for create action",
                    "details": {},
                    "suggested_next_tools": ["ov.utils.read"],
                }
            try:
                adata = store.get_adata(adata_id)
            except KeyError as exc:
                error_code = getattr(exc, "error_code", "missing_session_object")
                return {
                    "ok": False,
                    "tool_name": tool_name,
                    "error_code": error_code,
                    "message": str(exc) if error_code != "missing_session_object"
                               else f"Dataset not found: {adata_id}",
                    "details": getattr(exc, "details", {"adata_id": adata_id}),
                    "suggested_next_tools": ["ov.utils.read"],
                }
            # Inject adata (or derivative) as first positional arg
            kwargs = self._inject_adata_for_create(cls, spec, kwargs, adata)

        # Custom handler takes priority
        if action.handler is not None:
            instance = action.handler(cls, kwargs, store, args)
        else:
            instance = cls(**kwargs)

        # Store instance
        try:
            instance_id = store.create_instance(
                obj=instance,
                class_name=spec.full_name,
                metadata={
                    "tool_name": tool_name,
                    "create_args": {k: _safe_json(v) for k, v in args.items()
                                    if k not in ("action", "adata_id")},
                },
            )
        except SessionError as exc:
            return {
                "ok": False,
                "tool_name": tool_name,
                "error_code": exc.error_code,
                "message": str(exc),
                "details": exc.details,
                "suggested_next_tools": ["ov.get_limits", "ov.cleanup_runtime"],
            }

        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": f"Created {spec.tool_name} instance",
            "outputs": [
                {"type": "object_ref", "ref_type": "instance", "ref_id": instance_id},
            ],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_destroy(
        self,
        entry: dict,
        args: dict,
        store: SessionStore,
        spec: ClassWrapperSpec,
    ) -> dict:
        """Delete an instance from the store."""
        tool_name = entry.get("tool_name", "")
        instance_id = args.get("instance_id")
        if not instance_id:
            return {
                "ok": False,
                "tool_name": tool_name,
                "error_code": "invalid_arguments",
                "message": "instance_id is required for destroy action",
                "details": {},
                "suggested_next_tools": [],
            }

        try:
            store.delete_handle(instance_id)
        except KeyError:
            return {
                "ok": False,
                "tool_name": tool_name,
                "error_code": "missing_session_object",
                "message": f"Instance not found: {instance_id}",
                "details": {"instance_id": instance_id},
                "suggested_next_tools": [],
            }

        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": f"Destroyed {spec.tool_name} instance {instance_id}",
            "outputs": [],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_action(
        self,
        entry: dict,
        args: dict,
        store: SessionStore,
        spec: ClassWrapperSpec,
        action: ActionSpec,
    ) -> dict:
        """Call a method on an existing instance."""
        tool_name = entry.get("tool_name", "")

        # Get instance
        instance_id = args.get("instance_id")
        if not instance_id:
            return {
                "ok": False,
                "tool_name": tool_name,
                "error_code": "invalid_arguments",
                "message": f"instance_id is required for action '{action.name}'",
                "details": {},
                "suggested_next_tools": [],
            }

        try:
            instance = store.get_instance(instance_id)
        except KeyError as exc:
            error_code = getattr(exc, "error_code", "missing_session_object")
            return {
                "ok": False,
                "tool_name": tool_name,
                "error_code": error_code,
                "message": str(exc) if error_code != "missing_session_object"
                           else f"Instance not found: {instance_id}",
                "details": getattr(exc, "details", {"instance_id": instance_id}),
                "suggested_next_tools": [],
            }

        # Build kwargs for the method
        kwargs = self._extract_action_kwargs(action, args)

        # If action needs adata, inject it
        if action.needs_adata:
            adata_id = args.get("adata_id")
            if not adata_id:
                return {
                    "ok": False,
                    "tool_name": tool_name,
                    "error_code": "missing_session_object",
                    "message": f"adata_id is required for action '{action.name}'",
                    "details": {},
                    "suggested_next_tools": ["ov.utils.read"],
                }
            try:
                adata = store.get_adata(adata_id)
            except KeyError as exc:
                error_code = getattr(exc, "error_code", "missing_session_object")
                return {
                    "ok": False,
                    "tool_name": tool_name,
                    "error_code": error_code,
                    "message": str(exc) if error_code != "missing_session_object"
                               else f"Dataset not found: {adata_id}",
                    "details": getattr(exc, "details", {"adata_id": adata_id}),
                    "suggested_next_tools": ["ov.utils.read"],
                }
            kwargs["adata"] = adata

        # Custom handler takes priority
        if action.handler is not None:
            result = action.handler(instance, kwargs, store, args)
        else:
            method = getattr(instance, action.method, None)
            if method is None:
                return {
                    "ok": False,
                    "tool_name": tool_name,
                    "error_code": "execution_failed",
                    "message": f"Method '{action.method}' not found on instance",
                    "details": {},
                    "suggested_next_tools": [],
                }
            result = method(**kwargs)

        # Normalize result based on action.returns
        return self._normalize_action_result(
            result, entry, tool_name, action, instance_id, store, args,
        )

    # -- Import helper -------------------------------------------------------

    def _import_class(self, full_name: str):
        """Import and return the class object from its full_name.

        E.g. ``"omicverse.bulk._Deseq2.pyDEG"`` → import pyDEG from
        ``omicverse.bulk._Deseq2``.
        """
        module_path, class_name = full_name.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import module '{module_path}' for class tool "
                f"'{class_name}': {exc}"
            ) from exc
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise ImportError(
                f"Module '{module_path}' has no attribute '{class_name}'"
            )
        return cls

    # -- Parameter helpers ---------------------------------------------------

    def _extract_action_kwargs(self, action: ActionSpec, args: dict) -> dict:
        """Extract only the params defined for this action from args."""
        kwargs = {}
        for param_name in action.params:
            if param_name in args:
                kwargs[param_name] = args[param_name]
            elif "default" in action.params[param_name]:
                kwargs[param_name] = action.params[param_name]["default"]
        return kwargs

    def _inject_adata_for_create(
        self,
        cls,
        spec: ClassWrapperSpec,
        kwargs: dict,
        adata,
    ) -> dict:
        """Inject adata (or a derivative like adata.to_df()) for create.

        pyDEG takes a DataFrame (raw_data), not an AnnData.
        pySCSA and MetaCell take AnnData directly.
        """
        # pyDEG: extract DataFrame from adata
        if "pyDEG" in spec.full_name:
            layer = kwargs.pop("layer", "X")
            if layer == "X" or layer is None:
                if hasattr(adata, "to_df"):
                    kwargs["raw_data"] = adata.to_df()
                else:
                    kwargs["raw_data"] = adata.X
            else:
                if hasattr(adata, "to_df"):
                    kwargs["raw_data"] = adata.to_df(layer=layer)
                else:
                    kwargs["raw_data"] = adata.layers.get(layer, adata.X)
        else:
            # Most classes take adata as first positional arg
            kwargs["adata"] = adata

        return kwargs

    # -- Result normalization ------------------------------------------------

    def _normalize_action_result(
        self,
        result,
        entry: dict,
        tool_name: str,
        action: ActionSpec,
        instance_id: str,
        store: SessionStore,
        args: dict,
    ) -> dict:
        """Normalize method result based on ``action.returns``."""
        outputs: List[dict] = []
        state_updates: dict = {}

        if action.returns == "json":
            data = _to_json_safe(result)
            outputs.append({"type": "json", "data": data})

        elif action.returns == "object_ref":
            # Result is a new AnnData — store it
            if _is_anndata(result):
                new_adata_id = store.create_adata(result)
                outputs.append({
                    "type": "object_ref",
                    "ref_type": "adata",
                    "ref_id": new_adata_id,
                })
            else:
                data = _to_json_safe(result)
                outputs.append({"type": "json", "data": data})

        elif action.returns == "instance_id":
            outputs.append({
                "type": "object_ref",
                "ref_type": "instance",
                "ref_id": instance_id,
            })

        elif action.returns == "void":
            pass  # no outputs

        else:
            data = _to_json_safe(result)
            outputs.append({"type": "json", "data": data})

        summary = f"{tool_name} {action.name} completed"

        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": summary,
            "outputs": outputs,
            "state_updates": state_updates,
            "warnings": [],
        }

    # -- Error helpers -------------------------------------------------------

    def _unavailable(self, tool_name: str, full_name: str, reason: str) -> dict:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error_code": "tool_unavailable",
            "message": f"{tool_name} is a class-backed tool that is not available: {reason}",
            "details": {"full_name": full_name, "kind": "class", "reason": reason},
            "suggested_next_tools": [],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_anndata(obj) -> bool:
    return type(obj).__name__ in ("AnnData", "MuData")


def _is_dataframe(obj) -> bool:
    return type(obj).__name__ in ("DataFrame",)


def _to_json_safe(obj) -> Any:
    """Convert result to JSON-safe representation."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if _is_dataframe(obj):
        try:
            truncated = len(obj) > 200
            subset = obj.head(200)
            return {
                "columns": list(subset.columns),
                "data": subset.values.tolist(),
                "shape": list(obj.shape),
                "truncated": truncated,
            }
        except Exception:
            return str(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _safe_json(v) -> Any:
    """Make a value safe for JSON metadata storage."""
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)
