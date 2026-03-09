"""
Adapter for AnnData-stateful functions.

Handles tools that require ``adata_id`` and typically mutate the dataset
in place.  Records state changes (produced keys) after execution.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any, Dict, List, Optional, Set

from .base import BaseAdapter
from ..session_store import SessionStore, SessionError


class AdataAdapter(BaseAdapter):
    """Adapter for ``execution_class == "adata"`` tools."""

    def can_handle(self, entry: dict) -> bool:
        return entry.get("execution_class") == "adata"

    def invoke(self, entry: dict, args: dict, store: SessionStore) -> dict:
        ctx = self.build_call_context(entry, args, store)
        func = entry.get("_function")

        if func is None:
            return {
                "ok": False,
                "tool_name": entry.get("tool_name", ""),
                "error_code": "execution_failed",
                "message": "No callable function found in manifest entry",
                "details": {},
                "suggested_next_tools": [],
            }

        # Load AnnData from session
        adata_id = args.get("adata_id")
        if not adata_id:
            return {
                "ok": False,
                "tool_name": entry.get("tool_name", ""),
                "error_code": "missing_session_object",
                "message": "adata_id is required",
                "details": {},
                "suggested_next_tools": ["ov.utils.read"],
            }

        try:
            adata = store.get_adata(adata_id)
        except KeyError as exc:
            error_code = getattr(exc, "error_code", "missing_session_object")
            return {
                "ok": False,
                "tool_name": entry.get("tool_name", ""),
                "error_code": error_code,
                "message": str(exc) if error_code != "missing_session_object"
                           else f"Dataset not found: {adata_id}",
                "details": getattr(exc, "details", {"adata_id": adata_id}),
                "suggested_next_tools": ["ov.utils.read"],
            }

        # Snapshot pre-execution state
        pre_state = self.snapshot_pre_state(adata, entry)

        # Build kwargs
        kwargs = self.prepare_adata_kwargs(entry, args, adata)

        # Execute
        try:
            result = func(**kwargs)
        except Exception as exc:
            return self.normalize_exception(exc, entry)

        # If function returned a new adata (e.g. qc filters cells), update store
        if _is_anndata(result) and result is not adata:
            store.update_adata(adata_id, result)
            adata = result

        # Detect state changes
        state_updates = self.detect_state_updates(adata, pre_state, entry)

        # Build response
        return_contract = entry.get("return_contract", {})
        primary_output = return_contract.get("primary_output", "object_ref")

        if primary_output == "image":
            return self._build_image_response(result, entry, ctx, adata_id, state_updates, store)
        elif primary_output == "table":
            return self._build_table_response(result, entry, ctx, adata_id, state_updates)
        else:
            return self._build_ref_response(entry, ctx, adata_id, state_updates, adata)

    def prepare_adata_kwargs(self, entry: dict, args: dict, adata) -> dict:
        """Inject adata as first arg, pass remaining MCP args as kwargs."""
        kwargs = dict(args)
        kwargs.pop("adata_id", None)
        kwargs.pop("instance_id", None)

        # The function expects adata as first positional arg
        kwargs["adata"] = adata
        return kwargs

    def snapshot_pre_state(self, adata, entry: dict) -> dict:
        """Record the keys present in AnnData slots before execution."""
        state: Dict[str, Set[str]] = {}
        for slot in ("obsm", "obsp", "obs", "var", "layers", "uns", "varm"):
            obj = getattr(adata, slot, None)
            if obj is not None:
                try:
                    state[slot] = set(obj.keys())
                except (AttributeError, TypeError):
                    if hasattr(obj, "columns"):
                        state[slot] = set(obj.columns)
                    else:
                        state[slot] = set()
            else:
                state[slot] = set()
        state["shape"] = (adata.shape[0], adata.shape[1]) if hasattr(adata, "shape") else (0, 0)
        return state

    def detect_state_updates(
        self, adata, pre_state: dict, entry: dict
    ) -> dict:
        """Compare post-execution state to pre-state and report changes."""
        produced: Dict[str, List[str]] = {}
        for slot in ("obsm", "obsp", "obs", "var", "layers", "uns", "varm"):
            obj = getattr(adata, slot, None)
            if obj is None:
                continue
            try:
                current_keys = set(obj.keys())
            except (AttributeError, TypeError):
                if hasattr(obj, "columns"):
                    current_keys = set(obj.columns)
                else:
                    continue
            pre_keys = pre_state.get(slot, set())
            new_keys = current_keys - pre_keys
            if new_keys:
                produced[slot] = sorted(new_keys)

        # Shape change
        shape_change = None
        if hasattr(adata, "shape"):
            pre_shape = pre_state.get("shape", (0, 0))
            if (adata.shape[0], adata.shape[1]) != pre_shape:
                shape_change = {
                    "before": list(pre_shape),
                    "after": list(adata.shape),
                }

        result = {"produced": produced}
        if shape_change:
            result["shape_change"] = shape_change
        return result

    # -- Response builders ---------------------------------------------------

    def _build_ref_response(
        self, entry, ctx, adata_id, state_updates, adata
    ) -> dict:
        """Standard response for mutating tools — return updated adata ref."""
        outputs = [{
            "type": "object_ref",
            "ref_type": "adata",
            "ref_id": adata_id,
        }]

        # Add summary info
        summary_parts = [entry.get("description", entry.get("tool_name", ""))]
        produced = state_updates.get("produced", {})
        if produced:
            keys = []
            for slot, names in produced.items():
                keys.extend(f"{slot}[{n}]" for n in names)
            summary_parts.append(f"Produced: {', '.join(keys)}")
        if hasattr(adata, "shape"):
            summary_parts.append(f"Shape: {adata.shape[0]} x {adata.shape[1]}")

        return self.normalize_result(
            None, entry, ctx,
            outputs=outputs,
            state_updates=state_updates,
        ) | {"summary": " | ".join(summary_parts)}

    def _build_image_response(
        self, result, entry, ctx, adata_id, state_updates, store
    ) -> dict:
        """Response for plotting tools — save figure and return artifact."""
        outputs = []

        try:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            if fig.get_axes():
                path = tempfile.mktemp(suffix=".png", prefix="ov_plot_")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                try:
                    artifact_id = store.create_artifact(
                        path, "image/png",
                        artifact_type="image",
                        source_tool=entry.get("tool_name", ""),
                    )
                except SessionError:
                    # Quota exceeded for artifacts — skip artifact registration
                    # but don't fail the tool call itself
                    plt.close(fig)
                    outputs.append({
                        "type": "object_ref", "ref_type": "adata", "ref_id": adata_id,
                    })
                    return self.normalize_result(
                        result, entry, ctx, outputs=outputs,
                        state_updates=state_updates,
                        warnings_list=["Artifact quota exceeded; plot saved but not registered"],
                    ) | {"summary": "Plot generated (artifact quota exceeded)"}
                outputs.append({
                    "type": "image",
                    "artifact_id": artifact_id,
                    "path": path,
                    "content_type": "image/png",
                })
            else:
                plt.close(fig)
        except ImportError:
            pass

        # Also include adata ref if it was modified
        outputs.append({
            "type": "object_ref",
            "ref_type": "adata",
            "ref_id": adata_id,
        })

        return self.normalize_result(
            result, entry, ctx,
            outputs=outputs,
            state_updates=state_updates,
        )

    def _build_table_response(
        self, result, entry, ctx, adata_id, state_updates
    ) -> dict:
        """Response for tools that return DataFrames (e.g. get_markers)."""
        outputs = []

        if _is_dataframe(result):
            outputs.append({
                "type": "table",
                "data": _dataframe_to_json(result),
            })
        elif isinstance(result, dict):
            outputs.append({"type": "json", "data": result})
        elif result is not None:
            try:
                json.dumps(result)
                outputs.append({"type": "json", "data": result})
            except (TypeError, ValueError):
                outputs.append({"type": "text", "data": str(result)})

        return self.normalize_result(
            result, entry, ctx,
            outputs=outputs,
            state_updates=state_updates,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_anndata(obj) -> bool:
    type_name = type(obj).__name__
    return type_name in ("AnnData", "MuData")


def _is_dataframe(obj) -> bool:
    type_name = type(obj).__name__
    return type_name in ("DataFrame",)


def _dataframe_to_json(df, max_rows: int = 200) -> dict:
    truncated = len(df) > max_rows
    subset = df.head(max_rows)
    return {
        "columns": list(subset.columns),
        "data": subset.values.tolist(),
        "shape": list(df.shape),
        "truncated": truncated,
    }
