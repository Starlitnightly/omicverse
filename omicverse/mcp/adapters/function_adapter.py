"""
Adapter for stateless (non-AnnData) functions.

Handles tools like ``ov.utils.read`` that take scalar / path arguments and
return JSON-serializable data, DataFrames, or new AnnData objects.
"""

from __future__ import annotations

import json
from typing import Any

from .base import BaseAdapter
from ..session_store import SessionStore, SessionError


class FunctionAdapter(BaseAdapter):
    """Adapter for ``execution_class == "stateless"`` tools."""

    def can_handle(self, entry: dict) -> bool:
        return entry.get("execution_class") == "stateless"

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

        kwargs = self.prepare_function_kwargs(entry, args)

        try:
            result = func(**kwargs)
        except Exception as exc:
            return self.normalize_exception(exc, entry)

        return self._build_response(result, entry, ctx, store)

    def prepare_function_kwargs(self, entry: dict, args: dict) -> dict:
        """Map MCP arguments to Python keyword arguments.

        Strips MCP-only keys (like ``adata_id``) that don't belong in the
        Python call.
        """
        kwargs = dict(args)
        # Remove keys not in the function signature
        kwargs.pop("adata_id", None)
        kwargs.pop("instance_id", None)
        return kwargs

    def _build_response(
        self,
        result: Any,
        entry: dict,
        ctx: dict,
        store: SessionStore,
    ) -> dict:
        """Convert the raw Python return into a structured MCP response."""
        outputs = []

        # Check if result is AnnData-like (for ov.utils.read)
        if _is_anndata(result):
            try:
                adata_id = store.create_adata(result, metadata={"source": "function_adapter"})
            except SessionError as exc:
                return {
                    "ok": False,
                    "tool_name": entry.get("tool_name", ""),
                    "error_code": exc.error_code,
                    "message": str(exc),
                    "details": exc.details,
                    "suggested_next_tools": ["ov.get_limits", "ov.cleanup_runtime"],
                }
            outputs.append({
                "type": "object_ref",
                "ref_type": "adata",
                "ref_id": adata_id,
            })
            summary = f"Loaded dataset: {result.shape[0]} obs x {result.shape[1]} vars"
            return self.normalize_result(
                result, entry, ctx,
                outputs=outputs,
                state_updates={},
                warnings_list=[],
            ) | {"summary": summary}

        # Check if result is DataFrame-like
        if _is_dataframe(result):
            table_data = _dataframe_to_json(result)
            outputs.append({"type": "table", "data": table_data})
            return self.normalize_result(result, entry, ctx, outputs=outputs)

        # Scalar / dict / list — JSON output
        if result is None:
            outputs.append({"type": "json", "data": None})
        else:
            try:
                json.dumps(result)  # test serializable
                outputs.append({"type": "json", "data": result})
            except (TypeError, ValueError):
                outputs.append({"type": "text", "data": str(result)})

        return self.normalize_result(result, entry, ctx, outputs=outputs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_anndata(obj) -> bool:
    """Check if *obj* is an AnnData instance without importing anndata."""
    type_name = type(obj).__name__
    return type_name in ("AnnData", "MuData")


def _is_dataframe(obj) -> bool:
    type_name = type(obj).__name__
    return type_name in ("DataFrame",)


def _dataframe_to_json(df, max_rows: int = 100) -> dict:
    """Convert a DataFrame to a JSON-serializable dict."""
    truncated = len(df) > max_rows
    subset = df.head(max_rows)
    return {
        "columns": list(subset.columns),
        "data": subset.values.tolist(),
        "shape": list(df.shape),
        "truncated": truncated,
    }
