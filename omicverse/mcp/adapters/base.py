"""
Base adapter interface for MCP tool execution.

All adapters normalize Python function calls into the standard MCP response
envelope: ``{ok, tool_name, summary, outputs, state_updates, warnings}``.
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..session_store import SessionStore


class BaseAdapter(ABC):
    """Abstract base for all MCP tool adapters."""

    @abstractmethod
    def can_handle(self, entry: dict) -> bool:
        """Return ``True`` if this adapter handles the given manifest entry."""

    @abstractmethod
    def invoke(self, entry: dict, args: dict, store: SessionStore) -> dict:
        """Execute the tool and return a response envelope."""

    def build_call_context(
        self,
        entry: dict,
        args: dict,
        store: SessionStore,
    ) -> dict:
        """Assemble common context for a tool invocation."""
        return {
            "tool_name": entry.get("tool_name", ""),
            "full_name": entry.get("full_name", ""),
            "execution_class": entry.get("execution_class", ""),
            "args": args,
        }

    def normalize_result(
        self,
        result: Any,
        entry: dict,
        ctx: dict,
        outputs: Optional[List[dict]] = None,
        state_updates: Optional[dict] = None,
        warnings_list: Optional[List[str]] = None,
    ) -> dict:
        """Wrap a successful result into the standard response envelope."""
        tool_name = entry.get("tool_name", "")
        description = entry.get("description", "")
        summary = f"{description}" if description else f"{tool_name} completed"

        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": summary,
            "outputs": outputs or [],
            "state_updates": state_updates or {},
            "warnings": warnings_list or [],
        }

    def normalize_exception(self, exc: Exception, entry: dict) -> dict:
        """Wrap an exception into the standard error envelope."""
        tool_name = entry.get("tool_name", "")
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_str = "".join(tb[-3:])  # last 3 frames

        return {
            "ok": False,
            "tool_name": tool_name,
            "error_code": "execution_failed",
            "message": str(exc),
            "details": {"traceback": tb_str},
            "suggested_next_tools": [],
        }
