"""
MCP server for the OmicVerse registry.

Exposes registered tools via the Model Context Protocol, with built-in
meta tools for discovery and session management.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
import uuid
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl

from .manifest import build_registry_manifest
from .session_store import SessionStore, SessionError, TraceRecord, RuntimeLimits
from .executor import McpExecutor
from .local_oauth import LocalOAuthProvider, LOCAL_AUTH_SCOPES
from .adata_kernel_runtime import AdataKernelRuntime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Meta tool definitions (not from registry)
# ---------------------------------------------------------------------------

META_TOOLS: Dict[str, dict] = {
    "ov.list_tools": {
        "tool_name": "ov.list_tools",
        "description": "List available OmicVerse tools, optionally filtered by category",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (e.g. preprocessing, pl, single)",
                },
                "execution_class": {
                    "type": "string",
                    "enum": ["stateless", "adata", "class"],
                    "description": "Filter by execution class",
                },
            },
        },
    },
    "ov.search_tools": {
        "tool_name": "ov.search_tools",
        "description": "Search OmicVerse tools by keyword across names, aliases, and descriptions",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum results to return",
                },
            },
            "required": ["query"],
        },
    },
    "ov.describe_tool": {
        "tool_name": "ov.describe_tool",
        "description": "Get detailed description of a specific OmicVerse tool including parameters, prerequisites, and availability",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Canonical tool name (e.g. ov.pp.pca)",
                },
            },
            "required": ["tool_name"],
        },
    },
    # -- Session management meta tools ---------------------------------------
    "ov.get_session": {
        "tool_name": "ov.get_session",
        "description": "Get information about the current session (ID, handle counts, persist directory)",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "ov.list_handles": {
        "tool_name": "ov.list_handles",
        "description": "List all handles (adata, artifact, instance) in the current session",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["adata", "artifact", "instance"],
                    "description": "Filter by handle type",
                },
            },
        },
    },
    "ov.persist_adata": {
        "tool_name": "ov.persist_adata",
        "description": "Save an AnnData dataset to disk as .h5ad with metadata sidecar",
        "inputSchema": {
            "type": "object",
            "properties": {
                "adata_id": {
                    "type": "string",
                    "description": "Dataset handle to persist",
                },
                "path": {
                    "type": "string",
                    "description": "Optional explicit file path (default: auto-generated)",
                },
            },
            "required": ["adata_id"],
        },
    },
    "ov.restore_adata": {
        "tool_name": "ov.restore_adata",
        "description": "Restore an AnnData dataset from a .h5ad file into the current session",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to .h5ad file",
                },
                "adata_id": {
                    "type": "string",
                    "description": "Optional: reuse a specific adata_id",
                },
            },
            "required": ["path"],
        },
    },
    # -- Observability meta tools -------------------------------------------
    "ov.get_metrics": {
        "tool_name": "ov.get_metrics",
        "description": "Get aggregated session metrics (handle counts, tool call stats)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["session", "tools"],
                    "description": "Scope: 'session' for summary, 'tools' for per-tool stats",
                },
            },
        },
    },
    "ov.list_events": {
        "tool_name": "ov.list_events",
        "description": "List recent session events (handle lifecycle, tool calls)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Max events to return",
                },
                "event_type": {
                    "type": "string",
                    "description": "Filter by event type",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Filter by tool name",
                },
            },
        },
    },
    "ov.get_trace": {
        "tool_name": "ov.get_trace",
        "description": "Get details of a single tool call trace by trace_id",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_id": {
                    "type": "string",
                    "description": "Trace identifier",
                },
            },
            "required": ["trace_id"],
        },
    },
    "ov.list_traces": {
        "tool_name": "ov.list_traces",
        "description": "List recent tool call traces with timing and status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Max traces to return",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Filter by tool name",
                },
                "ok": {
                    "type": "boolean",
                    "description": "Filter by success/failure",
                },
            },
        },
    },
    # -- Artifact meta tools ------------------------------------------------
    "ov.list_artifacts": {
        "tool_name": "ov.list_artifacts",
        "description": "List artifacts in the current session with optional filters",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_type": {
                    "type": "string",
                    "enum": ["file", "image", "table", "json", "plot", "report", "export"],
                    "description": "Filter by artifact type",
                },
                "content_type": {
                    "type": "string",
                    "description": "Filter by MIME content type",
                },
                "source_tool": {
                    "type": "string",
                    "description": "Filter by producing tool name",
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Max artifacts to return",
                },
            },
        },
    },
    "ov.describe_artifact": {
        "tool_name": "ov.describe_artifact",
        "description": "Get full metadata for an artifact including file status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "Artifact handle ID",
                },
            },
            "required": ["artifact_id"],
        },
    },
    "ov.register_artifact": {
        "tool_name": "ov.register_artifact",
        "description": "Manually register an existing file as a session artifact",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to register",
                },
                "content_type": {
                    "type": "string",
                    "default": "application/octet-stream",
                    "description": "MIME content type",
                },
                "artifact_type": {
                    "type": "string",
                    "default": "file",
                    "description": "Artifact type (file, image, table, json, plot, report, export)",
                },
                "source_tool": {
                    "type": "string",
                    "description": "Tool that produced this artifact",
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata",
                },
            },
            "required": ["path"],
        },
    },
    "ov.delete_artifact": {
        "tool_name": "ov.delete_artifact",
        "description": "Delete an artifact handle, optionally deleting the underlying file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "Artifact handle to delete",
                },
                "delete_file": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also delete the file on disk (default: false)",
                },
            },
            "required": ["artifact_id"],
        },
    },
    "ov.cleanup_artifacts": {
        "tool_name": "ov.cleanup_artifacts",
        "description": "Batch cleanup artifacts by filters (default: dry-run preview only)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_type": {
                    "type": "string",
                    "description": "Only cleanup artifacts of this type",
                },
                "older_than_seconds": {
                    "type": "number",
                    "description": "Only cleanup artifacts older than N seconds",
                },
                "delete_files": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also delete files on disk (default: false)",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview only, do not delete (default: true)",
                },
            },
        },
    },
    "ov.export_artifacts_manifest": {
        "tool_name": "ov.export_artifacts_manifest",
        "description": "Export all session artifacts as a JSON manifest for auditing",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # -- Runtime safety meta tools -------------------------------------------
    "ov.get_limits": {
        "tool_name": "ov.get_limits",
        "description": "Get current quota and TTL configuration with usage counts",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "ov.cleanup_runtime": {
        "tool_name": "ov.cleanup_runtime",
        "description": "Run manual cleanup of expired runtime data (events, traces, artifacts)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["events", "traces", "artifacts", "all"],
                    "default": "all",
                    "description": "What to clean up (default: all)",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview only, do not delete (default: true)",
                },
                "delete_files": {
                    "type": "boolean",
                    "default": False,
                    "description": "For artifacts, also delete underlying files (default: false)",
                },
            },
        },
    },
    "ov.get_health": {
        "tool_name": "ov.get_health",
        "description": "Lightweight health summary with quota proximity warnings",
        "inputSchema": {"type": "object", "properties": {}},
    },
}


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------


def _extract_refs_in(arguments: dict) -> List[str]:
    """Extract handle references from tool arguments (safe summary)."""
    refs = []
    for key in ("adata_id", "instance_id"):
        val = arguments.get(key)
        if val and isinstance(val, str):
            refs.append(val)
    return refs


def _extract_refs_out(result: dict) -> List[str]:
    """Extract handle references from tool result outputs."""
    refs = []
    for output in result.get("outputs", []):
        ref_id = output.get("ref_id")
        if ref_id:
            refs.append(ref_id)
        data = output.get("data", {})
        if isinstance(data, dict):
            val = data.get("adata_id")
            if val:
                refs.append(val)
    return refs


def _summarize_arguments(arguments: dict, max_value_len: int = 120) -> dict:
    """Build a log-safe summary of tool arguments."""
    summary: Dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if key in {"adata_id", "instance_id", "path", "tool_name", "action"}:
            summary[key] = value
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            text = repr(value)
            summary[key] = value if len(text) <= max_value_len else f"{text[:max_value_len - 3]}..."
            continue
        if isinstance(value, dict):
            summary[key] = f"<dict keys={sorted(value.keys())[:10]}>"
            continue
        if isinstance(value, list):
            summary[key] = f"<list len={len(value)}>"
            continue
        summary[key] = f"<{type(value).__name__}>"
    return summary


def _summarize_payload(payload: Any, max_value_len: int = 160) -> Any:
    """Build a compact, log-safe summary for result details/messages."""
    if isinstance(payload, dict):
        return {
            key: _summarize_payload(value, max_value_len=max_value_len)
            for key, value in list(payload.items())[:20]
        }
    if isinstance(payload, list):
        return [
            _summarize_payload(value, max_value_len=max_value_len)
            for value in payload[:10]
        ]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        text = repr(payload)
        return payload if len(text) <= max_value_len else f"{text[:max_value_len - 3]}..."
    return f"<{type(payload).__name__}>"


# ---------------------------------------------------------------------------
# RegistryMcpServer
# ---------------------------------------------------------------------------


class RegistryMcpServer:
    """MCP server backed by the OmicVerse function registry.

    Parameters
    ----------
    phase : str
        Rollout phase(s) to expose.  ``"P0"`` for core pipeline,
        ``"P0+P0.5"`` for core + analysis/viz.
    session_id : str, optional
        Logical session identifier.  Defaults to ``"default"``.
    persist_dir : str, optional
        Directory for persisting adata.  Created lazily on first persist.
    """

    def __init__(
        self,
        phase: str = "P0",
        session_id: Optional[str] = None,
        persist_dir: Optional[str] = None,
        limits: Optional[RuntimeLimits] = None,
    ):
        self._phase = phase
        self._store = SessionStore(session_id=session_id, persist_dir=persist_dir, limits=limits)
        self._adata_runtime = AdataKernelRuntime(store=self._store)

        # Build manifest for the requested phase
        self._manifest = build_registry_manifest(phase=phase)

        # Create executor
        self._executor = McpExecutor(self._manifest, self._store)

        logger.info(
            "RegistryMcpServer initialized: phase=%s, %d tools, session=%s",
            phase,
            len(self._manifest),
            self._store.session_id,
        )

    @property
    def executor(self) -> McpExecutor:
        return self._executor

    @property
    def store(self) -> SessionStore:
        return self._store

    # -- MCP protocol methods -----------------------------------------------

    def list_tools(self) -> List[dict]:
        """Return MCP tool schemas for all visible tools (incl. meta tools)."""
        tools = []

        # Meta tools first
        for name, meta in META_TOOLS.items():
            tools.append({
                "name": name,
                "description": meta["description"],
                "inputSchema": meta["inputSchema"],
            })

        # Registry tools
        for entry in self._manifest:
            tools.append(self.build_tool_schema(entry))

        return tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Dispatch a tool call.

        Returns the executor response envelope.  Every call is wrapped
        with a trace record for observability.
        """
        # Session TTL gate
        if self._store.check_session_expired():
            self._store._obs_metrics["expired_handle_rejections_total"] += 1
            try:
                self._store.record_event("session_expired", {
                    "session_id": self._store.session_id,
                    "session_ttl": self._store.limits.session_ttl_seconds,
                })
            except Exception:
                pass
            return {
                "ok": False,
                "error_code": "session_expired",
                "message": f"Session {self._store.session_id!r} has expired",
                "details": {"session_id": self._store.session_id},
                "suggested_next_tools": [],
            }

        trace_id = uuid.uuid4().hex[:16]
        started_at = time.time()
        logger.info(
            "Tool call start trace_id=%s tool=%s args=%s",
            trace_id,
            name,
            _summarize_arguments(arguments),
        )

        # Dispatch
        if name == "ov.list_tools":
            result = self._handle_list_tools(arguments)
        elif name == "ov.search_tools":
            result = self._handle_search_tools(arguments)
        elif name == "ov.describe_tool":
            result = self._handle_describe_tool(arguments)
        elif name == "ov.get_session":
            result = self._handle_get_session(arguments)
        elif name == "ov.list_handles":
            result = self._handle_list_handles(arguments)
        elif name == "ov.persist_adata":
            result = self._handle_persist_adata(arguments)
        elif name == "ov.restore_adata":
            result = self._handle_restore_adata(arguments)
        elif name == "ov.get_metrics":
            result = self._handle_get_metrics(arguments)
        elif name == "ov.list_events":
            result = self._handle_list_events(arguments)
        elif name == "ov.get_trace":
            result = self._handle_get_trace(arguments)
        elif name == "ov.list_traces":
            result = self._handle_list_traces(arguments)
        elif name == "ov.list_artifacts":
            result = self._handle_list_artifacts(arguments)
        elif name == "ov.describe_artifact":
            result = self._handle_describe_artifact(arguments)
        elif name == "ov.register_artifact":
            result = self._handle_register_artifact(arguments)
        elif name == "ov.delete_artifact":
            result = self._handle_delete_artifact(arguments)
        elif name == "ov.cleanup_artifacts":
            result = self._handle_cleanup_artifacts(arguments)
        elif name == "ov.export_artifacts_manifest":
            result = self._handle_export_artifacts_manifest(arguments)
        elif name == "ov.get_limits":
            result = self._handle_get_limits(arguments)
        elif name == "ov.cleanup_runtime":
            result = self._handle_cleanup_runtime(arguments)
        elif name == "ov.get_health":
            result = self._handle_get_health(arguments)
        else:
            result = self._executor.execute_tool(name, arguments)

        # Record trace (fail-safe)
        finished_at = time.time()
        try:
            self._store.record_trace(TraceRecord(
                trace_id=trace_id,
                session_id=self._store.session_id,
                tool_name=name,
                tool_type=self._classify_tool_type(name),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at) * 1000,
                ok=result.get("ok", False),
                error_code=result.get("error_code"),
                handle_refs_in=_extract_refs_in(arguments),
                handle_refs_out=_extract_refs_out(result),
            ))
            # Record tool event for meta tools (executor records its own)
            if name in META_TOOLS:
                event_type = "tool_called" if result.get("ok") else "tool_failed"
                self._store.record_event(event_type, {
                    "tool_name": name,
                    "trace_id": trace_id,
                    "ok": result.get("ok"),
                    "error_code": result.get("error_code"),
                })
        except Exception:
            pass

        logger.info(
            "Tool call end trace_id=%s tool=%s ok=%s error_code=%s duration_ms=%.2f refs_in=%s refs_out=%s",
            trace_id,
            name,
            result.get("ok", False),
            result.get("error_code"),
            (finished_at - started_at) * 1000,
            _extract_refs_in(arguments),
            _extract_refs_out(result),
        )
        if not result.get("ok", False):
            logger.error(
                "Tool call failed trace_id=%s tool=%s error_code=%s message=%s details=%s",
                trace_id,
                name,
                result.get("error_code"),
                result.get("message"),
                _summarize_payload(result.get("details", {})),
            )

        return result

    async def call_tool_async(self, name: str, arguments: dict) -> dict:
        """Async dispatch used by MCP transports to preserve cancellation."""
        if self._store.check_session_expired():
            self._store._obs_metrics["expired_handle_rejections_total"] += 1
            try:
                self._store.record_event("session_expired", {
                    "session_id": self._store.session_id,
                    "session_ttl": self._store.limits.session_ttl_seconds,
                })
            except Exception:
                pass
            return {
                "ok": False,
                "error_code": "session_expired",
                "message": f"Session {self._store.session_id!r} has expired",
                "details": {"session_id": self._store.session_id},
                "suggested_next_tools": [],
            }

        trace_id = uuid.uuid4().hex[:16]
        started_at = time.time()
        logger.info(
            "Tool call start trace_id=%s tool=%s args=%s",
            trace_id,
            name,
            _summarize_arguments(arguments),
        )

        try:
            if self._should_route_to_adata_runtime(name, arguments):
                entry = self._executor.resolve_entry(name)
                if entry is None:
                    result = {
                        "ok": False,
                        "error_code": "tool_not_found",
                        "message": f"Unknown tool: {name}",
                        "details": {"tool_name": name},
                        "suggested_next_tools": [],
                    }
                else:
                    result = await self._adata_runtime.execute(entry, arguments)
            else:
                if name == "ov.persist_adata" and arguments.get("adata_id"):
                    try:
                        await self._adata_runtime.flush_dirty(arguments["adata_id"])
                    except Exception:
                        pass
                result = self.call_tool(name, arguments)
                return result
        except asyncio.CancelledError:
            logger.warning("Tool call cancelled trace_id=%s tool=%s", trace_id, name)
            raise
        except Exception as exc:
            result = {
                "ok": False,
                "error_code": "execution_failed",
                "message": str(exc),
                "details": {"tool_name": name},
                "suggested_next_tools": [],
            }

        finished_at = time.time()
        self._record_trace_and_logs(
            trace_id=trace_id,
            tool_name=name,
            arguments=arguments,
            result=result,
            started_at=started_at,
            finished_at=finished_at,
        )
        return result

    def _should_route_to_adata_runtime(self, name: str, arguments: dict) -> bool:
        if name in META_TOOLS:
            return False
        if "adata_id" not in arguments:
            return False
        entry = self._executor.resolve_entry(name)
        if entry is None:
            return False
        return entry.get("execution_class") == "adata"

    def _record_trace_and_logs(
        self,
        *,
        trace_id: str,
        tool_name: str,
        arguments: dict,
        result: dict,
        started_at: float,
        finished_at: float,
    ) -> None:
        try:
            self._store.record_trace(TraceRecord(
                trace_id=trace_id,
                session_id=self._store.session_id,
                tool_name=tool_name,
                tool_type=self._classify_tool_type(tool_name),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at) * 1000,
                ok=result.get("ok", False),
                error_code=result.get("error_code"),
                handle_refs_in=_extract_refs_in(arguments),
                handle_refs_out=_extract_refs_out(result),
            ))
            event_type = "tool_called" if result.get("ok") else "tool_failed"
            self._store.record_event(event_type, {
                "tool_name": tool_name,
                "trace_id": trace_id,
                "ok": result.get("ok"),
                "error_code": result.get("error_code"),
            })
        except Exception:
            pass

        duration_ms = (finished_at - started_at) * 1000
        logger.info(
            "Tool call end trace_id=%s tool=%s ok=%s error_code=%s duration_ms=%.1f",
            trace_id,
            tool_name,
            result.get("ok", False),
            result.get("error_code"),
            duration_ms,
        )
        if not result.get("ok", False):
            logger.error(
                "Tool call failed trace_id=%s tool=%s error_code=%s message=%s details=%s",
                trace_id,
                tool_name,
                result.get("error_code"),
                result.get("message", ""),
                _summarize_payload(result.get("details", {})),
            )

    # -- Discovery meta tool handlers ----------------------------------------

    def _handle_list_tools(self, args: dict) -> dict:
        """Implement ``ov.list_tools``."""
        category = args.get("category")
        execution_class = args.get("execution_class")

        tools = []
        for entry in self._manifest:
            if category and entry.get("category") != category:
                continue
            if execution_class and entry.get("execution_class") != execution_class:
                continue
            tools.append({
                "tool_name": entry["tool_name"],
                "description": entry.get("description", ""),
                "category": entry.get("category", ""),
                "execution_class": entry.get("execution_class", ""),
                "risk_level": entry.get("risk_level", ""),
                "status": entry.get("status", ""),
                "availability": entry.get("availability", {}).get("available", True),
            })

        return {
            "ok": True,
            "tool_name": "ov.list_tools",
            "summary": f"Found {len(tools)} tools",
            "outputs": [{"type": "json", "data": tools}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_search_tools(self, args: dict) -> dict:
        """Implement ``ov.search_tools``."""
        query = args.get("query", "").lower()
        max_results = args.get("max_results", 10)

        if not query:
            return {
                "ok": False,
                "error_code": "invalid_arguments",
                "message": "query is required",
                "details": {},
                "suggested_next_tools": [],
            }

        scored: List[tuple] = []
        for entry in self._manifest:
            score = self._search_score(query, entry)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, entry in scored[:max_results]:
            results.append({
                "tool_name": entry["tool_name"],
                "description": entry.get("description", ""),
                "category": entry.get("category", ""),
                "score": round(score, 3),
                "availability": entry.get("availability", {}).get("available", True),
            })

        return {
            "ok": True,
            "tool_name": "ov.search_tools",
            "summary": f"Found {len(results)} matches for '{query}'",
            "outputs": [{"type": "json", "data": results}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_describe_tool(self, args: dict) -> dict:
        """Implement ``ov.describe_tool``."""
        tool_name = args.get("tool_name", "")
        entry = self._executor.resolve_entry(tool_name)

        if entry is None:
            return {
                "ok": False,
                "error_code": "tool_not_found",
                "message": f"Unknown tool: {tool_name}",
                "details": {},
                "suggested_next_tools": ["ov.list_tools", "ov.search_tools"],
            }

        # Return full manifest entry (minus internal function ref)
        desc = {k: v for k, v in entry.items() if not k.startswith("_")}

        # For class tools: refresh runtime availability and ensure actions shown
        if entry.get("execution_class") == "class":
            from .class_specs import get_spec as _get_class_spec
            from .availability import check_class_availability
            spec = _get_class_spec(entry.get("full_name", ""))
            if spec is not None:
                avail_ok, avail_reason = check_class_availability(spec)
                desc.setdefault("availability", {})
                desc["availability"] = dict(desc["availability"])
                desc["availability"]["available"] = avail_ok
                desc["availability"]["reason"] = avail_reason
                if "class_actions" not in desc:
                    from .manifest import _build_class_actions_summary
                    desc["class_actions"] = _build_class_actions_summary(spec)

        return {
            "ok": True,
            "tool_name": "ov.describe_tool",
            "summary": f"Description for {tool_name}",
            "outputs": [{"type": "json", "data": desc}],
            "state_updates": {},
            "warnings": [],
        }

    # -- Session management meta tool handlers -------------------------------

    def _handle_get_session(self, args: dict) -> dict:
        """Implement ``ov.get_session``."""
        return {
            "ok": True,
            "tool_name": "ov.get_session",
            "summary": f"Session: {self._store.session_id}",
            "outputs": [{"type": "json", "data": self._store.session_info()}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_list_handles(self, args: dict) -> dict:
        """Implement ``ov.list_handles``."""
        handles = self._store.list_handles()
        type_filter = args.get("type")
        if type_filter:
            handles = [h for h in handles if h["type"] == type_filter]
        return {
            "ok": True,
            "tool_name": "ov.list_handles",
            "summary": f"Found {len(handles)} handles",
            "outputs": [{"type": "json", "data": handles}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_persist_adata(self, args: dict) -> dict:
        """Implement ``ov.persist_adata``."""
        adata_id = args.get("adata_id", "")
        path = args.get("path")

        if not adata_id:
            return {
                "ok": False,
                "error_code": "invalid_arguments",
                "message": "adata_id is required",
                "details": {},
                "suggested_next_tools": ["ov.list_handles"],
            }

        try:
            asyncio.run(self._adata_runtime.flush_dirty(adata_id))
        except RuntimeError:
            # Already inside an event loop or no loop available; best effort only.
            pass
        except Exception:
            pass

        try:
            result = self._store.persist_adata(adata_id, path=path)
        except SessionError as exc:
            return {
                "ok": False,
                "error_code": exc.error_code,
                "message": str(exc),
                "details": exc.details,
                "suggested_next_tools": ["ov.list_handles"],
            }
        except KeyError as exc:
            return {
                "ok": False,
                "error_code": "handle_not_found",
                "message": str(exc),
                "details": {"adata_id": adata_id},
                "suggested_next_tools": ["ov.list_handles"],
            }
        except Exception as exc:
            return {
                "ok": False,
                "error_code": "persistence_failed",
                "message": f"Failed to persist adata: {exc}",
                "details": {"adata_id": adata_id},
                "suggested_next_tools": [],
            }

        return {
            "ok": True,
            "tool_name": "ov.persist_adata",
            "summary": f"Persisted {adata_id} to {result['path']}",
            "outputs": [{"type": "json", "data": result}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_restore_adata(self, args: dict) -> dict:
        """Implement ``ov.restore_adata``."""
        path = args.get("path", "")
        adata_id = args.get("adata_id")

        if not path:
            return {
                "ok": False,
                "error_code": "invalid_arguments",
                "message": "path is required",
                "details": {},
                "suggested_next_tools": [],
            }

        try:
            new_id = self._store.restore_adata(path, adata_id=adata_id)
        except FileNotFoundError as exc:
            return {
                "ok": False,
                "error_code": "persistence_failed",
                "message": str(exc),
                "details": {"path": path},
                "suggested_next_tools": [],
            }
        except ImportError as exc:
            return {
                "ok": False,
                "error_code": "persistence_failed",
                "message": str(exc),
                "details": {"path": path},
                "suggested_next_tools": [],
            }
        except Exception as exc:
            return {
                "ok": False,
                "error_code": "persistence_failed",
                "message": f"Failed to restore adata: {exc}",
                "details": {"path": path},
                "suggested_next_tools": [],
            }

        return {
            "ok": True,
            "tool_name": "ov.restore_adata",
            "summary": f"Restored adata from {path} as {new_id}",
            "outputs": [{"type": "object_ref", "ref_type": "adata", "ref_id": new_id}],
            "state_updates": {},
            "warnings": [],
        }

    # -- Observability meta tool handlers ------------------------------------

    def _classify_tool_type(self, name: str) -> str:
        """Classify a tool as meta, class, or registry."""
        if name in META_TOOLS:
            return "meta"
        entry = self._executor.resolve_entry(name)
        if entry and entry.get("execution_class") == "class":
            return "class"
        return "registry"

    def _handle_get_metrics(self, args: dict) -> dict:
        """Implement ``ov.get_metrics``."""
        scope = args.get("scope", "session")
        metrics = self._store.get_metrics(scope=scope)
        return {
            "ok": True,
            "tool_name": "ov.get_metrics",
            "summary": f"Metrics (scope={scope})",
            "outputs": [{"type": "json", "data": metrics}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_list_events(self, args: dict) -> dict:
        """Implement ``ov.list_events``."""
        events = self._store.list_events(
            limit=args.get("limit", 50),
            event_type=args.get("event_type"),
            tool_name=args.get("tool_name"),
        )
        return {
            "ok": True,
            "tool_name": "ov.list_events",
            "summary": f"Found {len(events)} events",
            "outputs": [{"type": "json", "data": events}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_get_trace(self, args: dict) -> dict:
        """Implement ``ov.get_trace``."""
        trace_id = args.get("trace_id", "")
        if not trace_id:
            return {
                "ok": False,
                "error_code": "invalid_arguments",
                "message": "trace_id is required",
                "details": {},
                "suggested_next_tools": ["ov.list_traces"],
            }
        trace = self._store.get_trace(trace_id)
        if trace is None:
            return {
                "ok": False,
                "error_code": "handle_not_found",
                "message": f"Trace not found: {trace_id}",
                "details": {"trace_id": trace_id},
                "suggested_next_tools": ["ov.list_traces"],
            }
        return {
            "ok": True,
            "tool_name": "ov.get_trace",
            "summary": f"Trace {trace_id}",
            "outputs": [{"type": "json", "data": trace}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_list_traces(self, args: dict) -> dict:
        """Implement ``ov.list_traces``."""
        traces = self._store.list_traces(
            limit=args.get("limit", 50),
            tool_name=args.get("tool_name"),
            ok=args.get("ok"),
        )
        return {
            "ok": True,
            "tool_name": "ov.list_traces",
            "summary": f"Found {len(traces)} traces",
            "outputs": [{"type": "json", "data": traces}],
            "state_updates": {},
            "warnings": [],
        }

    # -- Artifact meta tool handlers -----------------------------------------

    def _handle_list_artifacts(self, args: dict) -> dict:
        """Implement ``ov.list_artifacts``."""
        artifacts = self._store.list_artifacts(
            artifact_type=args.get("artifact_type"),
            content_type=args.get("content_type"),
            source_tool=args.get("source_tool"),
            limit=args.get("limit", 50),
        )
        return {
            "ok": True,
            "tool_name": "ov.list_artifacts",
            "summary": f"Found {len(artifacts)} artifacts",
            "outputs": [{"type": "json", "data": artifacts}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_describe_artifact(self, args: dict) -> dict:
        """Implement ``ov.describe_artifact``."""
        artifact_id = args.get("artifact_id", "")
        if not artifact_id:
            return {
                "ok": False,
                "error_code": "invalid_arguments",
                "message": "artifact_id is required",
                "details": {},
                "suggested_next_tools": ["ov.list_artifacts"],
            }
        try:
            desc = self._store.describe_artifact(artifact_id)
        except SessionError as exc:
            return {
                "ok": False,
                "error_code": exc.error_code,
                "message": str(exc),
                "details": exc.details,
                "suggested_next_tools": ["ov.list_artifacts"],
            }
        except KeyError as exc:
            return {
                "ok": False,
                "error_code": "handle_not_found",
                "message": str(exc),
                "details": {"artifact_id": artifact_id},
                "suggested_next_tools": ["ov.list_artifacts"],
            }
        return {
            "ok": True,
            "tool_name": "ov.describe_artifact",
            "summary": f"Artifact {artifact_id}",
            "outputs": [{"type": "json", "data": desc}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_register_artifact(self, args: dict) -> dict:
        """Implement ``ov.register_artifact``."""
        path = args.get("path", "")
        if not path:
            return {
                "ok": False,
                "error_code": "invalid_arguments",
                "message": "path is required",
                "details": {},
                "suggested_next_tools": [],
            }
        import os
        warnings = []
        if not os.path.isfile(path):
            warnings.append(f"File does not exist: {path}")
        try:
            artifact_id = self._store.create_artifact(
                path=path,
                content_type=args.get("content_type", "application/octet-stream"),
                metadata=args.get("metadata"),
                artifact_type=args.get("artifact_type", "file"),
                source_tool=args.get("source_tool", ""),
            )
        except SessionError as exc:
            return {
                "ok": False,
                "error_code": exc.error_code,
                "message": str(exc),
                "details": exc.details,
                "suggested_next_tools": ["ov.get_limits", "ov.cleanup_runtime"],
            }
        return {
            "ok": True,
            "tool_name": "ov.register_artifact",
            "summary": f"Registered artifact {artifact_id}",
            "outputs": [{"type": "json", "data": {
                "artifact_id": artifact_id,
                "path": path,
                "file_exists": os.path.isfile(path),
            }}],
            "state_updates": {},
            "warnings": warnings,
        }

    def _handle_delete_artifact(self, args: dict) -> dict:
        """Implement ``ov.delete_artifact``."""
        artifact_id = args.get("artifact_id", "")
        if not artifact_id:
            return {
                "ok": False,
                "error_code": "invalid_arguments",
                "message": "artifact_id is required",
                "details": {},
                "suggested_next_tools": ["ov.list_artifacts"],
            }
        try:
            result = self._store.delete_artifact(
                artifact_id,
                delete_file=args.get("delete_file", False),
            )
        except SessionError as exc:
            return {
                "ok": False,
                "error_code": exc.error_code,
                "message": str(exc),
                "details": exc.details,
                "suggested_next_tools": ["ov.list_artifacts"],
            }
        except KeyError as exc:
            return {
                "ok": False,
                "error_code": "handle_not_found",
                "message": str(exc),
                "details": {"artifact_id": artifact_id},
                "suggested_next_tools": ["ov.list_artifacts"],
            }
        return {
            "ok": True,
            "tool_name": "ov.delete_artifact",
            "summary": f"Deleted artifact {artifact_id}",
            "outputs": [{"type": "json", "data": result}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_cleanup_artifacts(self, args: dict) -> dict:
        """Implement ``ov.cleanup_artifacts``."""
        result = self._store.cleanup_artifacts(
            artifact_type=args.get("artifact_type"),
            older_than_seconds=args.get("older_than_seconds"),
            delete_files=args.get("delete_files", False),
            dry_run=args.get("dry_run", True),
        )
        mode = "dry-run" if result["dry_run"] else "executed"
        return {
            "ok": True,
            "tool_name": "ov.cleanup_artifacts",
            "summary": f"Cleanup {mode}: {result['matched']} matched, {result['deleted']} deleted",
            "outputs": [{"type": "json", "data": result}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_export_artifacts_manifest(self, args: dict) -> dict:
        """Implement ``ov.export_artifacts_manifest``."""
        manifest = self._store.export_artifacts_manifest()
        return {
            "ok": True,
            "tool_name": "ov.export_artifacts_manifest",
            "summary": f"Exported manifest with {manifest['artifact_count']} artifacts",
            "outputs": [{"type": "json", "data": manifest}],
            "state_updates": {},
            "warnings": [],
        }

    # -- Runtime safety meta tool handlers -----------------------------------

    def _handle_get_limits(self, args: dict) -> dict:
        """Implement ``ov.get_limits``."""
        lim = self._store.limits
        stats = self._store.stats
        return {
            "ok": True,
            "tool_name": "ov.get_limits",
            "summary": "Current quota and TTL configuration",
            "outputs": [{"type": "json", "data": {
                "quotas": {
                    "max_adata_per_session": lim.max_adata_per_session,
                    "max_artifacts_per_session": lim.max_artifacts_per_session,
                    "max_instances_per_session": lim.max_instances_per_session,
                    "max_events_per_session": lim.max_events_per_session,
                    "max_traces_per_session": lim.max_traces_per_session,
                },
                "ttl": {
                    "event_ttl_seconds": lim.event_ttl_seconds,
                    "trace_ttl_seconds": lim.trace_ttl_seconds,
                    "artifact_ttl_seconds": lim.artifact_ttl_seconds,
                    "session_ttl_seconds": lim.session_ttl_seconds,
                },
                "usage": {
                    "adata_count": stats["adata_count"],
                    "artifact_count": stats["artifact_count"],
                    "instance_count": stats["instance_count"],
                    "event_count": len(self._store._events),
                    "trace_count": len(self._store._traces),
                },
            }}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_cleanup_runtime(self, args: dict) -> dict:
        """Implement ``ov.cleanup_runtime``."""
        target = args.get("target", "all")
        dry_run = args.get("dry_run", True)
        delete_files = args.get("delete_files", False)
        result = self._store.cleanup_runtime(
            target=target, dry_run=dry_run, delete_files=delete_files,
        )
        return {
            "ok": True,
            "tool_name": "ov.cleanup_runtime",
            "summary": f"Cleanup {'preview' if dry_run else 'executed'}: "
                       f"{result['total_deleted']} deleted",
            "outputs": [{"type": "json", "data": result}],
            "state_updates": {},
            "warnings": [],
        }

    def _handle_get_health(self, args: dict) -> dict:
        """Implement ``ov.get_health``."""
        lim = self._store.limits
        stats = self._store.stats
        metrics = self._store.get_metrics()
        now = time.time()
        session_age = round(now - self._store._created_at, 1)

        warnings: List[str] = []
        # Quota proximity warnings (>80%)
        for resource, current, limit in [
            ("adata", stats["adata_count"], lim.max_adata_per_session),
            ("artifact", stats["artifact_count"], lim.max_artifacts_per_session),
            ("instance", stats["instance_count"], lim.max_instances_per_session),
        ]:
            if limit > 0 and current / limit > 0.8:
                warnings.append(
                    f"{resource}: {current}/{limit} ({round(100*current/limit)}%)"
                )

        return {
            "ok": True,
            "tool_name": "ov.get_health",
            "summary": "Session health summary",
            "outputs": [{"type": "json", "data": {
                "session_id": self._store.session_id,
                "session_age_seconds": session_age,
                "session_expired": self._store.check_session_expired(),
                "handles": {
                    "adata": stats["adata_count"],
                    "artifacts": stats["artifact_count"],
                    "instances": stats["instance_count"],
                },
                "quota_rejections_total": metrics.get("quota_rejections_total", 0),
                "cleanup_runs_total": metrics.get("cleanup_runs_total", 0),
                "last_cleanup_at": self._store._last_cleanup_at,
                "warnings": warnings,
            }}],
            "state_updates": {},
            "warnings": warnings,
        }

    # -- Schema builder ------------------------------------------------------

    @staticmethod
    def build_tool_schema(entry: dict) -> dict:
        """Convert a manifest entry into an MCP tool schema."""
        return {
            "name": entry["tool_name"],
            "description": entry.get("description", ""),
            "inputSchema": entry.get("parameter_schema", {
                "type": "object",
                "properties": {},
            }),
        }

    # -- Search scoring ------------------------------------------------------

    @staticmethod
    def _search_score(query: str, entry: dict) -> float:
        """Score an entry against a search query."""
        score = 0.0
        query_lower = query.lower()

        # Exact match in tool_name
        tool_name = entry.get("tool_name", "").lower()
        if query_lower in tool_name:
            score += 2.0
        if tool_name.endswith("." + query_lower):
            score += 3.0

        # Match in aliases
        for alias in entry.get("aliases", []):
            if query_lower == alias.lower():
                score += 2.5
            elif query_lower in alias.lower():
                score += 1.0

        # Match in description
        description = entry.get("description", "").lower()
        if query_lower in description:
            score += 1.0

        # Fuzzy match on tool name
        ratio = SequenceMatcher(None, query_lower, tool_name).ratio()
        if ratio > 0.5:
            score += ratio

        # Match in category
        if query_lower == entry.get("category", "").lower():
            score += 1.5

        return score

    # -- MCP SDK transport helpers -------------------------------------------

    def _apply_mcp_runtime_safeguards(self, transport_name: str) -> None:
        """Disable noisy stdout-oriented runtime output for MCP transports."""
        monitor_disabled = False
        try:
            from .._monitor import set_monitor_display
            set_monitor_display(False)
            monitor_disabled = True
        except Exception:
            pass
        logger.info(
            "MCP %s safeguards enabled: tool stdout redirected to stderr; monitor_display=%s",
            transport_name,
            "off" if monitor_disabled else "unknown",
        )

    def _build_sdk_server(self):
        """Create and configure the MCP SDK server instance."""
        try:
            from mcp.server import Server
            from mcp.types import Tool, TextContent
        except ImportError:
            raise ImportError(
                "The 'mcp' package is required for MCP transport support. "
                "Install it with: pip install mcp"
            )

        server = Server("omicverse-registry")

        @server.list_tools()
        async def handle_list_tools():
            logger.info("Received MCP list_tools request")
            tools = self.list_tools()
            return [
                Tool(
                    name=t["name"],
                    description=t.get("description", ""),
                    inputSchema=t.get("inputSchema", {}),
                )
                for t in tools
            ]

        @server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            with contextlib.redirect_stdout(sys.stderr):
                result = await self.call_tool_async(name, arguments or {})
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        return server

    # -- Stdio transport (requires mcp SDK) ----------------------------------

    def run_stdio(self):
        """Start the MCP server on stdio transport.

        Requires the ``mcp`` package to be installed.
        """
        import asyncio
        asyncio.run(self._run_stdio_async())

    async def _run_stdio_async(self):
        """Async entrypoint for stdio MCP transport."""
        try:
            from mcp.server.stdio import stdio_server
        except ImportError:
            raise ImportError(
                "The 'mcp' package is required for stdio transport. "
                "Install it with: pip install mcp"
            )

        server = self._build_sdk_server()
        self._apply_mcp_runtime_safeguards("stdio")

        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP stdio transport ready; awaiting client messages")
            await server.run(read_stream, write_stream, server.create_initialization_options())

    # -- Streamable HTTP transport -------------------------------------------

    def run_streamable_http(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        path: str = "/mcp",
    ) -> None:
        """Start the MCP server on local Streamable HTTP transport."""
        import asyncio
        asyncio.run(self._run_streamable_http_async(host=host, port=port, path=path))

    async def _run_streamable_http_async(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        path: str = "/mcp",
    ) -> None:
        """Async entrypoint for Streamable HTTP MCP transport."""
        try:
            import uvicorn
            from starlette.applications import Starlette
            from starlette.middleware import Middleware
            from starlette.middleware.authentication import AuthenticationMiddleware
            from starlette.routing import Route
            from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
            from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
            from mcp.server.auth.routes import (
                build_resource_metadata_url,
                create_auth_routes,
                create_protected_resource_routes,
            )
            from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
            from mcp.server.fastmcp.server import StreamableHTTPASGIApp
            from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        except ImportError:
            raise ImportError(
                "Streamable HTTP transport requires: mcp, uvicorn, and starlette. "
                "Install them with: pip install mcp uvicorn starlette"
            )

        if not path.startswith("/"):
            path = "/" + path

        server = self._build_sdk_server()
        self._apply_mcp_runtime_safeguards("streamable-http")

        issuer_url = AnyHttpUrl(f"http://{host}:{port}")
        resource_url = AnyHttpUrl(f"http://{host}:{port}{path}")
        auth_settings = AuthSettings(
            issuer_url=issuer_url,
            resource_server_url=resource_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                default_scopes=LOCAL_AUTH_SCOPES,
                valid_scopes=LOCAL_AUTH_SCOPES,
            ),
            revocation_options=RevocationOptions(enabled=False),
            required_scopes=[],
        )
        auth_provider = LocalOAuthProvider()
        token_verifier = auth_provider.build_token_verifier()
        session_manager = StreamableHTTPSessionManager(app=server)
        streamable_http_app = StreamableHTTPASGIApp(session_manager)
        resource_metadata_url = build_resource_metadata_url(resource_url)
        routes = [
            Route(
                path,
                endpoint=RequireAuthMiddleware(
                    streamable_http_app,
                    auth_settings.required_scopes or [],
                    resource_metadata_url,
                ),
            ),
        ]
        routes.extend(
            create_auth_routes(
                provider=auth_provider,
                issuer_url=issuer_url,
                client_registration_options=auth_settings.client_registration_options,
                revocation_options=auth_settings.revocation_options,
            )
        )
        routes.extend(
            create_protected_resource_routes(
                resource_url=resource_url,
                authorization_servers=[issuer_url],
                scopes_supported=auth_settings.required_scopes,
            )
        )
        # Claude Code probes a few discovery aliases before settling on the main
        # authorization metadata endpoints. Serve identical JSON on those aliases
        # instead of returning bare 404 text.
        oauth_metadata_handler = routes[1].endpoint
        protected_metadata_handler = routes[-1].endpoint
        routes.extend([
            Route("/.well-known/oauth-authorization-server/mcp", endpoint=oauth_metadata_handler, methods=["GET", "OPTIONS"]),
            Route("/.well-known/openid-configuration", endpoint=oauth_metadata_handler, methods=["GET", "OPTIONS"]),
            Route("/.well-known/openid-configuration/mcp", endpoint=oauth_metadata_handler, methods=["GET", "OPTIONS"]),
            Route(f"{path}/.well-known/openid-configuration", endpoint=oauth_metadata_handler, methods=["GET", "OPTIONS"]),
            Route("/.well-known/oauth-protected-resource", endpoint=protected_metadata_handler, methods=["GET", "OPTIONS"]),
        ])
        app = Starlette(
            debug=False,
            routes=routes,
            middleware=[
                Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(token_verifier)),
                Middleware(AuthContextMiddleware),
            ],
            lifespan=lambda app: session_manager.run(),
        )

        logger.info(
            "MCP streamable-http transport ready on http://%s:%d%s",
            host,
            port,
            path,
        )
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
        )
        http_server = uvicorn.Server(config)
        await http_server.serve()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _get_version() -> str:
    """Return installed omicverse version, or 'unknown' if not found."""
    try:
        from importlib.metadata import version
        return version("omicverse")
    except Exception:
        return "unknown"


def main():
    """CLI entrypoint: ``python -m omicverse.mcp`` or ``omicverse-mcp``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="OmicVerse Registry MCP Server — exposes analysis tools via stdio or local HTTP MCP transports",
        epilog="All log output goes to stderr. In stdio mode, stdout is reserved for MCP JSON-RPC protocol.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s (omicverse {_get_version()})",
    )
    parser.add_argument(
        "--phase",
        default="P0+P0.5",
        help="Rollout phase(s) to expose. Options: P0 (core pipeline), "
             "P0+P0.5 (core + analysis/viz, default), P0+P0.5+P2 (all including class tools)",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="Transport to serve MCP over (default: stdio).",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Logical session identifier for handle isolation. "
             "Defaults to 'default'. Use unique IDs for multi-session setups.",
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Directory for persisting AnnData datasets via ov.persist_adata. "
             "Auto-created on first persist call if not specified.",
    )
    parser.add_argument(
        "--max-adata",
        type=int,
        default=None,
        help="Max AnnData handles per session (default: 50)",
    )
    parser.add_argument(
        "--max-artifacts",
        type=int,
        default=None,
        help="Max artifact handles per session (default: 200)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for streamable-http transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port for streamable-http transport (default: 8765).",
    )
    parser.add_argument(
        "--http-path",
        default="/mcp",
        help="Route path for streamable-http transport (default: /mcp).",
    )
    args = parser.parse_args()

    # All log output must go to stderr — stdout is reserved for MCP JSON-RPC.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # Build RuntimeLimits from CLI overrides
    limits_kwargs: dict = {}
    if args.max_adata is not None:
        limits_kwargs["max_adata_per_session"] = args.max_adata
    if args.max_artifacts is not None:
        limits_kwargs["max_artifacts_per_session"] = args.max_artifacts
    limits = RuntimeLimits(**limits_kwargs) if limits_kwargs else None

    srv = RegistryMcpServer(
        phase=args.phase,
        session_id=args.session_id,
        persist_dir=args.persist_dir,
        limits=limits,
    )
    logger.info(
        "Starting OmicVerse MCP server (transport=%s, phase=%s, %d tools, session=%s)",
        args.transport,
        args.phase,
        len(srv._manifest),
        srv._store.session_id,
    )
    if args.transport == "stdio":
        srv.run_stdio()
    else:
        srv.run_streamable_http(host=args.host, port=args.port, path=args.http_path)


if __name__ == "__main__":
    main()
