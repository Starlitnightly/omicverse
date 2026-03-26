"""ToolRuntime — tool dispatch hub and handler implementations.

Extracted from ``smart_agent.py`` to keep tool handling in one place.
All tool handler methods follow the pattern ``_tool_<name>(self, ...)``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..harness.runtime_state import runtime_state
from ..harness.tool_catalog import (
    get_tool_spec,
    get_visible_tool_schemas,
    normalize_tool_name,
    resolve_tool_search,
)
from ..._registry import _global_registry

if TYPE_CHECKING:
    from .analysis_executor import AnalysisExecutor
    from .permission_policy import PermissionPolicy
    from .protocol import AgentContext
    from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def _truncate_text(text: str, limit: int) -> str:
    """Truncate text for LLM-facing tool output while preserving total length."""
    if limit <= 0 or len(text) <= limit:
        return text
    suffix = f"\n... (truncated, total_chars={len(text)})"
    keep = max(0, limit - len(suffix))
    return text[:keep] + suffix


# Legacy OmicVerse-specific tool schemas that are not part of the
# Claude-style tool catalog but must still be exposed to the LLM.
LEGACY_AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "inspect_data",
        "description": "Inspect the AnnData or MuData object without modifying it.",
        "parameters": {
            "type": "object",
            "properties": {
                "aspect": {
                    "type": "string",
                    "enum": ["shape", "obs", "var", "obsm", "uns", "layers", "full"],
                }
            },
            "required": ["aspect"],
        },
    },
    {
        "name": "execute_code",
        "description": "Execute Python code against the current dataset.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["code", "description"],
        },
    },
    {
        "name": "run_snippet",
        "description": "Run read-only Python code on a shallow copy of the current dataset.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "search_functions",
        "description": "Search the OmicVerse function registry.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_skills",
        "description": "Search installed domain-specific skills.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "delegate",
        "description": "Delegate to an explore, plan, or execute subagent.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_type": {"type": "string", "enum": ["explore", "plan", "execute"]},
                "task": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["agent_type", "task"],
        },
    },
    {
        "name": "web_fetch",
        "description": "Fetch a URL and return readable content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web and return results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "number"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_download",
        "description": "Download a file from a URL to disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "filename": {"type": "string"},
                "directory": {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "finish",
        "description": "Declare the task complete and return the summary.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


class ToolRuntime:
    """Dispatch and handle all agent tool calls.

    This is the single source of truth for tool visibility, plan-mode
    gating, legacy tool names, deferred loading semantics, and tool
    dispatch.  Other modules should call into ``ToolRuntime`` rather than
    duplicating these concerns.

    Parameters
    ----------
    ctx : AgentContext
        The agent instance (accessed via protocol surface).
    executor : AnalysisExecutor
        The code execution engine (for execute_code / run_snippet).
    """

    def __init__(self, ctx: "AgentContext", executor: "AnalysisExecutor") -> None:
        from .tool_registry import build_default_registry

        self._ctx = ctx
        self._executor = executor
        self._subagent_controller: Any = None  # late-bound
        self._registry: "ToolRegistry" = build_default_registry()
        self._bind_handlers()

    def set_subagent_controller(self, controller: Any) -> None:
        """Late-bind the subagent controller to avoid circular deps."""
        self._subagent_controller = controller

    def _require_subagent_controller(self) -> Any:
        """Return the bound subagent controller or raise a clear runtime error."""
        if self._subagent_controller is None:
            raise RuntimeError(
                "Subagent controller is not initialized; Agent delegation is unavailable."
            )
        return self._subagent_controller

    @property
    def registry(self) -> "ToolRegistry":
        """The tool registry backing this runtime's dispatch."""
        return self._registry

    # ------------------------------------------------------------------
    # Tool facade — visibility, plan-mode gating, loaded-tool tracking
    # ------------------------------------------------------------------

    def get_visible_agent_tools(
        self, *, allowed_names: Optional[set[str]] = None
    ) -> list[dict[str, Any]]:
        """Return the currently visible tool schemas for the active session.

        Merges Claude-style catalog tools (core + loaded deferred) with the
        legacy OmicVerse tool schemas.  When *allowed_names* is provided, only
        tools whose name (or normalized name) appears in the set are returned.
        """
        session_id = self._ctx._get_runtime_session_id()
        loaded = runtime_state.get_loaded_tools(session_id)
        tools = get_visible_tool_schemas(loaded) + list(LEGACY_AGENT_TOOLS)
        if allowed_names is None:
            return tools
        normalized_allowed = {normalize_tool_name(name) for name in allowed_names}
        return [
            tool for tool in tools
            if tool["name"] in allowed_names
            or normalize_tool_name(tool["name"]) in normalized_allowed
        ]

    def get_loaded_tool_names(self) -> list[str]:
        """Return the set of currently loaded tool names for this session."""
        return runtime_state.get_loaded_tools(self._ctx._get_runtime_session_id())

    def tool_blocked_in_plan_mode(self, tool_name: str) -> bool:
        """Check whether *tool_name* is blocked while plan mode is active."""
        spec = get_tool_spec(tool_name)
        session_state = runtime_state.get_summary(
            self._ctx._get_runtime_session_id()
        )
        plan_mode = bool(
            (session_state.get("plan_mode") or {}).get("enabled", False)
        )
        if not plan_mode:
            return False
        if spec is not None:
            return spec.high_risk or spec.name in {
                "Bash", "Edit", "Write", "NotebookEdit", "EnterWorktree",
            }
        return tool_name in {"execute_code", "web_download"}

    # ------------------------------------------------------------------
    # Registry-driven handler bindings
    # ------------------------------------------------------------------

    def _bind_handlers(self) -> None:
        """Bind handler callables to the registry for all registered tools.

        Each handler has the uniform signature
        ``(args: dict, adata: Any, request: str) -> Any``.
        """
        r = self._registry
        # fmt: off
        r.register_handler("tool_search", lambda a, d, _: self._tool_tool_search(
            a.get("query", ""), max_results=a.get("max_results", 5)))
        r.register_handler("bash", lambda a, d, _: self._tool_bash(
            a.get("command", ""), description=a.get("description", ""),
            timeout=a.get("timeout", 120000),
            run_in_background=bool(a.get("run_in_background", False)),
            dangerouslyDisableSandbox=bool(a.get("dangerouslyDisableSandbox", False))))
        r.register_handler("read", lambda a, d, _: self._tool_read(
            a.get("file_path", ""), offset=a.get("offset", 0),
            limit=a.get("limit", 2000), pages=a.get("pages", "")))
        r.register_handler("edit", lambda a, d, _: self._tool_edit(
            a.get("file_path", ""), a.get("old_string", ""),
            a.get("new_string", ""), replace_all=bool(a.get("replace_all", False))))
        r.register_handler("write", lambda a, d, _: self._tool_write(
            a.get("file_path", ""), a.get("content", "")))
        r.register_handler("glob", lambda a, d, _: self._tool_glob(
            a.get("pattern", ""), root=a.get("root", ""),
            max_results=a.get("max_results", 200)))
        r.register_handler("grep", lambda a, d, _: self._tool_grep(
            a.get("pattern", ""), root=a.get("root", ""),
            glob=a.get("glob", ""), max_results=a.get("max_results", 200)))
        r.register_handler("notebook_edit", lambda a, d, _: self._tool_notebook_edit(
            a.get("file_path", ""), cell_index=a.get("cell_index", 0),
            source=a.get("source", ""), cell_type=a.get("cell_type", "")))
        r.register_handler("inspect_data", lambda a, d, _: self._tool_inspect_data(
            d, a.get("aspect", "full")))
        r.register_handler("execute_code", self._dispatch_execute_code)
        r.register_handler("run_snippet", self._dispatch_run_snippet)
        r.register_handler("search_functions", lambda a, d, _: self._tool_search_functions(
            a.get("query", "")))
        r.register_handler("agent", self._dispatch_agent)
        r.register_handler("ask_user_question", self._dispatch_ask_user_question)
        r.register_handler("task_create", lambda a, d, _: self._tool_create_task(
            a.get("title", ""), description=a.get("description", ""),
            status=a.get("status", "pending")))
        r.register_handler("task_get", lambda a, d, _: self._tool_get_task(
            a.get("task_id", "")))
        r.register_handler("task_list", lambda a, d, _: self._tool_list_tasks(
            status=a.get("status", "")))
        r.register_handler("task_output", lambda a, d, _: self._tool_task_output(
            a.get("task_id", ""), offset=a.get("offset", 0),
            limit=a.get("limit", 200)))
        r.register_handler("task_stop", lambda a, d, _: self._tool_task_stop(
            a.get("task_id", "")))
        r.register_handler("task_update", lambda a, d, _: self._tool_task_update(
            a.get("task_id", ""), a.get("status", ""),
            summary=a.get("summary", "")))
        r.register_handler("enter_plan_mode", lambda a, d, _: self._tool_enter_plan_mode(
            reason=a.get("reason", "")))
        r.register_handler("exit_plan_mode", lambda a, d, _: self._tool_exit_plan_mode(
            summary=a.get("summary", "")))
        r.register_handler("enter_worktree", lambda a, d, _: self._tool_enter_worktree(
            branch_name=a.get("branch_name", ""),
            path=a.get("path", ""), base_ref=a.get("base_ref", "HEAD")))
        r.register_handler("skill", lambda a, d, _: self._tool_skill(
            a.get("query", ""), mode=a.get("mode", "search")))
        r.register_handler("web_fetch", lambda a, d, _: self._tool_web_fetch(
            a.get("url", ""), prompt=a.get("prompt")))
        r.register_handler("web_search", lambda a, d, _: self._tool_web_search(
            a.get("query", ""), num_results=a.get("num_results", 5)))
        r.register_handler("list_mcp_resources", lambda a, d, _: self._tool_list_mcp_resources(
            server=a.get("server", "")))
        r.register_handler("read_mcp_resource", lambda a, d, _: self._tool_read_mcp_resource(
            a.get("server", ""), a.get("uri", "")))
        r.register_handler("web_download", lambda a, d, _: self._tool_web_download(
            a.get("url", ""), filename=a.get("filename"),
            directory=a.get("directory")))
        r.register_handler("finish", lambda a, d, _: self._tool_finish(
            a.get("summary", "")))
        # fmt: on

    # ------------------------------------------------------------------
    # Dispatch helpers (validation / async wrappers)
    # ------------------------------------------------------------------

    def _dispatch_execute_code(
        self, args: dict, adata: Any, request: str
    ) -> Any:
        code = args.get("code", "")
        if not code or not code.strip():
            return json.dumps(
                {"error": "execute_code requires a non-empty 'code' argument."},
                ensure_ascii=False,
            )
        return self._tool_execute_code(code, args.get("description", ""), adata)

    def _dispatch_run_snippet(
        self, args: dict, adata: Any, request: str
    ) -> Any:
        snippet = args.get("code", "")
        if not snippet or not snippet.strip():
            return json.dumps(
                {"error": "run_snippet requires a non-empty 'code' argument."},
                ensure_ascii=False,
            )
        return self._tool_run_snippet(snippet, adata)

    async def _dispatch_agent(
        self, args: dict, adata: Any, request: str
    ) -> Any:
        agent_type = args.get(
            "subagent_type", args.get("agent_type", "explore")
        )
        task = args.get("task", "")
        context = args.get("context", "")
        logger.info(
            "delegation_started agent_type=%s task=%s",
            agent_type,
            task[:80],
        )
        subagent_controller = self._require_subagent_controller()
        sub_result = await subagent_controller.run_subagent(
            agent_type=agent_type,
            task=task,
            adata=adata,
            context=context,
        )
        if agent_type == "execute":
            return {
                "adata": sub_result["adata"],
                "output": sub_result["result"],
            }
        return sub_result["result"]

    def _dispatch_ask_user_question(
        self, args: dict, adata: Any, request: str
    ) -> Any:
        question = args.get("question", "") or args.get("prompt", "")
        if not question:
            return json.dumps(
                {"error": "AskUserQuestion requires a non-empty 'question' argument."},
                ensure_ascii=False,
            )
        return self._tool_ask_user_question(
            question,
            header=args.get("header", ""),
            options=args.get("options", []),
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch_tool(
        self,
        tool_call: Any,
        current_adata: Any,
        request: str,
        *,
        permission_policy: Optional["PermissionPolicy"] = None,
    ) -> Any:
        """Dispatch a tool call through the registry and return the result.

        Resolution flow:
        1. Resolve the tool name via the registry (handles canonical names,
           catalog aliases, legacy aliases, and case normalization).
        2. Check the optional *permission_policy* (deny → immediate return).
        3. Check plan-mode blocking for the resolved canonical name.
        4. Look up the bound handler callable.
        5. Call the handler with ``(args, adata, request)`` and await if async.

        Parameters
        ----------
        permission_policy : PermissionPolicy, optional
            When provided, the tool is checked against the policy before
            dispatch.  Denied tools return an error message immediately
            without executing.  This is used by subagent isolation to
            enforce scoped tool restrictions.
        """
        raw_name = tool_call.name
        args = tool_call.arguments

        # Registry-driven name resolution (delegates to catalog for aliases)
        canonical = self._registry.resolve_name(raw_name)
        if not canonical:
            # Fallback through catalog normalization for edge cases
            canonical = normalize_tool_name(raw_name)

        # Permission policy check (when provided by caller)
        if permission_policy is not None:
            decision = permission_policy.check(canonical)
            if decision.is_denied:
                return (
                    f"Permission denied for {canonical}: {decision.reason}"
                )

        if self.tool_blocked_in_plan_mode(canonical):
            return (
                f"{canonical} is blocked because the session is "
                "currently in plan mode."
            )

        handler = self._registry.get_handler(canonical)
        if handler is None:
            return f"Unknown tool: {raw_name}"

        result = handler(args, current_adata, request)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    # ------------------------------------------------------------------
    # OmicVerse data tools
    # ------------------------------------------------------------------

    def _tool_inspect_data(self, adata: Any, aspect: str) -> str:
        if adata is None:
            return (
                "No dataset is loaded. Use web_download to download a "
                "dataset first, then load it with execute_code."
            )
        try:
            parts: List[str] = []
            dtype = type(adata).__name__
            is_mudata = dtype == "MuData"

            if is_mudata and aspect in ("full", "shape"):
                parts.append("Type: MuData")
                if hasattr(adata, "mod"):
                    mod_keys = list(adata.mod.keys())
                    parts.append(f"Modalities: {mod_keys}")
                    for mk in mod_keys:
                        mod = adata.mod[mk]
                        layers_keys = (
                            list(mod.layers.keys())
                            if getattr(mod, "layers", None) is not None
                            else []
                        )
                        parts.append(
                            f"  {mk}: {mod.shape[0]} cells x "
                            f"{mod.shape[1]} features, layers={layers_keys}"
                        )
                if hasattr(adata, "shape"):
                    parts.append(
                        f"Combined shape: {adata.shape[0]} obs x "
                        f"{adata.shape[1]} vars"
                    )

            if aspect in ("shape", "full") and not is_mudata:
                parts.append(
                    f"Shape: {adata.shape[0]} cells x {adata.shape[1]} genes"
                )
            if aspect in ("obs", "full"):
                cols = list(adata.obs.columns)
                parts.append(f"obs columns ({len(cols)}): {cols}")
                try:
                    parts.append(
                        f"obs.head(3):\n{adata.obs.head(3).to_string()}"
                    )
                except Exception:
                    pass
            if aspect in ("var", "full") and not is_mudata:
                cols = list(adata.var.columns)
                parts.append(f"var columns ({len(cols)}): {cols}")
                try:
                    parts.append(
                        f"var.head(3):\n{adata.var.head(3).to_string()}"
                    )
                except Exception:
                    pass
            if aspect in ("obsm", "full"):
                keys = (
                    list(adata.obsm.keys()) if hasattr(adata, "obsm") else []
                )
                parts.append(f"obsm keys: {keys}")
                for k in keys:
                    try:
                        parts.append(f"  {k}: shape {adata.obsm[k].shape}")
                    except Exception:
                        pass
            if aspect in ("uns", "full"):
                keys = (
                    list(adata.uns.keys()) if hasattr(adata, "uns") else []
                )
                parts.append(f"uns keys: {keys}")
            if aspect in ("layers", "full"):
                layers = getattr(adata, "layers", None)
                keys = list(layers.keys()) if layers is not None else []
                parts.append(f"layers: {keys}")
            return (
                "\n".join(parts) if parts else f"Unknown aspect: {aspect}"
            )
        except Exception as e:
            return f"Error inspecting data: {e}"

    def _tool_execute_code(
        self, code: str, description: str, adata: Any
    ) -> dict:
        from .analysis_executor import ProactiveCodeTransformer
        from .repair_loop import ExecutionRepairLoop

        code = ProactiveCodeTransformer().transform(code)
        if getattr(self._ctx, "_code_only_mode", False):
            capture = getattr(self._ctx, "_capture_code_only_snippet", None)
            if callable(capture):
                capture(code, description=description)
            return {
                "adata": adata,
                "output": (
                    "CODE ONLY MODE: captured generated Python code without "
                    "executing it."
                ),
            }

        prereq_warnings = self._executor.check_code_prerequisites(code, adata)

        self._executor._notebook_fallback_error = None

        # --- Structured self-healing loop ---
        repair_loop = ExecutionRepairLoop(self._executor, max_retries=3)

        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                repair_result = pool.submit(
                    _asyncio.run,
                    repair_loop.run(code, adata, phase="execution"),
                ).result()
        else:
            repair_result = _asyncio.run(
                repair_loop.run(code, adata, phase="execution")
            )

        if repair_result.success:
            result = repair_result.exec_result
            stdout = result.get("stdout", "") if isinstance(result, dict) else ""
            result_adata = (
                result.get("adata", adata)
                if isinstance(result, dict)
                else (result if result is not None else adata)
            )

            output_parts: List[str] = []
            debug_output_parts: List[str] = []
            notebook_err = getattr(
                self._executor, "_notebook_fallback_error", None
            )
            if notebook_err:
                notebook_msg = (
                    f"WARNING: notebook session execution failed with error:\n{notebook_err}\n"
                    f"Fell back to in-process execution. Please fix the code to avoid this error."
                )
                output_parts.append(notebook_msg)
                debug_output_parts.append(notebook_msg)

            # Annotate recovered attempts
            if len(repair_result.attempts) > 1:
                strategies = [
                    a.strategy for a in repair_result.attempts if not a.success
                ]
                recovery_msg = (
                    f"RECOVERED after {len(repair_result.attempts)} attempt(s) "
                    f"(strategies: {', '.join(strategies)})"
                )
                output_parts.append(recovery_msg)
                debug_output_parts.append(recovery_msg)

            if prereq_warnings:
                warning_msg = f"PREREQUISITE WARNINGS: {prereq_warnings}"
                output_parts.append(warning_msg)
                debug_output_parts.append(warning_msg)
            if stdout.strip():
                output_parts.append(
                    f"stdout:\n{_truncate_text(stdout, 3000)}"
                )
                debug_output_parts.append(f"stdout:\n{stdout}")
            try:
                result_msg = (
                    f"Result adata shape: {result_adata.shape[0]} cells x "
                    f"{result_adata.shape[1]} features"
                )
                output_parts.append(result_msg)
                debug_output_parts.append(result_msg)
            except Exception:
                result_msg = f"Result type: {type(result_adata).__name__}"
                output_parts.append(result_msg)
                debug_output_parts.append(result_msg)

            llm_output = (
                "\n".join(output_parts)
                if output_parts
                else "Code executed successfully (no output)."
            )
            debug_output = (
                "\n".join(debug_output_parts)
                if debug_output_parts
                else "Code executed successfully (no output)."
            )
            return {
                "adata": result_adata,
                "output": llm_output,
                "debug_output": debug_output,
                "stdout": stdout,
            }

        # All repair attempts failed — return structured diagnostic
        envelope = repair_result.final_envelope
        if envelope is not None:
            error_output = (
                f"ERROR: {envelope.exception}: {envelope.summary}\n\n"
                f"Phase: {envelope.phase}\n"
                f"Attempts: {envelope.retry_count + 1}/{repair_loop.max_retries + 1}\n"
                f"Traceback (last {len(envelope.traceback_excerpt)} chars):\n"
                f"{envelope.traceback_excerpt}"
            )
            if envelope.repair_hints:
                error_output += (
                    "\nRepair hints:\n"
                    + "\n".join(f"  - {h}" for h in envelope.repair_hints)
                )
        else:
            error_output = "ERROR: execution failed with no diagnostic envelope"

        debug_error_output = error_output
        if prereq_warnings:
            error_output = (
                f"PREREQUISITE WARNINGS: {prereq_warnings}\n\n"
                f"{error_output}"
            )
            debug_error_output = (
                f"PREREQUISITE WARNINGS: {prereq_warnings}\n\n"
                f"{debug_error_output}"
            )
        return {
            "adata": adata,
            "output": error_output,
            "debug_output": debug_error_output,
            "stdout": "",
        }

    def _tool_run_snippet(self, code: str, adata: Any) -> str:
        """Read-only snippet — no adata copy, no serialisation round-trip."""
        try:
            return self._executor.execute_snippet_readonly(code, adata)
        except Exception as e:
            return f"ERROR: {e}"

    def _tool_search_functions(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Please provide a non-empty function search query."

        matches: List[Dict[str, Any]] = []
        seen_full_names: set[str] = set()

        def _extend(entries: List[Dict[str, Any]]) -> None:
            for entry in entries or []:
                full_name = str(
                    entry.get("full_name")
                    or entry.get("short_name")
                    or entry.get("name")
                    or ""
                )
                if not full_name or full_name in seen_full_names:
                    continue
                seen_full_names.add(full_name)
                matches.append(entry)

        try:
            runtime_matches = list(_global_registry.find(query))
        except Exception:
            runtime_matches = []
        _extend(runtime_matches)

        static_search = getattr(self._ctx, "_collect_static_registry_entries", None)
        if callable(static_search):
            try:
                static_matches = list(static_search(query, max_entries=20))
            except TypeError:
                static_matches = list(static_search(query))
            except Exception:
                static_matches = []
            _extend(static_matches)

        if not matches:
            return f"No functions found matching '{query}'. Try broader keywords."

        results: List[str] = []
        for m in matches[:20]:
            fname = m.get("full_name", m.get("short_name", ""))
            sig = m.get("signature", "")
            desc = m.get("description", "")[:300]

            entry_text = f"  {fname}({sig})\n    {desc}"

            branch_parameter = m.get("branch_parameter")
            branch_value = m.get("branch_value")
            if branch_parameter and branch_value:
                entry_text += f"\n    Branch: {branch_parameter}='{branch_value}'"

            prereqs = m.get("prerequisites", {})
            req_funcs = prereqs.get("functions", [])
            if req_funcs:
                entry_text += "\n    Must run first: " + ", ".join(req_funcs)

            requires = m.get("requires", {})
            if requires:
                req_items = [
                    f"{k}['{v}']"
                    for k, vals in requires.items()
                    for v in vals
                ]
                entry_text += "\n    Requires: " + ", ".join(req_items)

            produces = m.get("produces", {})
            if produces:
                prod_items = [
                    f"{k}['{v}']"
                    for k, vals in produces.items()
                    for v in vals
                ]
                entry_text += "\n    Produces: " + ", ".join(prod_items)

            examples = m.get("examples", [])
            code_examples = [
                ex
                for ex in examples
                if ex.strip().startswith(("ov.", "sc."))
            ]
            if code_examples:
                entry_text += "\n    Example: " + code_examples[0]
            elif examples:
                entry_text += "\n    Example: " + examples[0]

            results.append(entry_text)
            if len(results) >= 10:
                break

        return (
            f"Found {len(results)} matching functions:\n"
            + "\n".join(results)
        )

    def _tool_search_skills(self, query: str) -> str:
        registry = self._ctx.skill_registry
        if not registry or not registry.skill_metadata:
            return "No domain skills available."

        query_lower = query.lower()
        scored = []
        for meta in registry.skill_metadata.values():
            searchable = f"{meta.name} {meta.description} {meta.slug}".lower()
            score = sum(1 for word in query_lower.split() if word in searchable)
            if score > 0:
                scored.append((meta, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            slugs = ", ".join(
                m.slug for m in registry.skill_metadata.values()
            )
            return f"No skills matched '{query}'. Available skills: {slugs}"

        results: List[str] = []
        for meta, _ in scored[:2]:
            try:
                full_skill = registry.load_full_skill(meta.slug)
                if full_skill:
                    provider = None
                    llm = self._ctx._llm
                    if llm and hasattr(llm, "config"):
                        provider = llm.config.provider
                    body = full_skill.prompt_instructions(
                        max_chars=4000, provider=provider
                    )
                    results.append(f"=== {full_skill.name} ===\n{body}")
            except Exception:
                pass

        if not results:
            return "Skills matched but content could not be loaded."

        return "\n\n".join(results)

    # ------------------------------------------------------------------
    # Web tools
    # ------------------------------------------------------------------

    def _tool_web_fetch(
        self, url: str, prompt: str = None, timeout: int = 15
    ) -> str:
        import urllib.error
        import urllib.request
        from html.parser import HTMLParser

        class _HTMLToText(HTMLParser):
            def __init__(self):
                super().__init__()
                self._pieces: list = []
                self._skip = False
                self._skip_tags = {
                    "script", "style", "noscript", "svg", "head",
                }

            def handle_starttag(self, tag, attrs):
                if tag in self._skip_tags:
                    self._skip = True
                elif tag in ("br", "hr"):
                    self._pieces.append("\n")
                elif tag in (
                    "p", "div", "tr", "li",
                    "h1", "h2", "h3", "h4", "h5", "h6",
                ):
                    self._pieces.append("\n")

            def handle_endtag(self, tag):
                if tag in self._skip_tags:
                    self._skip = False
                elif tag in (
                    "p", "div", "tr",
                    "h1", "h2", "h3", "h4", "h5", "h6",
                ):
                    self._pieces.append("\n")

            def handle_data(self, data):
                if not self._skip:
                    self._pieces.append(data)

            def get_text(self) -> str:
                raw = "".join(self._pieces)
                lines = []
                for line in raw.splitlines():
                    stripped = " ".join(line.split())
                    if stripped:
                        lines.append(stripped)
                return "\n".join(lines)

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "OmicVerseAgent/1.0 (research bot)",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw_bytes = resp.read(512_000)
                charset = "utf-8"
                if "charset=" in content_type:
                    charset = (
                        content_type.split("charset=")[-1]
                        .split(";")[0]
                        .strip()
                    )
                body = raw_bytes.decode(charset, errors="replace")

            if "html" in content_type.lower() or body.strip().startswith("<"):
                parser = _HTMLToText()
                parser.feed(body)
                text = parser.get_text()
            else:
                text = body

            max_chars = 4000 if prompt else 6000
            if len(text) > max_chars:
                text = (
                    text[:max_chars] + "\n\n... [truncated, page too long]"
                )

            header = f"Content from {url}:\n\n"
            if prompt:
                header = f"Content from {url} (focus: {prompt}):\n\n"
            return header + text

        except urllib.error.HTTPError as e:
            return f"HTTP error fetching {url}: {e.code} {e.reason}"
        except urllib.error.URLError as e:
            return f"URL error fetching {url}: {e.reason}"
        except Exception as e:
            return f"Error fetching {url}: {type(e).__name__}: {e}"

    def _tool_web_search(self, query: str, num_results: int = 5) -> str:
        import re as _re
        import urllib.error
        import urllib.parse
        import urllib.request

        num_results = max(1, min(int(num_results), 10))
        encoded_q = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{encoded_q}"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "OmicVerseAgent/1.0 (research bot)",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read(256_000).decode("utf-8", errors="replace")
        except Exception as e:
            return f"Search error: {type(e).__name__}: {e}"

        results: List[str] = []
        result_blocks = _re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
            r".*?"
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            body,
            _re.DOTALL,
        )

        if not result_blocks:
            links = _re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                body,
            )
            for href, title in links[:num_results]:
                clean_title = _re.sub(r"<[^>]+>", "", title).strip()
                actual_url = href
                uddg_match = _re.search(r"uddg=([^&]+)", href)
                if uddg_match:
                    actual_url = urllib.parse.unquote(uddg_match.group(1))
                results.append(f"- {clean_title}\n  {actual_url}")
        else:
            for href, title, snippet in result_blocks[:num_results]:
                clean_title = _re.sub(r"<[^>]+>", "", title).strip()
                clean_snippet = _re.sub(r"<[^>]+>", "", snippet).strip()
                actual_url = href
                uddg_match = _re.search(r"uddg=([^&]+)", href)
                if uddg_match:
                    actual_url = urllib.parse.unquote(uddg_match.group(1))
                results.append(
                    f"- {clean_title}\n  {actual_url}\n  {clean_snippet}"
                )

        if not results:
            return f"No results found for '{query}'."

        return (
            f"Search results for '{query}':\n\n" + "\n\n".join(results)
        )

    def _tool_web_download(
        self,
        url: str,
        filename: str = None,
        directory: str = None,
    ) -> str:
        import urllib.error
        import urllib.parse
        import urllib.request

        MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

        if not filename:
            path_part = urllib.parse.urlparse(url).path
            filename = os.path.basename(path_part) or "downloaded_file"

        save_dir = directory or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "OmicVerseAgent/1.0 (research bot)",
                },
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_SIZE:
                    size_gb = int(content_length) / (1024**3)
                    return (
                        f"File too large ({size_gb:.1f} GB). "
                        "Max allowed is 2 GB."
                    )

                downloaded = 0
                chunk_size = 1024 * 1024
                with open(save_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > MAX_SIZE:
                            f.close()
                            os.remove(save_path)
                            return (
                                "Download aborted: exceeded 2GB limit at "
                                f"{downloaded / (1024**3):.1f} GB."
                            )
                        f.write(chunk)

            size_bytes = os.path.getsize(save_path)
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                size_str = f"{size_bytes / (1024**2):.1f} MB"
            else:
                size_str = f"{size_bytes / (1024**3):.2f} GB"

            return (
                f"Downloaded successfully:\n"
                f"  File: {save_path}\n"
                f"  Size: {size_str}\n"
                f"You can now load this file with execute_code, e.g.:\n"
                f"  adata = ov.read('{save_path}')"
            )

        except urllib.error.HTTPError as e:
            return f"HTTP error downloading {url}: {e.code} {e.reason}"
        except urllib.error.URLError as e:
            return f"URL error downloading {url}: {e.reason}"
        except Exception as e:
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except OSError:
                    pass
            return f"Download error: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # Claude-style tool helpers
    # ------------------------------------------------------------------

    def _tool_tool_search(self, query: str, max_results: int = 5) -> str:
        session_id = self._ctx._get_runtime_session_id()
        payload = resolve_tool_search(
            query,
            loaded_tools=runtime_state.get_loaded_tools(session_id),
            max_results=max_results,
        )
        selected = payload.get("selected_tools", [])
        loaded_tools: tuple = ()
        if selected:
            loaded_tools = runtime_state.load_tools(session_id, selected)
        payload["loaded_tools"] = list(
            runtime_state.get_loaded_tools(session_id)
        )
        payload["newly_loaded"] = list(loaded_tools)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tool_read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        pages: str = "",
    ) -> str:
        path = self._ctx._resolve_local_path(file_path)
        if not path.exists():
            return f"File not found: {path}"
        if path.is_dir():
            return f"Path is a directory, not a file: {path}"

        suffix = path.suffix.lower()
        if suffix == ".ipynb":
            try:
                import nbformat
            except ImportError:
                return "Notebook reading requires nbformat to be installed."
            nb = nbformat.read(path, as_version=4)
            cells = []
            for idx, cell in enumerate(nb.cells):
                if idx < offset:
                    continue
                if len(cells) >= max(1, limit):
                    break
                source = (
                    cell.source
                    if isinstance(cell.source, str)
                    else "".join(cell.source)
                )
                preview = "\n".join(source.splitlines()[:20])
                cells.append(f"[{idx}] {cell.cell_type}\n{preview}")
            return (
                "\n\n".join(cells)
                if cells
                else f"No notebook cells in range for {path}"
            )

        if suffix == ".pdf":
            try:
                import pypdf
            except ImportError:
                return "PDF reading requires pypdf to be installed."
            reader = pypdf.PdfReader(str(path))
            page_numbers: List[int] = []
            if pages:
                for part in pages.split(","):
                    part = part.strip()
                    if "-" in part:
                        start, end = part.split("-", 1)
                        page_numbers.extend(
                            range(
                                max(1, int(start)),
                                min(len(reader.pages), int(end)) + 1,
                            )
                        )
                    elif part:
                        page_numbers.append(int(part))
            else:
                page_numbers = list(
                    range(1, min(len(reader.pages), 5) + 1)
                )
            snippets = []
            for page_no in page_numbers[:20]:
                text = reader.pages[page_no - 1].extract_text() or ""
                snippets.append(f"## Page {page_no}\n{text[:4000]}")
            return (
                "\n\n".join(snippets)
                if snippets
                else f"No readable PDF text in {path}"
            )

        mime = mimetypes.guess_type(path.name)[0] or ""
        if mime.startswith("image/"):
            stat = path.stat()
            return json.dumps(
                {
                    "type": "image",
                    "path": str(path),
                    "mime": mime,
                    "size_bytes": stat.st_size,
                },
                ensure_ascii=False,
                indent=2,
            )

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        segment = lines[max(0, offset) : max(0, offset) + max(1, limit)]
        numbered = [
            f"{idx + offset + 1:6d}\t{line[:2000]}"
            for idx, line in enumerate(segment)
        ]
        return "\n".join(numbered) if numbered else f"(empty file) {path}"

    def _tool_edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        self._ctx._ensure_server_tool_mode("Edit")
        if self.tool_blocked_in_plan_mode("Edit"):
            return "Edit is blocked while the session is in plan mode."
        path = self._ctx._resolve_local_path(file_path)
        if not path.exists():
            return f"File not found: {path}"
        self._ctx._request_tool_approval(
            "Edit",
            reason=f"Edit file {path}",
            payload={"file_path": str(path)},
        )
        content = path.read_text(encoding="utf-8", errors="replace")
        occurrences = content.count(old_string)
        if occurrences == 0:
            return f"Edit failed: old_string was not found in {path}"
        updated = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        path.write_text(updated, encoding="utf-8")
        return json.dumps(
            {
                "file_path": str(path),
                "replacements": occurrences if replace_all else 1,
                "status": "updated",
            },
            ensure_ascii=False,
            indent=2,
        )

    def _tool_write(self, file_path: str, content: str) -> str:
        self._ctx._ensure_server_tool_mode("Write")
        if self.tool_blocked_in_plan_mode("Write"):
            return "Write is blocked while the session is in plan mode."
        path = self._ctx._resolve_local_path(file_path)
        self._ctx._request_tool_approval(
            "Write",
            reason=f"Write file {path}",
            payload={"file_path": str(path)},
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps(
            {
                "file_path": str(path),
                "bytes_written": len(content.encode("utf-8")),
            },
            ensure_ascii=False,
            indent=2,
        )

    def _tool_glob(
        self, pattern: str, root: str = "", max_results: int = 200
    ) -> str:
        base = (
            self._ctx._resolve_local_path(root, allow_relative=True)
            if root
            else Path(self._ctx._refresh_runtime_working_directory())
        )
        matches = sorted(str(p) for p in base.glob(pattern))[
            : max(1, max_results)
        ]
        return json.dumps(
            {"root": str(base), "pattern": pattern, "matches": matches},
            ensure_ascii=False,
            indent=2,
        )

    def _tool_grep(
        self,
        pattern: str,
        root: str = "",
        glob: str = "",
        max_results: int = 200,
    ) -> str:
        base = (
            self._ctx._resolve_local_path(root, allow_relative=True)
            if root
            else Path(self._ctx._refresh_runtime_working_directory())
        )
        rg_path = shutil.which("rg")
        if rg_path:
            cmd = [rg_path, "-n", pattern, str(base)]
            if glob:
                cmd[1:1] = ["-g", glob]
        else:
            cmd = ["grep", "-R", "-n", pattern, str(base)]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if proc.returncode not in (0, 1):
            return (
                proc.stderr
                or proc.stdout
                or f"Grep failed with exit code {proc.returncode}"
            )
        lines = [
            line
            for line in proc.stdout.splitlines()
            if line.strip()
        ][: max(1, max_results)]
        return "\n".join(lines) if lines else f"No matches for {pattern}"

    def _tool_notebook_edit(
        self,
        file_path: str,
        cell_index: int,
        source: str,
        cell_type: str = "",
    ) -> str:
        self._ctx._ensure_server_tool_mode("NotebookEdit")
        if self.tool_blocked_in_plan_mode("NotebookEdit"):
            return (
                "NotebookEdit is blocked while the session is in plan mode."
            )
        path = self._ctx._resolve_local_path(file_path)
        self._ctx._request_tool_approval(
            "NotebookEdit",
            reason=f"Edit notebook {path}",
            payload={"file_path": str(path), "cell_index": cell_index},
        )
        try:
            import nbformat
        except ImportError:
            return "NotebookEdit requires nbformat to be installed."
        nb = nbformat.read(path, as_version=4)
        if cell_index < 0 or cell_index >= len(nb.cells):
            return f"Notebook cell index out of range: {cell_index}"
        nb.cells[cell_index].source = source
        if cell_type:
            nb.cells[cell_index].cell_type = cell_type
        nbformat.write(nb, path)
        return json.dumps(
            {
                "file_path": str(path),
                "cell_index": cell_index,
                "status": "updated",
            },
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Task management tools
    # ------------------------------------------------------------------

    def _tool_create_task(
        self, title: str, description: str = "", status: str = "pending"
    ) -> str:
        task = runtime_state.create_task(
            self._ctx._get_runtime_session_id(),
            title=title,
            description=description,
            status=status if status else "pending",
        )
        return json.dumps(task.to_dict(), ensure_ascii=False, indent=2)

    def _tool_get_task(self, task_id: str) -> str:
        task = runtime_state.get_task(
            self._ctx._get_runtime_session_id(), task_id
        )
        payload = (
            task.to_dict() if task is not None else {"error": "Task not found"}
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tool_list_tasks(self, status: str = "") -> str:
        tasks = runtime_state.list_tasks(
            self._ctx._get_runtime_session_id(), status=status
        )
        return json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2)

    def _tool_task_output(
        self, task_id: str, offset: int = 0, limit: int = 200
    ) -> str:
        payload = runtime_state.read_task_output(
            self._ctx._get_runtime_session_id(),
            task_id,
            offset=offset,
            limit=limit,
        )
        return json.dumps(
            payload or {"error": "Task not found"},
            ensure_ascii=False,
            indent=2,
        )

    def _tool_task_stop(self, task_id: str) -> str:
        self._ctx._ensure_server_tool_mode("TaskStop")
        updated = runtime_state.stop_task(
            self._ctx._get_runtime_session_id(), task_id
        )
        payload = (
            updated.to_dict()
            if updated is not None
            else {"error": "Task not found"}
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tool_task_update(
        self, task_id: str, status: str, summary: str = ""
    ) -> str:
        updated = runtime_state.update_task(
            self._ctx._get_runtime_session_id(),
            task_id,
            status=status,
            summary=summary,
        )
        payload = (
            updated.to_dict()
            if updated is not None
            else {"error": "Task not found"}
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Plan mode / worktree
    # ------------------------------------------------------------------

    def _tool_enter_plan_mode(self, reason: str = "") -> str:
        payload = runtime_state.enter_plan_mode(
            self._ctx._get_runtime_session_id(), reason=reason
        )
        return json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)

    def _tool_exit_plan_mode(self, summary: str = "") -> str:
        payload = runtime_state.exit_plan_mode(
            self._ctx._get_runtime_session_id(), reason=summary
        )
        return json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)

    def _tool_enter_worktree(
        self,
        branch_name: str = "",
        path: str = "",
        base_ref: str = "HEAD",
    ) -> str:
        self._ctx._ensure_server_tool_mode("EnterWorktree")
        if self.tool_blocked_in_plan_mode("EnterWorktree"):
            return (
                "EnterWorktree is blocked while the session is in plan mode."
            )
        repo_root = self._ctx._detect_repo_root()
        if repo_root is None:
            return json.dumps(
                {
                    "error": "No git repository found for worktree creation.",
                },
                ensure_ascii=False,
                indent=2,
            )
        branch = (
            branch_name.strip()
            or f"ovagent/{self._ctx._get_runtime_session_id()}"
        )
        if path:
            worktree_path = self._ctx._resolve_local_path(
                path, allow_relative=True
            )
        else:
            worktree_root = Path.home() / ".ovagent" / "worktrees"
            worktree_root.mkdir(parents=True, exist_ok=True)
            worktree_path = worktree_root / branch.replace("/", "_")
        self._ctx._request_tool_approval(
            "EnterWorktree",
            reason=f"Create or switch git worktree {worktree_path}",
            payload={
                "branch_name": branch,
                "path": str(worktree_path),
            },
        )
        if not worktree_path.exists():
            proc = subprocess.run(
                [
                    "git", "-C", str(repo_root),
                    "worktree", "add",
                    str(worktree_path), "-b", branch, base_ref,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                proc = subprocess.run(
                    [
                        "git", "-C", str(repo_root),
                        "worktree", "add",
                        str(worktree_path), branch,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            if proc.returncode != 0:
                return json.dumps(
                    {
                        "error": (
                            proc.stderr.strip()
                            or proc.stdout.strip()
                            or "git worktree add failed"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
        worktree = runtime_state.set_worktree(
            self._ctx._get_runtime_session_id(),
            path=str(worktree_path),
            repo_root=str(repo_root),
            branch=branch,
            base_branch=base_ref,
        )
        return json.dumps(worktree.to_dict(), ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Skill / MCP / User interaction
    # ------------------------------------------------------------------

    def _tool_skill(self, query: str, mode: str = "search") -> str:
        if mode == "load":
            return self._ctx._load_skill_guidance(query)
        exact = None
        registry = self._ctx.skill_registry
        if registry and registry.skill_metadata:
            for meta in registry.skill_metadata.values():
                if query.strip().lower() in {
                    meta.slug.lower(),
                    meta.name.lower(),
                }:
                    exact = meta.slug
                    break
        if exact:
            return self._ctx._load_skill_guidance(exact)
        return self._tool_search_skills(query)

    def _tool_list_mcp_resources(self, server: str = "") -> str:
        manifest_path = os.environ.get(
            "OV_AGENT_MCP_MANIFEST", ""
        ).strip()
        if not manifest_path:
            return json.dumps(
                {
                    "available": False,
                    "reason": "OV_AGENT_MCP_MANIFEST is not configured.",
                    "resources": [],
                },
                ensure_ascii=False,
                indent=2,
            )
        manifest = json.loads(
            Path(manifest_path).read_text(encoding="utf-8")
        )
        resources = manifest.get("resources", [])
        if server:
            resources = [
                item
                for item in resources
                if item.get("server") == server
            ]
        return json.dumps(
            {"available": True, "resources": resources},
            ensure_ascii=False,
            indent=2,
        )

    def _tool_read_mcp_resource(self, server: str, uri: str) -> str:
        manifest_path = os.environ.get(
            "OV_AGENT_MCP_MANIFEST", ""
        ).strip()
        if not manifest_path:
            return json.dumps(
                {
                    "available": False,
                    "reason": "OV_AGENT_MCP_MANIFEST is not configured.",
                },
                ensure_ascii=False,
                indent=2,
            )
        manifest = json.loads(
            Path(manifest_path).read_text(encoding="utf-8")
        )
        for item in manifest.get("resources", []):
            if item.get("server") == server and item.get("uri") == uri:
                target = item.get("path", "")
                if not target:
                    return json.dumps(
                        item, ensure_ascii=False, indent=2
                    )
                return self._tool_read(target)
        return json.dumps(
            {"error": "MCP resource not found"},
            ensure_ascii=False,
            indent=2,
        )

    def _tool_ask_user_question(
        self,
        question: str,
        header: str = "",
        options: Optional[list[str]] = None,
    ) -> str:
        from ..harness.contracts import make_turn_id  # noqa: F811

        session_id = self._ctx._get_runtime_session_id()
        trace = getattr(self._ctx, "_last_run_trace", None)
        record = runtime_state.create_question(
            session_id,
            turn_id=getattr(trace, "turn_id", ""),
            trace_id=getattr(trace, "trace_id", ""),
            question=question,
            header=header,
            options=list(options or []),
        )
        answer = self._ctx._request_interaction(
            {
                "kind": "question",
                "question_id": record.question_id,
                "question": question,
                "header": header,
                "options": list(options or []),
                "session_id": session_id,
                "trace_id": record.trace_id,
                "turn_id": record.turn_id,
            }
        )
        if isinstance(answer, dict):
            resolved = runtime_state.resolve_question(
                session_id,
                record.question_id,
                str(answer.get("answer", "")),
            )
        else:
            resolved = runtime_state.resolve_question(
                session_id,
                record.question_id,
                str(answer or ""),
            )
        payload = (
            resolved.to_dict()
            if resolved is not None
            else record.to_dict()
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Bash
    # ------------------------------------------------------------------

    def _background_bash_worker(
        self,
        task_id: str,
        *,
        command: str,
        cwd: str,
        timeout_ms: int,
    ) -> None:
        session_id = self._ctx._get_runtime_session_id()
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        runtime_state.bind_process(session_id, task_id, proc)
        assert proc.stdout is not None
        start = time.time()
        for line in proc.stdout:
            runtime_state.append_task_output(
                session_id, task_id, line.rstrip("\n")
            )
            if time.time() - start > timeout_ms / 1000:
                proc.terminate()
                runtime_state.update_task(
                    session_id,
                    task_id,
                    status="failed",
                    summary="Background command timed out",
                )
                return
        return_code = proc.wait()
        status = "completed" if return_code == 0 else "failed"
        runtime_state.update_task(
            session_id,
            task_id,
            status=status,
            summary=f"Exit code {return_code}",
        )

    def _tool_bash(
        self,
        command: str,
        description: str = "",
        timeout: int = 120000,
        run_in_background: bool = False,
        dangerouslyDisableSandbox: bool = False,
    ) -> str:
        self._ctx._ensure_server_tool_mode("Bash")
        if self.tool_blocked_in_plan_mode("Bash"):
            return "Bash is blocked while the session is in plan mode."
        reason = description or f"Run shell command: {command[:120]}"
        self._ctx._request_tool_approval(
            "Bash",
            reason=reason,
            payload={
                "command": command,
                "dangerouslyDisableSandbox": dangerouslyDisableSandbox,
            },
        )
        cwd = self._ctx._refresh_runtime_working_directory()
        if run_in_background:
            task = runtime_state.create_task(
                self._ctx._get_runtime_session_id(),
                title=description or command[:80],
                description=command,
                kind="background_command",
                status="in_progress",
                background=True,
                tool_name="Bash",
                metadata={"cwd": cwd, "command": command},
            )
            proc = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            runtime_state.attach_background_process(
                self._ctx._get_runtime_session_id(),
                task.task_id,
                proc,
            )
            return json.dumps(
                {
                    "task_id": task.task_id,
                    "status": "in_progress",
                    "cwd": cwd,
                    "command": command,
                },
                ensure_ascii=False,
                indent=2,
            )

        proc_result = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout)) / 1000,
            check=False,
        )
        output = (proc_result.stdout or "") + (
            (proc_result.stderr or "") if proc_result.stderr else ""
        )
        return json.dumps(
            {
                "cwd": cwd,
                "command": command,
                "returncode": proc_result.returncode,
                "stdout": proc_result.stdout,
                "stderr": proc_result.stderr,
                "output": output.strip(),
            },
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Terminal tools
    # ------------------------------------------------------------------

    def _tool_finish(self, summary: str) -> dict:
        """Return the terminal finish payload."""
        return {"finished": True, "summary": summary}
