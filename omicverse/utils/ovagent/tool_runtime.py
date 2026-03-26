"""ToolRuntime — tool dispatch hub and runtime facade.

Extracted from ``smart_agent.py`` to keep tool handling in one place.
Concrete handler implementations for IO, web, and workspace tool families
live in dedicated handler modules:

- ``tool_runtime_io``        — read / edit / write / glob / grep / notebook
- ``tool_runtime_web``       — web_fetch / web_search / web_download
- ``tool_runtime_workspace`` — tasks / plan mode / worktree / skill / MCP

Execution tools (execute_code, run_snippet, bash, agent, inspect_data,
search_functions, tool_search, finish) remain in this facade until the
next extraction pass.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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

from . import tool_runtime_io, tool_runtime_web, tool_runtime_workspace

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

    Concrete handler implementations for IO, web, and workspace tool
    families are delegated to ``tool_runtime_io``, ``tool_runtime_web``,
    and ``tool_runtime_workspace`` respectively.

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

        IO, web, and workspace handlers delegate to their extracted
        modules; execution and core tools are handled inline.
        """
        r = self._registry
        # fmt: off

        # -- Core / execution tools (remain in facade) --
        r.register_handler("tool_search", lambda a, d, _: self._tool_tool_search(
            a.get("query", ""), max_results=a.get("max_results", 5)))
        r.register_handler("bash", lambda a, d, _: self._tool_bash(
            a.get("command", ""), description=a.get("description", ""),
            timeout=a.get("timeout", 120000),
            run_in_background=bool(a.get("run_in_background", False)),
            dangerouslyDisableSandbox=bool(a.get("dangerouslyDisableSandbox", False))))
        r.register_handler("inspect_data", lambda a, d, _: self._tool_inspect_data(
            d, a.get("aspect", "full")))
        r.register_handler("execute_code", self._dispatch_execute_code)
        r.register_handler("run_snippet", self._dispatch_run_snippet)
        r.register_handler("search_functions", lambda a, d, _: self._tool_search_functions(
            a.get("query", "")))
        r.register_handler("agent", self._dispatch_agent)
        r.register_handler("ask_user_question", self._dispatch_ask_user_question)
        r.register_handler("finish", lambda a, d, _: self._tool_finish(
            a.get("summary", "")))

        # -- IO tools (delegated to tool_runtime_io) --
        r.register_handler("read", lambda a, d, _: tool_runtime_io.handle_read(
            self._ctx, a.get("file_path", ""), offset=a.get("offset", 0),
            limit=a.get("limit", 2000), pages=a.get("pages", "")))
        r.register_handler("edit", lambda a, d, _: tool_runtime_io.handle_edit(
            self._ctx, self.tool_blocked_in_plan_mode, a.get("file_path", ""),
            a.get("old_string", ""), a.get("new_string", ""),
            replace_all=bool(a.get("replace_all", False))))
        r.register_handler("write", lambda a, d, _: tool_runtime_io.handle_write(
            self._ctx, self.tool_blocked_in_plan_mode, a.get("file_path", ""),
            a.get("content", "")))
        r.register_handler("glob", lambda a, d, _: tool_runtime_io.handle_glob(
            self._ctx, a.get("pattern", ""), root=a.get("root", ""),
            max_results=a.get("max_results", 200)))
        r.register_handler("grep", lambda a, d, _: tool_runtime_io.handle_grep(
            self._ctx, a.get("pattern", ""), root=a.get("root", ""),
            glob=a.get("glob", ""), max_results=a.get("max_results", 200)))
        r.register_handler("notebook_edit", lambda a, d, _: tool_runtime_io.handle_notebook_edit(
            self._ctx, self.tool_blocked_in_plan_mode, a.get("file_path", ""),
            cell_index=a.get("cell_index", 0), source=a.get("source", ""),
            cell_type=a.get("cell_type", "")))

        # -- Web tools (delegated to tool_runtime_web) --
        r.register_handler("web_fetch", lambda a, d, _: tool_runtime_web.handle_web_fetch(
            a.get("url", ""), prompt=a.get("prompt")))
        r.register_handler("web_search", lambda a, d, _: tool_runtime_web.handle_web_search(
            a.get("query", ""), num_results=a.get("num_results", 5)))
        r.register_handler("web_download", lambda a, d, _: tool_runtime_web.handle_web_download(
            a.get("url", ""), filename=a.get("filename"),
            directory=a.get("directory")))

        # -- Workspace tools (delegated to tool_runtime_workspace) --
        r.register_handler("task_create", lambda a, d, _: tool_runtime_workspace.handle_create_task(
            self._ctx, a.get("title", ""), description=a.get("description", ""),
            status=a.get("status", "pending")))
        r.register_handler("task_get", lambda a, d, _: tool_runtime_workspace.handle_get_task(
            self._ctx, a.get("task_id", "")))
        r.register_handler("task_list", lambda a, d, _: tool_runtime_workspace.handle_list_tasks(
            self._ctx, status=a.get("status", "")))
        r.register_handler("task_output", lambda a, d, _: tool_runtime_workspace.handle_task_output(
            self._ctx, a.get("task_id", ""), offset=a.get("offset", 0),
            limit=a.get("limit", 200)))
        r.register_handler("task_stop", lambda a, d, _: tool_runtime_workspace.handle_task_stop(
            self._ctx, a.get("task_id", "")))
        r.register_handler("task_update", lambda a, d, _: tool_runtime_workspace.handle_task_update(
            self._ctx, a.get("task_id", ""), a.get("status", ""),
            summary=a.get("summary", "")))
        r.register_handler("enter_plan_mode", lambda a, d, _: tool_runtime_workspace.handle_enter_plan_mode(
            self._ctx, reason=a.get("reason", "")))
        r.register_handler("exit_plan_mode", lambda a, d, _: tool_runtime_workspace.handle_exit_plan_mode(
            self._ctx, summary=a.get("summary", "")))
        r.register_handler("enter_worktree", lambda a, d, _: tool_runtime_workspace.handle_enter_worktree(
            self._ctx, self.tool_blocked_in_plan_mode,
            branch_name=a.get("branch_name", ""),
            path=a.get("path", ""), base_ref=a.get("base_ref", "HEAD")))
        r.register_handler("skill", lambda a, d, _: tool_runtime_workspace.handle_skill(
            self._ctx, a.get("query", ""), mode=a.get("mode", "search")))
        r.register_handler("list_mcp_resources", lambda a, d, _: tool_runtime_workspace.handle_list_mcp_resources(
            server=a.get("server", "")))
        r.register_handler("read_mcp_resource", lambda a, d, _: tool_runtime_workspace.handle_read_mcp_resource(
            a.get("server", ""), a.get("uri", ""),
            read_fn=lambda path: tool_runtime_io.handle_read(self._ctx, path)))
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
        return tool_runtime_workspace.handle_ask_user_question(
            self._ctx,
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
    # OmicVerse data tools (execution family — remain in facade)
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
        extract_code_fn = getattr(self._ctx, "_extract_python_code", None)

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
                    repair_loop.run(code, adata, phase="execution",
                                   extract_code_fn=extract_code_fn),
                ).result()
        else:
            repair_result = _asyncio.run(
                repair_loop.run(code, adata, phase="execution",
                               extract_code_fn=extract_code_fn)
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

    # ------------------------------------------------------------------
    # Claude-style tool helpers (execution family — remain in facade)
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

    # ------------------------------------------------------------------
    # Bash (execution family — remains in facade)
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
