"""ToolRuntime — tool dispatch hub and runtime facade.

Extracted from ``smart_agent.py`` to keep tool handling in one place.
Concrete handler implementations live in dedicated handler modules:

- ``tool_runtime_exec``      — execute_code / run_snippet / bash / agent / inspect / search / finish
- ``tool_runtime_io``        — read / edit / write / glob / grep / notebook
- ``tool_runtime_web``       — web_fetch / web_search / web_download
- ``tool_runtime_workspace`` — tasks / plan mode / worktree / skill / MCP

ToolRuntime itself owns runtime state, registry binding, approval
integration, and dispatch routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from ..harness.runtime_state import runtime_state
from ..harness.tool_catalog import (
    get_tool_spec,
    get_visible_tool_schemas,
    normalize_tool_name,
    resolve_tool_search,
)
from ..._registry import _global_registry

from . import tool_runtime_exec, tool_runtime_io, tool_runtime_web, tool_runtime_workspace

if TYPE_CHECKING:
    from .analysis_executor import AnalysisExecutor
    from .permission_policy import PermissionPolicy
    from .protocol import AgentContext
    from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


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
        "description": "Search the OmicVerse function registry by keyword. Returns signatures, parameters, prerequisites, and usage examples. Always call this before generating code to get exact API details.",
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

    Concrete handler implementations are delegated to four handler modules:

    - ``tool_runtime_exec``      — execution, agent, bash, data inspection
    - ``tool_runtime_io``        — filesystem read/edit/write, glob, grep
    - ``tool_runtime_web``       — web fetch/search/download
    - ``tool_runtime_workspace`` — tasks, plan mode, worktree, skill, MCP

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

        All concrete tool implementations live in extracted handler modules;
        this method wires them into the registry with argument unpacking.
        """
        r = self._registry
        # fmt: off

        # -- Execution / core tools (delegated to tool_runtime_exec) --
        r.register_handler("tool_search", lambda a, d, _: tool_runtime_exec.handle_tool_search(
            self._ctx, a.get("query", ""), max_results=a.get("max_results", 5)))
        r.register_handler("bash", lambda a, d, _: tool_runtime_exec.handle_bash(
            self._ctx, self.tool_blocked_in_plan_mode,
            a.get("command", ""), description=a.get("description", ""),
            timeout=a.get("timeout", 120000),
            run_in_background=bool(a.get("run_in_background", False)),
            dangerouslyDisableSandbox=bool(a.get("dangerouslyDisableSandbox", False))))
        r.register_handler("inspect_data", lambda a, d, _: tool_runtime_exec.handle_inspect_data(
            d, a.get("aspect", "full")))
        r.register_handler("execute_code", self._dispatch_execute_code)
        r.register_handler("run_snippet", self._dispatch_run_snippet)
        r.register_handler("search_functions", lambda a, d, _: tool_runtime_exec.handle_search_functions(
            self._ctx, a.get("query", "")))
        r.register_handler("agent", self._dispatch_agent)
        r.register_handler("ask_user_question", self._dispatch_ask_user_question)
        r.register_handler("finish", lambda a, d, _: tool_runtime_exec.handle_finish(
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
    # Dispatch helpers (thin validation wrappers)
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
        return tool_runtime_exec.handle_execute_code(
            self._ctx, self._executor, code, args.get("description", ""), adata
        )

    def _dispatch_run_snippet(
        self, args: dict, adata: Any, request: str
    ) -> Any:
        snippet = args.get("code", "")
        if not snippet or not snippet.strip():
            return json.dumps(
                {"error": "run_snippet requires a non-empty 'code' argument."},
                ensure_ascii=False,
            )
        return tool_runtime_exec.handle_run_snippet(
            self._executor, snippet, adata
        )

    async def _dispatch_agent(
        self, args: dict, adata: Any, request: str
    ) -> Any:
        agent_type = args.get(
            "subagent_type", args.get("agent_type", "explore")
        )
        task = args.get("task", "")
        context = args.get("context", "")
        subagent_controller = self._require_subagent_controller()
        return await tool_runtime_exec.handle_agent(
            subagent_controller, agent_type, task, adata, context
        )

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
