"""AgentContext protocol — type-only contract for extracted ovagent modules.

All extracted modules (PromptBuilder, AnalysisExecutor, ToolRuntime,
SubagentController, TurnController, CodegenPipeline) depend on this
protocol instead of importing the concrete ``OmicVerseAgent`` class.
This breaks the circular import between ``smart_agent.py`` and the
``ovagent/`` subpackage.

The protocol is ``@runtime_checkable`` so extracted modules and tests can
validate duck-typed agent doubles at runtime when needed.

Isolation boundary
------------------
``AgentContext`` describes the *parent* agent surface.  Subagents do NOT
receive a full ``AgentContext``; instead they operate through a
``SubagentRuntime`` (see ``subagent_controller.py``) that exposes only
the scoped state a subagent is allowed to read or mutate:

* ``SubagentRuntime.permission_policy`` — per-tool allow/ask/deny decisions.
* ``SubagentRuntime.budget_manager``   — subagent-local token budget.
* ``SubagentRuntime.tool_schemas``     — snapshotted (frozen) tool schemas.
* ``SubagentRuntime.last_usage``       — subagent-local usage tracking.

This separation ensures that subagent turns cannot implicitly mutate
parent-agent state such as ``last_usage``, ``last_usage_breakdown``, or
the parent's live tool registry.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from ..agent_backend import OmicVerseLLMBackend
    from ..agent_config import AgentConfig
    from ..agent_reporter import EventLevel, Reporter
    from ..agent_sandbox import CodeSecurityScanner, SecurityConfig
    from ..context_compactor import ContextCompactor
    from ..filesystem_context import FilesystemContextManager
    from ..harness import RunTraceStore
    from ..session_history import SessionHistory
    from ..skill_registry import SkillRegistry
    from .runtime import OmicVerseRuntime


@runtime_checkable
class AgentContext(Protocol):
    """Minimal surface that extracted ovagent modules access on the agent."""

    # ---- core identifiers ----
    model: str
    provider: str
    endpoint: str
    api_key: Optional[str]

    # ---- subsystems ----
    _llm: Optional["OmicVerseLLMBackend"]
    _config: "AgentConfig"
    _security_config: "SecurityConfig"
    _security_scanner: "CodeSecurityScanner"
    _filesystem_context: Optional["FilesystemContextManager"]
    skill_registry: Optional["SkillRegistry"]
    _notebook_executor: Any
    _ov_runtime: Optional["OmicVerseRuntime"]
    _trace_store: Optional["RunTraceStore"]
    _session_history: Optional["SessionHistory"]
    _context_compactor: Optional["ContextCompactor"]
    _approval_handler: Any
    _reporter: "Reporter"

    # ---- runtime state ----
    last_usage: Any
    last_usage_breakdown: Dict[str, Any]
    _last_run_trace: Any
    _active_run_id: str
    _web_session_id: str
    _managed_api_env: Dict[str, str]

    # ---- code-only mode state (used by CodegenPipeline and ToolRuntime) ----
    _code_only_mode: bool
    _code_only_captured_code: str
    _code_only_captured_history: List[Dict[str, Any]]

    # ---- feature flags ----
    use_notebook_execution: bool
    enable_filesystem_context: bool

    # ---- class-level constants ----
    LEGACY_AGENT_TOOLS: List[Dict[str, Any]]

    # ---- methods ----
    def _emit(self, level: "EventLevel", message: str, category: str = "") -> None: ...

    def _get_harness_session_id(self) -> str: ...

    def _get_runtime_session_id(self) -> str: ...

    def _get_visible_agent_tools(
        self, *, allowed_names: Optional[set[str]] = None
    ) -> list[dict[str, Any]]: ...

    def _get_loaded_tool_names(self) -> list[str]: ...

    def _refresh_runtime_working_directory(self) -> str: ...

    @contextmanager
    def _temporary_api_keys(self): ...  # type: ignore[override]

    def _tool_blocked_in_plan_mode(self, tool_name: str) -> bool: ...

    def _detect_repo_root(self, cwd: Optional[Path] = None) -> Optional[Path]: ...

    def _resolve_local_path(
        self, file_path: str, *, allow_relative: bool = False
    ) -> Path: ...

    def _ensure_server_tool_mode(self, tool_name: str) -> None: ...

    def _request_interaction(self, payload: dict[str, Any]) -> Any: ...

    def _request_tool_approval(
        self, tool_name: str, *, reason: str, payload: dict[str, Any]
    ) -> None: ...

    def _load_skill_guidance(self, slug: str) -> str: ...

    def _extract_python_code(self, text: str) -> Optional[str]: ...

    def _normalize_registry_entry_for_codegen(self, entry: Dict[str, Any]) -> Dict[str, Any]: ...

    def _build_agentic_system_prompt(self) -> str: ...

    async def _run_agentic_loop(
        self, request: str, adata: Any,
        event_callback: Any = None,
        cancel_event: Any = None,
        history: Any = None,
        approval_handler: Any = None,
        request_content: Any = None,
    ) -> Any: ...
