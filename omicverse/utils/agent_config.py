"""
Grouped configuration dataclasses for OmicVerse Agent (P0-2).

Replaces the 15-parameter ``OmicVerseAgent.__init__`` with four focused
config objects.  ``AgentConfig.from_flat_kwargs`` preserves full backward
compatibility with the original constructor signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional

from .agent_sandbox import ApprovalMode, SecurityConfig, SecurityLevel


class SandboxFallbackPolicy(Enum):
    """What to do when notebook execution fails (P2-3)."""
    RAISE = "raise"                # Do not fall back; raise SandboxDeniedError
    WARN_AND_FALLBACK = "warn"     # Emit warning event, then fall back to in-process
    SILENT = "silent"              # Fall back silently (legacy behaviour)


@dataclass
class LLMConfig:
    """LLM connection settings."""
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    auth_mode: str = "environment"
    auth_provider: Optional[str] = None
    auth_file: Optional[Path] = None
    reasoning_effort: str = "high"  # "low" | "medium" | "high" for GPT-5


@dataclass
class ReflectionConfig:
    """Code reflection & result review settings."""
    enabled: bool = True
    iterations: int = 1           # clamped to 1-3 in __post_init__
    result_review: bool = True

    def __post_init__(self) -> None:
        self.iterations = max(1, min(3, self.iterations))


@dataclass
class ExecutionConfig:
    """Code execution environment settings."""
    use_notebook: bool = True
    max_prompts_per_session: int = 5
    storage_dir: Optional[Path] = None
    keep_notebooks: bool = True
    timeout: int = 600
    strict_kernel_validation: bool = True
    sandbox_fallback_policy: SandboxFallbackPolicy = SandboxFallbackPolicy.WARN_AND_FALLBACK
    # Self-repair settings
    auto_install_packages: bool = True
    package_blocklist: List[str] = field(default_factory=lambda: [
        "os", "sys", "subprocess", "shutil", "signal", "ctypes",
    ])
    max_execution_retries: int = 2
    validate_outputs: bool = True
    # Agentic loop settings
    max_agent_turns: int = 15      # max iterations in agentic loop


@dataclass
class SubagentConfig:
    """Configuration for a subagent type (explore / plan / execute)."""
    agent_type: str
    allowed_tools: List[str]
    max_turns: int
    can_mutate_adata: bool
    temperature: float = 0.2


SUBAGENT_CONFIGS = {
    "explore": SubagentConfig(
        agent_type="explore",
        allowed_tools=["inspect_data", "run_snippet", "search_functions",
                       "web_fetch", "web_search", "finish"],
        max_turns=5,
        can_mutate_adata=False,
        temperature=0.1,
    ),
    "plan": SubagentConfig(
        agent_type="plan",
        allowed_tools=[
            "inspect_data", "run_snippet", "search_functions",
            "search_skills", "web_fetch", "web_search", "finish",
        ],
        max_turns=8,
        can_mutate_adata=False,
        temperature=0.3,
    ),
    "execute": SubagentConfig(
        agent_type="execute",
        allowed_tools=[
            "inspect_data", "execute_code", "run_snippet",
            "search_functions", "web_fetch", "web_search",
            "web_download", "finish",
        ],
        max_turns=10,
        can_mutate_adata=True,
        temperature=0.1,
    ),
}


@dataclass
class ContextConfig:
    """Filesystem context management settings."""
    enabled: bool = True
    storage_dir: Optional[Path] = None


@dataclass
class HarnessConfig:
    """Harness tracing and replay settings."""
    enable_traces: bool = True
    trace_dir: Optional[Path] = None
    record_artifacts: bool = True
    server_only_validation: bool = True
    server_tool_mode: bool = True
    enable_claude_tool_catalog: bool = True
    deferred_tool_loading: bool = True
    cleanup_reports_dir: Optional[Path] = None
    include_recent_failures_in_prompt: bool = True
    max_recent_failures: int = 3
    enable_context_compaction: bool = True
    enable_mcp_registry: bool = True


@dataclass
class AgentConfig:
    """Aggregated agent configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    verbose: bool = True
    history_enabled: bool = False
    history_path: Optional[Path] = None

    # --------------- backward-compat factory ---------------

    @classmethod
    def from_flat_kwargs(cls, **kw: Any) -> "AgentConfig":
        """Build from the original OmicVerseAgent.__init__ keyword args."""
        sd = kw.get("notebook_storage_dir")
        cd = kw.get("context_storage_dir")
        td = kw.get("harness_trace_dir")
        rd = kw.get("harness_cleanup_reports_dir")

        # Security config from flat kwargs
        approval_raw = kw.get("approval_mode", "never")
        if isinstance(approval_raw, str):
            approval_raw = ApprovalMode(approval_raw)

        return cls(
            llm=LLMConfig(
                model=kw.get("model", "gemini-2.5-flash"),
                api_key=kw.get("api_key"),
                endpoint=kw.get("endpoint"),
                auth_mode=kw.get("auth_mode", "environment"),
                auth_provider=kw.get("auth_provider"),
                auth_file=Path(kw["auth_file"]).expanduser() if kw.get("auth_file") else None,
            ),
            reflection=ReflectionConfig(
                enabled=kw.get("enable_reflection", True),
                iterations=kw.get("reflection_iterations", 1),
                result_review=kw.get("enable_result_review", True),
            ),
            execution=ExecutionConfig(
                use_notebook=kw.get("use_notebook_execution", True),
                max_prompts_per_session=kw.get("max_prompts_per_session", 5),
                storage_dir=Path(sd) if sd else None,
                keep_notebooks=kw.get("keep_execution_notebooks", True),
                timeout=kw.get("notebook_timeout", 600),
                strict_kernel_validation=kw.get("strict_kernel_validation", True),
                auto_install_packages=kw.get("auto_install_packages", True),
                max_execution_retries=kw.get("max_execution_retries", 2),
                validate_outputs=kw.get("validate_outputs", True),
                max_agent_turns=kw.get("max_agent_turns", 15),
            ),
            context=ContextConfig(
                enabled=kw.get("enable_filesystem_context", True),
                storage_dir=Path(cd) if cd else None,
            ),
            harness=HarnessConfig(
                enable_traces=kw.get("enable_harness_traces", True),
                trace_dir=Path(td) if td else None,
                record_artifacts=kw.get("record_harness_artifacts", True),
                server_only_validation=kw.get("server_only_validation", True),
                server_tool_mode=kw.get("server_tool_mode", True),
                enable_claude_tool_catalog=kw.get("enable_claude_tool_catalog", True),
                deferred_tool_loading=kw.get("deferred_tool_loading", True),
                cleanup_reports_dir=Path(rd) if rd else None,
                include_recent_failures_in_prompt=kw.get("include_recent_failures_in_prompt", True),
                max_recent_failures=kw.get("max_recent_failures", 3),
                enable_context_compaction=kw.get("enable_context_compaction", True),
                enable_mcp_registry=kw.get("enable_mcp_registry", True),
            ),
            security=cls._build_security_config(kw, approval_raw),
        )

    @staticmethod
    def _build_security_config(kw: dict, approval_mode: ApprovalMode) -> SecurityConfig:
        """Build SecurityConfig, preferring security_level preset if given."""
        level_raw = kw.get("security_level")
        if level_raw:
            config = SecurityConfig.from_level(level_raw)
            # Allow overriding approval_mode even with a preset
            if "approval_mode" in kw:
                config.approval_mode = approval_mode
            return config
        return SecurityConfig(
            approval_mode=approval_mode,
            allow_dynamic_imports=kw.get("allow_dynamic_imports", False),
            restrict_introspection=kw.get("restrict_introspection", True),
        )
