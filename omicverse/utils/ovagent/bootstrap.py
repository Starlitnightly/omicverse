"""Agent subsystem initialization.

Extracted from ``OmicVerseAgent.__init__`` so that each bootstrap concern
has its own focused, testable function.  Every function here is a pure
factory — it creates and returns a subsystem value without storing it on
an agent instance.

All ``initialize_*`` functions accept an optional *event_bus* parameter.
When provided, status messages are emitted as structured events through
the bus instead of going to bare ``print()``.  When omitted a default
``EventBus(PrintReporter())`` is used so that the existing console output
is preserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .event_stream import EventBus, make_event_bus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bus(event_bus: Optional[EventBus]) -> EventBus:
    """Return *event_bus* or a default ``PrintReporter``-backed bus."""
    if event_bus is not None:
        return event_bus
    return make_event_bus()


def _is_under_root(path: Path, root: Path) -> bool:
    """Check whether *path* is a descendant of *root*."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def format_skill_overview(skill_registry: Any) -> str:
    """Generate a bullet overview of available project skills."""
    if not skill_registry or not getattr(skill_registry, "skill_metadata", None):
        return ""
    lines = [
        f"- **{skill.name}** — {skill.description}"
        for skill in sorted(
            skill_registry.skill_metadata.values(),
            key=lambda item: item.name.lower(),
        )
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Skill registry
# ---------------------------------------------------------------------------

def initialize_skill_registry(
    package_root: Optional[Path] = None,
    event_bus: Optional[EventBus] = None,
) -> Tuple[Any, str]:
    """Load skills from package install and CWD.

    Returns ``(registry_or_None, overview_text)``.
    """
    from ..skill_registry import (
        build_multi_path_skill_registry,
        discover_multi_path_skill_roots,
    )

    eb = _bus(event_bus)

    if package_root is None:
        # three levels up: ovagent → utils → omicverse → project root
        package_root = Path(__file__).resolve().parents[3]
    cwd = Path.cwd()

    try:
        registry = build_multi_path_skill_registry(package_root, cwd)
    except Exception as exc:
        eb.init_warning(f"⚠️  Failed to load Agent Skills: {exc}")
        return None, ""

    if not registry or not registry.skill_metadata:
        return None, ""

    overview_text = format_skill_overview(registry)

    # Discovery summary
    roots = discover_multi_path_skill_roots(package_root, cwd)
    counts: Dict[str, int] = {}
    for label, root in roots:
        counts[label] = sum(
            1
            for metadata in registry.skill_metadata.values()
            if _is_under_root(Path(metadata.path), root)
        )
    total = len(registry.skill_metadata)
    msg = f"   🧭 Loaded {total} skills (progressive disclosure)"
    summary_parts: List[str] = []
    if counts.get("Bundled"):
        summary_parts.append(f"{counts['Bundled']} bundled")
    if counts.get("Legacy Built-in"):
        summary_parts.append(f"{counts['Legacy Built-in']} legacy built-in")
    if counts.get("Workspace"):
        summary_parts.append(f"{counts['Workspace']} user-created")
    if summary_parts:
        msg += f" ({' + '.join(summary_parts)})"
    eb.init(msg, total=total)

    return registry, overview_text


# ---------------------------------------------------------------------------
# Notebook executor
# ---------------------------------------------------------------------------

def initialize_notebook_executor(
    *,
    use_notebook: bool,
    storage_dir: Optional[Path],
    max_prompts_per_session: int,
    keep_notebooks: bool,
    timeout: int,
    strict_kernel_validation: bool,
    event_bus: Optional[EventBus] = None,
) -> Tuple[bool, Any]:
    """Initialize notebook executor.

    Returns ``(use_notebook_flag, executor_or_None)``.  When initialization
    fails, ``use_notebook_flag`` is ``False`` regardless of the input.
    """
    eb = _bus(event_bus)

    if not use_notebook:
        eb.init("   ⚡ Using in-process execution (no session isolation)")
        return False, None

    try:
        from ..session_notebook_executor import SessionNotebookExecutor

        executor = SessionNotebookExecutor(
            max_prompts_per_session=max_prompts_per_session,
            storage_dir=storage_dir,
            keep_notebooks=keep_notebooks,
            timeout=timeout,
            strict_kernel_validation=strict_kernel_validation,
        )
        eb.init("   📓 Session-based notebook execution enabled")
        eb.init(f"      Conda environment: {executor.conda_env or 'default'}")
        eb.init(f"      Session limit: {max_prompts_per_session} prompts")
        eb.init(f"      Storage: {executor.storage_dir}")
        return True, executor
    except Exception as e:
        eb.init_warning(f"   ⚠️  Notebook execution initialization failed: {e}")
        eb.init("   ⚡ Falling back to in-process execution")
        return False, None


# ---------------------------------------------------------------------------
# Filesystem context
# ---------------------------------------------------------------------------

def initialize_filesystem_context(
    *,
    enabled: bool,
    storage_dir: Optional[Path],
    event_bus: Optional[EventBus] = None,
) -> Tuple[bool, Any]:
    """Initialize filesystem context manager.

    Returns ``(enabled_flag, context_or_None)``.
    """
    eb = _bus(event_bus)

    if not enabled:
        eb.init("   ⚡ Filesystem context disabled")
        return False, None

    try:
        from ..filesystem_context import FilesystemContextManager

        ctx = FilesystemContextManager(base_dir=storage_dir)
        eb.init("   📁 Filesystem context enabled")
        eb.init(f"      Session: {ctx.session_id}")
        eb.init(f"      Storage: {ctx._workspace_dir}")
        return True, ctx
    except Exception as e:
        logger.warning("Filesystem context initialization failed: %s", e)
        eb.init_warning(f"   ⚠️  Filesystem context disabled (init failed: {e})")
        return False, None


# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

def initialize_session_history(
    config: Any,
    event_bus: Optional[EventBus] = None,
) -> Any:
    """Initialize session history if enabled in *config*.

    Returns ``SessionHistory`` or ``None``.
    """
    if not getattr(config, "history_enabled", False):
        return None

    from ..session_history import SessionHistory

    hist = SessionHistory(path=config.history_path)
    _bus(event_bus).init("   📝 Session history enabled")
    return hist


# ---------------------------------------------------------------------------
# Harness tracing & context compaction
# ---------------------------------------------------------------------------

def initialize_tracing(
    config: Any,
    llm: Any,
    model: str,
    event_bus: Optional[EventBus] = None,
) -> Tuple[Any, Any]:
    """Initialize trace store and context compactor.

    Returns ``(trace_store_or_None, compactor_or_None)``.
    """
    from ..harness import RunTraceStore
    from ..context_compactor import ContextCompactor

    eb = _bus(event_bus)
    trace_store = None
    compactor = None

    harness_config = getattr(config, "harness", None)
    if harness_config is not None and getattr(harness_config, "enable_traces", False):
        trace_store = RunTraceStore(root=harness_config.trace_dir)
        eb.init("   🧰 Harness tracing enabled")
    if (
        harness_config is not None
        and getattr(harness_config, "enable_context_compaction", False)
        and llm
    ):
        compactor = ContextCompactor(llm, model)
        eb.init("   🗜️  Harness context compaction enabled")

    return trace_store, compactor


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def initialize_security(
    config: Any,
    event_bus: Optional[EventBus] = None,
) -> Tuple[Any, Any]:
    """Initialize security scanner from *config*.

    Returns ``(security_config, scanner)``.
    """
    from ..agent_sandbox import CodeSecurityScanner, SecurityConfig

    security_config = getattr(config, "security", SecurityConfig())
    scanner = CodeSecurityScanner(security_config)
    _bus(event_bus).init(
        f"   🛡️  Security scanner enabled (approval: {security_config.approval_mode.value})"
    )
    return security_config, scanner


# ---------------------------------------------------------------------------
# OmicVerse runtime (workflow / run-store bridge)
# ---------------------------------------------------------------------------

def initialize_ov_runtime(
    repo_root: Optional[Path],
    event_bus: Optional[EventBus] = None,
) -> Any:
    """Initialize ``OmicVerseRuntime``.

    Returns the runtime instance or ``None`` on failure.
    """
    from .runtime import OmicVerseRuntime

    try:
        rt = OmicVerseRuntime(repo_root=repo_root)
        if rt.workflow.exists:
            _bus(event_bus).init(f"   📋 Workflow policy loaded: {rt.workflow.path}")
        return rt
    except Exception as exc:
        logger.warning("OVAgent runtime initialization failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

def create_llm_backend(
    *,
    system_prompt: str,
    model: str,
    api_key: Optional[str],
    endpoint: str,
    max_tokens: int = 8192,
    temperature: float = 0.2,
) -> Any:
    """Create the internal ``OmicVerseLLMBackend``."""
    from ..agent_backend import OmicVerseLLMBackend

    return OmicVerseLLMBackend(
        system_prompt=system_prompt,
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Display helpers (reflection / review banners)
# ---------------------------------------------------------------------------

def display_reflection_config(
    enable_reflection: bool,
    reflection_iterations: int,
    enable_result_review: bool,
    event_bus: Optional[EventBus] = None,
) -> None:
    """Emit reflection and result-review configuration."""
    eb = _bus(event_bus)

    if enable_reflection:
        suffix = "s" if reflection_iterations > 1 else ""
        eb.init(
            f"   🔍 Reflection enabled: {reflection_iterations} iteration{suffix} "
            "(code review & validation)"
        )
    else:
        eb.init("   ⚡ Reflection disabled (faster execution, no code validation)")

    if enable_result_review:
        eb.init("   ✅ Result review enabled (output validation & assessment)")
    else:
        eb.init("   ⚡ Result review disabled (no output validation)")
