"""Agent subsystem initialization.

Extracted from ``OmicVerseAgent.__init__`` so that each bootstrap concern
has its own focused, testable function.  Every function here is a pure
factory — it creates and returns a subsystem value without storing it on
an agent instance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
) -> Tuple[Any, str]:
    """Load skills from package install and CWD.

    Returns ``(registry_or_None, overview_text)``.
    """
    from ..skill_registry import (
        build_multi_path_skill_registry,
        discover_multi_path_skill_roots,
    )

    if package_root is None:
        # three levels up: ovagent → utils → omicverse → project root
        package_root = Path(__file__).resolve().parents[3]
    cwd = Path.cwd()

    try:
        registry = build_multi_path_skill_registry(package_root, cwd)
    except Exception as exc:
        print(f"⚠️  Failed to load Agent Skills: {exc}")
        return None, ""

    if not registry or not registry.skill_metadata:
        return None, ""

    overview_text = format_skill_overview(registry)

    # Print discovery summary
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
    print(msg)

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
) -> Tuple[bool, Any]:
    """Initialize notebook executor.

    Returns ``(use_notebook_flag, executor_or_None)``.  When initialization
    fails, ``use_notebook_flag`` is ``False`` regardless of the input.
    """
    if not use_notebook:
        print("   ⚡ Using in-process execution (no session isolation)")
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
        print("   📓 Session-based notebook execution enabled")
        print(f"      Conda environment: {executor.conda_env or 'default'}")
        print(f"      Session limit: {max_prompts_per_session} prompts")
        print(f"      Storage: {executor.storage_dir}")
        return True, executor
    except Exception as e:
        print(f"   ⚠️  Notebook execution initialization failed: {e}")
        print("   ⚡ Falling back to in-process execution")
        return False, None


# ---------------------------------------------------------------------------
# Filesystem context
# ---------------------------------------------------------------------------

def initialize_filesystem_context(
    *,
    enabled: bool,
    storage_dir: Optional[Path],
) -> Tuple[bool, Any]:
    """Initialize filesystem context manager.

    Returns ``(enabled_flag, context_or_None)``.
    """
    if not enabled:
        print("   ⚡ Filesystem context disabled")
        return False, None

    try:
        from ..filesystem_context import FilesystemContextManager

        ctx = FilesystemContextManager(base_dir=storage_dir)
        print("   📁 Filesystem context enabled")
        print(f"      Session: {ctx.session_id}")
        print(f"      Storage: {ctx._workspace_dir}")
        return True, ctx
    except Exception as e:
        logger.warning("Filesystem context initialization failed: %s", e)
        print(f"   ⚠️  Filesystem context disabled (init failed: {e})")
        return False, None


# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

def initialize_session_history(config: Any) -> Any:
    """Initialize session history if enabled in *config*.

    Returns ``SessionHistory`` or ``None``.
    """
    if not getattr(config, "history_enabled", False):
        return None

    from ..session_history import SessionHistory

    hist = SessionHistory(path=config.history_path)
    print("   📝 Session history enabled")
    return hist


# ---------------------------------------------------------------------------
# Harness tracing & context compaction
# ---------------------------------------------------------------------------

def initialize_tracing(
    config: Any,
    llm: Any,
    model: str,
) -> Tuple[Any, Any]:
    """Initialize trace store and context compactor.

    Returns ``(trace_store_or_None, compactor_or_None)``.
    """
    from ..harness import RunTraceStore
    from ..context_compactor import ContextCompactor

    trace_store = None
    compactor = None

    harness_config = getattr(config, "harness", None)
    if harness_config is not None and getattr(harness_config, "enable_traces", False):
        trace_store = RunTraceStore(root=harness_config.trace_dir)
        print("   🧰 Harness tracing enabled")
    if (
        harness_config is not None
        and getattr(harness_config, "enable_context_compaction", False)
        and llm
    ):
        compactor = ContextCompactor(llm, model)
        print("   🗜️  Harness context compaction enabled")

    return trace_store, compactor


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def initialize_security(config: Any) -> Tuple[Any, Any]:
    """Initialize security scanner from *config*.

    Returns ``(security_config, scanner)``.
    """
    from ..agent_sandbox import CodeSecurityScanner, SecurityConfig

    security_config = getattr(config, "security", SecurityConfig())
    scanner = CodeSecurityScanner(security_config)
    print(
        f"   🛡️  Security scanner enabled (approval: {security_config.approval_mode.value})"
    )
    return security_config, scanner


# ---------------------------------------------------------------------------
# OmicVerse runtime (workflow / run-store bridge)
# ---------------------------------------------------------------------------

def initialize_ov_runtime(repo_root: Optional[Path]) -> Any:
    """Initialize ``OmicVerseRuntime``.

    Returns the runtime instance or ``None`` on failure.
    """
    from .runtime import OmicVerseRuntime

    try:
        rt = OmicVerseRuntime(repo_root=repo_root)
        if rt.workflow.exists:
            print(f"   📋 Workflow policy loaded: {rt.workflow.path}")
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
) -> None:
    """Print reflection and result-review configuration."""
    if enable_reflection:
        suffix = "s" if reflection_iterations > 1 else ""
        print(
            f"   🔍 Reflection enabled: {reflection_iterations} iteration{suffix} "
            "(code review & validation)"
        )
    else:
        print("   ⚡ Reflection disabled (faster execution, no code validation)")

    if enable_result_review:
        print("   ✅ Result review enabled (output validation & assessment)")
    else:
        print("   ⚡ Result review disabled (no output validation)")
