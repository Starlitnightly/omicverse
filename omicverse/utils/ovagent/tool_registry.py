"""Tool metadata contract and registry seam for OVAgent dispatch.

Defines the execution-facing tool metadata model (ToolMetadata) and the
registry (ToolRegistry) that maps canonical tool names to metadata and
handler callables.  The model reuses ToolDefinition from tool_catalog
where available and adds policy attributes (approval, parallelism,
output tier, isolation) that drive dispatch, scheduling, and permission
decisions.

This module is the stable contract for follow-on dispatch and scheduler
tasks.  The registry is a *seam*: metadata is registered declaratively,
and handler callables are bound at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

from ..harness.tool_catalog import (
    ToolCatalog,
    ToolDefinition,
    get_default_tool_catalog,
)
from .tool_runtime import LEGACY_AGENT_TOOLS


# ---------------------------------------------------------------------------
# Policy enums
# ---------------------------------------------------------------------------


class ApprovalClass(str, Enum):
    """Per-tool approval policy tier."""

    allow = "allow"
    ask = "ask"
    deny = "deny"


class ParallelClass(str, Enum):
    """Per-tool parallel-safety classification."""

    readonly = "readonly"
    stateful = "stateful"
    exclusive = "exclusive"


class OutputTier(str, Enum):
    """Expected output volume tier for context-budget decisions."""

    minimal = "minimal"
    standard = "standard"
    verbose = "verbose"


class IsolationMode(str, Enum):
    """Execution isolation requirement."""

    none = "none"
    sandbox = "sandbox"
    worktree = "worktree"


# ---------------------------------------------------------------------------
# Tool metadata model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolMetadata:
    """Execution-facing contract for a single tool.

    Carries everything that dispatch, scheduling, approval routing, and
    context budgeting need for one tool.  When a ``ToolDefinition`` from
    the catalog exists, it is referenced via *definition*; legacy tools
    that only have a raw dict schema use *legacy_schema* instead.
    """

    canonical_name: str
    handler_key: str
    approval_class: ApprovalClass
    parallel_class: ParallelClass
    output_tier: OutputTier
    isolation_mode: IsolationMode
    is_async: bool = False
    definition: Optional[ToolDefinition] = None
    legacy_schema: Optional[Dict[str, Any]] = field(default=None, hash=False)
    pre_exec_hook: Optional[str] = None
    post_exec_hook: Optional[str] = None
    normalize_result_hook: Optional[str] = None
    migration_notes: str = ""

    @property
    def schema(self) -> Dict[str, Any]:
        """Return the JSON-schema payload for this tool."""
        if self.definition is not None:
            return self.definition.to_tool_schema()
        if self.legacy_schema is not None:
            return dict(self.legacy_schema)
        return {
            "name": self.canonical_name,
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    @property
    def aliases(self) -> Tuple[str, ...]:
        """All known aliases (including the canonical name)."""
        if self.definition is not None:
            return self.definition.all_aliases
        return (self.canonical_name,)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for inspection and debugging."""
        return {
            "canonical_name": self.canonical_name,
            "handler_key": self.handler_key,
            "approval_class": self.approval_class.value,
            "parallel_class": self.parallel_class.value,
            "output_tier": self.output_tier.value,
            "isolation_mode": self.isolation_mode.value,
            "is_async": self.is_async,
            "has_definition": self.definition is not None,
            "has_legacy_schema": self.legacy_schema is not None,
            "pre_exec_hook": self.pre_exec_hook,
            "post_exec_hook": self.post_exec_hook,
            "normalize_result_hook": self.normalize_result_hook,
            "migration_notes": self.migration_notes,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry seam: maps canonical names to metadata and handler callables.

    Alias resolution delegates to the underlying ``ToolCatalog`` for
    catalog tools, so all existing normalization rules (snake_case,
    kebab-case, legacy aliases) continue to work unchanged.
    """

    def __init__(self, catalog: Optional[ToolCatalog] = None) -> None:
        self._catalog = catalog or get_default_tool_catalog()
        self._entries: Dict[str, ToolMetadata] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}

    # -- registration -------------------------------------------------------

    def register(self, metadata: ToolMetadata) -> None:
        """Register a tool's metadata.  Raises on duplicate canonical names."""
        name = metadata.canonical_name
        if name in self._entries:
            raise ValueError(f"Duplicate registration for canonical name: {name}")
        self._entries[name] = metadata

    def register_handler(self, handler_key: str, handler: Callable[..., Any]) -> None:
        """Bind a callable to a handler key at runtime."""
        self._handlers[handler_key] = handler

    # -- lookup -------------------------------------------------------------

    def resolve_name(self, name_or_alias: str) -> str:
        """Resolve a name or alias to a canonical tool name, or ``""``."""
        if not name_or_alias:
            return ""
        if name_or_alias in self._entries:
            return name_or_alias
        # Delegate to catalog for alias / normalization resolution.
        catalog_name = self._catalog.resolve_name(name_or_alias)
        if catalog_name and catalog_name in self._entries:
            return catalog_name
        return ""

    def get(self, name_or_alias: str) -> Optional[ToolMetadata]:
        """Look up metadata by canonical name or any known alias."""
        canonical = self.resolve_name(name_or_alias)
        return self._entries.get(canonical) if canonical else None

    def get_handler(self, name_or_alias: str) -> Optional[Callable[..., Any]]:
        """Look up the runtime handler for a tool."""
        metadata = self.get(name_or_alias)
        if metadata is None:
            return None
        return self._handlers.get(metadata.handler_key)

    # -- introspection ------------------------------------------------------

    def all_entries(self) -> Tuple[ToolMetadata, ...]:
        """Return all registered metadata entries."""
        return tuple(self._entries.values())

    def handler_keys(self) -> frozenset[str]:
        """Return all distinct handler keys referenced by registered entries."""
        return frozenset(m.handler_key for m in self._entries.values())

    def validate_handlers(self) -> list[str]:
        """Return handler keys that have no bound callable (unresolved seam)."""
        return sorted(key for key in self.handler_keys() if key not in self._handlers)

    def policy_summary(self) -> Dict[str, Dict[str, str]]:
        """Return a {name: {policy_field: value}} dict for all entries."""
        return {
            name: {
                "approval_class": meta.approval_class.value,
                "parallel_class": meta.parallel_class.value,
                "output_tier": meta.output_tier.value,
                "isolation_mode": meta.isolation_mode.value,
            }
            for name, meta in sorted(self._entries.items())
        }


# ---------------------------------------------------------------------------
# Default registry construction
# ---------------------------------------------------------------------------

# Policy tuples: (handler_key, approval, parallel, output, isolation, is_async)
_CatalogPolicy = Tuple[str, ApprovalClass, ParallelClass, OutputTier, IsolationMode, bool]

_CATALOG_TOOL_POLICIES: Dict[str, _CatalogPolicy] = {
    "ToolSearch":          ("tool_search",          ApprovalClass.allow, ParallelClass.readonly,  OutputTier.minimal,  IsolationMode.none,     False),
    "Bash":                ("bash",                 ApprovalClass.ask,   ParallelClass.stateful,  OutputTier.verbose,  IsolationMode.sandbox,  False),
    "Read":                ("read",                 ApprovalClass.allow, ParallelClass.readonly,  OutputTier.standard, IsolationMode.none,     False),
    "Edit":                ("edit",                 ApprovalClass.ask,   ParallelClass.stateful,  OutputTier.standard, IsolationMode.none,     False),
    "Write":               ("write",                ApprovalClass.ask,   ParallelClass.stateful,  OutputTier.standard, IsolationMode.none,     False),
    "Glob":                ("glob",                 ApprovalClass.allow, ParallelClass.readonly,  OutputTier.standard, IsolationMode.none,     False),
    "Grep":                ("grep",                 ApprovalClass.allow, ParallelClass.readonly,  OutputTier.standard, IsolationMode.none,     False),
    "NotebookEdit":        ("notebook_edit",        ApprovalClass.ask,   ParallelClass.stateful,  OutputTier.standard, IsolationMode.none,     False),
    "Agent":               ("agent",                ApprovalClass.ask,   ParallelClass.stateful,  OutputTier.verbose,  IsolationMode.none,     True),
    "AskUserQuestion":     ("ask_user_question",    ApprovalClass.allow, ParallelClass.exclusive, OutputTier.minimal,  IsolationMode.none,     False),
    "TaskCreate":          ("task_create",          ApprovalClass.allow, ParallelClass.stateful,  OutputTier.minimal,  IsolationMode.none,     False),
    "TaskGet":             ("task_get",             ApprovalClass.allow, ParallelClass.readonly,  OutputTier.minimal,  IsolationMode.none,     False),
    "TaskList":            ("task_list",            ApprovalClass.allow, ParallelClass.readonly,  OutputTier.minimal,  IsolationMode.none,     False),
    "TaskOutput":          ("task_output",          ApprovalClass.allow, ParallelClass.readonly,  OutputTier.standard, IsolationMode.none,     False),
    "TaskStop":            ("task_stop",            ApprovalClass.ask,   ParallelClass.stateful,  OutputTier.minimal,  IsolationMode.none,     False),
    "TaskUpdate":          ("task_update",          ApprovalClass.allow, ParallelClass.stateful,  OutputTier.minimal,  IsolationMode.none,     False),
    "EnterPlanMode":       ("enter_plan_mode",      ApprovalClass.allow, ParallelClass.exclusive, OutputTier.minimal,  IsolationMode.none,     False),
    "ExitPlanMode":        ("exit_plan_mode",       ApprovalClass.allow, ParallelClass.exclusive, OutputTier.minimal,  IsolationMode.none,     False),
    "EnterWorktree":       ("enter_worktree",       ApprovalClass.ask,   ParallelClass.exclusive, OutputTier.minimal,  IsolationMode.worktree, False),
    "Skill":               ("skill",                ApprovalClass.allow, ParallelClass.readonly,  OutputTier.standard, IsolationMode.none,     False),
    "WebFetch":            ("web_fetch",            ApprovalClass.allow, ParallelClass.readonly,  OutputTier.standard, IsolationMode.none,     False),
    "WebSearch":           ("web_search",           ApprovalClass.allow, ParallelClass.readonly,  OutputTier.standard, IsolationMode.none,     False),
    "ListMcpResourcesTool":("list_mcp_resources",   ApprovalClass.allow, ParallelClass.readonly,  OutputTier.minimal,  IsolationMode.none,     False),
    "ReadMcpResourceTool": ("read_mcp_resource",    ApprovalClass.allow, ParallelClass.readonly,  OutputTier.standard, IsolationMode.none,     False),
}

_CATALOG_MIGRATION_NOTES: Dict[str, str] = {
    "Agent": (
        "Requires late-bound subagent_controller; dispatch must check "
        "initialisation before calling handler."
    ),
    "Skill": (
        "Legacy schema 'search_skills' in LEGACY_AGENT_TOOLS normalises "
        "to this entry via catalog alias resolution."
    ),
    "WebFetch": (
        "Legacy schema 'web_fetch' in LEGACY_AGENT_TOOLS normalises to "
        "this entry via catalog alias resolution."
    ),
    "WebSearch": (
        "Legacy schema 'web_search' in LEGACY_AGENT_TOOLS normalises to "
        "this entry via catalog alias resolution."
    ),
}

# Legacy tool names that are aliases of catalog tools (handled by catalog
# normalization, NOT separate registry entries).
_LEGACY_CATALOG_ALIASES: frozenset[str] = frozenset({
    "delegate",       # → Agent
    "web_fetch",      # → WebFetch
    "web_search",     # → WebSearch
    "search_skills",  # → Skill
})

# Legacy-only tools: no catalog ToolDefinition, own registry entries.
# Tuple: (handler_key, approval, parallel, output, isolation, is_async, migration_notes)
_LegacyPolicy = Tuple[str, ApprovalClass, ParallelClass, OutputTier, IsolationMode, bool, str]

_LEGACY_TOOL_POLICIES: Dict[str, _LegacyPolicy] = {
    "inspect_data": (
        "inspect_data", ApprovalClass.allow, ParallelClass.readonly,
        OutputTier.standard, IsolationMode.none, False,
        "Requires adata parameter; handler is read-only.",
    ),
    "execute_code": (
        "execute_code", ApprovalClass.ask, ParallelClass.stateful,
        OutputTier.verbose, IsolationMode.sandbox, False,
        "Requires adata parameter; handler includes proactive-transform and retry logic.",
    ),
    "run_snippet": (
        "run_snippet", ApprovalClass.allow, ParallelClass.readonly,
        OutputTier.verbose, IsolationMode.sandbox, False,
        "Requires adata parameter; executes on a shallow copy (read-only).",
    ),
    "search_functions": (
        "search_functions", ApprovalClass.allow, ParallelClass.readonly,
        OutputTier.standard, IsolationMode.none, False,
        "",
    ),
    "web_download": (
        "web_download", ApprovalClass.ask, ParallelClass.stateful,
        OutputTier.standard, IsolationMode.none, False,
        "",
    ),
    "finish": (
        "finish", ApprovalClass.allow, ParallelClass.exclusive,
        OutputTier.minimal, IsolationMode.none, False,
        "Terminal tool; handler returns a fixed dict inline in dispatch_tool.",
    ),
}


def build_default_registry() -> ToolRegistry:
    """Construct the default registry from the live tool catalog and legacy tools.

    Every tool that ``dispatch_tool`` can route to gets a ``ToolMetadata``
    entry.  Legacy names that are aliases for catalog tools (delegate,
    web_fetch, web_search, search_skills) resolve through the catalog's
    alias normalization and do NOT get separate entries.
    """
    catalog = get_default_tool_catalog()
    registry = ToolRegistry(catalog)

    # ---- catalog tools ----------------------------------------------------
    for tool_def in catalog.all_tools():
        name = tool_def.name
        policy = _CATALOG_TOOL_POLICIES.get(name)
        if policy is None:
            continue
        handler_key, approval, parallel, output, isolation, is_async = policy
        registry.register(ToolMetadata(
            canonical_name=name,
            handler_key=handler_key,
            approval_class=approval,
            parallel_class=parallel,
            output_tier=output,
            isolation_mode=isolation,
            is_async=is_async,
            definition=tool_def,
            migration_notes=_CATALOG_MIGRATION_NOTES.get(name, ""),
        ))

    # ---- legacy-only tools ------------------------------------------------
    legacy_by_name: Dict[str, Dict[str, Any]] = {
        t["name"]: t for t in LEGACY_AGENT_TOOLS
    }
    for name, policy in _LEGACY_TOOL_POLICIES.items():
        handler_key, approval, parallel, output, isolation, is_async, migration = policy
        registry.register(ToolMetadata(
            canonical_name=name,
            handler_key=handler_key,
            approval_class=approval,
            parallel_class=parallel,
            output_tier=output,
            isolation_mode=isolation,
            is_async=is_async,
            legacy_schema=legacy_by_name.get(name),
            migration_notes=migration,
        ))

    return registry


__all__ = [
    "ApprovalClass",
    "IsolationMode",
    "OutputTier",
    "ParallelClass",
    "ToolMetadata",
    "ToolRegistry",
    "build_default_registry",
]
