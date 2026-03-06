"""
Claude-style tool catalog and loading helpers for OVAgent harness sessions.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Literal, Optional


ToolClassification = Literal["core", "deferred"]


def _camel_to_words(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", value)


def _snake_case(value: str) -> str:
    spaced = _camel_to_words(value)
    return re.sub(r"[^a-z0-9]+", "_", spaced.lower()).strip("_")


def _kebab_case(value: str) -> str:
    return _snake_case(value).replace("_", "-")


def _tokenize(value: str) -> list[str]:
    spaced = _camel_to_words(value)
    raw = re.split(r"[^a-z0-9]+", spaced.lower())
    return [token for token in raw if token]


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _snake_case(value))


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    group: str
    classification: ToolClassification
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    keywords: tuple[str, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)
    legacy_aliases: tuple[str, ...] = field(default_factory=tuple)
    server_only: bool = False
    high_risk: bool = False
    supports_background: bool = False

    @property
    def requires_approval(self) -> bool:
        """Backward-compatible alias for approval-gated tools."""
        return self.high_risk

    @property
    def all_aliases(self) -> tuple[str, ...]:
        generated = {
            self.name,
            self.name.lower(),
            _snake_case(self.name),
            _kebab_case(self.name),
        }
        generated.update(self.aliases)
        generated.update(self.legacy_aliases)
        return tuple(sorted(alias for alias in generated if alias))

    def search_corpus(self) -> str:
        parts = [
            self.name,
            self.group,
            self.description,
            " ".join(self.keywords),
            " ".join(self.aliases),
            " ".join(self.legacy_aliases),
            _camel_to_words(self.name),
        ]
        return " ".join(parts).lower()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requires_approval"] = self.requires_approval
        payload["all_aliases"] = list(self.all_aliases)
        return payload

    def to_tool_schema(self) -> dict[str, Any]:
        parameters = dict(self.parameters)
        if parameters and "type" not in parameters:
            parameters = {
                "type": "object",
                "properties": parameters.get("properties", {}),
                "required": parameters.get("required", []),
            }
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters or {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


@dataclass(frozen=True)
class ToolMatch:
    tool_name: str
    score: int
    reason: str
    already_loaded: bool = False
    classification: ToolClassification = "deferred"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolSearchResult:
    query: str
    mode: Literal["search", "select"]
    matches: tuple[ToolMatch, ...] = field(default_factory=tuple)
    selected_tools: tuple[str, ...] = field(default_factory=tuple)
    loaded_tools: tuple[str, ...] = field(default_factory=tuple)
    unresolved: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "matches": [match.to_dict() for match in self.matches],
            "selected_tools": list(self.selected_tools),
            "loaded_tools": list(self.loaded_tools),
            "unresolved": list(self.unresolved),
        }


def _tool(
    name: str,
    *,
    group: str,
    classification: ToolClassification,
    description: str,
    parameters: Optional[dict[str, Any]] = None,
    keywords: Iterable[str] = (),
    aliases: Iterable[str] = (),
    legacy_aliases: Iterable[str] = (),
    server_only: bool = False,
    high_risk: bool = False,
    supports_background: bool = False,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        group=group,
        classification=classification,
        description=description,
        parameters=dict(parameters or {}),
        keywords=tuple(keywords),
        aliases=tuple(aliases),
        legacy_aliases=tuple(legacy_aliases),
        server_only=server_only,
        high_risk=high_risk,
        supports_background=supports_background,
    )


CLAUDE_CODE_TOOLS: tuple[ToolDefinition, ...] = (
    _tool(
        "ToolSearch",
        group="core",
        classification="core",
        description="Search and select deferred tools by keyword or explicit selection string.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "number", "default": 5},
            },
            "required": ["query"],
        },
        keywords=("search", "tool loading", "deferred tools", "select"),
    ),
    _tool(
        "Bash",
        group="core",
        classification="core",
        description="Execute shell commands with optional background execution.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "timeout": {"type": "number"},
                "run_in_background": {"type": "boolean"},
                "dangerouslyDisableSandbox": {"type": "boolean"},
            },
            "required": ["command"],
        },
        keywords=("shell", "terminal", "command", "exec"),
        high_risk=True,
        supports_background=True,
        server_only=True,
    ),
    _tool(
        "Read",
        group="core",
        classification="core",
        description="Read a local file, image, PDF, or notebook by absolute path.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "number"},
                "limit": {"type": "number"},
                "pages": {"type": "string"},
            },
            "required": ["file_path"],
        },
        keywords=("file", "reader", "pdf", "image", "notebook"),
    ),
    _tool(
        "Edit",
        group="code",
        classification="deferred",
        description="Apply a targeted diff to an existing file.",
        keywords=("patch", "modify", "diff"),
        high_risk=True,
        server_only=True,
    ),
    _tool(
        "Write",
        group="code",
        classification="deferred",
        description="Create or fully rewrite a file.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
        keywords=("create file", "overwrite file", "save"),
        high_risk=True,
        server_only=True,
    ),
    _tool(
        "Glob",
        group="code",
        classification="deferred",
        description="Search for files by glob pattern.",
        keywords=("files", "pattern", "paths", "find"),
    ),
    _tool(
        "Grep",
        group="code",
        classification="deferred",
        description="Search file contents by text or pattern.",
        keywords=("content search", "ripgrep", "text search", "symbol"),
    ),
    _tool(
        "NotebookEdit",
        group="code",
        classification="deferred",
        description="Edit cells inside a Jupyter notebook.",
        keywords=("ipynb", "notebook", "cells", "jupyter"),
        high_risk=True,
        server_only=True,
    ),
    _tool(
        "Agent",
        group="agent",
        classification="deferred",
        description="Run a subagent on an isolated task with its own context window.",
        keywords=("subagent", "delegate", "parallel exploration", "worker"),
        legacy_aliases=("delegate",),
    ),
    _tool(
        "AskUserQuestion",
        group="agent",
        classification="deferred",
        description="Pause for user clarification, confirmation, or an explicit decision.",
        keywords=("clarify", "approval", "user question", "pause"),
        aliases=("AskQuestion",),
    ),
    _tool(
        "TaskCreate",
        group="agent",
        classification="deferred",
        description="Create a tracked task record for multi-step work.",
        keywords=("todo", "task tracking", "background task", "progress"),
    ),
    _tool(
        "TaskGet",
        group="agent",
        classification="deferred",
        description="Get details for one tracked task.",
        keywords=("task details", "status", "inspect task"),
    ),
    _tool(
        "TaskList",
        group="agent",
        classification="deferred",
        description="List tracked tasks and their statuses.",
        keywords=("tasks", "progress list", "queue"),
    ),
    _tool(
        "TaskOutput",
        group="agent",
        classification="deferred",
        description="Read captured output from a background task.",
        keywords=("logs", "background output", "stdout", "stderr"),
    ),
    _tool(
        "TaskStop",
        group="agent",
        classification="deferred",
        description="Stop or cancel a running task.",
        keywords=("cancel task", "terminate", "stop background"),
        high_risk=True,
        server_only=True,
    ),
    _tool(
        "TaskUpdate",
        group="agent",
        classification="deferred",
        description="Update task status, summary, metadata, or lifecycle details.",
        keywords=("task status", "mark complete", "progress update"),
    ),
    _tool(
        "EnterPlanMode",
        group="workflow",
        classification="deferred",
        description="Enter analysis-only mode without execution or edits.",
        keywords=("planning", "analysis only", "dry run", "architecture"),
    ),
    _tool(
        "ExitPlanMode",
        group="workflow",
        classification="deferred",
        description="Exit planning-only mode and resume normal execution.",
        keywords=("resume execution", "leave planning", "normal mode"),
    ),
    _tool(
        "EnterWorktree",
        group="workflow",
        classification="deferred",
        description="Switch into an isolated git worktree context.",
        keywords=("git worktree", "isolated branch", "sandbox repo"),
        high_risk=True,
        server_only=True,
    ),
    _tool(
        "Skill",
        group="workflow",
        classification="deferred",
        description="Load a user-invocable skill or workflow guide.",
        keywords=("skill", "workflow guide", "domain guidance"),
        legacy_aliases=("search_skills",),
    ),
    _tool(
        "WebFetch",
        group="web",
        classification="deferred",
        description="Fetch and read content from a specific URL.",
        keywords=("url", "fetch page", "documentation", "http"),
        legacy_aliases=("web_fetch",),
    ),
    _tool(
        "WebSearch",
        group="web",
        classification="deferred",
        description="Run a web search and return ranked search results.",
        keywords=("internet search", "search engine", "lookup"),
        legacy_aliases=("web_search",),
    ),
    _tool(
        "ListMcpResourcesTool",
        group="mcp",
        classification="deferred",
        description="List available MCP resources from configured servers.",
        keywords=("mcp", "resources", "servers", "discover"),
    ),
    _tool(
        "ReadMcpResourceTool",
        group="mcp",
        classification="deferred",
        description="Read one MCP resource from a configured server.",
        keywords=("mcp", "resource read", "server data"),
    ),
)


class ToolCatalog:
    """Catalog of Claude-style tools with search and loading helpers."""

    def __init__(self, tools: Iterable[ToolDefinition]) -> None:
        self._tools = tuple(tools)
        self._by_name = {tool.name: tool for tool in self._tools}
        alias_map: dict[str, str] = {}
        for tool in self._tools:
            alias_map[_normalize_name(tool.name)] = tool.name
            for alias in tool.all_aliases:
                alias_map[_normalize_name(alias)] = tool.name
        self._alias_map = alias_map

    def all_tools(self) -> tuple[ToolDefinition, ...]:
        return self._tools

    def core_tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool for tool in self._tools if tool.classification == "core")

    def deferred_tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool for tool in self._tools if tool.classification == "deferred")

    def core_tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.core_tools())

    def deferred_tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.deferred_tools())

    def get(self, name_or_alias: str) -> Optional[ToolDefinition]:
        canonical = self.resolve_name(name_or_alias)
        if not canonical:
            return None
        return self._by_name.get(canonical)

    def resolve_name(self, name_or_alias: str) -> str:
        return self._alias_map.get(_normalize_name(name_or_alias or ""), "")

    def legacy_aliases(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for tool in self._tools:
            for alias in tool.legacy_aliases:
                mapping[alias] = tool.name
        return mapping

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        loaded_tools: Optional[Iterable[str]] = None,
        include_core: bool = False,
    ) -> tuple[ToolMatch, ...]:
        loaded = {self.resolve_name(name) or name for name in (loaded_tools or ())}
        text = (query or "").strip()
        if not text:
            return ()

        mandatory_tokens: list[str] = []
        optional_tokens: list[str] = []
        for raw in re.split(r"\s+", text):
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("+") and len(raw) > 1:
                mandatory_tokens.extend(_tokenize(raw[1:]))
            else:
                optional_tokens.extend(_tokenize(raw))

        pool = self._tools if include_core else self.deferred_tools()
        matches: list[ToolMatch] = []
        normalized_query = _normalize_name(text)
        for tool in pool:
            corpus = tool.search_corpus()
            corpus_tokens = set(_tokenize(corpus))
            if mandatory_tokens and not all(token in corpus_tokens or token in corpus for token in mandatory_tokens):
                continue

            score = 0
            reasons: list[str] = []
            if normalized_query and normalized_query == _normalize_name(tool.name):
                score += 100
                reasons.append("exact name")
            if normalized_query and normalized_query in _normalize_name(tool.name):
                score += 40
                reasons.append("name match")
            if any(normalized_query == _normalize_name(alias) for alias in tool.all_aliases if normalized_query):
                score += 30
                reasons.append("alias match")
            for token in optional_tokens:
                if token in corpus_tokens:
                    score += 12
                    reasons.append(f"token:{token}")
                elif token in corpus:
                    score += 6
                    reasons.append(f"substring:{token}")
            if not optional_tokens and mandatory_tokens:
                score += 5
            if score <= 0:
                continue
            matches.append(
                ToolMatch(
                    tool_name=tool.name,
                    score=score,
                    reason=", ".join(dict.fromkeys(reasons)) or "match",
                    already_loaded=tool.name in loaded,
                    classification=tool.classification,
                )
            )
        matches.sort(key=lambda item: (-item.score, item.tool_name.lower()))
        return tuple(matches[:max_results])

    def select_tools(
        self,
        names: Iterable[str],
        *,
        loaded_tools: Optional[Iterable[str]] = None,
    ) -> ToolSearchResult:
        requested = [name for name in names]
        loaded = {self.resolve_name(name) or name for name in (loaded_tools or ())}
        selected: list[str] = []
        unresolved: list[str] = []
        for raw in requested:
            canonical = self.resolve_name(raw)
            if not canonical:
                unresolved.append(raw)
                continue
            if canonical not in loaded:
                loaded.add(canonical)
            selected.append(canonical)
        return ToolSearchResult(
            query="select:" + ",".join(requested),
            mode="select",
            selected_tools=tuple(selected),
            loaded_tools=tuple(sorted(loaded)),
            unresolved=tuple(unresolved),
        )

    def resolve_loading_query(
        self,
        query: str,
        *,
        loaded_tools: Optional[Iterable[str]] = None,
        max_results: int = 5,
        auto_load_search_matches: bool = False,
    ) -> ToolSearchResult:
        text = (query or "").strip()
        loaded = {self.resolve_name(name) or name for name in (loaded_tools or ())}
        if text.lower().startswith("select:"):
            raw_names = [part.strip() for part in text.split(":", 1)[1].split(",") if part.strip()]
            return self.select_tools(raw_names, loaded_tools=loaded)

        matches = self.search(text, max_results=max_results, loaded_tools=loaded)
        selected: list[str] = []
        if auto_load_search_matches:
            for match in matches:
                if match.tool_name not in loaded:
                    loaded.add(match.tool_name)
                selected.append(match.tool_name)
        return ToolSearchResult(
            query=text,
            mode="search",
            matches=matches,
            selected_tools=tuple(selected),
            loaded_tools=tuple(sorted(loaded)),
        )

    def tool_schemas(
        self,
        *,
        loaded_tools: Optional[Iterable[str]] = None,
        include_core: bool = True,
    ) -> list[dict[str, Any]]:
        loaded = {self.resolve_name(name) or name for name in (loaded_tools or ())}
        payload: list[dict[str, Any]] = []
        for tool in self._tools:
            if tool.classification == "core":
                if include_core:
                    payload.append(tool.to_tool_schema())
                continue
            if tool.name in loaded:
                payload.append(tool.to_tool_schema())
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_tools": [tool.to_dict() for tool in self.core_tools()],
            "deferred_tools": [tool.to_dict() for tool in self.deferred_tools()],
            "legacy_aliases": self.legacy_aliases(),
        }


@lru_cache(maxsize=1)
def get_default_tool_catalog() -> ToolCatalog:
    return ToolCatalog(CLAUDE_CODE_TOOLS)


CORE_TOOL_NAMES = frozenset(get_default_tool_catalog().core_tool_names())
DEFERRED_TOOL_NAMES = frozenset(get_default_tool_catalog().deferred_tool_names())


def resolve_tool_name(name_or_alias: str) -> str:
    return get_default_tool_catalog().resolve_name(name_or_alias)


def normalize_tool_name(name_or_alias: str) -> str:
    return resolve_tool_name(name_or_alias) or (name_or_alias or "")


def get_tool_spec(name_or_alias: str) -> Optional[ToolDefinition]:
    return get_default_tool_catalog().get(name_or_alias)


def get_default_loaded_tool_names() -> tuple[str, ...]:
    # Keep the existing high-value coordination/search tools available by
    # default so OVAgent remains useful without an initial ToolSearch turn.
    return tuple(sorted(
        set(CORE_TOOL_NAMES)
        | {"Agent", "AskUserQuestion", "Skill", "WebFetch", "WebSearch"}
    ))


def get_visible_tool_schemas(loaded_tools: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
    loaded = set(loaded_tools or get_default_loaded_tool_names())
    return get_default_tool_catalog().tool_schemas(loaded_tools=loaded, include_core=True)


def resolve_tool_search(
    query: str,
    *,
    loaded_tools: Optional[Iterable[str]] = None,
    max_results: int = 5,
    auto_load_search_matches: bool = False,
) -> dict[str, Any]:
    return resolve_loading_query(
        query,
        loaded_tools=loaded_tools,
        max_results=max_results,
        auto_load_search_matches=auto_load_search_matches,
    ).to_dict()


def search_claude_code_tools(
    query: str,
    *,
    max_results: int = 5,
    loaded_tools: Optional[Iterable[str]] = None,
) -> tuple[ToolMatch, ...]:
    return get_default_tool_catalog().search(
        query,
        max_results=max_results,
        loaded_tools=loaded_tools,
    )


def resolve_loading_query(
    query: str,
    *,
    loaded_tools: Optional[Iterable[str]] = None,
    max_results: int = 5,
    auto_load_search_matches: bool = False,
) -> ToolSearchResult:
    return get_default_tool_catalog().resolve_loading_query(
        query,
        loaded_tools=loaded_tools,
        max_results=max_results,
        auto_load_search_matches=auto_load_search_matches,
    )


__all__ = [
    "CLAUDE_CODE_TOOLS",
    "CORE_TOOL_NAMES",
    "DEFERRED_TOOL_NAMES",
    "ToolCatalog",
    "ToolClassification",
    "ToolDefinition",
    "ToolMatch",
    "ToolSearchResult",
    "get_default_tool_catalog",
    "get_default_loaded_tool_names",
    "get_tool_spec",
    "get_visible_tool_schemas",
    "normalize_tool_name",
    "resolve_loading_query",
    "resolve_tool_search",
    "resolve_tool_name",
    "search_claude_code_tools",
]
