"""Workspace tool handlers: tasks, plan mode, worktree, skill, MCP.

Extracted from ``tool_runtime.py`` during Phase 3 decomposition.
These handler functions receive the ``AgentContext`` (and optionally a
plan-mode checker or read callback) as explicit parameters.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from ..harness.runtime_state import runtime_state

if TYPE_CHECKING:
    from .protocol import AgentContext


# ------------------------------------------------------------------
# Task management tools
# ------------------------------------------------------------------


def handle_create_task(
    ctx: "AgentContext",
    title: str,
    description: str = "",
    status: str = "pending",
) -> str:
    """Create a new task."""
    task = runtime_state.create_task(
        ctx._get_runtime_session_id(),
        title=title,
        description=description,
        status=status if status else "pending",
    )
    return json.dumps(task.to_dict(), ensure_ascii=False, indent=2)


def handle_get_task(ctx: "AgentContext", task_id: str) -> str:
    """Get a task by ID."""
    task = runtime_state.get_task(
        ctx._get_runtime_session_id(), task_id
    )
    payload = (
        task.to_dict() if task is not None else {"error": "Task not found"}
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def handle_list_tasks(ctx: "AgentContext", status: str = "") -> str:
    """List tasks, optionally filtered by status."""
    tasks = runtime_state.list_tasks(
        ctx._get_runtime_session_id(), status=status
    )
    return json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2)


def handle_task_output(
    ctx: "AgentContext",
    task_id: str,
    offset: int = 0,
    limit: int = 200,
) -> str:
    """Read output from a task."""
    payload = runtime_state.read_task_output(
        ctx._get_runtime_session_id(),
        task_id,
        offset=offset,
        limit=limit,
    )
    return json.dumps(
        payload or {"error": "Task not found"},
        ensure_ascii=False,
        indent=2,
    )


def handle_task_stop(ctx: "AgentContext", task_id: str) -> str:
    """Stop a running task."""
    ctx._ensure_server_tool_mode("TaskStop")
    updated = runtime_state.stop_task(
        ctx._get_runtime_session_id(), task_id
    )
    payload = (
        updated.to_dict()
        if updated is not None
        else {"error": "Task not found"}
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def handle_task_update(
    ctx: "AgentContext",
    task_id: str,
    status: str,
    summary: str = "",
) -> str:
    """Update a task's status and summary."""
    updated = runtime_state.update_task(
        ctx._get_runtime_session_id(),
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


def handle_enter_plan_mode(
    ctx: "AgentContext", reason: str = ""
) -> str:
    """Enter plan mode."""
    payload = runtime_state.enter_plan_mode(
        ctx._get_runtime_session_id(), reason=reason
    )
    return json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)


def handle_exit_plan_mode(
    ctx: "AgentContext", summary: str = ""
) -> str:
    """Exit plan mode."""
    payload = runtime_state.exit_plan_mode(
        ctx._get_runtime_session_id(), reason=summary
    )
    return json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)


def handle_enter_worktree(
    ctx: "AgentContext",
    plan_mode_checker: Callable[[str], bool],
    branch_name: str = "",
    path: str = "",
    base_ref: str = "HEAD",
) -> str:
    """Create or switch to a git worktree."""
    ctx._ensure_server_tool_mode("EnterWorktree")
    if plan_mode_checker("EnterWorktree"):
        return (
            "EnterWorktree is blocked while the session is in plan mode."
        )
    repo_root = ctx._detect_repo_root()
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
        or f"ovagent/{ctx._get_runtime_session_id()}"
    )
    if path:
        worktree_path = ctx._resolve_local_path(
            path, allow_relative=True
        )
    else:
        worktree_root = Path.home() / ".ovagent" / "worktrees"
        worktree_root.mkdir(parents=True, exist_ok=True)
        worktree_path = worktree_root / branch.replace("/", "_")
    ctx._request_tool_approval(
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
        ctx._get_runtime_session_id(),
        path=str(worktree_path),
        repo_root=str(repo_root),
        branch=branch,
        base_branch=base_ref,
    )
    return json.dumps(worktree.to_dict(), ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# Skill / MCP / User interaction
# ------------------------------------------------------------------


def handle_search_skills(ctx: "AgentContext", query: str) -> str:
    """Search installed domain-specific skills."""
    registry = ctx.skill_registry
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
                llm = ctx._llm
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


def handle_skill(ctx: "AgentContext", query: str, mode: str = "search") -> str:
    """Search or load a skill by name/query."""
    if mode == "load":
        return ctx._load_skill_guidance(query)
    exact = None
    registry = ctx.skill_registry
    if registry and registry.skill_metadata:
        for meta in registry.skill_metadata.values():
            if query.strip().lower() in {
                meta.slug.lower(),
                meta.name.lower(),
            }:
                exact = meta.slug
                break
    if exact:
        return ctx._load_skill_guidance(exact)
    return handle_search_skills(ctx, query)


def handle_list_mcp_resources(server: str = "") -> str:
    """List available MCP resources."""
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


def handle_read_mcp_resource(
    server: str,
    uri: str,
    read_fn: Callable[..., str],
) -> str:
    """Read a specific MCP resource by server and URI.

    Parameters
    ----------
    read_fn : callable
        A callback that reads a file path (e.g. ``handle_read`` from
        ``tool_runtime_io``), used when the MCP resource resolves to a
        local file.
    """
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
            return read_fn(target)
    return json.dumps(
        {"error": "MCP resource not found"},
        ensure_ascii=False,
        indent=2,
    )


def handle_ask_user_question(
    ctx: "AgentContext",
    question: str,
    header: str = "",
    options: Optional[list[str]] = None,
) -> str:
    """Ask the user a question and return the resolved answer."""
    from ..harness.contracts import make_turn_id  # noqa: F811

    session_id = ctx._get_runtime_session_id()
    trace = getattr(ctx, "_last_run_trace", None)
    record = runtime_state.create_question(
        session_id,
        turn_id=getattr(trace, "turn_id", ""),
        trace_id=getattr(trace, "trace_id", ""),
        question=question,
        header=header,
        options=list(options or []),
    )
    answer = ctx._request_interaction(
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
