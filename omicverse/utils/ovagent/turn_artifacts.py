"""Artifact persistence and debug-output helpers for the agentic turn loop.

Extracted from ``turn_controller.py`` during Phase 2 decomposition.
These helpers handle harness history persistence, conversation logging,
and execute-code debug-output persistence to the workspace results directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..harness import (
    coerce_usage_payload,
)
from ..session_history import HistoryEntry

if TYPE_CHECKING:
    from .protocol import AgentContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Harness history persistence
# ---------------------------------------------------------------------------


def persist_harness_history(ctx: "AgentContext", request: str) -> None:
    """Persist the latest trace into session history when enabled."""
    if ctx._session_history is None or ctx._last_run_trace is None:
        return

    trace = ctx._last_run_trace

    def _step_type(step):
        return (
            step.get("step_type")
            if isinstance(step, dict)
            else step.step_type
        )

    def _step_name(step):
        return (
            step.get("name") if isinstance(step, dict) else step.name
        )

    def _step_data(step):
        return (
            step.get("data", {})
            if isinstance(step, dict)
            else step.data
        )

    generated_code = "\n\n".join(
        _step_data(step).get("code", "")
        for step in trace.steps
        if _step_type(step) == "code"
        and _step_data(step).get("code")
    )
    tool_names = [
        _step_name(step)
        for step in trace.steps
        if _step_type(step) == "tool_call" and _step_name(step)
    ]
    artifact_refs = [
        (
            artifact.to_dict()
            if hasattr(artifact, "to_dict")
            else dict(artifact)
        )
        for artifact in trace.artifacts
    ]
    ctx._session_history.append(
        HistoryEntry(
            session_id=(
                trace.session_id or ctx._get_harness_session_id()
            ),
            timestamp=trace.finished_at or trace.started_at,
            request=request,
            trace_id=trace.trace_id,
            generated_code=generated_code,
            result_summary=trace.result_summary,
            tool_names=tool_names,
            artifact_refs=artifact_refs,
            usage=(
                trace.usage
                or coerce_usage_payload(ctx.last_usage)
            ),
            usage_breakdown=ctx.last_usage_breakdown,
            success=trace.status == "success",
        )
    )


# ---------------------------------------------------------------------------
# Conversation logging
# ---------------------------------------------------------------------------


def save_conversation_log(messages: list) -> None:
    """Save the full conversation to a JSON file for debugging."""
    log_dir = os.environ.get("OV_AGENT_LOG_DIR")
    if not log_dir:
        return
    try:
        import datetime as _dt

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = log_path / f"agent_conversation_{ts}.json"

        def _safe(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            if isinstance(obj, (list, tuple)):
                return [_safe(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _safe(v) for k, v in obj.items()}
            return repr(obj)

        out_file.write_text(
            json.dumps(_safe(messages), indent=2, ensure_ascii=False)
        )
        logger.info("conversation_log_saved path=%s", out_file)
    except Exception as exc:
        logger.debug("Failed to save conversation log: %s", exc)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def slugify_for_filename(text: str, *, max_len: int = 48) -> str:
    """Sanitise *text* for safe use as a filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        return "tool"
    return cleaned[:max_len]


# ---------------------------------------------------------------------------
# Tool debug-output persistence
# ---------------------------------------------------------------------------


def persist_tool_debug_output(
    ctx: "AgentContext",
    tool_name: str,
    output: str,
    *,
    turn_index: int,
    tool_index: int,
    description: str = "",
) -> Optional[Path]:
    """Persist tool debug output to the workspace results directory."""
    fs_ctx = getattr(ctx, "_filesystem_context", None)
    workspace_dir = getattr(fs_ctx, "workspace_dir", None)
    if workspace_dir is None or not output:
        return None
    results_dir = Path(workspace_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = slugify_for_filename(description or tool_name)
    file_path = results_dir / (
        f"{tool_name}_turn_{turn_index:02d}_tool_{tool_index:02d}_{suffix}.log"
    )
    header = (
        f"tool={tool_name}\n"
        f"turn={turn_index}\n"
        f"tool_index={tool_index}\n"
        f"description={description or tool_name}\n"
        f"chars={len(output)}\n"
        "-----\n"
    )
    file_path.write_text(header + output, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Execute-code persistence
# ---------------------------------------------------------------------------


def persist_execute_code_source(
    ctx: "AgentContext",
    code: str,
    *,
    turn_index: int,
    tool_index: int,
    description: str = "",
) -> Optional[Path]:
    """Persist execute_code source to the workspace results directory."""
    fs_ctx = getattr(ctx, "_filesystem_context", None)
    workspace_dir = getattr(fs_ctx, "workspace_dir", None)
    if workspace_dir is None or not code:
        return None
    results_dir = Path(workspace_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = slugify_for_filename(description or "execute_code")
    file_path = results_dir / (
        f"execute_code_turn_{turn_index:02d}_tool_{tool_index:02d}_{suffix}.py"
    )
    file_path.write_text(code, encoding="utf-8")
    return file_path


def persist_execute_code_stdout(
    ctx: "AgentContext",
    stdout: str,
    *,
    turn_index: int,
    tool_index: int,
    description: str = "",
) -> Optional[Path]:
    """Persist execute_code stdout to the workspace results directory."""
    if not stdout:
        return None
    fs_ctx = getattr(ctx, "_filesystem_context", None)
    workspace_dir = getattr(fs_ctx, "workspace_dir", None)
    if workspace_dir is None:
        return None
    results_dir = Path(workspace_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = slugify_for_filename(description or "execute_code_stdout")
    file_path = results_dir / (
        f"execute_code_stdout_turn_{turn_index:02d}_tool_{tool_index:02d}_{suffix}.log"
    )
    header = (
        "tool=execute_code_stdout\n"
        f"turn={turn_index}\n"
        f"tool_index={tool_index}\n"
        f"description={description or 'execute_code_stdout'}\n"
        f"chars={len(stdout)}\n"
        "-----\n"
    )
    file_path.write_text(header + stdout, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Chunked logging helpers
# ---------------------------------------------------------------------------


def log_tool_debug_output(
    *,
    tool_name: str,
    output: str,
    turn_index: int,
    tool_index: int,
    path: Optional[Path] = None,
    chunk_size: int = 4000,
) -> None:
    """Log tool debug output in chunks."""
    if not output:
        return
    total_chunks = max(1, (len(output) + chunk_size - 1) // chunk_size)
    logger.info(
        "%s_result_saved turn=%d tool_index=%d chars=%d path=%s chunks=%d",
        tool_name,
        turn_index,
        tool_index,
        len(output),
        str(path) if path is not None else "",
        total_chunks,
    )
    for chunk_idx, start in enumerate(range(0, len(output), chunk_size), start=1):
        chunk = output[start : start + chunk_size]
        logger.info(
            "%s_result_chunk turn=%d tool_index=%d chunk=%d/%d path=%s\n%s",
            tool_name,
            turn_index,
            tool_index,
            chunk_idx,
            total_chunks,
            str(path) if path is not None else "",
            chunk,
        )


def log_execute_code_source(
    *,
    code: str,
    turn_index: int,
    tool_index: int,
    path: Optional[Path] = None,
    chunk_size: int = 4000,
) -> None:
    """Log execute_code source in chunks."""
    if not code:
        return
    total_chunks = max(1, (len(code) + chunk_size - 1) // chunk_size)
    logger.info(
        "execute_code_source_saved turn=%d tool_index=%d chars=%d path=%s chunks=%d",
        turn_index,
        tool_index,
        len(code),
        str(path) if path is not None else "",
        total_chunks,
    )
    for chunk_idx, start in enumerate(range(0, len(code), chunk_size), start=1):
        chunk = code[start : start + chunk_size]
        logger.info(
            "execute_code_source_chunk turn=%d tool_index=%d chunk=%d/%d path=%s\n%s",
            turn_index,
            tool_index,
            chunk_idx,
            total_chunks,
            str(path) if path is not None else "",
            chunk,
        )


def log_execute_code_stdout(
    *,
    stdout: str,
    turn_index: int,
    tool_index: int,
    path: Optional[Path] = None,
    chunk_size: int = 4000,
) -> None:
    """Log execute_code stdout in chunks."""
    if not stdout:
        return
    total_chunks = max(1, (len(stdout) + chunk_size - 1) // chunk_size)
    logger.info(
        "execute_code_stdout_saved turn=%d tool_index=%d chars=%d path=%s chunks=%d",
        turn_index,
        tool_index,
        len(stdout),
        str(path) if path is not None else "",
        total_chunks,
    )
    for chunk_idx, start in enumerate(range(0, len(stdout), chunk_size), start=1):
        chunk = stdout[start : start + chunk_size]
        logger.info(
            "execute_code_stdout_chunk turn=%d tool_index=%d chunk=%d/%d path=%s\n%s",
            turn_index,
            tool_index,
            chunk_idx,
            total_chunks,
            str(path) if path is not None else "",
            chunk,
        )
