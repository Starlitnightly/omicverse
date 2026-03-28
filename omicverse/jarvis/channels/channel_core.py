"""Shared channel-core abstractions for Jarvis channel implementations.

Centralizes session-resolution helpers, request-building logic, result
processing, web-bridge interaction, and text utilities that were previously
duplicated across every channel file.

Channel-specific transport behaviour (how messages are *sent*) stays in each
channel module.  This module only owns the runtime and gateway boilerplate that
is identical regardless of transport.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agent_bridge import AgentBridge
from .._bridge_session import resolve_bridge_session_id
from ..channel_media import build_channel_request, prepare_channel_delivery_figures
from ..gateway.routing import GatewaySessionRegistry, SessionKey

logger = logging.getLogger(__name__)


# ── Shared dataclass ─────────────────────────────────────────────────────────

@dataclass
class RunningTask:
    """Track an in-flight asyncio analysis task."""
    task: asyncio.Task
    request: str
    started_at: float


# ── Text utilities ───────────────────────────────────────────────────────────

_ARTIFACT_EXTS = r"pdf|csv|tsv|txt|xlsx|html|json|h5ad|png|jpg|svg"


def text_chunks(text: str, limit: int = 2000) -> List[str]:
    """Split *text* into paragraph-aware chunks of at most *limit* characters."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    buf = ""
    for para in text.split("\n\n"):
        cand = f"{buf}\n\n{para}".strip() if buf else para
        if len(cand) <= limit:
            buf = cand
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= limit:
            buf = para
            continue
        pos = 0
        while pos < len(para):
            chunks.append(para[pos : pos + limit])
            pos += limit
        buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def strip_local_paths(text: str) -> str:
    """Remove local filesystem path references from agent output text."""
    t = text or ""
    t = re.sub(r"`[^`\n]*(?:/[^`\n]*){2,}`", "", t)
    t = re.sub(r"/(?:Users|home|tmp|private|root)/\S+", "", t)
    t = re.sub(r"~[/\\]\S+", "", t)
    t = re.sub(
        rf"\.?/?(?:\w[\w/-]*/)+\w[\w.-]*\.(?:{_ARTIFACT_EXTS})",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ── Request building ─────────────────────────────────────────────────────────

def build_full_request(session: Any, text: str, *, channel_label: str) -> str:
    """Build a full analysis request enriched with session context.

    Thin wrapper around :func:`channel_media.build_channel_request` so that
    channels don't need to import that function directly.
    """
    return build_channel_request(session, text, channel_label=channel_label)


# ── Web-bridge helpers ───────────────────────────────────────────────────────

def get_prior_history(
    session_manager: Any,
    channel: str,
    scope_type: str,
    scope_id: str,
    session: Any,
) -> list:
    """Retrieve prior conversation history from the web bridge (if attached)."""
    web_bridge = getattr(session_manager, "gateway_web_bridge", None)
    if not web_bridge:
        return []
    return web_bridge.get_prior_history_simple(
        channel,
        scope_type,
        scope_id,
        session_id=resolve_bridge_session_id(session),
    )


def notify_turn_complete(
    session_manager: Any,
    *,
    channel: str,
    scope_type: str,
    scope_id: str,
    session: Any,
    user_text: str,
    llm_text: str,
    adata: Any = None,
    figures: Optional[list] = None,
) -> None:
    """Mirror a completed analysis turn into the web session."""
    web_bridge = getattr(session_manager, "gateway_web_bridge", None)
    if web_bridge is None:
        return
    try:
        kwargs: Dict[str, Any] = dict(
            channel=channel,
            scope_type=scope_type,
            scope_id=scope_id,
            user_text=user_text,
            llm_text=llm_text,
            adata=adata,
            session_id=resolve_bridge_session_id(session),
        )
        if figures is not None:
            kwargs["figures"] = figures
        web_bridge.on_turn_complete_simple(**kwargs)
    except Exception:
        logger.debug("notify_turn_complete: web bridge callback failed", exc_info=True)


# ── Result processing ────────────────────────────────────────────────────────

def process_result_state(
    session: Any,
    result: Any,
    user_text: str,
) -> tuple:
    """Apply common post-analysis state updates to *session*.

    Returns ``(delivery_figures, adata_info)`` for the channel to use when
    composing its transport-specific reply.
    """
    if result.adata is not None:
        session.adata = result.adata
        try:
            session.save_adata()
            session.prompt_count += 1
        except Exception:
            logger.debug("process_result_state: save_adata/prompt_count update failed", exc_info=True)
    if result.usage is not None:
        session.last_usage = result.usage
    delivery_figures = prepare_channel_delivery_figures(session, result.figures)
    try:
        adata = session.adata
        adata_info = (
            f"{adata.n_obs:,} cells x {adata.n_vars:,} genes"
            if adata is not None
            else ""
        )
    except Exception:
        logger.debug("process_result_state: adata info extraction failed", exc_info=True)
        adata_info = ""
    try:
        session.append_memory_log(
            request=user_text,
            summary=(result.summary or "分析完成"),
            adata_info=adata_info,
        )
    except Exception:
        logger.debug("process_result_state: append_memory_log failed", exc_info=True)
    return delivery_figures, adata_info


def format_analysis_error(result: Any, llm_buf: str, *, max_llm: int = 1200) -> str:
    """Format a standard analysis-error message from *result*."""
    err_text = f"分析出错: {result.error}"
    if result.diagnostics:
        hints = "\n".join(f"- {item}" for item in result.diagnostics[:4])
        err_text += f"\n\n诊断:\n{hints}"
    if llm_buf.strip():
        err_text += f"\n\n模型输出:\n{llm_buf[:max_llm]}"
    return err_text


def default_summary(session: Any) -> str:
    """Return a fallback analysis-complete summary string."""
    if session.adata is not None:
        adata = session.adata
        return f"分析完成\n{adata.n_obs:,} cells x {adata.n_vars:,} genes"
    return "分析完成"


# ── Command data structures ─────────────────────────────────────────────────

@dataclass
class StatusInfo:
    """Structured status data collected from session state.

    Channels call :func:`gather_status` to populate this, then format it
    for their transport (plain text, HTML, etc.).
    """
    adata_shape: Optional[Tuple[int, int]] = None
    obs_columns: List[str] = field(default_factory=list)
    kernel_name: Optional[str] = None
    prompt_count: Optional[int] = None
    max_prompts: Any = None
    session_id: Optional[str] = None
    is_running: bool = False
    running_request: str = ""
    workspace_path: Optional[str] = None


@dataclass
class UsageInfo:
    """Structured token-usage data from the session's last run."""
    input_tokens: str = "?"
    output_tokens: str = "?"
    total_tokens: str = "?"
    cache_read: str = ""
    cache_creation: str = ""
    has_data: bool = False


@dataclass
class WorkspaceInfo:
    """Structured workspace listing."""
    path: str = ""
    h5ad_files: List[Tuple[str, Optional[float]]] = field(default_factory=list)
    h5ad_total: int = 0
    has_agents_md: bool = False
    has_today_memory: bool = False


@dataclass
class SaveResult:
    """Outcome of a :func:`perform_save` attempt."""
    success: bool = False
    no_data: bool = False
    path: Optional[str] = None
    adata_shape: Optional[Tuple[int, int]] = None
    error: Optional[str] = None


# ── Command data gathering ──────────────────────────────────────────────────

def gather_status(
    session: Any,
    *,
    session_manager: Any = None,
    is_running: bool = False,
    running_request: str = "",
) -> StatusInfo:
    """Collect status information from session state.

    Parameters
    ----------
    session : channel session object
    session_manager : optional session manager (needed for kernel name)
    is_running : whether an analysis task is currently running
    running_request : the text of the running request (for display)
    """
    info = StatusInfo(is_running=is_running, running_request=running_request)
    if session.adata is not None:
        a = session.adata
        info.adata_shape = (a.n_obs, a.n_vars)
        try:
            info.obs_columns = list(a.obs.columns[:8])
        except Exception:
            logger.debug("gather_status: obs columns extraction failed", exc_info=True)
    if session_manager is not None:
        try:
            info.kernel_name = session_manager.get_active_kernel(session.user_id)
        except Exception:
            logger.debug("gather_status: get_active_kernel failed", exc_info=True)
    try:
        kst = session.kernel_status()
        if kst:
            info.prompt_count = kst.get("prompt_count", 0)
            info.max_prompts = kst.get("max_prompts", "?")
            info.session_id = kst.get("session_id")
    except Exception:
        logger.debug("gather_status: kernel_status retrieval failed", exc_info=True)
    try:
        info.workspace_path = str(session.agent.workspace_dir)
    except Exception:
        logger.debug("gather_status: workspace_path extraction failed", exc_info=True)
    return info


def _usage_attr(obj: Any, *names: str, default: str = "?") -> str:
    """Extract a formatted attribute from a usage object."""
    for name in names:
        v = getattr(obj, name, None)
        if v is not None:
            return f"{v:,}" if isinstance(v, int) else str(v)
    return default


def gather_usage(session: Any) -> UsageInfo:
    """Collect token-usage data from the session's last run."""
    usage = getattr(session, "last_usage", None)
    if usage is None:
        return UsageInfo(has_data=False)
    return UsageInfo(
        input_tokens=_usage_attr(usage, "input_tokens"),
        output_tokens=_usage_attr(usage, "output_tokens"),
        total_tokens=_usage_attr(usage, "total_tokens"),
        cache_read=_usage_attr(usage, "cache_read_input_tokens", default=""),
        cache_creation=_usage_attr(usage, "cache_creation_input_tokens", default=""),
        has_data=True,
    )


def gather_workspace(session: Any) -> WorkspaceInfo:
    """Collect workspace listing data from the session."""
    from datetime import datetime

    ws = session.workspace
    h5ad_files = session.list_h5ad_files()
    agents_md = session.get_agents_md()
    today_log = session.memory_dir / f"{datetime.now().date()}.md"

    files: List[Tuple[str, Optional[float]]] = []
    for f in h5ad_files[:10]:
        try:
            mb = f.stat().st_size / 1_048_576
            files.append((f.name, mb))
        except OSError:
            files.append((f.name, None))

    return WorkspaceInfo(
        path=str(ws),
        h5ad_files=files,
        h5ad_total=len(h5ad_files),
        has_agents_md=bool(agents_md),
        has_today_memory=today_log.exists(),
    )


def perform_save(session: Any) -> SaveResult:
    """Execute the /save command: persist current adata and return a result.

    This is a *synchronous* call.  Channels that need async should wrap it
    with ``asyncio.to_thread``.
    """
    if session.adata is None:
        return SaveResult(no_data=True)
    try:
        path = session.save_adata()
        if not path or not Path(path).exists():
            return SaveResult(error="保存失败，请重试。")
        a = session.adata
        return SaveResult(
            success=True,
            path=str(path),
            adata_shape=(a.n_obs, a.n_vars),
        )
    except Exception as exc:
        return SaveResult(error=str(exc))


# ── Default plain-text formatters ───────────────────────────────────────────

def format_status_plain(info: StatusInfo) -> str:
    """Format *StatusInfo* as plain text (QQ, Discord, WeChat, iMessage, …)."""
    lines: List[str] = []
    if info.adata_shape:
        n_obs, n_vars = info.adata_shape
        lines.append(f"{n_obs:,} cells x {n_vars:,} genes")
        if info.obs_columns:
            lines.append(f"obs: {', '.join(info.obs_columns)}")
    else:
        lines.append("暂无数据")
    if info.kernel_name:
        lines.append(f"kernel: {info.kernel_name}")
    if info.prompt_count is not None:
        lines.append(f"prompts: {info.prompt_count}/{info.max_prompts or '?'}")
    if info.is_running:
        lines.append("分析中（可 /cancel）")
    if info.workspace_path:
        lines.append(f"工作区: {info.workspace_path}")
    return "\n".join(lines)


def format_usage_plain(info: UsageInfo) -> str:
    """Format *UsageInfo* as plain text."""
    if not info.has_data:
        return "暂无用量数据，请先进行一次分析。"
    lines = [
        "Token 用量（最近一次）",
        f"输入: {info.input_tokens}",
        f"输出: {info.output_tokens}",
        f"合计: {info.total_tokens}",
    ]
    if info.cache_read and info.cache_read != "?":
        lines.append(f"缓存读取: {info.cache_read}")
    if info.cache_creation and info.cache_creation != "?":
        lines.append(f"缓存写入: {info.cache_creation}")
    return "\n".join(lines)


def format_workspace_plain(info: WorkspaceInfo) -> str:
    """Format *WorkspaceInfo* as plain text."""
    lines = [f"Workspace: {info.path}", ""]
    if info.h5ad_files:
        lines.append(f"数据文件 ({info.h5ad_total})")
        for name, mb in info.h5ad_files:
            if mb is not None:
                lines.append(f"- {name} ({mb:.1f} MB)")
            else:
                lines.append(f"- {name}")
        if info.h5ad_total > len(info.h5ad_files):
            lines.append(f"... 还有 {info.h5ad_total - len(info.h5ad_files)} 个")
    else:
        lines.append("数据文件 (空)")
    lines += [
        "",
        f"AGENTS.md {'OK' if info.has_agents_md else '-'}",
        f"今日记忆 {'OK' if info.has_today_memory else '-'}",
    ]
    return "\n".join(lines)


# ── Shared request-queue helpers ───────────────────────────────────────────

def coalesce_pending_requests(
    items: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Merge a queue of pending analysis requests into one.

    Each *item* is expected to have ``"text"`` and optional
    ``"request_content"`` keys.  Returns ``(coalesced_text, merged_content)``.
    """
    parts: List[str] = []
    request_content: List[Dict[str, Any]] = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if text:
            parts.append(text)
        request_content.extend(list(item.get("request_content") or []))
    return "\n\n".join(parts).strip(), request_content


def command_parts(text: str) -> Tuple[str, str]:
    """Split ``/command tail`` text into ``(command, tail)`` pair."""
    tokens = text.split()
    cmd = tokens[0].lower() if tokens else ""
    tail = text.split(None, 1)[1].strip() if len(tokens) > 1 else ""
    return cmd, tail


def format_save_result_plain(result: SaveResult) -> str:
    """Format *SaveResult* as plain text."""
    if result.no_data:
        return "没有数据，请先 /load 或完成分析。"
    if result.error:
        return f"保存失败: {result.error}"
    if result.success and result.adata_shape:
        n_obs, n_vars = result.adata_shape
        return (
            f"已保存 current.h5ad\n"
            f"{n_obs:,} cells x {n_vars:,} genes\n"
            f"路径: {result.path}"
        )
    return "保存失败，请重试。"
