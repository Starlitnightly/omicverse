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
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..agent_bridge import AgentBridge
from .._bridge_session import resolve_bridge_session_id
from ..channel_media import build_channel_request, prepare_channel_delivery_figures
from ..gateway.routing import GatewaySessionRegistry, SessionKey


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
    t = re.sub(r"/(?:Users|home|tmp|var|opt|root|data|mnt|private)/\S+", "", t)
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
        pass


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
            pass
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
        adata_info = ""
    try:
        session.append_memory_log(
            request=user_text,
            summary=(result.summary or "分析完成"),
            adata_info=adata_info,
        )
    except Exception:
        pass
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
