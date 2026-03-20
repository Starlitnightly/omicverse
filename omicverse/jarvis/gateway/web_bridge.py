"""
WebSessionBridge — mirrors jarvis channel turns into omicverse-web AgentSessions.

This module is the glue between the jarvis channel layer and the omicverse-web
SessionManager.  It is intentionally dependency-free at import time: the web
SessionManager is passed in at construction, so this file can be imported even
when omicverse-web is not installed (the bridge simply won't be created).

Usage (from jarvis/cli.py, after GatewayServer is started)::

    from omicverse_web.services.agent_session_service import SessionManager as WebSM
    from omicverse.jarvis.gateway.web_bridge import WebSessionBridge

    web_sm = WebSM(max_sessions=20)
    bridge = WebSessionBridge(web_sm)
    sm.gateway_web_bridge = bridge   # attach to jarvis SessionManager
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime.models import ConversationRoute

logger = logging.getLogger("omicverse.jarvis.gateway.web_bridge")


def _route_to_web_session_id(
    channel: str,
    scope_type: str,
    scope_id: str,
    thread_id: Optional[str] = None,
) -> str:
    """Derive a stable 16-hex-char web session_id from channel routing info.

    Uses SHA-1 (same algorithm as ``GatewaySessionRegistry`` in omicverse-web)
    so that the web and jarvis sides always agree on session IDs.
    """
    key = f"{channel}:{scope_type}:{scope_id}"
    if thread_id:
        key += f":thread:{thread_id}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class WebSessionBridge:
    """Writes completed jarvis turns into the shared omicverse-web SessionManager.

    Parameters
    ----------
    web_session_manager:
        The omicverse-web ``SessionManager`` instance.
    memory_store:
        Optional ``MemoryStore`` — when set, each completed turn is also
        persisted as a memory document for long-term recall.
    """

    def __init__(self, web_session_manager: Any, memory_store: Any = None) -> None:
        self._sm = web_session_manager
        self._memory_store = memory_store

    # ------------------------------------------------------------------
    # Session lookup
    # ------------------------------------------------------------------

    def session_id_for_route(
        self,
        channel: str,
        scope_type: str,
        scope_id: str,
        thread_id: Optional[str] = None,
    ) -> str:
        """Return the stable web session_id for a channel routing triple."""
        return _route_to_web_session_id(channel, scope_type, scope_id, thread_id)

    def session_id_for_conversation_route(self, route: "ConversationRoute") -> str:
        """Convenience overload that takes a ``ConversationRoute`` directly."""
        return _route_to_web_session_id(
            route.channel,
            route.scope_type,
            route.scope_id,
            route.thread_id,
        )

    def ensure_session(self, route: "ConversationRoute") -> Any:
        """Get-or-create the web AgentSession for a ConversationRoute."""
        sid = self.session_id_for_conversation_route(route)
        return self._sm.get_or_create(sid)

    # ------------------------------------------------------------------
    # Turn write-back (called after analysis completes)
    # ------------------------------------------------------------------

    def on_turn_complete(
        self,
        route: "ConversationRoute",
        user_text: str,
        llm_text: str,
        adata: Any = None,
    ) -> None:
        """Mirror a completed analysis turn into the web AgentSession.

        This is the primary integration point.  Call it from
        ``MessageRuntime._run_analysis`` (or from each channel's own
        ``_run_analysis``) after the LLM reply is ready.

        Parameters
        ----------
        route:
            The jarvis ``ConversationRoute`` identifying the channel source.
        user_text:
            The user's original message text.
        llm_text:
            The full LLM reply accumulated during the turn.
        adata:
            Updated AnnData object, if the analysis mutated it.
        """
        try:
            sid = self.session_id_for_conversation_route(route)
            web_session = self._sm.get_or_create(sid)
            # Tag user message with channel source for clarity in the web UI
            tagged_user = f"[{route.channel}] {user_text}"
            web_session.add_message("user", tagged_user)
            if llm_text:
                web_session.add_message("assistant", llm_text)
            if adata is not None:
                self._sm.set_shared_adata(adata)
            logger.debug(
                "WebSessionBridge: synced turn for route=%s → web_session=%s",
                route.route_key(),
                sid,
            )
            # Persist to memory store if configured
            if self._memory_store is not None and llm_text:
                try:
                    self._memory_store.create_document(
                        title=user_text[:120],
                        content=f"**User:** {user_text}\n\n**Assistant:** {llm_text}",
                        tags=[route.channel],
                        channel=route.channel,
                        session_id=sid,
                    )
                except Exception:
                    pass
        except Exception:
            logger.exception(
                "WebSessionBridge.on_turn_complete: failed to sync route=%s",
                getattr(route, "route_key", lambda: "?")(),
            )

    def on_turn_complete_simple(
        self,
        channel: str,
        scope_type: str,
        scope_id: str,
        user_text: str,
        llm_text: str,
        adata: Any = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """String-based overload for channels that don't use ConversationRoute.

        Feishu, QQ and iMessage channels pass routing info as plain strings.
        """
        try:
            sid = _route_to_web_session_id(channel, scope_type, scope_id, thread_id)
            web_session = self._sm.get_or_create(sid)
            tagged_user = f"[{channel}] {user_text}"
            web_session.add_message("user", tagged_user)
            if llm_text:
                web_session.add_message("assistant", llm_text)
            if adata is not None:
                self._sm.set_shared_adata(adata)
            logger.debug(
                "WebSessionBridge: synced turn for channel=%s scope=%s:%s → web_session=%s",
                channel, scope_type, scope_id, sid,
            )
            if self._memory_store is not None and llm_text:
                try:
                    self._memory_store.create_document(
                        title=user_text[:120],
                        content=f"**User:** {user_text}\n\n**Assistant:** {llm_text}",
                        tags=[channel],
                        channel=channel,
                        session_id=sid,
                    )
                except Exception:
                    pass
        except Exception:
            logger.exception(
                "WebSessionBridge.on_turn_complete_simple: failed for channel=%s",
                channel,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def list_web_sessions(self) -> list[dict]:
        """Return summaries of all web sessions (for /api/gateway/sessions)."""
        try:
            return self._sm.list_sessions()
        except Exception:
            return []
