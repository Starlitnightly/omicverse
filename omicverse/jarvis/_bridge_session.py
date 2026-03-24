from __future__ import annotations

from typing import Any, Optional


def _normalize_session_id(value: Any) -> str:
    if value is None:
        return ""
    session_id = str(value).strip()
    return session_id


def resolve_bridge_session_id(session: Any, *, create: bool = True) -> str:
    """Return the active agent session id for bridge routing.

    When ``create`` is True, this eagerly materializes the notebook session when
    the executor supports it so `get_prior_history*()` and `on_turn_complete*()`
    can route by the same stable session id instead of falling back to chat
    scope ids.
    """
    if session is None:
        return ""

    agent = getattr(session, "agent", None)
    if agent is not None:
        try:
            info = agent.get_current_session_info()
        except Exception:
            info = None
        if isinstance(info, dict):
            session_id = _normalize_session_id(info.get("session_id"))
            if session_id:
                return session_id

    try:
        status = session.kernel_status()
    except Exception:
        status = None
    if isinstance(status, dict):
        session_id = _normalize_session_id(status.get("session_id"))
        if session_id:
            return session_id

    executor = getattr(agent, "_notebook_executor", None)
    if executor is None:
        return ""

    current_session: Optional[dict] = getattr(executor, "current_session", None)
    if isinstance(current_session, dict):
        session_id = _normalize_session_id(current_session.get("session_id"))
        if session_id:
            return session_id

    ensure_session = getattr(executor, "_ensure_session", None)
    if create and callable(ensure_session):
        try:
            current_session = ensure_session()
        except Exception:
            current_session = getattr(executor, "current_session", None)
        if isinstance(current_session, dict):
            session_id = _normalize_session_id(current_session.get("session_id"))
            if session_id:
                return session_id

    start_new_session = getattr(executor, "_start_new_session", None)
    should_start_new_session = getattr(executor, "_should_start_new_session", None)
    if create and callable(start_new_session):
        should_start = True
        if callable(should_start_new_session):
            try:
                should_start = bool(should_start_new_session())
            except Exception:
                should_start = True
        elif getattr(executor, "current_session", None):
            should_start = False
        if should_start:
            try:
                start_new_session()
            except Exception:
                return ""
            current_session = getattr(executor, "current_session", None)
            if isinstance(current_session, dict):
                session_id = _normalize_session_id(current_session.get("session_id"))
                if session_id:
                    return session_id

    return ""
