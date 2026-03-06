"""
Channel routing and deterministic session keys (OpenClaw-style core).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SessionKey:
    channel: str
    scope_type: str   # dm/group/channel/thread/topic
    scope_id: str
    thread_id: Optional[str] = None

    def as_key(self) -> str:
        if self.thread_id:
            return f"{self.channel}:{self.scope_type}:{self.scope_id}:thread:{self.thread_id}"
        return f"{self.channel}:{self.scope_type}:{self.scope_id}"


class GatewaySessionRegistry:
    """
    Deterministic key -> synthetic user id mapper for SessionManager.

    SessionManager currently indexes sessions by int user_id. This registry keeps
    a stable mapping so we can route by channel/chat/thread without rewriting
    the full jarvis session stack.
    """

    def __init__(self, session_manager: Any) -> None:
        self._sm = session_manager
        self._key_to_uid: Dict[str, int] = {}

    def get_or_create(self, session_key: SessionKey) -> Any:
        key = session_key.as_key()
        uid = self._key_to_uid.get(key)
        if uid is None:
            uid = self._stable_uid(key)
            self._key_to_uid[key] = uid
        return self._sm.get_or_create(uid)

    @staticmethod
    def _stable_uid(key: str) -> int:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        # Keep it positive and bounded for readability.
        return int(digest[:15], 16) % 10_000_000_000

