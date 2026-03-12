from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..gateway.routing import SessionKey


@dataclass(frozen=True)
class ConversationRoute:
    channel: str
    scope_type: str
    scope_id: str
    thread_id: Optional[str] = None
    sender_id: Optional[str] = None

    def route_key(self) -> str:
        return self.to_session_key().as_key()

    @property
    def conversation_kind(self) -> str:
        if self.thread_id:
            return "thread"
        if self.scope_type in {"dm", "direct", "private", "p2p"}:
            return "direct"
        return "group"

    @property
    def is_direct(self) -> bool:
        return self.conversation_kind == "direct"

    @property
    def is_group(self) -> bool:
        return self.conversation_kind == "group"

    @property
    def is_thread(self) -> bool:
        return self.conversation_kind == "thread"

    def to_session_key(self) -> SessionKey:
        return SessionKey(
            channel=self.channel,
            scope_type=self.scope_type,
            scope_id=self.scope_id,
            thread_id=self.thread_id,
        )


@dataclass(frozen=True)
class MessageEnvelope:
    route: ConversationRoute
    text: str
    sender_id: str
    sender_username: Optional[str] = None
    message_id: Optional[str] = None
    trigger: str = "message"
    explicit_trigger: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    should_ignore: bool = False
    should_ack: bool = False
    should_start: bool = False
    should_queue: bool = False
    should_quick_chat: bool = False
    reason: str = ""


@dataclass(frozen=True)
class DeliveryEvent:
    route: ConversationRoute
    kind: str
    text: str = ""
    text_format: str = "plain"
    mode: str = "send"
    target: Optional[str] = None
    binary: Optional[bytes] = None
    filename: Optional[str] = None
    caption: str = ""
    controls: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeTaskState:
    route: ConversationRoute
    running: bool
    request: str = ""
    started_at: float = 0.0
    pending_count: int = 0
