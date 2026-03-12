from __future__ import annotations

from typing import Any

from ..gateway.routing import GatewaySessionRegistry
from .models import ConversationRoute


class MessageRouter:
    """Resolve runtime conversation routes into Jarvis sessions."""

    def __init__(self, session_manager: Any) -> None:
        self._registry = GatewaySessionRegistry(session_manager)

    def get_session(self, route: ConversationRoute) -> Any:
        return self._registry.get_or_create(route.to_session_key())
