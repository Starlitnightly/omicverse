"""Gateway primitives (routing/session keys) for Jarvis channels."""

from .routing import GatewaySessionRegistry, SessionKey
from .web_bridge import WebSessionBridge

__all__ = ["GatewaySessionRegistry", "SessionKey", "WebSessionBridge"]

