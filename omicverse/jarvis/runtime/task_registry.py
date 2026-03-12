from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import ConversationRoute, MessageEnvelope, RuntimeTaskState


@dataclass
class _RegistryEntry:
    route: ConversationRoute
    task: Optional[asyncio.Task] = None
    active_envelope: Optional[MessageEnvelope] = None
    started_at: float = 0.0
    pending: List[MessageEnvelope] = field(default_factory=list)


class TaskRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, _RegistryEntry] = {}

    def snapshot(self, route: ConversationRoute) -> RuntimeTaskState:
        entry = self._entries.get(route.route_key())
        running = bool(entry and entry.task and not entry.task.done())
        return RuntimeTaskState(
            route=route,
            running=running,
            request=(entry.active_envelope.text if entry and entry.active_envelope else ""),
            started_at=entry.started_at if entry else 0.0,
            pending_count=len(entry.pending) if entry else 0,
        )

    def task_for(self, route: ConversationRoute) -> Optional[asyncio.Task]:
        entry = self._entries.get(route.route_key())
        return entry.task if entry else None

    def start(self, route: ConversationRoute, *, envelope: MessageEnvelope, task: asyncio.Task) -> None:
        key = route.route_key()
        entry = self._entries.get(key)
        if entry is None:
            entry = _RegistryEntry(route=route)
            self._entries[key] = entry
        entry.route = route
        entry.task = task
        entry.active_envelope = envelope
        entry.started_at = time.time()

    def finish(self, route: ConversationRoute) -> None:
        key = route.route_key()
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.task = None
        entry.active_envelope = None
        entry.started_at = 0.0
        if not entry.pending:
            self._entries.pop(key, None)

    def enqueue(self, envelope: MessageEnvelope) -> int:
        route = envelope.route
        key = route.route_key()
        entry = self._entries.get(key)
        if entry is None:
            entry = _RegistryEntry(route=route)
            self._entries[key] = entry
        entry.pending.append(envelope)
        return len(entry.pending)

    def pop_pending(self, route: ConversationRoute) -> List[MessageEnvelope]:
        key = route.route_key()
        entry = self._entries.get(key)
        if entry is None:
            return []
        pending = list(entry.pending)
        entry.pending.clear()
        if entry.task is None:
            self._entries.pop(key, None)
        return pending

    async def cancel(self, route: ConversationRoute, *, timeout: float = 5.0) -> bool:
        key = route.route_key()
        entry = self._entries.get(key)
        if entry is None:
            return False
        entry.pending.clear()
        task = entry.task
        if task is None or task.done():
            if task is None:
                self._entries.pop(key, None)
            return False
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        return True
