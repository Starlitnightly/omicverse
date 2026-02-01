"""
Structured event reporting for OmicVerse Agent (P1-1).

Replaces hard-coded ``print()`` calls with a pluggable Reporter interface.
Three built-in implementations:

* **PrintReporter** – reproduces current emoji-rich terminal output (default).
* **SilentReporter** – logs to ``logging`` only (tests / batch pipelines).
* **CallbackReporter** – forwards events to a user-supplied callable
  (Jupyter widgets, Web UI, progress bars, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class EventLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class AgentEvent:
    """A single structured agent event."""
    level: EventLevel
    message: str
    category: str = ""              # "init" / "execution" / "reflection" / "result"
    data: Optional[Dict[str, Any]] = field(default=None, repr=False)


@runtime_checkable
class Reporter(Protocol):
    """Minimal interface that all reporters must satisfy."""
    def emit(self, event: AgentEvent) -> None: ...


# ---------------------------------------------------------------------------
# Built-in reporters
# ---------------------------------------------------------------------------

_LEVEL_ICONS = {
    EventLevel.DEBUG:   "",
    EventLevel.INFO:    "\u2139\ufe0f ",
    EventLevel.WARNING: "\u26a0\ufe0f ",
    EventLevel.ERROR:   "\u274c ",
    EventLevel.SUCCESS: "\u2705 ",
}


class PrintReporter:
    """Default reporter: prints to stdout (matches legacy behaviour)."""

    def emit(self, event: AgentEvent) -> None:
        icon = _LEVEL_ICONS.get(event.level, "")
        print(f"{icon}{event.message}")


class SilentReporter:
    """Logs events via ``logging`` only – no stdout output."""

    def emit(self, event: AgentEvent) -> None:
        log_fn = getattr(logger, event.level.value, logger.info)
        log_fn("[%s] %s", event.category or "-", event.message)


class CallbackReporter:
    """Forwards every event to a user-supplied callback."""

    def __init__(self, callback: Callable[[AgentEvent], None]) -> None:
        self._cb = callback

    def emit(self, event: AgentEvent) -> None:
        self._cb(event)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_reporter(
    *,
    verbose: bool = True,
    callback: Optional[Callable[[AgentEvent], None]] = None,
    reporter: Optional[Reporter] = None,
) -> Reporter:
    """Create a reporter based on configuration.

    Priority: explicit *reporter* > *callback* > *verbose* flag.
    """
    if reporter is not None:
        return reporter
    if callback is not None:
        return CallbackReporter(callback)
    return PrintReporter() if verbose else SilentReporter()
