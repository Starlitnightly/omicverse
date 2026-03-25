"""Structured event stream for OVAgent runtime observability.

Provides an ``EventBus`` that wraps the ``Reporter`` protocol and routes
turn/tool/recovery/init status through structured events instead of ad-hoc
``print()`` calls.  Every runtime subsystem emits events through this bus
so that consumers (Jarvis, Claw, harness, tests) can subscribe uniformly.

The ``EventBus`` preserves trace-output compatibility: when backed by a
``PrintReporter`` the output is identical to the previous ``print()``-based
diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..agent_reporter import AgentEvent, EventLevel, Reporter, PrintReporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EventBus — structured reporter wrapper
# ---------------------------------------------------------------------------


class EventBus:
    """Thin facade over a ``Reporter`` that adds category-aware helpers.

    Parameters
    ----------
    reporter : Reporter
        Underlying reporter implementation (PrintReporter, SilentReporter,
        CallbackReporter, or any user-supplied implementation).

    The bus exposes convenience methods for common runtime events so that
    call sites don't need to construct ``AgentEvent`` objects manually.
    All methods ultimately delegate to ``reporter.emit()``.
    """

    def __init__(self, reporter: Reporter) -> None:
        self._reporter = reporter

    # -- generic emission ---------------------------------------------------

    def emit(
        self,
        level: EventLevel,
        message: str,
        *,
        category: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a structured agent event."""
        self._reporter.emit(
            AgentEvent(level=level, message=message, category=category, data=data)
        )

    # -- init events --------------------------------------------------------

    def init(self, message: str, **data: Any) -> None:
        """Emit an initialisation status message (already formatted)."""
        self.emit(EventLevel.DEBUG, message, category="init", data=data or None)

    def init_warning(self, message: str, **data: Any) -> None:
        """Emit an initialisation warning."""
        self.emit(EventLevel.WARNING, message, category="init", data=data or None)

    def init_error(self, message: str, **data: Any) -> None:
        """Emit an initialisation error."""
        self.emit(EventLevel.ERROR, message, category="init", data=data or None)

    # -- turn lifecycle events ----------------------------------------------

    def turn_start(self, turn: int, max_turns: int) -> None:
        """Emit turn-start banner."""
        self.emit(
            EventLevel.DEBUG,
            f"   \U0001f504 Turn {turn}/{max_turns}",
            category="turn",
            data={"turn": turn, "max_turns": max_turns},
        )

    def turn_cancelled(self, reason: str) -> None:
        """Emit cancellation notice."""
        self.emit(
            EventLevel.DEBUG,
            f"   \u26d4 {reason}",
            category="turn",
            data={"cancelled": True, "reason": reason},
        )

    def turn_max_reached(self, max_turns: int) -> None:
        """Emit max-turns warning."""
        self.emit(
            EventLevel.DEBUG,
            f"   \u26a0\ufe0f  Max turns ({max_turns}) reached, returning current result",
            category="turn",
            data={"max_turns": max_turns},
        )

    # -- tool events --------------------------------------------------------

    def tool_dispatch(self, tool_name: str, arg_keys: list[str]) -> None:
        """Emit tool dispatch notice."""
        self.emit(
            EventLevel.DEBUG,
            f"   \U0001f527 Tool: {tool_name}({', '.join(f'{k}=' for k in arg_keys)})",
            category="tool",
            data={"tool_name": tool_name, "arg_keys": arg_keys},
        )

    def tool_result(self, tool_name: str, description: str) -> None:
        """Emit successful tool result."""
        self.emit(
            EventLevel.DEBUG,
            f"      \u2705 {description}",
            category="tool",
            data={"tool_name": tool_name},
        )

    def tool_finished(self, summary: str) -> None:
        """Emit finish-tool acknowledgement."""
        self.emit(
            EventLevel.DEBUG,
            f"   \u2705 Finished: {summary}",
            category="tool",
            data={"finished": True, "summary": summary},
        )

    def tool_delegated(self, agent_type: str, task: str) -> None:
        """Emit subagent delegation notice."""
        self.emit(
            EventLevel.DEBUG,
            f"   -> Delegating to {agent_type} subagent: {task[:80]}...",
            category="tool",
            data={"agent_type": agent_type},
        )

    # -- subagent events ----------------------------------------------------

    def subagent_turn(self, agent_type: str, turn: int, max_turns: int) -> None:
        """Emit subagent turn progress."""
        self.emit(
            EventLevel.DEBUG,
            f"      \U0001f504 [{agent_type}] Turn {turn}/{max_turns}",
            category="subagent",
            data={"agent_type": agent_type, "turn": turn, "max_turns": max_turns},
        )

    def subagent_tool(self, agent_type: str, tool_name: str, arg_keys: list[str]) -> None:
        """Emit subagent tool call."""
        self.emit(
            EventLevel.DEBUG,
            f"      \U0001f527 [{agent_type}] {tool_name}({', '.join(f'{k}=' for k in arg_keys)})",
            category="subagent",
            data={"agent_type": agent_type, "tool_name": tool_name},
        )

    def subagent_finished(self, agent_type: str, summary: str) -> None:
        """Emit subagent completion."""
        self.emit(
            EventLevel.DEBUG,
            f"      \u2705 [{agent_type}] Finished: {summary[:120]}",
            category="subagent",
            data={"agent_type": agent_type},
        )

    # -- llm / response events ----------------------------------------------

    def llm_response(self, summary: str) -> None:
        """Emit LLM text response snippet."""
        self.emit(
            EventLevel.DEBUG,
            f"   \U0001f4ac Agent response: {summary[:200]}",
            category="llm",
            data={"summary_length": len(summary)},
        )

    # -- recovery / execution events ----------------------------------------

    def recovery_attempt(self, phase: str, message: str) -> None:
        """Emit recovery/diagnosis attempt."""
        self.emit(
            EventLevel.DEBUG,
            message,
            category="recovery",
            data={"phase": phase},
        )

    def execution_status(self, message: str, **data: Any) -> None:
        """Emit execution-path status."""
        self.emit(
            EventLevel.DEBUG,
            message,
            category="execution",
            data=data or None,
        )

    # -- conversation log ---------------------------------------------------

    def conversation_log_saved(self, path: str) -> None:
        """Emit conversation log save notice."""
        self.emit(
            EventLevel.DEBUG,
            f"   \U0001f4dd Conversation log saved: {path}",
            category="trace",
            data={"path": path},
        )

    # -- request lifecycle --------------------------------------------------

    def request_start(
        self,
        request: str,
        *,
        dataset_shape: Optional[tuple] = None,
    ) -> None:
        """Emit request-processing banner."""
        sep = "=" * 70
        lines = [
            f"\n{sep}",
            "\U0001f916 OmicVerse Agent Processing Request",
            sep,
            f'Request: "{request}"',
        ]
        if dataset_shape is not None:
            lines.append(f"Dataset: {dataset_shape[0]} cells \u00d7 {dataset_shape[1]} genes")
        else:
            lines.append("Dataset: None (knowledge query)")
        lines.append(f"{sep}\n")
        self.emit(
            EventLevel.DEBUG,
            "\n".join(lines),
            category="lifecycle",
            data={"request": request, "dataset_shape": dataset_shape},
        )

    def request_success(self) -> None:
        """Emit success banner."""
        sep = "=" * 70
        self.emit(
            EventLevel.DEBUG,
            f"\n{sep}\n\u2705 SUCCESS - Agentic loop completed!\n{sep}\n",
            category="lifecycle",
        )

    def request_error(self, error: str) -> None:
        """Emit error banner."""
        sep = "=" * 70
        self.emit(
            EventLevel.DEBUG,
            f"\n{sep}\n\u274c ERROR - Agentic loop failed: {error}\n{sep}\n",
            category="lifecycle",
            data={"error": error},
        )

    def mode_info(self, message: str) -> None:
        """Emit execution mode information."""
        self.emit(EventLevel.DEBUG, message, category="lifecycle")


def make_event_bus(reporter: Optional[Reporter] = None) -> EventBus:
    """Create an EventBus, defaulting to PrintReporter when none supplied."""
    return EventBus(reporter or PrintReporter())
