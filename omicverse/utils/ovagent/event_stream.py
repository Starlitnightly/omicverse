"""RuntimeEventEmitter — unified observability surface for OVAgent runtime.

Routes all runtime state through structured logger sinks and the harness
event stream.  Print output is **not** part of this contract — it is
non-authoritative and optional.

Observability contract
----------------------
* Every dispatch, result, retry, and completion event is emitted through
  **both** the Python logger and the harness event callback (when present).
* Consumers of runtime state must read from the logger or event stream,
  never from stdout.
* CLI banners (bootstrap.py, auth.py) remain explicitly scoped as
  non-authoritative human feedback and are NOT routed through this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

from ..harness import build_stream_event

if TYPE_CHECKING:
    from ..harness import RunTraceRecorder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RuntimeEventEmitter
# ---------------------------------------------------------------------------


class RuntimeEventEmitter:
    """Emit structured runtime events through logger + harness event stream.

    This is the single observability surface for the agentic turn loop,
    tool dispatch, subagent execution, and completion feedback.

    Parameters
    ----------
    recorder : RunTraceRecorder, optional
        Harness trace recorder.  Events are appended to the trace when
        provided.
    event_callback : callable, optional
        Async callback that receives harness stream events (dict).
    source : str
        Label identifying the emitting subsystem (``"turn_loop"``,
        ``"subagent"``, ``"tool_runtime"``).
    """

    def __init__(
        self,
        *,
        recorder: Optional["RunTraceRecorder"] = None,
        event_callback: Optional[
            Callable[..., Coroutine[Any, Any, None]]
        ] = None,
        source: str = "runtime",
    ) -> None:
        self._recorder = recorder
        self._event_callback = event_callback
        self._source = source

    # ------------------------------------------------------------------
    # Core emit
    # ------------------------------------------------------------------

    async def emit(
        self,
        event_type: str,
        content: Any,
        *,
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
        step_id: str = "",
        category: str = "",
        latency_ms: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit a structured event through all sinks.

        1. Always log via the module logger.
        2. Append to trace recorder when available.
        3. Forward to harness event callback when available.
        """
        event = build_stream_event(
            event_type,
            content,
            turn_id=turn_id,
            trace_id=trace_id,
            session_id=session_id,
            step_id=step_id,
            category=category,
            latency_ms=latency_ms,
        )
        if extra:
            event.update(extra)

        # 1. Structured log
        logger.info(
            "runtime_event source=%s type=%s category=%s content=%s",
            self._source,
            event_type,
            category or "-",
            _summarize(content),
        )

        # 2. Trace recorder
        if self._recorder is not None:
            self._recorder.add_event(event)

        # 3. Harness event callback
        if self._event_callback is not None:
            await self._event_callback(event)

    # ------------------------------------------------------------------
    # Convenience methods — turn lifecycle
    # ------------------------------------------------------------------

    async def turn_started(
        self,
        turn: int,
        max_turns: int,
        tool_choice: str,
        *,
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> None:
        """Emit a turn-started event."""
        logger.info(
            "turn_started turn=%d/%d tool_choice=%s",
            turn + 1,
            max_turns,
            tool_choice,
        )
        await self.emit(
            "status",
            {
                "turn": turn + 1,
                "max_turns": max_turns,
                "tool_choice": tool_choice,
            },
            turn_id=turn_id,
            trace_id=trace_id,
            session_id=session_id,
            category="turn_lifecycle",
        )

    async def turn_cancelled(
        self,
        phase: str,
        *,
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> None:
        """Emit a cancellation event."""
        logger.info("turn_cancelled phase=%s", phase)
        await self.emit(
            "done",
            f"Cancelled ({phase})",
            turn_id=turn_id,
            trace_id=trace_id,
            session_id=session_id,
            category="lifecycle",
            extra={"cancelled": True},
        )

    async def max_turns_reached(
        self,
        max_turns: int,
        *,
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> None:
        """Emit a max-turns-reached event."""
        logger.warning("max_turns_reached max_turns=%d", max_turns)
        await self.emit(
            "done",
            f"Reached max turns ({max_turns})",
            turn_id=turn_id,
            trace_id=trace_id,
            session_id=session_id,
            category="lifecycle",
            extra={"max_turns": True},
        )

    # ------------------------------------------------------------------
    # Convenience methods — tool dispatch
    # ------------------------------------------------------------------

    async def tool_dispatched(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        batch_id: str = "",
        parallel: bool = False,
        step_id: str = "",
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> None:
        """Emit a tool-dispatch event."""
        logger.info(
            "tool_dispatched name=%s args=[%s] batch=%s parallel=%s",
            name,
            ", ".join(f"{k}=" for k in arguments),
            batch_id,
            parallel,
        )

    async def tool_completed(
        self,
        name: str,
        *,
        status: str = "success",
        description: str = "",
        step_id: str = "",
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> None:
        """Emit a tool-completion event."""
        logger.info(
            "tool_completed name=%s status=%s description=%s",
            name,
            status,
            description[:120] if description else "-",
        )

    async def execution_completed(
        self,
        description: str,
        *,
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> None:
        """Emit an execute_code completion event."""
        logger.info("execution_completed description=%s", description[:120])

    async def delegation_completed(
        self,
        agent_type: str,
        task: str = "",
        *,
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> None:
        """Emit a delegation-started/completed event."""
        logger.info(
            "delegation agent_type=%s task=%s",
            agent_type,
            task[:80] if task else "-",
        )

    async def task_finished(
        self,
        summary: str,
        *,
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> None:
        """Emit a task-finished event."""
        logger.info("task_finished summary=%s", summary[:120])

    # ------------------------------------------------------------------
    # Convenience methods — subagent lifecycle
    # ------------------------------------------------------------------

    async def subagent_turn(
        self,
        agent_type: str,
        turn: int,
        max_turns: int,
    ) -> None:
        """Log a subagent turn progression."""
        logger.info(
            "subagent_turn agent_type=%s turn=%d/%d",
            agent_type,
            turn + 1,
            max_turns,
        )

    async def subagent_tool(
        self,
        agent_type: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Log a subagent tool dispatch."""
        logger.info(
            "subagent_tool agent_type=%s tool=%s args=[%s]",
            agent_type,
            tool_name,
            ", ".join(f"{k}=" for k in arguments),
        )

    async def subagent_finished(
        self,
        agent_type: str,
        summary: str,
    ) -> None:
        """Log subagent completion."""
        logger.info(
            "subagent_finished agent_type=%s summary=%s",
            agent_type,
            summary[:120],
        )

    # ------------------------------------------------------------------
    # Convenience — conversation log
    # ------------------------------------------------------------------

    def conversation_log_saved(self, path: str) -> None:
        """Log that a conversation log was persisted (non-critical)."""
        logger.info("conversation_log_saved path=%s", path)

    # ------------------------------------------------------------------
    # Convenience — agent response
    # ------------------------------------------------------------------

    async def agent_response(
        self,
        content: str,
        *,
        turn_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> None:
        """Log an assistant text response (no tool calls)."""
        logger.info(
            "agent_response content=%s",
            content[:200] if content else "-",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize(content: Any, max_len: int = 120) -> str:
    """Return a short string summary of *content* for log lines."""
    if content is None:
        return "-"
    if isinstance(content, str):
        return content[:max_len]
    if isinstance(content, dict):
        keys = ", ".join(content.keys())
        return f"{{{keys}}}"[:max_len]
    return str(content)[:max_len]
