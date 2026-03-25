"""Dependency-safe tool scheduler for OVAgent turn execution.

Partitions an ordered sequence of tool calls into execution batches
based on ``ParallelClass`` policy metadata from the ``ToolRegistry``.
Read-only tool calls that appear consecutively may execute concurrently
within a single batch, while stateful, approval-gated, and
isolation-sensitive tools are serialized into their own batches.

Contract
--------
* **Input**: ordered list of tool calls (as received from the LLM).
* **Output**: ordered list of ``ExecutionBatch`` objects.
* **Deterministic ordering**: results are always returned in the
  original call order regardless of which calls ran in parallel.
* **Trace coherence**: each tool call retains its position index so
  that the caller can emit step/event traces in the correct sequence.

This module is the stable scheduling seam between the turn loop
(``TurnController``) and tool dispatch (``ToolRuntime``).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional, Sequence, Tuple

from .tool_registry import ParallelClass, ToolRegistry


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


def _make_batch_id() -> str:
    return "batch_" + uuid.uuid4().hex[:12]


@dataclass
class ScheduledCall:
    """A single tool call annotated with its original position and policy."""

    index: int
    tool_call: Any  # provider-specific tool-call object (has .name, .id, .arguments)
    canonical_name: str
    parallel_class: ParallelClass
    batch_id: str = ""


@dataclass
class ExecutionBatch:
    """A group of tool calls that share an execution strategy.

    * ``parallel=True`` means all calls in the batch MAY run concurrently.
    * ``parallel=False`` means they MUST run sequentially (in list order).
    """

    batch_id: str
    calls: List[ScheduledCall]
    parallel: bool

    @property
    def size(self) -> int:
        return len(self.calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "parallel": self.parallel,
            "size": self.size,
            "calls": [
                {
                    "index": c.index,
                    "canonical_name": c.canonical_name,
                    "parallel_class": c.parallel_class.value,
                }
                for c in self.calls
            ],
        }


@dataclass
class ScheduleResult:
    """Complete schedule for one turn's tool calls."""

    batches: List[ExecutionBatch]
    total_calls: int

    @property
    def total_batches(self) -> int:
        return len(self.batches)

    @property
    def has_parallel(self) -> bool:
        return any(b.parallel and b.size > 1 for b in self.batches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_batches": self.total_batches,
            "has_parallel": self.has_parallel,
            "batches": [b.to_dict() for b in self.batches],
        }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class ToolScheduler:
    """Partition tool calls into dependency-safe execution batches.

    The scheduler walks the ordered list of tool calls and groups
    consecutive ``ParallelClass.readonly`` calls into a single parallel
    batch.  Any ``stateful`` or ``exclusive`` call forces a batch
    boundary:

    * ``exclusive`` tools always run alone in their own batch.
    * ``stateful`` tools are placed in a serial (non-parallel) batch.
    * Unknown tools (not in the registry) default to ``stateful``
      to guarantee safety.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def _classify(self, tool_call: Any) -> Tuple[str, ParallelClass]:
        """Resolve name and parallel class for a tool call."""
        raw_name = tool_call.name
        canonical = self._registry.resolve_name(raw_name)
        if not canonical:
            # Fall back: treat as canonical if no alias found
            canonical = raw_name

        meta = self._registry.get(canonical)
        if meta is not None:
            return canonical, meta.parallel_class

        # Unknown tool → conservative default
        return canonical, ParallelClass.stateful

    def schedule(self, tool_calls: Sequence[Any]) -> ScheduleResult:
        """Partition *tool_calls* into execution batches.

        Returns a ``ScheduleResult`` whose batches preserve the original
        call ordering.  The caller should iterate batches in order and,
        within each parallel batch, may dispatch all calls concurrently
        while still collecting results in index order.
        """
        if not tool_calls:
            return ScheduleResult(batches=[], total_calls=0)

        batches: List[ExecutionBatch] = []
        pending_readonly: List[ScheduledCall] = []

        def _flush_readonly() -> None:
            if not pending_readonly:
                return
            bid = _make_batch_id()
            for sc in pending_readonly:
                sc.batch_id = bid
            batches.append(ExecutionBatch(
                batch_id=bid,
                calls=list(pending_readonly),
                parallel=True,
            ))
            pending_readonly.clear()

        for idx, tc in enumerate(tool_calls):
            canonical, par_class = self._classify(tc)
            sc = ScheduledCall(
                index=idx,
                tool_call=tc,
                canonical_name=canonical,
                parallel_class=par_class,
            )

            if par_class == ParallelClass.readonly:
                pending_readonly.append(sc)

            elif par_class == ParallelClass.exclusive:
                # Flush any pending readonly, then isolate this call
                _flush_readonly()
                bid = _make_batch_id()
                sc.batch_id = bid
                batches.append(ExecutionBatch(
                    batch_id=bid,
                    calls=[sc],
                    parallel=False,
                ))

            else:
                # stateful (or anything else): flush readonly, serial batch
                _flush_readonly()
                bid = _make_batch_id()
                sc.batch_id = bid
                batches.append(ExecutionBatch(
                    batch_id=bid,
                    calls=[sc],
                    parallel=False,
                ))

        # Flush trailing readonly calls
        _flush_readonly()

        return ScheduleResult(batches=batches, total_calls=len(tool_calls))


# ---------------------------------------------------------------------------
# Batch executor helper
# ---------------------------------------------------------------------------


async def execute_batch(
    batch: ExecutionBatch,
    dispatch_fn: Callable[..., Coroutine],
) -> List[Tuple[int, Any]]:
    """Execute a batch of tool calls and return results in index order.

    Parameters
    ----------
    batch : ExecutionBatch
        The batch to execute.
    dispatch_fn : async callable
        ``async def dispatch_fn(scheduled_call: ScheduledCall) -> Any``
        — the per-call dispatch function.

    Returns
    -------
    list of (index, result) tuples, sorted by the original call index.
    """
    if not batch.calls:
        return []

    if batch.parallel and len(batch.calls) > 1:
        # Concurrent execution with deterministic result ordering
        async def _run(sc: ScheduledCall) -> Tuple[int, Any]:
            result = await dispatch_fn(sc)
            return (sc.index, result)

        pairs = await asyncio.gather(*[_run(sc) for sc in batch.calls])
        # Sort by original index to guarantee deterministic ordering
        return sorted(pairs, key=lambda p: p[0])

    # Serial execution (single call, or non-parallel batch)
    results: List[Tuple[int, Any]] = []
    for sc in batch.calls:
        result = await dispatch_fn(sc)
        results.append((sc.index, result))
    return results


__all__ = [
    "ExecutionBatch",
    "ScheduleResult",
    "ScheduledCall",
    "ToolScheduler",
    "execute_batch",
]
