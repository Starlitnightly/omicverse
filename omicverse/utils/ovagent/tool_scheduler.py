"""ToolScheduler — dependency-safe parallel tool-call scheduling.

Accepts a batch of tool calls (from a single LLM response), consults
the ``ToolDispatchRegistry`` for per-tool policy metadata, partitions
calls into execution *waves*, and runs independent tools concurrently
within each wave while preserving ordering for dependent calls.

Event and result output is returned in the **original request order**
regardless of actual execution order, so downstream consumers and tests
see deterministic sequencing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence

from ..harness.tool_catalog import normalize_tool_name

if TYPE_CHECKING:
    from .tool_runtime import ToolDispatchRegistry, ToolRegistryEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ScheduledCall:
    """A single tool call annotated with scheduling metadata."""

    index: int               # position in the original batch
    tool_call: Any           # raw tool-call object from LLM response
    canonical_name: str      # normalize_tool_name result
    entry: Optional["ToolRegistryEntry"]  # registry entry (None if unknown)
    parallel_safe: bool      # can run concurrently with other safe calls
    depends_on_adata: bool   # needs current adata
    modifies_adata: bool     # returns updated adata
    is_finish: bool          # terminates the loop


@dataclass
class ScheduledResult:
    """Result for a single tool call, keyed by original index."""

    index: int
    tool_call: Any
    canonical_name: str
    result: Any
    elapsed_s: float = 0.0


@dataclass
class ExecutionWave:
    """A group of tool calls that execute together.

    If ``parallel`` is True, all calls in the wave run concurrently via
    ``asyncio.gather``.  Otherwise they run sequentially in index order.
    """

    calls: List[ScheduledCall]
    parallel: bool = False


@dataclass
class BatchResult:
    """Aggregate result of executing a full tool-call batch."""

    results: List[ScheduledResult]      # in original index order
    waves: List[ExecutionWave]          # execution plan used
    parallel_calls: int = 0             # how many calls ran in parallel
    sequential_calls: int = 0           # how many ran sequentially
    total_elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# ToolScheduler
# ---------------------------------------------------------------------------


class ToolScheduler:
    """Dependency-safe parallel scheduler for tool-call batches.

    The scheduler is stateless per-batch: call ``plan()`` to get an
    execution plan, then ``execute_batch()`` to run it.  Or use
    ``execute_batch()`` directly which plans and executes in one step.

    Parameters
    ----------
    registry : ToolDispatchRegistry
        The dispatch registry used to look up per-tool policy metadata.
    """

    def __init__(self, registry: "ToolDispatchRegistry") -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def classify(self, tool_calls: Sequence[Any]) -> List[ScheduledCall]:
        """Annotate each tool call with scheduling metadata."""
        classified: List[ScheduledCall] = []
        for i, tc in enumerate(tool_calls):
            canonical = normalize_tool_name(tc.name) or tc.name
            entry = self._registry.lookup(canonical)
            if entry is None:
                # Unknown tool — treat as sequential for safety
                classified.append(ScheduledCall(
                    index=i,
                    tool_call=tc,
                    canonical_name=canonical,
                    entry=None,
                    parallel_safe=False,
                    depends_on_adata=False,
                    modifies_adata=False,
                    is_finish=(canonical == "finish"),
                ))
            else:
                policy = entry.policy
                classified.append(ScheduledCall(
                    index=i,
                    tool_call=tc,
                    canonical_name=canonical,
                    entry=entry,
                    parallel_safe=policy.parallel_safe,
                    depends_on_adata=policy.needs_adata,
                    modifies_adata=policy.returns_adata,
                    is_finish=(canonical == "finish"),
                ))
        return classified

    def plan(self, tool_calls: Sequence[Any]) -> List[ExecutionWave]:
        """Partition tool calls into execution waves.

        Algorithm:
        1. Scan calls left-to-right.
        2. Accumulate parallel-safe, non-adata-modifying calls into a
           pending parallel wave.
        3. When a sequential call is encountered (not parallel_safe, or
           modifies adata, or is ``finish``), flush the pending parallel
           wave first, then emit the sequential call as its own wave.
        4. After the adata-modifying wave completes, subsequent calls
           that depend on adata must wait for it.

        This produces a correct topological ordering that captures
        parallelism for the common case (multiple read-only / info tools)
        while preserving strict sequencing for data-modifying or
        inherently-sequential tools.
        """
        classified = self.classify(tool_calls)
        if not classified:
            return []

        # Single call — no scheduling overhead
        if len(classified) == 1:
            return [ExecutionWave(calls=classified, parallel=False)]

        waves: List[ExecutionWave] = []
        pending_parallel: List[ScheduledCall] = []
        # Track whether a preceding call modifies adata — if so,
        # subsequent adata-dependent calls must be sequential.
        adata_dirty = False

        def _flush_parallel() -> None:
            nonlocal pending_parallel
            if not pending_parallel:
                return
            waves.append(ExecutionWave(
                calls=list(pending_parallel),
                parallel=len(pending_parallel) > 1,
            ))
            pending_parallel = []

        for call in classified:
            needs_sequential = (
                not call.parallel_safe
                or call.modifies_adata
                or call.is_finish
                or (call.depends_on_adata and adata_dirty)
            )

            if needs_sequential:
                _flush_parallel()
                waves.append(ExecutionWave(
                    calls=[call], parallel=False,
                ))
                if call.modifies_adata:
                    adata_dirty = True
            else:
                pending_parallel.append(call)

        _flush_parallel()
        return waves

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_batch(
        self,
        tool_calls: Sequence[Any],
        dispatcher: Callable[..., Any],
        current_adata: Any,
        *,
        on_before: Optional[Callable[..., Any]] = None,
        on_after: Optional[Callable[..., Any]] = None,
    ) -> BatchResult:
        """Plan and execute a tool-call batch.

        Parameters
        ----------
        tool_calls :
            Raw tool-call objects from the LLM response.
        dispatcher :
            ``async (tool_call, adata, request) -> result``.
            Typically ``ToolRuntime.dispatch_tool``.
        current_adata :
            The current dataset object, threaded through adata-modifying
            tools.
        on_before :
            Optional ``async (ScheduledCall) -> None`` hook called before
            each tool dispatch (for event emission).
        on_after :
            Optional ``async (ScheduledResult, updated_adata) -> None``
            hook called after each tool completes (for event emission).

        Returns
        -------
        BatchResult
            Results in original index order, plus execution statistics.
        """
        t_batch_start = time.monotonic()
        waves = self.plan(tool_calls)

        results: dict[int, ScheduledResult] = {}
        parallel_count = 0
        sequential_count = 0
        adata = current_adata

        for wave in waves:
            if wave.parallel and len(wave.calls) > 1:
                # -- parallel execution --
                parallel_count += len(wave.calls)
                wave_results = await self._execute_parallel(
                    wave.calls, dispatcher, adata, on_before, on_after,
                )
                for sr in wave_results:
                    results[sr.index] = sr
            else:
                # -- sequential execution --
                for call in wave.calls:
                    sequential_count += 1
                    sr, new_adata = await self._execute_single(
                        call, dispatcher, adata, on_before, on_after,
                    )
                    results[sr.index] = sr
                    if call.modifies_adata and isinstance(sr.result, dict) and "adata" in sr.result:
                        adata = sr.result["adata"]

        # Return results in original order
        ordered = [results[i] for i in sorted(results.keys())]
        total_elapsed = time.monotonic() - t_batch_start

        logger.info(
            "tool_scheduler_batch waves=%d parallel=%d sequential=%d "
            "elapsed_s=%.3f",
            len(waves), parallel_count, sequential_count, total_elapsed,
        )

        return BatchResult(
            results=ordered,
            waves=waves,
            parallel_calls=parallel_count,
            sequential_calls=sequential_count,
            total_elapsed_s=total_elapsed,
        )

    async def _execute_single(
        self,
        call: ScheduledCall,
        dispatcher: Callable[..., Any],
        adata: Any,
        on_before: Optional[Callable[..., Any]],
        on_after: Optional[Callable[..., Any]],
    ) -> tuple[ScheduledResult, Any]:
        """Execute a single tool call."""
        if on_before is not None:
            cb = on_before(call)
            if asyncio.iscoroutine(cb):
                await cb

        t0 = time.monotonic()
        result = await dispatcher(call.tool_call, adata)
        elapsed = time.monotonic() - t0

        sr = ScheduledResult(
            index=call.index,
            tool_call=call.tool_call,
            canonical_name=call.canonical_name,
            result=result,
            elapsed_s=elapsed,
        )

        new_adata = adata
        if call.modifies_adata and isinstance(result, dict) and "adata" in result:
            new_adata = result["adata"]

        if on_after is not None:
            cb = on_after(sr, new_adata)
            if asyncio.iscoroutine(cb):
                await cb

        return sr, new_adata

    async def _execute_parallel(
        self,
        calls: List[ScheduledCall],
        dispatcher: Callable[..., Any],
        adata: Any,
        on_before: Optional[Callable[..., Any]],
        on_after: Optional[Callable[..., Any]],
    ) -> List[ScheduledResult]:
        """Execute multiple parallel-safe calls concurrently.

        Events (on_before/on_after) are emitted in **original index
        order** even though execution happens concurrently.  We achieve
        this by:
        1. Emitting all on_before hooks in order before starting.
        2. Gathering results concurrently.
        3. Emitting all on_after hooks in order after completion.
        """
        # Phase 1: emit on_before in original order
        if on_before is not None:
            for call in sorted(calls, key=lambda c: c.index):
                cb = on_before(call)
                if asyncio.iscoroutine(cb):
                    await cb

        # Phase 2: dispatch concurrently
        async def _dispatch(call: ScheduledCall) -> ScheduledResult:
            t0 = time.monotonic()
            result = await dispatcher(call.tool_call, adata)
            elapsed = time.monotonic() - t0
            return ScheduledResult(
                index=call.index,
                tool_call=call.tool_call,
                canonical_name=call.canonical_name,
                result=result,
                elapsed_s=elapsed,
            )

        gathered = await asyncio.gather(
            *[_dispatch(c) for c in calls],
            return_exceptions=True,
        )

        # Handle exceptions — wrap in error results
        results: List[ScheduledResult] = []
        for call, res in zip(calls, gathered):
            if isinstance(res, BaseException):
                logger.error(
                    "tool_scheduler_parallel_error tool=%s error=%s",
                    call.canonical_name, res,
                )
                results.append(ScheduledResult(
                    index=call.index,
                    tool_call=call.tool_call,
                    canonical_name=call.canonical_name,
                    result=f"Error executing {call.canonical_name}: {res}",
                    elapsed_s=0.0,
                ))
            else:
                results.append(res)

        # Phase 3: emit on_after in original index order
        results.sort(key=lambda r: r.index)
        if on_after is not None:
            for sr in results:
                cb = on_after(sr, adata)
                if asyncio.iscoroutine(cb):
                    await cb

        return results
