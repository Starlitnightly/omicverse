"""Tests for the dependency-safe tool scheduler.

Validates that:
- Consecutive readonly tool calls are batched for parallel execution.
- Stateful tools break readonly batches and run serially.
- Exclusive tools always get their own isolated batch.
- Unknown tools default to stateful (conservative safety).
- Result ordering is deterministic regardless of parallel execution.
- Empty and single-call edge cases are handled correctly.
- Trace metadata (batch_id, parallel flag) is emitted correctly.
- The execute_batch helper preserves index ordering under concurrency.

These tests run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""

import asyncio
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Scheduler tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs (matches test_runtime_contracts.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SAVED = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils"]
}
for name in ["omicverse", "omicverse.utils"]:
    sys.modules.pop(name, None)

_ov_pkg = types.ModuleType("omicverse")
_ov_pkg.__path__ = [str(PACKAGE_ROOT)]
_ov_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = _ov_pkg

_utils_pkg = types.ModuleType("omicverse.utils")
_utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
_utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = _utils_pkg
_ov_pkg.utils = _utils_pkg

from omicverse.utils.ovagent.tool_scheduler import (
    ExecutionBatch,
    ScheduleResult,
    ScheduledCall,
    ToolScheduler,
    execute_batch,
)
from omicverse.utils.ovagent.tool_registry import (
    ApprovalClass,
    IsolationMode,
    OutputTier,
    ParallelClass,
    ToolMetadata,
    ToolRegistry,
    build_default_registry,
)

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeToolCall:
    """Minimal tool-call double with .name, .id, .arguments."""

    def __init__(self, name: str, tc_id: str = "", arguments: dict = None):
        self.name = name
        self.id = tc_id or f"call_{name}"
        self.arguments = arguments or {}


def _build_test_registry() -> ToolRegistry:
    """Build a registry with known tools for deterministic scheduling tests."""
    registry = build_default_registry()
    return registry


# ---------------------------------------------------------------------------
# ToolScheduler.schedule — partitioning tests
# ---------------------------------------------------------------------------


class TestSchedulePartitioning:
    """Test that schedule() correctly partitions tool calls into batches."""

    def setup_method(self):
        self.registry = _build_test_registry()
        self.scheduler = ToolScheduler(self.registry)

    def test_empty_tool_calls(self):
        result = self.scheduler.schedule([])
        assert result.total_calls == 0
        assert result.total_batches == 0
        assert result.batches == []
        assert not result.has_parallel

    def test_single_readonly_tool(self):
        calls = [FakeToolCall("Read")]
        result = self.scheduler.schedule(calls)
        assert result.total_calls == 1
        assert result.total_batches == 1
        batch = result.batches[0]
        assert batch.parallel is True  # single readonly → parallel batch
        assert batch.size == 1
        assert batch.calls[0].canonical_name == "Read"
        assert batch.calls[0].parallel_class == ParallelClass.readonly

    def test_consecutive_readonly_batched(self):
        """Multiple consecutive readonly tools should be in one parallel batch."""
        calls = [
            FakeToolCall("Read"),
            FakeToolCall("Glob"),
            FakeToolCall("Grep"),
        ]
        result = self.scheduler.schedule(calls)
        assert result.total_calls == 3
        assert result.total_batches == 1
        batch = result.batches[0]
        assert batch.parallel is True
        assert batch.size == 3
        assert result.has_parallel

    def test_stateful_breaks_batch(self):
        """A stateful tool should flush the readonly batch."""
        calls = [
            FakeToolCall("Read"),
            FakeToolCall("Glob"),
            FakeToolCall("Bash"),  # stateful
            FakeToolCall("Grep"),
        ]
        result = self.scheduler.schedule(calls)
        assert result.total_batches == 3
        # Batch 0: Read + Glob (parallel)
        assert result.batches[0].parallel is True
        assert result.batches[0].size == 2
        # Batch 1: Bash (serial)
        assert result.batches[1].parallel is False
        assert result.batches[1].size == 1
        assert result.batches[1].calls[0].canonical_name == "Bash"
        # Batch 2: Grep (parallel, single)
        assert result.batches[2].parallel is True
        assert result.batches[2].size == 1

    def test_exclusive_isolates(self):
        """Exclusive tools get their own batch and flush predecessors."""
        calls = [
            FakeToolCall("Read"),
            FakeToolCall("AskUserQuestion"),  # exclusive
            FakeToolCall("Glob"),
        ]
        result = self.scheduler.schedule(calls)
        assert result.total_batches == 3
        # Batch 0: Read (parallel)
        assert result.batches[0].parallel is True
        assert result.batches[0].calls[0].canonical_name == "Read"
        # Batch 1: AskUserQuestion (serial/exclusive)
        assert result.batches[1].parallel is False
        assert result.batches[1].calls[0].canonical_name == "AskUserQuestion"
        # Batch 2: Glob (parallel)
        assert result.batches[2].parallel is True
        assert result.batches[2].calls[0].canonical_name == "Glob"

    def test_all_stateful_all_serial(self):
        """All stateful tools should each get their own serial batch."""
        calls = [
            FakeToolCall("Bash"),
            FakeToolCall("Edit"),
            FakeToolCall("Write"),
        ]
        result = self.scheduler.schedule(calls)
        assert result.total_batches == 3
        for batch in result.batches:
            assert batch.parallel is False
            assert batch.size == 1
        assert not result.has_parallel

    def test_unknown_tool_defaults_stateful(self):
        """Tools not in the registry should default to stateful."""
        calls = [
            FakeToolCall("Read"),
            FakeToolCall("totally_unknown_tool"),
            FakeToolCall("Glob"),
        ]
        result = self.scheduler.schedule(calls)
        assert result.total_batches == 3
        # Unknown tool breaks the readonly batch
        assert result.batches[1].parallel is False
        assert result.batches[1].calls[0].canonical_name == "totally_unknown_tool"

    def test_mixed_sequence_preserves_order(self):
        """Complex mixed sequence should produce correct batch ordering."""
        calls = [
            FakeToolCall("Read"),       # readonly
            FakeToolCall("Grep"),       # readonly
            FakeToolCall("Bash"),       # stateful → flush
            FakeToolCall("Read"),       # readonly
            FakeToolCall("finish"),     # exclusive → flush
            FakeToolCall("Read"),       # readonly
        ]
        result = self.scheduler.schedule(calls)
        assert result.total_calls == 6
        # Batch 0: Read + Grep
        assert result.batches[0].parallel is True
        assert result.batches[0].size == 2
        # Batch 1: Bash
        assert result.batches[1].parallel is False
        # Batch 2: Read
        assert result.batches[2].parallel is True
        assert result.batches[2].size == 1
        # Batch 3: finish (exclusive)
        assert result.batches[3].parallel is False
        # Batch 4: Read
        assert result.batches[4].parallel is True

    def test_index_preserved_across_batches(self):
        """Each ScheduledCall should carry its original index."""
        calls = [
            FakeToolCall("Read"),
            FakeToolCall("Bash"),
            FakeToolCall("Glob"),
        ]
        result = self.scheduler.schedule(calls)
        all_indices = []
        for batch in result.batches:
            for sc in batch.calls:
                all_indices.append(sc.index)
        assert all_indices == [0, 1, 2]

    def test_batch_ids_unique(self):
        """Each batch should have a unique batch_id."""
        calls = [
            FakeToolCall("Read"),
            FakeToolCall("Bash"),
            FakeToolCall("Glob"),
        ]
        result = self.scheduler.schedule(calls)
        batch_ids = [b.batch_id for b in result.batches]
        assert len(set(batch_ids)) == len(batch_ids)

    def test_calls_carry_batch_id(self):
        """ScheduledCalls within a batch should carry the batch's ID."""
        calls = [FakeToolCall("Read"), FakeToolCall("Glob")]
        result = self.scheduler.schedule(calls)
        batch = result.batches[0]
        for sc in batch.calls:
            assert sc.batch_id == batch.batch_id

    def test_legacy_tool_names_resolved(self):
        """Legacy tool names should resolve through the registry."""
        calls = [FakeToolCall("inspect_data")]
        result = self.scheduler.schedule(calls)
        assert result.total_calls == 1
        sc = result.batches[0].calls[0]
        assert sc.canonical_name == "inspect_data"
        assert sc.parallel_class == ParallelClass.readonly


# ---------------------------------------------------------------------------
# ScheduleResult serialisation
# ---------------------------------------------------------------------------


class TestScheduleResultSerialization:
    """Test to_dict() serialisation for trace/debug coherence."""

    def test_to_dict_round_trip(self):
        registry = _build_test_registry()
        scheduler = ToolScheduler(registry)
        calls = [FakeToolCall("Read"), FakeToolCall("Bash")]
        result = scheduler.schedule(calls)
        d = result.to_dict()
        assert d["total_calls"] == 2
        assert d["total_batches"] == 2
        assert isinstance(d["batches"], list)
        for batch_dict in d["batches"]:
            assert "batch_id" in batch_dict
            assert "parallel" in batch_dict
            assert "size" in batch_dict
            assert "calls" in batch_dict


# ---------------------------------------------------------------------------
# execute_batch — async execution tests
# ---------------------------------------------------------------------------


class TestExecuteBatch:
    """Test the execute_batch helper for deterministic result ordering."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_serial_batch_single_call(self):
        sc = ScheduledCall(
            index=0,
            tool_call=FakeToolCall("Read"),
            canonical_name="Read",
            parallel_class=ParallelClass.readonly,
            batch_id="batch_1",
        )
        batch = ExecutionBatch(
            batch_id="batch_1", calls=[sc], parallel=False,
        )

        async def dispatch(s):
            return f"result_{s.canonical_name}"

        results = self._run(execute_batch(batch, dispatch))
        assert len(results) == 1
        assert results[0] == (0, "result_Read")

    def test_parallel_batch_deterministic_order(self):
        """Results from parallel execution must come back in index order."""
        calls = []
        for i, name in enumerate(["Read", "Glob", "Grep"]):
            calls.append(ScheduledCall(
                index=i,
                tool_call=FakeToolCall(name),
                canonical_name=name,
                parallel_class=ParallelClass.readonly,
                batch_id="batch_p",
            ))
        batch = ExecutionBatch(
            batch_id="batch_p", calls=calls, parallel=True,
        )

        # Simulate varying execution times to verify ordering
        async def dispatch(sc):
            # Reverse delay: last index finishes first
            await asyncio.sleep(0.01 * (3 - sc.index))
            return f"result_{sc.index}"

        results = self._run(execute_batch(batch, dispatch))
        assert len(results) == 3
        # Must be in original index order despite reverse completion
        assert results[0] == (0, "result_0")
        assert results[1] == (1, "result_1")
        assert results[2] == (2, "result_2")

    def test_empty_batch(self):
        batch = ExecutionBatch(
            batch_id="batch_e", calls=[], parallel=True,
        )

        async def dispatch(sc):
            return "should not be called"

        results = self._run(execute_batch(batch, dispatch))
        assert results == []

    def test_parallel_single_call_runs_serial(self):
        """A parallel batch with only one call should run serially."""
        sc = ScheduledCall(
            index=0,
            tool_call=FakeToolCall("Read"),
            canonical_name="Read",
            parallel_class=ParallelClass.readonly,
            batch_id="batch_s",
        )
        batch = ExecutionBatch(
            batch_id="batch_s", calls=[sc], parallel=True,
        )

        call_count = 0

        async def dispatch(s):
            nonlocal call_count
            call_count += 1
            return "ok"

        results = self._run(execute_batch(batch, dispatch))
        assert len(results) == 1
        assert call_count == 1


# ---------------------------------------------------------------------------
# Integration: scheduler + executor end-to-end
# ---------------------------------------------------------------------------


class TestSchedulerExecutorIntegration:
    """End-to-end test: schedule then execute all batches."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_full_pipeline_deterministic_results(self):
        """Schedule → execute → results arrive in original call order."""
        registry = _build_test_registry()
        scheduler = ToolScheduler(registry)

        calls = [
            FakeToolCall("Read"),       # 0: readonly
            FakeToolCall("Glob"),       # 1: readonly
            FakeToolCall("Bash"),       # 2: stateful
            FakeToolCall("Grep"),       # 3: readonly
            FakeToolCall("Read"),       # 4: readonly
        ]
        schedule = scheduler.schedule(calls)
        assert schedule.total_calls == 5

        execution_log = []

        async def dispatch(sc):
            execution_log.append(sc.canonical_name)
            return f"output_{sc.index}"

        all_results = []

        async def run_all():
            for batch in schedule.batches:
                batch_results = await execute_batch(batch, dispatch)
                all_results.extend(batch_results)

        self._run(run_all())

        # Results must be in original index order
        indices = [idx for idx, _ in all_results]
        assert indices == [0, 1, 2, 3, 4]

        # All dispatches happened
        assert len(execution_log) == 5

    def test_parallel_actually_concurrent(self):
        """Verify that parallel batches overlap in time."""
        registry = _build_test_registry()
        scheduler = ToolScheduler(registry)

        calls = [FakeToolCall("Read"), FakeToolCall("Glob")]
        schedule = scheduler.schedule(calls)
        assert schedule.total_batches == 1
        assert schedule.batches[0].parallel is True

        import time
        timestamps = {}

        async def dispatch(sc):
            timestamps[sc.canonical_name] = time.monotonic()
            await asyncio.sleep(0.05)
            return "ok"

        async def run():
            await execute_batch(schedule.batches[0], dispatch)

        self._run(run())

        # Both should have started within a very small window
        t_read = timestamps["Read"]
        t_glob = timestamps["Glob"]
        assert abs(t_read - t_glob) < 0.03, (
            f"Expected concurrent start, got delta={abs(t_read - t_glob):.3f}s"
        )


# ---------------------------------------------------------------------------
# has_parallel property
# ---------------------------------------------------------------------------


class TestHasParallel:
    def test_single_readonly_not_counted(self):
        """A single readonly call in a parallel batch is not 'has_parallel'."""
        registry = _build_test_registry()
        scheduler = ToolScheduler(registry)
        result = scheduler.schedule([FakeToolCall("Read")])
        assert not result.has_parallel

    def test_two_readonly_is_parallel(self):
        registry = _build_test_registry()
        scheduler = ToolScheduler(registry)
        result = scheduler.schedule([FakeToolCall("Read"), FakeToolCall("Glob")])
        assert result.has_parallel


# ---------------------------------------------------------------------------
# Scheduler contract: the acceptance criterion test
# ---------------------------------------------------------------------------


class TestSchedulerContract:
    """Validates the acceptance criterion:

    Turn processing no longer executes every tool call in a naive serial
    loop; scheduler policy metadata determines which calls may batch in
    parallel; result ordering and finish semantics remain deterministic;
    traces and tool-result messages remain coherent under concurrency.
    """

    def test_readonly_tools_batched_not_serial(self):
        """Policy metadata drives batching: readonly → parallel batch."""
        registry = _build_test_registry()
        scheduler = ToolScheduler(registry)
        calls = [FakeToolCall("Read"), FakeToolCall("Glob"), FakeToolCall("Grep")]
        schedule = scheduler.schedule(calls)

        # NOT a naive serial loop: all three in one parallel batch
        assert schedule.total_batches == 1
        assert schedule.batches[0].parallel is True
        assert schedule.batches[0].size == 3

    def test_stateful_forces_serial(self):
        """Stateful tools are correctly serialized by policy."""
        registry = _build_test_registry()
        scheduler = ToolScheduler(registry)
        calls = [FakeToolCall("Bash"), FakeToolCall("Edit")]
        schedule = scheduler.schedule(calls)
        assert all(not b.parallel for b in schedule.batches)

    def test_deterministic_result_ordering(self):
        """Results always come back in original call order."""
        registry = _build_test_registry()
        scheduler = ToolScheduler(registry)
        calls = [
            FakeToolCall("Read"),
            FakeToolCall("Glob"),
            FakeToolCall("Bash"),
            FakeToolCall("Grep"),
        ]
        schedule = scheduler.schedule(calls)

        async def dispatch(sc):
            # Simulate reversed completion for parallel calls
            if sc.parallel_class == ParallelClass.readonly:
                await asyncio.sleep(0.01 * (4 - sc.index))
            return f"r{sc.index}"

        async def run():
            all_results = []
            for batch in schedule.batches:
                batch_results = await execute_batch(batch, dispatch)
                all_results.extend(batch_results)
            return all_results

        results = asyncio.run(run())
        indices = [idx for idx, _ in results]
        assert indices == [0, 1, 2, 3], f"Expected [0,1,2,3], got {indices}"

    def test_trace_metadata_coherent(self):
        """Each scheduled call carries batch_id and parallel_class for traces."""
        registry = _build_test_registry()
        scheduler = ToolScheduler(registry)
        calls = [FakeToolCall("Read"), FakeToolCall("Bash")]
        schedule = scheduler.schedule(calls)

        for batch in schedule.batches:
            for sc in batch.calls:
                assert sc.batch_id == batch.batch_id
                assert isinstance(sc.parallel_class, ParallelClass)

        # Serialise for trace inspection
        d = schedule.to_dict()
        assert d["total_calls"] == 2
        assert len(d["batches"]) == 2
        for bd in d["batches"]:
            assert "batch_id" in bd
            assert "parallel" in bd
            for cd in bd["calls"]:
                assert "parallel_class" in cd
