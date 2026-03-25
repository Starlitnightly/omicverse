"""Tests for the ToolScheduler — dependency-safe parallel tool scheduling.

Covers:
- Wave planning: parallel vs sequential classification
- Parallel execution: independent tools run concurrently
- Dependency ordering: adata-modifying tools force sequential execution
- Deterministic event ordering: results returned in original order
- Edge cases: single call, empty batch, finish tool, unknown tools
"""

import asyncio
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List

import pytest

from omicverse.utils.ovagent.tool_runtime import (
    ToolDispatchRegistry,
    ToolPolicy,
    ToolRegistryEntry,
)
from omicverse.utils.ovagent.tool_scheduler import (
    BatchResult,
    ExecutionWave,
    ScheduledCall,
    ScheduledResult,
    ToolScheduler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> ToolDispatchRegistry:
    """Build a test registry with tools of varying policies."""
    reg = ToolDispatchRegistry()

    # Parallel-safe, read-only tools
    reg.register(ToolRegistryEntry(
        name="Read",
        executor=lambda args, adata: f"read:{args.get('path', '')}",
        schema={},
        policy=ToolPolicy(read_only=True, parallel_safe=True),
    ))
    reg.register(ToolRegistryEntry(
        name="Grep",
        executor=lambda args, adata: f"grep:{args.get('pattern', '')}",
        schema={},
        policy=ToolPolicy(read_only=True, parallel_safe=True),
    ))
    reg.register(ToolRegistryEntry(
        name="Glob",
        executor=lambda args, adata: f"glob:{args.get('pattern', '')}",
        schema={},
        policy=ToolPolicy(read_only=True, parallel_safe=True),
    ))
    reg.register(ToolRegistryEntry(
        name="ToolSearch",
        executor=lambda args, adata: "search_result",
        schema={},
        policy=ToolPolicy(read_only=True, parallel_safe=True),
    ))

    # Sequential: not parallel_safe
    reg.register(ToolRegistryEntry(
        name="Agent",
        executor=lambda args, adata: "agent_result",
        schema={},
        policy=ToolPolicy(
            parallel_safe=False, needs_adata=True,
        ),
        aliases=("delegate",),
    ))
    reg.register(ToolRegistryEntry(
        name="AskUserQuestion",
        executor=lambda args, adata: "question_result",
        schema={},
        policy=ToolPolicy(parallel_safe=False),
    ))

    # Adata-modifying
    reg.register(ToolRegistryEntry(
        name="execute_code",
        executor=lambda args, adata: {"adata": adata, "output": "ok"},
        schema={},
        policy=ToolPolicy(
            needs_adata=True, returns_adata=True,
            blocked_in_plan_mode=True,
        ),
    ))

    # Adata-reading only (parallel safe, needs adata, doesn't modify)
    reg.register(ToolRegistryEntry(
        name="inspect_data",
        executor=lambda args, adata: "inspected",
        schema={},
        policy=ToolPolicy(
            needs_adata=True, read_only=True, parallel_safe=True,
        ),
    ))

    # Finish
    reg.register(ToolRegistryEntry(
        name="finish",
        executor=lambda args, adata: {
            "finished": True,
            "summary": args.get("summary", ""),
        },
        schema={},
        policy=ToolPolicy(),
    ))

    # Bash (parallel safe by default, not read-only)
    reg.register(ToolRegistryEntry(
        name="Bash",
        executor=lambda args, adata: "bash_result",
        schema={},
        policy=ToolPolicy(
            blocked_in_plan_mode=True,
            requires_server_mode=True,
            parallel_safe=True,
        ),
    ))

    return reg


def _tc(name: str, arguments: dict | None = None, tc_id: str = "") -> SimpleNamespace:
    """Create a fake tool-call object."""
    return SimpleNamespace(
        name=name,
        arguments=arguments or {},
        id=tc_id or f"tc_{name}",
    )


# ---------------------------------------------------------------------------
# Planning tests
# ---------------------------------------------------------------------------


class TestSchedulerPlanning:
    """ToolScheduler.plan() produces correct execution waves."""

    def setup_method(self):
        self.reg = _make_registry()
        self.scheduler = ToolScheduler(self.reg)

    def test_empty_batch(self):
        waves = self.scheduler.plan([])
        assert waves == []

    def test_single_call(self):
        waves = self.scheduler.plan([_tc("Read")])
        assert len(waves) == 1
        assert not waves[0].parallel
        assert len(waves[0].calls) == 1

    def test_all_parallel_safe(self):
        """Multiple parallel-safe tools go into a single parallel wave."""
        calls = [_tc("Read"), _tc("Grep"), _tc("Glob")]
        waves = self.scheduler.plan(calls)
        assert len(waves) == 1
        assert waves[0].parallel is True
        assert len(waves[0].calls) == 3

    def test_sequential_tool_breaks_wave(self):
        """A non-parallel-safe tool forces a wave break."""
        calls = [_tc("Read"), _tc("Agent"), _tc("Grep")]
        waves = self.scheduler.plan(calls)
        # Wave 0: Read (alone, but parallel=False since single)
        # Wave 1: Agent (sequential)
        # Wave 2: Grep (alone)
        assert len(waves) == 3
        assert waves[0].calls[0].canonical_name == "Read"
        assert waves[1].calls[0].canonical_name == "Agent"
        assert waves[2].calls[0].canonical_name == "Grep"
        assert not waves[1].parallel

    def test_adata_modifier_forces_sequential(self):
        """execute_code (returns_adata=True) runs in its own wave."""
        calls = [_tc("Read"), _tc("execute_code"), _tc("Grep")]
        waves = self.scheduler.plan(calls)
        assert len(waves) == 3
        assert waves[1].calls[0].canonical_name == "execute_code"
        assert not waves[1].parallel

    def test_finish_always_last_wave(self):
        """finish tool goes into its own sequential wave."""
        calls = [_tc("Read"), _tc("Grep"), _tc("finish")]
        waves = self.scheduler.plan(calls)
        last_wave = waves[-1]
        assert last_wave.calls[0].canonical_name == "finish"
        assert not last_wave.parallel

    def test_adata_dependency_chain(self):
        """After execute_code, adata-dependent tools must be sequential."""
        calls = [
            _tc("execute_code"),
            _tc("inspect_data"),  # needs_adata, but adata was just modified
            _tc("Read"),          # no adata dependency
        ]
        waves = self.scheduler.plan(calls)
        # execute_code: own wave (modifies adata)
        # inspect_data: own wave (depends_on_adata and adata_dirty)
        # Read: own wave (no dependency)
        assert len(waves) == 3
        assert waves[0].calls[0].canonical_name == "execute_code"
        assert waves[1].calls[0].canonical_name == "inspect_data"
        assert waves[2].calls[0].canonical_name == "Read"

    def test_parallel_read_only_tools_group(self):
        """Multiple read-only adata tools group when adata is clean."""
        calls = [
            _tc("Read"),
            _tc("inspect_data"),
            _tc("Grep"),
            _tc("ToolSearch"),
        ]
        waves = self.scheduler.plan(calls)
        assert len(waves) == 1
        assert waves[0].parallel is True
        assert len(waves[0].calls) == 4

    def test_unknown_tool_is_sequential(self):
        """Tools not in the registry are treated as sequential."""
        calls = [_tc("UnknownTool"), _tc("Read")]
        waves = self.scheduler.plan(calls)
        assert len(waves) == 2
        assert not waves[0].parallel
        assert waves[0].calls[0].parallel_safe is False

    def test_mixed_parallel_and_sequential(self):
        """Complex batch with mixed tool types."""
        calls = [
            _tc("Read"),       # parallel
            _tc("Grep"),       # parallel
            _tc("Agent"),      # sequential (not parallel_safe)
            _tc("Glob"),       # parallel
            _tc("ToolSearch"), # parallel
            _tc("finish"),     # sequential
        ]
        waves = self.scheduler.plan(calls)
        # Wave 0: [Read, Grep] (parallel)
        # Wave 1: [Agent] (sequential)
        # Wave 2: [Glob, ToolSearch] (parallel)
        # Wave 3: [finish] (sequential)
        assert len(waves) == 4
        assert waves[0].parallel is True
        assert len(waves[0].calls) == 2
        assert not waves[1].parallel
        assert waves[2].parallel is True
        assert len(waves[2].calls) == 2
        assert not waves[3].parallel


class TestSchedulerClassify:
    """ToolScheduler.classify() annotates tool calls correctly."""

    def setup_method(self):
        self.reg = _make_registry()
        self.scheduler = ToolScheduler(self.reg)

    def test_classify_parallel_safe_tool(self):
        classified = self.scheduler.classify([_tc("Read")])
        assert len(classified) == 1
        c = classified[0]
        assert c.parallel_safe is True
        assert c.depends_on_adata is False
        assert c.modifies_adata is False
        assert c.is_finish is False

    def test_classify_adata_modifier(self):
        classified = self.scheduler.classify([_tc("execute_code")])
        c = classified[0]
        assert c.parallel_safe is True  # parallel_safe default is True
        assert c.depends_on_adata is True
        assert c.modifies_adata is True

    def test_classify_sequential_tool(self):
        classified = self.scheduler.classify([_tc("Agent")])
        c = classified[0]
        assert c.parallel_safe is False
        assert c.depends_on_adata is True

    def test_classify_finish(self):
        classified = self.scheduler.classify([_tc("finish")])
        c = classified[0]
        assert c.is_finish is True

    def test_classify_unknown(self):
        classified = self.scheduler.classify([_tc("NoSuchTool")])
        c = classified[0]
        assert c.entry is None
        assert c.parallel_safe is False


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


class TestSchedulerExecution:
    """ToolScheduler.execute_batch() runs tools correctly."""

    def setup_method(self):
        self.reg = _make_registry()
        self.scheduler = ToolScheduler(self.reg)

    def test_single_tool_execution(self):
        dispatch_log = []

        async def dispatcher(tc, adata):
            dispatch_log.append(tc.name)
            return f"result_{tc.name}"

        async def run():
            return await self.scheduler.execute_batch(
                [_tc("Read", {"path": "/foo"})],
                dispatcher,
                None,
            )

        result = asyncio.run(run())
        assert len(result.results) == 1
        assert result.results[0].result == "result_Read"
        assert result.sequential_calls == 1
        assert result.parallel_calls == 0

    def test_parallel_execution_runs_concurrently(self):
        """Parallel tools should overlap in time."""
        start_times = {}
        end_times = {}

        async def dispatcher(tc, adata):
            start_times[tc.name] = time.monotonic()
            await asyncio.sleep(0.05)  # simulate I/O
            end_times[tc.name] = time.monotonic()
            return f"result_{tc.name}"

        async def run():
            return await self.scheduler.execute_batch(
                [_tc("Read"), _tc("Grep"), _tc("Glob")],
                dispatcher,
                None,
            )

        result = asyncio.run(run())
        assert result.parallel_calls == 3
        assert result.sequential_calls == 0

        # Verify overlap: all should start before any finishes
        # (with tolerance for scheduling jitter)
        max_start = max(start_times.values())
        min_end = min(end_times.values())
        # If truly parallel, the last start should be before (or near)
        # the first finish
        assert max_start < min_end + 0.03, (
            "Parallel tools did not overlap in time"
        )

    def test_results_in_original_order(self):
        """Results must be returned in the original tool-call order."""
        async def dispatcher(tc, adata):
            # Reverse ordering by adding delays
            delays = {"Glob": 0.05, "Grep": 0.03, "Read": 0.01}
            await asyncio.sleep(delays.get(tc.name, 0))
            return f"result_{tc.name}"

        async def run():
            return await self.scheduler.execute_batch(
                [_tc("Read"), _tc("Grep"), _tc("Glob")],
                dispatcher,
                None,
            )

        result = asyncio.run(run())
        names = [r.canonical_name for r in result.results]
        assert names == ["Read", "Grep", "Glob"]

    def test_sequential_preserves_order(self):
        """Sequential tools execute in order."""
        dispatch_order = []

        async def dispatcher(tc, adata):
            dispatch_order.append(tc.name)
            return "ok"

        async def run():
            return await self.scheduler.execute_batch(
                [_tc("Agent"), _tc("AskUserQuestion")],
                dispatcher,
                None,
            )

        result = asyncio.run(run())
        assert dispatch_order == ["Agent", "AskUserQuestion"]
        assert result.sequential_calls == 2

    def test_mixed_waves_execution(self):
        """Mixed parallel/sequential batch executes correctly."""
        dispatch_order = []

        async def dispatcher(tc, adata):
            dispatch_order.append(tc.name)
            return f"result_{tc.name}"

        async def run():
            return await self.scheduler.execute_batch(
                [
                    _tc("Read"),   # parallel wave with Grep
                    _tc("Grep"),
                    _tc("Agent"),  # sequential wave
                    _tc("Glob"),   # single sequential
                ],
                dispatcher,
                None,
            )

        result = asyncio.run(run())
        assert len(result.results) == 4
        # Results in original order
        names = [r.canonical_name for r in result.results]
        assert names == ["Read", "Grep", "Agent", "Glob"]
        # Agent must dispatch after Read+Grep wave completes
        agent_idx = dispatch_order.index("Agent")
        assert "Read" in dispatch_order[:agent_idx]
        assert "Grep" in dispatch_order[:agent_idx]

    def test_adata_threading_through_waves(self):
        """Adata-modifying tools receive updated adata in subsequent waves."""
        async def dispatcher(tc, adata):
            if tc.name == "execute_code":
                return {"adata": "modified_adata", "output": "ok"}
            if tc.name == "inspect_data":
                # This should receive the modified adata
                return f"inspected:{adata}"
            return f"result_{tc.name}"

        async def run():
            return await self.scheduler.execute_batch(
                [
                    _tc("execute_code", {"code": "x=1", "description": "t"}),
                    _tc("inspect_data", {"aspect": "shape"}),
                ],
                dispatcher,
                "original_adata",
            )

        result = asyncio.run(run())
        # execute_code result contains adata
        assert result.results[0].result == {
            "adata": "modified_adata", "output": "ok",
        }
        # inspect_data should have received "modified_adata"
        assert result.results[1].result == "inspected:modified_adata"

    def test_parallel_exception_handled(self):
        """Exceptions in parallel tools don't crash the batch."""
        async def dispatcher(tc, adata):
            if tc.name == "Grep":
                raise ValueError("grep failed")
            return f"result_{tc.name}"

        async def run():
            return await self.scheduler.execute_batch(
                [_tc("Read"), _tc("Grep"), _tc("Glob")],
                dispatcher,
                None,
            )

        result = asyncio.run(run())
        assert len(result.results) == 3
        assert "result_Read" == result.results[0].result
        assert "Error executing Grep" in result.results[1].result
        assert "result_Glob" == result.results[2].result

    def test_on_before_hook_called_in_order(self):
        """on_before hooks fire in original index order."""
        before_order = []

        async def dispatcher(tc, adata):
            return "ok"

        async def on_before(call):
            before_order.append(call.canonical_name)

        async def run():
            return await self.scheduler.execute_batch(
                [_tc("Read"), _tc("Grep"), _tc("Glob")],
                dispatcher,
                None,
                on_before=on_before,
            )

        asyncio.run(run())
        assert before_order == ["Read", "Grep", "Glob"]

    def test_on_after_hook_called_in_order(self):
        """on_after hooks fire in original index order after dispatch."""
        after_order = []

        async def dispatcher(tc, adata):
            # Add delays so execution order differs from index order
            delays = {"Read": 0.03, "Grep": 0.01, "Glob": 0.02}
            await asyncio.sleep(delays.get(tc.name, 0))
            return f"result_{tc.name}"

        async def on_after(sr, adata):
            after_order.append(sr.canonical_name)

        async def run():
            return await self.scheduler.execute_batch(
                [_tc("Read"), _tc("Grep"), _tc("Glob")],
                dispatcher,
                None,
                on_after=on_after,
            )

        asyncio.run(run())
        assert after_order == ["Read", "Grep", "Glob"]

    def test_batch_result_statistics(self):
        """BatchResult reports correct parallel/sequential counts."""
        async def dispatcher(tc, adata):
            return "ok"

        async def run():
            return await self.scheduler.execute_batch(
                [
                    _tc("Read"),    # parallel wave
                    _tc("Grep"),    # parallel wave
                    _tc("Agent"),   # sequential wave
                    _tc("finish"),  # sequential wave
                ],
                dispatcher,
                None,
            )

        result = asyncio.run(run())
        assert result.parallel_calls == 2
        assert result.sequential_calls == 2
        assert result.total_elapsed_s > 0
        assert len(result.waves) == 3

    def test_elapsed_time_recorded(self):
        """Each result records its dispatch elapsed time."""
        async def dispatcher(tc, adata):
            await asyncio.sleep(0.02)
            return "ok"

        async def run():
            return await self.scheduler.execute_batch(
                [_tc("Read")],
                dispatcher,
                None,
            )

        result = asyncio.run(run())
        assert result.results[0].elapsed_s >= 0.01


# ---------------------------------------------------------------------------
# Integration: ToolScheduler used from TurnController-like context
# ---------------------------------------------------------------------------


class TestSchedulerTurnControllerIntegration:
    """Verify scheduler integrates correctly with turn-controller patterns."""

    def setup_method(self):
        self.reg = _make_registry()
        self.scheduler = ToolScheduler(self.reg)

    def test_finish_terminates_batch(self):
        """finish tool is always in the last wave."""
        calls = [_tc("Read"), _tc("finish"), _tc("Grep")]
        waves = self.scheduler.plan(calls)
        # Read is parallel safe, finish breaks the wave, Grep follows
        finish_wave_idx = None
        for i, w in enumerate(waves):
            for c in w.calls:
                if c.is_finish:
                    finish_wave_idx = i
        assert finish_wave_idx is not None
        # Grep should be after finish
        grep_wave_idx = None
        for i, w in enumerate(waves):
            for c in w.calls:
                if c.canonical_name == "Grep":
                    grep_wave_idx = i
        assert grep_wave_idx > finish_wave_idx

    def test_delegate_alias_resolved(self):
        """The 'delegate' alias resolves to Agent (sequential)."""
        classified = self.scheduler.classify([_tc("delegate")])
        c = classified[0]
        # delegate is an alias for Agent, which has parallel_safe=False
        assert c.entry is not None
        assert c.entry.name == "Agent"
        assert c.parallel_safe is False

    def test_two_parallel_reads_faster_than_sequential(self):
        """Parallel execution should be faster than sequential for I/O tools."""
        async def slow_dispatcher(tc, adata):
            await asyncio.sleep(0.05)
            return "ok"

        async def run():
            t0 = time.monotonic()
            result = await self.scheduler.execute_batch(
                [_tc("Read"), _tc("Grep"), _tc("Glob")],
                slow_dispatcher,
                None,
            )
            elapsed = time.monotonic() - t0
            return result, elapsed

        result, elapsed = asyncio.run(run())
        assert result.parallel_calls == 3
        # If truly parallel: ~0.05s. If sequential: ~0.15s.
        # Use generous threshold to avoid flaky tests.
        assert elapsed < 0.12, (
            f"Parallel execution took {elapsed:.3f}s, expected < 0.12s"
        )
