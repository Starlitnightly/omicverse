"""TurnController — agentic orchestration loop + follow-up gate.

Extracted from ``smart_agent.py``.  The TurnController owns the main
turn-by-turn loop (LLM call → tool dispatch → result append → repeat)
including stall detection, cancellation, and conversation logging.

Policy helpers (FollowUpGate, ConvergenceMonitor) and artifact persistence
helpers are imported from ``turn_followup`` and ``turn_artifacts`` respectively
and re-exported here for backward compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..harness import (
    RunTraceRecorder,
    build_stream_event,
    coerce_usage_payload,
)
from ..harness.runtime_state import runtime_state
from ..harness.tool_catalog import normalize_tool_name
from ..session_history import HistoryEntry

from .context_budget import (
    BudgetSliceType,
    ContextBudgetManager,
)
from .event_stream import RuntimeEventEmitter
from .tool_registry import OutputTier
from .tool_scheduler import ToolScheduler, execute_batch, ScheduledCall

# --- Extracted helpers (re-exported for backward compatibility) ---
from .turn_followup import FollowUpGate, ConvergenceMonitor  # noqa: F401
from .turn_artifacts import (
    persist_harness_history,
    save_conversation_log,
    slugify_for_filename,
    persist_tool_debug_output,
    persist_execute_code_source,
    persist_execute_code_stdout,
    log_tool_debug_output,
    log_execute_code_source,
    log_execute_code_stdout,
)

if TYPE_CHECKING:
    from .prompt_builder import PromptBuilder
    from .protocol import AgentContext
    from .tool_runtime import ToolRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TurnController
# ---------------------------------------------------------------------------

class TurnController:
    """Run the main agentic turn loop.

    Parameters
    ----------
    ctx : AgentContext
        The agent instance (accessed via protocol surface).
    prompt_builder : PromptBuilder
        Prompt construction helper.
    tool_runtime : ToolRuntime
        Tool dispatch hub.
    """

    def __init__(
        self,
        ctx: "AgentContext",
        prompt_builder: "PromptBuilder",
        tool_runtime: "ToolRuntime",
    ) -> None:
        self._ctx = ctx
        self._prompt_builder = prompt_builder
        self._tool_runtime = tool_runtime

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_tool_results(
        self, messages: list, tool_results: list
    ) -> None:
        """Append tool results to the conversation in the correct format."""
        if not tool_results:
            return

        provider_info = None
        llm = self._ctx._llm
        if llm:
            from ..model_config import ModelConfig, get_provider

            provider_info = get_provider(llm.config.provider)
            is_openai_responses = (
                llm.config.provider == "openai"
                and ModelConfig.requires_responses_api(llm.config.model)
            )
        else:
            is_openai_responses = False

        model_name = llm.config.model if llm else ""
        is_anthropic_wire = (
            (
                provider_info is not None
                and provider_info.wire_api.value == "anthropic"
            )
            or "claude" in model_name.lower()
        )

        if is_openai_responses:
            for tc_id, tc_name, tc_output in tool_results:
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc_id,
                        "output": tc_output,
                    }
                )
        elif is_anthropic_wire:
            content_blocks = []
            for tc_id, tc_name, tc_output in tool_results:
                content_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc_id,
                        "content": tc_output,
                    }
                )
            messages.append({"role": "user", "content": content_blocks})
        else:
            for tc_id, tc_name, tc_output in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": tc_name,
                        "content": tc_output,
                    }
                )

    # --- Backward-compatible delegation to turn_artifacts helpers ---

    def _persist_harness_history(self, request: str) -> None:
        """Persist the latest trace into session history when enabled."""
        persist_harness_history(self._ctx, request)

    @staticmethod
    def _save_conversation_log(messages: list) -> None:
        """Save the full conversation to a JSON file for debugging."""
        save_conversation_log(messages)

    @staticmethod
    def _slugify_for_filename(text: str, *, max_len: int = 48) -> str:
        return slugify_for_filename(text, max_len=max_len)

    def _persist_tool_debug_output(
        self,
        tool_name: str,
        output: str,
        *,
        turn_index: int,
        tool_index: int,
        description: str = "",
    ) -> Optional[Path]:
        return persist_tool_debug_output(
            self._ctx,
            tool_name,
            output,
            turn_index=turn_index,
            tool_index=tool_index,
            description=description,
        )

    def _persist_execute_code_source(
        self,
        code: str,
        *,
        turn_index: int,
        tool_index: int,
        description: str = "",
    ) -> Optional[Path]:
        return persist_execute_code_source(
            self._ctx,
            code,
            turn_index=turn_index,
            tool_index=tool_index,
            description=description,
        )

    def _persist_execute_code_stdout(
        self,
        stdout: str,
        *,
        turn_index: int,
        tool_index: int,
        description: str = "",
    ) -> Optional[Path]:
        return persist_execute_code_stdout(
            self._ctx,
            stdout,
            turn_index=turn_index,
            tool_index=tool_index,
            description=description,
        )

    @staticmethod
    def _log_tool_debug_output(
        *,
        tool_name: str,
        output: str,
        turn_index: int,
        tool_index: int,
        path: Optional[Path] = None,
        chunk_size: int = 4000,
    ) -> None:
        log_tool_debug_output(
            tool_name=tool_name,
            output=output,
            turn_index=turn_index,
            tool_index=tool_index,
            path=path,
            chunk_size=chunk_size,
        )

    @staticmethod
    def _log_execute_code_source(
        *,
        code: str,
        turn_index: int,
        tool_index: int,
        path: Optional[Path] = None,
        chunk_size: int = 4000,
    ) -> None:
        log_execute_code_source(
            code=code,
            turn_index=turn_index,
            tool_index=tool_index,
            path=path,
            chunk_size=chunk_size,
        )

    @staticmethod
    def _log_execute_code_stdout(
        *,
        stdout: str,
        turn_index: int,
        tool_index: int,
        path: Optional[Path] = None,
        chunk_size: int = 4000,
    ) -> None:
        log_execute_code_stdout(
            stdout=stdout,
            turn_index=turn_index,
            tool_index=tool_index,
            path=path,
            chunk_size=chunk_size,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_agentic_loop(
        self,
        request: str,
        adata: Any,
        event_callback=None,
        cancel_event=None,
        history=None,
        approval_handler=None,
        request_content=None,
    ) -> Any:
        """Execute the agentic loop: LLM decides tools to call iteratively."""
        ctx = self._ctx
        config = ctx._config
        max_turns = config.execution.max_agent_turns if config else 15
        workflow_config = getattr(
            getattr(getattr(ctx, "_ov_runtime", None), "workflow", None),
            "config",
            None,
        )
        if workflow_config is not None and getattr(
            workflow_config, "max_turns", 0
        ):
            max_turns = int(workflow_config.max_turns)

        recorder = RunTraceRecorder(
            request=request,
            model=ctx.model,
            provider=ctx.provider,
            session_id=ctx._get_harness_session_id(),
        )
        ctx._last_run_trace = recorder.trace
        final_summary = ""
        runtime_session_id = ctx._get_runtime_session_id()
        analysis_run = None

        if ctx._ov_runtime is not None:
            try:
                analysis_run = ctx._ov_runtime.start_analysis_run(
                    request=request,
                    model=ctx.model,
                    provider=ctx.provider,
                    session_id=recorder.trace.session_id,
                    metadata={
                        "provider": ctx.provider,
                        "runtime_session_id": runtime_session_id,
                    },
                )
                ctx._active_run_id = analysis_run.run_id
                recorder.add_artifact(
                    "analysis_run",
                    label=analysis_run.run_id,
                    path=str(
                        ctx._ov_runtime.run_store.root
                        / analysis_run.run_id
                    ),
                    description="OVAgent analysis run record",
                    metadata={"run_id": analysis_run.run_id},
                )
            except Exception as exc:
                logger.warning(
                    "Failed to start analysis run record: %s", exc
                )
                analysis_run = None

        def _finalize_analysis_run(status: str, summary: str) -> None:
            if analysis_run is None or ctx._ov_runtime is None:
                return
            try:
                ctx._ov_runtime.finish_analysis_run(
                    analysis_run.run_id,
                    status=status,
                    summary=summary,
                    trace_id=recorder.trace.trace_id,
                    artifacts=[
                        artifact.to_dict()
                        for artifact in recorder.trace.artifacts
                    ],
                    metadata={
                        "turn_id": recorder.trace.turn_id,
                        "session_id": recorder.trace.session_id,
                        "usage": coerce_usage_payload(ctx.last_usage),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to finalize analysis run %s: %s",
                    analysis_run.run_id,
                    exc,
                )

        def _sync_runtime_trace() -> None:
            runtime_summary = runtime_state.get_summary(
                runtime_session_id
            )
            recorder.trace.loaded_tools = list(
                runtime_summary.get("loaded_tools", [])
            )
            recorder.trace.plan_mode = bool(
                (runtime_summary.get("plan_mode") or {}).get(
                    "enabled", False
                )
            )
            recorder.trace.worktree = dict(
                runtime_summary.get("worktree") or {}
            )
            recorder.trace.task_ids = [
                task.get("task_id", "")
                for task in runtime_summary.get("tasks", [])
                if task.get("task_id")
            ]
            recorder.trace.question_ids = [
                question.get("question_id", "")
                for question in runtime_summary.get(
                    "pending_questions", []
                )
                if question.get("question_id")
            ]

        _sync_runtime_trace()

        emitter = RuntimeEventEmitter(
            recorder=recorder,
            event_callback=event_callback,
            source="turn_loop",
        )

        async def emit(event):
            recorder.add_event(event)
            if event_callback:
                await event_callback(event)

        if analysis_run is not None:
            await emit(
                build_stream_event(
                    "status",
                    {
                        "run_id": analysis_run.run_id,
                        "workflow_path": str(
                            ctx._ov_runtime.workflow.path
                        ),
                    },
                    turn_id=recorder.trace.turn_id,
                    trace_id=recorder.trace.trace_id,
                    session_id=recorder.trace.session_id,
                    category="runtime",
                )
            )

        # Build initial messages
        system_prompt = self._prompt_builder.build_agentic_system_prompt()
        session_id = ctx._get_harness_session_id()
        if ctx._session_history is not None and session_id:
            history_context = (
                ctx._session_history.build_context_for_llm(session_id)
            )
            if history_context:
                system_prompt += "\n\n" + history_context

        harness_config = getattr(config, "harness", None)
        if (
            ctx._trace_store is not None
            and harness_config is not None
            and getattr(
                harness_config,
                "include_recent_failures_in_prompt",
                False,
            )
        ):
            failures_context = (
                ctx._trace_store.build_context_for_prompt(
                    limit=getattr(
                        harness_config, "max_recent_failures", 3
                    ),
                )
            )
            if failures_context:
                system_prompt += "\n\n" + failures_context

        if (
            ctx._context_compactor is not None
            and ctx._context_compactor.needs_compaction(
                system_prompt, request
            )
        ):
            compaction = await ctx._context_compactor.compact_bundle(
                system_prompt
            )
            recorder.add_step(
                "context_compaction",
                summary="context compacted for handoff",
                data={
                    "original_tokens": compaction.original_tokens,
                    "compacted_tokens": compaction.compacted_tokens,
                    "summary": compaction.summary,
                },
            )
            if harness_config is not None and getattr(
                harness_config, "record_artifacts", False
            ):
                recorder.add_artifact(
                    "context_summary",
                    label="compacted_system_prompt",
                    description=compaction.summary[:240],
                    metadata={
                        "original_tokens": compaction.original_tokens,
                        "compacted_tokens": compaction.compacted_tokens,
                    },
                )
            system_prompt = compaction.handoff_text

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append(
            {
                "role": "user",
                "content": self._prompt_builder.build_initial_user_message(
                    request,
                    adata,
                    extra_content=request_content,
                ),
            },
        )

        # --- Token-aware context budget manager ---
        budget_model = (
            getattr(getattr(ctx._llm, "config", None), "model", None)
            or ctx.model
            or ""
        )
        budget_manager = ContextBudgetManager(model=budget_model)
        budget_manager.record(
            BudgetSliceType.system_prompt,
            system_prompt,
            content_key="system_prompt",
        )

        current_adata = adata
        completed_without_tool_calls = False
        meaningful_tool_call_seen = False
        no_tool_retry_count = 0
        convergence_monitor = ConvergenceMonitor(
            request
        )
        llm_model_name = (
            getattr(getattr(ctx._llm, "config", None), "model", None)
            or ctx.model
        )
        _is_anthropic_model = "claude" in str(llm_model_name).lower()
        max_no_tool_retries = 3 if _is_anthropic_model else 2
        chat_timeout = float(
            os.environ.get("OV_AGENT_CHAT_TIMEOUT_SECONDS", "120")
        )
        ctx._approval_handler = approval_handler

        def _is_cancelled():
            return cancel_event is not None and cancel_event.is_set()

        try:
            for turn in range(max_turns):
                # --- Cancel checkpoint: before LLM call ---
                if _is_cancelled():
                    recorder.add_step(
                        "status", summary="cancelled_before_llm"
                    )
                    recorder.finish(
                        status="cancelled",
                        result_summary="Cancelled before LLM call",
                        usage=ctx.last_usage,
                    )
                    _finalize_analysis_run(
                        "cancelled", "Cancelled before LLM call"
                    )
                    if ctx._trace_store is not None:
                        recorder.save(ctx._trace_store)
                    await emitter.turn_cancelled(
                        "before_llm_call",
                        turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id,
                        session_id=recorder.trace.session_id,
                    )
                    self._save_conversation_log(messages)
                    return current_adata

                tool_choice = FollowUpGate.select_tool_choice(
                    request=request,
                    adata=current_adata,
                    turn_index=turn,
                    had_meaningful_tool_call=meaningful_tool_call_seen,
                    forced_retry=no_tool_retry_count > 0,
                )
                if convergence_monitor.should_force_tool_choice():
                    tool_choice = "required"
                    convergence_monitor._force_execute_next = False
                    logger.info(
                        "convergence_force_tool_choice turn=%d",
                        turn + 1,
                    )

                await emitter.turn_started(
                    turn,
                    max_turns,
                    tool_choice,
                    turn_id=recorder.trace.turn_id,
                    trace_id=recorder.trace.trace_id,
                    session_id=recorder.trace.session_id,
                )
                t_llm_start = time.time()
                stall_retries = 0
                response = None
                while response is None:
                    try:
                        response = await asyncio.wait_for(
                            ctx._llm.chat(
                                messages,
                                tools=ctx._get_visible_agent_tools(),
                                tool_choice=tool_choice,
                            ),
                            timeout=chat_timeout + 15,
                        )
                    except (asyncio.TimeoutError, Exception) as exc:
                        is_timeout = isinstance(
                            exc, asyncio.TimeoutError
                        ) or "timeout" in str(exc).lower()
                        if is_timeout and stall_retries < 1:
                            stall_retries += 1
                            logger.warning(
                                "llm_chat_stall_retry attempt=%d turn=%d",
                                stall_retries,
                                turn + 1,
                            )
                            recorder.add_step(
                                "status",
                                summary=(
                                    f"llm_chat_stall_retry_{stall_retries}"
                                ),
                            )
                            continue
                        logger.error(
                            "llm_chat_stall_final turn=%d error=%s",
                            turn + 1,
                            exc,
                        )
                        recorder.add_step(
                            "error",
                            summary=(
                                f"LLM stall after "
                                f"{stall_retries} retries: {exc}"
                            ),
                        )
                        recorder.finish(
                            status="stall",
                            result_summary=f"LLM stall: {exc}",
                            usage=ctx.last_usage,
                        )
                        _finalize_analysis_run(
                            "stall", f"LLM stall: {exc}"
                        )
                        if ctx._trace_store is not None:
                            recorder.save(ctx._trace_store)
                        await emit(
                            build_stream_event(
                                "done",
                                (
                                    "LLM stall detected after tool "
                                    f"execution (retries={stall_retries})"
                                ),
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                session_id=recorder.trace.session_id,
                                category="lifecycle",
                            )
                            | {"stall": True, "error": str(exc)}
                        )
                        self._save_conversation_log(messages)
                        ctx._last_run_trace = recorder.trace
                        return current_adata

                logger.info(
                    "agentic_llm_call_done turn=%d/%d elapsed_s=%.2f "
                    "has_tool_calls=%s content_len=%d",
                    turn + 1,
                    max_turns,
                    time.time() - t_llm_start,
                    bool(response.tool_calls),
                    len(response.content or ""),
                )

                if response.usage:
                    ctx.last_usage = response.usage

                if response.raw_message:
                    if isinstance(response.raw_message, list):
                        messages.extend(response.raw_message)
                    else:
                        messages.append(response.raw_message)
                elif response.content:
                    messages.append(
                        {"role": "assistant", "content": response.content}
                    )

                # Text-only response
                if not response.tool_calls:
                    if response.content:
                        final_summary = (
                            response.content
                            if isinstance(response.content, str)
                            else str(response.content)
                        )
                        recorder.add_step(
                            "llm_chunk",
                            summary=final_summary[:200],
                        )
                        await emitter.agent_response(
                            final_summary,
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                        )
                        await emit(
                            build_stream_event(
                                "llm_chunk",
                                response.content,
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                session_id=recorder.trace.session_id,
                                category="assistant",
                            )
                        )
                    should_follow_up = (
                        FollowUpGate.should_continue_after_text(
                            request=request,
                            response_content=final_summary or "",
                            adata=current_adata,
                            had_meaningful_tool_call=(
                                meaningful_tool_call_seen
                            ),
                        )
                    )
                    logger.info(
                        "follow_up_gate turn=%d should_continue=%s "
                        "retry=%d/%d meaningful_tool=%s",
                        turn + 1,
                        should_follow_up,
                        no_tool_retry_count,
                        max_no_tool_retries,
                        meaningful_tool_call_seen,
                    )
                    if (
                        should_follow_up
                        and no_tool_retry_count < max_no_tool_retries
                    ):
                        no_tool_retry_count += 1
                        guidance = (
                            FollowUpGate.build_no_tool_follow_up(
                                request,
                                retry_count=no_tool_retry_count,
                                max_retries=max_no_tool_retries,
                            )
                        )
                        messages.append(
                            {"role": "user", "content": guidance}
                        )
                        recorder.add_step(
                            "status",
                            summary=(
                                "follow_up_required_after_"
                                "text_only_response"
                            ),
                            data={
                                "retry_count": no_tool_retry_count,
                                "max_retries": max_no_tool_retries,
                                "tool_choice": tool_choice,
                            },
                        )
                        await emit(
                            build_stream_event(
                                "status",
                                {
                                    "follow_up_required": True,
                                    "retry_count": no_tool_retry_count,
                                    "max_retries": max_no_tool_retries,
                                    "reason": (
                                        "action_request_text_only_response"
                                    ),
                                },
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                session_id=recorder.trace.session_id,
                                category="runtime",
                            )
                        )
                        continue
                    if (
                        should_follow_up
                        and not meaningful_tool_call_seen
                    ):
                        logger.warning(
                            "follow_up_exhausted turn=%d retries=%d",
                            turn + 1,
                            no_tool_retry_count,
                        )
                        recorder.add_step(
                            "status",
                            summary=(
                                "follow_up_exhausted_no_tool_calls"
                            ),
                        )
                        await emit(
                            build_stream_event(
                                "status",
                                {
                                    "follow_up_exhausted": True,
                                    "retry_count": no_tool_retry_count,
                                },
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                session_id=recorder.trace.session_id,
                                category="runtime",
                            )
                        )
                    completed_without_tool_calls = True
                    break

                # Process tool calls
                tool_results: List[tuple] = []
                finished = False
                no_tool_retry_count = 0

                # --- Schedule tool calls into batches ---
                scheduler = ToolScheduler(
                    self._tool_runtime.registry
                )
                schedule = scheduler.schedule(response.tool_calls)
                logger.info(
                    "tool_schedule turn=%d batches=%d parallel=%s "
                    "total_calls=%d",
                    turn + 1,
                    schedule.total_batches,
                    schedule.has_parallel,
                    schedule.total_calls,
                )

                # Pre-allocate result slots indexed by original position
                _ordered_results: List[Optional[tuple]] = [
                    None
                ] * schedule.total_calls
                # Map from call index → step_id for trace coherence
                _step_ids: Dict[int, str] = {}

                for batch in schedule.batches:
                    if _is_cancelled():
                        recorder.add_step(
                            "status",
                            summary="cancelled_before_tool_dispatch",
                        )
                        recorder.finish(
                            status="cancelled",
                            result_summary=(
                                "Cancelled before tool dispatch"
                            ),
                            usage=ctx.last_usage,
                        )
                        _finalize_analysis_run(
                            "cancelled",
                            "Cancelled before tool dispatch",
                        )
                        if ctx._trace_store is not None:
                            recorder.save(ctx._trace_store)
                        await emitter.turn_cancelled(
                            "before_tool_dispatch",
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                        )
                        self._save_conversation_log(messages)
                        return current_adata

                    # Emit item_started for each call in the batch
                    for sc in batch.calls:
                        tc = sc.tool_call
                        canonical = sc.canonical_name
                        tool_step_id = recorder.add_step(
                            "tool_call",
                            name=canonical or tc.name,
                            summary=f"{canonical or tc.name} dispatched",
                            data={
                                "arguments": tc.arguments,
                                "batch_id": batch.batch_id,
                                "parallel": batch.parallel,
                            },
                        )
                        _step_ids[sc.index] = tool_step_id
                        await emitter.tool_dispatched(
                            canonical or tc.name,
                            tc.arguments,
                            batch_id=batch.batch_id,
                            parallel=batch.parallel,
                            step_id=tool_step_id,
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                        )
                        await emit(
                            build_stream_event(
                                "item_started",
                                {
                                    "item_type": "tool_call",
                                    "name": canonical or tc.name,
                                    "batch_id": batch.batch_id,
                                    "parallel": batch.parallel,
                                },
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                step_id=tool_step_id,
                                session_id=recorder.trace.session_id,
                                category="item",
                            )
                        )
                        await emit(
                            build_stream_event(
                                "tool_call",
                                {
                                    "name": canonical or tc.name,
                                    "arguments": tc.arguments,
                                },
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                step_id=tool_step_id,
                                session_id=recorder.trace.session_id,
                                category="tool",
                            )
                        )

                    # Dispatch the batch
                    async def _dispatch_scheduled(
                        sc: ScheduledCall,
                    ) -> Any:
                        return await self._tool_runtime.dispatch_tool(
                            sc.tool_call, current_adata, request
                        )

                    batch_results = await execute_batch(
                        batch, _dispatch_scheduled
                    )

                    # Process results in original index order
                    for call_idx, result in batch_results:
                        sc = batch.calls[
                            call_idx - batch.calls[0].index
                        ]
                        tc = sc.tool_call
                        canonical = sc.canonical_name
                        tool_step_id = _step_ids[call_idx]
                        tool_index = sc.index + 1

                        debug_tool_output = None
                        if (
                            canonical != "finish"
                            and FollowUpGate.tool_counts_as_meaningful_progress(
                                canonical or tc.name
                            )
                        ):
                            meaningful_tool_call_seen = True

                        if (
                            canonical
                            in ("execute_code", "Agent", "delegate")
                            and isinstance(result, dict)
                            and "adata" in result
                        ):
                            current_adata = result["adata"]
                            tool_output = result.get(
                                "output", "Code executed."
                            )
                            debug_tool_output = result.get(
                                "debug_output", tool_output
                            )
                            if canonical == "execute_code":
                                code = tc.arguments.get("code", "")
                                stdout_text = str(result.get("stdout", "") or "")
                                description = tc.arguments.get(
                                    "description", "Code executed"
                                )
                                recorder.add_step(
                                    "code",
                                    name=canonical,
                                    summary=description,
                                    data={"code": code},
                                )
                                recorder.add_artifact(
                                    "code",
                                    label=description,
                                )
                                logger.info(
                                    "\u2705 %s", description
                                )
                                code_path = self._persist_execute_code_source(
                                    code,
                                    turn_index=turn + 1,
                                    tool_index=tool_index,
                                    description=description,
                                )
                                self._log_execute_code_source(
                                    code=code,
                                    turn_index=turn + 1,
                                    tool_index=tool_index,
                                    path=code_path,
                                )
                                stdout_path = self._persist_execute_code_stdout(
                                    stdout_text,
                                    turn_index=turn + 1,
                                    tool_index=tool_index,
                                    description=description,
                                )
                                self._log_execute_code_stdout(
                                    stdout=stdout_text,
                                    turn_index=turn + 1,
                                    tool_index=tool_index,
                                    path=stdout_path,
                                )
                                debug_path = self._persist_tool_debug_output(
                                    canonical,
                                    str(debug_tool_output or tool_output),
                                    turn_index=turn + 1,
                                    tool_index=tool_index,
                                    description=description,
                                )
                                self._log_tool_debug_output(
                                    tool_name=canonical,
                                    output=str(debug_tool_output or tool_output),
                                    turn_index=turn + 1,
                                    tool_index=tool_index,
                                    path=debug_path,
                                )
                                await emitter.execution_completed(
                                    description,
                                    turn_id=recorder.trace.turn_id,
                                    trace_id=recorder.trace.trace_id,
                                    session_id=recorder.trace.session_id,
                                )
                                await emit(
                                    build_stream_event(
                                        "code",
                                        code,
                                        turn_id=recorder.trace.turn_id,
                                        trace_id=recorder.trace.trace_id,
                                        session_id=recorder.trace.session_id,
                                        category="execution",
                                    )
                                )
                            else:
                                await emitter.delegation_completed(
                                    tc.arguments.get("agent_type", ""),
                                    turn_id=recorder.trace.turn_id,
                                    trace_id=recorder.trace.trace_id,
                                    session_id=recorder.trace.session_id,
                                )

                            shape = (
                                (
                                    current_adata.shape[0],
                                    current_adata.shape[1],
                                )
                                if hasattr(current_adata, "shape")
                                else None
                            )
                            recorder.add_step(
                                "result",
                                name=canonical or tc.name,
                                summary="adata updated",
                                data={"shape": shape},
                            )
                            await emit(
                                build_stream_event(
                                    "result",
                                    current_adata,
                                    turn_id=recorder.trace.turn_id,
                                    trace_id=recorder.trace.trace_id,
                                    session_id=recorder.trace.session_id,
                                    category="execution",
                                )
                                | {"shape": shape}
                            )
                        elif canonical == "finish":
                            summary = tc.arguments.get(
                                "summary", "Task completed"
                            )
                            final_summary = summary
                            recorder.add_step(
                                "done", name=canonical, summary=summary
                            )
                            await emitter.task_finished(
                                summary,
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                session_id=recorder.trace.session_id,
                            )
                            tool_output = f"Task finished: {summary}"
                            finished = True
                        elif isinstance(result, str):
                            tool_output = result
                        else:
                            tool_output = str(result)

                        if debug_tool_output is None:
                            debug_tool_output = tool_output

                        # Tier-driven truncation via budget manager
                        meta = self._tool_runtime.registry.get(
                            canonical
                        )
                        output_tier = (
                            meta.output_tier
                            if meta is not None
                            else OutputTier.standard
                        )
                        tool_output = budget_manager.truncate_output(
                            tool_output, output_tier
                        )
                        budget_manager.record(
                            BudgetSliceType.tool_output,
                            tool_output,
                            content_key=canonical or tc.name,
                            tier=output_tier,
                        )

                        await emit(
                            build_stream_event(
                                "tool_result",
                                {
                                    "name": canonical or tc.name,
                                    "output": tool_output,
                                },
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                step_id=tool_step_id,
                                session_id=recorder.trace.session_id,
                                category="tool",
                            )
                        )

                        try:
                            parsed_tool_output = json.loads(tool_output)
                        except Exception as e:
                            logger.debug("turn_controller: failed to parse tool output as JSON (%s)", e)
                            parsed_tool_output = None

                        if isinstance(parsed_tool_output, dict):
                            if canonical == "ToolSearch":
                                await emit(
                                    build_stream_event(
                                        "status",
                                        {
                                            "loaded_tools": (
                                                parsed_tool_output.get(
                                                    "loaded_tools", []
                                                )
                                            ),
                                            "newly_loaded": (
                                                parsed_tool_output.get(
                                                    "newly_loaded", []
                                                )
                                            ),
                                        },
                                        turn_id=recorder.trace.turn_id,
                                        trace_id=recorder.trace.trace_id,
                                        step_id=tool_step_id,
                                        session_id=recorder.trace.session_id,
                                        category="runtime",
                                    )
                                )
                            elif canonical in {
                                "EnterPlanMode", "ExitPlanMode",
                            }:
                                await emit(
                                    build_stream_event(
                                        "status",
                                        {"plan_mode": parsed_tool_output},
                                        turn_id=recorder.trace.turn_id,
                                        trace_id=recorder.trace.trace_id,
                                        step_id=tool_step_id,
                                        session_id=recorder.trace.session_id,
                                        category="runtime",
                                    )
                                )
                            elif canonical == "EnterWorktree":
                                await emit(
                                    build_stream_event(
                                        "status",
                                        {"worktree": parsed_tool_output},
                                        turn_id=recorder.trace.turn_id,
                                        trace_id=recorder.trace.trace_id,
                                        step_id=tool_step_id,
                                        session_id=recorder.trace.session_id,
                                        category="runtime",
                                    )
                                )
                            elif canonical in {
                                "TaskCreate", "TaskUpdate",
                                "TaskStop", "Bash",
                            }:
                                await emit(
                                    build_stream_event(
                                        "task_update",
                                        parsed_tool_output,
                                        turn_id=recorder.trace.turn_id,
                                        trace_id=recorder.trace.trace_id,
                                        step_id=tool_step_id,
                                        session_id=recorder.trace.session_id,
                                        category="task",
                                    )
                                )

                        _ordered_results[call_idx] = (
                            tc.id, tc.name, tool_output
                        )
                        _sync_runtime_trace()
                        await emit(
                            build_stream_event(
                                "item_completed",
                                {
                                    "item_type": "tool_call",
                                    "name": canonical or tc.name,
                                    "status": "success",
                                    "batch_id": batch.batch_id,
                                },
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                step_id=tool_step_id,
                                session_id=recorder.trace.session_id,
                                category="item",
                            )
                        )

                # Collect results preserving original order
                tool_results = [
                    r for r in _ordered_results if r is not None
                ]

                self._append_tool_results(messages, tool_results)

                # --- Convergence steering ---
                convergence_monitor.record_turn(
                    [tc.name for tc in response.tool_calls]
                )
                if (
                    not finished
                    and convergence_monitor.should_inject()
                ):
                    steering = (
                        convergence_monitor.build_steering_message()
                    )
                    messages.append(
                        {"role": "user", "content": steering}
                    )
                    logger.info(
                        "convergence_steering_injected turn=%d "
                        "level=%d",
                        turn + 1,
                        convergence_monitor._escalation,
                    )
                    recorder.add_step(
                        "status",
                        summary="convergence_steering_injected",
                        data={
                            "escalation_level": (
                                convergence_monitor._escalation
                            ),
                            "consecutive_readonly": (
                                convergence_monitor
                                ._consecutive_readonly
                            ),
                        },
                    )

                # --- Incremental compaction checkpoint ---
                if (
                    not finished
                    and budget_manager.should_compact()
                    and len(messages) > 6
                ):
                    # Build a short summary of tool results from
                    # this turn for the checkpoint
                    turn_tool_names = [
                        tc.name for tc in response.tool_calls
                    ]
                    cp_summary = (
                        f"Tools called: {', '.join(turn_tool_names)}. "
                        f"Budget utilization: "
                        f"{budget_manager.utilization:.0%}."
                    )
                    budget_manager.add_checkpoint(
                        turn_index=turn,
                        summary=cp_summary,
                        messages_covered=len(messages),
                    )
                    logger.info(
                        "budget_checkpoint turn=%d "
                        "utilization=%.2f messages=%d",
                        turn + 1,
                        budget_manager.utilization,
                        len(messages),
                    )
                    recorder.add_step(
                        "status",
                        summary="budget_compaction_checkpoint",
                        data=budget_manager.to_dict(),
                    )

                if finished:
                    summary = final_summary or tc.arguments.get(
                        "summary", "Task completed"
                    )
                    recorder.finish(
                        status="success",
                        result_summary=summary,
                        usage=ctx.last_usage,
                    )
                    _finalize_analysis_run("success", summary)
                    if ctx._trace_store is not None:
                        recorder.save(ctx._trace_store)
                    await emit(
                        build_stream_event(
                            "done",
                            summary,
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                            category="lifecycle",
                        )
                    )
                    if ctx.last_usage:
                        await emit(
                            build_stream_event(
                                "usage",
                                ctx.last_usage,
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                session_id=recorder.trace.session_id,
                                category="usage",
                            )
                        )
                    self._save_conversation_log(messages)
                    ctx._last_run_trace = recorder.trace
                    return current_adata

            # Post-loop
            if completed_without_tool_calls:
                recorder.finish(
                    status="success",
                    result_summary=final_summary or "Completed",
                    usage=ctx.last_usage,
                )
                _finalize_analysis_run(
                    "success", final_summary or "Completed"
                )
                if ctx._trace_store is not None:
                    recorder.save(ctx._trace_store)
                await emit(
                    build_stream_event(
                        "done",
                        final_summary or "Completed",
                        turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id,
                        session_id=recorder.trace.session_id,
                        category="lifecycle",
                    )
                )
            else:
                final_summary = final_summary or (
                    f"Reached max turns ({max_turns})"
                )
                recorder.finish(
                    status="max_turns",
                    result_summary=final_summary,
                    usage=ctx.last_usage,
                )
                _finalize_analysis_run("max_turns", final_summary)
                if ctx._trace_store is not None:
                    recorder.save(ctx._trace_store)
                await emitter.max_turns_reached(
                    max_turns,
                    turn_id=recorder.trace.turn_id,
                    trace_id=recorder.trace.trace_id,
                    session_id=recorder.trace.session_id,
                )
            if ctx.last_usage:
                await emit(
                    build_stream_event(
                        "usage",
                        ctx.last_usage,
                        turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id,
                        session_id=recorder.trace.session_id,
                        category="usage",
                    )
                )
            self._save_conversation_log(messages)
            ctx._last_run_trace = recorder.trace
            return current_adata
        except Exception as exc:
            recorder.add_step("error", summary=str(exc))
            recorder.finish(
                status="error",
                result_summary=str(exc),
                usage=ctx.last_usage,
            )
            _finalize_analysis_run("error", str(exc))
            if ctx._trace_store is not None:
                recorder.save(ctx._trace_store)
            ctx._last_run_trace = recorder.trace
            try:
                await emit(
                    build_stream_event(
                        "done",
                        f"Error: {exc}",
                        turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id,
                        session_id=recorder.trace.session_id,
                        category="lifecycle",
                    )
                    | {"error": True}
                )
            except Exception as _emit_exc:
                logger.debug("turn_controller: failed to emit error done event: %s", _emit_exc)
            raise
        finally:
            ctx._approval_handler = None
            ctx._active_run_id = ""
