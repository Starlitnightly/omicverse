"""TurnController — agentic orchestration loop + follow-up gate.

Extracted from ``smart_agent.py``.  The TurnController owns the main
turn-by-turn loop (LLM call → tool dispatch → result append → repeat)
including stall detection, cancellation, and conversation logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..harness import (
    RunTraceRecorder,
    build_stream_event,
    coerce_usage_payload,
)
from ..harness.runtime_state import runtime_state
from ..harness.tool_catalog import normalize_tool_name
from ..session_history import HistoryEntry

if TYPE_CHECKING:
    from .prompt_builder import PromptBuilder
    from .protocol import AgentContext
    from .tool_runtime import ToolRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Follow-up gate helper
# ---------------------------------------------------------------------------

class FollowUpGate:
    """Stateless helpers for the follow-up / retry heuristic."""

    URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
    ACTION_REQUEST_PATTERN = re.compile(
        r"\b(analy[sz]e|download|fetch|get|open|read|inspect|load|run|"
        r"execute|search|lookup|look up|find|process|parse|clone|fix|"
        r"edit|write)\b",
        re.IGNORECASE,
    )
    PROMISSORY_PATTERN = re.compile(
        r"\b(let me|i(?:'ll| will| can)|going to|start by|"
        r"first(?:,)?\s+i(?:'ll| will)|can continue\?|"
        r"could continue\?|re-?start)\b",
        re.IGNORECASE,
    )
    BLOCKER_PATTERN = re.compile(
        r"\b(can(?:not|'t)|unable|failed|error|need your|"
        r"please provide|approval required|missing|not installed|"
        r"permission denied)\b",
        re.IGNORECASE,
    )
    RESULT_PATTERN = re.compile(
        r"\b(found|fetched|downloaded|loaded|read|parsed|"
        r"here (?:is|are)|summary|supplementary|links?)\b",
        re.IGNORECASE,
    )

    @classmethod
    def request_requires_tool_action(
        cls, request: str, adata: Any
    ) -> bool:
        text = (request or "").strip()
        if not text:
            return False
        if cls.URL_PATTERN.search(text):
            return True
        if adata is not None:
            return True
        lowered = text.lower()
        if any(
            marker in lowered
            for marker in (
                "\u6570\u636e", "dataset", "\u4e0b\u8f7d",
                "\u5206\u6790", "\u5904\u7406", "\u8bfb\u53d6",
                "\u6253\u5f00", "\u641c\u7d22",
            )
        ):
            return True
        return bool(cls.ACTION_REQUEST_PATTERN.search(text))

    @classmethod
    def response_is_promissory(cls, content: str) -> bool:
        text = (content or "").strip()
        if not text:
            return False
        if cls.PROMISSORY_PATTERN.search(text):
            return True
        chinese_markers = (
            "\u6211\u5148", "\u8ba9\u6211", "\u6211\u4f1a",
            "\u6211\u5c06", "\u5148\u83b7\u53d6",
            "\u5148\u4e0b\u8f7d", "\u5148\u8bfb\u53d6",
            "\u5148\u53bb", "\u73b0\u5728\u5f00\u59cb",
            "\u91cd\u65b0\u5f00\u59cb",
            "\u53ef\u4ee5\u7ee7\u7eed\u5417",
            "\u7ee7\u7eed\u5417",
        )
        lowered = text.lower()
        return (
            any(marker in text for marker in chinese_markers)
            or lowered.startswith("okay, i")
        )

    @classmethod
    def select_tool_choice(
        cls,
        *,
        request: str,
        adata: Any,
        turn_index: int,
        had_meaningful_tool_call: bool,
        forced_retry: bool,
    ) -> str:
        if forced_retry:
            return "required"
        if (
            turn_index == 0
            and not had_meaningful_tool_call
            and cls.request_requires_tool_action(request, adata)
        ):
            return "required"
        return "auto"

    @classmethod
    def should_continue_after_text(
        cls,
        *,
        request: str,
        response_content: str,
        adata: Any,
        had_meaningful_tool_call: bool,
    ) -> bool:
        text = (response_content or "").strip()
        if not text:
            return False
        if had_meaningful_tool_call:
            return False
        if cls.BLOCKER_PATTERN.search(text):
            return False
        needs_action = cls.request_requires_tool_action(request, adata)
        if cls.response_is_promissory(text) and needs_action:
            # Only follow up when there is actually a task to execute.
            # Pure offers like "I can help you" without actionable context
            # should not trigger a forced tool-call turn.
            return True
        if needs_action and not cls.RESULT_PATTERN.search(text):
            return True
        return False

    @classmethod
    def build_no_tool_follow_up(
        cls,
        request: str,
        *,
        retry_count: int = 0,
        max_retries: int = 2,
    ) -> str:
        if retry_count >= max_retries - 1:
            base = (
                "IMPORTANT: You MUST call a tool in this response. "
                "Do NOT respond with text only. Use one of your available "
                "tools now. If you cannot proceed, call the 'finish' tool "
                "with a summary of what went wrong."
            )
        else:
            base = (
                "Do not describe future actions without taking them. "
                "Either call the appropriate tool now or provide the "
                "final answer only if the task is already complete."
            )
        if cls.URL_PATTERN.search(request or ""):
            return (
                base
                + " The user provided a URL, so fetch it in this turn "
                "with `WebFetch`/`web_fetch` before continuing."
            )
        return base


# ---------------------------------------------------------------------------
# Convergence monitor — soft steering for read-only tool plateaus
# ---------------------------------------------------------------------------

class ConvergenceMonitor:
    """Detect read-only-tool plateaus and inject escalating steering messages.

    Fires when the LLM calls only read-only tools (run_snippet, inspect_data,
    search_functions, search_skills) for several consecutive turns without
    ever using execute_code, and the output contract still has unproduced
    artifacts.
    """

    READ_ONLY_TOOLS = frozenset({
        "run_snippet", "inspect_data", "search_functions",
        "search_skills", "RunSnippet", "InspectData",
        "SearchFunctions", "SearchSkills",
    })
    ARTIFACT_TOOLS = frozenset({
        "execute_code", "ExecuteCode",
    })
    THRESHOLD = 2
    ESCALATION_LEVELS = 3

    def __init__(self, initial_prompt: str):
        self._consecutive_readonly = 0
        self._execute_code_seen = False
        self._escalation = 0
        self._force_execute_next = False
        self._required_artifacts = self._parse_output_contract(
            initial_prompt
        )

    @staticmethod
    def _parse_output_contract(prompt: str) -> List[str]:
        """Extract artifact IDs from the OUTPUT CONTRACT block."""
        artifacts: List[str] = []
        in_contract = False
        for line in prompt.split("\n"):
            if "OUTPUT CONTRACT" in line:
                in_contract = True
                continue
            if in_contract:
                stripped = line.strip()
                if stripped.startswith("* "):
                    part = stripped[2:].split(":")[0].strip()
                    if part:
                        artifacts.append(part)
                elif (
                    stripped
                    and not stripped.startswith("-")
                    and not stripped.startswith("*")
                ):
                    in_contract = False
        return artifacts

    def record_turn(self, tool_names: List[str]) -> None:
        """Call after each turn's tool dispatch completes."""
        normalized = {
            normalize_tool_name(n) or n for n in tool_names
        }
        if normalized & self.ARTIFACT_TOOLS:
            self._execute_code_seen = True
            self._consecutive_readonly = 0
            return
        if normalized and normalized <= self.READ_ONLY_TOOLS:
            self._consecutive_readonly += 1
        else:
            self._consecutive_readonly = 0

    def should_inject(self) -> bool:
        """True when steering message should be injected."""
        if self._execute_code_seen:
            return False
        if not self._required_artifacts:
            return False
        if self._consecutive_readonly < self.THRESHOLD:
            return False
        if self._escalation >= self.ESCALATION_LEVELS:
            return False
        return True

    def should_force_tool_choice(self) -> bool:
        """True when tool_choice should be forced to 'required'."""
        if self._execute_code_seen:
            return False
        return self._force_execute_next

    def build_steering_message(self) -> str:
        """Return escalating steering text. Advances escalation level."""
        self._escalation += 1
        artifacts_str = ", ".join(self._required_artifacts)
        if self._escalation == 1:
            return (
                "You have been exploring for several turns. The task "
                f"requires producing these artifacts: [{artifacts_str}]. "
                "You have enough context now. Call execute_code() with "
                "the full analysis pipeline to generate these outputs. "
                "Do NOT call run_snippet again — it cannot save files."
            )
        if self._escalation == 2:
            return (
                "IMPORTANT: You have explored extensively but have not "
                "produced any required artifacts yet. The output "
                f"contract requires: [{artifacts_str}]. Use "
                "execute_code() NOW to generate these files. "
                "run_snippet is read-only and CANNOT save files or "
                "produce artifacts. Only execute_code() can do that."
            )
        # Level 3: set force flag for tool_choice override
        self._force_execute_next = True
        return (
            "URGENT: No artifacts have been produced. The task WILL "
            "FAIL unless you call execute_code() immediately to "
            f"create: [{artifacts_str}]. You MUST call execute_code "
            "with complete code that imports all needed libraries, "
            "processes the data, and saves every required output file. "
            "Do NOT call run_snippet or inspect_data."
        )


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

    def _persist_harness_history(self, request: str) -> None:
        """Persist the latest trace into session history when enabled."""
        ctx = self._ctx
        if ctx._session_history is None or ctx._last_run_trace is None:
            return

        trace = ctx._last_run_trace

        def _step_type(step):
            return (
                step.get("step_type")
                if isinstance(step, dict)
                else step.step_type
            )

        def _step_name(step):
            return (
                step.get("name") if isinstance(step, dict) else step.name
            )

        def _step_data(step):
            return (
                step.get("data", {})
                if isinstance(step, dict)
                else step.data
            )

        generated_code = "\n\n".join(
            _step_data(step).get("code", "")
            for step in trace.steps
            if _step_type(step) == "code"
            and _step_data(step).get("code")
        )
        tool_names = [
            _step_name(step)
            for step in trace.steps
            if _step_type(step) == "tool_call" and _step_name(step)
        ]
        artifact_refs = [
            (
                artifact.to_dict()
                if hasattr(artifact, "to_dict")
                else dict(artifact)
            )
            for artifact in trace.artifacts
        ]
        ctx._session_history.append(
            HistoryEntry(
                session_id=(
                    trace.session_id or ctx._get_harness_session_id()
                ),
                timestamp=trace.finished_at or trace.started_at,
                request=request,
                trace_id=trace.trace_id,
                generated_code=generated_code,
                result_summary=trace.result_summary,
                tool_names=tool_names,
                artifact_refs=artifact_refs,
                usage=(
                    trace.usage
                    or coerce_usage_payload(ctx.last_usage)
                ),
                usage_breakdown=ctx.last_usage_breakdown,
                success=trace.status == "success",
            )
        )

    @staticmethod
    def _save_conversation_log(messages: list) -> None:
        """Save the full conversation to a JSON file for debugging."""
        log_dir = os.environ.get("OV_AGENT_LOG_DIR")
        if not log_dir:
            return
        try:
            import datetime as _dt
            from pathlib import Path

            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = log_path / f"agent_conversation_{ts}.json"

            def _safe(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [_safe(v) for v in obj]
                if isinstance(obj, dict):
                    return {k: _safe(v) for k, v in obj.items()}
                return repr(obj)

            out_file.write_text(
                json.dumps(_safe(messages), indent=2, ensure_ascii=False)
            )
            print(f"   \U0001f4dd Conversation log saved: {out_file}")
        except Exception as exc:
            logger.debug("Failed to save conversation log: %s", exc)

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
                    print("   \u26d4 Cancelled before LLM call")
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
                    await emit(
                        build_stream_event(
                            "done",
                            "Cancelled",
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                            category="lifecycle",
                        )
                        | {"cancelled": True}
                    )
                    self._save_conversation_log(messages)
                    return current_adata

                print(f"   \U0001f504 Turn {turn + 1}/{max_turns}")
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

                logger.info(
                    "agentic_llm_call_start turn=%d/%d tool_choice=%s "
                    "messages=%d post_tool=%s",
                    turn + 1,
                    max_turns,
                    tool_choice,
                    len(messages),
                    "yes" if meaningful_tool_call_seen else "no",
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
                        print(
                            "   \U0001f4ac Agent response: "
                            f"{final_summary[:200]}"
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

                for tc in response.tool_calls:
                    canonical = normalize_tool_name(tc.name)
                    if _is_cancelled():
                        print(
                            "   \u26d4 Cancelled before tool dispatch"
                        )
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
                        await emit(
                            build_stream_event(
                                "done",
                                "Cancelled",
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                session_id=recorder.trace.session_id,
                                category="lifecycle",
                            )
                            | {"cancelled": True}
                        )
                        self._save_conversation_log(messages)
                        return current_adata

                    tool_step_id = recorder.add_step(
                        "tool_call",
                        name=canonical or tc.name,
                        summary=f"{canonical or tc.name} dispatched",
                        data={"arguments": tc.arguments},
                    )
                    print(
                        f"   \U0001f527 Tool: {canonical or tc.name}"
                        f"({', '.join(f'{k}=' for k in tc.arguments)})"
                    )

                    await emit(
                        build_stream_event(
                            "item_started",
                            {
                                "item_type": "tool_call",
                                "name": canonical or tc.name,
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

                    result = await self._tool_runtime.dispatch_tool(
                        tc, current_adata, request
                    )
                    if canonical not in {
                        "finish", "TaskGet", "TaskList",
                        "TaskOutput", "ToolSearch",
                    }:
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
                        if canonical == "execute_code":
                            code = tc.arguments.get("code", "")
                            recorder.add_step(
                                "code",
                                name=canonical,
                                summary=tc.arguments.get(
                                    "description", "Code executed"
                                ),
                                data={"code": code},
                            )
                            recorder.add_artifact(
                                "code",
                                label=tc.arguments.get(
                                    "description", "execute_code"
                                ),
                            )
                            print(
                                "      \u2705 "
                                + tc.arguments.get(
                                    "description", "Code executed"
                                )
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
                            print(
                                "      \u2705 delegate("
                                + tc.arguments.get("agent_type", "")
                                + ") completed"
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
                        print(f"   \u2705 Finished: {summary}")
                        tool_output = f"Task finished: {summary}"
                        finished = True
                    elif isinstance(result, str):
                        tool_output = result
                    else:
                        tool_output = str(result)

                    # Truncate
                    max_tool_output_chars = 8000
                    if canonical in {
                        "WebFetch", "WebSearch",
                        "web_fetch", "web_search",
                    }:
                        max_tool_output_chars = 4000
                    if len(tool_output) > max_tool_output_chars:
                        tool_output = (
                            tool_output[: max_tool_output_chars - 500]
                            + "\n... (truncated)"
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

                    tool_results.append(
                        (tc.id, tc.name, tool_output)
                    )
                    _sync_runtime_trace()
                    await emit(
                        build_stream_event(
                            "item_completed",
                            {
                                "item_type": "tool_call",
                                "name": canonical or tc.name,
                                "status": "success",
                            },
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            step_id=tool_step_id,
                            session_id=recorder.trace.session_id,
                            category="item",
                        )
                    )

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
                logger.warning(
                    "agentic_loop_max_turns max_turns=%d "
                    "meaningful_tool=%s",
                    max_turns,
                    meaningful_tool_call_seen,
                )
                print(
                    f"   \u26a0\ufe0f  Max turns ({max_turns}) reached, "
                    "returning current result"
                )
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
                await emit(
                    build_stream_event(
                        "done",
                        final_summary,
                        turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id,
                        session_id=recorder.trace.session_id,
                        category="lifecycle",
                    )
                    | {"max_turns": True}
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
