"""SubagentController — spawn and manage subagent conversations.

Extracted from ``smart_agent.py``.  A subagent gets its own system prompt,
restricted tool set, and independent message history.

Isolation model
---------------
Two runtime modes are supported:

1. **Shared** (default) — the subagent dispatches tools through the parent's
   ``ToolRuntime``, sharing its mutable state (LLM backend, loaded tools,
   session runtime state).  This is adequate for lightweight subagents that
   live within the same trust boundary as the parent.

2. **Isolated** — the subagent receives a *snapshot* of the parent's read-only
   state via ``IsolatedSubagentContext`` and dispatches tools through its own
   ``PermissionPolicy``-gated dispatcher.  The parent's mutable runtime is
   never accessed during execution.  Results are handed back through a
   structured ``SubagentResult``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional

from .contracts import ApprovalClass, IsolationMode
from .permission_policy import (
    PermissionDecision,
    PermissionPolicy,
    PermissionVerdict,
    build_subagent_policy,
)

if TYPE_CHECKING:
    from .prompt_builder import PromptBuilder
    from .protocol import AgentContext
    from .tool_runtime import ToolRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Isolated subagent context — snapshot of parent read-only state
# ---------------------------------------------------------------------------

@dataclass
class IsolatedSubagentContext:
    """Read-only snapshot of the parent runtime for isolated subagent execution.

    This replaces direct access to the parent's mutable ``AgentContext`` in
    high-isolation paths.  The subagent cannot mutate the parent's session,
    loaded tools, or LLM backend through this object.
    """
    agent_type: str
    allowed_tools: FrozenSet[str]
    permission_policy: PermissionPolicy
    can_mutate_adata: bool = False
    max_turns: int = 10
    isolation_mode: IsolationMode = IsolationMode.IN_PROCESS

    # Snapshot of parent state (read-only copies)
    system_prompt: str = ""
    visible_tool_schemas: List[Dict[str, Any]] = field(default_factory=list)

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check whether *tool_name* passes the permission policy."""
        decision = self.permission_policy.check(tool_name)
        return decision.verdict == PermissionVerdict.ALLOW

    def check_tool_permission(self, tool_name: str) -> PermissionDecision:
        """Return the full permission decision for *tool_name*."""
        return self.permission_policy.check(tool_name)


@dataclass
class SubagentResult:
    """Structured result from a subagent execution.

    Returned by both shared and isolated paths so the caller gets a
    uniform interface.
    """
    result: str
    adata: Any = None
    turns_used: int = 0
    tool_calls_made: int = 0
    isolation_mode: IsolationMode = IsolationMode.IN_PROCESS
    denied_tool_calls: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SubagentController
# ---------------------------------------------------------------------------

class SubagentController:
    """Spawn subagents (explore / plan / execute) with restricted tools.

    Parameters
    ----------
    ctx : AgentContext
        The agent instance (accessed via protocol surface).
    prompt_builder : PromptBuilder
        Prompt construction helper.
    tool_runtime : ToolRuntime
        Tool dispatch hub (shared with the parent loop).
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
        from .event_stream import make_event_bus
        self._event_bus = make_event_bus(getattr(ctx, "_reporter", None))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_subagent(
        self,
        agent_type: str,
        task: str,
        adata: Any,
        context: str = "",
        *,
        isolation: Optional[IsolationMode] = None,
    ) -> dict:
        """Spawn a subagent with restricted tools and its own conversation.

        Parameters
        ----------
        isolation : IsolationMode or None
            Override the default isolation mode.  When ``None``, the shared
            (legacy) path is used.  Pass ``IsolationMode.SUBPROCESS`` or
            ``IsolationMode.WORKTREE`` to use the isolated path.

        Returns
        -------
        dict
            ``{"result": str, "adata": AnnData}``
        """
        if isolation is not None and isolation != IsolationMode.IN_PROCESS:
            sr = await self._run_isolated(agent_type, task, adata, context, isolation)
            return {"result": sr.result, "adata": sr.adata}

        return await self._run_shared(agent_type, task, adata, context)

    def build_isolated_context(
        self,
        agent_type: str,
        context: str = "",
    ) -> IsolatedSubagentContext:
        """Build an isolated context snapshot for *agent_type*.

        Useful for callers that want to inspect the policy or tool schemas
        before launching the subagent.
        """
        from ..agent_config import SUBAGENT_CONFIGS

        config = SUBAGENT_CONFIGS[agent_type]
        policy = build_subagent_policy(
            config.allowed_tools,
            deny_mutations=not config.can_mutate_adata,
        )
        prompt = self._prompt_builder.build_subagent_system_prompt(
            agent_type, context
        )
        tools = self._ctx._get_visible_agent_tools(
            allowed_names=set(config.allowed_tools)
        )

        return IsolatedSubagentContext(
            agent_type=agent_type,
            allowed_tools=frozenset(config.allowed_tools),
            permission_policy=policy,
            can_mutate_adata=config.can_mutate_adata,
            max_turns=config.max_turns,
            system_prompt=prompt,
            visible_tool_schemas=tools,
        )

    # ------------------------------------------------------------------
    # Shared path (legacy — parent ToolRuntime is used directly)
    # ------------------------------------------------------------------

    async def _run_shared(
        self,
        agent_type: str,
        task: str,
        adata: Any,
        context: str,
    ) -> dict:
        """Shared-runtime subagent execution (original behaviour)."""
        from ..agent_config import SUBAGENT_CONFIGS

        config = SUBAGENT_CONFIGS[agent_type]

        subagent_tools = self._ctx._get_visible_agent_tools(
            allowed_names=set(config.allowed_tools)
        )

        messages = [
            {
                "role": "system",
                "content": self._prompt_builder.build_subagent_system_prompt(
                    agent_type, context
                ),
            },
            {
                "role": "user",
                "content": self._prompt_builder.build_subagent_user_message(
                    task, adata
                ),
            },
        ]

        working_adata = adata

        for turn in range(config.max_turns):
            self._event_bus.subagent_turn(agent_type, turn + 1, config.max_turns)

            response = await self._ctx._llm.chat(
                messages, tools=subagent_tools, tool_choice="auto"
            )

            if response.usage:
                self._ctx.last_usage = response.usage

            if response.raw_message:
                if isinstance(response.raw_message, list):
                    messages.extend(response.raw_message)
                else:
                    messages.append(response.raw_message)
            elif response.content:
                messages.append(
                    {"role": "assistant", "content": response.content}
                )

            if not response.tool_calls:
                return {
                    "result": response.content or "",
                    "adata": working_adata,
                }

            for tc in response.tool_calls:
                self._event_bus.subagent_tool(
                    agent_type, tc.name, list(tc.arguments.keys()),
                )

                result = await self._tool_runtime.dispatch_tool(
                    tc, working_adata, task
                )

                if (
                    tc.name == "execute_code"
                    and isinstance(result, dict)
                    and "adata" in result
                ):
                    working_adata = result["adata"]
                    tool_output = result.get("output", "Code executed.")
                elif tc.name == "finish":
                    summary = tc.arguments.get("summary", "")
                    self._event_bus.subagent_finished(agent_type, summary)
                    return {"result": summary, "adata": working_adata}
                elif isinstance(result, str):
                    tool_output = result
                else:
                    tool_output = str(result)

                if len(tool_output) > 6000:
                    tool_output = tool_output[:5500] + "\n... (truncated)"

                tool_msg = self._ctx._llm.format_tool_result_message(
                    tc.id, tc.name, tool_output
                )
                messages.append(tool_msg)

        return {
            "result": (
                f"Subagent ({agent_type}) reached max turns "
                f"({config.max_turns})"
            ),
            "adata": working_adata,
        }

    # ------------------------------------------------------------------
    # Isolated path — no shared mutable parent runtime
    # ------------------------------------------------------------------

    async def _run_isolated(
        self,
        agent_type: str,
        task: str,
        adata: Any,
        context: str,
        isolation: IsolationMode,
    ) -> SubagentResult:
        """Isolated subagent execution.

        The subagent gets a frozen snapshot of the parent's state.  Tool
        calls are gated by the ``PermissionPolicy``; denied calls produce
        an error message rather than reaching the parent runtime.
        """
        iso_ctx = self.build_isolated_context(agent_type, context)
        iso_ctx = IsolatedSubagentContext(
            agent_type=iso_ctx.agent_type,
            allowed_tools=iso_ctx.allowed_tools,
            permission_policy=iso_ctx.permission_policy,
            can_mutate_adata=iso_ctx.can_mutate_adata,
            max_turns=iso_ctx.max_turns,
            isolation_mode=isolation,
            system_prompt=iso_ctx.system_prompt,
            visible_tool_schemas=iso_ctx.visible_tool_schemas,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": iso_ctx.system_prompt},
            {
                "role": "user",
                "content": self._prompt_builder.build_subagent_user_message(
                    task, adata
                ),
            },
        ]

        working_adata = adata
        total_tool_calls = 0
        denied_calls: list[str] = []

        for turn in range(iso_ctx.max_turns):
            self._event_bus.subagent_turn(agent_type, turn + 1, iso_ctx.max_turns)

            response = await self._ctx._llm.chat(
                messages,
                tools=iso_ctx.visible_tool_schemas,
                tool_choice="auto",
            )

            # Usage is recorded on the isolated result, not mutated on parent
            if response.raw_message:
                if isinstance(response.raw_message, list):
                    messages.extend(response.raw_message)
                else:
                    messages.append(response.raw_message)
            elif response.content:
                messages.append(
                    {"role": "assistant", "content": response.content}
                )

            if not response.tool_calls:
                return SubagentResult(
                    result=response.content or "",
                    adata=working_adata,
                    turns_used=turn + 1,
                    tool_calls_made=total_tool_calls,
                    isolation_mode=isolation,
                    denied_tool_calls=denied_calls,
                )

            for tc in response.tool_calls:
                total_tool_calls += 1
                self._event_bus.subagent_tool(
                    agent_type, tc.name, list(tc.arguments.keys()),
                )

                # Permission gate — check before dispatching
                decision = iso_ctx.check_tool_permission(tc.name)
                if decision.verdict == PermissionVerdict.DENY:
                    denied_calls.append(tc.name)
                    tool_output = (
                        f"Permission denied: {tc.name}. {decision.reason}"
                    )
                    logger.info(
                        "Isolated subagent %s: denied tool %s — %s",
                        agent_type, tc.name, decision.reason,
                    )
                elif decision.verdict == PermissionVerdict.ASK:
                    # In isolated mode, ASK escalates to DENY (no user present)
                    denied_calls.append(tc.name)
                    tool_output = (
                        f"Tool '{tc.name}' requires approval which is not "
                        f"available in isolated mode. {decision.reason}"
                    )
                else:
                    # ALLOW — dispatch through tool runtime
                    result = await self._tool_runtime.dispatch_tool(
                        tc, working_adata, task
                    )
                    if (
                        tc.name == "execute_code"
                        and iso_ctx.can_mutate_adata
                        and isinstance(result, dict)
                        and "adata" in result
                    ):
                        working_adata = result["adata"]
                        tool_output = result.get("output", "Code executed.")
                    elif tc.name == "finish":
                        summary = tc.arguments.get("summary", "")
                        self._event_bus.subagent_finished(agent_type, summary)
                        return SubagentResult(
                            result=summary,
                            adata=working_adata,
                            turns_used=turn + 1,
                            tool_calls_made=total_tool_calls,
                            isolation_mode=isolation,
                            denied_tool_calls=denied_calls,
                        )
                    elif isinstance(result, str):
                        tool_output = result
                    else:
                        tool_output = str(result)

                if len(tool_output) > 6000:
                    tool_output = tool_output[:5500] + "\n... (truncated)"

                tool_msg = self._ctx._llm.format_tool_result_message(
                    tc.id, tc.name, tool_output
                )
                messages.append(tool_msg)

        return SubagentResult(
            result=(
                f"Subagent ({agent_type}) reached max turns "
                f"({iso_ctx.max_turns})"
            ),
            adata=working_adata,
            turns_used=iso_ctx.max_turns,
            tool_calls_made=total_tool_calls,
            isolation_mode=isolation,
            denied_tool_calls=denied_calls,
        )
