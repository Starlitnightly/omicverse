"""SubagentController — spawn and manage subagent conversations.

Extracted from ``smart_agent.py``.  A subagent gets its own system prompt,
restricted tool set, independent message history, and — critically — an
isolated ``SubagentRuntime`` that prevents mutable-state leakage between
parent and child turns.

Isolation contract
------------------
* Subagent **does not** write to the parent's ``ctx.last_usage``.
* Subagent operates under a ``PermissionPolicy`` scoped to its allowed
  tool set; tools outside the set are denied before dispatch.
* Budget tracking uses a subagent-local ``ContextBudgetManager``.
* Tool schemas are snapshotted at subagent creation time and not
  re-read from the parent during the subagent's turn loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

from .context_budget import (
    BudgetSliceType,
    ContextBudgetManager,
    create_subagent_budget_manager,
)
from .permission_policy import (
    PermissionDecision,
    PermissionPolicy,
    create_subagent_policy,
)
from .tool_registry import OutputTier

if TYPE_CHECKING:
    from .prompt_builder import PromptBuilder
    from .protocol import AgentContext
    from .tool_runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Subagent runtime — isolated execution state for one subagent run
# ---------------------------------------------------------------------------


@dataclass
class SubagentRuntime:
    """Scoped execution context for a single subagent run.

    Captures everything the subagent needs and provides its own mutable
    state space.  The parent agent's mutable context is **not** directly
    accessible through this object.

    Attributes
    ----------
    agent_type : str
        The subagent kind (``"explore"``, ``"plan"``, ``"execute"``).
    max_turns : int
        Maximum number of LLM turns for this subagent run.
    permission_policy : PermissionPolicy
        Scoped permission evaluator — only allowed tools pass.
    budget_manager : ContextBudgetManager
        Subagent-local token budget tracker.
    tool_schemas : list
        Snapshotted tool schemas visible to this subagent.
    last_usage : Any
        LLM usage tracking — subagent-local, **not** shared with parent.
    can_mutate_adata : bool
        Whether this subagent type is allowed to mutate ``adata``.
    """

    agent_type: str
    max_turns: int
    permission_policy: PermissionPolicy
    budget_manager: ContextBudgetManager
    tool_schemas: List[Any] = field(default_factory=list)
    last_usage: Any = None
    can_mutate_adata: bool = False

    def record_usage(self, usage: Any) -> None:
        """Record LLM usage in the subagent's own state."""
        self.last_usage = usage

    def check_tool_permission(self, tool_name: str) -> PermissionDecision:
        """Evaluate whether *tool_name* is permitted in this subagent."""
        return self.permission_policy.check(tool_name)


# ---------------------------------------------------------------------------
# Controller
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
        Tool dispatch hub (used for dispatch only; mutable state is NOT
        shared with the subagent).
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
    # Subagent runtime factory
    # ------------------------------------------------------------------

    def _create_subagent_runtime(
        self,
        agent_type: str,
        allowed_tools: FrozenSet[str],
        max_turns: int,
        can_mutate_adata: bool = False,
    ) -> SubagentRuntime:
        """Build an isolated ``SubagentRuntime`` for one subagent run.

        This is the single factory point for subagent isolation.  It:

        1. Creates a ``PermissionPolicy`` scoped to *allowed_tools*.
        2. Snapshots the visible tool schemas (no live reference to parent).
        3. Creates a subagent-local budget manager.
        """
        # Permission policy scoped to allowed tools
        policy = create_subagent_policy(
            self._tool_runtime.registry,
            allowed_tools=allowed_tools,
        )

        # Budget manager — subagent-local
        budget_model = (
            getattr(
                getattr(self._ctx._llm, "config", None), "model", None
            )
            or self._ctx.model
            or ""
        )
        budget_manager = create_subagent_budget_manager(model=budget_model)

        # Snapshot tool schemas — the subagent sees a frozen list, not a
        # live reference into the parent's tool registry.
        tool_schemas = list(
            self._tool_runtime.get_visible_agent_tools(
                allowed_names=set(allowed_tools)
            )
        )

        return SubagentRuntime(
            agent_type=agent_type,
            max_turns=max_turns,
            permission_policy=policy,
            budget_manager=budget_manager,
            tool_schemas=tool_schemas,
            can_mutate_adata=can_mutate_adata,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_subagent(
        self,
        agent_type: str,
        task: str,
        adata: Any,
        context: str = "",
    ) -> dict:
        """Spawn a subagent with restricted tools and its own conversation.

        The subagent runs inside a ``SubagentRuntime`` that isolates its
        mutable state (usage tracking, budget, permission decisions) from
        the parent agent.

        Returns
        -------
        dict
            ``{"result": str, "adata": AnnData, "last_usage": Any}``
        """
        config = self._ctx._config.get_subagent_config(agent_type)

        # Create isolated runtime
        runtime = self._create_subagent_runtime(
            agent_type=agent_type,
            allowed_tools=frozenset(config.allowed_tools),
            max_turns=config.max_turns,
            can_mutate_adata=config.can_mutate_adata,
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

        for turn in range(runtime.max_turns):
            logger.info(
                "subagent_turn agent_type=%s turn=%d/%d",
                agent_type,
                turn + 1,
                runtime.max_turns,
            )

            response = await self._ctx._llm.chat(
                messages, tools=runtime.tool_schemas, tool_choice="auto"
            )

            # Record usage in subagent runtime — NOT on parent ctx
            if response.usage:
                runtime.record_usage(response.usage)

            if response.raw_message:
                if isinstance(response.raw_message, list):
                    for msg in response.raw_message:
                        if isinstance(msg, dict):
                            messages.append(msg)
                        else:
                            logger.warning(
                                "subagent: skipping non-dict raw_message element type=%s",
                                type(msg).__name__,
                            )
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
                    "last_usage": runtime.last_usage,
                }

            for tc in response.tool_calls:
                logger.info(
                    "subagent_tool agent_type=%s tool=%s args=[%s]",
                    agent_type,
                    tc.name,
                    ", ".join(f"{k}=" for k in tc.arguments),
                )

                # Permission check before dispatch
                decision = runtime.check_tool_permission(tc.name)
                if decision.is_denied:
                    tool_output = (
                        f"Permission denied for tool '{tc.name}': "
                        f"{decision.reason}"
                    )
                    tool_msg = self._ctx._llm.format_tool_result_message(
                        tc.id, tc.name, tool_output
                    )
                    messages.append(tool_msg)
                    continue

                result = await self._tool_runtime.dispatch_tool(
                    tc, working_adata, task,
                    permission_policy=runtime.permission_policy,
                )

                if (
                    tc.name == "execute_code"
                    and isinstance(result, dict)
                    and "adata" in result
                ):
                    if runtime.can_mutate_adata:
                        working_adata = result["adata"]
                    tool_output = result.get("output", "Code executed.")
                elif tc.name == "finish":
                    summary = tc.arguments.get("summary", "")
                    logger.info(
                        "subagent_finished agent_type=%s summary=%s",
                        agent_type,
                        summary[:120],
                    )
                    return {
                        "result": summary,
                        "adata": working_adata,
                        "last_usage": runtime.last_usage,
                    }
                elif isinstance(result, str):
                    tool_output = result
                else:
                    tool_output = str(result)

                # Tier-driven truncation via subagent-local budget manager
                meta = self._tool_runtime.registry.get(tc.name)
                output_tier = (
                    meta.output_tier
                    if meta is not None
                    else OutputTier.standard
                )
                tool_output = runtime.budget_manager.truncate_output(
                    tool_output, output_tier
                )
                runtime.budget_manager.record(
                    BudgetSliceType.tool_output,
                    tool_output,
                    content_key=tc.name,
                    tier=output_tier,
                )

                tool_msg = self._ctx._llm.format_tool_result_message(
                    tc.id, tc.name, tool_output
                )
                messages.append(tool_msg)

        return {
            "result": (
                f"Subagent ({agent_type}) reached max turns "
                f"({runtime.max_turns})"
            ),
            "adata": working_adata,
            "last_usage": runtime.last_usage,
        }
