"""SubagentController — spawn and manage subagent conversations.

Extracted from ``smart_agent.py``.  A subagent gets its own system prompt,
restricted tool set, and independent message history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .prompt_builder import PromptBuilder
    from .protocol import AgentContext
    from .tool_runtime import ToolRuntime


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

    async def run_subagent(
        self,
        agent_type: str,
        task: str,
        adata: Any,
        context: str = "",
    ) -> dict:
        """Spawn a subagent with restricted tools and its own conversation.

        Returns
        -------
        dict
            ``{"result": str, "adata": AnnData}``
        """
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
            print(
                f"      \U0001f504 [{agent_type}] "
                f"Turn {turn + 1}/{config.max_turns}"
            )

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
                print(
                    f"      \U0001f527 [{agent_type}] "
                    f"{tc.name}({', '.join(f'{k}=' for k in tc.arguments)})"
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
                    print(
                        f"      \u2705 [{agent_type}] "
                        f"Finished: {summary[:120]}"
                    )
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
