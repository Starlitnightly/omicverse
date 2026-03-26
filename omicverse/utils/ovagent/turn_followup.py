"""Follow-up gate and convergence monitor for the agentic turn loop.

Extracted from ``turn_controller.py`` during Phase 2 decomposition.
FollowUpGate provides stateless helpers for the follow-up / retry heuristic.
ConvergenceMonitor detects read-only-tool plateaus and injects escalating
steering messages.
"""

from __future__ import annotations

import re
from typing import Any, List

from ..harness.tool_catalog import normalize_tool_name


# ---------------------------------------------------------------------------
# Follow-up gate helper
# ---------------------------------------------------------------------------

class FollowUpGate:
    """Stateless helpers for the follow-up / retry heuristic."""

    NON_COMPLETING_TOOLS = frozenset({
        "inspectdata",
        "runsnippet",
        "searchfunctions",
        "searchskills",
        "toolsearch",
        "read",
        "glob",
        "grep",
        "ls",
        "taskget",
        "tasklist",
        "taskoutput",
        "enterplanmode",
        "exitplanmode",
    })

    URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
    ACTION_REQUEST_PATTERN = re.compile(
        r"\b(analy[sz]e|download|fetch|get|open|read|inspect|load|run|"
        r"execute|search|lookup|look up|find|process|parse|clone|fix|"
        r"edit|write|plot|draw|figure|visuali[sz]e|visualization|"
        r"chart|graph|image|png|umap|tsne|heatmap|send)\b",
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
                "\u6253\u5f00", "\u641c\u7d22", "\u7ed8\u56fe",
                "\u753b\u56fe", "\u56fe", "\u56fe\u7247",
                "\u53d1\u56fe", "\u53d1\u9001", "umap",
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

    @classmethod
    def tool_counts_as_meaningful_progress(cls, tool_name: str) -> bool:
        canonical = normalize_tool_name(tool_name) or tool_name or ""
        key = str(canonical).replace("_", "").lower()
        return key not in cls.NON_COMPLETING_TOOLS


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
                "Do NOT call run_snippet again \u2014 it cannot save files."
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
