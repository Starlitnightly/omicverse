import os

import pytest

from omicverse.utils.context_compactor import ContextCompactor, HANDOFF_PROMPT


_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Harness tests are server-only and require OV_AGENT_RUN_HARNESS_TESTS=1.",
)


class _FakeLLM:
    def __init__(self):
        self.prompts = []

    async def run(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "compacted summary"


@pytest.mark.asyncio
async def test_context_compactor_wraps_summary_with_handoff_prompt():
    llm = _FakeLLM()
    compactor = ContextCompactor(llm, "gpt-5")

    result = await compactor.compact_bundle("A" * 200)

    assert result.summary == "compacted summary"
    assert result.handoff_text.startswith(HANDOFF_PROMPT)
    assert "compacted summary" in result.handoff_text
    assert result.original_tokens > 0
    assert result.compacted_tokens > 0
    assert result.compacted_tokens >= len(result.summary.split())
    assert llm.prompts and "Source context:" in llm.prompts[0]
