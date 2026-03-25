import os

import pytest
from tests.utils._web_test_support import load_service_module

_MODULE = load_service_module("omicverse_web.services.agent_service", "agent_service.py")

get_harness_capabilities = _MODULE.get_harness_capabilities
stream_agent_events = _MODULE.stream_agent_events


_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Harness tests are server-only and require OV_AGENT_RUN_HARNESS_TESTS=1.",
)


class _FakeAgent:
    async def stream_async(self, prompt, adata, cancel_event=None, history=None, approval_handler=None, request_content=None):
        yield {
            "type": "tool_call",
            "content": {"name": "inspect_data", "arguments": {"aspect": "shape"}},
            "trace_id": "trace_123",
            "step_id": "step_123",
            "category": "tool",
        }
        yield {
            "type": "done",
            "content": "done",
            "trace_id": "trace_123",
            "category": "lifecycle",
        }


def test_stream_agent_events_preserves_trace_metadata():
    handle = stream_agent_events(_FakeAgent(), "inspect", None, session_id="sess_web")
    events = []
    for raw in handle:
        if raw.startswith("data: "):
            events.append(raw)

    joined = "\n".join(events)
    assert "trace_123" in joined
    assert "step_123" in joined
    assert "\"category\": \"tool\"" in joined


def test_harness_capabilities_expose_initialize_contract():
    caps = get_harness_capabilities()
    assert caps["version"] >= 1
    assert caps["supports"]["sse_streaming"] is True
    assert caps["supports"]["trace_replay"] is True
    assert caps["supports"]["approval_requests"] is True
    assert caps["supports"]["approval_resume"] is True
    assert caps["supports"]["item_lifecycle"] is True
