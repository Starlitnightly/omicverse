import os
import importlib.util
from pathlib import Path
import sys
import types

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_WEB_ROOT = _ROOT / "omicverse_web"
_SERVICES_ROOT = _WEB_ROOT / "services"
_SERVICE_PATH = _SERVICES_ROOT / "agent_service.py"

web_pkg = types.ModuleType("omicverse_web")
web_pkg.__path__ = [str(_WEB_ROOT)]
services_pkg = types.ModuleType("omicverse_web.services")
services_pkg.__path__ = [str(_SERVICES_ROOT)]
sys.modules.setdefault("omicverse_web", web_pkg)
sys.modules.setdefault("omicverse_web.services", services_pkg)

_SPEC = importlib.util.spec_from_file_location("omicverse_web.services.agent_service", _SERVICE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

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
    async def stream_async(self, prompt, adata, cancel_event=None, history=None, approval_handler=None):
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
