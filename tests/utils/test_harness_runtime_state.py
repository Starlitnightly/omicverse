import os
import threading

import pytest
from tests.utils._web_test_support import load_service_module


_AGENT_SERVICE = load_service_module("omicverse_web.services.agent_service", "agent_service.py")
_SESSION_SERVICE = load_service_module(
    "omicverse_web.services.agent_session_service",
    "agent_session_service.py",
)

get_harness_capabilities = _AGENT_SERVICE.get_harness_capabilities
AgentSession = _SESSION_SERVICE.AgentSession
ApprovalRequest = _SESSION_SERVICE.ApprovalRequest

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Harness tests are server-only and require OV_AGENT_RUN_HARNESS_TESTS=1.",
)


def test_agent_session_summary_tracks_runtime_state():
    session = AgentSession(session_id="ses_tool_state")
    session.add_message("user", "inspect", turn_id="turn_1")
    session.register_turn("turn_1", threading.Event())
    session.register_trace("trace_1")
    session.register_approval(ApprovalRequest(
        approval_id="approval_1",
        turn_id="turn_1",
        session_id="ses_tool_state",
        title="Run command",
        message="Need approval before running a risky tool.",
    ))

    summary = session.to_summary()

    assert summary["session_id"] == "ses_tool_state"
    assert summary["message_count"] == 1
    assert summary["active_turn_id"] == "turn_1"
    assert summary["trace_count"] == 1
    assert summary["last_trace_id"] == "trace_1"
    assert summary["pending_approvals"] == 1


def test_agent_session_approval_lifecycle_updates_pending_counts():
    session = AgentSession(session_id="ses_approval")
    approval = ApprovalRequest(
        approval_id="approval_2",
        turn_id="turn_2",
        session_id="ses_approval",
        title="Execution approval required",
        message="Generated code requires approval before execution.",
    )
    session.register_approval(approval)

    assert len(session.get_pending_approvals()) == 1

    resolved = session.resolve_approval("approval_2", "approve")

    assert resolved is not None
    assert resolved["status"] == "approved"
    assert resolved["decision"] == "approve"
    assert session.get_pending_approvals() == []


def test_harness_handshake_payload_covers_runtime_state_and_server_gate():
    session = AgentSession(session_id="ses_handshake")
    session.register_trace("trace_handshake")
    payload = {
        "capabilities": get_harness_capabilities(),
        "session": session.to_summary(),
    }

    supports = payload["capabilities"]["supports"]

    assert payload["capabilities"]["version"] >= 1
    assert payload["capabilities"]["server_only_validation"] is True
    assert payload["capabilities"]["harness_test_env"] == "OV_AGENT_RUN_HARNESS_TESTS"
    assert supports["sse_streaming"] is True
    assert supports["trace_replay"] is True
    assert supports["session_history"] is True
    assert supports["turn_cancel"] is True
    assert supports["item_lifecycle"] is True
    assert supports["approval_requests"] is True
    assert supports["approval_resume"] is True
    assert payload["session"]["session_id"] == "ses_handshake"
    assert payload["session"]["last_trace_id"] == "trace_handshake"


def test_harness_capabilities_include_lifecycle_events_needed_by_web_clients():
    caps = get_harness_capabilities()
    event_types = set(caps["event_types"])

    assert {"item_started", "item_completed", "approval_request", "approval_resolved"}.issubset(event_types)
