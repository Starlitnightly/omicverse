import os

import pytest

from omicverse.utils.harness import (
    RunTraceRecorder,
    RunTraceStore,
    build_stream_event,
)


_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Harness tests are server-only and require OV_AGENT_RUN_HARNESS_TESTS=1.",
)


def test_build_stream_event_preserves_trace_metadata():
    event = build_stream_event(
        "tool_call",
        {"name": "inspect_data", "arguments": {"aspect": "shape"}},
        turn_id="turn_x",
        session_id="sess_x",
        trace_id="trace_x",
        step_id="step_x",
        category="tool",
        latency_ms=12.5,
    )

    assert event["type"] == "tool_call"
    assert event["turn_id"] == "turn_x"
    assert event["session_id"] == "sess_x"
    assert event["trace_id"] == "trace_x"
    assert event["step_id"] == "step_x"
    assert event["category"] == "tool"
    assert event["latency_ms"] == 12.5


def test_run_trace_recorder_round_trip(tmp_path):
    store = RunTraceStore(root=tmp_path / "harness")
    recorder = RunTraceRecorder(
        request="run qc",
        model="gpt-5.2",
        provider="openai",
        session_id="sess_1",
        adata_shape=(100, 200),
        history_size=2,
    )

    recorder.add_event(build_stream_event("status", "started", trace_id=recorder.trace.trace_id))
    recorder.add_step("tool_call", name="inspect_data", summary="inspect dispatched", data={"arguments": {"aspect": "shape"}})
    recorder.finish(status="success", success=True, summary="completed")
    path = recorder.save(store)

    loaded = store.load(recorder.trace.trace_id)
    assert path.exists()
    assert loaded["trace_id"] == recorder.trace.trace_id
    assert loaded["session_id"] == "sess_1"
    assert loaded["adata_shape"] == [100, 200]
    assert loaded["history_size"] == 2
    assert loaded["status"] == "success"


def test_run_trace_store_rejects_path_traversal(tmp_path):
    store = RunTraceStore(root=tmp_path / "harness")
    outside = tmp_path / "secret.json"
    outside.write_text('{"secret": true}', encoding="utf-8")

    with pytest.raises(ValueError):
        store.load("../secret")
