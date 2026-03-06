import os

import pytest

from omicverse.utils.harness.server_cli import (
    load_trace_payload,
    run_cleanup,
    run_scenario,
)
from omicverse.utils.harness.trace_store import RunTraceRecorder, RunTraceStore


_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Harness tests are server-only and require OV_AGENT_RUN_HARNESS_TESTS=1.",
)


def test_server_cli_helpers_round_trip(tmp_path):
    store = RunTraceStore(root=tmp_path / "harness")
    recorder = RunTraceRecorder(
        request="qc",
        model="gpt-5.2",
        provider="openai",
        session_id="sess_cli",
    )
    recorder.add_step("tool_call", name="inspect_data", summary="inspect", data={"arguments": {"aspect": "shape"}})
    recorder.finish(status="success", success=True, summary="done")
    recorder.save(store)

    loaded = load_trace_payload(recorder.trace.trace_id, root=str(store.root))
    scenario = run_scenario(
        recorder.trace.trace_id,
        scenario_name="cli-smoke",
        expected_tools=["inspect_data"],
        root=str(store.root),
    )
    cleanup = run_cleanup(
        trace_root=str(store.root),
        docs_root=str(tmp_path / "missing_docs"),
        repo_root=str(tmp_path / "repo"),
        save_report=True,
    )

    assert loaded["trace_id"] == recorder.trace.trace_id
    assert scenario["passed"] is True
    assert cleanup["summary"]["total_findings"] >= 1
    assert "report_path" in cleanup
