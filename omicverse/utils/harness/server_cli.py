"""
Server-only harness CLI helpers for replay, scenario evaluation, and cleanup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .cleanup import HarnessCleaner
from .contracts import ScenarioSpec
from .trace_store import RunTraceStore, ScenarioRunner


SERVER_ONLY_ENV = "OV_AGENT_RUN_HARNESS_TESTS"


def require_server_only_mode() -> None:
    enabled = os.environ.get(SERVER_ONLY_ENV, "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        raise RuntimeError(
            f"Harness replay/scenario/cleanup commands are server-only. "
            f"Set {SERVER_ONLY_ENV}=1 in the dedicated validation environment."
        )


def make_trace_store(root: Optional[str] = None) -> RunTraceStore:
    return RunTraceStore(root=Path(root).expanduser() if root else None)


def load_trace_payload(trace_id: str, *, root: Optional[str] = None) -> dict[str, Any]:
    store = make_trace_store(root)
    return store.load(trace_id)


def summarize_trace(payload: dict[str, Any]) -> dict[str, Any]:
    tools = []
    for step in payload.get("steps", []):
        tool_name = step.get("name") or step.get("tool_name")
        if tool_name:
            tools.append(tool_name)
    return {
        "trace_id": payload.get("trace_id", ""),
        "turn_id": payload.get("turn_id", ""),
        "session_id": payload.get("session_id", ""),
        "request": payload.get("request", ""),
        "model": payload.get("model", ""),
        "provider": payload.get("provider", ""),
        "status": payload.get("status", ""),
        "result_summary": payload.get("result_summary", ""),
        "tool_names": tools,
        "event_count": len(payload.get("events", [])),
        "step_count": len(payload.get("steps", [])),
        "artifact_count": len(payload.get("artifacts", [])),
    }


def run_scenario(
    trace_id: str,
    *,
    scenario_name: str,
    expected_events: Optional[list[str]] = None,
    expected_tools: Optional[list[str]] = None,
    root: Optional[str] = None,
) -> dict[str, Any]:
    payload = load_trace_payload(trace_id, root=root)
    runner = ScenarioRunner()
    result = runner.evaluate(
        ScenarioSpec(
            name=scenario_name,
            expected_event_types=list(expected_events or []),
            expected_tool_names=list(expected_tools or []),
        ),
        payload,
    )
    return {
        "scenario_name": result.scenario_name,
        "trace_id": result.trace_id,
        "passed": result.passed,
        "issues": result.issues,
        "metadata": result.metadata,
    }


def run_cleanup(
    *,
    trace_root: Optional[str] = None,
    docs_root: Optional[str] = None,
    repo_root: Optional[str] = None,
    save_report: bool = False,
    report_name: str = "cleanup_report",
) -> dict[str, Any]:
    store = make_trace_store(trace_root)
    cleaner = HarnessCleaner(
        store=store,
        docs_root=Path(docs_root).expanduser() if docs_root else None,
        repo_root=Path(repo_root).expanduser() if repo_root else None,
    )
    report = cleaner.run()
    payload = report.to_dict()
    if save_report:
        report_path = cleaner.save_report(report, name=report_name)
        payload["report_path"] = str(report_path)
    return payload


def dump_json(payload: dict[str, Any], output: Optional[str] = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        Path(output).expanduser().write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
