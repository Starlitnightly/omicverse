"""
Trace persistence and replay helpers for OVAgent harness runs.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from .contracts import (
    ArtifactRef,
    HarnessEvent,
    RunTrace,
    ScenarioResult,
    ScenarioSpec,
    StepTrace,
    make_step_id,
    make_trace_id,
    make_turn_id,
)


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if hasattr(value, "shape"):
        return {
            "repr": type(value).__name__,
            "shape": tuple(getattr(value, "shape", ())),
        }
    return repr(value)


class RunTraceStore:
    """Filesystem-backed store for run traces and cleanup reports."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or (Path.home() / ".ovagent" / "harness")
        self.traces_dir = self.root / "traces"
        self.reports_dir = self.root / "reports"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: RunTrace) -> Path:
        path = self.traces_dir / f"{trace.trace_id}.json"
        path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, trace_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", trace_id):
            raise ValueError("Invalid trace identifier")
        path = self.traces_dir / f"{trace_id}.json"
        traces_root = self.traces_dir.resolve()
        candidate = path.resolve()
        if candidate.parent != traces_root:
            raise ValueError("Invalid trace path")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_recent(self, limit: int = 20) -> list[Path]:
        files = sorted(self.traces_dir.glob("trace_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[:limit]

    def save_report(self, name: str, payload: dict[str, Any]) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.reports_dir / f"{name}_{ts}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def build_context_for_prompt(self, limit: int = 3) -> str:
        files = self.list_recent(limit=max(limit * 3, limit))
        lines = ["## Recent failing agent traces"]
        found = 0
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("status") == "success":
                continue
            lines.append(f"- Request: {payload.get('request', '')}")
            lines.append(f"  Error: {payload.get('result_summary', payload.get('status', 'unknown'))}")
            found += 1
            if found >= limit:
                break
        return "\n".join(lines) if found else ""


class RunTraceRecorder:
    """In-memory recorder that captures events and step-level traces."""

    def __init__(
        self,
        *,
        request: str,
        model: str,
        provider: str,
        adata_shape: Optional[tuple[int, int]] = None,
        turn_id: Optional[str] = None,
        session_id: str = "",
        history_size: int = 0,
    ) -> None:
        self.trace = RunTrace(
            trace_id=make_trace_id(),
            turn_id=turn_id or make_turn_id(),
            request=request,
            model=model,
            provider=provider,
            session_id=session_id,
            adata_shape=adata_shape,
            history_size=history_size,
        )
        self._active_steps: dict[str, StepTrace] = {}

    def add_event(self, payload: dict[str, Any]) -> None:
        self.trace.events.append(_safe_json(HarnessEvent.from_dict(payload).to_dict()))

    def start_step(
        self,
        step_type: str,
        *,
        name: str = "",
        summary: str = "",
        data: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        step_id: Optional[str] = None,
    ) -> str:
        actual_step_id = step_id or make_step_id()
        payload = dict(data or {})
        if metadata:
            payload["metadata"] = dict(metadata)
        step = StepTrace(
            step_id=actual_step_id,
            step_type=step_type,
            name=name,
            summary=summary,
            data=payload,
        )
        self._active_steps[actual_step_id] = step
        return actual_step_id

    def add_step(
        self,
        step_type: str,
        *,
        name: str = "",
        summary: str = "",
        data: Optional[dict[str, Any]] = None,
    ) -> str:
        step = StepTrace(
            step_id=make_step_id(),
            step_type=step_type,
            name=name,
            summary=summary,
            data=dict(data or {}),
        )
        self.trace.steps.append(step)
        return step.step_id

    def finish_step(
        self,
        step_id: str,
        *,
        status: str,
        output_summary: str = "",
        error: str = "",
        artifact_refs: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        step = self._active_steps.pop(step_id, None)
        if step is None:
            return
        step.finish(
            status=status,
            output_summary=output_summary,
            error=error,
            artifact_refs=artifact_refs,
            metadata=metadata,
        )
        self.trace.steps.append(step)
        if artifact_refs:
            self.trace.artifacts.extend(artifact_refs)

    def add_artifact(
        self,
        kind: str,
        *,
        uri: str = "",
        path: str = "",
        label: str = "",
        description: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.trace.artifacts.append(
            ArtifactRef(
                kind=kind,
                label=label,
                path=path or uri,
                uri=uri,
                description=description,
                metadata=dict(metadata or {}),
            )
        )

    def finish(
        self,
        *,
        status: str,
        success: bool = False,
        summary: str = "",
        result_summary: str = "",
        error: str = "",
        usage: Optional[dict[str, Any]] = None,
        usage_breakdown: Optional[dict[str, Any]] = None,
    ) -> None:
        self.trace.finish(
            status=status,
            success=success,
            summary=summary,
            result_summary=result_summary,
            error=error,
            usage=usage,
            usage_breakdown=usage_breakdown,
        )

    def save(self, store: RunTraceStore) -> Path:
        return store.save(self.trace)


class ScenarioRunner:
    """Evaluate a stored trace against a simple scenario spec."""

    def evaluate(self, spec: ScenarioSpec, trace: RunTrace | dict[str, Any]) -> ScenarioResult:
        payload = trace if isinstance(trace, dict) else trace.to_dict()
        issues: list[str] = []

        actual_events = [event.get("type") for event in payload.get("events", [])]
        actual_tools = [
            step.get("name") or step.get("tool_name")
            for step in payload.get("steps", [])
            if step.get("name") or step.get("tool_name")
        ]

        for expected in spec.expected_event_types:
            if expected not in actual_events:
                issues.append(f"missing event type: {expected}")
        for expected in spec.expected_tool_names:
            if expected not in actual_tools:
                issues.append(f"missing tool: {expected}")

        return ScenarioResult(
            scenario_name=spec.name,
            trace_id=payload.get("trace_id", ""),
            passed=not issues,
            issues=issues,
            metadata=dict(spec.metadata),
        )
