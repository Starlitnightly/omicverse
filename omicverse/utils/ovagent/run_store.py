"""File-backed analysis run store for OVAgent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
import uuid
from typing import Any, Optional

from .workflow import WorkflowDocument


def _now() -> float:
    return time.time()


def _make_run_id() -> str:
    return "run_" + uuid.uuid4().hex[:12]


@dataclass
class AnalysisRun:
    run_id: str
    request: str
    model: str
    provider: str
    session_id: str = ""
    status: str = "started"
    summary: str = ""
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    finished_at: Optional[float] = None
    trace_ids: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    workflow: dict[str, Any] = field(default_factory=dict)

    def finish(
        self,
        *,
        status: str,
        summary: str = "",
        trace_id: str = "",
        artifacts: Optional[list[dict[str, Any]]] = None,
        warnings: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.status = status
        self.summary = summary or self.summary
        self.finished_at = _now()
        self.updated_at = self.finished_at
        if trace_id:
            self.add_trace(trace_id)
        if artifacts:
            self.artifacts.extend(artifacts)
        if warnings:
            self.warnings.extend(warnings)
        if metadata:
            merged = dict(self.metadata)
            merged.update(metadata)
            self.metadata = merged

    def add_trace(self, trace_id: str) -> None:
        if trace_id and trace_id not in self.trace_ids:
            self.trace_ids.append(trace_id)
            self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnalysisRun":
        return cls(**payload)


class RunStore:
    """Persist analysis runs under ``~/.ovagent/runs`` by default."""

    def __init__(self, root: Optional[Path] = None):
        self.root = (root or (Path.home() / ".ovagent" / "runs")).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def _manifest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "bundle.json"

    def _summary_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "summary.md"

    def start_run(
        self,
        *,
        request: str,
        model: str,
        provider: str,
        session_id: str = "",
        workflow: Optional[WorkflowDocument] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AnalysisRun:
        run = AnalysisRun(
            run_id=_make_run_id(),
            request=request,
            model=model,
            provider=provider,
            session_id=session_id,
            metadata=dict(metadata or {}),
            workflow=workflow.to_dict() if workflow is not None else {},
        )
        run_dir = self._run_dir(run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        if workflow is not None and workflow.raw_text:
            (run_dir / "WORKFLOW.snapshot.md").write_text(workflow.raw_text, encoding="utf-8")
        self.save_run(run)
        return run

    def save_run(self, run: AnalysisRun) -> None:
        manifest_path = self._manifest_path(run.run_id)
        manifest_path.write_text(
            json.dumps(run.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._summary_path(run.run_id).write_text(
            self._build_summary_markdown(run),
            encoding="utf-8",
        )

    def load_run(self, run_id: str) -> AnalysisRun:
        payload = json.loads(self._manifest_path(run_id).read_text(encoding="utf-8"))
        return AnalysisRun.from_dict(payload)

    def append_trace(self, run_id: str, trace_id: str) -> AnalysisRun:
        run = self.load_run(run_id)
        run.add_trace(trace_id)
        self.save_run(run)
        return run

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        summary: str = "",
        trace_id: str = "",
        artifacts: Optional[list[dict[str, Any]]] = None,
        warnings: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AnalysisRun:
        run = self.load_run(run_id)
        run.finish(
            status=status,
            summary=summary,
            trace_id=trace_id,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )
        self.save_run(run)
        return run

    def list_runs(self, limit: int = 20) -> list[AnalysisRun]:
        manifests = sorted(self.root.glob("run_*/bundle.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        runs: list[AnalysisRun] = []
        for manifest in manifests[:limit]:
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                runs.append(AnalysisRun.from_dict(payload))
            except Exception:
                continue
        return runs

    def build_bundle(self, run_id: str) -> dict[str, Any]:
        run = self.load_run(run_id)
        return {
            "run": run.to_dict(),
            "paths": {
                "run_dir": str(self._run_dir(run_id)),
                "bundle": str(self._manifest_path(run_id)),
                "summary": str(self._summary_path(run_id)),
            },
        }

    def _build_summary_markdown(self, run: AnalysisRun) -> str:
        lines = [
            f"# Analysis Run {run.run_id}",
            "",
            f"- Status: `{run.status}`",
            f"- Model: `{run.model}`",
            f"- Provider: `{run.provider}`",
        ]
        if run.session_id:
            lines.append(f"- Session: `{run.session_id}`")
        lines.extend([
            f"- Created: `{run.created_at}`",
            f"- Updated: `{run.updated_at}`",
            "",
            "## Request",
            "",
            run.request,
            "",
        ])
        if run.summary:
            lines.extend([
                "## Summary",
                "",
                run.summary,
                "",
            ])
        if run.trace_ids:
            lines.extend([
                "## Traces",
                "",
                *[f"- `{trace_id}`" for trace_id in run.trace_ids],
                "",
            ])
        if run.artifacts:
            lines.extend([
                "## Artifacts",
                "",
                *[
                    f"- `{artifact.get('kind', 'artifact')}`: {artifact.get('label') or artifact.get('path') or artifact.get('uri') or artifact.get('description', '')}"
                    for artifact in run.artifacts
                ],
                "",
            ])
        if run.warnings:
            lines.extend([
                "## Warnings",
                "",
                *[f"- {warning}" for warning in run.warnings],
                "",
            ])
        return "\n".join(lines).strip() + "\n"
