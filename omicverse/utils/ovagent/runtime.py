"""Workflow-aware runtime helpers for OVAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .run_store import AnalysisRun, RunStore
from .workflow import WorkflowDocument, load_workflow_document, resolve_repo_root


class OmicVerseRuntime:
    """Minimal workflow/run-store bridge for the existing smart agent."""

    def __init__(
        self,
        *,
        repo_root: Optional[Path] = None,
        workflow_path: Optional[Path] = None,
        run_root: Optional[Path] = None,
    ):
        self.repo_root = repo_root or resolve_repo_root()
        self.workflow_path = workflow_path
        self.workflow = load_workflow_document(self.repo_root, workflow_path)
        self.run_store = RunStore(root=run_root)

    def reload_workflow(self) -> WorkflowDocument:
        self.workflow = load_workflow_document(self.repo_root, self.workflow_path)
        return self.workflow

    def compose_system_prompt(self, base_prompt: str) -> str:
        workflow = self.reload_workflow()
        if not workflow.body and not workflow.config.default_tools:
            return base_prompt
        return base_prompt.rstrip() + "\n\n" + workflow.build_prompt_block()

    def start_analysis_run(
        self,
        *,
        request: str,
        model: str,
        provider: str,
        session_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> AnalysisRun:
        workflow = self.reload_workflow()
        return self.run_store.start_run(
            request=request,
            model=model,
            provider=provider,
            session_id=session_id,
            workflow=workflow,
            metadata=metadata,
        )

    def append_trace(self, run_id: str, trace_id: str) -> AnalysisRun:
        return self.run_store.append_trace(run_id, trace_id)

    def finish_analysis_run(
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
        return self.run_store.finish_run(
            run_id,
            status=status,
            summary=summary,
            trace_id=trace_id,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )
