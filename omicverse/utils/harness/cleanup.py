"""
Recurring cleanup and drift-report helpers for the OVAgent harness.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .trace_store import RunTraceStore


@dataclass
class CleanupFinding:
    kind: str
    message: str
    path: str = ""
    severity: str = "info"


@dataclass
class CleanupReport:
    generated_at: float
    findings: list[CleanupFinding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "findings": [asdict(f) for f in self.findings],
            "summary": self.summary,
        }


class HarnessCleaner:
    """Generate lightweight cleanup/drift reports for harness assets."""

    def __init__(
        self,
        *,
        store: Optional[RunTraceStore] = None,
        docs_root: Optional[Path] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.store = store or RunTraceStore()
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.docs_root = docs_root or (self.repo_root / "docs" / "harness")

    def run(self) -> CleanupReport:
        report = CleanupReport(generated_at=time.time())
        report.findings.extend(self._find_missing_harness_docs())
        report.findings.extend(self._find_orphaned_trace_files())
        report.findings.extend(self._find_duplicate_web_event_contract())
        report.summary = {
            "total_findings": len(report.findings),
            "severities": self._count_by("severity", report.findings),
            "kinds": self._count_by("kind", report.findings),
        }
        return report

    def save_report(self, report: CleanupReport, name: str = "cleanup_report") -> Path:
        return self.store.save_report(name, report.to_dict())

    def _find_missing_harness_docs(self) -> list[CleanupFinding]:
        findings: list[CleanupFinding] = []
        required = [
            "index.md",
            "core-beliefs.md",
            "runtime-contract.md",
            "server-validation.md",
            "cleanup-policy.md",
        ]
        for rel in required:
            path = self.docs_root / rel
            if not path.exists():
                findings.append(
                    CleanupFinding(
                        kind="missing_doc",
                        severity="warning",
                        path=str(path),
                        message=f"missing harness system-of-record doc: {rel}",
                    )
                )
        return findings

    def _find_orphaned_trace_files(self) -> list[CleanupFinding]:
        findings: list[CleanupFinding] = []
        for path in self.store.list_recent(limit=200):
            if path.stat().st_size == 0:
                findings.append(
                    CleanupFinding(
                        kind="empty_trace",
                        severity="warning",
                        path=str(path),
                        message="empty trace file should be removed or regenerated",
                    )
                )
        return findings

    def _find_duplicate_web_event_contract(self) -> list[CleanupFinding]:
        findings: list[CleanupFinding] = []
        candidates = [
            self.repo_root / "omicclaw" / "services" / "agent_service.py",
            self.repo_root / "omicverse_web" / "services" / "agent_service.py",
        ]
        agent_service = next((path for path in candidates if path.exists()), None)
        if agent_service is None:
            return findings
        text = agent_service.read_text(encoding="utf-8")
        if "AGENT_EVENT_TYPES =" in text or "class AgentEvent" in text:
            findings.append(
                CleanupFinding(
                    kind="duplicate_contract",
                    severity="warning",
                    path=str(agent_service),
                    message="web service still defines its own agent event contract instead of importing the core harness schema",
                )
            )
        return findings

    @staticmethod
    def _count_by(key: str, findings: list[CleanupFinding]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in findings:
            value = getattr(finding, key)
            counts[value] = counts.get(value, 0) + 1
        return counts
