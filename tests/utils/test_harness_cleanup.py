import os
from pathlib import Path

import pytest

from omicverse.utils.harness import HarnessCleaner, RunTraceStore


_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Harness tests are server-only and require OV_AGENT_RUN_HARNESS_TESTS=1.",
)


def test_cleanup_report_detects_missing_docs(tmp_path):
    repo_root = tmp_path / "repo"
    docs_root = repo_root / "docs" / "harness"
    docs_root.mkdir(parents=True)

    # Intentionally create only one required doc.
    (docs_root / "index.md").write_text("# Harness\n", encoding="utf-8")

    store = RunTraceStore(root=tmp_path / "store")
    cleaner = HarnessCleaner(store=store, docs_root=docs_root, repo_root=repo_root)
    report = cleaner.run()

    assert report.summary["total_findings"] >= 1
    assert any(f.kind == "missing_doc" for f in report.findings)


def test_cleanup_report_can_be_saved(tmp_path):
    repo_root = tmp_path / "repo"
    docs_root = repo_root / "docs" / "harness"
    docs_root.mkdir(parents=True, exist_ok=True)
    for name in ("index.md", "core-beliefs.md", "runtime-contract.md", "server-validation.md", "cleanup-policy.md"):
        (docs_root / name).write_text("# ok\n", encoding="utf-8")

    agent_service = repo_root / "omicverse_web" / "services"
    agent_service.mkdir(parents=True, exist_ok=True)
    (agent_service / "agent_service.py").write_text("AGENT_EVENT_TYPES = ('x',)\n", encoding="utf-8")

    store = RunTraceStore(root=tmp_path / "store")
    cleaner = HarnessCleaner(store=store, docs_root=docs_root, repo_root=repo_root)
    report = cleaner.run()
    path = cleaner.save_report(report)

    assert path.exists()
    assert path.suffix == ".json"
    assert any(f.kind == "duplicate_contract" for f in report.findings)
