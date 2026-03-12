"""Static consistency tests for MCP verified status documentation and tooling.

Verifies that the verified process document exists, is cross-referenced,
and that the version report script supports the source field.
No network access, no heavy imports.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/mcp/ -> repo root


# ---------------------------------------------------------------------------
# Verified process document
# ---------------------------------------------------------------------------


class TestVerifiedProcessDoc:
    """Verify docs/mcp_verified_process.md exists and has required content."""

    @pytest.fixture(autouse=True)
    def _load_doc(self):
        self._doc_path = _REPO_ROOT / "docs" / "mcp_verified_process.md"
        assert self._doc_path.exists(), f"Missing {self._doc_path}"
        self._doc = self._doc_path.read_text()

    def test_verified_process_doc_exists(self):
        assert len(self._doc) > 200

    def test_verified_process_mentions_core_runtime(self):
        assert "core-runtime" in self._doc, (
            "mcp_verified_process.md does not mention core-runtime"
        )

    def test_verified_process_has_upgrade_checklist(self):
        doc_lower = self._doc.lower()
        has_upgrade = "upgrade" in doc_lower or "promotion" in doc_lower
        has_checklist = "checklist" in doc_lower or "- [ ]" in self._doc
        assert has_upgrade and has_checklist, (
            "mcp_verified_process.md missing upgrade/promotion checklist"
        )

    def test_verified_process_has_rollback_section(self):
        doc_lower = self._doc.lower()
        assert (
            "rollback" in doc_lower
            or "demotion" in doc_lower
            or "demote" in doc_lower
        ), "mcp_verified_process.md missing rollback/demotion section"


# ---------------------------------------------------------------------------
# Verified stacks promotion content
# ---------------------------------------------------------------------------


class TestVerifiedStacksPromotionContent:
    """Verify mcp_verified_stacks.md has promotion-related content."""

    @pytest.fixture(autouse=True)
    def _load_doc(self):
        self._doc_path = _REPO_ROOT / "docs" / "mcp_verified_stacks.md"
        assert self._doc_path.exists(), f"Missing {self._doc_path}"
        self._doc = self._doc_path.read_text()

    def test_verified_stacks_has_promotion_checklist(self):
        doc_lower = self._doc.lower()
        assert "core-runtime" in self._doc, (
            "mcp_verified_stacks.md does not mention core-runtime"
        )
        has_promotion = (
            "promot" in doc_lower
            or "candidate" in doc_lower
            or "pending" in doc_lower
        )
        assert has_promotion, (
            "mcp_verified_stacks.md missing promotion/candidate info "
            "for core-runtime"
        )

    def test_verified_stacks_artifact_naming(self):
        assert "mcp-versions-core-runtime" in self._doc, (
            "mcp_verified_stacks.md does not mention artifact naming "
            "pattern 'mcp-versions-core-runtime'"
        )

    def test_verified_stacks_cross_references_process_doc(self):
        assert "mcp_verified_process.md" in self._doc, (
            "mcp_verified_stacks.md does not cross-reference "
            "mcp_verified_process.md"
        )


# ---------------------------------------------------------------------------
# Workflow artifact consistency
# ---------------------------------------------------------------------------


class TestWorkflowArtifactConsistency:
    """Verify workflow artifact names match documentation."""

    def test_workflow_core_runtime_artifact_name_matches_doc(self):
        wf_path = (
            _REPO_ROOT / ".github" / "workflows" / "mcp-tests.yml"
        )
        assert wf_path.exists()
        wf = wf_path.read_text()

        doc_path = _REPO_ROOT / "docs" / "mcp_verified_stacks.md"
        assert doc_path.exists()
        doc = doc_path.read_text()

        artifact_name = "mcp-versions-core-runtime-py"
        assert artifact_name in wf, (
            f"Workflow missing artifact name pattern: {artifact_name}"
        )
        assert "mcp-versions-core-runtime" in doc, (
            "Verified stacks doc missing artifact name pattern"
        )


# ---------------------------------------------------------------------------
# Version report source field
# ---------------------------------------------------------------------------


class TestVersionReportSourceField:
    """Verify the version report script supports the --source flag."""

    def test_version_report_source_field(self):
        script = _REPO_ROOT / "scripts" / "ci" / "mcp-report-versions.py"
        assert script.exists()

        result = subprocess.run(
            [
                sys.executable, str(script),
                "--profile", "fast-mock",
                "--source", "ci",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Version report with --source failed: {result.stderr}"
        )
        data = json.loads(result.stdout)
        assert "source" in data, (
            "Version report output missing 'source' field"
        )
        assert data["source"] == "ci"

    def test_version_report_source_default_is_local(self):
        script = _REPO_ROOT / "scripts" / "ci" / "mcp-report-versions.py"
        result = subprocess.run(
            [sys.executable, str(script), "--profile", "fast-mock"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data.get("source") == "local", (
            "Default source should be 'local'"
        )
