"""Static consistency tests for MCP CI version matrix infrastructure.

Verifies that the version report script, verified stacks documentation, and
workflow integration are consistent.  No network access, no heavy imports.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/mcp/ -> repo root

PROFILE_NAMES = [
    "fast-mock",
    "core-runtime",
    "scientific-runtime",
    "extended-runtime",
]

STATUS_TERMS = ["verified", "constrained", "best_effort"]

_VERSION_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "mcp-report-versions.py"


# ---------------------------------------------------------------------------
# Version report script
# ---------------------------------------------------------------------------


class TestVersionReportScript:
    """Verify scripts/ci/mcp-report-versions.py exists and works."""

    def test_version_report_script_exists(self):
        assert _VERSION_SCRIPT.exists(), f"Missing {_VERSION_SCRIPT}"

    def test_version_report_script_runs(self):
        result = subprocess.run(
            [sys.executable, str(_VERSION_SCRIPT), "--profile", "fast-mock"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Version report failed: {result.stderr}"
        )
        data = json.loads(result.stdout)
        assert isinstance(data, dict)

    def test_version_report_output_schema(self):
        result = subprocess.run(
            [sys.executable, str(_VERSION_SCRIPT), "--profile", "core-runtime"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)

        required_keys = {
            "schema_version", "profile", "timestamp",
            "platform", "python", "packages",
        }
        assert required_keys.issubset(data.keys()), (
            f"Missing keys: {required_keys - data.keys()}"
        )
        assert data["profile"] == "core-runtime"
        assert isinstance(data["packages"], dict)
        assert data["schema_version"] == 1

    def test_version_report_all_profiles_accepted(self):
        for profile in PROFILE_NAMES:
            result = subprocess.run(
                [sys.executable, str(_VERSION_SCRIPT), "--profile", profile],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, (
                f"Profile {profile!r} rejected: {result.stderr}"
            )

    def test_version_report_invalid_profile_rejected(self):
        result = subprocess.run(
            [sys.executable, str(_VERSION_SCRIPT), "--profile", "nonexistent"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0, (
            "Invalid profile should be rejected"
        )


# ---------------------------------------------------------------------------
# Verified stacks documentation
# ---------------------------------------------------------------------------


class TestVerifiedStacksDoc:
    """Verify docs/mcp_verified_stacks.md exists and is complete."""

    @pytest.fixture(autouse=True)
    def _load_doc(self):
        self._doc_path = _REPO_ROOT / "docs" / "mcp_verified_stacks.md"
        assert self._doc_path.exists(), f"Missing {self._doc_path}"
        self._doc = self._doc_path.read_text()

    def test_verified_stacks_doc_exists(self):
        assert len(self._doc) > 100

    def test_verified_stacks_contains_status_terms(self):
        for term in STATUS_TERMS:
            assert term in self._doc, (
                f"Status term {term!r} missing from mcp_verified_stacks.md"
            )

    def test_verified_stacks_mentions_all_profiles(self):
        for name in PROFILE_NAMES:
            assert name in self._doc, (
                f"Profile {name!r} missing from mcp_verified_stacks.md"
            )

    def test_ci_profiles_doc_cross_references_verified_stacks(self):
        doc_path = _REPO_ROOT / "docs" / "mcp_ci_profiles.md"
        assert doc_path.exists()
        content = doc_path.read_text()
        assert "mcp_verified_stacks.md" in content, (
            "mcp_ci_profiles.md does not cross-reference mcp_verified_stacks.md"
        )


# ---------------------------------------------------------------------------
# Workflow integration
# ---------------------------------------------------------------------------


class TestWorkflowVersionIntegration:
    """Verify the workflow uploads version artifacts."""

    @pytest.fixture(autouse=True)
    def _load_workflow(self):
        self._wf_path = (
            _REPO_ROOT / ".github" / "workflows" / "mcp-tests.yml"
        )
        assert self._wf_path.exists(), f"Missing {self._wf_path}"
        self._wf = self._wf_path.read_text()

    def test_workflow_has_artifact_upload(self):
        assert "upload-artifact" in self._wf, (
            "Workflow missing upload-artifact step"
        )

    def test_workflow_path_triggers_include_version_script(self):
        assert "mcp-report-versions.py" in self._wf, (
            "Workflow path triggers do not include mcp-report-versions.py"
        )


# ---------------------------------------------------------------------------
# Script integration
# ---------------------------------------------------------------------------


class TestScriptsCallVersionReport:
    """Verify all CI scripts call the version report tool."""

    def test_scripts_call_version_report(self):
        for profile in PROFILE_NAMES:
            script_path = (
                _REPO_ROOT / "scripts" / "ci" / f"mcp-{profile}.sh"
            )
            assert script_path.exists(), f"Missing script: {script_path}"
            content = script_path.read_text()
            assert "mcp-report-versions.py" in content, (
                f"Script {script_path.name} does not call mcp-report-versions.py"
            )
