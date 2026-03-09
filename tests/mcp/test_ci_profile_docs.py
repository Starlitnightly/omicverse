"""Consistency tests for CI profile documentation, scripts, and workflow.

Verifies that profile names, markers, and commands stay in sync across
documentation, shell scripts, and the GitHub Actions workflow file.
No CI platform connectivity required — all checks are local file reads.
"""

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Constants — single source of truth for profile definitions
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/mcp/ → repo root

PROFILE_NAMES = [
    "fast-mock",
    "core-runtime",
    "scientific-runtime",
    "extended-runtime",
]

PROFILE_MARKERS = {
    "fast-mock": "not real_runtime",
    "core-runtime": "core",
    "scientific-runtime": "scientific",
    "extended-runtime": "extended",
}


# ---------------------------------------------------------------------------
# Documentation consistency
# ---------------------------------------------------------------------------


class TestCIProfileDocs:
    """Verify docs/mcp_ci_profiles.md is consistent with profile definitions."""

    @pytest.fixture(autouse=True)
    def _load_doc(self):
        self._doc_path = _REPO_ROOT / "docs" / "mcp_ci_profiles.md"
        assert self._doc_path.exists(), f"Missing {self._doc_path}"
        self._doc = self._doc_path.read_text()

    def test_ci_profiles_doc_exists(self):
        assert len(self._doc) > 100

    def test_all_profiles_mentioned_in_doc(self):
        for name in PROFILE_NAMES:
            assert name in self._doc, f"Profile {name!r} missing from mcp_ci_profiles.md"

    def test_profile_markers_in_doc(self):
        for name, marker in PROFILE_MARKERS.items():
            assert marker in self._doc, (
                f"Marker {marker!r} for profile {name!r} missing from mcp_ci_profiles.md"
            )

    def test_runtime_matrix_mentions_profiles(self):
        matrix_path = _REPO_ROOT / "docs" / "mcp_runtime_matrix.md"
        assert matrix_path.exists()
        matrix = matrix_path.read_text()
        assert "CI Profile" in matrix, "mcp_runtime_matrix.md missing 'CI Profile' column"
        for name in PROFILE_NAMES:
            assert name in matrix, f"Profile {name!r} missing from mcp_runtime_matrix.md"


# ---------------------------------------------------------------------------
# Script consistency
# ---------------------------------------------------------------------------


class TestCIProfileScripts:
    """Verify scripts/ci/mcp-*.sh exist and contain correct content."""

    def _script_path(self, profile: str) -> Path:
        return _REPO_ROOT / "scripts" / "ci" / f"mcp-{profile}.sh"

    def test_scripts_exist(self):
        for name in PROFILE_NAMES:
            path = self._script_path(name)
            assert path.exists(), f"Missing script: {path}"

    def test_scripts_executable(self):
        for name in PROFILE_NAMES:
            path = self._script_path(name)
            assert os.access(path, os.X_OK), f"Script not executable: {path}"

    def test_scripts_contain_correct_marker(self):
        for name, marker in PROFILE_MARKERS.items():
            path = self._script_path(name)
            content = path.read_text()
            assert f'-m "{marker}"' in content, (
                f"Script {path.name} missing marker flag: -m \"{marker}\""
            )

    def test_scripts_use_strict_mode(self):
        for name in PROFILE_NAMES:
            path = self._script_path(name)
            content = path.read_text()
            assert "set -euo pipefail" in content, (
                f"Script {path.name} missing strict mode"
            )

    def test_scripts_target_mcp_tests(self):
        for name in PROFILE_NAMES:
            path = self._script_path(name)
            content = path.read_text()
            assert "tests/mcp/" in content, (
                f"Script {path.name} missing tests/mcp/ target"
            )


# ---------------------------------------------------------------------------
# Workflow consistency
# ---------------------------------------------------------------------------


class TestCIWorkflow:
    """Verify .github/workflows/mcp-tests.yml references all profiles."""

    @pytest.fixture(autouse=True)
    def _load_workflow(self):
        self._wf_path = _REPO_ROOT / ".github" / "workflows" / "mcp-tests.yml"
        assert self._wf_path.exists(), f"Missing {self._wf_path}"
        self._wf = self._wf_path.read_text()

    def test_workflow_file_exists(self):
        assert len(self._wf) > 100

    def test_workflow_references_all_profiles(self):
        for name in PROFILE_NAMES:
            assert name in self._wf, (
                f"Profile {name!r} missing from mcp-tests.yml"
            )

    def test_workflow_references_scripts(self):
        for name in PROFILE_NAMES:
            script_ref = f"mcp-{name}.sh"
            assert script_ref in self._wf, (
                f"Script reference {script_ref!r} missing from mcp-tests.yml"
            )
