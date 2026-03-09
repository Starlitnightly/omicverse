"""Static consistency tests for MCP CI dependency files.

Verifies that requirements files exist, are referenced by scripts and workflow,
and that the workflow no longer uses bare unconstrained pip installs for
profile-specific dependencies.
"""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/mcp/ -> repo root

PROFILES_WITH_REQUIREMENTS = {
    "scientific-runtime": "mcp-scientific-runtime.txt",
    "extended-runtime": "mcp-extended-runtime.txt",
}


class TestCIDependencyFiles:
    """Verify requirements files and their integration with scripts/workflow."""

    def test_requirements_files_exist(self):
        for profile, filename in PROFILES_WITH_REQUIREMENTS.items():
            path = _REPO_ROOT / "requirements" / filename
            assert path.exists(), (
                f"Missing requirements file for {profile}: {path}"
            )

    def test_requirements_files_non_empty(self):
        for profile, filename in PROFILES_WITH_REQUIREMENTS.items():
            path = _REPO_ROOT / "requirements" / filename
            lines = [
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            assert len(lines) >= 1, (
                f"Requirements file for {profile} has no active package lines"
            )

    def test_workflow_no_bare_pip_installs(self):
        wf_path = _REPO_ROOT / ".github" / "workflows" / "mcp-tests.yml"
        assert wf_path.exists(), f"Missing workflow file: {wf_path}"
        content = wf_path.read_text()
        # The workflow should not have bare "pip install scvelo" etc.
        assert "pip install scvelo" not in content, (
            "Workflow still has bare 'pip install scvelo' — "
            "use --install flag instead"
        )
        assert "pip install SEACells" not in content, (
            "Workflow still has bare 'pip install SEACells' — "
            "use --install flag instead"
        )

    def test_scripts_reference_requirements(self):
        for profile, filename in PROFILES_WITH_REQUIREMENTS.items():
            script_path = (
                _REPO_ROOT / "scripts" / "ci" / f"mcp-{profile}.sh"
            )
            assert script_path.exists(), f"Missing script: {script_path}"
            content = script_path.read_text()
            assert filename in content, (
                f"Script {script_path.name} does not reference {filename}"
            )

    def test_profiles_doc_mentions_requirements(self):
        doc_path = _REPO_ROOT / "docs" / "mcp_ci_profiles.md"
        assert doc_path.exists(), f"Missing doc: {doc_path}"
        content = doc_path.read_text()
        for _profile, filename in PROFILES_WITH_REQUIREMENTS.items():
            assert filename in content, (
                f"docs/mcp_ci_profiles.md does not mention {filename}"
            )

    def test_scripts_support_install_flag(self):
        all_profiles = [
            "fast-mock",
            "core-runtime",
            "scientific-runtime",
            "extended-runtime",
        ]
        for profile in all_profiles:
            script_path = (
                _REPO_ROOT / "scripts" / "ci" / f"mcp-{profile}.sh"
            )
            assert script_path.exists(), f"Missing script: {script_path}"
            content = script_path.read_text()
            assert "--install" in content, (
                f"Script {script_path.name} does not support --install flag"
            )
