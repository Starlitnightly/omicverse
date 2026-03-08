"""Tests to verify documentation stays consistent with code."""

import pathlib

import pytest
from omicverse.mcp.server import META_TOOLS


# Resolve project root (tests/mcp/ -> project root)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


class TestDocsConsistency:
    def test_meta_tools_count_matches_docs(self):
        """README and integration doc should reference the correct meta tool count."""
        readme = (_PROJECT_ROOT / "README.md").read_text()
        assert "20 built-in meta tools" in readme
        assert len(META_TOOLS) == 20

    def test_phase_names_consistent(self):
        """Quickstart should reference all three phase names."""
        quickstart = (_PROJECT_ROOT / "docs" / "mcp_quickstart.md").read_text()
        assert "P0" in quickstart
        assert "P0.5" in quickstart or "P0+P0.5" in quickstart
        assert "P2" in quickstart

    def test_all_meta_tool_names_in_quickstart(self):
        """Every META_TOOLS key should appear in the quickstart doc."""
        quickstart = (_PROJECT_ROOT / "docs" / "mcp_quickstart.md").read_text()
        for name in META_TOOLS:
            assert name in quickstart, f"Meta tool {name!r} not found in mcp_quickstart.md"

    def test_cli_command_in_readme(self):
        readme = (_PROJECT_ROOT / "README.md").read_text()
        assert "python -m omicverse.mcp" in readme

    def test_cli_command_in_quickstart(self):
        quickstart = (_PROJECT_ROOT / "docs" / "mcp_quickstart.md").read_text()
        assert "python -m omicverse.mcp" in quickstart

    def test_entrypoint_in_pyproject(self):
        pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text()
        assert "omicverse-mcp" in pyproject

    def test_mcp_extra_in_pyproject(self):
        pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text()
        assert "mcp" in pyproject

    def test_integration_doc_exists(self):
        assert (_PROJECT_ROOT / "docs" / "mcp_integration.md").exists()
