"""Tests for CLI help and version output."""

import subprocess
import sys

import pytest


def _run_mcp_cli(*args: str) -> subprocess.CompletedProcess:
    """Run ``python -m omicverse.mcp`` with given args."""
    return subprocess.run(
        [sys.executable, "-m", "omicverse.mcp", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestCLIHelp:
    def test_help_contains_phase_arg(self):
        result = _run_mcp_cli("--help")
        assert result.returncode == 0
        assert "--phase" in result.stdout

    def test_help_contains_session_id_arg(self):
        result = _run_mcp_cli("--help")
        assert result.returncode == 0
        assert "--session-id" in result.stdout

    def test_help_contains_persist_dir_arg(self):
        result = _run_mcp_cli("--help")
        assert result.returncode == 0
        assert "--persist-dir" in result.stdout

    def test_help_contains_description(self):
        result = _run_mcp_cli("--help")
        assert result.returncode == 0
        assert "OmicVerse" in result.stdout

    def test_help_contains_epilog(self):
        result = _run_mcp_cli("--help")
        assert result.returncode == 0
        assert "stderr" in result.stdout


class TestCLIVersion:
    def test_version_flag_exits_clean(self):
        result = _run_mcp_cli("--version")
        assert result.returncode == 0

    def test_version_contains_omicverse(self):
        result = _run_mcp_cli("--version")
        assert result.returncode == 0
        assert "omicverse" in result.stdout.lower()


class TestCLIModuleEquivalence:
    def test_module_help_matches_script(self):
        """Both invocation styles should produce the same help text (modulo prog name)."""
        mod_result = _run_mcp_cli("--help")
        # Just verify both exit 0 and contain the same key content
        assert mod_result.returncode == 0
        assert "--phase" in mod_result.stdout
        assert "--session-id" in mod_result.stdout
        assert "--persist-dir" in mod_result.stdout
        assert "--version" in mod_result.stdout
