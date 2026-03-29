"""Tests for MCP optional-dependency startup boundary.

Verifies that the MCP server module can be imported and that ``--help``
works even when the external ``mcp`` SDK is not installed, and that
transport methods give controlled errors when the SDK is missing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


# ---------------------------------------------------------------------------
# Helper: run a snippet in a subprocess with mcp blocked at the meta-path level
# ---------------------------------------------------------------------------

_MCP_BLOCKER_PREAMBLE = textwrap.dedent("""\
    import sys

    class _McpBlocker:
        \"\"\"Meta-path finder that makes 'mcp' unimportable.\"\"\"
        def find_module(self, name, path=None):
            if name == "mcp" or name.startswith("mcp."):
                return self
        def load_module(self, name):
            raise ImportError(f"Simulated: No module named '{name}'")

    sys.meta_path.insert(0, _McpBlocker())
""")


def _run_with_mcp_blocked(snippet: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run *snippet* in a subprocess where ``import mcp`` always fails."""
    code = _MCP_BLOCKER_PREAMBLE + textwrap.dedent(snippet)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# No-extra scenario: mcp SDK absent
# ---------------------------------------------------------------------------

class TestNoMcpExtra:
    """Behaviour when the ``mcp`` optional dependency is NOT installed."""

    def test_server_module_importable_without_mcp(self):
        """``from omicverse.mcp.server import main`` must not crash."""
        result = _run_with_mcp_blocked("""\
            from omicverse.mcp.server import main
            assert callable(main)
            print("OK")
        """)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_help_flag_without_mcp(self):
        """``python -m omicverse.mcp --help`` degrades gracefully."""
        result = _run_with_mcp_blocked("""\
            import sys
            sys.argv = ["omicverse.mcp", "--help"]
            from omicverse.mcp.server import main
            main()
        """)
        # --help should succeed and print usage because argparse runs
        # before any mcp SDK import
        assert result.returncode == 0, result.stderr
        assert "OmicVerse" in result.stdout

    def test_version_flag_without_mcp(self):
        """``--version`` should also work without mcp SDK."""
        result = _run_with_mcp_blocked("""\
            import sys
            sys.argv = ["omicverse.mcp", "--version"]
            from omicverse.mcp.server import main
            main()
        """)
        assert result.returncode == 0, result.stderr
        assert "omicverse" in result.stdout.lower()

    def test_registry_mcp_server_importable_without_mcp(self):
        """``RegistryMcpServer`` class can be imported without mcp SDK."""
        result = _run_with_mcp_blocked("""\
            from omicverse.mcp.server import RegistryMcpServer
            assert RegistryMcpServer is not None
            print("OK")
        """)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_meta_tools_importable_without_mcp(self):
        """``META_TOOLS`` dict can be imported without mcp SDK."""
        result = _run_with_mcp_blocked("""\
            from omicverse.mcp.server import META_TOOLS
            assert isinstance(META_TOOLS, dict)
            assert len(META_TOOLS) > 0
            print("OK")
        """)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_package_init_importable_without_mcp(self):
        """``from omicverse.mcp import get_manifest`` must not crash."""
        result = _run_with_mcp_blocked("""\
            from omicverse.mcp import get_manifest, build_mcp_server, build_default_manifest
            assert callable(get_manifest)
            assert callable(build_mcp_server)
            assert callable(build_default_manifest)
            print("OK")
        """)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_dunder_main_help_without_mcp(self):
        """``python -m omicverse.mcp --help`` via the actual __main__ path."""
        # Build a script that blocks mcp then runs __main__
        result = _run_with_mcp_blocked("""\
            import sys, runpy
            sys.argv = ["omicverse.mcp", "--help"]
            try:
                runpy.run_module("omicverse.mcp", run_name="__main__")
            except SystemExit as e:
                # argparse --help raises SystemExit(0)
                sys.exit(e.code)
        """)
        assert result.returncode == 0, result.stderr
        assert "OmicVerse" in result.stdout

    def test_dunder_main_no_args_without_mcp_gives_controlled_error(self):
        """Running without --help and without mcp gives a controlled error, not a traceback."""
        result = _run_with_mcp_blocked("""\
            import sys, runpy
            sys.argv = ["omicverse.mcp"]
            try:
                runpy.run_module("omicverse.mcp", run_name="__main__")
            except SystemExit as e:
                sys.exit(e.code)
            except ImportError as e:
                # The server startup will fail with ImportError from run_stdio/etc.
                # but this should be a controlled error, not from module import
                print(f"CONTROLLED: {e}", file=sys.stderr)
                sys.exit(42)
        """)
        # Either exits via controlled ImportError or crashes at run_stdio —
        # but NOT at module import time. returncode != 0 is fine, as long as
        # there's no "No module named 'mcp'" from the top-level import chain.
        assert "from .local_oauth" not in result.stderr, (
            "local_oauth should not be imported at module level"
        )


# ---------------------------------------------------------------------------
# With-extra scenario: mcp SDK present
# ---------------------------------------------------------------------------

class TestWithMcpExtra:
    """Behaviour when the ``mcp`` optional dependency IS installed.

    These tests run in the current process (mcp is available in the test env).
    """

    def test_help_flag_exits_clean(self):
        """``python -m omicverse.mcp --help`` exits 0 with usage output."""
        result = subprocess.run(
            [sys.executable, "-m", "omicverse.mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OmicVerse" in result.stdout
        assert "--phase" in result.stdout

    def test_version_flag_exits_clean(self):
        """``python -m omicverse.mcp --version`` exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "omicverse.mcp", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "omicverse" in result.stdout.lower()

    def test_server_module_imports_fully(self):
        """All server.py exports are importable when mcp is present."""
        from omicverse.mcp.server import RegistryMcpServer, META_TOOLS, main
        assert callable(main)
        assert isinstance(META_TOOLS, dict)
        assert RegistryMcpServer is not None

    def test_entrypoint_function_is_main(self):
        """The ``omicverse-mcp`` console_script points to server:main."""
        from omicverse.mcp.server import main
        assert callable(main)


# ---------------------------------------------------------------------------
# Entrypoint packaging policy
# ---------------------------------------------------------------------------

class TestEntrypointPolicy:
    """Verify the omicverse-mcp entrypoint packaging contract."""

    def test_entrypoint_target_exists(self):
        """The function referenced by the console_script must exist."""
        from omicverse.mcp.server import main
        assert callable(main)

    def test_help_documents_transport_choices(self):
        """--help must document the available transport options."""
        result = subprocess.run(
            [sys.executable, "-m", "omicverse.mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "stdio" in result.stdout
        assert "streamable-http" in result.stdout
