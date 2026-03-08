"""Smoke tests for MCP server startup entry points."""

import subprocess
import sys

import pytest
from omicverse.mcp.server import RegistryMcpServer, META_TOOLS, main
from omicverse.mcp.manifest import build_registry_manifest
from omicverse.mcp.session_store import SessionStore
from omicverse.mcp.executor import McpExecutor
from tests.mcp.conftest import build_mock_registry


@pytest.fixture
def mock_server():
    """RegistryMcpServer backed by mock registry (no real imports)."""
    reg = build_mock_registry()
    manifest = build_registry_manifest(registry=reg, phase="P0+P0.5")

    srv = RegistryMcpServer.__new__(RegistryMcpServer)
    srv._phase = "P0+P0.5"
    srv._store = SessionStore()
    srv._manifest = manifest
    srv._executor = McpExecutor(manifest, srv._store)
    return srv


class TestServerInstantiation:
    def test_mock_server_creates(self, mock_server):
        assert mock_server._phase == "P0+P0.5"
        assert len(mock_server._manifest) > 0

    def test_list_tools_includes_meta(self, mock_server):
        tools = mock_server.list_tools()
        names = {t["name"] for t in tools}
        for meta_name in META_TOOLS:
            assert meta_name in names

    def test_list_tools_includes_registry(self, mock_server):
        tools = mock_server.list_tools()
        names = {t["name"] for t in tools}
        # At least one P0 tool should be present
        assert any(n.startswith("ov.") and n not in META_TOOLS for n in names)

    def test_tool_schema_shape(self, mock_server):
        tools = mock_server.list_tools()
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t

    def test_call_meta_list_tools(self, mock_server):
        result = mock_server.call_tool("ov.list_tools", {})
        assert result["ok"] is True
        assert len(result["outputs"][0]["data"]) > 0

    def test_call_registry_tool(self, mock_server):
        result = mock_server.call_tool("ov.utils.read", {"path": "test.h5ad"})
        assert result["ok"] is True


class TestCLIEntrypoint:
    def test_help_flag_exits_clean(self):
        """``python -m omicverse.mcp --help`` should exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "omicverse.mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OmicVerse" in result.stdout

    def test_main_function_is_callable(self):
        assert callable(main)


class TestPackageAPI:
    def test_build_mcp_server_import(self):
        from omicverse.mcp import build_mcp_server
        assert callable(build_mcp_server)

    def test_get_manifest_import(self):
        from omicverse.mcp import get_manifest
        assert callable(get_manifest)

    def test_build_default_manifest_import(self):
        from omicverse.mcp import build_default_manifest
        assert callable(build_default_manifest)


from tests.mcp._env import skip_no_core


@skip_no_core
@pytest.mark.core
@pytest.mark.real_runtime
class TestRealRegistryHydration:
    """Verify that the real registry gets populated via ensure_registry_populated().

    These tests import real OmicVerse modules, so they require anndata/scanpy.
    """

    def setup_method(self):
        # Reset hydration flag so each test triggers a fresh import pass
        import omicverse.mcp.manifest as m
        m._HYDRATED = False

    def test_real_registry_has_p0_tools(self):
        """build_registry_manifest(registry=None) should hydrate and return P0 tools."""
        manifest = build_registry_manifest(phase="P0")
        names = {e["tool_name"] for e in manifest}
        assert "ov.pp.pca" in names
        assert "ov.utils.read" in names
        assert len(manifest) >= 9

    def test_real_registry_has_p05_tools(self):
        """P0+P0.5 should include marker and plotting tools."""
        manifest = build_registry_manifest(phase="P0+P0.5")
        names = {e["tool_name"] for e in manifest}
        assert "ov.single.find_markers" in names
        assert len(manifest) >= 15

    def test_hydration_is_idempotent(self):
        """Calling ensure_registry_populated() twice should not break anything."""
        from omicverse.mcp.manifest import ensure_registry_populated
        ensure_registry_populated()
        ensure_registry_populated()
        manifest = build_registry_manifest(phase="P0")
        assert len(manifest) >= 9
