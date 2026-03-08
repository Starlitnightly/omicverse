"""Tests for RegistryMcpServer."""

import pytest
from omicverse.mcp.server import RegistryMcpServer, META_TOOLS
from omicverse.mcp.manifest import build_registry_manifest
from omicverse.mcp.session_store import SessionStore
from omicverse.mcp.executor import McpExecutor
from tests.mcp.conftest import build_mock_registry


@pytest.fixture
def server():
    """Create a server backed by mock registry."""
    reg = build_mock_registry()
    manifest = build_registry_manifest(registry=reg, phase="P0+P0.5")

    srv = RegistryMcpServer.__new__(RegistryMcpServer)
    srv._phase = "P0+P0.5"
    srv._store = SessionStore()
    srv._manifest = manifest
    srv._executor = McpExecutor(manifest, srv._store)
    return srv


class TestListTools:
    def test_includes_meta_tools(self, server):
        tools = server.list_tools()
        names = {t["name"] for t in tools}
        assert "ov.list_tools" in names
        assert "ov.search_tools" in names
        assert "ov.describe_tool" in names

    def test_includes_registry_tools(self, server):
        tools = server.list_tools()
        names = {t["name"] for t in tools}
        assert "ov.pp.pca" in names
        assert "ov.utils.read" in names

    def test_tool_schema_shape(self, server):
        tools = server.list_tools()
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t


class TestCallTool:
    def test_meta_list_tools(self, server):
        result = server.call_tool("ov.list_tools", {})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert isinstance(data, list)
        assert len(data) > 0

    def test_meta_list_tools_category_filter(self, server):
        result = server.call_tool("ov.list_tools", {"category": "preprocessing"})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        for item in data:
            assert item["category"] == "preprocessing"

    def test_meta_search_tools(self, server):
        result = server.call_tool("ov.search_tools", {"query": "pca"})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert len(data) > 0
        assert data[0]["tool_name"] == "ov.pp.pca"

    def test_meta_search_empty_query(self, server):
        result = server.call_tool("ov.search_tools", {"query": ""})
        assert result["ok"] is False

    def test_meta_describe_tool(self, server):
        result = server.call_tool("ov.describe_tool", {"tool_name": "ov.pp.pca"})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["tool_name"] == "ov.pp.pca"
        assert "parameter_schema" in data
        assert "dependency_contract" in data

    def test_meta_describe_unknown(self, server):
        result = server.call_tool("ov.describe_tool", {"tool_name": "ov.pp.nonexistent"})
        assert result["ok"] is False
        assert result["error_code"] == "tool_not_found"

    def test_registry_tool_call(self, server):
        result = server.call_tool("ov.utils.read", {"path": "test.h5ad"})
        assert result["ok"] is True


class TestSearchScoring:
    def test_exact_match_scores_high(self, server):
        result = server.call_tool("ov.search_tools", {"query": "pca"})
        data = result["outputs"][0]["data"]
        # pca should be first
        assert data[0]["tool_name"] == "ov.pp.pca"

    def test_description_match(self, server):
        result = server.call_tool("ov.search_tools", {"query": "quality control"})
        data = result["outputs"][0]["data"]
        names = [d["tool_name"] for d in data]
        assert "ov.pp.qc" in names
