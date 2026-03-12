"""Meta tools backed by a real registry (anndata + scanpy).

Verifies that meta tools like ``ov.list_tools``, ``ov.get_health``, etc.
work correctly when the server is backed by real OmicVerse registrations.
"""

import pytest

from omicverse.mcp.server import RegistryMcpServer
from tests.mcp._env import skip_no_core


@skip_no_core
@pytest.mark.core
@pytest.mark.real_runtime
class TestMetaToolsRealRegistry:
    """Meta tools backed by real registry."""

    @pytest.fixture
    def real_server(self):
        return RegistryMcpServer(phase="P0+P0.5", session_id="test_meta")

    def test_list_tools_real_count(self, real_server):
        result = real_server.call_tool("ov.list_tools", {})
        assert result["ok"] is True
        tools = result["outputs"][0]["data"]
        # Real registry: meta tools + P0/P0.5 pipeline tools
        assert len(tools) >= 12

    def test_get_health_real_session(self, real_server):
        result = real_server.call_tool("ov.get_health", {})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert "session_id" in data
        assert data["session_expired"] is False

    def test_get_limits_real_session(self, real_server):
        result = real_server.call_tool("ov.get_limits", {})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert "quotas" in data
        assert "usage" in data

    def test_search_tools_finds_real_tools(self, real_server):
        result = real_server.call_tool("ov.search_tools", {"query": "pca"})
        assert result["ok"] is True
        matches = result["outputs"][0]["data"]
        assert len(matches) >= 1

    def test_describe_tool_real_tool(self, real_server):
        result = real_server.call_tool("ov.describe_tool", {"tool_name": "ov.pp.pca"})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["tool_name"] == "ov.pp.pca"

    def test_get_metrics_real_session(self, real_server):
        result = real_server.call_tool("ov.get_metrics", {"scope": "session"})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert "tool_calls_total" in data
