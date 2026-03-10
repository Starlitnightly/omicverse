"""Tests for observability meta tools (ov.get_metrics, ov.list_events, etc.)."""

from __future__ import annotations

from omicverse.mcp.session_store import SessionStore
from omicverse.mcp.executor import McpExecutor
from omicverse.mcp.server import RegistryMcpServer


def _make_mock_server(mock_registry, **kwargs):
    from omicverse.mcp.manifest import build_registry_manifest

    manifest = build_registry_manifest(registry=mock_registry)
    srv = RegistryMcpServer.__new__(RegistryMcpServer)
    srv._phase = "P0+P0.5"
    srv._store = SessionStore(**kwargs)
    srv._manifest = manifest
    srv._executor = McpExecutor(manifest, srv._store)
    return srv


# ---------------------------------------------------------------------------
# ov.get_metrics
# ---------------------------------------------------------------------------


class TestGetMetricsTool:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.get_metrics", {})
        assert result["ok"] is True
        assert result["tool_name"] == "ov.get_metrics"

    def test_returns_session_metrics(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.get_metrics", {})
        data = result["outputs"][0]["data"]
        assert "session_id" in data
        assert "adata_count" in data
        assert "tool_calls_total" in data

    def test_scope_tools_returns_tool_stats(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        result = srv.call_tool("ov.get_metrics", {"scope": "tools"})
        data = result["outputs"][0]["data"]
        assert "tool_stats" in data

    def test_metrics_after_multiple_calls(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        srv.call_tool("ov.get_session", {})
        srv.call_tool("ov.list_tools", {})
        result = srv.call_tool("ov.get_metrics", {})
        data = result["outputs"][0]["data"]
        # 3 preceding calls + the get_metrics call itself
        assert data["tool_calls_total"] >= 3


# ---------------------------------------------------------------------------
# ov.list_events
# ---------------------------------------------------------------------------


class TestListEventsTool:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.list_events", {})
        assert result["ok"] is True
        assert result["tool_name"] == "ov.list_events"

    def test_returns_events_list(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        from tests.mcp.conftest import _make_mock_adata
        srv._store.create_adata(_make_mock_adata(5, 5))
        result = srv.call_tool("ov.list_events", {})
        data = result["outputs"][0]["data"]
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_filter_by_event_type(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        from tests.mcp.conftest import _make_mock_adata
        srv._store.create_adata(_make_mock_adata(5, 5))
        srv._store.create_artifact("/tmp/x.png", "image/png")
        result = srv.call_tool("ov.list_events", {"event_type": "adata_created"})
        data = result["outputs"][0]["data"]
        assert len(data) >= 1
        assert all(e["event_type"] == "adata_created" for e in data)

    def test_limit_parameter(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        for _ in range(10):
            srv._store.record_event("test", {})
        result = srv.call_tool("ov.list_events", {"limit": 3})
        data = result["outputs"][0]["data"]
        assert len(data) == 3


# ---------------------------------------------------------------------------
# ov.get_trace
# ---------------------------------------------------------------------------


class TestGetTraceTool:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        traces = srv._store.list_traces()
        trace_id = traces[0]["trace_id"]
        result = srv.call_tool("ov.get_trace", {"trace_id": trace_id})
        assert result["ok"] is True

    def test_unknown_trace_returns_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.get_trace", {"trace_id": "nonexistent"})
        assert result["ok"] is False
        assert result["error_code"] == "handle_not_found"

    def test_missing_trace_id_returns_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.get_trace", {})
        assert result["ok"] is False
        assert result["error_code"] == "invalid_arguments"


# ---------------------------------------------------------------------------
# ov.list_traces
# ---------------------------------------------------------------------------


class TestListTracesTool:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.list_traces", {})
        assert result["ok"] is True

    def test_filter_by_tool_name(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        srv.call_tool("ov.get_session", {})
        result = srv.call_tool("ov.list_traces", {"tool_name": "ov.list_tools"})
        data = result["outputs"][0]["data"]
        assert all(t["tool_name"] == "ov.list_tools" for t in data)

    def test_filter_by_ok_status(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        srv.call_tool("ov.get_trace", {"trace_id": "nonexistent"})
        result = srv.call_tool("ov.list_traces", {"ok": False})
        data = result["outputs"][0]["data"]
        assert len(data) >= 1
        assert all(not t["ok"] for t in data)


# ---------------------------------------------------------------------------
# Observability tools in list_tools
# ---------------------------------------------------------------------------


class TestObservabilityToolsInListTools:
    def test_new_tools_appear_in_list_tools(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        tools = srv.list_tools()
        tool_names = [t["name"] for t in tools]
        assert "ov.get_metrics" in tool_names
        assert "ov.list_events" in tool_names
        assert "ov.get_trace" in tool_names
        assert "ov.list_traces" in tool_names
