"""Tests for tool call trace recording."""

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
# Trace creation
# ---------------------------------------------------------------------------


class TestTraceCreation:
    def test_tool_call_creates_trace(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        traces = srv._store.list_traces()
        assert len(traces) == 1

    def test_trace_has_timing_fields(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        traces = srv._store.list_traces()
        t = traces[0]
        assert "started_at" in t
        assert "duration_ms" in t
        assert t["duration_ms"] >= 0

    def test_trace_has_ok_field(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        traces = srv._store.list_traces()
        assert traces[0]["ok"] is True

    def test_trace_has_tool_type_meta(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        traces = srv._store.list_traces()
        assert traces[0]["tool_type"] == "meta"

    def test_trace_has_tool_type_registry(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        from tests.mcp.conftest import _make_mock_adata
        adata_id = srv._store.create_adata(_make_mock_adata(100, 500))
        srv.call_tool("ov.pp.qc", {"adata_id": adata_id})
        traces = srv._store.list_traces(tool_name="ov.pp.qc")
        assert len(traces) == 1
        assert traces[0]["tool_type"] == "registry"

    def test_trace_has_handle_refs_in(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        from tests.mcp.conftest import _make_mock_adata
        adata_id = srv._store.create_adata(_make_mock_adata(100, 500))
        srv.call_tool("ov.pp.qc", {"adata_id": adata_id})

        traces = srv._store.list_traces(tool_name="ov.pp.qc")
        trace_id = traces[0]["trace_id"]
        full = srv._store.get_trace(trace_id)
        assert adata_id in full["handle_refs_in"]

    def test_trace_has_handle_refs_out(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.utils.read", {"path": "test.h5ad"})
        ref_id = result["outputs"][0]["ref_id"]

        traces = srv._store.list_traces(tool_name="ov.utils.read")
        trace_id = traces[0]["trace_id"]
        full = srv._store.get_trace(trace_id)
        assert ref_id in full["handle_refs_out"]

    def test_failed_tool_trace_has_error_code(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.pp.qc", {"adata_id": "adata_nonexistent"})
        traces = srv._store.list_traces(tool_name="ov.pp.qc")
        assert len(traces) == 1
        assert traces[0]["ok"] is False
        assert traces[0]["error_code"] is not None


# ---------------------------------------------------------------------------
# Trace retrieval
# ---------------------------------------------------------------------------


class TestTraceRetrieval:
    def test_get_trace_by_id(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        traces = srv._store.list_traces()
        trace_id = traces[0]["trace_id"]
        full = srv._store.get_trace(trace_id)
        assert full is not None
        assert full["trace_id"] == trace_id
        assert "finished_at" in full
        assert "handle_refs_in" in full
        assert "handle_refs_out" in full

    def test_get_trace_unknown_returns_none(self):
        store = SessionStore()
        assert store.get_trace("nonexistent_trace") is None

    def test_list_traces_default_limit(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        for _ in range(60):
            srv.call_tool("ov.list_tools", {})
        traces = srv._store.list_traces()
        assert len(traces) == 50  # default limit

    def test_list_traces_filter_by_tool_name(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        srv.call_tool("ov.get_session", {})
        traces = srv._store.list_traces(tool_name="ov.get_session")
        assert len(traces) == 1
        assert traces[0]["tool_name"] == "ov.get_session"

    def test_list_traces_filter_by_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        srv.call_tool("ov.get_trace", {"trace_id": "nonexistent"})
        ok_traces = srv._store.list_traces(ok=True)
        fail_traces = srv._store.list_traces(ok=False)
        assert len(ok_traces) >= 1
        assert len(fail_traces) >= 1
        assert all(t["ok"] for t in ok_traces)
        assert all(not t["ok"] for t in fail_traces)
