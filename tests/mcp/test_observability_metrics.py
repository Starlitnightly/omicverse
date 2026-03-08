"""Tests for session-level observability metrics."""

from __future__ import annotations

import tempfile

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
# Session-level metrics
# ---------------------------------------------------------------------------


class TestSessionMetrics:
    def test_initial_metrics_zero(self):
        store = SessionStore()
        m = store.get_metrics()
        assert m["tool_calls_total"] == 0
        assert m["tool_calls_failed"] == 0
        assert m["persisted_adata_count"] == 0
        assert m["restored_adata_count"] == 0

    def test_metrics_after_tool_call(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        m = srv._store.get_metrics()
        assert m["tool_calls_total"] == 1

    def test_metrics_after_failed_tool(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.nonexistent_tool", {})
        m = srv._store.get_metrics()
        assert m["tool_calls_total"] == 1
        assert m["tool_calls_failed"] == 1

    def test_metrics_persisted_count(self, mock_registry):
        persist_dir = tempfile.mkdtemp(prefix="ov_test_")
        srv = _make_mock_server(mock_registry, persist_dir=persist_dir)
        from tests.mcp.conftest import _make_mock_adata
        adata_id = srv._store.create_adata(_make_mock_adata(5, 5))
        srv.call_tool("ov.persist_adata", {"adata_id": adata_id})
        m = srv._store.get_metrics()
        assert m["persisted_adata_count"] == 1

    def test_metrics_restored_count(self, mock_registry):
        from tests.mcp.conftest import _make_mock_adata
        from unittest.mock import patch
        import types

        persist_dir = tempfile.mkdtemp(prefix="ov_test_")
        srv = _make_mock_server(mock_registry, persist_dir=persist_dir)
        adata_id = srv._store.create_adata(_make_mock_adata(5, 5))
        result = srv._store.persist_adata(adata_id)
        path = result["path"]

        mock_ad = types.ModuleType("anndata")
        mock_ad.read_h5ad = lambda p: _make_mock_adata(5, 5)
        with patch.dict("sys.modules", {"anndata": mock_ad}):
            srv.call_tool("ov.restore_adata", {"path": path})

        m = srv._store.get_metrics()
        assert m["restored_adata_count"] >= 1

    def test_metrics_handle_counts(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        from tests.mcp.conftest import _make_mock_adata
        srv._store.create_adata(_make_mock_adata(5, 5))
        srv._store.create_artifact("/tmp/x.png", "image/png")
        m = srv._store.get_metrics()
        assert m["adata_count"] == 1
        assert m["artifact_count"] == 1


# ---------------------------------------------------------------------------
# Per-tool stats
# ---------------------------------------------------------------------------


class TestToolStats:
    def test_tool_stats_populated(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        m = srv._store.get_metrics(scope="tools")
        assert "ov.list_tools" in m["tool_stats"]

    def test_tool_stats_call_count_increments(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        srv.call_tool("ov.list_tools", {})
        ts = srv._store.get_metrics(scope="tools")["tool_stats"]["ov.list_tools"]
        assert ts["call_count"] == 2

    def test_tool_stats_fail_count_increments(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.get_trace", {"trace_id": "nonexistent"})
        ts = srv._store.get_metrics(scope="tools")["tool_stats"]["ov.get_trace"]
        assert ts["fail_count"] == 1

    def test_tool_stats_last_called_at_updates(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        ts = srv._store.get_metrics(scope="tools")["tool_stats"]["ov.list_tools"]
        assert ts["last_called_at"] > 0

    def test_tool_stats_multiple_tools(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv.call_tool("ov.list_tools", {})
        srv.call_tool("ov.get_session", {})
        ts = srv._store.get_metrics(scope="tools")["tool_stats"]
        assert "ov.list_tools" in ts
        assert "ov.get_session" in ts
        assert ts["ov.list_tools"]["call_count"] == 1
        assert ts["ov.get_session"]["call_count"] == 1


# ---------------------------------------------------------------------------
# Scope variants
# ---------------------------------------------------------------------------


class TestMetricsScope:
    def test_scope_session_returns_combined(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        m = srv._store.get_metrics(scope="session")
        assert "session_id" in m
        assert "adata_count" in m
        assert "tool_calls_total" in m

    def test_scope_tools_returns_tool_stats(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        m = srv._store.get_metrics(scope="tools")
        assert "tool_stats" in m
        assert "session_id" not in m
