"""Tests for session management meta tools."""

from __future__ import annotations

import os
import tempfile

import pytest
from unittest.mock import patch

from omicverse.mcp.session_store import SessionStore
from omicverse.mcp.executor import McpExecutor
from omicverse.mcp.server import RegistryMcpServer, META_TOOLS


def _make_mock_server(mock_registry, **kwargs):
    """Create a mock RegistryMcpServer bypassing real registry."""
    from omicverse.mcp.manifest import build_registry_manifest

    manifest = build_registry_manifest(registry=mock_registry)
    srv = RegistryMcpServer.__new__(RegistryMcpServer)
    srv._phase = "P0+P0.5"
    srv._store = SessionStore(**kwargs)
    srv._manifest = manifest
    srv._executor = McpExecutor(manifest, srv._store)
    return srv


# ---------------------------------------------------------------------------
# ov.get_session
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.get_session", {})
        assert result["ok"] is True
        assert result["tool_name"] == "ov.get_session"

    def test_returns_session_id(self, mock_registry):
        srv = _make_mock_server(mock_registry, session_id="test-123")
        result = srv.call_tool("ov.get_session", {})
        data = result["outputs"][0]["data"]
        assert data["session_id"] == "test-123"

    def test_returns_handle_counts(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        from tests.mcp.conftest import _make_mock_adata
        srv._store.create_adata(_make_mock_adata(5, 5))
        result = srv.call_tool("ov.get_session", {})
        data = result["outputs"][0]["data"]
        assert data["stats"]["adata_count"] == 1


# ---------------------------------------------------------------------------
# ov.list_handles
# ---------------------------------------------------------------------------


class TestListHandles:
    def test_empty_list(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.list_handles", {})
        assert result["ok"] is True
        assert result["outputs"][0]["data"] == []

    def test_after_create_adata(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        from tests.mcp.conftest import _make_mock_adata
        adata_id = srv._store.create_adata(_make_mock_adata(5, 5))
        result = srv.call_tool("ov.list_handles", {})
        data = result["outputs"][0]["data"]
        assert len(data) == 1
        assert data[0]["handle_id"] == adata_id
        assert data[0]["type"] == "adata"

    def test_type_filter_adata(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        from tests.mcp.conftest import _make_mock_adata
        srv._store.create_adata(_make_mock_adata(5, 5))
        srv._store.create_artifact("/tmp/x.png", "image/png")
        result = srv.call_tool("ov.list_handles", {"type": "adata"})
        data = result["outputs"][0]["data"]
        assert len(data) == 1
        assert all(h["type"] == "adata" for h in data)

    def test_type_filter_instance(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv._store.create_instance({"x": 1}, "TestClass")
        result = srv.call_tool("ov.list_handles", {"type": "instance"})
        data = result["outputs"][0]["data"]
        assert len(data) == 1
        assert data[0]["type"] == "instance"

    def test_multiple_handle_types(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        from tests.mcp.conftest import _make_mock_adata
        srv._store.create_adata(_make_mock_adata(5, 5))
        srv._store.create_artifact("/tmp/x.png", "image/png")
        srv._store.create_instance({"x": 1}, "TestClass")
        result = srv.call_tool("ov.list_handles", {})
        data = result["outputs"][0]["data"]
        types = {h["type"] for h in data}
        assert types == {"adata", "artifact", "instance"}


# ---------------------------------------------------------------------------
# ov.persist_adata
# ---------------------------------------------------------------------------


class TestPersistAdataTool:
    def test_persist_via_call_tool(self, mock_registry):
        persist_dir = tempfile.mkdtemp(prefix="ov_test_")
        srv = _make_mock_server(mock_registry, persist_dir=persist_dir)
        from tests.mcp.conftest import _make_mock_adata
        adata_id = srv._store.create_adata(_make_mock_adata(5, 5))

        result = srv.call_tool("ov.persist_adata", {"adata_id": adata_id})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["adata_id"] == adata_id
        assert os.path.isfile(data["path"])

    def test_persist_unknown_returns_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.persist_adata", {"adata_id": "adata_nonexistent"})
        assert result["ok"] is False
        assert result["error_code"] == "handle_not_found"

    def test_persist_returns_path(self, mock_registry):
        persist_dir = tempfile.mkdtemp(prefix="ov_test_")
        srv = _make_mock_server(mock_registry, persist_dir=persist_dir)
        from tests.mcp.conftest import _make_mock_adata
        adata_id = srv._store.create_adata(_make_mock_adata(5, 5))

        result = srv.call_tool("ov.persist_adata", {"adata_id": adata_id})
        assert "path" in result["outputs"][0]["data"]

    def test_persist_missing_adata_id_returns_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.persist_adata", {})
        assert result["ok"] is False
        assert result["error_code"] == "invalid_arguments"


# ---------------------------------------------------------------------------
# ov.restore_adata
# ---------------------------------------------------------------------------


class TestRestoreAdataTool:
    def test_restore_nonexistent_returns_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.restore_adata", {"path": "/nonexistent/file.h5ad"})
        assert result["ok"] is False
        assert result["error_code"] == "persistence_failed"

    def test_restore_returns_adata_ref(self, mock_registry):
        """Mock anndata.read_h5ad to test restore flow."""
        srv = _make_mock_server(mock_registry)

        # Create a dummy file
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            f.write(b"dummy")
            path = f.name

        try:
            from tests.mcp.conftest import _make_mock_adata
            import types
            mock_ad = types.ModuleType("anndata")
            mock_ad.read_h5ad = lambda p: _make_mock_adata(10, 20)

            with patch.dict("sys.modules", {"anndata": mock_ad}):
                result = srv.call_tool("ov.restore_adata", {"path": path})

            assert result["ok"] is True
            assert result["outputs"][0]["ref_type"] == "adata"
            assert result["outputs"][0]["ref_id"].startswith("adata_")
        finally:
            os.unlink(path)

    def test_restore_missing_path_returns_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.restore_adata", {})
        assert result["ok"] is False
        assert result["error_code"] == "invalid_arguments"


# ---------------------------------------------------------------------------
# Meta tools appear in list_tools
# ---------------------------------------------------------------------------


class TestNewToolsInListTools:
    def test_new_meta_tools_appear_in_list_tools(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        tools = srv.list_tools()
        tool_names = [t["name"] for t in tools]
        assert "ov.get_session" in tool_names
        assert "ov.list_handles" in tool_names
        assert "ov.persist_adata" in tool_names
        assert "ov.restore_adata" in tool_names

    def test_meta_tool_schemas_correct_shape(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        tools = srv.list_tools()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_total_meta_tools_count(self):
        """Verify all 20 meta tools are defined."""
        assert len(META_TOOLS) == 20
        expected = {
            "ov.list_tools", "ov.search_tools", "ov.describe_tool",
            "ov.get_session", "ov.list_handles",
            "ov.persist_adata", "ov.restore_adata",
            "ov.get_metrics", "ov.list_events",
            "ov.get_trace", "ov.list_traces",
            "ov.list_artifacts", "ov.describe_artifact",
            "ov.register_artifact", "ov.delete_artifact",
            "ov.cleanup_artifacts", "ov.export_artifacts_manifest",
            "ov.get_limits", "ov.cleanup_runtime", "ov.get_health",
        }
        assert set(META_TOOLS.keys()) == expected
