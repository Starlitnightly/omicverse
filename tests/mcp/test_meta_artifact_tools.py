"""Tests for artifact meta tools (ov.list_artifacts, ov.describe_artifact, etc.)."""

from __future__ import annotations

import os
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
# ov.list_artifacts
# ---------------------------------------------------------------------------


class TestListArtifactsTool:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.list_artifacts", {})
        assert result["ok"] is True
        assert result["tool_name"] == "ov.list_artifacts"

    def test_empty_list(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.list_artifacts", {})
        data = result["outputs"][0]["data"]
        assert data == []

    def test_filter_params_work(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv._store.create_artifact("/a.png", "image/png", artifact_type="image",
                                   source_tool="ov.pl.umap")
        srv._store.create_artifact("/b.csv", "text/csv", artifact_type="table",
                                   source_tool="ov.tl.rank_genes")
        result = srv.call_tool("ov.list_artifacts", {"artifact_type": "image"})
        data = result["outputs"][0]["data"]
        assert len(data) == 1
        assert data[0]["artifact_type"] == "image"


# ---------------------------------------------------------------------------
# ov.describe_artifact
# ---------------------------------------------------------------------------


class TestDescribeArtifactTool:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        aid = srv._store.create_artifact("/tmp/x.png", "image/png")
        result = srv.call_tool("ov.describe_artifact", {"artifact_id": aid})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["artifact_id"] == aid
        assert "file_exists" in data

    def test_unknown_artifact_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.describe_artifact", {"artifact_id": "artifact_nope"})
        assert result["ok"] is False
        assert result["error_code"] == "handle_not_found"

    def test_missing_artifact_id_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.describe_artifact", {})
        assert result["ok"] is False
        assert result["error_code"] == "invalid_arguments"


# ---------------------------------------------------------------------------
# ov.register_artifact
# ---------------------------------------------------------------------------


class TestRegisterArtifactTool:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            result = srv.call_tool("ov.register_artifact", {"path": path})
            assert result["ok"] is True
            data = result["outputs"][0]["data"]
            assert "artifact_id" in data
            assert data["file_exists"] is True
        finally:
            os.unlink(path)

    def test_missing_path_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.register_artifact", {})
        assert result["ok"] is False
        assert result["error_code"] == "invalid_arguments"

    def test_artifact_type_passed(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.register_artifact", {
            "path": "/tmp/report.pdf",
            "artifact_type": "report",
            "content_type": "application/pdf",
        })
        assert result["ok"] is True
        aid = result["outputs"][0]["data"]["artifact_id"]
        h = srv._store.get_artifact(aid)
        assert h.artifact_type == "report"
        assert h.content_type == "application/pdf"

    def test_file_not_exist_still_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.register_artifact", {
            "path": "/tmp/not_here_12345.txt",
        })
        assert result["ok"] is True
        assert len(result["warnings"]) == 1
        data = result["outputs"][0]["data"]
        assert data["file_exists"] is False


# ---------------------------------------------------------------------------
# ov.delete_artifact
# ---------------------------------------------------------------------------


class TestDeleteArtifactTool:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        aid = srv._store.create_artifact("/tmp/x.txt", "text/plain")
        result = srv.call_tool("ov.delete_artifact", {"artifact_id": aid})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["deleted_handle"] is True

    def test_missing_artifact_id_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.delete_artifact", {})
        assert result["ok"] is False
        assert result["error_code"] == "invalid_arguments"

    def test_delete_file_flag(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"bye")
            path = f.name
        aid = srv._store.create_artifact(path, "text/plain")
        result = srv.call_tool("ov.delete_artifact", {
            "artifact_id": aid, "delete_file": True,
        })
        assert result["ok"] is True
        assert result["outputs"][0]["data"]["deleted_file"] is True
        assert not os.path.isfile(path)

    def test_unknown_artifact_error(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.delete_artifact", {"artifact_id": "artifact_nope"})
        assert result["ok"] is False
        assert result["error_code"] == "handle_not_found"


# ---------------------------------------------------------------------------
# ov.cleanup_artifacts
# ---------------------------------------------------------------------------


class TestCleanupArtifactsTool:
    def test_dry_run_default(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv._store.create_artifact("/tmp/a.txt", "text/plain")
        result = srv.call_tool("ov.cleanup_artifacts", {})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["dry_run"] is True
        assert data["matched"] == 1
        assert data["deleted"] == 0
        # Artifact still exists
        assert len(srv._store.list_artifacts()) == 1

    def test_actual_cleanup(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv._store.create_artifact("/tmp/a.txt", "text/plain")
        srv._store.create_artifact("/tmp/b.txt", "text/plain")
        result = srv.call_tool("ov.cleanup_artifacts", {"dry_run": False})
        data = result["outputs"][0]["data"]
        assert data["deleted"] == 2
        assert len(srv._store.list_artifacts()) == 0


# ---------------------------------------------------------------------------
# ov.export_artifacts_manifest
# ---------------------------------------------------------------------------


class TestExportManifestTool:
    def test_returns_ok(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        result = srv.call_tool("ov.export_artifacts_manifest", {})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert "session_id" in data
        assert "artifact_count" in data
        assert data["artifact_count"] == 0

    def test_contains_artifacts(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        srv._store.create_artifact("/tmp/x.png", "image/png", artifact_type="image")
        result = srv.call_tool("ov.export_artifacts_manifest", {})
        data = result["outputs"][0]["data"]
        assert data["artifact_count"] == 1
        assert len(data["artifacts"]) == 1
        assert data["artifacts"][0]["artifact_type"] == "image"


# ---------------------------------------------------------------------------
# Tools in list_tools
# ---------------------------------------------------------------------------


class TestArtifactToolsInListTools:
    def test_new_tools_appear_in_list_tools(self, mock_registry):
        srv = _make_mock_server(mock_registry)
        tools = srv.list_tools()
        tool_names = [t["name"] for t in tools]
        assert "ov.list_artifacts" in tool_names
        assert "ov.describe_artifact" in tool_names
        assert "ov.register_artifact" in tool_names
        assert "ov.delete_artifact" in tool_names
        assert "ov.cleanup_artifacts" in tool_names
        assert "ov.export_artifacts_manifest" in tool_names
