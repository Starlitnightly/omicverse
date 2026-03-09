"""Tests for McpExecutor."""

import pytest
from omicverse.mcp.executor import (
    McpExecutor,
    TOOL_NOT_FOUND,
    INVALID_ARGUMENTS,
    MISSING_SESSION_OBJECT,
    MISSING_DATA_REQUIREMENTS,
)
from omicverse.mcp.session_store import SessionStore
from tests.mcp.conftest import _make_mock_adata


class TestResolveEntry:
    def test_finds_known_tool(self, executor_with_mock):
        entry = executor_with_mock.resolve_entry("ov.pp.pca")
        assert entry is not None
        assert entry["tool_name"] == "ov.pp.pca"

    def test_returns_none_for_unknown(self, executor_with_mock):
        assert executor_with_mock.resolve_entry("ov.pp.nonexistent") is None


class TestExecuteTool:
    def test_unknown_tool_returns_error(self, executor_with_mock):
        result = executor_with_mock.execute_tool("ov.pp.nonexistent", {})
        assert result["ok"] is False
        assert result["error_code"] == TOOL_NOT_FOUND

    def test_missing_required_arg(self, executor_with_mock):
        # ov.utils.read requires "path"
        result = executor_with_mock.execute_tool("ov.utils.read", {})
        assert result["ok"] is False
        assert result["error_code"] == INVALID_ARGUMENTS

    def test_stateless_tool_executes(self, executor_with_mock):
        result = executor_with_mock.execute_tool(
            "ov.utils.read", {"path": "test.h5ad"}
        )
        assert result["ok"] is True
        assert result["outputs"][0]["type"] == "object_ref"

    def test_adata_tool_missing_session(self, executor_with_mock):
        result = executor_with_mock.execute_tool(
            "ov.pp.scale", {"adata_id": "adata_nonexistent"}
        )
        assert result["ok"] is False
        assert result["error_code"] == MISSING_SESSION_OBJECT

    def test_adata_tool_missing_data_requirement(self, executor_with_mock):
        # Create adata without 'scaled' layer, then try pca (requires scaled)
        adata = _make_mock_adata(50, 200)
        adata_id = executor_with_mock.store.create_adata(adata)
        result = executor_with_mock.execute_tool(
            "ov.pp.pca", {"adata_id": adata_id}
        )
        assert result["ok"] is False
        assert result["error_code"] == MISSING_DATA_REQUIREMENTS

    def test_adata_tool_succeeds_with_prereqs(self, executor_with_mock):
        # Create adata WITH 'scaled' layer
        adata = _make_mock_adata(50, 200)
        adata.layers["scaled"] = adata.X
        adata_id = executor_with_mock.store.create_adata(adata)
        result = executor_with_mock.execute_tool(
            "ov.pp.pca", {"adata_id": adata_id}
        )
        assert result["ok"] is True


class TestPrerequisiteChecking:
    def test_no_prereqs_passes(self, executor_with_mock):
        adata = _make_mock_adata(50, 200)
        adata_id = executor_with_mock.store.create_adata(adata)
        # qc has no requires
        result = executor_with_mock.execute_tool(
            "ov.pp.qc", {"adata_id": adata_id}
        )
        assert result["ok"] is True

    def test_missing_prereqs_returns_suggestions(self, executor_with_mock):
        adata = _make_mock_adata(50, 200)
        adata_id = executor_with_mock.store.create_adata(adata)
        # neighbors requires obsm['X_pca']
        result = executor_with_mock.execute_tool(
            "ov.pp.neighbors", {"adata_id": adata_id}
        )
        assert result["ok"] is False
        assert result["error_code"] == MISSING_DATA_REQUIREMENTS


class TestErrorResponses:
    def test_error_has_required_fields(self, executor_with_mock):
        result = executor_with_mock.execute_tool("ov.nonexistent", {})
        assert "ok" in result
        assert "error_code" in result
        assert "message" in result
        assert "details" in result
        assert "suggested_next_tools" in result

    def test_success_has_required_fields(self, executor_with_mock):
        result = executor_with_mock.execute_tool(
            "ov.utils.read", {"path": "test.h5ad"}
        )
        assert "ok" in result
        assert "tool_name" in result
        assert "outputs" in result
