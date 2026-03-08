"""Tests for AdataAdapter."""

import pytest
from omicverse.mcp.adapters.adata_adapter import AdataAdapter
from omicverse.mcp.session_store import SessionStore
from tests.mcp.conftest import _make_mock_adata


@pytest.fixture
def adapter():
    return AdataAdapter()


@pytest.fixture
def store_with_adata():
    store = SessionStore()
    adata = _make_mock_adata(100, 500)
    adata_id = store.create_adata(adata)
    return store, adata_id, adata


class TestCanHandle:
    def test_accepts_adata(self, adapter):
        assert adapter.can_handle({"execution_class": "adata"})

    def test_rejects_stateless(self, adapter):
        assert not adapter.can_handle({"execution_class": "stateless"})


class TestInvokeAdata:
    def test_mutating_tool(self, adapter, store_with_adata):
        store, adata_id, adata = store_with_adata

        def scale(adata, max_value=10, **kwargs):
            adata.layers["scaled"] = "mock_scaled"
            return adata

        entry = {
            "tool_name": "ov.pp.scale",
            "_function": scale,
            "execution_class": "adata",
            "description": "Scale data",
            "dependency_contract": {"produces": {"layers": ["scaled"]}},
            "return_contract": {"primary_output": "object_ref"},
        }
        result = adapter.invoke(entry, {"adata_id": adata_id}, store)
        assert result["ok"] is True
        assert result["outputs"][0]["type"] == "object_ref"
        assert result["outputs"][0]["ref_id"] == adata_id
        # Check state updates
        produced = result["state_updates"].get("produced", {})
        assert "layers" in produced
        assert "scaled" in produced["layers"]

    def test_missing_adata_id(self, adapter):
        store = SessionStore()
        entry = {
            "tool_name": "ov.pp.scale",
            "_function": lambda adata: adata,
            "execution_class": "adata",
            "description": "",
            "return_contract": {"primary_output": "object_ref"},
        }
        result = adapter.invoke(entry, {}, store)
        assert result["ok"] is False
        assert result["error_code"] == "missing_session_object"

    def test_unknown_adata_id(self, adapter):
        store = SessionStore()
        entry = {
            "tool_name": "ov.pp.scale",
            "_function": lambda adata: adata,
            "execution_class": "adata",
            "description": "",
            "return_contract": {"primary_output": "object_ref"},
        }
        result = adapter.invoke(entry, {"adata_id": "adata_nonexistent"}, store)
        assert result["ok"] is False
        assert result["error_code"] == "missing_session_object"

    def test_exception_in_tool(self, adapter, store_with_adata):
        store, adata_id, _ = store_with_adata

        def bad_func(adata):
            raise RuntimeError("computation failed")

        entry = {
            "tool_name": "ov.pp.bad",
            "_function": bad_func,
            "execution_class": "adata",
            "description": "",
            "return_contract": {"primary_output": "object_ref"},
        }
        result = adapter.invoke(entry, {"adata_id": adata_id}, store)
        assert result["ok"] is False
        assert "computation failed" in result["message"]


class TestSnapshotAndDetect:
    def test_detects_new_keys(self, adapter, store_with_adata):
        store, adata_id, adata = store_with_adata
        pre = adapter.snapshot_pre_state(adata, {})
        # Simulate adding keys
        adata.obsm["X_pca"] = "mock"
        adata.uns["pca"] = {"params": {}}
        updates = adapter.detect_state_updates(adata, pre, {})
        assert "X_pca" in updates["produced"].get("obsm", [])
        assert "pca" in updates["produced"].get("uns", [])

    def test_no_changes_empty_produced(self, adapter, store_with_adata):
        _, _, adata = store_with_adata
        pre = adapter.snapshot_pre_state(adata, {})
        updates = adapter.detect_state_updates(adata, pre, {})
        assert updates["produced"] == {}
