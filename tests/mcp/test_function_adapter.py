"""Tests for FunctionAdapter."""

import pytest
from omicverse.mcp.adapters.function_adapter import FunctionAdapter
from omicverse.mcp.session_store import SessionStore


@pytest.fixture
def adapter():
    return FunctionAdapter()


@pytest.fixture
def store():
    return SessionStore()


class TestCanHandle:
    def test_accepts_stateless(self, adapter):
        assert adapter.can_handle({"execution_class": "stateless"})

    def test_rejects_adata(self, adapter):
        assert not adapter.can_handle({"execution_class": "adata"})

    def test_rejects_class(self, adapter):
        assert not adapter.can_handle({"execution_class": "class"})


class TestInvokeStateless:
    def test_simple_function(self, adapter, store):
        def func(x: int = 1, y: str = "hello"):
            return {"x": x, "y": y}

        entry = {
            "tool_name": "ov.test.func",
            "_function": func,
            "execution_class": "stateless",
            "description": "Test function",
        }
        result = adapter.invoke(entry, {"x": 42, "y": "world"}, store)
        assert result["ok"] is True
        assert result["outputs"][0]["type"] == "json"
        assert result["outputs"][0]["data"]["x"] == 42

    def test_returns_none(self, adapter, store):
        def func():
            return None

        entry = {"tool_name": "ov.test.none", "_function": func,
                 "execution_class": "stateless", "description": ""}
        result = adapter.invoke(entry, {}, store)
        assert result["ok"] is True

    def test_exception_handled(self, adapter, store):
        def func():
            raise ValueError("test error")

        entry = {"tool_name": "ov.test.err", "_function": func,
                 "execution_class": "stateless", "description": ""}
        result = adapter.invoke(entry, {}, store)
        assert result["ok"] is False
        assert result["error_code"] == "execution_failed"
        assert "test error" in result["message"]

    def test_missing_function(self, adapter, store):
        entry = {"tool_name": "ov.test.nofunc", "_function": None,
                 "execution_class": "stateless", "description": ""}
        result = adapter.invoke(entry, {}, store)
        assert result["ok"] is False

    def test_anndata_return_creates_ref(self, adapter, store):
        """When a stateless function returns AnnData, it should create adata_id."""
        from tests.mcp.conftest import _make_mock_adata

        def func(path: str = "test.h5ad"):
            return _make_mock_adata(50, 200)

        entry = {"tool_name": "ov.utils.read", "_function": func,
                 "execution_class": "stateless", "description": "Read data"}
        result = adapter.invoke(entry, {"path": "test.h5ad"}, store)
        assert result["ok"] is True
        assert result["outputs"][0]["type"] == "object_ref"
        assert result["outputs"][0]["ref_id"].startswith("adata_")
        # Verify stored
        adata_id = result["outputs"][0]["ref_id"]
        adata = store.get_adata(adata_id)
        assert adata.shape == (50, 200)
