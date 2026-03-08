"""
End-to-end smoke test for the P0 pipeline.

Validates: read → qc → scale → pca → neighbors → umap → leiden
"""

import pytest
from omicverse.mcp.manifest import build_registry_manifest
from omicverse.mcp.session_store import SessionStore
from omicverse.mcp.executor import McpExecutor
from tests.mcp.conftest import build_mock_registry


@pytest.fixture
def pipeline_executor():
    """Executor wired to mock registry for P0 tools."""
    reg = build_mock_registry()
    manifest = build_registry_manifest(registry=reg, phase="P0")
    store = SessionStore()
    return McpExecutor(manifest, store)


class TestP0Pipeline:
    def test_full_pipeline(self, pipeline_executor):
        ex = pipeline_executor

        # 1. read
        r = ex.execute_tool("ov.utils.read", {"path": "test_data.h5ad"})
        assert r["ok"] is True, f"read failed: {r.get('message')}"
        adata_id = r["outputs"][0]["ref_id"]
        assert adata_id.startswith("adata_")

        # 2. qc
        r = ex.execute_tool("ov.pp.qc", {"adata_id": adata_id})
        assert r["ok"] is True, f"qc failed: {r.get('message')}"

        # 3. scale
        r = ex.execute_tool("ov.pp.scale", {"adata_id": adata_id})
        assert r["ok"] is True, f"scale failed: {r.get('message')}"

        # Verify scaled layer exists
        adata = ex.store.get_adata(adata_id)
        assert "scaled" in adata.layers

        # 4. pca
        r = ex.execute_tool("ov.pp.pca", {"adata_id": adata_id, "n_pcs": 30})
        assert r["ok"] is True, f"pca failed: {r.get('message')}"
        adata = ex.store.get_adata(adata_id)
        assert "X_pca" in adata.obsm

        # 5. neighbors
        r = ex.execute_tool("ov.pp.neighbors", {"adata_id": adata_id})
        assert r["ok"] is True, f"neighbors failed: {r.get('message')}"
        adata = ex.store.get_adata(adata_id)
        assert "neighbors" in adata.uns

        # 6. umap
        r = ex.execute_tool("ov.pp.umap", {"adata_id": adata_id})
        assert r["ok"] is True, f"umap failed: {r.get('message')}"
        adata = ex.store.get_adata(adata_id)
        assert "X_umap" in adata.obsm

        # 7. leiden
        r = ex.execute_tool(
            "ov.pp.leiden",
            {"adata_id": adata_id, "resolution": 1.0},
        )
        assert r["ok"] is True, f"leiden failed: {r.get('message')}"
        adata = ex.store.get_adata(adata_id)
        assert "leiden" in adata.obs

    def test_pipeline_state_updates_reported(self, pipeline_executor):
        """Each step should report what it produced."""
        ex = pipeline_executor

        r = ex.execute_tool("ov.utils.read", {"path": "test.h5ad"})
        adata_id = r["outputs"][0]["ref_id"]

        # scale should report layers produced
        r = ex.execute_tool("ov.pp.scale", {"adata_id": adata_id})
        updates = r.get("state_updates", {})
        produced = updates.get("produced", {})
        assert "layers" in produced

    def test_pipeline_prereq_order_enforced(self, pipeline_executor):
        """Running pca before scale should fail due to missing 'scaled' layer."""
        ex = pipeline_executor

        r = ex.execute_tool("ov.utils.read", {"path": "test.h5ad"})
        adata_id = r["outputs"][0]["ref_id"]

        # Skip scale, go straight to pca → should fail
        r = ex.execute_tool("ov.pp.pca", {"adata_id": adata_id})
        assert r["ok"] is False
        assert r["error_code"] == "missing_data_requirements"

    def test_pipeline_umap_before_neighbors_fails(self, pipeline_executor):
        """Running umap before neighbors should fail."""
        ex = pipeline_executor

        r = ex.execute_tool("ov.utils.read", {"path": "test.h5ad"})
        adata_id = r["outputs"][0]["ref_id"]

        ex.execute_tool("ov.pp.scale", {"adata_id": adata_id})
        ex.execute_tool("ov.pp.pca", {"adata_id": adata_id})

        # Skip neighbors, go to umap → should fail
        r = ex.execute_tool("ov.pp.umap", {"adata_id": adata_id})
        assert r["ok"] is False
        assert r["error_code"] == "missing_data_requirements"


class TestStoreLayersWorkflow:
    def test_store_and_retrieve(self, pipeline_executor):
        ex = pipeline_executor

        r = ex.execute_tool("ov.utils.read", {"path": "test.h5ad"})
        adata_id = r["outputs"][0]["ref_id"]

        # Store original
        r = ex.execute_tool(
            "ov.utils.store_layers",
            {"adata_id": adata_id, "layers": "original"},
        )
        assert r["ok"] is True

        adata = ex.store.get_adata(adata_id)
        assert "layers_original" in adata.uns

        # Retrieve
        r = ex.execute_tool(
            "ov.utils.retrieve_layers",
            {"adata_id": adata_id, "layers": "original"},
        )
        assert r["ok"] is True
