"""End-to-end P0 pipeline tests using real anndata + scanpy.

Each test calls the MCP server's ``call_tool()`` with a real
``RegistryMcpServer`` backed by the actual OmicVerse registry.
"""

import pytest

from omicverse.mcp.server import RegistryMcpServer
from tests.mcp._env import skip_no_core


def _write_test_h5ad(tmp_path, n_obs=200, n_vars=2000):
    """Write a realistic .h5ad with negative-binomial counts."""
    import anndata
    import numpy as np

    rng = np.random.default_rng(42)
    # Negative binomial: overdispersed counts like real scRNA-seq
    means = rng.exponential(scale=10, size=n_vars)
    X = np.zeros((n_obs, n_vars), dtype="float32")
    for j in range(n_vars):
        X[:, j] = rng.negative_binomial(n=2, p=2 / (2 + means[j]), size=n_obs)
    adata = anndata.AnnData(X=X)
    var_names = [f"Gene_{i}" for i in range(n_vars)]
    for i in range(20):
        var_names[i] = f"MT-{i}"
    adata.var_names = var_names
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    path = str(tmp_path / "test.h5ad")
    adata.write_h5ad(path)
    return path


@skip_no_core
@pytest.mark.core
@pytest.mark.real_runtime
class TestRealP0Pipeline:
    """Execute the P0 pipeline via call_tool with real anndata."""

    @pytest.fixture
    def real_server(self):
        return RegistryMcpServer(phase="P0+P0.5", session_id="test_p0")

    def _read_adata(self, server, path):
        """Helper: read an h5ad and return the ref_id."""
        result = server.call_tool("ov.utils.read", {"path": path})
        assert result["ok"] is True, f"read failed: {result}"
        return result["outputs"][0]["ref_id"]

    def test_read_creates_adata_handle(self, real_server, tmp_path):
        path = _write_test_h5ad(tmp_path)
        result = real_server.call_tool("ov.utils.read", {"path": path})
        assert result["ok"] is True
        output = result["outputs"][0]
        assert output["type"] == "object_ref"
        assert "ref_id" in output

    def test_qc_on_real_adata(self, real_server, tmp_path):
        path = _write_test_h5ad(tmp_path)
        adata_id = self._read_adata(real_server, path)
        result = real_server.call_tool("ov.pp.qc", {"adata_id": adata_id})
        assert result["ok"] is True, f"qc failed: {result.get('message', '')}"

    def test_scale_on_real_adata(self, real_server, tmp_path):
        path = _write_test_h5ad(tmp_path)
        adata_id = self._read_adata(real_server, path)
        real_server.call_tool("ov.pp.qc", {"adata_id": adata_id})
        result = real_server.call_tool("ov.pp.scale", {"adata_id": adata_id})
        assert result["ok"] is True, f"scale failed: {result.get('message', '')}"

    def test_pca_on_real_adata(self, real_server, tmp_path):
        path = _write_test_h5ad(tmp_path)
        adata_id = self._read_adata(real_server, path)
        real_server.call_tool("ov.pp.qc", {"adata_id": adata_id})
        real_server.call_tool("ov.pp.scale", {"adata_id": adata_id})
        result = real_server.call_tool("ov.pp.pca", {"adata_id": adata_id})
        assert result["ok"] is True, f"pca failed: {result.get('message', '')}"

    def test_full_p0_pipeline(self, real_server, tmp_path):
        """read -> qc -> scale -> pca (-> neighbors -> umap -> leiden if torch available)."""
        path = _write_test_h5ad(tmp_path)
        adata_id = self._read_adata(real_server, path)

        # Core steps (no torch dependency)
        core_steps = [
            ("ov.pp.qc", {"adata_id": adata_id}),
            ("ov.pp.scale", {"adata_id": adata_id}),
            ("ov.pp.pca", {"adata_id": adata_id}),
        ]
        for tool_name, args in core_steps:
            result = real_server.call_tool(tool_name, args)
            assert result["ok"] is True, f"{tool_name} failed: {result.get('message', '')}"

        # Graph steps require torch for neighbor computation; skip if unavailable
        graph_steps = [
            ("ov.pp.neighbors", {"adata_id": adata_id}),
            ("ov.pp.umap", {"adata_id": adata_id}),
            ("ov.pp.leiden", {"adata_id": adata_id}),
        ]
        for tool_name, args in graph_steps:
            result = real_server.call_tool(tool_name, args)
            if not result["ok"] and "torch" in result.get("message", ""):
                pytest.skip("torch not installed — skipping graph steps")
            assert result["ok"] is True, f"{tool_name} failed: {result.get('message', '')}"

    def test_list_handles_after_read(self, real_server, tmp_path):
        """After reading, the adata handle should appear in list_handles."""
        path = _write_test_h5ad(tmp_path)
        adata_id = self._read_adata(real_server, path)
        result = real_server.call_tool("ov.list_handles", {})
        assert result["ok"] is True
        handles = result["outputs"][0]["data"]
        ids = [h["handle_id"] for h in handles] if isinstance(handles, list) else []
        assert adata_id in ids, f"adata {adata_id} not in handles: {handles}"
