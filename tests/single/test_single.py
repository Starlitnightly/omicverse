import pytest
import omicverse as ov


@pytest.fixture(scope="module")
def adata():
    adata = ov.datasets.pbmc3k()
    return adata


@pytest.fixture(scope="module")
def processed_adata(adata):
    adata = ov.pp.qc(
        adata, tresh={"mito_perc": 0.05, "nUMIs": 500, "detected_genes": 250}
    )
    adata = ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)
    ov.pp.scale(adata)
    return adata

def test_pp(processed_adata):
    assert processed_adata.layers["scaled"] is not None

def test_pca_umap(processed_adata):
    ov.pp.pca(processed_adata, layer="scaled", n_pcs=5)
    ov.pp.neighbors(processed_adata, use_rep="scaled|original|X_pca")
    ov.pp.umap(processed_adata)
    assert processed_adata.obsm["X_umap"] is not None
    #assert processed_adata.obsm["scaled|original|X_pca"] is not None

def test_cpu_gpu_mixed_init():
    ov.settings.cpu_gpu_mixed_init()
    assert ov.settings.mode == "cpu-gpu-mixed"

