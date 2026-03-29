import sys
import types

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import omicverse as ov

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="omicverse.pl currently requires Python 3.10+ due to existing type syntax in other pl modules",
)


@pytest.fixture
def toy_adata():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["A", "A", "B", "B", "A", "B"]),
            "score": np.linspace(0.1, 0.9, 6),
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    adata = AnnData(rng.normal(size=(6, 3)), obs=obs, var=var)
    adata.obsm["X_umap"] = rng.normal(size=(6, 2))
    adata.obsm["X_pca"] = rng.normal(size=(6, 2))
    adata.obsm["scaled|original|X_pca"] = adata.obsm["X_pca"].copy()
    adata.obsm["spatial"] = rng.uniform(0, 10, size=(6, 2))
    adata.uns["scaled|original|pca_var_ratios"] = np.array([0.5, 0.3, 0.2])
    adata.uns["cell_type_colors"] = np.array(["#1f77b4", "#ff7f0e"], dtype="U16")
    adata.uns["spatial"] = {
        "lib1": {
            "images": {"hires": np.ones((12, 12, 3), dtype=float)},
            "scalefactors": {
                "spot_diameter_fullres": 1.0,
                "tissue_hires_scalef": 1.0,
            },
        }
    }
    return adata


def test_public_plot_set_smoke():
    ov.pl.plot_set(scanpy=False, show_monitor=False, dpi=60, figsize=4)
    assert plt.rcParams["figure.dpi"] == 60


def test_public_embedding_smoke(toy_adata):
    ax = ov.pl.embedding(
        toy_adata,
        basis="X_umap",
        color="cell_type",
        frameon=False,
        show=False,
    )
    assert ax is not None
    assert hasattr(ax, "scatter")


def test_public_pca_smoke(toy_adata):
    ax = ov.pl.pca(toy_adata, color="cell_type", show=False)
    assert ax is not None
    assert hasattr(ax, "scatter")


def test_public_spatial_smoke(toy_adata):
    result = ov.pl.spatial(
        toy_adata,
        color="cell_type",
        show=False,
    )
    assert result is not None
    axes = result if isinstance(result, (list, tuple)) else [result]
    assert axes
    assert all(hasattr(ax, "imshow") or hasattr(ax, "scatter") for ax in axes)


def test_public_venn_smoke(tmp_path):
    ax = ov.pl.venn(
        sets={"A": {1, 2}, "B": {2, 3}},
        out=str(tmp_path),
        ax=False,
        ext="png",
    )
    assert ax is False
    assert (tmp_path / "Intersections_2.txt").exists()
    assert (tmp_path / "Venn_2.png").exists()


def test_public_plot_pca_variance_ratio_smoke(toy_adata):
    ov.pl.plot_pca_variance_ratio(toy_adata, n_pcs=3, show=False, save=False)
    ax = plt.gca()
    assert ax.get_xlabel() == "ranking"


def test_public_plot_embedding_celltype_smoke(toy_adata):
    fig, axes = ov.pl.plot_embedding_celltype(
        toy_adata,
        basis="umap",
        celltype_key="cell_type",
        title="Toy",
    )
    assert fig is not None
    assert len(axes) == 2


def test_public_embedding_density_smoke(toy_adata):
    ax = ov.pl.embedding_density(
        toy_adata,
        basis="X_umap",
        groupby="cell_type",
        target_clusters="A",
        show=False,
    )
    assert ax is not None
    assert "temp_density" in toy_adata.obs


def test_public_violin_old_smoke(toy_adata):
    fig, ax = plt.subplots()
    ov.pl.violin_old(
        toy_adata,
        keys="g1",
        groupby="cell_type",
        ax=ax,
    )
    assert ax.get_xlabel() == "cell_type"
    assert ax.get_ylabel() == "g1"


def test_public_half_violin_boxplot_smoke(toy_adata):
    ax = ov.pl.half_violin_boxplot(
        toy_adata,
        keys="g1",
        groupby="cell_type",
        show=False,
    )
    assert ax is not None
    assert ax.get_xlabel() == "cell_type"


def test_public_marker_heatmap_smoke_with_fake_pycomplexheatmap(monkeypatch, toy_adata):
    fake = types.ModuleType("PyComplexHeatmap")
    fake.__version__ = "1.8.0"

    class DummyDotClustermapPlotter:
        def __init__(self, *args, **kwargs):
            self.ax_heatmap = plt.gca()
            self.cmap_legend_kws = {}

    def _dummy(*args, **kwargs):
        return object()

    fake.DotClustermapPlotter = DummyDotClustermapPlotter
    fake.HeatmapAnnotation = _dummy
    fake.anno_simple = _dummy
    fake.anno_label = _dummy
    fake.AnnotationBase = object
    monkeypatch.setitem(sys.modules, "PyComplexHeatmap", fake)

    fig, ax = ov.pl.marker_heatmap(
        toy_adata,
        groupby="cell_type",
        marker_genes_dict={"A": ["g1"], "B": ["g2"]},
        use_raw=False,
        show_rownames=True,
        show_colnames=True,
    )

    assert fig is not None
    assert ax is not None
