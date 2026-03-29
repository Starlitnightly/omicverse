import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

import omicverse as ov


def _toy_adata():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["A", "A", "B", "B", "A", "B"]),
            "group": pd.Categorical(["g1", "g2", "g1", "g2", "g1", "g2"]),
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    adata = AnnData(rng.normal(size=(6, 3)), obs=obs, var=var)
    adata.obsm["X_umap"] = rng.normal(size=(6, 2))
    adata.uns["cell_type_colors"] = np.array(["#1f77b4", "#ff7f0e"], dtype="U16")
    adata.uns["group_colors"] = np.array(["#2ca02c", "#d62728"], dtype="U16")
    return adata


def test_plot_cellproportion_warns_and_delegates():
    adata = _toy_adata()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = ov.pl.plot_cellproportion(
            adata,
            celltype_clusters="cell_type",
            visual_clusters="group",
            visual_name="Condition",
            legend=True,
        )

    assert fig is not None
    assert ax.get_xlabel() == "Condition"
    if caught:
        assert "ov.pl.cellproportion" in str(caught[0].message)


def test_plot_embedding_celltype_warns_and_delegates():
    adata = _toy_adata()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = ov.pl.plot_embedding_celltype(
            adata,
            basis="umap",
            celltype_key="cell_type",
            title="Toy",
        )

    assert fig is not None
    assert len(axes) == 2
    assert caught
    assert "ov.pl.embedding_celltype" in str(caught[0].message)


def test_violin_old_warns_and_delegates():
    adata = _toy_adata()
    fig, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ov.pl.violin_old(adata, keys="g1", groupby="cell_type", ax=ax)

    assert ax.get_xlabel() == "cell_type"
    assert caught
    assert "ov.pl.violin" in str(caught[0].message)
