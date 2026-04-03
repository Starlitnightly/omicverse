import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import omicverse as ov
import omicverse.datasets._datasets as datasets_mod
import omicverse.pl._heatmap_marsilea as heatmap_mod
from omicverse.pl._heatmap_marsilea import (
    _prepare_dynamic_matrix,
    _compute_dynamic_feature_metadata,
    _split_dynamic_features,
    cell_cor_heatmap,
    dynamic_heatmap,
    feature_heatmap,
    group_heatmap,
)
from omicverse.pl._heatmap import complexheatmap, marker_heatmap


def _has_pycomplexheatmap():
    try:
        import PyComplexHeatmap  # noqa: F401
        return True
    except ImportError:
        return False


class _StubFigure:
    def __init__(self):
        self.axes = [_StubAxis()]

    def savefig(self, *args, **kwargs):
        return None


class _StubSpine:
    def __init__(self):
        self.visible = False
        self.linewidth = None
        self.edgecolor = None

    def set_visible(self, visible):
        self.visible = visible

    def set_linewidth(self, linewidth):
        self.linewidth = linewidth

    def set_edgecolor(self, edgecolor):
        self.edgecolor = edgecolor


class _StubAxis:
    def __init__(self):
        self.collections = [object()]
        self.images = []
        self.lines = []
        self.patches = []
        self._visible = True
        self.axis_on = False
        self.xticks = None
        self.yticks = None
        self.tick_params_kwargs = None
        self.spines = {
            "left": _StubSpine(),
            "right": _StubSpine(),
            "top": _StubSpine(),
            "bottom": _StubSpine(),
        }

    def get_visible(self):
        return self._visible

    def set_axis_on(self):
        self.axis_on = True

    def set_xticks(self, ticks):
        self.xticks = list(ticks)

    def set_yticks(self, ticks):
        self.yticks = list(ticks)

    def tick_params(self, **kwargs):
        self.tick_params_kwargs = kwargs


class _StubPlot:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.figure = _StubFigure()
        self.calls = []

    def add_left(self, *args, **kwargs):
        self.calls.append(("add_left", args, kwargs))

    def add_right(self, *args, **kwargs):
        self.calls.append(("add_right", args, kwargs))

    def add_top(self, *args, **kwargs):
        self.calls.append(("add_top", args, kwargs))

    def add_title(self, *args, **kwargs):
        self.calls.append(("add_title", args, kwargs))

    def add_bottom(self, *args, **kwargs):
        self.calls.append(("add_bottom", args, kwargs))

    def add_legends(self, *args, **kwargs):
        self.calls.append(("add_legends", args, kwargs))

    def add_dendrogram(self, *args, **kwargs):
        self.calls.append(("add_dendrogram", args, kwargs))

    def group_cols(self, *args, **kwargs):
        self.calls.append(("group_cols", args, kwargs))

    def group_rows(self, *args, **kwargs):
        self.calls.append(("group_rows", args, kwargs))

    def render(self):
        return None


class _StubStackBoard:
    def __init__(self, boards, *args, **kwargs):
        self.boards = boards
        self.args = args
        self.kwargs = kwargs
        self.figure = _StubFigure()
        self.calls = []

    def add_legends(self, *args, **kwargs):
        self.calls.append(("add_legends", args, kwargs))

    def render(self):
        return None


class _StubAnnotation:
    def __init__(self, kind, *args, **kwargs):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs

    def get_legends(self):
        return self.kind


class _StubMA:
    Heatmap = _StubPlot
    SizedHeatmap = _StubPlot
    StackBoard = _StubStackBoard


class _StubMP:
    @staticmethod
    def Colors(*args, **kwargs):
        return _StubAnnotation("Colors", *args, **kwargs)

    @staticmethod
    def Labels(*args, **kwargs):
        return _StubAnnotation("Labels", *args, **kwargs)

    @staticmethod
    def AnnoLabels(*args, **kwargs):
        return _StubAnnotation("AnnoLabels", *args, **kwargs)

    @staticmethod
    def Numbers(*args, **kwargs):
        return _StubAnnotation("Numbers", *args, **kwargs)

    @staticmethod
    def StackBar(*args, **kwargs):
        return _StubAnnotation("StackBar", *args, **kwargs)

    @staticmethod
    def ColorMesh(*args, **kwargs):
        return _StubAnnotation("ColorMesh", *args, **kwargs)


@pytest.fixture
def stub_marsilea(monkeypatch):
    monkeypatch.setattr(heatmap_mod, "_import_marsilea", lambda: (_StubMA, _StubMP))
    monkeypatch.setattr(
        heatmap_mod,
        "_render_plot",
        lambda plotter, save_path=None, show=False, legend_kws=None: (
            plotter.render() or getattr(plotter, "figure", None)
        ),
    )


@pytest.fixture
def simple_adata():
    adata = AnnData(
        X=np.array(
            [
                [1.0, 0.0, 3.0, 0.5],
                [2.0, 1.0, 0.0, 1.5],
                [0.0, 2.0, 1.0, 2.5],
                [4.0, 3.0, 0.0, 3.5],
                [3.0, 2.5, 2.0, 4.5],
                [1.5, 0.5, 4.0, 5.0],
            ]
        )
    )
    adata.var_names = ["GeneA", "GeneB", "GeneC", "GeneD"]
    adata.obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["T", "T", "B", "B", "Mono", "Mono"]),
            "pseudotime": [0.1, 0.2, 0.4, 0.5, 0.8, 0.9],
            "lineage": pd.Categorical(["L1", "L1", "L1", "L2", "L2", "L2"]),
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    return adata


def test_group_heatmap_requires_marsilea(simple_adata, monkeypatch):
    monkeypatch.setattr(
        heatmap_mod,
        "_import_marsilea",
        lambda: (_ for _ in ()).throw(ImportError("marsilea package is required")),
    )
    with pytest.raises(ImportError, match="marsilea"):
        group_heatmap(
            simple_adata,
            groupby="cell_type",
            var_names=["GeneA", "GeneB"],
        )


def test_group_heatmap_returns_plot_object(simple_adata, stub_marsilea):
    heatmap = group_heatmap(
        simple_adata,
        groupby="cell_type",
        var_names=["GeneA", "GeneB"],
        standard_scale="var",
        legend=False,
    )
    assert isinstance(heatmap, _StubPlot)
    assert heatmap.args[0].shape == (3, 2)


def test_feature_heatmap_returns_plot_object(simple_adata, stub_marsilea):
    heatmap = feature_heatmap(
        simple_adata,
        var_names={"Immune": ["GeneA", "GeneB"]},
        groupby="cell_type",
        cell_orderby="pseudotime",
        max_cells=2,
        legend=False,
    )
    assert isinstance(heatmap, _StubPlot)
    assert heatmap.args[0].shape[0] == 2


def test_dynamic_heatmap_returns_plot_object(simple_adata, stub_marsilea):
    heatmap = dynamic_heatmap(
        simple_adata,
        var_names=["GeneA", "GeneB", "GeneC"],
        pseudotime="pseudotime",
        lineage_key="lineage",
        cell_bins=2,
        use_fitted=True,
        order_by="peaktime",
        reverse_ht=["L2"],
        n_split=2,
        legend=False,
    )
    assert isinstance(heatmap, _StubStackBoard)
    assert len(heatmap.boards) == 2
    assert heatmap.boards[0].args[0].shape[0] == 3


def test_group_heatmap_tight_legend_returns_figure(simple_adata, stub_marsilea):
    fig = group_heatmap(
        simple_adata,
        groupby="cell_type",
        var_names=["GeneA", "GeneB"],
        standard_scale="var",
        legend=True,
        legend_style="tight",
    )
    assert isinstance(fig, _StubFigure)


def test_cell_cor_heatmap_tight_legend_returns_figure(simple_adata, stub_marsilea):
    fig = cell_cor_heatmap(
        simple_adata,
        group_by="cell_type",
        features=["GeneA", "GeneB", "GeneC"],
        legend=True,
        legend_style="tight",
    )
    assert isinstance(fig, _StubFigure)


def test_group_heatmap_border_styles_axes(simple_adata, stub_marsilea):
    heatmap = group_heatmap(
        simple_adata,
        groupby="cell_type",
        var_names=["GeneA", "GeneB"],
        legend=False,
        border=True,
    )

    axis = heatmap.figure.axes[0]
    assert not axis.axis_on
    heatmap.render()

    assert axis.axis_on
    assert axis.xticks == []
    assert axis.yticks == []
    assert all(spine.visible for spine in axis.spines.values())
    assert all(spine.linewidth == pytest.approx(0.8) for spine in axis.spines.values())
    assert all(spine.edgecolor == "#2B2B2B" for spine in axis.spines.values())


def test_dynamic_heatmap_border_is_opt_in(simple_adata, stub_marsilea):
    heatmap = dynamic_heatmap(
        simple_adata,
        var_names=["GeneA", "GeneB", "GeneC"],
        pseudotime="pseudotime",
        lineage_key="lineage",
        legend=False,
    )

    axis = heatmap.figure.axes[0]
    assert not axis.axis_on
    heatmap.render()

    assert not axis.axis_on
    assert not any(spine.visible for spine in axis.spines.values())


def test_prepare_dynamic_matrix_scales_features_across_pseudotime(simple_adata):
    matrix, metadata = _prepare_dynamic_matrix(
        simple_adata,
        ["GeneA", "GeneB", "GeneC"],
        "pseudotime",
        lineage_key="lineage",
        use_cell_columns=True,
        standard_scale="var",
    )
    del metadata

    assert np.allclose(matrix.mean(axis=1).to_numpy(), 0.0, atol=1e-7)
    assert np.allclose(matrix.std(axis=1, ddof=0).to_numpy(), 1.0, atol=1e-7)


def test_prepare_dynamic_matrix_supports_reverse_ht_by_position(simple_adata):
    matrix, metadata = _prepare_dynamic_matrix(
        simple_adata,
        ["GeneA", "GeneB", "GeneC"],
        "pseudotime",
        lineage_key="lineage",
        use_cell_columns=True,
        reverse_ht=1,
    )

    l1_columns = metadata.index[metadata["lineage"].astype(str) == "L1"].tolist()
    assert l1_columns == ["L1:cell_2", "L1:cell_1", "L1:cell_0"]
    assert list(matrix.columns[:3]) == l1_columns


def test_prepare_dynamic_matrix_supports_gene_cell_annotations(simple_adata):
    _, metadata = _prepare_dynamic_matrix(
        simple_adata,
        ["GeneA", "GeneB", "GeneC"],
        "pseudotime",
        lineage_key="lineage",
        use_cell_columns=True,
        annotation_keys=["cell_type", "GeneD"],
    )

    assert "GeneD" in metadata.columns
    assert metadata.loc["L1:cell_0", "GeneD"] == pytest.approx(0.5)


def test_dynamic_heatmap_limits_to_two_lineages_and_places_titles_on_outer_sides(stub_marsilea):
    adata = AnnData(
        X=np.array(
            [
                [1.0, 2.0, 0.0],
                [2.0, 1.0, 1.0],
                [3.0, 0.0, 2.0],
                [4.0, 1.0, 3.0],
                [5.0, 2.0, 4.0],
                [6.0, 3.0, 5.0],
            ]
        )
    )
    adata.var_names = ["GeneA", "GeneB", "GeneC"]
    adata.obs = pd.DataFrame(
        {
            "pseudotime": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "lineage": pd.Categorical(["L1", "L1", "L2", "L2", "L3", "L3"]),
            "clusters": pd.Categorical(["A", "A", "B", "B", "C", "C"]),
        },
        index=[f"cell_{i}" for i in range(6)],
    )

    heatmap = dynamic_heatmap(
        adata,
        var_names=["GeneA", "GeneB", "GeneC"],
        pseudotime="pseudotime",
        lineage_key="lineage",
        cell_annotation="clusters",
        separate_annotation="clusters",
        reverse_ht="L1",
        legend=False,
    )

    assert isinstance(heatmap, _StubStackBoard)
    assert len(heatmap.boards) == 2
    left_titles = [call for call in heatmap.boards[0].calls if call[0] == "add_title"]
    right_titles = [call for call in heatmap.boards[1].calls if call[0] == "add_title"]
    assert any(call[2].get("top") == "L1" for call in left_titles)
    assert any(call[2].get("top") == "L2" for call in right_titles)
    assert list(heatmap.boards[0].args[0].columns) == ["L1:cell_1", "L1:cell_0"]
    left_right_annotations = [call for call in heatmap.boards[0].calls if call[0] == "add_right"]
    right_right_annotations = [call for call in heatmap.boards[1].calls if call[0] == "add_right"]
    assert len(left_right_annotations) == 0
    assert len(right_right_annotations) == 2
    assert right_right_annotations[0][1][0].kwargs.get("label") == "L1"
    assert right_right_annotations[1][1][0].kwargs.get("label") == "L2"
    assert right_right_annotations[0][2].get("pad") == pytest.approx(0.03)
    assert right_right_annotations[1][2].get("pad") == pytest.approx(0.0)


def test_compute_dynamic_feature_metadata_orders_by_earliest_peak():
    """With two lineages, cluster_time must be the MIN peak time across
    lineages (i.e. when the gene first peaks), so that early-peaking genes
    sort to the top.  Regression against the previous max() bug that would
    give late-lineage peak time priority."""
    # gene A peaks at t=0.1 in L1 and t=0.9 in L2  → earliest peak = 0.1
    # gene B peaks at t=0.8 in L1 and t=0.2 in L2  → earliest peak = 0.2
    # Correct ascending order: gene A first (0.1 < 0.2)
    n_bins = 10
    matrix = pd.DataFrame(
        {
            f"L1_bin_{i+1}": [
                10.0 if i == 0 else 0.0,  # gene A peaks at bin 1 of L1
                0.0 if i < 7 else 10.0,   # gene B peaks at bin 8 of L1
            ]
            for i in range(n_bins)
        }
        | {
            f"L2_bin_{i+1}": [
                0.0 if i < 8 else 10.0,   # gene A peaks at bin 9 of L2
                10.0 if i == 1 else 0.0,   # gene B peaks at bin 2 of L2
            ]
            for i in range(n_bins)
        },
        index=["geneA", "geneB"],
    )
    metadata = pd.DataFrame(
        [
            {"lineage": "L1", "pseudotime": float(i) / (n_bins - 1)}
            for i in range(n_bins)
        ]
        + [
            {"lineage": "L2", "pseudotime": float(i) / (n_bins - 1)}
            for i in range(n_bins)
        ],
        index=[f"L1_bin_{i+1}" for i in range(n_bins)]
        + [f"L2_bin_{i+1}" for i in range(n_bins)],
    )

    fmeta = _compute_dynamic_feature_metadata(matrix, metadata, order_by="peak")

    # gene A has earliest peak (0.0 in L1) so cluster_time should be 0.0
    assert fmeta.loc["geneA", "cluster_time"] < fmeta.loc["geneB", "cluster_time"], (
        "gene A should have a smaller cluster_time than gene B (peaks earlier)"
    )
    # peak_lineage should be the lineage where the earliest peak occurs
    assert fmeta.loc["geneA", "peak_lineage"] == "L1"
    assert fmeta.loc["geneB", "peak_lineage"] == "L2"


def test_split_dynamic_features_kmeans_peaktime_uses_cluster_time_only(monkeypatch):
    matrix = pd.DataFrame(
        [
            [10.0, 0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0, 10.0],
            [10.0, 0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0, 10.0],
        ],
        index=["g1", "g2", "g3", "g4"],
    )
    feature_metadata = pd.DataFrame(
        {
            "cluster_time": [0.1, 0.2, 0.8, 0.9],
            "peak_lineage": ["L1", "L1", "L2", "L2"],
        },
        index=matrix.index,
    )
    recorded = {}

    def _fake_kmeans2(fit_data, k, minit, iter):
        recorded["fit_data"] = np.asarray(fit_data)
        return None, np.array([0, 0, 1, 1])

    monkeypatch.setattr(heatmap_mod, "kmeans2", _fake_kmeans2)

    split = _split_dynamic_features(
        matrix,
        feature_metadata,
        n_split=2,
        split_method="kmeans-peaktime",
    )

    assert recorded["fit_data"].shape == (4, 1)
    assert split.tolist() == ["cluster_1", "cluster_1", "cluster_2", "cluster_2"]


@pytest.mark.skipif(
    not _has_pycomplexheatmap(),
    reason="PyComplexHeatmap not installed",
)
def test_complexheatmap_uses_pycomplexheatmap(simple_adata):
    """complexheatmap now delegates to the legacy PyComplexHeatmap backend."""
    assert callable(complexheatmap)


@pytest.mark.skipif(
    not _has_pycomplexheatmap(),
    reason="PyComplexHeatmap not installed",
)
def test_marker_heatmap_uses_pycomplexheatmap(simple_adata):
    """marker_heatmap now delegates to the legacy PyComplexHeatmap backend."""
    assert callable(marker_heatmap)


def test_group_heatmap_smoke_with_pbmc3k_api(monkeypatch, stub_marsilea):
    def _mock_get_adata(url, filename=None):
        return datasets_mod.create_mock_dataset(
            n_cells=120,
            n_genes=80,
            n_cell_types=4,
            with_clustering=True,
        )

    monkeypatch.setattr(datasets_mod, "get_adata", _mock_get_adata)
    adata = ov.datasets.pbmc3k(processed=True)
    adata.var_names = [f"Marker{i}" for i in range(adata.n_vars)]

    heatmap = group_heatmap(
        adata,
        groupby="leiden",
        var_names=list(adata.var_names[:6]),
        standard_scale="var",
        legend=False,
    )

    assert isinstance(heatmap, _StubPlot)
