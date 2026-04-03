from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sys
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path


matplotlib.use("Agg")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _ccc_plot_data import (
    make_comm_adata as build_comm_adata,
    make_comm_adata_shifted as build_comm_adata_shifted,
    make_comm_adata_with_receiver_only_group as build_comm_adata_with_receiver_only_group,
)

try:
    import omicverse as ov
    import omicverse.pl._ccc as ccc_mod
except Exception as exc:  # pragma: no cover - environment guard for optional deps
    ov = None
    ccc_mod = None
    pytestmark = pytest.mark.skip(reason=f"omicverse import failed in test env: {exc}")

@pytest.fixture()
def comm_adata() -> AnnData:
    return build_comm_adata()


@pytest.fixture()
def comparison_comm_adata() -> AnnData:
    return build_comm_adata_shifted()


@pytest.fixture()
def comm_adata_with_receiver_only_group() -> AnnData:
    return build_comm_adata_with_receiver_only_group()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _assert_figure_and_axes(fig, ax) -> None:
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


@pytest.mark.parametrize(
    ("plot_type", "kwargs"),
    [
        ("heatmap", {"display_by": "aggregation"}),
        ("focused_heatmap", {"display_by": "aggregation"}),
        ("heatmap", {"display_by": "interaction", "sender_use": "EVT_1", "facet_by": "sender", "top_n": 2}),
        ("dot", {"display_by": "aggregation", "top_n": 2}),
        ("dot", {"display_by": "interaction", "sender_use": "EVT_1", "facet_by": "sender", "top_n": 2}),
        ("bubble", {"display_by": "aggregation", "top_n": 2}),
        ("bubble_lr", {"pair_lr_use": "MDK_SDC1"}),
        ("bubble", {"display_by": "interaction", "receiver_use": "dNK1", "facet_by": "receiver", "top_n": 2}),
        ("pathway_bubble", {"signaling": "MK", "top_n": 2}),
        ("role_heatmap", {"pattern": "incoming", "top_n": 2}),
        ("role_network", {}),
        ("role_network_marsilea", {}),
    ],
)
def test_ccc_heatmap_variants_return_figure_and_axes(comm_adata: AnnData, plot_type: str, kwargs: dict) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type=plot_type,
        show=False,
        **kwargs,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_diff_heatmap_returns_figure_and_axes(
    comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="diff_heatmap",
        pattern="all",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_diff_heatmap_prefers_diverging_cmap_and_text_for_small_discrete_matrix(
    monkeypatch, comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    captured: dict[str, object] = {}

    def _fake_diff(*args, **kwargs):
        matrix = pd.DataFrame(
            np.ones((3, 3), dtype=float),
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        return matrix, "Delta score"

    def _fake_cluster(matrix, **kwargs):
        return matrix

    def _fake_plot(matrix, **kwargs):
        captured["kwargs"] = kwargs
        fig, ax = plt.subplots()
        return fig, ax

    monkeypatch.setattr(ccc_mod, "_diff_role_matrix", _fake_diff)
    monkeypatch.setattr(ccc_mod, "_apply_cluster_order", _fake_cluster)
    monkeypatch.setattr(ccc_mod, "_plot_heatmap_matrix", _fake_plot)

    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="diff_heatmap",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert captured["kwargs"]["cmap"] == "RdBu_r"
    assert captured["kwargs"]["add_text"] is True


def test_ccc_heatmap_diff_heatmap_requires_comparison_adata(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="comparison_adata"):
        ov.pl.ccc_heatmap(
            comm_adata,
            plot_type="diff_heatmap",
            show=False,
        )


def test_ccc_heatmap_aggregation_routes_through_cellchatviz_backend(monkeypatch, comm_adata: AnnData) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netVisual_heatmap_marsilea(self, **kwargs):
            called["kwargs"] = kwargs
            return "plotter"

    def _fake_build(adata, *, palette=None):
        called["palette"] = palette
        called["adata"] = adata
        return _StubViz()

    def _fake_render(plotter, *, title=None, add_custom_legends=False):
        fig, ax = plt.subplots()
        called["plotter"] = plotter
        return fig, ax

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", _fake_build)
    monkeypatch.setattr(ccc_mod, "_render_plotter_figure", _fake_render)

    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="heatmap",
        display_by="aggregation",
        signaling="MK",
        show=False,
    )

    _assert_figure_and_axes(fig, ax)
    assert called["plotter"] == "plotter"
    assert called["kwargs"]["signaling"] == ["MK"]


def test_ccc_heatmap_aggregation_rejects_interaction_filters(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="interaction_use"):
        ov.pl.ccc_heatmap(
            comm_adata,
            plot_type="heatmap",
            display_by="aggregation",
            interaction_use="TGFB1 - TGFBR1",
            show=False,
        )


def test_ccc_heatmap_role_heatmap_rejects_sender_filter(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="sender_use"):
        ov.pl.ccc_heatmap(
            comm_adata,
            plot_type="role_heatmap",
            sender_use="EVT_1",
            show=False,
        )


def test_ccc_heatmap_pathway_bubble_rejects_pair_lr_filter(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="pair_lr_use"):
        ov.pl.ccc_heatmap(
            comm_adata,
            plot_type="pathway_bubble",
            signaling="MK",
            pair_lr_use="MDK_SDC1",
            show=False,
        )


@pytest.mark.parametrize(
    ("plot_type", "kwargs"),
    [
        ("circle", {}),
        ("circle_focused", {}),
        ("individual_outgoing", {}),
        ("individual_incoming", {}),
        ("individual", {"signaling": "MK"}),
        ("arrow", {}),
        ("arrow", {"display_by": "interaction"}),
        ("sigmoid", {}),
        ("sigmoid", {"display_by": "interaction"}),
        ("embedding_network", {}),
        ("pathway", {"signaling": "MK"}),
        ("chord", {"signaling": "MK"}),
        ("lr_chord", {"pair_lr_use": "MDK_SDC1"}),
        ("gene_chord", {"signaling": "MK"}),
        ("diffusion", {}),
        ("individual_lr", {"pair_lr_use": "MDK_SDC1"}),
        ("bipartite", {"ligand": "TGFB1"}),
    ],
)
def test_ccc_network_plot_base_variants_return_figure_and_axes(
    comm_adata: AnnData, plot_type: str, kwargs: dict
) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type=plot_type,
        show=False,
        **kwargs,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_network_plot_diff_network_returns_figure_and_axes(
    comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="diff_network",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


@pytest.mark.parametrize(
    ("plot_type", "method_name"),
    [
        ("individual_outgoing", "netVisual_individual_circle"),
        ("individual_incoming", "netVisual_individual_circle_incoming"),
    ],
)
def test_ccc_network_plot_individual_circle_variants_use_more_visible_edge_defaults(
    monkeypatch, comm_adata: AnnData, plot_type: str, method_name: str
) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netVisual_individual_circle(self, **kwargs):
            called["kwargs"] = kwargs
            fig, _ = plt.subplots()
            return fig

        def netVisual_individual_circle_incoming(self, **kwargs):
            called["kwargs"] = kwargs
            fig, _ = plt.subplots()
            return fig

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", lambda adata, *, palette=None: _StubViz())

    fig, ax = ov.pl.ccc_network_plot(comm_adata, plot_type=plot_type, show=False)
    _assert_figure_and_axes(fig, ax)
    assert called["kwargs"]["edge_width_max"] == 12


def test_ccc_network_plot_diff_network_uses_cell_type_node_colors(
    comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="diff_network",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    facecolors = []
    for collection in ax.collections:
        if hasattr(collection, "get_facecolors"):
            colors = collection.get_facecolors()
            if len(colors):
                facecolors.extend([tuple(round(channel, 3) for channel in color[:3]) for color in colors])
    assert facecolors
    assert any(color != (0.957, 0.945, 0.918) for color in facecolors)


@pytest.mark.parametrize(
    ("plot_type", "expected_title"),
    [
        ("scatter", "Outgoing vs incoming communication"),
        ("role_scatter", "Signaling role scatter"),
    ],
)
def test_ccc_stat_plot_scatter_variants_route_through_cellchatviz_scatter(
    monkeypatch, comm_adata: AnnData, plot_type: str, expected_title: str
) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netAnalysis_signalingRole_scatter(self, **kwargs):
            called["kwargs"] = kwargs
            fig, ax = plt.subplots()
            ax.text(0.1, 0.2, "EVT_1")
            return fig, ax

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", lambda adata, *, palette=None: _StubViz())

    fig, ax = ov.pl.ccc_stat_plot(comm_adata, plot_type=plot_type, show=False)
    _assert_figure_and_axes(fig, ax)
    assert called["kwargs"]["x_measure"] == "outdegree"
    assert called["kwargs"]["y_measure"] == "indegree"
    assert called["kwargs"]["title"] == expected_title


@pytest.mark.parametrize(
    ("plot_type", "kwargs"),
    [
        ("bar", {}),
        ("sankey", {}),
        ("box", {}),
        ("violin", {}),
        ("scatter", {}),
        ("role_scatter", {}),
        ("role_network", {}),
        ("role_network_marsilea", {}),
        ("pathway_summary", {}),
        ("gene", {"signaling": "TGFb"}),
    ],
)
def test_ccc_stat_plot_core_variants_return_figure_and_axes(
    comm_adata: AnnData, plot_type: str, kwargs: dict
) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type=plot_type,
        top_n=2,
        show=False,
        **kwargs,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_stat_plot_interaction_sankey_returns_figure_and_axes(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="sankey",
        display_by="interaction",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    interaction_texts = [text for text in ax.texts if "CXCL12" in text.get_text() or "TGFB1" in text.get_text()]
    assert interaction_texts
    assert all(text.get_rotation() == 0 for text in interaction_texts)
    receiver_rectangles = [
        patch
        for patch in ax.patches
        if hasattr(patch, "get_facecolor")
        and hasattr(patch, "get_width")
        and hasattr(patch, "get_x")
        and patch.get_width() > 0.03
        and patch.get_x() >= 0.89
    ]
    assert receiver_rectangles
    assert any(tuple(round(channel, 3) for channel in patch.get_facecolor()[:3]) != (0.851, 0.851, 0.851) for patch in receiver_rectangles)


@pytest.mark.parametrize("plot_type", ["arrow", "sigmoid"])
def test_ccc_network_plot_interaction_flow_has_middle_stage_labels(comm_adata: AnnData, plot_type: str) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type=plot_type,
        display_by="interaction",
        top_n=3,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    texts = [text.get_text() for text in ax.texts]
    assert "Ligand-Receptor" in texts
    assert any("CXCL12" in text or "TGFB1" in text or "MDK" in text for text in texts)
    middle_rectangles = [
        patch
        for patch in ax.patches
        if hasattr(patch, "get_width")
        and hasattr(patch, "get_height")
        and hasattr(patch, "get_x")
        and 0.4 <= patch.get_x() <= 0.55
        and patch.get_width() >= 0.09
        and patch.get_height() >= 0.07
    ]
    assert middle_rectangles


def test_build_flow_plot_frames_prunes_dense_interaction_branches() -> None:
    long_df = pd.DataFrame(
        {
            "sender": ["S1", "S2", "S3", "S4", "S5", "S6"] * 2,
            "receiver": ["R1", "R2", "R3", "R4", "R5", "R6"] * 2,
            "interaction": ["L1_R1"] * 6 + ["L2_R2"] * 6,
            "pair_lr": ["L1_R1"] * 6 + ["L2_R2"] * 6,
            "score": np.linspace(12.0, 1.0, 12),
            "significant": np.ones(12, dtype=float),
        }
    )
    node_df, edge_df, column_titles = ccc_mod._build_flow_plot_frames(
        long_df,
        display_by="interaction",
        value="sum",
        top_n=4,
    )

    assert column_titles == [(0.0, "Sender"), (0.5, "Ligand-Receptor"), (1.0, "Receiver")]
    interaction_nodes = node_df.loc[node_df["column"] == "interaction", "label"].astype(str).tolist()
    assert set(interaction_nodes) == {"L1_R1", "L2_R2"}
    sender_edges = edge_df.loc[edge_df["from_id"].astype(str).str.startswith("sender::"), :]
    receiver_edges = edge_df.loc[edge_df["to_id"].astype(str).str.startswith("receiver::"), :]
    assert sender_edges.groupby("to_id").size().max() <= 3
    assert receiver_edges.groupby("from_id").size().max() <= 3


def test_ccc_network_plot_bipartite_aligns_ligand_and_receptor_labels(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type="bipartite",
        ligand="TGFB1",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)

    text_lookup = {
        " ".join(text.get_text().split()): text
        for text in ax.texts
        if text.get_text().strip()
    }
    assert all(not label.startswith("Ligand focus:") for label in text_lookup)

    ligand_text = text_lookup["TGFB1"]
    receptor_raw = (
        comm_adata.var.loc[comm_adata.var["ligand"].astype(str) == "TGFB1", "receptor"].astype(str).iloc[0]
    )
    receptor_tokens = [token for token in receptor_raw.replace("_", " ").split() if token]
    receptor_label = next(
        label
        for label in text_lookup
        if all(any(token.lower() in part.lower() for part in label.split()) for token in receptor_tokens)
    )
    receptor_text = text_lookup[receptor_label]
    assert abs(ligand_text.get_position()[1] - receptor_text.get_position()[1]) < 1e-6


def test_ccc_stat_plot_lr_contribution_returns_figure_and_axes(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="lr_contribution",
        signaling="TGFb",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_stat_plot_bar_cleans_complex_interaction_labels(comm_adata: AnnData) -> None:
    adata = comm_adata.copy()
    adata.var = adata.var.copy()
    adata.var["interaction_name_2"] = ""
    adata.var.loc[adata.var_names[0], "interaction_name"] = "FN1_integrin_a5b1_complex"
    adata.var.loc[adata.var_names[0], "interacting_pair"] = "FN1_integrin_a5b1_complex"
    adata.layers["means"][:, 0] = 50.0

    fig, ax = ov.pl.ccc_stat_plot(
        adata,
        plot_type="bar",
        top_n=1,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    labels = [" ".join(tick.get_text().split()) for tick in ax.get_yticklabels() if tick.get_text()]
    assert any("FN1 integrin a5b1" in label for label in labels)
    assert all("complex" not in label.lower() for label in labels)


def test_ccc_stat_plot_gene_cleans_complex_gene_labels(comm_adata: AnnData) -> None:
    adata = comm_adata.copy()
    adata.var = adata.var.copy()
    adata.var.loc[adata.var_names[0], "classification"] = "TGFb"
    adata.var.loc[adata.var_names[0], "pathway_name"] = "TGFb"
    adata.var.loc[adata.var_names[0], "gene_a"] = "complex:IL27"
    adata.var.loc[adata.var_names[0], "gene_b"] = "complex:IL1_receptor"
    adata.layers["means"][:, 0] = 60.0

    fig, ax = ov.pl.ccc_stat_plot(
        adata,
        plot_type="gene",
        signaling="TGFb",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    labels = " ".join(tick.get_text() for tick in ax.get_yticklabels() if tick.get_text())
    assert "complex:" not in labels.lower()
    assert "receptor" not in labels.lower()
    assert "IL27" in labels or "IL1" in labels


def test_ccc_stat_plot_gene_uses_source_expression_adata(comm_adata: AnnData) -> None:
    genes = list(
        dict.fromkeys(
            comm_adata.var["gene_a"].astype(str).tolist() + comm_adata.var["gene_b"].astype(str).tolist()
        )
    )
    obs = pd.DataFrame(
        {
            "cell_labels": ["EVT_1", "EVT_1", "dNK1", "dNK1", "VCT", "VCT"],
        },
        index=[f"cell_{idx}" for idx in range(6)],
    )
    matrix = np.arange(len(obs.index) * len(genes), dtype=float).reshape(len(obs.index), len(genes)) + 1.0
    expr_adata = AnnData(matrix, obs=obs, var=pd.DataFrame(index=genes))

    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="gene",
        signaling="TGFb",
        source_adata=expr_adata,
        source_groupby="cell_labels",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert ax.get_xlabel() == "cell_labels"
    assert "expression" in ax.get_title().lower()
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]
    assert any("(L)" in label or "(R)" in label for label in ytick_labels)


@pytest.mark.parametrize(
    ("plot_type", "kwargs"),
    [
        ("comparison", {"compare_by": "overall"}),
        ("comparison", {"compare_by": "celltype", "pattern": "incoming"}),
        ("ranknet", {}),
        ("role_change", {"idents_use": "EVT_1"}),
    ],
)
def test_ccc_stat_plot_comparison_variants_return_figure_and_axes(
    comm_adata: AnnData,
    comparison_comm_adata: AnnData,
    plot_type: str,
    kwargs: dict,
) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type=plot_type,
        top_n=3,
        show=False,
        **kwargs,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_stat_plot_box_sender_facets_by_sender(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="box",
        facet_by="sender",
        top_n=3,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    titles = {axis.get_title() for axis in fig.axes if axis.get_title()}
    assert "EVT_1" in titles
    assert len(titles) >= 2


def test_ccc_stat_plot_role_change_labels_signaling_for_selected_identity(
    comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="role_change",
        idents_use="EVT_1",
        top_n=3,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    text_labels = [text.get_text() for text in ax.texts if text.get_text()]
    pathway_labels = set(comm_adata.var["classification"].astype(str).tolist())
    assert any(label in pathway_labels for label in text_labels)
    assert "EVT_1" not in text_labels


@pytest.mark.parametrize("plot_type", ["comparison", "ranknet", "role_change"])
def test_ccc_stat_plot_comparison_variants_require_comparison_adata(
    comm_adata: AnnData, plot_type: str
) -> None:
    with pytest.raises(ValueError, match="comparison_adata"):
        ov.pl.ccc_stat_plot(
            comm_adata,
            plot_type=plot_type,
            show=False,
        )


def test_ccc_heatmap_save_writes_output(comm_adata: AnnData, tmp_path) -> None:
    output = tmp_path / "ccc_heatmap.png"
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="bubble",
        display_by="interaction",
        save=str(output),
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert output.exists()


def test_ccc_network_plot_diff_network_requires_comparison_adata(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="comparison_adata"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="diff_network",
            show=False,
        )


def test_ccc_network_plot_pathway_requires_signaling(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="signaling"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="pathway",
            show=False,
        )


def test_ccc_network_plot_pathway_rejects_interaction_filters(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="interaction_use"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="pathway",
            signaling="MK",
            interaction_use="TGFB1 - TGFBR1",
            show=False,
        )


def test_ccc_network_plot_diffusion_rejects_sender_filter(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="sender_use"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="diffusion",
            sender_use="EVT_1",
            show=False,
        )


def test_ccc_network_plot_embedding_requires_positions(comm_adata: AnnData) -> None:
    comm_adata = comm_adata.copy()
    comm_adata.uns.pop("node_positions", None)
    with pytest.raises(ValueError, match="node_positions"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="embedding_network",
            show=False,
        )


def test_ccc_stat_plot_raises_for_empty_filtered_data(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="No communication records remain"):
        ov.pl.ccc_stat_plot(
            comm_adata,
            plot_type="lr_contribution",
            signaling="NotAPathway",
            show=False,
        )


def test_ccc_stat_plot_lr_contribution_routes_through_cellchatviz_backend(monkeypatch, comm_adata: AnnData) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netAnalysis_contribution(self, **kwargs):
            called["kwargs"] = kwargs
            fig, ax = plt.subplots()
            return None, fig, (ax,)

    def _fake_build(adata, *, palette=None):
        called["adata"] = adata
        called["palette"] = palette
        return _StubViz()

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", _fake_build)

    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="lr_contribution",
        signaling="TGFb",
        top_n=4,
        show=False,
    )

    _assert_figure_and_axes(fig, ax)
    assert called["kwargs"]["signaling"] == ["TGFb"]
    assert called["kwargs"]["top_pairs"] == 4


def test_ccc_stat_plot_lr_contribution_rejects_sender_filter_with_signaling(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="sender_use"):
        ov.pl.ccc_stat_plot(
            comm_adata,
            plot_type="lr_contribution",
            signaling="TGFb",
            sender_use="EVT_1",
            show=False,
        )


def test_ccc_stat_plot_pathway_summary_rejects_pair_lr_filter(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="pair_lr_use"):
        ov.pl.ccc_stat_plot(
            comm_adata,
            plot_type="pathway_summary",
            pair_lr_use="MDK_SDC1",
            show=False,
        )


def test_ccc_heatmap_sender_facet_groups_columns_without_sender_filter(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="dot",
        display_by="interaction",
        facet_by="sender",
        top_n=2,
        top_n_pairs=0,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    ticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert ticklabels == [
        "EVT_1 -> EVT_1",
        "EVT_1 -> VCT",
        "EVT_1 -> dNK1",
        "VCT -> EVT_1",
        "VCT -> VCT",
        "VCT -> dNK1",
        "dNK1 -> EVT_1",
        "dNK1 -> VCT",
        "dNK1 -> dNK1",
    ]


def test_ccc_heatmap_focused_heatmap_avoids_extra_annotation_text(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="focused_heatmap",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    non_title_texts = [
        text.get_text().strip()
        for axis in fig.axes
        for text in axis.texts
        if text.get_text().strip() and "focused communication heatmap" not in text.get_text().strip().lower()
    ]
    assert non_title_texts == []


def test_cellchatviz_role_network_variants_hide_value_one_annotations_and_marsilea_uses_dark_text(
    comm_adata: AnnData,
) -> None:
    viz = ov.pl.CellChatViz(comm_adata)
    n_cells = len(viz.cell_types)
    viz.centrality_scores = {
        "outdegree": np.array([1.0, 0.42, 0.0][:n_cells], dtype=float),
        "indegree": np.array([0.18, 1.0, 0.0][:n_cells], dtype=float),
        "flow_betweenness": np.array([0.0, 0.27, 1.0][:n_cells], dtype=float),
        "information": np.array([0.66, 0.0, 1.0][:n_cells], dtype=float),
    }

    fig = viz.netAnalysis_signalingRole_network(show_values=True)
    ax = fig.axes[0]
    texts = [text.get_text().strip() for text in ax.texts if text.get_text().strip()]
    assert "1.00" not in texts
    assert any(text in {"0.42", "0.18", "0.27", "0.66"} for text in texts)
    plt.close(fig)

    plotter = viz.netAnalysis_signalingRole_network_marsilea(show_values=True)
    fig = ccc_mod._render_plot(plotter)
    text_items = [
        text
        for axis in fig.axes
        for text in axis.texts
        if text.get_text().strip() and text.get_text().strip().replace(".", "", 1).isdigit()
    ]
    assert text_items
    assert all(text.get_text().strip() != "1.00" for text in text_items)
    assert all(text.get_color() != "white" for text in text_items)
    plt.close(fig)


def test_ccc_heatmap_interaction_top_n_also_limits_pair_columns(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="heatmap",
        display_by="interaction",
        facet_by="sender",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert len(ax.get_xticklabels()) <= 2


def test_ccc_heatmap_accepts_named_palette_string(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type="circle",
        palette="Set1",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_pair_lr_filter_prefers_interacting_pair(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="heatmap",
        display_by="interaction",
        pair_lr_use="MDK_SDC1",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    ticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert ticklabels == ["MDK - SDC1"]


def test_ccc_scatter_includes_receiver_only_groups(comm_adata_with_receiver_only_group: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata_with_receiver_only_group,
        plot_type="scatter",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    labels = {text.get_text() for text in ax.texts}
    assert labels == {"EVT_1", "dNK1", "SCT"}


def test_ccc_violin_uses_non_vertical_interaction_labels(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="violin",
        top_n=4,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert all(tick.get_rotation() == 0 for tick in ax.get_xticklabels())
