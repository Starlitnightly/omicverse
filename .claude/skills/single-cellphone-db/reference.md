# CellPhoneDB v5 visualization quick commands

```python
import omicverse as ov

ov.plot_set()

adata = ov.read("data/cpdb/normalised_log_counts.h5ad")
adata = adata[adata.obs["cell_labels"].isin([
    "eEVT", "iEVT", "EVT_1", "EVT_2", "DC", "dNK1", "dNK2", "dNK3",
    "VCT", "VCT_CCC", "VCT_fusing", "VCT_p", "GC", "SCT"
])]
adata.obs["cell_labels"] = adata.obs["cell_labels"].astype("category").cat.remove_unused_categories()

cpdb_results, adata_cpdb = ov.single.run_cellphonedb_v5(
    adata,
    cpdb_file_path="./cellphonedb.zip",
    celltype_key="cell_labels",
    min_cell_fraction=0.005,
    min_genes=200,
    min_cells=3,
    iterations=1000,
    threshold=0.1,
    pvalue=0.05,
    threads=10,
    output_dir="./cpdb_results",
    cleanup_temp=True,
)

color_dict = dict(zip(
    adata.obs["cell_labels"].cat.categories,
    adata.uns["cell_labels_colors"],
))

# High-level public plotting APIs
ov.pl.ccc_heatmap(
    adata_cpdb,
    plot_type="heatmap",
    signaling=["Signaling by Interleukin"],
    cmap="Reds",
)

ov.pl.ccc_heatmap(
    adata_cpdb,
    plot_type="focused_heatmap",
    signaling=["Signaling by Fibroblast growth factor", "Signaling by Galectin"],
    min_interaction_threshold=0.2,
    cmap="Reds",
)

ov.pl.ccc_heatmap(
    adata_cpdb,
    plot_type="pathway_bubble",
    signaling=["Signaling by Fibroblast growth factor", "Signaling by Galectin"],
    sender_use=["eEVT", "dNK1"],
    pvalue_threshold=0.01,
    min_expression=0.1,
    top_n=100,
    group_pathways=True,
    transpose=True,
    cmap="RdBu_r",
)

ov.pl.ccc_heatmap(
    adata_cpdb,
    plot_type="bubble_lr",
    sender_use=["eEVT", "dNK1"],
    receiver_use=["VCT", "SCT"],
    pair_lr_use=["NCAM1_FGFR1", "TGFB1_TGFBR1"],
    pvalue_threshold=1.0,
    min_expression=0.0,
    top_n=50,
    transpose=True,
)

ov.pl.ccc_heatmap(
    adata_cpdb,
    plot_type="role_heatmap",
    pattern="incoming",
    signaling=None,
    cmap="Greens",
)

ov.pl.ccc_heatmap(
    adata_cpdb,
    plot_type="role_network_marsilea",
    signaling=None,
    cmap="Greens",
)

ov.pl.ccc_network_plot(
    adata_cpdb,
    plot_type="circle_focused",
    signaling=["Signaling by Fibroblast growth factor"],
    sender_use=["eEVT", "dNK1"],
    receiver_use=["VCT", "SCT"],
    figsize=(6, 6),
)

ov.pl.ccc_network_plot(
    adata_cpdb,
    plot_type="individual",
    signaling="Signaling by Fibroblast growth factor",
    sender_use=["eEVT", "dNK1"],
    receiver_use=["VCT", "SCT"],
    layout="hierarchy",
)

ov.pl.ccc_network_plot(
    adata_cpdb,
    plot_type="chord",
    signaling="Signaling by Fibroblast growth factor",
    sender_use=["eEVT", "dNK1"],
    receiver_use=["VCT", "SCT"],
    rotate_names=True,
)

ov.pl.ccc_network_plot(
    adata_cpdb,
    plot_type="gene_chord",
    signaling="Signaling by Fibroblast growth factor",
    sender_use=["eEVT", "dNK1"],
    receiver_use=["VCT", "SCT"],
    rotate_names=True,
)

ov.pl.ccc_network_plot(
    adata_cpdb,
    plot_type="lr_chord",
    pair_lr_use=["NCAM1_FGFR1"],
    sender_use=["eEVT", "dNK1"],
    receiver_use=["VCT", "SCT"],
    rotate_names=True,
)

ov.pl.ccc_network_plot(
    adata_cpdb,
    plot_type="diffusion",
    figsize=(7, 6),
)

ov.pl.ccc_stat_plot(
    adata_cpdb,
    plot_type="pathway_summary",
    top_n=15,
    strength_threshold=0.5,
    pvalue_threshold=0.05,
)

ov.pl.ccc_stat_plot(
    adata_cpdb,
    plot_type="lr_contribution",
    signaling="Signaling by Fibroblast growth factor",
    top_n=15,
)

ov.pl.ccc_stat_plot(
    adata_cpdb,
    plot_type="role_scatter",
    signaling=None,
)

# Low-level CellChatViz API for custom workflows
viz = ov.pl.CellChatViz(adata_cpdb, palette=color_dict)

count_matrix, weight_matrix = viz.compute_aggregated_network(
    pvalue_threshold=0.05,
    use_means=True,
)
viz.netVisual_circle(
    weight_matrix,
    title="Interaction weights/strength",
    cmap="Reds",
    vertex_size_max=10,
    figsize=(5, 5),
)

pathway_comm = viz.compute_pathway_communication(
    method="mean",
    min_lr_pairs=2,
    min_expression=0.1,
)
sig_pathways, summary = viz.get_significant_pathways_v2(
    pathway_comm,
    strength_threshold=0.5,
    pvalue_threshold=0.05,
    min_significant_pairs=1,
)

viz.netVisual_heatmap_marsilea(
    signaling=["Signaling by Fibroblast growth factor"],
    color_heatmap="Reds",
    add_dendrogram=False,
)

viz.netVisual_heatmap_marsilea_focused(
    signaling=["Signaling by Fibroblast growth factor"],
    min_interaction_threshold=0.2,
    color_heatmap="Reds",
    add_dendrogram=False,
)

viz.netVisual_bubble_marsilea(
    sources_use=["eEVT", "dNK1"],
    signaling=["Signaling by Fibroblast growth factor", "Signaling by Galectin"],
    pvalue_threshold=0.01,
    mean_threshold=0.1,
    top_interactions=200,
    cmap="RdBu_r",
    show_pvalue=True,
    show_mean=True,
    show_count=True,
    add_dendrogram=False,
    group_pathways=True,
    transpose=True,
)

viz.netVisual_bubble_lr(
    sources_use=["eEVT", "dNK1"],
    targets_use=["VCT", "SCT"],
    lr_pairs=["NCAM1_FGFR1", "TGFB1_TGFBR1"],
    pvalue_threshold=1.0,
    mean_threshold=0.0,
    show_all_pairs=True,
    transpose=True,
)

viz.netVisual_chord_gene(
    sources_use=["eEVT", "dNK1"],
    signaling=["Signaling by Fibroblast growth factor"],
    pvalue_threshold=0.001,
    mean_threshold=0.1,
    rotate_names=True,
    figsize=(5, 5),
)

viz.netAnalysis_computeCentrality()
viz.netAnalysis_signalingRole_network_marsilea(
    signaling=None,
    measures=None,
    color_heatmap="Greens",
    width=8,
    height=2,
    add_dendrogram=False,
    add_cell_colors=True,
    add_importance_bars=True,
    show_values=False,
)
viz.netAnalysis_signalingRole_scatter(
    signaling=None,
    x_measure="outdegree",
    y_measure="indegree",
    figsize=(4, 4),
)
viz.netAnalysis_contribution(
    signaling="Signaling by Fibroblast growth factor",
    pvalue_threshold=0.05,
    top_pairs=15,
    figsize=(10, 4),
)

viz.netVisual_diffusion(
    similarity_type="functional",
    layout="spring",
    figsize=(7, 6),
)
```
