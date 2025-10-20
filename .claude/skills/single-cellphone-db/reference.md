# CellPhoneDB v5 quick commands

```python
import scanpy as sc
import omicverse as ov
import pandas as pd
import matplotlib.pyplot as plt

ov.plot_set()

adata = ov.read('data/cpdb/normalised_log_counts.h5ad')
adata = adata[adata.obs['cell_labels'].isin([
    'eEVT','iEVT','EVT_1','EVT_2','DC','dNK1','dNK2','dNK3',
    'VCT','VCT_CCC','VCT_fusing','VCT_p','GC','SCT'
])]
adata.obs['cell_labels'] = adata.obs['cell_labels'].astype('category').cat.remove_unused_categories()
print(adata.X.max())

cpdb_results, adata_cpdb = ov.single.run_cellphonedb_v5(
    adata,
    cpdb_file_path='./cellphonedb.zip',
    celltype_key='cell_labels',
    min_cell_fraction=0.005,
    min_genes=200,
    min_cells=3,
    iterations=1000,
    threshold=0.1,
    pvalue=0.05,
    threads=10,
    output_dir='./cpdb_results',
    cleanup_temp=True,
)

ov.utils.save(cpdb_results, 'data/cpdb/gex_cpdb_test.pkl')
adata_cpdb.write('data/cpdb/gex_cpdb_ad.h5ad')

color_dict = dict(zip(
    adata.obs['cell_labels'].cat.categories,
    adata.uns['cell_labels_colors']
))
viz = ov.pl.CellChatViz(adata_cpdb, palette=color_dict)

count_matrix, weight_matrix = viz.compute_aggregated_network(pvalue_threshold=0.05, use_means=True)
viz.netVisual_circle(weight_matrix, title="Interaction weights/strength", cmap='Reds', vertex_size_max=10, figsize=(5, 5))
viz.netVisual_circle(count_matrix, title="Number of interactions", cmap='Reds', vertex_size_max=10, figsize=(5, 5))

viz.netVisual_individual_circle(pvalue_threshold=0.05, vertex_size_max=10, edge_width_max=10, cmap='Blues', figsize=(20, 20), ncols=4)
viz.netVisual_individual_circle_incoming(pvalue_threshold=0.05, cmap='Reds', figsize=(20, 20), ncols=4, vertex_size_max=10)

pathway_comm = viz.compute_pathway_communication(method='mean', min_lr_pairs=2, min_expression=0.1)
sig_pathways, summary = viz.get_significant_pathways_v2(pathway_comm, strength_threshold=0.5, pvalue_threshold=0.05, min_significant_pairs=1)
pathways_show = ["Signaling by Interleukin"]

viz.netVisual_aggregate(signaling=pathways_show, layout='circle', figsize=(5, 5), vertex_size_max=10)
viz.netVisual_chord_cell(signaling=pathways_show[0], group_celltype=None, count_min=10, figsize=(5, 5), normalize_to_sender=True)
viz.netVisual_heatmap_marsilea(signaling=pathways_show, color_heatmap='Reds', add_dendrogram=False)

viz.netVisual_chord_LR(ligand_receptor_pairs='NCAM1_FGFR1', count_min=1, figsize=(5, 5), sources=['eEVT','dNK1'], rotate_names=True)

viz.netVisual_bubble_marsilea(
    sources_use=['eEVT','dNK1'],
    signaling=['Signaling by Fibroblast growth factor','Signaling by Galectin'],
    pvalue_threshold=0.01,
    mean_threshold=0.1,
    top_interactions=200,
    cmap='RdBu_r',
    show_pvalue=False,
    show_mean=False,
    show_count=True,
    add_dendrogram=False,
    font_size=10,
    group_pathways=True,
    figsize=(14, 2),
    transpose=True,
    title='Cell-Cell Communication Analysis',
    remove_isolate=False,
)

viz.netVisual_chord_gene(
    sources_use=['eEVT','dNK1'],
    signaling=['Signaling by Fibroblast growth factor'],
    pvalue_threshold=0.001,
    mean_threshold=4,
    gap=0.03,
    use_gradient=True,
    sort='size',
    directed=True,
    rotate_names=True,
    figsize=(4, 4),
)

centrality_scores = viz.netAnalysis_computeCentrality()
viz.netAnalysis_signalingRole_network_marsilea(signaling=None, measures=None, color_heatmap='Greens', width=8, height=2, font_size=10, title='EVT: Signaling Role Analysis', add_dendrogram=False, add_cell_colors=True, add_importance_bars=False, show_values=False)
viz.netAnalysis_signalingRole_scatter(signaling=None, x_measure='outdegree', y_measure='indegree', title='', figsize=(4, 4))
viz.netAnalysis_signalingRole_heatmap(pattern='outgoing', signaling=None, row_scale=False, figsize=(6, 6), cmap='Greens', show_totals=True)
viz.netAnalysis_signalingRole_heatmap(pattern='incoming', signaling=None, row_scale=False, figsize=(6, 6), cmap='Greens', show_totals=True)
```
