# SCENIC quick commands

## Full pipeline

```python
import omicverse as ov
import scanpy as sc
import glob

ov.plot_set()

# --- Load data ---
adata = ov.single.mouse_hsc_nestorowa16()  # Example: mouse hematopoiesis
# Or: adata = ov.read("your_data.h5ad")

# --- Database paths (download first — see SKILL.md) ---
db_glob = 'scenic_db/*.feather'
motif_path = 'scenic_db/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl'

# --- Initialize ---
# n_jobs: match to available CPU cores for parallel motif enrichment
scenic = ov.single.SCENIC(adata, db_glob=db_glob, motif_path=motif_path, n_jobs=12)

# --- Stage 1: GRN inference ---
# method='regdiffusion': deep learning GRN (10x faster than GRNBoost2)
# layer: must be RAW counts (not log-normalized)
edgelist = scenic.cal_grn(method='regdiffusion', layer='raw_count')

# --- Stage 2+3: Regulon pruning + AUCell scoring ---
# rho_mask_dropouts=True: handles scRNA-seq dropout noise in correlation
# thresholds: cisTarget enrichment thresholds for module pruning
regulon_ad = scenic.cal_regulons(rho_mask_dropouts=True, seed=42)

# --- Inspect results ---
print(f"Regulons found: {len(scenic.regulons)}")
print(f"AUCell matrix shape: {scenic.auc_mtx.shape}")
scenic.auc_mtx.head()
```

## Regulon Specificity Scores (RSS)

```python
# RSS: identifies which TFs are most specific to each cell type
# Uses Jensen-Shannon divergence (0=ubiquitous, 1=perfectly specific)
rss = ov.single.regulon_specificity_scores(scenic.auc_mtx, adata.obs['cell_type'])
rss.head()

# Plot top regulons per cell type
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ct in enumerate(['HSC', 'Monocyte', 'Erythroid']):
    rss[ct].sort_values(ascending=False).head(10).plot.barh(ax=axes[i])
    axes[i].set_title(f'Top TFs: {ct}')
plt.tight_layout()
```

## Binary regulon activity

```python
from pyscenic.binarize import binarize

# Convert continuous AUCell scores to binary (on/off) per regulon
# num_workers: CPU workers for threshold estimation
binary_mtx, thresholds = binarize(scenic.auc_mtx, num_workers=12)
```

## Visualization

```python
# Regulon activity on UMAP embedding
ov.pl.embedding(regulon_ad, basis='X_umap', color=['Ets1(+)', 'E2f8(+)', 'Bhlhe40(+)'])

# GRN network graph (if network was built)
# build_correlation_network_umap_layout returns: G (networkx), pos (coordinates), corr_matrix
G, pos, corr = ov.single.build_correlation_network_umap_layout(
    embedding_df, correlation_threshold=0.95, umap_neighbors=15
)
G = ov.single.add_tf_regulation(G, tf_gene_dict)
ov.single.plot_grn(G, pos, tf_list=['Ets1', 'E2f8'], temporal_df=temporal_df,
                   tf_gene_dict=tf_gene_dict, top_tf_target_num=5, figsize=(8, 8))
```

## Save and reload

```python
# Save SCENIC object (pickle)
import pickle
with open('scenic_obj.pkl', 'wb') as f:
    pickle.dump(scenic, f)

# Save regulon AnnData (h5ad)
regulon_ad.write('scenic_regulon_ad.h5ad')

# Save AUCell matrix
scenic.auc_mtx.to_csv('aucell_matrix.csv')
```

## Temporal analysis (advanced)

```python
# Gene temporal center: when along pseudotime each gene peaks
temporal_df = ov.single.batch_calculate_gene_temporal_centers(adata, pseudotime_key='dpt_pseudotime')

# Cluster genes by temporal pattern
clusters = ov.single.get_temporal_gene_clusters(temporal_df, n_clusters=5)

# Save full temporal analysis
ov.single.save_temporal_analysis_results(temporal_df, clusters, output_dir='temporal_results/')
```
