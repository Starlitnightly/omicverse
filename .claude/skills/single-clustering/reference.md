# Single-cell clustering quick commands

```python
import omicverse as ov
import scanpy as sc
import scvelo as scv
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

ov.plot_set()
adata = scv.datasets.dentategyrus()
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=3000)
adata.raw = adata  # Preserve full gene set before HVG subsetting
adata = adata[:, adata.var.highly_variable_features]

ov.pp.scale(adata)
ov.pp.pca(adata, layer='scaled', n_pcs=50)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, use_rep='scaled|original|X_pca')

# --- Standard clustering methods ---
# resolution: higher = more clusters. Start with 1.0 and adjust.
# For ~3k cells: resolution 0.5-1.0 gives 5-15 clusters; resolution 2+ gives 20+
ov.utils.cluster(adata, method='leiden', resolution=1)
ov.utils.cluster(adata, method='louvain', resolution=1)

# scICE: ensemble clustering. resolution_range defines sweep bounds, n_boot controls stability.
# Higher n_boot = more robust but slower. n_steps controls resolution grid granularity.
ov.utils.cluster(adata, method='scICE', use_rep='scaled|original|X_pca',
                  resolution_range=(4, 20), n_boot=50, n_steps=11)

# GMM: Gaussian Mixture Model. n_components = expected number of clusters.
# covariance_type='full' allows elliptical clusters (best for biology, but slowest).
# Use 'diag' for speed on large datasets.
ov.utils.cluster(adata, method='GMM', use_rep='scaled|original|X_pca',
                  n_components=21, covariance_type='full', tol=1e-9, max_iter=1000)

# --- Topic modeling (LDA) ---
# Treats gene expression as a document and discovers latent "topics" (cell programs).
# learning_rate: Adam optimizer LR. Lower (1e-4) for stability, higher (1e-2) for speed.
LDA_obj = ov.utils.LDA_topic(adata, feature_type='expression',
                             highly_variable_key='highly_variable_features',
                             layers='counts', learning_rate=1e-3)
LDA_obj.predicted(13)  # 13 = number of topics to extract
# LDA_threshold: cells with max topic probability below this are "unassigned"
LDA_obj.get_results_rfc(adata, use_rep='scaled|original|X_pca',
                        LDA_threshold=0.4, num_topics=13)

# --- cNMF program discovery ---
# components: range of K values to test. n_iter: factorizations per K.
cnmf_obj = ov.single.cNMF(adata, components=np.arange(5, 11), n_iter=20,
                          num_highvar_genes=2000, output_dir='example/cNMF', name='demo')
cnmf_obj.factorize(worker_i=0, total_workers=4)  # Parallelize across workers
cnmf_obj.combine(skip_missing_files=True)
# density_threshold: filter unstable programs. Higher = stricter (2.0 typical).
cnmf_obj.consensus(k=7, density_threshold=2.0)
cnmf_obj.get_results(adata)
# cNMF_threshold: minimum program usage to assign a cell to a program
cnmf_obj.get_results_rfc(adata, use_rep='scaled|original|X_pca', cNMF_threshold=0.5)

# Evaluate clustering quality via Adjusted Rand Index (0=random, 1=perfect match)
ari = adjusted_rand_score(adata.obs['clusters'], adata.obs['leiden'])
```

## Batch correction

```python
adata1 = ov.read('neurips2021_s1d3.h5ad'); adata1.obs['batch'] = 's1d3'
adata2 = ov.read('neurips2021_s2d1.h5ad'); adata2.obs['batch'] = 's2d1'
adata3 = ov.read('neurips2021_s3d7.h5ad'); adata3.obs['batch'] = 's3d7'
adata = sc.concat([adata1, adata2, adata3], merge='same')
adata = ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}, batch_key='batch')
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=3000)
adata.raw = adata
adata = adata[:, adata.var.highly_variable_features]
ov.pp.scale(adata)
ov.pp.pca(adata, layer='scaled', n_pcs=50)

# Each method stores corrected embedding in adata.obsm['X_<method>']
# harmony: fast, linear correction. Good default for moderate batch effects.
adata = ov.single.batch_correction(adata, batch_key='batch', methods='harmony', n_pcs=50)
# combat: parametric batch adjustment. Works well for bulk-like effects.
adata = ov.single.batch_correction(adata, batch_key='batch', methods='combat', n_pcs=50)
# scanorama: mutual nearest neighbors. Good for partially overlapping batches.
adata = ov.single.batch_correction(adata, batch_key='batch', methods='scanorama', n_pcs=50)
# scVI: deep generative model. Best for complex batch effects but needs GPU.
# n_layers/n_latent control network depth/bottleneck; gene_likelihood='nb' for count data.
adata = ov.single.batch_correction(adata, batch_key='batch', methods='scVI', n_layers=2,
                                   n_latent=30, gene_likelihood='nb')
# CellANOVA: controls for known confounders. control_dict maps pools to batches.
adata = ov.single.batch_correction(adata, batch_key='batch', methods='CellANOVA', n_pcs=50,
                                   control_dict={'pool1': ['s1d3', 's2d1']})

# Benchmark all methods with scib-metrics
from scib_metrics.benchmark import Benchmarker
bm = Benchmarker(adata, batch_key='batch', label_key='cell_type',
                 embedding_obsm_keys=['X_pca', 'X_combat', 'X_harmony', 'X_cellanova',
                                      'X_scanorama', 'X_mira_topic', 'X_mira_feature', 'X_scVI'])
bm.benchmark()
bm.plot_results_table(min_max_scale=False)
```
