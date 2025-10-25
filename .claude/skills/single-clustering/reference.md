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
adata.raw = adata
adata = adata[:, adata.var.highly_variable_features]

ov.pp.scale(adata)
ov.pp.pca(adata, layer='scaled', n_pcs=50)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, use_rep='scaled|original|X_pca')

ov.utils.cluster(adata, method='leiden', resolution=1)
ov.utils.cluster(adata, method='louvain', resolution=1)
ov.utils.cluster(adata, method='scICE', use_rep='scaled|original|X_pca',
                  resolution_range=(4, 20), n_boot=50, n_steps=11)
ov.utils.cluster(adata, method='GMM', use_rep='scaled|original|X_pca',
                  n_components=21, covariance_type='full', tol=1e-9, max_iter=1000)

LDA_obj = ov.utils.LDA_topic(adata, feature_type='expression',
                             highly_variable_key='highly_variable_features',
                             layers='counts', learning_rate=1e-3)
LDA_obj.predicted(13)
LDA_obj.get_results_rfc(adata, use_rep='scaled|original|X_pca',
                        LDA_threshold=0.4, num_topics=13)

cnmf_obj = ov.single.cNMF(adata, components=np.arange(5, 11), n_iter=20,
                          num_highvar_genes=2000, output_dir='example/cNMF', name='demo')
cnmf_obj.factorize(worker_i=0, total_workers=4)
cnmf_obj.combine(skip_missing_files=True)
cnmf_obj.consensus(k=7, density_threshold=2.0)
cnmf_obj.get_results(adata)
cnmf_obj.get_results_rfc(adata, use_rep='scaled|original|X_pca', cNMF_threshold=0.5)

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

adata = ov.single.batch_correction(adata, batch_key='batch', methods='harmony', n_pcs=50)
adata = ov.single.batch_correction(adata, batch_key='batch', methods='combat', n_pcs=50)
adata = ov.single.batch_correction(adata, batch_key='batch', methods='scanorama', n_pcs=50)
adata = ov.single.batch_correction(adata, batch_key='batch', methods='scVI', n_layers=2,
                                   n_latent=30, gene_likelihood='nb')
adata = ov.single.batch_correction(adata, batch_key='batch', methods='CellANOVA', n_pcs=50,
                                   control_dict={'pool1': ['s1d3', 's2d1']})

from scib_metrics.benchmark import Benchmarker
bm = Benchmarker(adata, batch_key='batch', label_key='cell_type',
                 embedding_obsm_keys=['X_pca', 'X_combat', 'X_harmony', 'X_cellanova',
                                      'X_scanorama', 'X_mira_topic', 'X_mira_feature', 'X_scVI'])
bm.benchmark()
bm.plot_results_table(min_max_scale=False)
```
