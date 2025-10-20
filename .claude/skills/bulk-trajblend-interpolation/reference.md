# BulkTrajBlend quick commands

```python
import omicverse as ov
import scanpy as sc
import scvelo as scv
from omicverse.utils import mde
import matplotlib.pyplot as plt

ov.plot_set()

adata = scv.datasets.dentategyrus()
print(adata.obs['clusters'].value_counts())

bulk_df = ov.utils.read('data/GSE74985_mergedCount.txt.gz', index_col=0)
bulk_df = ov.bulk.Matrix_ID_mapping(bulk_df, 'genesets/pair_GRCm39.tsv')

bulktb = ov.bulk2single.BulkTrajBlend(
    bulk_seq=bulk_df,
    single_seq=adata,
    bulk_group=['dg_d_1', 'dg_d_2', 'dg_d_3'],
    celltype_key='clusters',
)

bulktb.vae_configure(cell_target_num=100)

vae_net = bulktb.vae_train(
    batch_size=512,
    learning_rate=1e-4,
    hidden_size=256,
    epoch_num=3500,
    vae_save_dir='data/bulk2single/save_model',
    vae_save_name='dg_btb_vae',
    generate_save_dir='data/bulk2single/output',
    generate_save_name='dg_btb',
)

bulktb.vae_load('data/bulk2single/save_model/dg_btb_vae.pth')

generated = bulktb.vae_generate(leiden_size=25)

overview_ax = ov.bulk2single.bulk2single_plot_cellprop(
    generated,
    celltype_key='clusters',
)
plt.grid(False)

bulktb.gnn_configure(max_epochs=2000, use_rep='X', neighbor_rep='X_pca')
bulktb.gnn_train()
bulktb.gnn_load('save_model/gnn.pth')

communities = bulktb.gnn_generate()
print(communities.head())

bulktb.nocd_obj.adata.obsm['X_mde'] = mde(bulktb.nocd_obj.adata.obsm['X_pca'])
sc.pl.embedding(
    bulktb.nocd_obj.adata,
    basis='X_mde',
    color=['clusters', 'nocd_n'],
    wspace=0.4,
    palette=ov.utils.pyomic_palette(),
)

interpolated = bulktb.interpolation('OPC')
sc.pp.highly_variable_genes(interpolated, min_mean=0.0125, max_mean=3, min_disp=0.5)
interpolated = interpolated[:, interpolated.var.highly_variable]
sc.pp.scale(interpolated, max_value=10)
sc.tl.pca(interpolated, n_comps=100)

adata_copy = adata.copy()
sc.pp.normalize_total(adata_copy, target_sum=1e4)
sc.pp.log1p(adata_copy)
adata_copy.raw = adata_copy
sc.pp.highly_variable_genes(adata_copy, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata_copy = adata_copy[:, adata_copy.var.highly_variable]
sc.pp.scale(adata_copy, max_value=10)
sc.tl.pca(adata_copy, n_comps=100)

adata_copy.obsm['X_mde'] = mde(adata_copy.obsm['X_pca'])
interpolated.obsm['X_mde'] = mde(interpolated.obsm['X_pca'])

overlay_palette = sc.pl.palettes.default_102

overlay = ov.utils.embedding(
    adata_copy,
    basis='X_mde',
    color=['clusters'],
    wspace=0.4,
    frameon='small',
    palette=overlay_palette,
)

ov.utils.embedding(
    interpolated,
    basis='X_mde',
    color=['clusters'],
    wspace=0.4,
    frameon='small',
    palette=overlay_palette,
)

from omicverse.single import pyVIA

v_raw = pyVIA(
    adata=adata_copy,
    adata_key='X_pca',
    adata_ncomps=100,
    basis='X_mde',
    clusters='clusters',
    knn=20,
    random_seed=4,
    root_user=['nIPC'],
    dataset='group',
)
v_raw.run()

v_interp = pyVIA(
    adata=interpolated,
    adata_key='X_pca',
    adata_ncomps=100,
    basis='X_mde',
    clusters='clusters',
    knn=15,
    random_seed=4,
    root_user=['Neuroblast'],
    dataset='group',
)
v_interp.run()

fig, ax = v_raw.plot_stream(basis='X_mde', clusters='clusters', density_grid=0.8)
fig, ax = v_interp.plot_stream(basis='X_mde', clusters='clusters', density_grid=0.8)

v_raw.get_pseudotime(adata_copy)
sc.pp.neighbors(adata_copy, n_neighbors=15, use_rep='X_pca')
ov.utils.cal_paga(adata_copy, use_time_prior='pt_via', vkey='paga', groups='clusters')
ov.utils.plot_paga(adata_copy, basis='mde', size=50, alpha=0.1, title='PAGA Raw', min_edge_width=2)

v_interp.get_pseudotime(interpolated)
sc.pp.neighbors(interpolated, n_neighbors=15, use_rep='X_pca')
ov.utils.cal_paga(interpolated, use_time_prior='pt_via', vkey='paga', groups='clusters')
ov.utils.plot_paga(interpolated, basis='mde', size=50, alpha=0.1, title='PAGA Interpolated', min_edge_width=2)
```
