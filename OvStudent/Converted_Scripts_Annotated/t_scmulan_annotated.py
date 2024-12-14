```python
# Line 1: import os -- import os
# Line 3: import scanpy as sc -- import scanpy as sc
# Line 4: import omicverse as ov -- import omicverse as ov
# Line 5: ov.plot_set() -- ov.plot_set()
# Line 8: adata = sc.read('./data/liver_test.h5ad') -- adata = sc.read('./data/liver_test.h5ad')
# Line 10: adata -- adata
# Line 13: from scipy.sparse import csc_matrix -- from scipy.sparse import csc_matrix
# Line 14: adata.X = csc_matrix(adata.X) -- adata.X = csc_matrix(adata.X)
# Line 16: adata_GS_uniformed = ov.externel.scMulan.GeneSymbolUniform(input_adata=adata, -- adata_GS_uniformed = ov.externel.scMulan.GeneSymbolUniform(input_adata=adata,
# Line 17:                                  output_dir="./data", --                                  output_dir="./data",
# Line 18:                                  output_prefix='liver') --                                  output_prefix='liver')
# Line 22: adata_GS_uniformed=sc.read_h5ad('./data/liver_uniformed.h5ad') -- adata_GS_uniformed=sc.read_h5ad('./data/liver_uniformed.h5ad')
# Line 24: adata_GS_uniformed -- adata_GS_uniformed
# Line 28: if adata_GS_uniformed.X.max() > 10: -- if adata_GS_uniformed.X.max() > 10:
# Line 29:     sc.pp.normalize_total(adata_GS_uniformed, target_sum=1e4) --     sc.pp.normalize_total(adata_GS_uniformed, target_sum=1e4)
# Line 30:     sc.pp.log1p(adata_GS_uniformed) --     sc.pp.log1p(adata_GS_uniformed)
# Line 35: ckp_path = './ckpt/ckpt_scMulan.pt' -- ckp_path = './ckpt/ckpt_scMulan.pt'
# Line 37: scml = ov.externel.scMulan.model_inference(ckp_path, adata_GS_uniformed) -- scml = ov.externel.scMulan.model_inference(ckp_path, adata_GS_uniformed)
# Line 38: base_process = scml.cuda_count() -- base_process = scml.cuda_count()
# Line 40: scml.get_cell_types_and_embds_for_adata(parallel=True, n_process = 1) -- scml.get_cell_types_and_embds_for_adata(parallel=True, n_process = 1)
# Line 43: adata_mulan = scml.adata.copy() -- adata_mulan = scml.adata.copy()
# Line 46: ov.pp.scale(adata_mulan) -- ov.pp.scale(adata_mulan)
# Line 47: ov.pp.pca(adata_mulan) -- ov.pp.pca(adata_mulan)
# Line 50: ov.pp.mde(adata_mulan,embedding_dim=2,n_neighbors=15, basis='X_mde', -- ov.pp.mde(adata_mulan,embedding_dim=2,n_neighbors=15, basis='X_mde',
# Line 51:           n_pcs=10, use_rep='scaled|original|X_pca',) --           n_pcs=10, use_rep='scaled|original|X_pca',)
# Line 54: ov.pl.embedding(adata_mulan,basis='X_mde', -- ov.pl.embedding(adata_mulan,basis='X_mde',
# Line 55:                 color=["cell_type_from_scMulan",], --                 color=["cell_type_from_scMulan",],
# Line 56:                 ncols=1,frameon='small') --                 ncols=1,frameon='small')
# Line 58: adata_mulan.obsm['X_umap']=adata_mulan.obsm['X_mde'] -- adata_mulan.obsm['X_umap']=adata_mulan.obsm['X_mde']
# Line 61: ov.externel.scMulan.cell_type_smoothing(adata_mulan, threshold=0.1) -- ov.externel.scMulan.cell_type_smoothing(adata_mulan, threshold=0.1)
# Line 65: ov.pl.embedding(adata_mulan,basis='X_mde', -- ov.pl.embedding(adata_mulan,basis='X_mde',
# Line 66:                 color=["cell_type_from_mulan_smoothing","cell_type"], --                 color=["cell_type_from_mulan_smoothing","cell_type"],
# Line 67:                 ncols=1,frameon='small') --                 ncols=1,frameon='small')
# Line 69: adata_mulan -- adata_mulan
# Line 71: top_celltypes = adata_mulan.obs.cell_type_from_scMulan.value_counts().index[:20] -- top_celltypes = adata_mulan.obs.cell_type_from_scMulan.value_counts().index[:20]
# Line 74: selected_cell_types = top_celltypes -- selected_cell_types = top_celltypes
# Line 75: ov.externel.scMulan.visualize_selected_cell_types(adata_mulan,selected_cell_types,smoothing=True) -- ov.externel.scMulan.visualize_selected_cell_types(adata_mulan,selected_cell_types,smoothing=True)
```