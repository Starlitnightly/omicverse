# Quick Command Reference

## t_mofa.ipynb
```python
import omicverse as ov
rna = ov.utils.read('data/sample/rna_p_n_raw.h5ad')
atac = ov.utils.read('data/sample/atac_p_n_raw.h5ad')
test_mofa = ov.single.pyMOFA(omics=[rna, atac], omics_name=['RNA', 'ATAC'])
test_mofa.mofa_preprocess()
test_mofa.mofa_run(outfile='models/brac_rna_atac.hdf5')
rna = ov.utils.read('data/sample/rna_test.h5ad')
rna = ov.single.factor_exact(rna, hdf5_path='data/sample/MOFA_POS.hdf5')
ov.single.factor_correlation(adata=rna, cluster='cell_type', factor_list=[1,2,3,4,5])
pymofa = ov.single.pyMOFAART(model_path='data/sample/MOFA_POS.hdf5')
pymofa.plot_r2()
```

## t_mofa_glue.ipynb
```python
pair_obj = ov.single.GLUE_pair(rna, atac)
pair_obj.correlation()
test_mofa = ov.single.pyMOFA(omics=[rna1, atac1], omics_name=['RNA', 'ATAC'])
test_mofa.mofa_preprocess()
test_mofa.mofa_run(outfile='models/chen_rna_atac.hdf5')
pymofa = ov.single.pyMOFAART(model_path='models/chen_rna_atac.hdf5')
pymofa.get_factors(rna1)
pymofa.plot_cor(rna1, 'cell_type', figsize=(4,6))
```

## t_simba.ipynb
```python
import omicverse as ov
from omicverse.utils import mde
adata = ov.utils.read('simba_adata_raw.h5ad')
simba = ov.single.pySIMBA(adata, 'result_human_pancreas')
simba.preprocess(batch_key='batch', min_n_cells=3, method='lib_size', n_top_genes=3000, n_bins=5)
simba.gen_graph()
simba.train(num_workers=6)
simba.load('result_human_pancreas/pbg/graph0')
adata = simba.batch_correction()
adata.obsm['X_mde'] = mde(adata.obsm['X_simba'])
```

## t_tosica.ipynb
```python
ov.utils.download_tosica_gmt()
tosica = ov.single.pyTOSICA(
    adata=ref_adata,
    gmt_path='genesets/GO_bp.gmt',
    depth=1,
    label_name='Celltype',
    project_path='hGOBP_demo',
    batch_size=8,
)
tosica.train(epochs=5)
tosica.save()
tosica.load()
new_adata = tosica.predicted(pre_adata=query_adata)
```

## t_stavia.ipynb
```python
import scvelo as scv
adata = scv.datasets.dentategyrus()
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)
ov.pp.scale(adata)
ov.pp.pca(adata, layer='scaled', n_pcs=50)
ov.pp.neighbors(adata, use_rep='scaled|original|X_pca', n_neighbors=15, n_pcs=30)
ov.pp.umap(adata, min_dist=1)
ncomps, knn = 30, 15
v0 = VIA.core.VIA(
    data=adata.obsm['scaled|original|X_pca'][:, :ncomps],
    true_label=adata.obs['clusters'],
    edgepruning_clustering_resolution=0.15,
    cluster_graph_pruning=0.15,
    knn=knn,
    root_user=['nIPC'],
    resolution_parameter=1.5,
    dataset='',
    random_seed=4,
    memory=10,
)
v0.run_VIA()
adata.obs['pt_via'] = v0.single_cell_pt_markov
```
