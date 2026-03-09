# Quick Command Reference

## MOFA on paired scRNA + scATAC (t_mofa.ipynb)
```python
import omicverse as ov
import os

# Load each modality as a SEPARATE AnnData (not concatenated)
rna = ov.utils.read('data/sample/rna_p_n_raw.h5ad')
atac = ov.utils.read('data/sample/atac_p_n_raw.h5ad')

# Validate: omics list and omics_name must match in length
omics = [rna, atac]
omics_name = ['RNA', 'ATAC']
assert len(omics) == len(omics_name), "omics and omics_name must have same length"

test_mofa = ov.single.pyMOFA(omics=omics, omics_name=omics_name)
test_mofa.mofa_preprocess()  # Selects HVGs per modality

os.makedirs('models', exist_ok=True)  # Output dir must exist before mofa_run
test_mofa.mofa_run(outfile='models/brac_rna_atac.hdf5')

# Inspect factors downstream
rna = ov.utils.read('data/sample/rna_test.h5ad')
rna = ov.single.factor_exact(rna, hdf5_path='data/sample/MOFA_POS.hdf5')
ov.single.factor_correlation(adata=rna, cluster='cell_type', factor_list=[1, 2, 3, 4, 5])

pymofa = ov.single.pyMOFAART(model_path='data/sample/MOFA_POS.hdf5')
pymofa.plot_r2()  # Variance explained per factor
```

## GLUE pairing then MOFA (t_mofa_glue.ipynb)
```python
# Start from GLUE-derived embeddings (unpaired RNA + ATAC)
pair_obj = ov.single.GLUE_pair(rna, atac)
pair_obj.correlation()  # Aligns unpaired cells via shared embedding

# Then run MOFA as usual
test_mofa = ov.single.pyMOFA(omics=[rna1, atac1], omics_name=['RNA', 'ATAC'])
test_mofa.mofa_preprocess()
test_mofa.mofa_run(outfile='models/chen_rna_atac.hdf5')

pymofa = ov.single.pyMOFAART(model_path='models/chen_rna_atac.hdf5')
pymofa.get_factors(rna1)
pymofa.plot_cor(rna1, 'cell_type', figsize=(4, 6))  # Factor-cluster correlations
```

## SIMBA batch integration (t_simba.ipynb)
```python
import omicverse as ov
from omicverse.utils import mde

adata = ov.utils.read('simba_adata_raw.h5ad')

# Validate batch column exists
assert 'batch' in adata.obs.columns, "Need 'batch' column in adata.obs"

simba = ov.single.pySIMBA(adata, 'result_human_pancreas')
# MUST call preprocess() before gen_graph() — otherwise KeyError on binned features
simba.preprocess(batch_key='batch', min_n_cells=3, method='lib_size', n_top_genes=3000, n_bins=5)
simba.gen_graph()
simba.train(num_workers=6)  # CPU workers for PyTorch-BigGraph; increase for faster training
simba.load('result_human_pancreas/pbg/graph0')  # Load trained checkpoint
adata = simba.batch_correction()  # Returns AnnData with X_simba embedding
adata.obsm['X_mde'] = mde(adata.obsm['X_simba'])  # Optional: MDE for visualization
```

## TOSICA reference transfer (t_tosica.ipynb)
```python
import os

# Download GMT gene-set files first (writes to genesets/ in cwd)
ov.utils.download_tosica_gmt()
gmt_path = 'genesets/GO_bp.gmt'
assert os.path.isfile(gmt_path), f"GMT file not found at {gmt_path}"

tosica = ov.single.pyTOSICA(
    adata=ref_adata,
    gmt_path=gmt_path,           # Must be a FILE PATH, not a database name string
    depth=1,                      # depth=2 doubles memory usage
    label_name='Celltype',        # Column in ref_adata.obs with cell type labels
    project_path='hGOBP_demo',    # Directory for model checkpoints
    batch_size=8,
)
tosica.train(epochs=5)
tosica.save()
tosica.load()  # Reload from checkpoint
new_adata = tosica.predicted(pre_adata=query_adata)  # Transfer labels to query
```

## StaVIA trajectory (t_stavia.ipynb)
```python
import scvelo as scv
import pyVIA as VIA

adata = scv.datasets.dentategyrus()
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)
ov.pp.scale(adata)
ov.pp.pca(adata, layer='scaled', n_pcs=50)

# Verify PCA and neighbors exist before VIA
assert 'X_pca' in adata.obsm, "PCA required before VIA"
ov.pp.neighbors(adata, use_rep='scaled|original|X_pca', n_neighbors=15, n_pcs=30)
ov.pp.umap(adata, min_dist=1)

ncomps, knn = 30, 15
v0 = VIA.core.VIA(
    data=adata.obsm['scaled|original|X_pca'][:, :ncomps],
    true_label=adata.obs['clusters'],
    edgepruning_clustering_resolution=0.15,
    cluster_graph_pruning=0.15,
    knn=knn,
    root_user=['nIPC'],           # Must be a value in true_label
    resolution_parameter=1.5,
    dataset='',
    random_seed=4,
    memory=10,
)
v0.run_VIA()
adata.obs['pt_via'] = v0.single_cell_pt_markov  # Pseudotime
```
