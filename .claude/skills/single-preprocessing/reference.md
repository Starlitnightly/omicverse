# Quick commands: Single-cell preprocessing

## Environment setup
```bash
# Base CPU workflow
pip install omicverse scanpy matplotlib

# Mixed CPU–GPU (OmicVerse >=1.7.0)
conda create -n ov-cpugpu python=3.11 -y
conda activate ov-cpugpu
pip install omicverse[full] rapids-singlecell

# Pure GPU stack (RAPIDS 24.04 example)
conda create -n rapids python=3.11 -y
conda install -n rapids rapids=24.04 -c rapidsai -c conda-forge -c nvidia -y
conda install -n rapids cudf=24.04 cuml=24.04 cugraph=24.04 cuxfilter=24.04 cucim=24.04 pylibraft=24.04 raft-dask=24.04 cuvs=24.04 -c rapidsai -c conda-forge -c nvidia -y
conda activate rapids
pip install rapids-singlecell
curl -sSL https://raw.githubusercontent.com/Starlitnightly/omicverse/refs/heads/master/install.sh | bash -s
```

## Data download
```bash
mkdir -p data write
wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz
cd data
tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz
cd ..
```

## Notebook boilerplate
```python
import scanpy as sc
import omicverse as ov
ov.plot_set(font_path='Arial')
%load_ext autoreload
%autoreload 2

adata = sc.read_10x_mtx(
    'data/filtered_gene_bc_matrices/hg19/',
    var_names='gene_symbols',
    cache=True,
)
ov.utils.store_layers(adata, layers='counts')
```

## QC and normalisation
```python
adata = ov.pp.qc(
    adata,
    tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250},
    doublets_method='scrublet',
)
adata = ov.pp.preprocess(
    adata,
    mode='shiftlog|pearson',
    n_HVGs=2000,
    target_sum=5e5,
)
adata.raw = adata
```

## Optional mixed CPU–GPU extras
```python
X_counts_recovered, size_factors = ov.pp.recover_counts(
    adata.X,
    5e5,
    5e6,
    log_base=None,
    chunk_size=10000,
)
adata.layers['recover_counts'] = X_counts_recovered
adata.uns['size_factors'] = size_factors
```

## GPU-only helpers
```python
ov.pp.anndata_to_GPU(adata)
ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)
ov.pp.neighbors(
    adata,
    n_neighbors=15,
    n_pcs=50,
    use_rep='scaled|original|X_pca',
    method='cagra',
)
ov.pp.anndata_to_CPU(adata)
```

## Dimensionality reduction and clustering
```python
ov.pp.scale(adata)
ov.pp.pca(adata, layer='scaled', n_pcs=50)
ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50, use_rep='scaled|original|X_pca')
ov.pp.umap(adata)
ov.pp.mde(adata, embedding_dim=2, n_neighbors=15, use_rep='scaled|original|X_pca')
ov.pp.leiden(adata, resolution=1)
```

## Visualisation and export
```python
ov.pl.embedding(adata, basis='X_mde', color=['leiden', 'CST3'], frameon='small')
adata.write('write/pbmc3k_preprocessed.h5ad')
```
