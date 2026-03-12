# OmicVerse I/O quick commands

## Universal reader

```python
import omicverse as ov

# Auto-detect format by extension
adata = ov.read('data.h5ad')           # → AnnData
df = ov.read('counts.csv')             # → DataFrame (comma-sep)
df = ov.read('metadata.tsv')           # → DataFrame (tab-sep)
df = ov.read('data.csv.gz')            # → DataFrame (gzipped)

# Rust backend for large h5ad (requires snapatac2)
adata = ov.read('large.h5ad', backend='rust')
# adata.close() when done
```

## h5ad files

```python
import omicverse as ov

# Standard read
adata = ov.io.read_h5ad('sample.h5ad')

# Backed mode (memory-efficient for large files)
adata = ov.io.read_h5ad('large.h5ad', backed='r')

# Write h5ad (use anndata directly)
adata.write_h5ad('output.h5ad')
```

## 10x Genomics H5

```python
import omicverse as ov

# Basic read (gene expression only, auto-detects v2/v3)
adata = ov.io.read_10x_h5('filtered_feature_bc_matrix.h5')

# Keep all feature types (GEX + ADT + CRISPR)
adata = ov.io.read_10x_h5('filtered_feature_bc_matrix.h5', gex_only=False)

# Multi-genome: select specific genome
adata = ov.io.read_10x_h5('raw_feature_bc_matrix.h5', genome='GRCh38')
```

## 10x Matrix Market directory

```python
import omicverse as ov

# Standard Cell Ranger output
adata = ov.io.read_10x_mtx('filtered_feature_bc_matrix/')

# Use Ensembl IDs instead of gene symbols
adata = ov.io.read_10x_mtx('filtered_feature_bc_matrix/', var_names='gene_ids')

# STARsolo uncompressed output
adata = ov.io.read_10x_mtx('Solo.out/Gene/filtered/', compressed=False)

# Custom prefix (e.g., 'sample1_')
adata = ov.io.read_10x_mtx('data/', prefix='sample1_')
```

## Visium (standard Space Ranger)

```python
import omicverse as ov

# Standard read from Space Ranger outs/
adata = ov.io.spatial.read_visium('spaceranger/outs/')

# Use raw counts instead of filtered
adata = ov.io.spatial.read_visium(
    'spaceranger/outs/',
    count_file='raw_feature_bc_matrix.h5',
)

# Custom library ID
adata = ov.io.spatial.read_visium(
    'spaceranger/outs/',
    library_id='sample_A',
)

# Skip image loading (faster)
adata = ov.io.spatial.read_visium('spaceranger/outs/', load_images=False)

# Access spatial data after loading:
# adata.obsm['spatial']                             — pixel coordinates
# adata.uns['spatial'][lib_id]['images']['hires']    — high-res image
# adata.uns['spatial'][lib_id]['scalefactors']       — scale factors
# adata.obs['in_tissue']                             — tissue mask
```

## Visium HD

```python
import omicverse as ov

# Auto-detect bin vs segmentation
adata = ov.io.read_visium_hd('spaceranger_hd/outs/')

# Bin-level with specific bin size
adata = ov.io.read_visium_hd_bin(
    'spaceranger_hd/outs/binned_outputs/square_016um/',
    binsize=16,
)

# Cell segmentation (requires geopandas + shapely)
adata = ov.io.read_visium_hd_seg(
    'spaceranger_hd/outs/segmented_outputs/',
)
# adata.obs['geometry'] — WKT polygon strings per cell
```

## Nanostring SMI (CosMx)

```python
import omicverse as ov

adata = ov.io.read_nanostring(
    path='cosmx_output/',
    counts_file='exprMat_file.csv',
    meta_file='metadata_file.csv',
    fov_file='fov_positions_file.csv',  # optional
)
# adata.obsm['spatial']      — cell center coordinates (local px)
# adata.obsm['spatial_fov']  — global FOV coordinates (if fov_file provided)
# adata.obs['geometry']      — WKT polygon strings
```

## CSV / TSV

```python
import omicverse as ov

# ov.io.read_csv wraps pandas.read_csv
df = ov.io.read_csv('expression.csv', index_col=0)
df = ov.io.read_csv('metadata.tsv', sep='\t')

# Or use ov.read() for auto-detection
df = ov.read('expression.csv')      # auto comma
df = ov.read('metadata.tsv')        # auto tab
```

## Object serialization

```python
import omicverse as ov

# Save any Python object (cloudpickle → pickle fallback)
ov.io.save(model_obj, 'my_model.pkl')
ov.io.save(gene_list, 'markers.pkl')

# Load back
model_obj = ov.io.load('my_model.pkl')
gene_list = ov.io.load('markers.pkl')
```

## Batch loading pattern

```python
import omicverse as ov
from pathlib import Path

# Load multiple h5ad files
data_dir = Path('samples/')
adatas = {
    p.stem: ov.read(str(p))
    for p in sorted(data_dir.glob('*.h5ad'))
}

# Load multiple 10x directories
sample_dirs = ['sample1/', 'sample2/', 'sample3/']
adatas = {
    d: ov.io.read_10x_mtx(d)
    for d in sample_dirs
}
```

## Format conversion

```python
import omicverse as ov

# CSV → h5ad
df = ov.read('counts.csv', index_col=0)
import anndata
adata = anndata.AnnData(df)
adata.write_h5ad('counts.h5ad')

# 10x MTX → h5ad
adata = ov.io.read_10x_mtx('filtered_feature_bc_matrix/')
adata.write_h5ad('sample.h5ad')
```
