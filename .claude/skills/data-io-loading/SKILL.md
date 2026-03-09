---
name: data-io-loading
title: OmicVerse data I/O
description: "OmicVerse data I/O: use ov.read(), ov.io.read_h5ad, read_10x_h5, read_10x_mtx, read_visium, read_visium_hd, read_nanostring instead of scanpy. Covers h5ad, 10x, spatial, CSV formats."
---

# OmicVerse Data I/O

OmicVerse provides its own data readers under `ov.io`. These replace scanpy's IO functions with better format handling, spatial geometry support, and Rust backend options. When working in an OmicVerse project, always use `ov.io.*` for data loading — never fall back to `sc.read_*` or `scanpy.read_*`.

## Why this matters

OmicVerse's readers are not thin wrappers — they are independent implementations that handle edge cases scanpy misses:
- **10x H5/MTX**: Proper v2/v3 format detection, flexible prefix/compression options
- **Visium**: Auto-resolves tissue positions (parquet > csv > legacy csv), loads images + scale factors
- **Visium HD**: Cell segmentation with GeoJSON→WKT polygon conversion (not available in scanpy at all)
- **Nanostring SMI**: Auto-detects column names across CosMx format variants (not in scanpy)

## Migration table: scanpy → OmicVerse

| Task | DON'T use | Use instead |
|------|-----------|-------------|
| Read any file | `sc.read(path)` | `ov.read(path)` |
| Read h5ad | `sc.read_h5ad(f)` | `ov.read(f)` or `ov.io.read_h5ad(f)` |
| Read 10x H5 | `sc.read_10x_h5(f)` | `ov.io.read_10x_h5(f)` |
| Read 10x MTX dir | `sc.read_10x_mtx(d)` | `ov.io.read_10x_mtx(d)` |
| Read Visium | `sc.read_visium(d)` | `ov.io.spatial.read_visium(d)` |
| Read Visium HD | *(not available)* | `ov.io.read_visium_hd(d)` |
| Read Nanostring | *(not available)* | `ov.io.read_nanostring(d, counts, meta)` |
| Read CSV/TSV | `pd.read_csv(f)` | `ov.read(f)` or `ov.io.read_csv(f)` |
| Save Python object | `pickle.dump(...)` | `ov.io.save(obj, path)` |
| Load Python object | `pickle.load(...)` | `ov.io.load(path)` |

## Access paths

```
ov.read(path)                          # Top-level universal reader (lazy attr)
ov.io.read_h5ad(filename)              # h5ad
ov.io.read_10x_h5(filename)            # 10x Genomics H5
ov.io.read_10x_mtx(path)              # 10x Matrix Market directory
ov.io.spatial.read_visium(path)        # Visium (standard Space Ranger)
ov.io.read_visium_hd(path)            # Visium HD (auto-detect bin vs seg)
ov.io.read_visium_hd_bin(path)        # Visium HD bin-level
ov.io.read_visium_hd_seg(path)        # Visium HD cell segmentation
ov.io.read_nanostring(path, ...)      # Nanostring SMI / CosMx
ov.io.read_csv(**kwargs)              # CSV/TSV wrapper
ov.io.save(obj, path)                 # Pickle serialization
ov.io.load(path)                      # Pickle deserialization
```

Note: `read_visium` (standard) is under `ov.io.spatial`, not directly under `ov.io`. All other readers are at `ov.io` level.

## Universal reader: `ov.read(path, backend='python')`

Auto-detects format by file extension and returns the appropriate object:

| Extension | Returns | Backend |
|-----------|---------|---------|
| `.h5ad` | `AnnData` | Python (anndata) or Rust (snapatac2) |
| `.csv` | `DataFrame` | pandas |
| `.tsv`, `.txt` | `DataFrame` | pandas (tab-separated) |
| `.csv.gz`, `.tsv.gz`, `.txt.gz` | `DataFrame` | pandas (gzip) |

```python
import omicverse as ov

# h5ad → AnnData
adata = ov.read('pbmc3k.h5ad')

# CSV → DataFrame
df = ov.read('counts.csv')

# Gzipped TSV → DataFrame
df = ov.read('metadata.tsv.gz')

# Rust backend for large h5ad files (requires snapatac2)
adata = ov.read('large_dataset.h5ad', backend='rust')
# Remember: call adata.close() when done with Rust backend
```

## Single-cell readers

### `ov.io.read_h5ad(filename, **kwargs)`

Direct h5ad reader. All kwargs forwarded to `anndata.read_h5ad()`.

```python
adata = ov.io.read_h5ad('sample.h5ad')
adata = ov.io.read_h5ad('large.h5ad', backed='r')  # Backed mode for large files
```

### `ov.io.read_10x_h5(filename, *, genome=None, gex_only=True)`

Read 10x Genomics HDF5 count matrices. Handles both legacy (v2) and v3+ formats automatically.

```python
adata = ov.io.read_10x_h5('filtered_feature_bc_matrix.h5')

# Multi-genome file: filter by genome
adata = ov.io.read_10x_h5('raw_feature_bc_matrix.h5', genome='GRCh38')

# Keep all feature types (Gene Expression + Antibody Capture + CRISPR Guide)
adata = ov.io.read_10x_h5('filtered_feature_bc_matrix.h5', gex_only=False)
```

### `ov.io.read_10x_mtx(path, *, var_names='gene_symbols', make_unique=True, gex_only=True, prefix=None, compressed=True)`

Read 10x Matrix Market directory (contains `matrix.mtx`, `features.tsv`/`genes.tsv`, `barcodes.tsv`).

```python
adata = ov.io.read_10x_mtx('filtered_feature_bc_matrix/')

# Use Ensembl gene IDs instead of symbols
adata = ov.io.read_10x_mtx('filtered_feature_bc_matrix/', var_names='gene_ids')

# STARsolo output (uncompressed files)
adata = ov.io.read_10x_mtx('Solo.out/Gene/filtered/', compressed=False)
```

## Spatial readers

### `ov.io.spatial.read_visium(path, *, count_file='filtered_feature_bc_matrix.h5', library_id=None, load_images=True, ...)`

Read standard 10x Visium Space Ranger output. Loads count matrix, tissue positions, images, and scale factors.

```python
adata = ov.io.spatial.read_visium('spaceranger_output/outs/')

# Use raw counts
adata = ov.io.spatial.read_visium('outs/', count_file='raw_feature_bc_matrix.h5')

# Skip image loading (faster, less memory)
adata = ov.io.spatial.read_visium('outs/', load_images=False)
```

Output structure:
- `adata.obsm['spatial']` — spot pixel coordinates
- `adata.uns['spatial'][library_id]['images']` — hires/lowres images
- `adata.uns['spatial'][library_id]['scalefactors']` — scale factors
- `adata.obs['in_tissue']`, `array_row`, `array_col` — tissue position metadata

### `ov.io.read_visium_hd(path, ...)` / `read_visium_hd_bin` / `read_visium_hd_seg`

Read Visium HD data. The unified `read_visium_hd` auto-detects bin vs segmentation format.

```python
# Auto-detect
adata = ov.io.read_visium_hd('spaceranger_hd_output/outs/')

# Explicit bin-level (specify bin size)
adata = ov.io.read_visium_hd_bin('outs/binned_outputs/square_016um/', binsize=16)

# Cell segmentation (includes GeoJSON polygon geometry)
adata = ov.io.read_visium_hd_seg('outs/segmented_outputs/')
# adata.obs['geometry'] contains WKT polygon strings
```

### `ov.io.read_nanostring(path, counts_file, meta_file, fov_file=None)`

Read Nanostring Spatial Molecular Imager (CosMx) data.

```python
adata = ov.io.read_nanostring(
    path='cosmx_output/',
    counts_file='exprMat_file.csv',
    meta_file='metadata_file.csv',
    fov_file='fov_positions_file.csv',  # optional
)
# adata.obsm['spatial'] — cell center coordinates
# adata.obs['geometry'] — cell polygon WKT strings
```

## Serialization

```python
# Save any Python object (uses cloudpickle with pickle fallback)
ov.io.save(my_model, 'model.pkl')

# Load it back
my_model = ov.io.load('model.pkl')
```

## Defensive validation

```python
from pathlib import Path

# Before reading: verify file exists
path = Path('data.h5ad')
assert path.exists(), f"File not found: {path}"

# Before read_10x_mtx: verify directory structure
mtx_dir = Path('filtered_feature_bc_matrix/')
assert (mtx_dir / 'matrix.mtx.gz').exists() or (mtx_dir / 'matrix.mtx').exists(), \
    f"No matrix.mtx found in {mtx_dir}"

# Before read_visium: verify Space Ranger output
outs_dir = Path('outs/')
assert (outs_dir / 'filtered_feature_bc_matrix.h5').exists(), \
    f"No count matrix in {outs_dir}. Is this a Space Ranger output directory?"
assert (outs_dir / 'spatial').is_dir(), \
    f"No spatial/ directory in {outs_dir}"
```

## Troubleshooting

- **`FileNotFoundError` from `read_10x_h5`**: Verify the `.h5` file path is correct. Cell Ranger output is typically at `outs/filtered_feature_bc_matrix.h5`.
- **`ValueError: The type is not supported` from `ov.read()`**: The file extension is not recognized. Use format-specific readers (`read_10x_h5`, `read_10x_mtx`) for non-standard extensions.
- **`ImportError: snapatac2` from `ov.read(..., backend='rust')`**: Install with `pip install snapatac2`. The Rust backend is optional.
- **Duplicate gene names warning**: `read_10x_mtx` with `var_names='gene_symbols'` auto-deduplicates by default (`make_unique=True`). If you need original names, set `make_unique=False`.
- **`read_visium` missing tissue positions**: The reader auto-detects `.parquet`, `.csv`, and legacy `.csv` formats. If using a custom directory layout, verify the `spatial/` subdirectory contains a tissue positions file.
- **Visium HD segmentation missing polygons**: Requires `geopandas` and `shapely`. Install with `pip install geopandas shapely`.
- **Large h5ad OOM**: Use backed mode `ov.io.read_h5ad('large.h5ad', backed='r')` or Rust backend `ov.read('large.h5ad', backend='rust')`.

## Quick copy-paste commands

See [`reference.md`](reference.md) for complete code blocks organized by format.
