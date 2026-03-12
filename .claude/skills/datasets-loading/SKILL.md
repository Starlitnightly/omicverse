---
name: datasets-loading
title: OmicVerse built-in datasets and mock data
description: "OmicVerse built-in datasets: pbmc3k, pancreas, dentategyrus, zebrafish, immune, spatial, multiome, plus create_mock_dataset() and predefined_signatures GMT gene sets."
---

# OmicVerse Built-in Datasets

`ov.datasets` provides 30+ ready-to-use datasets with automatic download, caching, and fallback to mock data. Use these instead of manually downloading files or relying on `scanpy.datasets`.

## When to Use This Module

- **Tutorials/demos**: Load standard benchmarks (PBMC3k, Paul15, dentate gyrus) with one function call
- **Testing pipelines**: Use `create_mock_dataset()` to generate synthetic data without downloads
- **Gene set analysis**: Use `predefined_signatures` for curated GMT gene sets (cell cycle, gender, mitochondrial, tissue-specific)
- **Velocity workflows**: Load pre-formatted datasets with spliced/unspliced layers

## Dataset Catalog

### Single-Cell

| Function | Cells | Genes | Description |
|----------|-------|-------|-------------|
| `ov.datasets.pbmc3k()` | 2,700 | 32,738 | 10x PBMC3k (raw or processed) |
| `ov.datasets.pbmc8k()` | ~8,000 | — | 10x PBMC 8k |
| `ov.datasets.paul15()` | 2,730 | 3,451 | Myeloid progenitors |
| `ov.datasets.krumsiek11()` | 640 | 11 | Myeloid differentiation simulation |
| `ov.datasets.bone_marrow()` | 5,780 | 27,876 | Bone marrow hematopoietic |
| `ov.datasets.hematopoiesis()` | — | — | Processed hematopoiesis |
| `ov.datasets.hematopoiesis_raw()` | — | — | Raw hematopoiesis |
| `ov.datasets.sc_ref_Lymph_Node()` | ~10,000 | ~15,000 | Lymph node reference |
| `ov.datasets.bhattacherjee()` | ~5,000 | ~2,000 | Mouse PFC cocaine study |
| `ov.datasets.human_tfs()` | — | — | Human TF list (DataFrame) |

### RNA Velocity & Trajectories

| Function | Cells | Genes | Description |
|----------|-------|-------|-------------|
| `ov.datasets.dentate_gyrus()` | 18,213 | 27,998 | Dentate gyrus (loom) |
| `ov.datasets.dentate_gyrus_scvelo()` | 2,930 | 13,913 | DG subset from scVelo |
| `ov.datasets.zebrafish()` | 4,181 | 16,940 | Zebrafish developmental |
| `ov.datasets.pancreatic_endocrinogenesis()` | — | — | Pancreatic epithelial |
| `ov.datasets.pancreas_cellrank()` | 2,930 | 13,913 | Pancreas cellrank benchmark |
| `ov.datasets.scnt_seq_neuron_splicing()` | 13,476 | 44,021 | scNT-seq neuron splicing |
| `ov.datasets.scnt_seq_neuron_labeling()` | 3,060 | 24,078 | scNT-seq neuron labeling |
| `ov.datasets.sceu_seq_rpe1()` | ~2,930 | ~13,913 | scEU-seq RPE1 |
| `ov.datasets.sceu_seq_organoid()` | 3,831 | 9,157 | scEU-seq organoid |
| `ov.datasets.haber()` | 7,216 | 27,998 | Intestinal epithelium |
| `ov.datasets.chromaffin()` | — | — | Chromaffin cell lineage |
| `ov.datasets.hg_forebrain_glutamatergic()` | 1,720 | 32,738 | Human forebrain |
| `ov.datasets.toggleswitch()` | 200 | 2 | Two-gene simulation |

### Spatial & Multiome

| Function | Description |
|----------|-------------|
| `ov.datasets.seqfish()` | SeqFISH spatial transcriptomics |
| `ov.datasets.multi_brain_5k()` | 10x E18 mouse brain multiome (MuData) |

### Bulk RNA-seq & Deconvolution

| Function | Description |
|----------|-------------|
| `ov.datasets.burczynski06()` | UC/CD PBMC bulk (127 samples) |
| `ov.datasets.moignard15()` | Embryo hematopoiesis qRT-PCR |
| `ov.datasets.decov_bulk_covid_bulk()` | COVID-19 PBMC bulk |
| `ov.datasets.decov_bulk_covid_single()` | COVID-19 PBMC single-cell ref |

### Synthetic

| Function | Description |
|----------|-------------|
| `ov.datasets.create_mock_dataset()` | Configurable synthetic scRNA-seq |
| `ov.datasets.blobs()` | Gaussian blob clusters |

## Mock Data Generation

Use `create_mock_dataset()` when you need data without network access or for pipeline testing:

```python
import omicverse as ov

# Basic mock dataset
adata = ov.datasets.create_mock_dataset(
    n_cells=2000,
    n_genes=1500,
    n_cell_types=6,
    with_clustering=False,
    random_state=42,
)
# adata.obs: cell_type, sample_id, condition, tissue
# adata.var: gene_symbols, highly_variable

# With full preprocessing (normalized, PCA, UMAP, leiden)
adata = ov.datasets.create_mock_dataset(
    n_cells=5000,
    n_genes=3000,
    n_cell_types=10,
    with_clustering=True,
)
```

**Features:**
- Negative binomial expression distribution
- Cell-type-specific marker genes (2-5x expression multiplier)
- Gene names: `Gene_0001`, `Gene_0002`, ...
- `with_clustering=True` adds: normalization, HVG, scaling, PCA, UMAP, leiden

## Predefined Gene Set Signatures

Pre-loaded GMT files for common scoring tasks:

```python
from omicverse.datasets import predefined_signatures, load_signatures_from_file

# Available signature keys
print(list(predefined_signatures.keys()))
# ['cell_cycle_human', 'cell_cycle_mouse', 'gender_human', 'gender_mouse',
#  'mitochondrial_genes_human', 'mitochondrial_genes_mouse',
#  'ribosomal_genes_human', 'ribosomal_genes_mouse',
#  'apoptosis_human', 'apoptosis_mouse',
#  'human_lung', 'mouse_lung', 'mouse_brain', 'mouse_liver', 'emt_human']

# Load a signature → dict[str, list[str]]
cell_cycle = load_signatures_from_file(predefined_signatures['cell_cycle_human'])
# {'S_genes': ['MCM5', 'PCNA', ...], 'G2M_genes': ['HMGB2', 'CDK1', ...]}

# Use with scoring
import scanpy as sc
sc.tl.score_genes_cell_cycle(adata, s_genes=cell_cycle['S_genes'],
                              g2m_genes=cell_cycle['G2M_genes'])
```

## Critical API Reference

```python
# CORRECT: use ov.datasets for standard benchmarks
adata = ov.datasets.pbmc3k()

# WRONG: manually downloading what's already built-in
# import urllib.request
# urllib.request.urlretrieve('https://...', 'pbmc3k.h5ad')  # unnecessary!
# adata = ov.read('pbmc3k.h5ad')

# CORRECT: pbmc3k(processed=True) for pre-processed version
adata = ov.datasets.pbmc3k(processed=True)

# WRONG: loading raw then manually preprocessing for a demo
# adata = ov.datasets.pbmc3k()
# sc.pp.normalize_total(adata)  # unnecessary if you just need a quick demo

# CORRECT: mock data for testing (no network needed)
adata = ov.datasets.create_mock_dataset(n_cells=500, n_genes=200)

# WRONG: creating synthetic data manually with numpy
# X = np.random.poisson(1, (500, 200))  # missing metadata, layers, etc.
```

## Caching Behavior

- **Default cache directory:** `./data/` (relative to working directory)
- **Skip if exists:** All functions check for existing files before downloading
- **Mirror fallback:** Stanford and Figshare mirrors for reliability
- **Mock fallback:** Most functions generate mock data if download fails (network issues)
- **`var_names_make_unique()`** called automatically after loading

## Troubleshooting

- **Download timeout / 403 error**: Some datasets use `download_data_requests()` with custom headers. If persistent, manually download the file to `./data/` with the expected filename and the function will find it.
- **`ModuleNotFoundError: No module named 'muon'`** when calling `multi_brain_5k()`: Install muon: `pip install muon`. This function returns MuData, not AnnData.
- **Mock dataset has no `.raw` or `layers['counts']`**: Add manually after creation: `ov.utils.store_layers(adata, layers='counts')` and `adata.raw = adata`.
- **`load_signatures_from_file` returns empty dict**: Verify the GMT file path. Use `predefined_signatures['key']` which resolves to the bundled file via `importlib.resources`.
- **Dentate gyrus loom download is slow**: The loom file is large (~200MB). Use `ov.datasets.dentate_gyrus_scvelo()` for the smaller pre-processed subset (2,930 cells).

## Dependencies
- Core: `omicverse`, `scanpy`, `anndata`, `numpy`, `pandas`
- Downloads: `tqdm`, `requests` (for mirror fallback)
- Multiome: `muon` (only for `multi_brain_5k()`)
- Signatures: `importlib.resources` (stdlib)

## Examples
- "Load the PBMC3k dataset and run the standard preprocessing pipeline."
- "Create a mock dataset with 5000 cells and 8 cell types for testing my clustering workflow."
- "Load cell cycle gene signatures and score my adata for S and G2M phase genes."

## References
- Quick copy/paste commands: [`reference.md`](reference.md)
