# Quick commands: Datasets loading

## Single-cell datasets

```python
import omicverse as ov

# PBMC3k — raw
adata = ov.datasets.pbmc3k()

# PBMC3k — processed (with clustering)
adata = ov.datasets.pbmc3k(processed=True)

# PBMC 8k
adata = ov.datasets.pbmc8k()

# Paul15 myeloid progenitors
adata = ov.datasets.paul15()

# Krumsiek11 myeloid simulation
adata = ov.datasets.krumsiek11()

# Bone marrow
adata = ov.datasets.bone_marrow()

# Hematopoiesis (processed)
adata = ov.datasets.hematopoiesis()

# Hematopoiesis (raw)
adata = ov.datasets.hematopoiesis_raw()

# Lymph node reference
adata = ov.datasets.sc_ref_Lymph_Node()

# Bhattacherjee mouse PFC
adata = ov.datasets.bhattacherjee()

# Human transcription factor list
df_tfs = ov.datasets.human_tfs()
```

## RNA velocity and trajectory datasets

```python
# Dentate gyrus (full loom, ~200MB)
adata = ov.datasets.dentate_gyrus()

# Dentate gyrus (scVelo subset, smaller)
adata = ov.datasets.dentate_gyrus_scvelo()

# Zebrafish developmental
adata = ov.datasets.zebrafish()

# Pancreatic endocrinogenesis
adata = ov.datasets.pancreatic_endocrinogenesis()

# Pancreas cellrank benchmark
adata = ov.datasets.pancreas_cellrank()

# scNT-seq neuron splicing
adata = ov.datasets.scnt_seq_neuron_splicing()

# scNT-seq neuron labeling
adata = ov.datasets.scnt_seq_neuron_labeling()

# scEU-seq RPE1
adata = ov.datasets.sceu_seq_rpe1()

# scEU-seq organoid
adata = ov.datasets.sceu_seq_organoid()

# Haber intestinal epithelium
adata = ov.datasets.haber()

# Human forebrain glutamatergic
adata = ov.datasets.hg_forebrain_glutamatergic()

# Chromaffin cells
adata = ov.datasets.chromaffin()

# Toggle switch (2-gene simulation)
adata = ov.datasets.toggleswitch()
```

## Spatial and multiome datasets

```python
# SeqFISH spatial
adata = ov.datasets.seqfish()

# 10x E18 mouse brain multiome (returns MuData)
import muon as mu
mdata = ov.datasets.multi_brain_5k()
```

## Bulk RNA-seq and deconvolution

```python
# Burczynski06 UC/CD bulk (127 samples, 22283 genes)
adata = ov.datasets.burczynski06()

# Moignard15 embryo hematopoiesis qRT-PCR
adata = ov.datasets.moignard15()

# COVID-19 bulk
adata_bulk = ov.datasets.decov_bulk_covid_bulk()

# COVID-19 single-cell reference
adata_sc = ov.datasets.decov_bulk_covid_single()
```

## Synthetic data generation

```python
# Basic mock dataset
adata = ov.datasets.create_mock_dataset(
    n_cells=2000,
    n_genes=1500,
    n_cell_types=6,
    with_clustering=False,
    random_state=42,
)

# Mock with full preprocessing (PCA, UMAP, leiden)
adata = ov.datasets.create_mock_dataset(
    n_cells=5000,
    n_genes=3000,
    n_cell_types=10,
    with_clustering=True,
)

# Gaussian blobs for clustering benchmarks
adata = ov.datasets.blobs(
    n_variables=11,
    n_centers=5,
    cluster_std=1.0,
    n_observations=640,
)
```

## Gene set signatures

```python
from omicverse.datasets import predefined_signatures, load_signatures_from_file

# List available signatures
print(list(predefined_signatures.keys()))
# cell_cycle_human, cell_cycle_mouse, gender_human, gender_mouse,
# mitochondrial_genes_human, mitochondrial_genes_mouse,
# ribosomal_genes_human, ribosomal_genes_mouse,
# apoptosis_human, apoptosis_mouse,
# human_lung, mouse_lung, mouse_brain, mouse_liver, emt_human

# Load cell cycle genes
cc = load_signatures_from_file(predefined_signatures['cell_cycle_human'])
print(cc.keys())  # dict_keys(['S_genes', 'G2M_genes'])

# Score cell cycle
import scanpy as sc
sc.tl.score_genes_cell_cycle(
    adata,
    s_genes=cc['S_genes'],
    g2m_genes=cc['G2M_genes'],
)

# Load tissue-specific signatures
lung_sigs = load_signatures_from_file(predefined_signatures['human_lung'])
brain_sigs = load_signatures_from_file(predefined_signatures['mouse_brain'])

# Load custom GMT file
custom_sigs = load_signatures_from_file('path/to/custom_genesets.gmt')
```

## Caching and download utilities

```python
from omicverse.datasets import download_data, download_data_requests, get_adata

# Download any file
local_path = download_data(
    url='https://example.com/data.h5ad',
    file_path='my_data.h5ad',
    dir='./data',
)

# Download with custom headers (reduces 403 errors)
local_path = download_data_requests(
    url='https://example.com/data.h5ad',
    file_path='my_data.h5ad',
    dir='./data',
)

# Download and load as AnnData directly
adata = get_adata(
    url='https://example.com/data.h5ad',
    filename='data.h5ad',
)
```

## Key function signatures

```python
# Dataset loaders (common pattern)
ov.datasets.pbmc3k(processed=False)                    # → AnnData
ov.datasets.paul15(url=..., filename='paul15.h5')       # → AnnData
ov.datasets.dentate_gyrus(url=..., filename=None)       # → AnnData
ov.datasets.dentate_gyrus_scvelo(filename=...)          # → AnnData
ov.datasets.zebrafish(filename=...)                      # → AnnData
ov.datasets.bone_marrow(filename=...)                    # → AnnData
ov.datasets.hematopoiesis(filename=...)                  # → AnnData
ov.datasets.multi_brain_5k()                             # → MuData | None

# Mock data
ov.datasets.create_mock_dataset(
    n_cells=2000, n_genes=1500, n_cell_types=6,
    with_clustering=False, random_state=42,
)  # → AnnData

ov.datasets.blobs(
    n_variables=11, n_centers=5, cluster_std=1.0,
    n_observations=640, random_state=0,
)  # → AnnData

# Signatures
load_signatures_from_file(input_file)  # → Dict[str, List[str]]

# Download utilities
download_data(url, file_path=None, dir='./data')        # → str (path)
download_data_requests(url, file_path=None, dir='./data') # → str (path)
get_adata(url, filename=None)                             # → AnnData | None
```