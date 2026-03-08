# ATACformer

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

ATAC-seq-native transformer, peak-based (not gene-based) input, chromatin accessibility specialist

!!! tip "When to choose ATACformer"

    User has ATAC-seq data, chromatin accessibility profiles, or peak-based (not gene expression) inputs

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | ATACformer |
| **Version** | v1.0 |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | ATAC |
| **Species** | human |
| **Gene IDs** | custom (peak-based) |
| **Embedding Dim** | 512 |
| **GPU Required** | Yes |
| **Min VRAM** | 16 GB |
| **Recommended VRAM** | 32 GB |
| **CPU Fallback** | No |
| **Adapter Status** | ⚠️ partial |

---

## Quick Start

```python
import omicverse as ov

# 1. Check model spec
info = ov.fm.describe_model("atacformer")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "atacformer", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="atacformer",
    adata_path="your_data.h5ad",
    output_path="output_atacformer.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_atacformer.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | custom (peak-based) |
| **Preprocessing** | Input must be ATAC-seq peak matrix (not gene expression). Follow standard scATAC-seq preprocessing (LSI/TF-IDF). |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_atacformer` | `adata.obsm` | Cell embeddings (512-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_atacformer.h5ad")
embeddings = adata.obsm["X_atacformer"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_atacformer")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/Atacformer/Atacformer](https://github.com/Atacformer/Atacformer)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [ATACformer Tutorial Notebook](t_fm_atacformer.ipynb).
