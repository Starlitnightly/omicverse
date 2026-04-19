# scMulan

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

Native multi-omics joint modeling (RNA+ATAC+Protein simultaneously), designed for CITE-seq/10x Multiome

!!! tip "When to choose scMulan"

    User has multi-omics data (CITE-seq, 10x Multiome, RNA+ATAC+Protein), or wants joint multi-modal embedding

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | scMulan |
| **Version** | v1.0 |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA, ATAC, Protein, Multi-omics |
| **Species** | human |
| **Gene IDs** | symbol |
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
info = ov.fm.describe_model("scmulan")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "scmulan", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="scmulan",
    adata_path="your_data.h5ad",
    output_path="output_scmulan.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_scmulan.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | For multi-omics, organize data as MuData with separate modalities. For RNA-only, standard preprocessing. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_scmulan` | `adata.obsm` | Cell embeddings (512-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_scmulan.h5ad")
embeddings = adata.obsm["X_scmulan"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_scmulan")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/SuperBianC/scMulan](https://github.com/SuperBianC/scMulan)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [scMulan Tutorial Notebook](t_fm_scmulan.ipynb).
