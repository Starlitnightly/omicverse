# Nicheformer

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

Niche-aware spatial transformer, jointly models spatial coordinates and gene expression

!!! tip "When to choose Nicheformer"

    User has spatial transcriptomics data (Visium, MERFISH, Slide-seq) and wants niche-aware or spatial-context embeddings

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | Nicheformer |
| **Version** | v1.0 |
| **Tasks** | `embed`, `integrate`, `spatial` |
| **Modalities** | RNA, Spatial |
| **Species** | human, mouse |
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
info = ov.fm.describe_model("nicheformer")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "nicheformer", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="nicheformer",
    adata_path="your_data.h5ad",
    output_path="output_nicheformer.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_nicheformer.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard preprocessing. For spatial tasks, include spatial coordinates in `adata.obsm['spatial']`. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_nicheformer` | `adata.obsm` | Cell embeddings (512-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_nicheformer.h5ad")
embeddings = adata.obsm["X_nicheformer"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_nicheformer")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/theislab/nicheformer](https://github.com/theislab/nicheformer)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [Nicheformer Tutorial Notebook](t_fm_nicheformer.ipynb).
