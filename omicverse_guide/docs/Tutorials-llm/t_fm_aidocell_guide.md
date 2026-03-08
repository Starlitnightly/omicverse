# AIDO.Cell

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

Dense transformer optimized for unsupervised cell clustering without predefined labels

!!! tip "When to choose AIDO.Cell"

    User wants unsupervised clustering, label-free cell grouping, or dense transformer embeddings for discovery

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | AIDO.Cell |
| **Version** | v1.0 |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA |
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
info = ov.fm.describe_model("aidocell")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "aidocell", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="aidocell",
    adata_path="your_data.h5ad",
    output_path="output_aidocell.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_aidocell.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard preprocessing. Model optimizes for unsupervised cluster separation. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_aidocell` | `adata.obsm` | Cell embeddings (512-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_aidocell.h5ad")
embeddings = adata.obsm["X_aidocell"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_aidocell")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/genbio-ai/AIDO](https://github.com/genbio-ai/AIDO)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [AIDO.Cell Tutorial Notebook](t_fm_aidocell.ipynb).
