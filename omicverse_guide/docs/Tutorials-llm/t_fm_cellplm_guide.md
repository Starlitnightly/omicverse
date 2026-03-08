# CellPLM

✅ **Status:** ready | **Version:** v1.0

---

## Overview

Cell-centric (not gene-centric) architecture, highest batch throughput (batch_size=128), fast inference

!!! tip "When to choose CellPLM"

    User needs fast inference, high throughput, million-cell scale processing, or cell-level (not gene-level) modeling

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | CellPLM |
| **Version** | v1.0 |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA |
| **Species** | human |
| **Gene IDs** | symbol |
| **Embedding Dim** | 512 |
| **GPU Required** | Yes |
| **Min VRAM** | 8 GB |
| **Recommended VRAM** | 16 GB |
| **CPU Fallback** | Yes |
| **Adapter Status** | ✅ ready |

---

## Quick Start

```python
import omicverse as ov

# 1. Check model spec
info = ov.fm.describe_model("cellplm")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "cellplm", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="cellplm",
    adata_path="your_data.h5ad",
    output_path="output_cellplm.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_cellplm.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard preprocessing. Model handles tokenization internally. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_cellplm` | `adata.obsm` | Cell embeddings (512-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_cellplm.h5ad")
embeddings = adata.obsm["X_cellplm"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_cellplm")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/OmicsML/CellPLM](https://github.com/OmicsML/CellPLM)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [CellPLM Tutorial Notebook](t_cellplm.ipynb).
