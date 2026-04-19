# CellFM

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

MLP architecture (not transformer), trained on ~126M cells (largest training corpus)

!!! tip "When to choose CellFM"

    User explicitly wants MLP-based (not transformer) model, or wants the largest pretraining scale (~126M cells)

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | CellFM |
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
info = ov.fm.describe_model("cellfm")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "cellfm", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="cellfm",
    adata_path="your_data.h5ad",
    output_path="output_cellfm.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_cellfm.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard preprocessing. Model uses MLP layers instead of attention. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_cellfm` | `adata.obsm` | Cell embeddings (512-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_cellfm.h5ad")
embeddings = adata.obsm["X_cellfm"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_cellfm")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/cellverse/CellFM](https://github.com/cellverse/CellFM)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [CellFM Tutorial Notebook](t_fm_cellfm.ipynb).
