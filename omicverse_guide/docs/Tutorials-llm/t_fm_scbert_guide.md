# scBERT

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

Compact 200-dim embeddings, BERT-style masked gene pretraining, lightweight model

!!! tip "When to choose scBERT"

    User needs compact 200-dim embeddings, BERT-style pretraining, or a lightweight model for constrained hardware

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | scBERT |
| **Version** | v1.0 |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA |
| **Species** | human |
| **Gene IDs** | symbol |
| **Embedding Dim** | 200 |
| **GPU Required** | Yes |
| **Min VRAM** | 8 GB |
| **Recommended VRAM** | 16 GB |
| **CPU Fallback** | Yes |
| **Adapter Status** | ⚠️ partial |

---

## Quick Start

```python
import omicverse as ov

# 1. Check model spec
info = ov.fm.describe_model("scbert")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "scbert", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="scbert",
    adata_path="your_data.h5ad",
    output_path="output_scbert.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_scbert.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard log-normalization and gene selection. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_scBERT` | `adata.obsm` | Cell embeddings (200-dim) |
| `scbert_pred` | `adata.obs` | Predicted cell type labels |

```python
import scanpy as sc

adata = sc.read_h5ad("output_scbert.h5ad")
embeddings = adata.obsm["X_scBERT"]  # shape: (n_cells, 200)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_scBERT")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/TencentAILabHealthcare/scBERT](https://github.com/TencentAILabHealthcare/scBERT)
- **Paper:** [https://www.nature.com/articles/s42256-022-00534-z](https://www.nature.com/articles/s42256-022-00534-z)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [scBERT Tutorial Notebook](t_fm_scbert.ipynb).
