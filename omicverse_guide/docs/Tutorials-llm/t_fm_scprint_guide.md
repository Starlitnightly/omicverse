# scPRINT

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

Protein-coding gene focus with built-in denoising, robust batch integration

!!! tip "When to choose scPRINT"

    User mentions denoising, protein-coding genes, ambient RNA removal, or wants built-in noise reduction

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | scPRINT |
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
info = ov.fm.describe_model("scprint")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "scprint", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="scprint",
    adata_path="your_data.h5ad",
    output_path="output_scprint.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_scprint.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard preprocessing. Model includes built-in denoising during inference. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_scprint` | `adata.obsm` | Cell embeddings (512-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_scprint.h5ad")
embeddings = adata.obsm["X_scprint"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_scprint")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/scprint/scPRINT](https://github.com/scprint/scPRINT)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [scPRINT Tutorial Notebook](t_fm_scprint.ipynb).
