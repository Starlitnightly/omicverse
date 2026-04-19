# GeneCompass

⚠️ **Status:** partial | **Version:** 120M-cells

---

## Overview

Prior-knowledge-enhanced pretraining (gene regulatory networks + pathway info), 120M cell training corpus

!!! tip "When to choose GeneCompass"

    User mentions prior knowledge, gene regulatory networks, pathway-informed embeddings, or mouse+human cross-species

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | GeneCompass |
| **Version** | 120M-cells |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA |
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
info = ov.fm.describe_model("genecompass")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "genecompass", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="genecompass",
    adata_path="your_data.h5ad",
    output_path="output_genecompass.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_genecompass.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Follow GeneCompass preprocessing. Supports both human and mouse gene symbols. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_genecompass` | `adata.obsm` | Cell embeddings (512-dim) |
| `genecompass_pred` | `adata.obs` | Predicted cell type labels |

```python
import scanpy as sc

adata = sc.read_h5ad("output_genecompass.h5ad")
embeddings = adata.obsm["X_genecompass"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_genecompass")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/xCompass-AI/GeneCompass](https://github.com/xCompass-AI/GeneCompass)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [GeneCompass Tutorial Notebook](t_fm_genecompass.ipynb).
