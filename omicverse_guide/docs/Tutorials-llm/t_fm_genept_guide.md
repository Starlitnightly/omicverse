# GenePT

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

API-based GPT-3.5 gene embeddings (1536-dim), no local GPU required, gene-level (not cell-level)

!!! tip "When to choose GenePT"

    User wants gene-level embeddings (not cell-level), has no local GPU, or wants API-based OpenAI embeddings

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | GenePT |
| **Version** | v1.0 |
| **Tasks** | `embed` |
| **Modalities** | RNA |
| **Species** | human |
| **Gene IDs** | symbol |
| **Embedding Dim** | 1536 |
| **GPU Required** | No |
| **Min VRAM** | 0 GB |
| **Recommended VRAM** | 0 GB |
| **CPU Fallback** | Yes |
| **Adapter Status** | ⚠️ partial |

---

## Quick Start

```python
import omicverse as ov

# 1. Check model spec
info = ov.fm.describe_model("genept")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "genept", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="genept",
    adata_path="your_data.h5ad",
    output_path="output_genept.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_genept.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | No local preprocessing needed. Requires OpenAI API key for embedding generation. |
| **Data format** | AnnData (`.h5ad`) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_genept` | `adata.obsm` | Cell embeddings (1536-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_genept.h5ad")
embeddings = adata.obsm["X_genept"]  # shape: (n_cells, 1536)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_genept")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/yiqunchen/GenePT](https://github.com/yiqunchen/GenePT)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [GenePT Tutorial Notebook](t_fm_genept.ipynb).
