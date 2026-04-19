# scPlantLLM

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

Plant-specific single-cell model, handles polyploidy and plant gene nomenclature

!!! tip "When to choose scPlantLLM"

    User has plant single-cell data (Arabidopsis, rice, maize, etc.) or mentions polyploidy

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | scPlantLLM |
| **Version** | v1.0 |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA |
| **Species** | plant (Arabidopsis, rice, maize, etc.) |
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
info = ov.fm.describe_model("scplantllm")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "scplantllm", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="scplantllm",
    adata_path="your_data.h5ad",
    output_path="output_scplantllm.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_scplantllm.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard preprocessing with plant gene nomenclature. Model handles polyploidy-specific challenges. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_scplantllm` | `adata.obsm` | Cell embeddings (512-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_scplantllm.h5ad")
embeddings = adata.obsm["X_scplantllm"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_scplantllm")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/scPlantLLM/scPlantLLM](https://github.com/scPlantLLM/scPlantLLM)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [scPlantLLM Tutorial Notebook](t_fm_scplantllm.ipynb).
