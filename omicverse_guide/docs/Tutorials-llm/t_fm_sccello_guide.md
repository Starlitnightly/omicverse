# scCello

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

Cell ontology-aligned embeddings, zero-shot cell type annotation with hierarchical coherence

!!! tip "When to choose scCello"

    User wants zero-shot cell type annotation, ontology-consistent predictions, or hierarchical cell-type labeling

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | scCello |
| **Version** | v1.0 |
| **Tasks** | `embed`, `integrate`, `annotate` |
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
info = ov.fm.describe_model("sccello")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "sccello", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="sccello",
    adata_path="your_data.h5ad",
    output_path="output_sccello.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_sccello.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard preprocessing. Annotations are aligned to the Cell Ontology hierarchy. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |
| **Label key** | `.obs` column for cell type labels (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_sccello` | `adata.obsm` | Cell embeddings (512-dim) |
| `sccello_pred` | `adata.obs` | Predicted cell type labels |

```python
import scanpy as sc

adata = sc.read_h5ad("output_sccello.h5ad")
embeddings = adata.obsm["X_sccello"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_sccello")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/cellarium-ai/scCello](https://github.com/cellarium-ai/scCello)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [scCello Tutorial Notebook](t_fm_sccello.ipynb).
