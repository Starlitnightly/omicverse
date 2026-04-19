# scFoundation

✅ **Status:** ready | **Version:** xTrimoGene

---

## Overview

Large-scale asymmetric transformer (xTrimoGene), custom 19264 gene vocabulary, pre-trained for perturbation/drug response

!!! tip "When to choose scFoundation"

    User needs perturbation prediction, drug response modeling, or works with the xTrimoGene gene vocabulary

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | scFoundation |
| **Version** | xTrimoGene |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA |
| **Species** | human |
| **Gene IDs** | custom (19,264 gene set) |
| **Embedding Dim** | 512 |
| **GPU Required** | Yes |
| **Min VRAM** | 16 GB |
| **Recommended VRAM** | 32 GB |
| **CPU Fallback** | No |
| **Adapter Status** | ✅ ready |

---

## Quick Start

```python
import omicverse as ov

# 1. Check model spec
info = ov.fm.describe_model("scfoundation")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "scfoundation", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="scfoundation",
    adata_path="your_data.h5ad",
    output_path="output_scfoundation.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_scfoundation.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | custom (19,264 gene set) |
| **Preprocessing** | Match genes to model vocabulary. Follow xTrimoGene preprocessing pipeline. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_scfoundation` | `adata.obsm` | Cell embeddings (512-dim) |
| `scfoundation_pred` | `adata.obs` | Predicted cell type labels |

```python
import scanpy as sc

adata = sc.read_h5ad("output_scfoundation.h5ad")
embeddings = adata.obsm["X_scfoundation"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_scfoundation")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/biomap-research/scFoundation](https://github.com/biomap-research/scFoundation)
- **Paper:** [https://www.nature.com/articles/s41592-024-02305-7](https://www.nature.com/articles/s41592-024-02305-7)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [scFoundation Tutorial Notebook](t_scfoundation.ipynb).
