# scGPT

✅ **Status:** ready | **Version:** whole-human-2024

---

## Overview

Multi-modal transformer (RNA+ATAC+Spatial), attention-based gene interaction modeling

!!! tip "When to choose scGPT"

    User needs multi-modal analysis (RNA+ATAC or spatial), or explicit attention-based gene interaction maps

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | scGPT |
| **Version** | whole-human-2024 |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA, ATAC, Spatial |
| **Species** | human, mouse |
| **Gene IDs** | symbol (HGNC) |
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
info = ov.fm.describe_model("scgpt")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "scgpt", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="scgpt",
    adata_path="your_data.h5ad",
    output_path="output_scgpt.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_scgpt.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol (HGNC) |
| **Preprocessing** | Normalize to 1e4 via `sc.pp.normalize_total`, then bin into 51 expression bins. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_scGPT` | `adata.obsm` | Cell embeddings (512-dim) |
| `scgpt_pred` | `adata.obs` | Predicted cell type labels |

```python
import scanpy as sc

adata = sc.read_h5ad("output_scgpt.h5ad")
embeddings = adata.obsm["X_scGPT"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_scGPT")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo)
- **Paper:** [https://www.nature.com/articles/s41592-024-02201-0](https://www.nature.com/articles/s41592-024-02201-0)
- **Documentation:** [https://scgpt.readthedocs.io/](https://scgpt.readthedocs.io/)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [scGPT Tutorial Notebook](t_scgpt.ipynb).
