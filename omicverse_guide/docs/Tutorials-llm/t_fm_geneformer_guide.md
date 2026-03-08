# Geneformer

✅ **Status:** ready | **Version:** v2-106M

---

## Overview

Rank-value encoded transformer, Ensembl gene IDs, CPU-capable, network biology pretraining

!!! tip "When to choose Geneformer"

    User has Ensembl gene IDs, needs CPU-only inference, or wants gene-network-aware embeddings

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | Geneformer |
| **Version** | v2-106M |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA |
| **Species** | human |
| **Gene IDs** | ensembl (ENSG...) |
| **Embedding Dim** | 512 |
| **GPU Required** | No |
| **Min VRAM** | 4 GB |
| **Recommended VRAM** | 16 GB |
| **CPU Fallback** | Yes |
| **Adapter Status** | ✅ ready |

---

## Quick Start

```python
import omicverse as ov

# 1. Check model spec
info = ov.fm.describe_model("geneformer")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "geneformer", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="geneformer",
    adata_path="your_data.h5ad",
    output_path="output_geneformer.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_geneformer.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | ensembl (ENSG...) |
| **Preprocessing** | Rank-value encoding. Use `geneformer.preprocess()` for proper tokenization. Strip Ensembl version suffix (`.15`) if present. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

!!! warning "Gene ID Conversion"

    Geneformer requires Ensembl IDs (e.g., `ENSG00000141510`). If your data uses gene symbols, convert with:
    ```python
    # ov.fm.preprocess_validate() will detect this and suggest auto-fixes
    check = ov.fm.preprocess_validate("data.h5ad", "geneformer", "embed")
    print(check["auto_fixes"])  # Shows conversion suggestions
    ```

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_geneformer` | `adata.obsm` | Cell embeddings (512-dim) |
| `geneformer_pred` | `adata.obs` | Predicted cell type labels |

```python
import scanpy as sc

adata = sc.read_h5ad("output_geneformer.h5ad")
embeddings = adata.obsm["X_geneformer"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_geneformer")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://huggingface.co/ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- **Paper:** [https://www.nature.com/articles/s41586-023-06139-9](https://www.nature.com/articles/s41586-023-06139-9)
- **Documentation:** [https://geneformer.readthedocs.io/](https://geneformer.readthedocs.io/)
- **License:** Apache 2.0 (code)

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [Geneformer Tutorial Notebook](t_geneformer.ipynb).
