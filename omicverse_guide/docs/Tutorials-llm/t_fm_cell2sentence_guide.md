# Cell2Sentence

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

Converts cells to text sentences for LLM fine-tuning, 768-dim LLM embeddings

!!! tip "When to choose Cell2Sentence"

    User wants to leverage general-purpose LLMs, convert cells to text, or use LLM fine-tuning workflows

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | Cell2Sentence |
| **Version** | v1.0 |
| **Tasks** | `embed` |
| **Modalities** | RNA |
| **Species** | human |
| **Gene IDs** | symbol |
| **Embedding Dim** | 768 |
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
info = ov.fm.describe_model("cell2sentence")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "cell2sentence", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="cell2sentence",
    adata_path="your_data.h5ad",
    output_path="output_cell2sentence.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_cell2sentence.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Requires fine-tuning on reference data. Gene expression is converted to ranked gene sentences. |
| **Data format** | AnnData (`.h5ad`) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_cell2sentence` | `adata.obsm` | Cell embeddings (768-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_cell2sentence.h5ad")
embeddings = adata.obsm["X_cell2sentence"]  # shape: (n_cells, 768)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_cell2sentence")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/vandijklab/cell2sentence](https://github.com/vandijklab/cell2sentence)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [Cell2Sentence Tutorial Notebook](t_fm_cell2sentence.ipynb).
