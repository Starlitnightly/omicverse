# ChatCell

⚠️ **Status:** partial | **Version:** v1.0

---

## Overview

Conversational chat interface for single-cell analysis, zero-shot annotation via dialogue

!!! tip "When to choose ChatCell"

    User wants interactive chat-based cell analysis, conversational annotation, or dialogue-driven exploration

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | ChatCell |
| **Version** | v1.0 |
| **Tasks** | `embed`, `annotate` |
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
info = ov.fm.describe_model("chatcell")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "chatcell", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="chatcell",
    adata_path="your_data.h5ad",
    output_path="output_chatcell.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_chatcell.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard preprocessing. Conversational prompts can guide annotation. |
| **Data format** | AnnData (`.h5ad`) |
| **Label key** | `.obs` column for cell type labels (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_chatcell` | `adata.obsm` | Cell embeddings (512-dim) |
| `chatcell_pred` | `adata.obs` | Predicted cell type labels |

```python
import scanpy as sc

adata = sc.read_h5ad("output_chatcell.h5ad")
embeddings = adata.obsm["X_chatcell"]  # shape: (n_cells, 512)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_chatcell")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/chatcell/CHATCELL](https://github.com/chatcell/CHATCELL)
- **License:** Check upstream LICENSE

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [ChatCell Tutorial Notebook](t_fm_chatcell.ipynb).
