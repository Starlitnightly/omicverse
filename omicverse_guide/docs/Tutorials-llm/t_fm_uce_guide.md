# UCE

✅ **Status:** ready | **Version:** 4-layer

---

## Overview

Broadest species support (7 species), 1280-dim embeddings, universal cell embedding via protein structure

!!! tip "When to choose UCE"

    User has non-human/non-mouse species (zebrafish, frog, pig, macaque, lemur), or needs cross-species comparison

---

## Specifications

| Property | Value |
|----------|-------|
| **Model** | UCE |
| **Version** | 4-layer |
| **Tasks** | `embed`, `integrate` |
| **Modalities** | RNA |
| **Species** | human, mouse, zebrafish, mouse_lemur, macaque, frog, pig |
| **Gene IDs** | symbol |
| **Embedding Dim** | 1280 |
| **GPU Required** | Yes |
| **Min VRAM** | 16 GB |
| **Recommended VRAM** | 16 GB |
| **CPU Fallback** | No |
| **Adapter Status** | ✅ ready |

---

## Quick Start

```python
import omicverse as ov

# 1. Check model spec
info = ov.fm.describe_model("uce")

# 2. Profile your data
profile = ov.fm.profile_data("your_data.h5ad")

# 3. Validate compatibility
check = ov.fm.preprocess_validate("your_data.h5ad", "uce", "embed")

# 4. Run inference
result = ov.fm.run(
    task="embed",
    model_name="uce",
    adata_path="your_data.h5ad",
    output_path="output_uce.h5ad",
    device="auto",
)

# 5. Interpret results
metrics = ov.fm.interpret_results("output_uce.h5ad", task="embed")
```

---

## Input Requirements

| Requirement | Detail |
|-------------|--------|
| **Gene ID scheme** | symbol |
| **Preprocessing** | Standard log-normalization. Model handles tokenization internally. |
| **Data format** | AnnData (`.h5ad`) |
| **Batch key** | `.obs` column for batch integration (optional) |

---

## Output Keys

After running `ov.fm.run()`, results are stored in the AnnData object:

| Key | Location | Description |
|-----|----------|-------------|
| `X_uce` | `adata.obsm` | Cell embeddings (1280-dim) |

```python
import scanpy as sc

adata = sc.read_h5ad("output_uce.h5ad")
embeddings = adata.obsm["X_uce"]  # shape: (n_cells, 1280)

# Downstream analysis
sc.pp.neighbors(adata, use_rep="X_uce")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["leiden"])
```

---

## Resources

- **Repository / Checkpoint:** [https://github.com/snap-stanford/UCE](https://github.com/snap-stanford/UCE)
- **Paper:** [https://www.nature.com/articles/s41592-024-02201-0](https://www.nature.com/articles/s41592-024-02201-0)
- **Documentation:** [https://github.com/snap-stanford/UCE](https://github.com/snap-stanford/UCE)
- **License:** MIT License

---

## Hands-On Tutorial

For a step-by-step walkthrough with code, see the [UCE Tutorial Notebook](t_uce.ipynb).
