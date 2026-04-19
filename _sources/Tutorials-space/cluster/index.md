# Spatial clustering and denoising expressions

Spatial clustering, which shares an analogy with single-cell clustering, has expanded the scope of tissue physiology studies from cell-centroid to structure-centroid with spatially resolved transcriptomics (SRT) data.

Here, we present **five spatial clustering methods** in OmicVerse — one notebook per method, all driven by the same canonical fixture and clustered through [`pymclustR`](https://pypi.org/project/pymclustR/) (a pure-Python re-implementation of CRAN [`mclust`](https://github.com/cran/mclust) — no `rpy2` / R install needed).

We made several improvements while integrating the `GraphST`, `BINARY`, `Banksy`, `CAST` and `STAGATE` algorithms in OmicVerse:

- Removed the preprocessing that comes bundled with `GraphST` and used the same preprocessing as the rest of OmicVerse.
- Optimised the dimensional display of `GraphST` — PCA is now a self-contained computational step.
- Implemented `mclust` in pure Python (`pymclustR`), removing the R-language dependency. The legacy `method='mclust_R'` (rpy2 bridge to CRAN `mclust`) is still available for backwards compatibility.
- Provided a unified interface `ov.space.cluster` so that any embedder can be swapped at the same call site.

## Notebooks (one per spatial embedder)

| # | Embedder | Notebook | Best for | Reference |
|--:|---|---|---|---|
| 1 | **GraphST** | [`t_cluster_graphst.ipynb`](t_cluster_graphst.ipynb) | spot-level Visium / Slide-seq; benchmarks-best in *Nat. Methods* 2024-04 | Long et al., *Nat. Commun.* 2023 — [10.1038/s41467-023-36796-3](https://doi.org/10.1038/s41467-023-36796-3) |
| 2 | **BINARY**  | [`t_cluster_binary.ipynb`](t_cluster_binary.ipynb)   | very large or very sparse spatial datasets where binary presence/absence already carries the signal | Lin et al., *Cell Genomics* 2024 — [10.1016/j.xgen.2024.100565](https://doi.org/10.1016/j.xgen.2024.100565) |
| 3 | **STAGATE** | [`t_cluster_stagate.ipynb`](t_cluster_stagate.ipynb) | spot-level Visium; bonus denoised expression matrix for marker plots | Dong & Zhang, *Nat. Commun.* 2022 — [10.1038/s41467-022-29439-6](https://doi.org/10.1038/s41467-022-29439-6) |
| 4 | **CAST**    | [`t_cluster_cast.ipynb`](t_cluster_cast.ipynb)       | multi-slice single-cell-resolution data (Xenium, MERFISH, NanoString) | Tang et al., *Nat. Methods* 2024 — [10.1038/s41592-024-02410-7](https://doi.org/10.1038/s41592-024-02410-7) |
| 5 | **BANKSY**  | [`t_cluster_banksy.ipynb`](t_cluster_banksy.ipynb)   | unifies cell-typing and tissue-domain segmentation via a single mixing parameter `λ` | Singhal et al., *Nat. Genet.* 2024 — [10.1038/s41588-024-01664-3](https://doi.org/10.1038/s41588-024-01664-3) |

All five run on the **Maynard 151676 dorsolateral prefrontal cortex** Visium sample (3 460 spots × 10 747 genes) using the canonical pre-processed fixture from `omicverse-test`:

```text
omicverse-test/notebooks/data/cluster_svg.h5ad
omicverse-test/data/151676/151676_truth.txt   # ground-truth layer labels
```

## Recommendation

| Resolution / scale | Recommended embedder(s) |
|---|---|
| Spot-level Visium / Slide-seq | GraphST, BINARY, STAGATE |
| Single-cell-resolution (NanoString / Xenium / MERFISH) | BINARY, CAST |
| Multi-slice (3-D / batched) | CAST, STAligner |
| Cell-typing **and** tissue-domain in one pass | BANKSY |

For the Gaussian-mixture clusterer that follows the embedding step, use `method='pymclustR'` for a pure-Python pipeline; if you already have rpy2 + R + mclust set up, `method='mclust_R'` is still supported.

## Citations

If you use these notebooks please cite the relevant embedder paper plus omicverse and the pymclustR port:

- **GraphST** — Long, Y., Ang, K.S., Li, M. *et al.* Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST. *Nat. Commun.* 14, 1155 (2023). <https://doi.org/10.1038/s41467-023-36796-3>
- **BINARY** — Lin S., Cui Y., Zhao F., Yang Z., Song J., Yao J., et al. Complete spatially resolved gene expression is not necessary for identifying spatial domains. *Cell Genomics* 4, 100565 (2024). <https://doi.org/10.1016/j.xgen.2024.100565>
- **STAGATE** — Dong, K., Zhang, S. Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder. *Nat. Commun.* 13, 1739 (2022). <https://doi.org/10.1038/s41467-022-29439-6>
- **CAST** — Tang, Z., Luo, S., Zeng, H. *et al.* Search and match across spatial omics samples at single-cell resolution. *Nat. Methods* 21, 1818–1829 (2024). <https://doi.org/10.1038/s41592-024-02410-7>
- **BANKSY** — Singhal, V., Chou, N., Lee, J. *et al.* BANKSY unifies cell typing and tissue domain segmentation for scalable spatial omics data analysis. *Nat. Genet.* 56, 431–441 (2024). <https://doi.org/10.1038/s41588-024-01664-3>
- **mclust** — Scrucca L., Fop M., Murphy T.B., Raftery A.E. mclust 5: clustering, classification and density estimation using Gaussian finite mixture models. *R Journal* 8, 289–317 (2016). <https://journal.r-project.org/archive/2016-1/scrucca-fop-murphy-etal.pdf>
- **pymclustR** — Pure-Python port of CRAN mclust used in these tutorials. <https://pypi.org/project/pymclustR/>
