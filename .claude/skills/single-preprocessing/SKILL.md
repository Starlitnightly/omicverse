---
name: single-cell-preprocessing-with-omicverse
title: Single-cell preprocessing with omicverse
description: Walk through omicverse's single-cell preprocessing tutorials to QC PBMC3k data, normalise counts, detect HVGs, and run PCA/embedding pipelines on CPU, CPU–GPU mixed, or GPU stacks.
---

# Single-cell preprocessing with omicverse

## Overview
Follow this skill when a user needs to reproduce the preprocessing workflow from the omicverse notebooks [`t_preprocess.ipynb`](../../omicverse_guide/docs/Tutorials-single/t_preprocess.ipynb), [`t_preprocess_cpu.ipynb`](../../omicverse_guide/docs/Tutorials-single/t_preprocess_cpu.ipynb), and [`t_preprocess_gpu.ipynb`](../../omicverse_guide/docs/Tutorials-single/t_preprocess_gpu.ipynb). The tutorials operate on the 10x PBMC3k dataset and cover QC filtering, normalisation, highly variable gene (HVG) detection, dimensionality reduction, and downstream embeddings.

## Instructions
1. **Set up the environment**
   - Import `omicverse as ov` and `scanpy as sc`, then call `ov.plot_set(font_path='Arial')` (or `ov.ov_plot_set()` in legacy notebooks) to standardise figure styling.
   - Encourage `%load_ext autoreload` and `%autoreload 2` when iterating inside notebooks so code edits propagate without restarting the kernel.
2. **Prepare input data**
   - Download the PBMC3k filtered matrix from 10x Genomics (`pbmc3k_filtered_gene_bc_matrices.tar.gz`) and extract it under `data/filtered_gene_bc_matrices/hg19/`.
   - Load the matrix via `sc.read_10x_mtx(..., var_names='gene_symbols', cache=True)` and keep a writable folder like `write/` for exports.
3. **Perform quality control (QC)**
   - Run `ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}, doublets_method='scrublet')` for the CPU/CPU–GPU pipelines; omit `doublets_method` on pure GPU where Scrublet is not yet supported.
   - Review the returned AnnData summary to confirm doublet rates and QC thresholds; advise adjusting cut-offs for different species or sequencing depths.
4. **Store raw counts before transformations**
   - Call `ov.utils.store_layers(adata, layers='counts')` immediately after QC so the original counts remain accessible for later recovery and comparison.
5. **Normalise and select HVGs**
   - Use `ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000, target_sum=5e5)` to apply shift-log normalisation followed by Pearson residual HVG detection (set `target_sum=None` on GPU, which keeps defaults).
   - For CPU–GPU mixed runs, demonstrate `ov.pp.recover_counts(...)` to invert normalisation and store reconstructed counts in `adata.layers['recover_counts']`.
6. **Manage `.raw` and layer recovery**
   - Snapshot normalised data to `.raw` with `adata.raw = adata` (or `adata.raw = adata.copy()`), and show `ov.utils.retrieve_layers(adata_counts, layers='counts')` to compare normalised vs. raw intensities.
7. **Scale, reduce, and embed**
   - Scale features using `ov.pp.scale(adata)` (layers hold scaled matrices) followed by `ov.pp.pca(adata, layer='scaled', n_pcs=50)`.
   - Construct neighbourhood graphs with:
     - `sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, use_rep='scaled|original|X_pca')` for the baseline notebook.
     - `ov.pp.neighbors(..., use_rep='scaled|original|X_pca')` on CPU–GPU to leverage accelerated routines.
     - `ov.pp.neighbors(..., method='cagra')` on GPU to call RAPIDS graph primitives.
   - Generate embeddings via `ov.utils.mde(...)`, `ov.pp.umap(adata)`, `ov.pp.mde(...)`, `ov.pp.tsne(...)`, or `ov.pp.sude(...)` depending on the notebook variant.
8. **Cluster and annotate**
   - Run `ov.pp.leiden(adata, resolution=1)` after neighbour graph construction; CPU–GPU pipelines also showcase `ov.pp.score_genes_cell_cycle` before clustering.
   - Plot embeddings with `ov.pl.embedding(...)` or `ov.utils.embedding(...)`, colouring by `leiden` clusters and marker genes.
9. **Document outputs**
   - Encourage saving intermediate AnnData objects (`adata.write('write/pbmc3k_preprocessed.h5ad')`) and figure exports using Matplotlib’s `plt.savefig(...)` to preserve QC summaries and embeddings.
10. **Notebook-specific notes**
    - *Baseline (`t_preprocess.ipynb`)*: Focuses on CPU execution with Scanpy neighbours; emphasise storing counts before and after `retrieve_layers` demonstrations.
    - *CPU–GPU mixed (`t_preprocess_cpu.ipynb`)*: Highlights Omicverse ≥1.7.0 mixed acceleration. Include timing magics (%%time) to showcase speedups and call out `doublets_method='scrublet'` support.
    - *GPU (`t_preprocess_gpu.ipynb`)*: Requires a CUDA-capable GPU, RAPIDS 24.04 stack, and `rapids-singlecell`. Mention the `ov.pp.anndata_to_GPU`/`ov.pp.anndata_to_CPU` transfers and `method='cagra'` neighbours. Note the current warning that pure-GPU pipelines depend on RAPIDS updates.
11. **Troubleshooting tips**
    - If `sc.read_10x_mtx` fails, verify the extracted folder structure and ensure gene symbols are available via `var_names='gene_symbols'`.
    - Address GPU import errors by confirming the conda environment matches the RAPIDS version for the installed CUDA driver (`nvidia-smi`).
    - For `ov.pp.preprocess` dimension mismatches, ensure QC filtered out empty barcodes so HVG selection does not encounter zero-variance features.
    - When embeddings lack expected fields (e.g., `scaled|original|X_pca` missing), re-run `ov.pp.scale` and `ov.pp.pca` to rebuild the cached layers.

## Examples
- "Download PBMC3k counts, run QC with Scrublet, normalise with `shiftlog|pearson`, and compute MDE + UMAP embeddings on CPU." 
- "Set up the mixed CPU–GPU workflow in a fresh conda env, recover raw counts after normalisation, and score cell cycle phases before Leiden clustering." 
- "Provision a RAPIDS environment, transfer AnnData to GPU, run `method='cagra'` neighbours, and return embeddings to CPU for plotting."

## References
- Detailed walkthrough notebooks: [`t_preprocess.ipynb`](../../omicverse_guide/docs/Tutorials-single/t_preprocess.ipynb), [`t_preprocess_cpu.ipynb`](../../omicverse_guide/docs/Tutorials-single/t_preprocess_cpu.ipynb), [`t_preprocess_gpu.ipynb`](../../omicverse_guide/docs/Tutorials-single/t_preprocess_gpu.ipynb)
- Quick copy/paste commands: [`reference.md`](reference.md)
