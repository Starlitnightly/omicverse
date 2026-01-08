# Registered Functions — GPU Support Overview

## Legend
- [x] Supported: GPU acceleration available (native, parameter-enabled, or auto-enabled by environment).
- [ ] Not supported: CPU-only implementation.
- To enable GPU, initialize the environment first if needed: `ov.settings.gpu_init()` (RAPIDS) or `ov.settings.cpu_gpu_mixed_init()` (mixed). Actual GPU usage also depends on whether dependencies are installed with GPU support.

## Settings
- [x] `ov.settings.gpu_init`: Initialize RAPIDS/GPU mode.
- [x] `ov.settings.cpu_gpu_mixed_init`: Initialize CPU–GPU mixed mode.

## Preprocessing (ov.pp)
- [x] `ov.pp.anndata_to_GPU`: Move AnnData to GPU (RAPIDS).
- [ ] `ov.pp.anndata_to_CPU`: Move data back to CPU.
- [x] `ov.pp.preprocess`: End-to-end preprocessing (gpu <span class="tag tag-rapids">rapids</span>).
    - `mode='shiftlog|pearson'`
        - [x] normalize_total/log1p (gpu <span class="tag tag-rapids">rapids</span>).
        - [x] HVGs = pearson_residuals (gpu <span class="tag tag-rapids">rapids</span>).
    - `mode='pearson|pearson'`
        - [x] normalize_pearson_residuals (gpu <span class="tag tag-rapids">rapids</span>).
        - [x] HVGs = pearson_residuals (gpu <span class="tag tag-rapids">rapids</span>).
- [x] `ov.pp.scale`: Scaling (gpu <span class="tag tag-rapids">rapids</span>).
- [x] `ov.pp.pca`: PCA (gpu <span class="tag tag-rapids">rapids</span> | <span class="tag tag-mixed">cpu-gpu-mixed</span>[<span class="tag tag-torch">torch</span>|<span class="tag tag-mlx">mlx</span>]).
- [ ] `ov.pp.neighbors`: KNN graph (by `method`).
    - [ ] `method='umap'` (UMAP-based neighbor estimation, CPU).
    - [ ] `method='gauss'` (Gaussian kernel, CPU).
    - [x] `method='rapids'` (gpu <span class="tag tag-rapids">rapids</span>).
- [ ] `ov.pp.umap`: UMAP (by implementation).
    - [ ] Scanpy UMAP (`settings.mode='cpu'`).
    - [x] RAPIDS UMAP (`settings.mode='gpu'`, gpu <span class="tag tag-rapids">rapids</span>).
    - [x] PyMDE/torch path (`settings.mode='cpu-gpu-mixed'`, <span class="tag tag-mixed">cpu-gpu-mixed</span>[<span class="tag tag-torch">torch</span>]).
- [x] `ov.pp.qc`: Quality control (gpu <span class="tag tag-rapids">rapids</span> | <span class="tag tag-mixed">cpu-gpu-mixed</span>[<span class="tag tag-torch">torch</span>]).
- [ ] `ov.pp.score_genes_cell_cycle`: Cell cycle scoring.
- [x] `ov.pp.sude`: SUDE dimensionality reduction (<span class="tag tag-mixed">cpu-gpu-mixed</span>[<span class="tag tag-torch">torch</span>]).

## Utils (ov.utils)
- [x] `ov.utils.mde`: Minimum Distortion Embedding (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [ ] `ov.utils.cluster`: Multi-algorithm clustering (per algorithm below).
    - [x] Leiden (<span class="tag tag-mixed">cpu</span>[<span class="tag tag-torch">igraph</span>]<span class="tag tag-mixed">cpu-gpu-mixed</span>[<span class="tag tag-torch">pyg</span>]).
    - [ ] Louvain (Scanpy, CPU).
    - [ ] KMeans (scikit-learn, CPU).
    - [ ] GMM/mclust (scikit-learn, CPU).
    - [ ] mclust_R (R package mclust, CPU).
    - [ ] schist (schist library, CPU).
    - [ ] scICE (currently invoked with `use_gpu=False`).
- [ ] `ov.utils.refine_label`: Neighborhood voting label refinement.
- [ ] `ov.utils.weighted_knn_trainer`: Train weighted KNN.
- [ ] `ov.utils.weighted_knn_transfer`: Weighted KNN label transfer.

## Single-cell (ov.single)
- [ ] `ov.single.batch_correction`: Batch correction (per method below).
    - [ ] harmony (Harmony, CPU).
    - [ ] combat (Scanpy Combat, CPU).
    - [ ] scanorama (Scanorama, CPU).
    - [x] scVI (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
    - [ ] CellANOVA (CPU).
- [x] `ov.single.MetaCell`: SEACells (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [ ] `ov.single.TrajInfer`: Trajectory inference (per method below).
    - [ ] palantir (CPU).
    - [ ] diffusion_map (CPU).
    - [ ] slingshot (CPU).
- [x] `ov.single.Fate`: TimeFateKernel (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [x] `ov.single.pyCEFCON`: CEFCON driver discovery (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [x] `ov.single.gptcelltype_local`: Local LLM annotation (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [x] `ov.single.cNMF`: cNMF (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [ ] `ov.single.CellVote`: Multi-method voting.
  * [ ] `scsa_anno` (SCSA, CPU).
  * [ ] `gpt_anno` (online GPT, CPU/network).
  * [ ] `gbi_anno` (GPTBioInsightor, CPU/network).
  * [ ] `popv_anno` (PopV, CPU).
- [ ] `ov.single.gptcelltype`: Online GPT annotation.
- [ ] `ov.single.mouse_hsc_nestorowa16`: Load dataset.
- [ ] `ov.single.load_human_prior_interaction_network`: Load prior network.
- [ ] `ov.single.convert_human_to_mouse_network`: Cross-species symbol conversion.

## Spatial (ov.space)
- [x] `ov.space.pySTAGATE`: STAGATE spatial clustering (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [ ] `ov.space.clusters`: Multi-method spatial clustering (per method below).
    - [x] STAGATE (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
    - [x] GraphST (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
    - [x] CAST (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
    - [x] BINARY (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [ ] `ov.space.merge_cluster`: Merge clusters.
- [ ] `ov.space.Cal_Spatial_Net`: Build spatial neighbor graph.
- [x] `ov.space.pySTAligner`: STAligner integration (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [x] `ov.space.pySpaceFlow`: SpaceFlow spatial embedding (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [ ] `ov.space.Tangram`: Tangram deconvolution (per mode below).
    - [x] `mode='clusters'` (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
    - [x] `mode='cells'` (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [ ] `ov.space.svg`: Spatially variable genes (stats-based, not explicit GPU).
- [x] `ov.space.CAST`: CAST integration (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
- [ ] `ov.space.crop_space_visium`: Crop spatial image/coordinates.
- [ ] `ov.space.rotate_space_visium`: Rotate spatial image/coordinates.
- [ ] `ov.space.map_spatial_auto`: Auto mapping (per method below).
    - [x] `method='torch'` (<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]).
    - [ ] `method='phase'` (NumPy, CPU).
    - [ ] `method='feature'` (feature-based matching, CPU).
    - [ ] `method='hybrid'` (hybrid pipeline, CPU).
- [ ] `ov.space.map_spatial_manual`: Manual offset mapping.
- [ ] `ov.space.read_visium_10x`: Read Visium data.
- [x] `ov.space.visium_10x_hd_cellpose_he`: H&E segmentation (`gpu=True`).
- [ ] `ov.space.visium_10x_hd_cellpose_expand`: Label expansion.
- [x] `ov.space.visium_10x_hd_cellpose_gex`: GEX segmentation/mapping (`gpu=True`).
- [ ] `ov.space.salvage_secondary_labels`: Merge labels.
- [ ] `ov.space.bin2cell`: Bin-to-cell conversion.

## External (ov.external)
- [x] `ov.external.GraphST.GraphST`: GraphST (`device` supports GPU).
- [ ] `ov.bulk.pyWGCNA`: WGCNA (CPU implementation).

## Plotting (ov.pl)
- [ ] `ov.pl.*` (`_single/_bulk/_density/_dotplot/_violin/_general/_palette`): plotting APIs.

## Bulk (ov.bulk)
- [ ] `ov.bulk.*` (`_Deseq2/_Enrichment/_combat/_network/_tcga`): statistics, enrichment, and network analysis.
