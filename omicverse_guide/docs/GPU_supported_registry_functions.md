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
- [x] `ov.pp.preprocess`: End-to-end preprocessing (gpu[rapids]).
  - `mode='shiftlog|pearson'`
    - [x] normalize_total/log1p (gpu[rapids]).
    - [x] HVGs = pearson_residuals (gpu[rapids]).
  - `mode='pearson|pearson'`
    - [x] normalize_pearson_residuals (gpu[rapids]).
    - [x] HVGs = pearson_residuals (gpu[rapids]).
- [x] `ov.pp.scale`: Scaling (gpu[rapids]).
- [x] `ov.pp.pca`: PCA (gpu[rapids] | cpu-gpu-mixed[torch|mlx]).
- `ov.pp.neighbors`: KNN graph (by `method`).
  - [ ] `method='umap'` (UMAP-based neighbor estimation, CPU).
  - [ ] `method='gauss'` (Gaussian kernel, CPU).
  - [x] `method='rapids'` (gpu[rapids]).
- `ov.pp.umap`: UMAP (by implementation).
  - [ ] Scanpy UMAP (`settings.mode='cpu'`).
  - [x] RAPIDS UMAP (`settings.mode='gpu'`, gpu[rapids]).
  - [x] PyMDE/torch path (`settings.mode='cpu-gpu-mixed'`, cpu-gpu-mixed[torch]).
- [x] `ov.pp.qc`: Quality control (gpu[rapids] | cpu-gpu-mixed[torch]).
- [ ] `ov.pp.score_genes_cell_cycle`: Cell cycle scoring.
- [ ] `ov.pp.sude`: SUDE dimensionality reduction (CPU).

## Utils (ov.utils)
- [x] `ov.utils.mde`: Minimum Distortion Embedding (all[torch]).
- `ov.utils.cluster`: Multi-algorithm clustering (per algorithm below).
  - [ ] Leiden (Scanpy, CPU).
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
- `ov.single.batch_correction`: Batch correction (per method below).
  - [ ] harmony (Harmony, CPU).
  - [ ] combat (Scanpy Combat, CPU).
  - [ ] scanorama (Scanorama, CPU).
  - [x] scVI (all[torch]).
  - [ ] CellANOVA (CPU).
- [x] `ov.single.MetaCell`: SEACells (all[torch]).
- `ov.single.TrajInfer`: Trajectory inference (per method below).
  - [ ] palantir (CPU).
  - [ ] diffusion_map (CPU).
  - [ ] slingshot (CPU).
- [x] `ov.single.Fate`: TimeFateKernel (all[torch]).
- [x] `ov.single.pyCEFCON`: CEFCON driver discovery (all[torch]).
- [x] `ov.single.gptcelltype_local`: Local LLM annotation (all[torch]).
- [ ] `ov.single.cNMF`: cNMF (CPU implementation).
- [ ] `ov.single.CellVote`: Multi-method voting.
  - [ ] `scsa_anno` (SCSA, CPU).
  - [ ] `gpt_anno` (online GPT, CPU/network).
  - [ ] `gbi_anno` (GPTBioInsightor, CPU/network).
  - [ ] `popv_anno` (PopV, CPU).
- [ ] `ov.single.gptcelltype`: Online GPT annotation.
- [ ] `ov.single.mouse_hsc_nestorowa16`: Load dataset.
- [ ] `ov.single.load_human_prior_interaction_network`: Load prior network.
- [ ] `ov.single.convert_human_to_mouse_network`: Cross-species symbol conversion.

## Spatial (ov.space)
- [x] `ov.space.pySTAGATE`: STAGATE spatial clustering (all[torch]).
- `ov.space.clusters`: Multi-method spatial clustering (per method below).
  - [x] STAGATE (all[torch]).
  - [x] GraphST (all[torch]).
  - [x] CAST (all[torch]).
  - [x] BINARY (all[torch]).
- [ ] `ov.space.merge_cluster`: Merge clusters.
- [ ] `ov.space.Cal_Spatial_Net`: Build spatial neighbor graph.
- [x] `ov.space.pySTAligner`: STAligner integration (all[torch]).
- [x] `ov.space.pySpaceFlow`: SpaceFlow spatial embedding (all[torch]).
- `ov.space.Tangram`: Tangram deconvolution (per mode below).
  - [x] `mode='clusters'` (all[torch]).
  - [x] `mode='cells'` (all[torch]).
- [ ] `ov.space.svg`: Spatially variable genes (stats-based, not explicit GPU).
- [x] `ov.space.CAST`: CAST integration (all[torch]).
- [ ] `ov.space.crop_space_visium`: Crop spatial image/coordinates.
- [ ] `ov.space.rotate_space_visium`: Rotate spatial image/coordinates.
- `ov.space.map_spatial_auto`: Auto mapping (per method below).
  - [x] `method='torch'` (all[torch]).
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
