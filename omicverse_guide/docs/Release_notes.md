# Release Notes

## v 1.0.0
- First public release.

## v 1.1.7
### bulk module:
- Added Deseq2, including `pyDEseq` functions: `deseq2_normalize`, `estimateSizeFactors`, `estimateDispersions`, `Matrix_ID_mapping`.
- Included TCGA with `TCGA`.
- Introduced Enrichment with functions `geneset_enrichment`, `geneset_plot`.

### single module:
- Integrated scdrug with functions `autoResolution`, `writeGEP`, `Drug_Response`.
- Added cpdb with functions `cpdb_network_cal`, `cpdb_plot_network`, `cpdb_plot_interaction`, `cpdb_interaction_filtered`.
- Included scgsea with functions `geneset_aucell`, `pathway_aucell`, `pathway_aucell_enrichment`, `pathway_enrichment`, `pathway_enrichment_plot`.

## v 1.1.8
### single module:
- Addressed errors in cpdb, including import errors and color issues in `cpdb_plot_network`.
- Introduced `cpdb_submeans_exacted` in cpdb for easy sub-network extraction.

## v 1.1.9
### bulk2single module:
- Added the `bulk2single` module.
- Fixed model load error from bulk2space.
- Resolved early stop issues from bulk2space.
- Included more user-friendly input methods and visualizations.
- Added loss history visualization.

### utils module:
- Introduced `pyomic_palette` in the plot module.

## v 1.1.10
- Updated all code references.

### single module:
- Fixed non-valid parameters in `single.mofa.mofa_run` function.
- Added layer raw count addition in `single.scanpy_lazy` function.
- Introduced `utils.plot_boxplot` for plotting box plots with jittered points.
- Added `bulk.pyDEseq.plot_boxplot` for plotting box plots with jittered points for specific genes.

## v 1.2.0
### bulk module:
- Fixed non-valid `cutoff` parameter in `bulk.geneset_enrichment`.
- Added modules: `pyPPI`, `pyGSEA`, `pyWGCNA`, `pyTCGA`, `pyDEG`.

### bulk2single module:
- Introduced `bulk2single.save` for manual model saving.

## v 1.2.1-4
### single module:
- Added `pySCSA` module with functions: `cell_anno`, `cell_anno_print`, `cell_auto_anno`, `get_model_tissue`.
- Implemented doublet cell filtering in `single.scanpy_lazy`.
- Added `single.scanpy_cellanno_from_dict` for easier annotation.
- Updated SCSA database from [CellMarker2.0](http://bio-bigdata.hrbmu.edu.cn/CellMarker/).
- Fixed errors in SCSA database keys: `Ensembl_HGNC` and `Ensembl_Mouse`.

## v 1.2.5
### single module:
- Added `pyVIA` module with functions: `run`, `plot_piechart_graph`, `plot_stream`, `plot_trajectory_gams`, `plot_lineage_probability`, `plot_gene_trend`, `plot_gene_trend_heatmap`, `plot_clustergraph`.
- Fixed warning error in `utils.pyomic_plot_set`.
- Updated requirements, including `pybind11`, `hnswlib`, `termcolor`, `pygam`, `pillow`, `gdown`.

## v 1.2.6
### single module:
- Added `pyVIA.get_piechart_dict` and `pyVIA.get_pseudotime`.

## v 1.2.7
### bulk2single module:
- Added `Single2Spatial` module with functions: `load`, `save`, `train`, `spot_assess`.
- Fixed installation errors for packages in pip.

## v 1.2.8
- Fixed pip installation errors.

### bulk2single module:
- Replaced `deep-forest` in `Single2Spatial` with `Neuron Network` for classification tasks.
- Accelerated the entire Single2Spatial inference process using GPU and batch-level estimation by modifying the `predicted_size` setting.

## v 1.2.9
### bulk module:
- Fixed duplicates_index mapping in `Matrix_ID_mapping`.
- Resolved hub genes plot issues in `pyWGCNA.plot_sub_network`.
- Fixed backupgene in `pyGSEA.geneset_enrichment` to support rare species.
- Added matrix plot module in `pyWGCNA.plot_matrix`.

### single module:
- Added `rank_genes_groups` check in `pySCSA`.

### bulk2single module:
- Fixed import error of `deepforest`.

## v 1.2.10
- Renamed the package to `omicverse`.

### single module:
- Fixed argument error in `pySCSA`.

### bulk2single module:
- Updated plot arguments in `bulk2single`.

## v 1.2.11
### bulk module:
- Fixed `wilcoxon` method in `pyDEG.deg_analysis`.
- Added parameter setting for treatment and control group names in `pyDEG.plot_boxplot`.
- Fixed figure display issues in `pyWGCNA.plot_matrix`.
- Fixed category correlation failed by one-hot in `pyWGCNA.analysis_meta_correlation`.
- Fixed network display issues in `pyWGCNA.plot_sub_network` and updated `utils.plot_network` to avoid errors.

## v 1.3.0
### bulk module:
- Added `DEseq2` method to `pyDEG.deg_analysis`.
- Introduced `pyGSEA` module in `bulk`.
- Renamed raw `pyGSEA` to `pyGSE` in `bulk`.
- Added `get_gene_annotation` in `utils` for gene name transformation.

## v 1.3.1
### single module:
- Added `get_celltype_marker` method.

### single module:
- Added `GLUE_pair`, `pyMOFA`, `pyMOFAART` module.
- Added tutorials for `Multi omics analysis by MOFA and GLUE`.
- Updated tutorial for `Multi omics analysis by MOFA`.

## v 1.4.0
### bulk2single module:
- Added `BulkTrajBlend` method.

### single module:
- Fixed errors in `scnocd` model.
- Added `save`, `load`, and `get_pair_dict` in `scnocd` model.

### utils module:
- Added `mde` method.
- Added `gz` format support for `utils.read`.

## v 1.4.1
### preprocess module:
- Added `pp` (preprocess) module with `qc` (quantity control), `hvg` (high variable feature), `pca`.
- Added `data_files` for cell cycle calculation from [Cellula](https://github.com/andrecossa5/Cellula/) and [pegasus](https://github.com/lilab-bcb/pegasus/).

## v 1.4.3
###

 preprocess module:
- Fixed sparse preprocess error in `pp`.
- Fixed trajectory import error in `via`.
- Added gene correlation analysis of trajectory.

## v 1.4.4
### single module:
- Added `panglaodb` database to `pySCSA` module.
- Fixed errors in `pySCSA.cell_auto_anno` when some cell types are not found in clusters.
- Fixed errors in `pySCSA.cell_anno` when `rank_genes_groups` are not consistent with clusters.
- Added `pySIMBA` module in single for batch correction.

### preprocess module:
- Added `store_layers` and `retrieve_layers` in `ov.utils`.
- Added `plot_embedding_celltype` and `plot_cellproportion` in `ov.utils`.

## v 1.4.5
### single module:
- Added `MetaTiME` module to perform cell type annotation automatically in TME.

## v 1.4.12
- Updated `conda install omicverse -c conda-forge`.

### single module:
- Added `pyTOSICA` module to perform cell type migration from reference scRNA-seq in Transformer model.
- Added `atac_concat_get_index`, `atac_concat_inner`, `atac_concat_outer` functions to merge/concatenate scATAC data.
- Fixed `MetaTime.predicted` when Unknown cell type appears.

### preprocess module:
- Added `plot_embedding` in `ov.utils` to plot UMAP in a special color dictionary.

## v 1.4.13
### bulk module:
- Added `mad_filtered` to filter robust genes when calculating the network in `ov.bulk.pyWGCNA` module.
- Fixed `string_interaction` in `ov.bulk.pyPPI` for string-db updates.

### preprocess module:
- Changed `mode` argument of `pp.preprocess` to control preprocessing steps.
- Added `ov.utils.embedding`, `ov.utils.neighbors`, and `ov.utils.stacking_vol`.

## v 1.4.14
### preprocess module:
- Added `batch_key` in `pp.preprocess` and `pp.qc`.

### utils module:
- Added `plot_ConvexHull` to visualize the boundary of clusters.
- Added `weighted_knn_trainer` and `weighted_knn_transfer` for multi-adata integration.

### single module:
- Fixed import errors in `mofa`.

## v 1.4.17
### bulk module:
- Fixed compatibility issues with `pydeseq2` version `0.4.0`.
- Added `bulk.batch_correction` for multi-bulk RNA-seq/microarray samples.

### single module:
- Added `single.batch_correction` for multi-single cell datasets.

### preprocess module:
- Added parameter `layers_add` in `pp.scale`.

## v 1.5.0
### single module:
- Added `cellfategenie` to calculate timing-associated genes/genesets.
- Fixed the name error in `atac_concat_outer`.
- Added more kwargs for `batch_correction`.

### utils module:
- Added `plot_heatmap` to visualize the heatmap of pseudotime.
- Fixed `embedding` when the version of `mpl` is larger than `3.7.0`.
- Added `geneset_wordcloud` to visualize geneset heatmaps of pseudotime.

## v 1.5.1
### single module:
- Added `scLTNN` to infer cell trajectory.

### bulk2single module:
- Updated cell fraction prediction with `TAPE` in bulk2single.
- Fixed group and normalization issues in bulk2single.

### utils module:
- Added `Ro/e` calculation (by: Haihao Zhang).
- Added `cal_paga` and `plot_paga` to visualize the state transfer matrix.
- Fixed the `read` function.

## v 1.5.2
### bulk2single Module:
- Resolved a matrix error occurring when gene symbols are not unique.
- Addressed the `interpolation` issue in `BulkTrajBlend` when target cells do not exist.
- Corrected the `generate` function in `BulkTrajBlend`.
- Rectified the argument for `vae_configure` in `BulkTrajBlend` when `cell_target_num` is set to None.
- Introduced the parameter `max_single_cells` for input in `BulkTrajBlend`.
- Defaulted to using `scaden` for deconvolution in Bulk RNA-seq.

### single Module:
- Fixed an error in `pyVIA` when the root is set to None.
- Added the `TrajInfer` module for inferring cell trajectories.
- Integrated `Palantir` and `Diffusion_map` into the `TrajInfer` module.
- Corrected the parameter error in `batch_correction`.

### utils Module:
- Introduced `plot_pca_variance_ratio` for visualizing the ratio of PCA variance.
- Added the `cluster` and `filtered` module for clustering the cells
- Integrated `MiRA` to calculate the LDA topic

## v 1.5.3
### single Module:
- Added `scVI` and `MIRA` to remove batch effect

### space Module:
- Added `STAGATE` to cluster and denoisy the spatial RNA-seq 

### pp Module:
- Added `doublets` argument of `ov.pp.qc` to control doublets('Default'=True)

## v 1.5.4
### bulk Module:
- Fixed an error in `pyDEG.deg_analysis` when `n_cpus` can not be set in `pyDeseq2(v0.4.3)`

### single Module:
- Fixed an argument error in `single.batch_correction` of combat

### utils Module:
- Added `venn4` plot to visualize
- Fixed the label visualization of `plot_network`
- Added `ondisk` argument of `LDA_topic`

### space Module:
- Added `Tangram` to mapping the scRNA-seq to stRNA-seq

## v 1.5.5
### pp Module:
- Added `max_cells_ratio` and `max_genes_ratio` to control the max threshold in qc of scRNA-seq

### single Module:
- Added `SEACells` model to calculate the metacells from scRNA-seq

### space Module:
- Added `STAligner` to integrate multi stRNA-seq

## v 1.5.6
### pp Module
- Added `mt_startswith` argument to control the `qc` in mouse or other species.

### utils Module
- Added `schist` method to cluster the single cell RNA-seq

### single Module
- Fixed the import error of `palantir` in SEACells
- Added `CEFCON` model to identify the driver regulators of cell fate decisions

### bulk2single Module
- Added `use_rep` and `neighbor_rep` argument to configure the nocd 

### space Module
- Added `SpaceFlow` to identify the pseudo-spatial map

## v 1.5.8

### pp Module
- Added `score_genes_cell_cycle` function to calculate the cell cycle

### bulk Module
- Fixed `dds.plot_volcano` text plot error when the version of `adjustText` larger than `0.9`

### single Module
- Optimised `MetaCell.load` model loading logic
- Fixed an error when loading the model usng `MetaCell.load`
- Added tutorials of `Metacells`

### pl Module

Add `pl` as a unified drawing prefix for the next release, to replace the drawing functionality in the original utils, while retaining the drawing in the original utils.

- Added `embedding` to plot the embedding of scRNA-seq using `ov.pl.embedding`
- Added `optim_palette` to provide a spatially constrained approach that generates discriminate color assignments for visualizing single-cell spatial data in various scenarios
- Added `cellproportion` to plot the proportion of stack bar of scRNA-seq
- Added `embedding_celltype` to plot the figures both celltype proportion and embedding
- Added `ConvexHull` to plot the ConvexHull around the target cells
- Added `embedding_adjust` to adjust the text of celltype legend in embedding
- Added `embedding_density` to plot the category density in the cells
- Added `bardotplot` to plot the bardotplot between different groups.
- Added `add_palue` to plot the p-threshold between different groups.
- Added `embedding_multi` to support the `mudata` object
- Added `purple_color` to visualize the purple palette.
- Added `venn` to plot the venn from set 2 to set 4
- Added `boxplot` to visualize the boxdotplot
- Added `volcano` to visualzize the result of differential expressed genes

## v 1.5.9

### single Module

- Added `slingshot` in `single.TrajInfer`
- Fixed some error of `scLTNN`
- Added `GPU` mode to preprocess the data
- Added `cNMF` to calculate the nmf

### space Module

- Added `Spatrio` to mapping the scRNA-seq to stRNA-seq

## v 1.6.0

Move `CEFCON`,`GNTD`,`mofapy2`,`spaceflow`,`spatrio`,`STAligner`,`tosica` from root to external module.

### space Module

- Added `STT` in `omicverse.space` to calculate the spatial transition tensor.
- Added `scSLAT` in `omicverse.external` to align of different spatial slices.
- Added `PROST` in `omicverse.external` and `svg` in `omicverse.space` to identify the spatially variable genes and domain.

### single Module

- Added `get_results_rfc` in `omicverse.single.cNMF` to predict the precise cluster in complex scRNA-seq/stRNA-seq
- Added `get_results_rfc` in `omicverse.utils.LDA_topic` to predict the precise cluster in complex scRNA-seq/stRNA-seq
- Added `gptcelltype` in `omicverse.single` to annotate celltype using large language model #82.

### pl Module

- Added `plot_spatial` in `omicverse.pl` to visual the spot proportion of cells when deconvolution

## v 1.6.2

Support Raw Windows platform

- Added `mde` in `omicverse.pp` to accerate the umap calculation.

## v 1.6.3

- Added  `ov.setting.cpu_init` to change the environment to CPU.
- Move module `tape`,`SEACells` and `palantir` to `external`

### Single Module
- Added `CytoTrace2` to predict cellular potency categories and absolute developmental potential from single-cell RNA-sequencing data.
- Added `cpdb_exact_target` and `cpdb_exact_source` to exact the means of special ligand/receptor
- Added `gptcelltype_local` to identify the celltype using local LLM #96 #99

### Bulk Module
- Added `MaxBaseMean` columns in dds.result to help people ignore the empty samples.
  
### Space Module
- Added `**kwargs` in `STT.compute_pathway`
- Added `GraphST` to identify the spatial domain

### pl Module
- Added `cpdb_network`, `cpdb_chord`, `cpdb_heatmap`, `cpdb_interacting_network`,`cpdb_interacting_heatmap` and `cpdb_group_heatmap` to visualize the result of CellPhoneDB

### utils Module
- Added `mclust_py` to identify the Gaussian Mixture cluster
- Added `mclust` methdo in `cluster` function

## v 1.6.4

### Bulk Module

- Optimised pyGSEA's `geneset_plot` visualisation of coordinate effects
- Fixed an error of `pyTCGA.survival_analysis` when the matrix is sparse. #62, #68, #95
- Added tqdm to visualize the process of `pyTCGA.survial_analysis_all`
- Fixed an error of `data_drop_duplicates_index` with remove duplicate indexes to retain only the highest expressed genes #45
- Added `geneset_plot_multi` in `ov.bulk` to visualize the multi results of enrichment. #103

### Single Module

- Added `mellon_density` to calculate the cell density. #103

### PP Module
- Fixed an error of `ov.pp.pca` when pcs smaller than 13. #102
- Added `COMPOSITE` in `ov.pp.qc`'s method to predicted doublet cells. #103
- Added `species` argument in `score_genes_cell_cycle` to calculate the cell phase without gene manual input

## v 1.6.6

### Pl Module

- Fixed the 'celltyep_key' error of `ov.pl.cpdb_group_heatmap` #109
- Fixed an error in `ov.utils.roe` when some expected frequencies are less than expected value.
- Added `cellstackarea` to visual the Percent stacked area chart of celltype in samples.

### Single Module
- Fixed the bug of `ov.single.cytotrace2` when adata.X is not sparse data. #115, #116
- Fixed the groupby error in `ov.single.get_obs_value` of SEACells.
- Fixed the error of cNMF #107, #85
- Fixed the plot error when `Pycomplexheatmap` version > 1.7 #136


### Bulk Module

- Fixed an key error in `ov.bulk.Matrix_ID_mapping`
- Added `enrichment_multi_concat` in `ov.bulk` to concat the result of enrichment.
- Fixed the pandas version error in gseapy #137

### Bulk2Single Module

- Added `adata.var_names_make_unique()` to avoid mat shape error if gene not unique. #100

### Space Module

- Fixed an error in `construct_landscape` of `ov.space.STT`
- Fixed an error of `get_image_idx_1D` in `ov.space.svg` #117
- Added `COMMOT` to calculate the cell-cell interaction of spatial RNA-seq.
- Added `starfysh` to deconvolute spatial transcriptomic without scRNA-seq (#108)

### PP Module

- Updated constraint error of ov.pp.mde #129
- Fixed type error of `float128` #134


## v 1.6.7

### Space Module

- Added `n_jobs` argument to adjust thread in `extenel.STT.pl.plot_tensor_single`
- Fixed an error in `extenel.STT.tl.construct_landscape`
- Updated the tutorial of `COMMOT` and `Flowsig`
  

### Pl Module

- Added `legend_awargs` to adjust the legend set in `pl.cellstackarea` and `pl.cellproportion`

### Single Module

- Fixed the error of `get_results` and `get_results_rfc` in `cNMF` module. (#143) (#139)
- Added `sccaf` to obtain the best clusters.
- Fixed the `.str` error in cytotrace2 (#146)

### Bulk Module

- Fixed the import error of `gseapy` in `bulk.geneset_enrichment`
- Optimized code logic for offline enrichment analysis, added background parameter
- Added `pyWGCNA` package replace the raw calculation of pyWGCNA (#162)

### Bulk2Single Module

- Remove `_stat_axis` in `bulk2single_data_prepare` and use `index` instead of it (#160).

### PP Module

- Fixed a return bugs in `pp.regress_and_scale` (#156)
- Fixed a scanpy version error when using `ov.pp.pca` (#154)

## v 1.6.8

### Bulk Module

- Fixed the error of log_init in gsea_obj.enrichment (#184)
- Added `ax` argument to visualize the `geneset_plot`

### Space Module

- Added CAST to integrate multi slice
- Added `crop_space_visium` in `omicverse.tl` to crop the sub area of space data

### Pl Module

- Added `legend` argument to visualize the `cpdb_heatmap`
- Added `text_show` argument to visualize the `cellstackarea`
- Added `ForbiddenCity` color system

## v 1.6.9

### PP Module

- Added `recover_counts` to recover `counts` after `ov.pp.preprocess`
- removed the lognorm layers added in `ov.pp.pca`

### Single Module

- Added `MultiMap` module to integrate multi species
- Added `CellVote` to vote the best cells
- Added `CellANOVA` to integrate samples and correct the batch effect
- Added `StaVia` to calculate the pseudotime and infer trajectory.

### Space Module

- Added `ov.space.cluster` to identify the spatial domain
- Added `Binary` for spatial cluster
- Added `Spateo` to calculate the SVG

## v 1.7.0

Added `cpu-gpu-mixed` to accelerate the analysis of scrna-seq using GPU.
Changed the logo presentation of Omicverse to `ov.plot_set`

### Bulk Module
- Added `limma`, `edgeR` in different expression gene analysis. (#238)
- Fixed the version error of `DEseq2` analysis.

### Single Module
- Added `lazy` function to calculate all function of scrna-seq (#291)
- Added `generate_scRNA_report` and `generate_reference_table` to generate the report and reference (#291) (#292)
- Fixed `geneset_prepare` not being able to read gmt not split by `\t\t` (#235) (#238)
- Added `geneset_aucell_tmp`,`pathway_aucell_tmp`,`pathway_aucell_enrichment_tmp` to test the chunk_size (#238)
- Added data enhancement of `Fate`
- Added `plot_atlas_view_ov` in VIA
- Fixed an error when the matrix is too large in `recover_counts`.
- Added `forceatlas2` to calculate the `X_force_directed`.
- Added `milo` and `scCODA` to analysis different celltype abundance.
- Added `memento` to analysis different gene expression.

### Space Module
- Added `GASTON` to learn a topographic map of a tissue slice from spatially resolved transcriptomics (SRT) data (#238)
- Added super kwargs in `plot_tensor_single` of STT.
- Updated `COMMOT` using GPU-accerlate
  
### Plot Module
- Added `dotplot_doublegroup` to visual the genes in doublegroup.
- Added `transpose` argument of `cpdb_interacting_heatmap` to transpose the figure.
- Added `calculate_gene_density` to plot the gene's density. 


## v 1.7.1

### Single Module
- Fixed some error of `ov.single.lazy`.
- Fixed the format of `ov.single.generate_scRNA_report`
- Updated some functions of `palantir`
- Added `CellOntologyMapper` to map cell name.


## v 1.7.2

### Pl Module
- Optimated the plot effect of `ov.pl.box_plot`
- Optimated the plot effect of `ov.pl.volcano`
Optimated the plot effect of `ov.pl.violin`
- Added beautiful dotplot than scanpy (#318)
- Added the similar visualization function of CellChat. (#313)

### Space Module
- Added 3D cell-cell interaction analysis in `COMMOT` (#315)

### Single Module
- Fixed the error of pathway_enrichment. (#184)
- Added SCENIC module with GPU-accerlate. (#331) 

### utils Module
- Added scICE to calculate the best cluster (#329)

## v 1.7.6

### LLM Module
- Added `GeneFromer`, `scGPT`, `scFoundation`, `UCE`, `CellPLM` to call directly in OmicVerse.

### Pl Module
- Optimized the visualization effect of embedding.
- Added `ov.pl.umap`, `ov.pl.pca`, `ov.pl.mde`, and `ov.pl.tsne` 


## v 1.7.8

Implemented lazy loading system that reduces `import omicverse` time by **40%** (from ~7.8s to ~4.7s).
Added GPU-accelerated PCA support for Apple Silicon (MLX) and CUDA (TorchDR) devices.
Introduced Smart Agent System with natural language processing for 50+ AI models from 8 providers.
Added and fixed the `anndata-rs` to support million size's datasets (#336)

### PP Module
- Added GPU-accelerated PCA in `ov.pp.pca()` with MLX support for Apple Silicon MPS devices
- Added TorchDR-based PCA acceleration in `ov.pp.pca()` for NVIDIA CUDA devices
- Added smart device detection and automatic backend selection in `init_pca()` and `pca()` functions
- Added graceful fallback to CPU implementation when GPU acceleration fails
- Added enhanced verbose output with device selection information and emoji indicators
- Added optimal component determination based on variance contribution thresholds in `init_pca()`
- Added GPU-accelerated SUDE dimensionality reduction in `ov.pp.sude()` with MLX/CUDA support
- Optimize the `ov.pp.qc` and added ribosome and hb-genes to know more information of data quantity.

### Datasets Module
- Complete elimination of scanpy dependencies for faster loading
- Added dynamo-style dataset framework with comprehensive collection
- Added robust download system with progress tracking and caching
- Added enhanced mock data generation with realistic structure
- Added support for h5ad, loom, xlsx, and compressed formats

### Agent Module
- Added multi-provider LLM support (OpenAI, Anthropic, Google, DeepSeek, Qwen, Moonshot, Grok, Zhipu AI)
- Added natural language processing for both English and Chinese
- Added code generation architecture with local execution
- Added function registry system with multi-language aliases
- Added smart API key management and provider-specific configuration

### Bulk Module
- Added `BayesPrime` and `Scaden` to deconvoluted Bulk RNA-seq's celltype proportion.
- Added `alignment` to alignment the fastq to counts.

### Single Module
- Added `ov.single.Annotation` and `ov.single.AnnotationRef` to annotate the cell type automatically.
- Added `ov.alignment.single` to alignment the scRNA-seq to counts directly.

## v 1.7.9

Implemented **smart lazy loading system** that dramatically reduces `import omicverse` time by **85.6x** (from ~16.57s to ~0.19s).
Enhanced RNA-seq alignment workflow with comprehensive toolkit for FASTQ processing and counting.
Optimized dataset management with nested directory creation for better organization.

### Performance Optimization

**Lazy Loading System**:
- Implemented module-level lazy loading using `__getattr__` mechanism for all major modules
- Added attribute-level lazy loading for frequently-used functions (read, palette, Agent, etc.)
- Introduced intelligent caching system to ensure instant access after first load
- Reduced initial import time from **16.57 seconds to 0.19 seconds** (85.6x speedup)
- Maintained full backward compatibility - all existing code works without modification
- Preserved complete IDE support with tab completion via `__dir__()` implementation
- Fixed circular import issues by delaying settings module initialization
- **MkDocs API documentation generation fully compatible** with lazy loading

**Benefits for Users**:
- ⚡ Instant startup for Jupyter notebooks and scripts
- 🎯 Load only what you use - modules imported on first access
- 💾 Reduced memory footprint for simple tasks
- 🔄 Second access is cached and instant (< 0.001s)

### Alignment Module

**New Comprehensive RNA-seq Alignment Toolkit**:

Added complete end-to-end workflow for processing raw sequencing data:

- **`ov.alignment.prefetch`**: Download SRA datasets from NCBI with built-in retry logic
- **`ov.alignment.fqdump`**: Convert SRA to FASTQ format with parallel processing support
- **`ov.alignment.parallel_fastq_dump`**: High-performance parallel FASTQ extraction
- **`ov.alignment.fastp`**: Quality control and adapter trimming for FASTQ files
- **`ov.alignment.STAR`**: RNA-seq alignment using STAR aligner with customizable parameters
- **`ov.alignment.featureCount`**: Gene-level read counting (renamed from `count` to avoid conflicts)
- **`ov.alignment.single`**: One-command scRNA-seq alignment with kb-python (kallisto|bustools)
- **`ov.alignment.ref`**: Build kallisto|bustools reference index for alignment
- **`ov.alignment.count`**: Quantify gene expression from aligned reads

**Key Features**:
- Unified API for both bulk RNA-seq (STAR + featureCount) and scRNA-seq (kb-python) workflows
- Built-in support for RNA velocity analysis with kb-python
- Parallel processing capabilities for faster data conversion
- Automatic handling of paired-end and single-end reads
- Technology-specific filtering for bulk vs single-cell data
- Integration with SRA toolkit for seamless data download

**Example Workflow**:
```python
# Download and process bulk RNA-seq
ov.alignment.prefetch('SRR1234567', output_dir='./data')
ov.alignment.fqdump('SRR1234567', output_dir='./fastq')
ov.alignment.fastp('sample_1.fastq.gz', 'sample_2.fastq.gz', output_prefix='clean')
ov.alignment.STAR(fastq1='clean_1.fastq.gz', fastq2='clean_2.fastq.gz',
                  genome_dir='./genome', output_prefix='aligned')
ov.alignment.featureCount(bam='aligned.bam', annotation='genes.gtf', output='counts.txt')

# Or use one-command scRNA-seq alignment
ov.alignment.single(
    fastq=['read1.fastq.gz', 'read2.fastq.gz'],
    index='./kb_index',
    output_dir='./kb_output',
    technology='10xv3'
)
```

### PP Module
- Fixed an HVG (Highly Variable Genes) selection issue in `ov.pp.preprocess`
- Improved preprocessing pipeline stability and accuracy
- Refactored PCA implementation to utilize `torch_pca` for GPU acceleration (replacing TorchDR)
- Enhanced support for sparse matrices in PCA computation
- Updated PCA embedding basis from `X_pca` to `PCA` for clarity and consistency
- Improved error handling with try-except blocks in PCA computation
- Fixed PCA GPU mode support with sparse matrices to avoid memory errors

### Single Module
- Added `CONCORD` method to `ov.single.batch_correction` for single-cell data integration
- Enhanced batch correction capabilities with state-of-the-art algorithm
- **Fixed critical performance issue in pySCENIC**: Reverted inefficient correlation calculation optimization that caused memory issues and slowdowns in scRNA-seq data
- Removed misleading warnings about dropout genes in SCENIC correlation calculations
- Restored memory-efficient pairwise correlation computation (prevents OOM with >20k genes)
- SCENIC now uses original approach: calculate correlations only for specific TF-target pairs instead of creating full gene×gene matrices
- Added `ov.single.find_markers` for unified marker gene identification supporting five methods: `cosg`, `t-test`, `t-test_overestim_var`, `wilcoxon`, and `logreg`; statistical methods are natively ported from scanpy with no scanpy runtime dependency and numerically consistent results (rtol=1e-4)
- Added `ov.single.get_markers` to extract top marker genes from results as a `DataFrame` or `dict`, with support for single/multiple cluster filtering and optional filtering by `min_logfoldchange`, `min_score`, and `min_pval_adj`; output includes `pct_group` and `pct_rest` columns showing cell expression proportions within and outside each cluster

### Space Module
- Added `FlashDeconv` for fast, GPU-free deconvolution in Visium spatial transcriptomics
- Added `Banksy` clustering method for spatial domain identification
- Updated spatial analysis documentation with new clustering approaches

### Web Module
- Launched `Omicverse-Notebook` for browser-based interactive analysis without local installation
- Launched `Omicverse-Web` for web-based data analysis without coding requirements
- Democratized bioinformatics analysis for researchers without programming background

### Agent Module
- Enhanced `ov.Agent` with improved natural language processing for data analysis
- Expanded LLM provider support and model selection
- Optimized code generation and execution pipeline

### Pl Module
- Enhanced categorical legend handling for scatterplot embeddings
- Added `legend_loc='on data'` option for direct annotation on plots
- Improved visualization clarity for complex datasets
- Added `ov.pl.markers_dotplot` as a cleaner drop-in for `rank_genes_groups_dotplot` with improved defaults (`standard_scale='var'`, `cmap='Spectral_r'`, `dendrogram=False`)
- Fixed `KeyError` in `rank_genes_groups_df` when cluster names are numeric strings (e.g., leiden `'0'`, `'1'`); now correctly handles structured arrays, DataFrames, and plain 2D arrays from all marker methods

### Datasets Module
- Added comprehensive dataset URLs for easier data access
- Expanded data downloading utilities with progress tracking
- **Fixed dataset download to create nested target directories automatically**
- Improved dataset utilities with better error handling
- Refreshed download behaviors for more reliable data fetching

### Docs
- Strengthened data handling documentation in dotplot and DEG analysis tutorials
- Updated the scTour clustering tutorial with latest best practices
- Added comprehensive release notes for v1.7.9
- Enhanced alignment module documentation with end-to-end workflows

### Bug Fixes
- Resolved circular import issues between `_settings` and `utils` modules
- Fixed compatibility issues with latest package versions (zarr, pandas, etc.)
- Improved error handling in parallel processing functions

### Single Module
**Enhanced DEG Analysis with Expression Percentages**: Added cell expression percentage information to differential expression results

- Added `pct_ctrl` column showing percentage of cells expressing each gene in control group (0-100%)
- Added `pct_test` column showing percentage of cells expressing each gene in test group (0-100%)
- Added `pct_diff` column showing the difference in expression percentage (pct_test - pct_ctrl)
- Works with all DEG methods: `wilcoxon`, `t-test`, and `memento-de`
- Enables better marker gene identification by filtering genes based on expression prevalence
- Similar to dotplot circle size information, helps identify genes with widespread vs. sparse expression patterns

**Example Usage**:
```python
deg_obj = ov.single.DEG(adata, condition='condition',
                        ctrl_group='Control', test_group='Treatment')
deg_obj.run(celltype_key='cell_label', celltype_group=['T_cells'])
results = deg_obj.get_results()
# Now includes pct_ctrl, pct_test, pct_diff columns
```

### Compatibility
**NumPy 2.0 Compatibility**: Fixed all NPY201 compatibility issues to ensure seamless support for both NumPy 1.x and 2.x

**Fixed Issues (31 total)**:

1. **`np.in1d` → `np.isin`** (9 instances)
   - `omicverse/bulk/_dynamicTree.py`: 3 instances (lines 697, 741)
   - `omicverse/single/_cosg.py`: 1 instance (line 77)
   - `omicverse/external/GNTD/_preprocessing.py`: 2 instances
   - `omicverse/external/scdiffusion/guided_diffusion/cell_datasets_WOT.py`: 1 instance
   - Other external modules: 2 instances

2. **`np.row_stack` → `np.vstack`** (13 instances)
   - `omicverse/external/CAST/CAST_Projection.py`: 2 instances
   - `omicverse/external/CAST/visualize.py`: 2 instances
   - `omicverse/external/scSLAT/viz/multi_dataset.py`: multiple instances
   - `omicverse/single/_mdic3.py`: 1 instance

3. **`np.product` → `np.prod`** (4 instances)
   - `omicverse/external/umap_pytorch/model.py`: 2 instances
   - `omicverse/external/umap_pytorch/modules.py`: 2 instances

4. **`np.trapz` compatibility wrapper** (2 instances)
   - Added compatibility wrapper in:
     - `omicverse/external/VIA/plotting_via.py`
     - `omicverse/external/VIA/plotting_via_ov.py`
   - Uses `numpy.trapezoid` (NumPy 2.0+) with fallback to `numpy.trapz` (NumPy 1.x)

**Backward Compatibility**:
- ✅ All changes maintain full backward compatibility with NumPy 1.x (1.13+)
- ✅ `np.isin` available since NumPy 1.13
- ✅ `np.vstack` available in all NumPy versions
- ✅ `np.prod` available in all NumPy versions
- ✅ Custom compatibility wrapper handles `trapz`/`trapezoid` transition

## v 1.7.10

### Scope
- This release note summarizes changes from commit `cd3d151` (version set to `1.7.10rc1`) to current `HEAD`.
- Total code delta in this window: `252 files changed`, `+46,992 / -9,752`.

### Agent & Runtime
- Upgraded `ov.Agent` architecture to modern agentic tool-calling workflows with subagent delegation (v4/v5 evolution).
- Improved GPT-5.2 robustness, response parsing, and backend error handling.
- Added harness runtime components for execution contracts, tool catalog, runtime state, tracing, and cleanup policies.
- Strengthened sandbox behavior with restricted import controls for internal modules.
- Added web bridge and session-level execution improvements for agent workflows.

### New Modules
- Added `omicverse.biocontext` for biomedical knowledge queries via BioContext MCP tooling.
- Added `omicverse.fm` (foundation-model adapters, routing, registry, and API).
- Added structured `omicverse.io` namespaces for general/single/bulk/spatial I/O paths.
- Added `omicverse.jarvis` multi-channel bot framework (Feishu/QQ/Telegram) with bridge support.

### Core OmicVerse Improvements
- Continued enhancements across `pp`, `pl`, `single`, `space`, and `utils` modules.
- Fixed circular import between preprocessing utility internals (`_utils.py` and `_scale.py` path).
- Added/updated function-level metadata and documentation quality in key analysis modules (preprocessing, annotation, trajectory, spatial, datasets, bulk).
- Extended dataset utilities with new signature resources and improved loading pathways.

### Registry & Help System
- Improved registry behavior and module import exposure in package entrypoints.
- Enhanced function/class registration metadata coverage for agent discoverability.
- Registry help generation now better aligns with class constructor documentation in class-based tools.

### Web & UI
- Single-cell analysis UI received iterative upgrades:
  - Better code cell management and undo behavior
  - Improved AnnData slot detail retrieval and display
  - Better DataFrame rendering and integration
  - Plot density/point style control refinements
  - i18n and UX polish for analysis panels
- `omicverse_web` service layer expanded with session-oriented agent service support.

### Developer Experience & Testing
- Added FM test suite and multiple harness/ovagent test modules.
- Removed obsolete legacy-priority and complexity-classifier test paths.
- Added workflow and harness documentation pages for runtime contracts and operational guidance.

### Documentation
- Updated and expanded agent architecture and streaming API docs.
- Updated `t_preprocess_cpu.ipynb` to match latest GPU/version detection behavior.
- Added bilingual and deployment-oriented guidance for Jarvis and agent-related workflows.

