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

### PP Module
- Fixed an HVG issue on `ov.pp.preprocess`.

### Single Module
- Added `CONCORD` to integrate the single-cell data in `ov.single.batch_correction`

### Space Module
- Added `FlashDeconv` to perform deconvolution in Visium profile.
- Added `Banksy` clustering method and update documentation

### Web Module
- Added `Omicverse-Notebook` and `Omicverse-Web` to analysis data without code.

### Agent Module
- Added `ov.Agent` to perform the analysis using LLM.

### Pl Module
- Enhanced categorical legend handling for scatterplot embeddings, including `legend_loc='on data'`.

### Datasets Module
- Added dataset URLs and expanded data downloading utilities.
- Improved dataset utilities and refreshed download behaviors.

### Docs
- Strengthened data handling notes in dotplot and DEG analysis.
- Updated the scTour clustering tutorial and release notes.



