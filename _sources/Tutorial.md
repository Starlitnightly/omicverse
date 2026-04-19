# Tutorials

This page is the markdown overview for the tutorial structure defined in `mkdocs.yml`. It is meant to help readers scan the current OmicVerse tutorial map before jumping into notebooks or topic-specific guides.

## Bulk

- [Bulk tutorial index](Tutorials-bulk/index.md)
- Upstream
  - [Bulk RNA-seq alignment with STAR](Tutorials-bulk/t_mapping_STAR.ipynb)
  - [Bulk RNA-seq alignment with kb-python](Tutorials-bulk/t_mapping_kbpython.ipynb)
- Preprocessing
  - [Batch correction in Bulk RNA-seq or microarray data](Tutorials-bulk/t_bulk_combat.ipynb)
- Downstream
  - [Different expression analysis](Tutorials-bulk/t_deg.ipynb)
  - [Different expression analysis with DEseq2](Tutorials-bulk/t_deseq2.ipynb)
  - [Protein-Protein interaction (PPI) analysis by String-db](Tutorials-bulk/t_network.ipynb)
  - [WGCNA (Weighted gene co-expression network analysis) analysis](Tutorials-bulk/t_wgcna.ipynb)
- Deconvolution
  - [Bulk deconvolution with reference scRNA-seq](Tutorials-bulk/t_decov_bulk.ipynb)
- Others
  - [TCGA database preprocess](Tutorials-bulk/t_tcga.ipynb)

## Metabolomics

- [Metabolomics tutorial index](Tutorials-metabol/index.md)
- Preprocessing and univariate analysis
  - [Metabolomics preprocessing and univariate statistics](Tutorials-metabol/t_metabol_01_intro.ipynb)
- Multivariate discrimination
  - [Multivariate discrimination with PLS-DA and OPLS-DA](Tutorials-metabol/t_metabol_02_multivariate.ipynb)
- Pathway enrichment
  - [Metabolite-set enrichment analysis (MSEA)](Tutorials-metabol/t_metabol_03_pathway.ipynb)
- Untargeted LC-MS
  - [Untargeted LC-MS and mummichog pathway inference](Tutorials-metabol/t_metabol_04_untargeted.ipynb)
- Lipidomics
  - [Lipidomics with LIPID MAPS and LION](Tutorials-metabol/t_metabol_05_lipidomics.ipynb)

## Microbiome

- [Microbiome tutorial index](Tutorials-microbiome/index.md)
- Amplicon (16S / ITS / 18S)
  - [16S amplicon pipeline (cutadapt + vsearch UNOISE3 + SINTAX)](Tutorials-microbiome/t_16s_amplicon.ipynb)
  - [16S phylogeny: MAFFT + FastTree → Faith PD + UniFrac](Tutorials-microbiome/t_16s_phylogeny.ipynb)
  - [DADA2 backend (pure-Python via pydada2)](Tutorials-microbiome/t_16s_dada2.ipynb)
  - [Differential abundance: Wilcoxon vs DESeq2 vs ANCOM-BC](Tutorials-microbiome/t_16s_da_comparison.ipynb)

## Single

- [Single-cell tutorial index](Tutorials-single/index.md)
- Alignment
  - [Alignment of single-cell RNA-seq data](Tutorials-single/t_alignment_1k.ipynb)
  - [Alignment of single-cell RNA-seq data for RNA velocity analysis](Tutorials-single/t_alignment_velocity.ipynb)
- Preprocessing
  - [Preprocessing the data of scRNA-seq [CPU]](Tutorials-single/t_preprocess_cpu.ipynb)
  - [Preprocessing the data of scRNA-seq [GPU]](Tutorials-single/t_preprocess_gpu.ipynb)
  - [Preprocessing the data of scRNA-seq [Rust / out-of-memory]](Tutorials-single/t_preprocess_rust.ipynb)
  - [Clustering space](Tutorials-single/t_cluster.ipynb)
  - [Data integration and batch correction](Tutorials-single/t_single_batch.ipynb)
  - [Consensus Non-negative Matrix factorization (cNMF)](Tutorials-single/t_cnmf.ipynb)
  - [Lazy analysis of scRNA-seq](Tutorials-single/t_lazy.ipynb)
- Annotation
  - [Reference-free automated single-cell cell type annotation](Tutorials-single/t_anno_noref.ipynb)
  - [Reference automated single-cell cell type annotation](Tutorials-single/t_anno_ref.ipynb)
  - [Automatic cell type annotation with GPT/Other](Tutorials-single/t_gptanno.ipynb)
  - [Mapping Cell Names to the Cell Ontology](Tutorials-single/t_cellmatch.ipynb)
  - [Celltype auto annotation with SCSA](Tutorials-single/t_cellanno.ipynb)
  - [Celltype auto annotation with MetaTiME](Tutorials-single/t_metatime.ipynb)
  - [Celltype annotation migration(mapping) with TOSICA](Tutorials-single/t_tosica.ipynb)
  - [Celltype auto annotation with scMulan](Tutorials-single/t_scmulan.ipynb)
  - [Consensus annotation with CellVote](Tutorials-single/t_cellvote.md)
- Trajectory
  - [Prediction of absolute developmental potential using CytoTrace2](Tutorials-single/t_cytotrace.ipynb)
  - [Basic Trajectory Inference](Tutorials-single/t_traj.ipynb)
  - [Trajectory Inference with StaVIA](Tutorials-single/t_stavia.ipynb)
  - [Timing-associated genes analysis with TimeFateKernel](Tutorials-single/t_cellfate_gene.ipynb)
  - [Identify the driver regulators of cell fate decisions](Tutorials-single/t_cellfate.ipynb)
- Cell Structure
  - [Inference of MetaCell from Single-Cell RNA-seq](Tutorials-single/t_metacells.ipynb)
  - [Differential expression and celltype analysis [All Cell]](Tutorials-single/t_deg_single.ipynb)
  - [Differential expression analysis [Meta Cell]](Tutorials-single/t_scdeg.ipynb)
  - [Gene Regulatory Network Analysis with SCENIC](Tutorials-single/t_scenic.ipynb)
  - [Pathway analysis with AUCell](Tutorials-single/t_aucell.ipynb)
  - [Cell interaction with CellPhoneDB](Tutorials-single/t_cellphonedb.ipynb)
  - [Drug response predict with scDrug](Tutorials-single/t_scdrug.ipynb)
  - [Batch Correction with SIMBA](Tutorials-single/t_simba.ipynb)
- Velocity
  - [Velocity Basic Calculation](Tutorials-velo/t_velo.ipynb)
  - [Velocity Optimization](Tutorials-velo/t_graphvelo.ipynb)
- Multi-omics
  - [Multi omics analysis by MOFA](Tutorials-single/t_mofa.ipynb)
  - [Multi omics analysis by MOFA and GLUE](Tutorials-single/t_mofa_glue.ipynb)
  - [Celltype annotation transfer in multi-omics](Tutorials-single/t_anno_trans.ipynb)

## Multi-Omics

- [Multi-Omics tutorial index](Tutorials-Multi-Omics/index.md)
- [Bulk ↔ Single-cell ↔ Spatial overview](Tutorials-Multi-Omics/bulk-single/index.md)
- [Bulk RNA-seq generate 'interrupted' cells to interpolate scRNA-seq](Tutorials-Multi-Omics/bulk-single/t_bulktrajblend.ipynb)
- [Bulk RNA-seq to Single RNA-seq](Tutorials-Multi-Omics/bulk-single/t_bulk2single.ipynb)
- [Single RNA-seq to Spatial RNA-seq](Tutorials-Multi-Omics/bulk-single/t_single2spatial.ipynb)

## Space

- [Space tutorial overview](Tutorials-space/index.md)
- Preprocess
  - [Crop and Rotation of spatial transcriptomic data](Tutorials-space/t_crop_rotate.ipynb)
  - [Visium 10x HD Cellpose](Tutorials-space/t_cellpose.ipynb)
  - [Analyze Nanostring data](Tutorials-space/t_nanostring_preprocess.ipynb)
  - [Analyze Xenium data](Tutorials-space/t_xenium_preprocess.ipynb)
  - [Analyze Visium HD data](Tutorials-space/t_visium_hd_preprocess.ipynb)
  - [Spatial clustering and denoising expressions](Tutorials-space/t_cluster_space.ipynb)
  - [Spatial integration and clustering](Tutorials-space/t_staligner.ipynb)
- Deconvolution
  - [Identifying Pseudo-Spatial Map](Tutorials-space/t_spaceflow.ipynb)
  - [Spatial deconvolution with reference scRNA-seq](Tutorials-space/t_decov.ipynb)
  - [FlashDeconv (fast, GPU-free deconvolution)](Tutorials-space/t_flashdeconv.ipynb)
  - [Spatial deconvolution without reference scRNA-seq](Tutorials-space/t_starfysh_new.ipynb)
- Downstream
  - [Spatial transition tensor of single cells](Tutorials-space/t_stt.ipynb)
  - [Spatial Communication](Tutorials-space/t_commot_flowsig.ipynb)
  - [Spatial IsoDepth Calculation](Tutorials-space/t_gaston.ipynb)
  - [Single cell spatial alignment tools](Tutorials-space/t_slat.ipynb)

## Foundation Model

- [Foundation model overview](Tutorials-llm/index.md)
- [ov.fm API Overview](Tutorials-llm/t_fm_guide.md)
- [ov.fm Quick Start](Tutorials-llm/t_fm.ipynb)
- Skill-ready models
  - [scGPT guide](Tutorials-llm/t_fm_scgpt_guide.md) and [tutorial](Tutorials-llm/t_scgpt.ipynb)
  - [Geneformer guide](Tutorials-llm/t_fm_geneformer_guide.md) and [tutorial](Tutorials-llm/t_geneformer.ipynb)
  - [UCE guide](Tutorials-llm/t_fm_uce_guide.md) and [tutorial](Tutorials-llm/t_uce.ipynb)
  - [scFoundation guide](Tutorials-llm/t_fm_scfoundation_guide.md) and [tutorial](Tutorials-llm/t_scfoundation.ipynb)
  - [CellPLM guide](Tutorials-llm/t_fm_cellplm_guide.md) and [tutorial](Tutorials-llm/t_cellplm.ipynb)
- Core and specialized models
  - [scBERT](Tutorials-llm/t_fm_scbert_guide.md)
  - [GeneCompass](Tutorials-llm/t_fm_genecompass_guide.md)
  - [Nicheformer](Tutorials-llm/t_fm_nicheformer_guide.md)
  - [scMulan](Tutorials-llm/t_fm_scmulan_guide.md)
  - [tGPT](Tutorials-llm/t_fm_tgpt_guide.md)
  - [CellFM](Tutorials-llm/t_fm_cellfm_guide.md)
  - [scCello](Tutorials-llm/t_fm_sccello_guide.md)
  - [scPRINT](Tutorials-llm/t_fm_scprint_guide.md)
  - [AIDO.Cell](Tutorials-llm/t_fm_aidocell_guide.md)
  - [PULSAR](Tutorials-llm/t_fm_pulsar_guide.md)
  - [Tabula](Tutorials-llm/t_fm_tabula_guide.md)
  - [ATACformer](Tutorials-llm/t_fm_atacformer_guide.md)
  - [scPlantLLM](Tutorials-llm/t_fm_scplantllm_guide.md)
  - [LangCell](Tutorials-llm/t_fm_langcell_guide.md)
  - [Cell2Sentence](Tutorials-llm/t_fm_cell2sentence_guide.md)
  - [GenePT](Tutorials-llm/t_fm_genept_guide.md)
  - [ChatCell](Tutorials-llm/t_fm_chatcell_guide.md)

## Plotting

- [Plotting tutorial overview](Tutorials-plotting/index.md)
- [Visualization of single cell RNA-seq](Tutorials-plotting/t_visualize_single.ipynb)
- [Visualization of bulk RNA-seq](Tutorials-plotting/t_visualize_bulk.ipynb)
- [Palette optimization for publication-quality single-cell & spatial plots](Tutorials-plotting/t_palette.ipynb)
- [Color system](Tutorials-plotting/t_visualize_colorsystem.ipynb)

## OmicClaw

- Gateway and channels
  - [Overview](Tutorials-jarvis/t_msg_bot_overview.md)
  - [Setup and Auth](Tutorials-jarvis/t_setup_auth.md)
  - [Telegram Tutorial](Tutorials-jarvis/t_channel_telegram.md)
  - [Feishu Tutorial](Tutorials-jarvis/t_channel_feishu.md)
  - [iMessage Tutorial](Tutorials-jarvis/t_channel_imessage.md)
  - [QQ Tutorial](Tutorials-jarvis/t_channel_qq.md)
  - [Session Workflow](Tutorials-jarvis/t_session_commands.md)
  - [Common Issues](Tutorials-jarvis/t_troubleshooting.md)
- MCP server
  - [Overview](Tutorials-llm/t_mcp_guide.md)
  - [Quick Start](Tutorials-llm/t_mcp_quickstart.md)
  - [Full Start](Tutorials-llm/t_mcp_full_start.md)
  - [Tool Catalog](Tutorials-llm/t_mcp_tools.md)
  - [Clients and Deployment](Tutorials-llm/t_mcp_clients.md)
  - [Runtime and Troubleshooting](Tutorials-llm/t_mcp_runtime.md)
  - [Reference](Tutorials-llm/t_mcp_reference.md)
  - [Claude Code Walkthrough](Tutorials-llm/t_mcp_claude_code.md)
- General notebook / pipeline workflows
  - [J.A.R.V.I.S. with PBMC3k](Tutorials-llm/t_ov_agent_pbmc3k.ipynb)
  - [J.A.R.V.I.S. with Ten-Task Suite](Tutorials-llm/ov_agent_ten_task_suite.ipynb)
