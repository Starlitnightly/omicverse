r"""
Single-cell omics analysis utilities.

This module provides comprehensive tools for single-cell RNA-seq analysis including:
- Cell type annotation and automated identification
- Trajectory inference and pseudotime analysis
- Gene regulatory network analysis
- Cell-cell communication analysis
- Multi-modal integration (RNA + ATAC)
- Drug response prediction
- Pathway enrichment and functional analysis

Key classes:
    pyMOFA: Multi-Omics Factor Analysis
    pyVIA: Velocity and pseudotime analysis 
    pySIMBA: Single-cell integration and batch alignment
    pyTOSICA: Trajectory inference and cell fate analysis
    pyCEFCON: Cell-cell communication analysis
    MetaCell: Metacell construction and analysis
    pyDEG: Single-cell differential expression

Annotation tools:
    pySCSA: Automated cell type annotation
    MetaTiME: Tumor microenvironment annotation
    gptcelltype: GPT-based cell type identification
    CellVote: Consensus cell type annotation

Trajectory analysis:
    TrajInfer: Trajectory inference framework
    scLTNN: Lineage tracing with neural networks
    cytotrace2: Developmental potential scoring

Examples:
    >>> import omicverse as ov
    >>> # Automated cell annotation
    >>> ov.single.pySCSA(adata, tissue='lung', species='human')
    >>> 
    >>> # Trajectory analysis
    >>> via = ov.single.pyVIA(adata)
    >>> via.run_via()
    >>> 
    >>> # Multi-omics integration
    >>> mofa = ov.single.pyMOFA(data_dict)
    >>> mofa.build_mofa()
"""

# Heavy functionality lives in submodules and is imported lazily.

# Core gene selection and differential expression
from ._cosg import cosg
from ._anno import pySCSA,MetaTiME,scanpy_lazy,scanpy_cellanno_from_dict,get_celltype_marker
from ._nocd import scnocd
from ._mofa import (
    pyMOFAART,pyMOFA,GLUE_pair,
    factor_exact,factor_correlation,
    get_weights,glue_pair,get_r2_from_hdf5,
    convert_r2_to_matrix,factor_group_correlation_mdata,
    plot_factor_group_associations,plot_factor_boxplots,
    plot_factors_violin,plot_weights,
)
from ._scdrug import (
    autoResolution,writeGEP,Drug_Response
)
from ._cpdb import (
    cpdb_network_cal,cpdb_plot_network,
    cpdb_plot_interaction,
    cpdb_interaction_filtered,
    cpdb_submeans_exacted,cpdb_exact_target,
    cpdb_exact_source,cellphonedb_v5,
    run_cellphonedb_v5
)

from ._scgsea import (
    geneset_aucell,pathway_aucell,pathway_aucell_enrichment,
    geneset_aucell_tmp,pathway_aucell_tmp,pathway_aucell_enrichment_tmp,
    pathway_enrichment,pathway_enrichment_plot
)
#from ._via import pyVIA,scRNA_hematopoiesis
from ._simba import pySIMBA
from ._tosica import pyTOSICA
from ._atac import atac_concat_get_index,atac_concat_inner,atac_concat_outer
from ._batch import batch_correction
from ._cellfategenie import Fate,gene_trends,mellon_density
from ._ltnn import scLTNN,plot_origin_tesmination,find_related_gene
from ._traj import TrajInfer,fle
from ._diffusionmap import diffmap
from ._cefcon import pyCEFCON,convert_human_to_mouse_network,load_human_prior_interaction_network,mouse_hsc_nestorowa16
from ._aucell import aucell
from ._metacell import MetaCell,plot_metacells,get_obs_value
from ._mdic3 import pyMDIC3
from ._cnmf import *
from ._gptcelltype import gptcelltype,gpt4celltype,get_cluster_celltype
from ._cytotrace2 import cytotrace2
from ._gptcelltype_local import gptcelltype_local
from ._sccaf import SCCAF_assessment,plot_roc,SCCAF_optimize_all,color_long
from ._multimap import TFIDF_LSI,Wrapper,Integration,Batch
from ._scdiffusion import scDiffusion
from ._cellvote import get_cluster_celltype,CellVote
from ._deg_ct import DCT,DEG
from ._lazy_function import lazy
from ._lazy_report import generate_scRNA_report
from ._lazy_checkpoint import lazy_checkpoint, resume_from_checkpoint, list_checkpoints, cleanup_checkpoints
from ._lazy_step_by_step import (
    lazy_step_qc, lazy_step_preprocess, lazy_step_scale, lazy_step_pca,
    lazy_step_cell_cycle, lazy_step_harmony, lazy_step_scvi, 
    lazy_step_select_best_method, lazy_step_mde, lazy_step_clustering,
    lazy_step_final_embeddings, lazy_step_by_step_guide
)
from ._diffusionmap import diffmap
from ._cellmatch import CellOntologyMapper,download_cl
from ._scenic import SCENIC,build_correlation_network_umap_layout,add_tf_regulation,plot_grn
from ._annotation import Annotation
from ._annotation_ref import AnnotationRef
from ._velo import Velo,velocity_embedding
from ._milo_dev import Milo


__all__ = [
    # Core analysis functions
    'cosg',
    'lazy',
    'lazy_checkpoint', 
    'resume_from_checkpoint',
    'list_checkpoints',
    'cleanup_checkpoints',
    'lazy_step_qc',
    'lazy_step_preprocess',
    'lazy_step_scale',
    'lazy_step_pca',
    'lazy_step_cell_cycle',
    'lazy_step_harmony',
    'lazy_step_scvi',
    'lazy_step_select_best_method',
    'lazy_step_mde',
    'lazy_step_clustering',
    'lazy_step_final_embeddings',
    'lazy_step_by_step_guide',
    'aucell',
    
    # Cell type annotation
    'pySCSA',
    'MetaTiME', 
    'scanpy_lazy',
    'scanpy_cellanno_from_dict',
    'get_celltype_marker',
    'gptcelltype',
    'gpt4celltype',
    'get_cluster_celltype',
    'gptcelltype_local',
    'CellVote',
    'CellOntologyMapper',
    'download_cl',
    
    # Multi-omics integration
    'pyMOFAART',
    'pyMOFA',
    'GLUE_pair',
    'factor_exact',
    'factor_correlation',
    'get_weights',
    'glue_pair',
    'get_r2_from_hdf5',
    'pySIMBA',
    'pyTOSICA',
    'TFIDF_LSI',
    'Wrapper',
    'Integration',
    'Batch',
    'convert_r2_to_matrix',
    'factor_group_correlation_mdata',
    'plot_factor_group_associations',
    'plot_factor_boxplots',
    'plot_factors_violin',
    'plot_weights',
    
    # Trajectory and pseudotime analysis
    #'pyVIA',
    #'scRNA_hematopoiesis',
    'TrajInfer',
    'fle',
    'diffmap',
    'scLTNN',
    'plot_origin_tesmination',
    'find_related_gene',
    'cytotrace2',
    
    # Cell fate and development
    'Fate',
    'gene_trends',
    'mellon_density',
    
    # Network and communication analysis
    'scnocd',
    'pyCEFCON',
    'convert_human_to_mouse_network',
    'load_human_prior_interaction_network',
    'mouse_hsc_nestorowa16',
    'cpdb_network_cal',
    'cpdb_plot_network',
    'cpdb_plot_interaction',
    'cpdb_interaction_filtered',
    'cpdb_submeans_exacted',
    'cpdb_exact_target',
    'cpdb_exact_source',
    'cellphonedb_v5',
    'run_cellphonedb_v5',
    # Pathway and functional analysis
    'geneset_aucell',
    'pathway_aucell',
    'pathway_aucell_enrichment',
    'geneset_aucell_tmp',
    'pathway_aucell_tmp',
    'pathway_aucell_enrichment_tmp',
    'pathway_enrichment',
    'pathway_enrichment_plot',
    
    # Drug response analysis
    'autoResolution',
    'writeGEP',
    'Drug_Response',
    'scDiffusion',
    
    # ATAC-seq analysis
    'atac_concat_get_index',
    'atac_concat_inner',
    'atac_concat_outer',
    
    # Batch correction and preprocessing  
    'batch_correction',
    
    # Quality control and assessment
    'SCCAF_assessment',
    'plot_roc',
    'SCCAF_optimize_all',
    'color_long',
    
    # Metacells and aggregation
    'MetaCell',
    'plot_metacells',
    'get_obs_value',
    
    # Differential expression
    'DCT',
    'DEG',
    'pyMDIC3',
    
    # Additional analysis tools  
    'cnmf',
    'generate_scRNA_report',
    'SCENIC',
    'build_correlation_network_umap_layout',
    'add_tf_regulation',
    'plot_grn',
    'Velo',
    'velocity_embedding',
    'Annotation',   # cell type annotation
    'AnnotationRef', # cell type annotation with reference
    'Milo',
]

