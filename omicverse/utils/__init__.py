r"""
Utility functions for data handling and analysis.

This module provides essential utility functions for data I/O, manipulation,
visualization, clustering, and various computational tasks across the omicverse
ecosystem. It serves as the foundation for many higher-level analysis functions.

Data I/O and manipulation:
    read: Universal data reader for various formats
    read_csv, read_10x_mtx, read_h5ad, read_10x_h5: Format-specific readers
    data_downloader: Download datasets and models
    geneset_prepare: Prepare gene sets for analysis
    get_gene_annotation: Gene annotation utilities
    
Visualization utilities:
    palette: Color palette management
    ov_plot_set, plot_set: Global plot settings
    Various plotting helper functions
    
Clustering and analysis:
    cluster: General clustering functions
    LDA_topic: Topic modeling for single-cell data
    filtered, refine_label: Data filtering and label refinement
    
Statistical and computational tools:
    correlation_pseudotime: Pseudotime correlation analysis
    np_mean, np_std: Numpy-based statistical functions
    neighbors: Neighbor graph construction
    mde: Minimum distortion embedding
    
Data structures and conversion:
    anndata_sparse: Sparse matrix utilities for AnnData
    store_layers, retrieve_layers: Layer management
    
Specialized analysis:
    roe: Ratio of expression analysis
    cal_paga, plot_paga: PAGA trajectory analysis
    venny4py: Venn diagram utilities
    syn: Synthetic data generation
    scatterplot: Advanced scatter plot functions
    knn: K-nearest neighbors utilities
    heatmap: Heatmap generation and customization
    lsi: Latent semantic indexing

Examples:
    >>> import omicverse as ov
    >>> # Data reading
    >>> adata = ov.utils.read('data.h5ad')
    >>> 
    >>> # Clustering
    >>> labels = ov.utils.cluster(
    ...     adata.obsm['X_pca'], 
    ...     method='leiden', 
    ...     resolution=0.5
    ... )
    >>> 
    >>> # Visualization setup
    >>> ov.utils.ov_plot_set()
    >>> 
    >>> # Download data
    >>> ov.utils.data_downloader('pbmc3k')
"""

# All functions imported via wildcard imports from submodules
from ._data import (
    read,read_csv,read_10x_mtx,read_h5ad,read_10x_h5,convert_to_pandas,
    download_CaDRReS_model,download_GDSC_data,
    download_pathway_database,download_geneid_annotation_pair,
    gtf_to_pair_tsv,download_tosica_gmt,geneset_prepare,get_gene_annotation,
    correlation_pseudotime,store_layers,retrieve_layers,easter_egg,
    save,load,convert_adata_for_rust,
    anndata_sparse,np_mean,np_std,
    load_signatures_from_file,predefined_signatures
    )
from ._anndata_rust_patch import patch_rust_adata
from ._plot import (
    plot_set,plotset,ov_plot_set,pyomic_palette,palette,blue_palette,orange_palette,
    red_palette,green_palette,plot_text_set,ticks_range,plot_boxplot,plot_network,
    plot_cellproportion,plot_embedding_celltype,geneset_wordcloud,
    plot_pca_variance_ratio,gen_mpl_labels
)
#from ._genomics import *
from ._mde import mde
from ._syn import logger, pancreas, synthetic_iid, url_datadir
from ._scatterplot import diffmap, draw_graph, embedding, pca, spatial, tsne, umap
from ._knn import weighted_knn_trainer, weighted_knn_transfer
from ._heatmap import (
    additional_colors,
    adjust_palette,
    clip,
    default_color,
    default_palette,
    get_colors,
    interpret_colorkey,
    is_categorical,
    is_list,
    is_list_of_str,
    is_list_or_array,
    is_view,
    make_dense,
    plot_heatmap,
    set_colors_for_categorical_obs,
    strings_to_categoricals,
    to_list,
)
from ._roe import roe, roe_plot_heatmap, transform_roe_values
from ._odds_ratio import odds_ratio, plot_odds_ratio_heatmap
from ._shannon_diversity import shannon_diversity, compare_shannon_diversity, plot_shannon_diversity
from ._resolution import optimal_resolution, plot_resolution_optimization, resolution_stability_analysis
from ._paga import cal_paga,plot_paga,PAGA_tree
from ._cluster import cluster,LDA_topic,filtered,refine_label
from ._venn import venny4py
from ._lsi import Array, lsi, tfidf
from ._neighboors import neighbors,calc_kBET,calc_kSIM

# Import smart_agent module to make it accessible and expose key entrypoints
# Store verifier with a private name first to ensure reference is preserved
from . import agent_backend, smart_agent
from . import verifier as _verifier_module
from .agent_backend import BackendConfig, OmicVerseLLMBackend, Usage
from .smart_agent import Agent, OmicVerseAgent, list_supported_models

# P0-2 / P0-3 / P1-1 / P2-1 / P2-2: New agent infrastructure modules
from .agent_config import AgentConfig, SandboxFallbackPolicy
from .agent_errors import (
    OVAgentError, WorkflowNeedsFallback, ProviderError,
    ConfigError, ExecutionError, SandboxDeniedError,
)
from .agent_reporter import AgentEvent, EventLevel, Reporter, make_reporter
from .context_compactor import ContextCompactor, estimate_tokens
from .session_history import SessionHistory, HistoryEntry
from .mcp_client import MCPClientManager, MCPTool, MCPServerInfo
from .biocontext_bridge import BioContextBridge

# Python 3.10 compatibility: Provide __getattr__ to dynamically return verifier
# This ensures getattr(omicverse.utils, 'verifier') works in unittest.mock.patch
def __getattr__(name):
    """Dynamically return module attributes for Python 3.10 compatibility.

    This is required because unittest.mock.patch uses getattr() to resolve
    module paths, and in Python 3.10 submodule imports don't automatically
    become accessible as attributes of the parent module.
    """
    if name == 'verifier':
        return _verifier_module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Also make verifier accessible via normal attribute access
verifier = _verifier_module

# Explicit public exports for stable, non-wildcard imports
__all__ = [
    # @ _data
    "read",
    "read_csv",
    "read_10x_mtx",
    "read_h5ad",
    "read_10x_h5",
    "convert_to_pandas",
    "download_CaDRReS_model",
    "download_GDSC_data",
    "download_pathway_database",
    "download_geneid_annotation_pair",
    "gtf_to_pair_tsv",
    "download_tosica_gmt",
    "geneset_prepare",
    "get_gene_annotation",
    "correlation_pseudotime",
    "store_layers",
    "retrieve_layers",
    "easter_egg",
    "save",
    "load",
    "convert_adata_for_rust",
    "anndata_sparse",
    "np_mean",
    "np_std",
    "load_signatures_from_file",
    "predefined_signatures",
    # @ _anndata_rust_patch
    "patch_rust_adata",
    # @ _plot
    "plot_set",
    "plotset",
    "ov_plot_set",
    "pyomic_palette",
    "palette",
    "blue_palette",
    "orange_palette",
    "red_palette",
    "green_palette",
    "plot_text_set",
    "ticks_range",
    "plot_boxplot",
    "plot_network",
    "plot_cellproportion",
    "plot_embedding_celltype",
    "geneset_wordcloud",
    "plot_pca_variance_ratio",
    "gen_mpl_labels",
    # @ _mde
    "mde",
    # @ _syn
    "logger",
    "pancreas",
    "synthetic_iid",
    "url_datadir",
    # @ _scatterplot
    "diffmap",
    "draw_graph",
    "embedding",
    "pca",
    "spatial",
    "tsne",
    "umap",
    # @ _knn
    "weighted_knn_trainer",
    "weighted_knn_transfer",
    # @ _heatmap
    "additional_colors",
    "adjust_palette",
    "clip",
    "default_color",
    "default_palette",
    "get_colors",
    "interpret_colorkey",
    "is_categorical",
    "is_list",
    "is_list_of_str",
    "is_list_or_array",
    "is_view",
    "make_dense",
    "plot_heatmap",
    "set_colors_for_categorical_obs",
    "strings_to_categoricals",
    "to_list",
    # @ _roe
    "roe",
    "roe_plot_heatmap",
    "transform_roe_values",
    # @ _odds_ratio
    "odds_ratio",
    "plot_odds_ratio_heatmap",
    # @ _shannon_diversity
    "shannon_diversity",
    "compare_shannon_diversity",
    "plot_shannon_diversity",
    # @ _resolution
    "optimal_resolution",
    "plot_resolution_optimization",
    "resolution_stability_analysis",
    # @ _paga
    "cal_paga",
    "plot_paga",
    "PAGA_tree",
    # @ _cluster
    "cluster",
    "LDA_topic",
    "filtered",
    "refine_label",
    # @ _venn
    "venny4py",
    # @ _lsi
    "Array",
    "lsi",
    "tfidf",
    # @ _neighboors
    "neighbors",
    "calc_kBET",
    "calc_kSIM",
    # @ agent_backend
    "agent_backend",
    "BackendConfig",
    "OmicVerseLLMBackend",
    "Usage",
    # @ smart_agent
    "smart_agent",
    "Agent",
    "OmicVerseAgent",
    "list_supported_models",
    # @ verifier
    "verifier",
    # @ mcp_client
    "MCPClientManager",
    "MCPTool",
    "MCPServerInfo",
    # @ biocontext_bridge
    "BioContextBridge",
]
