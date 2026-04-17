"""
monocle2_py: Pure Python implementation of Monocle2 for single-cell trajectory analysis.

Input: AnnData objects (from scanpy/anndata).
All analysis state is stored in ``adata.obs`` / ``adata.var`` /
``adata.uns['monocle']`` / ``adata.obsm`` — the library never writes
intermediate files to disk.
"""

from .core import (
    set_ordering_filter,
    detect_genes,
    estimate_size_factors,
    estimate_dispersions,
    dispersion_table,
    estimate_t,
    relative2abs,
)
from .dimension_reduction import reduce_dimension
from .ordering import order_cells
from .differential import (
    differential_gene_test,
    BEAM,
    fit_model,
    gen_smooth_curves,
)
from .clustering import cluster_cells, cluster_genes
from .plotting import (
    plot_cell_trajectory,
    plot_genes_in_pseudotime,
    plot_genes_branched_heatmap,
    plot_genes_branched_pseudotime,
    plot_cell_clusters,
    plot_genes_jitter,
    plot_genes_violin,
    plot_ordering_genes,
    plot_pseudotime_heatmap,
    plot_complex_cell_trajectory,
    plot_multiple_branches_pseudotime,
    plot_multiple_branches_heatmap,
    plot_rho_delta,
    plot_pc_variance_explained,
)
from .ddrtree import DDRTree
from .utils import cal_ABCs, cal_ILRs

__version__ = "0.1.0"
__all__ = [
    # core / preprocessing
    "set_ordering_filter",
    "detect_genes",
    "estimate_size_factors",
    "estimate_dispersions",
    "dispersion_table",
    "estimate_t",
    "relative2abs",
    # dimension reduction & ordering
    "reduce_dimension",
    "order_cells",
    # differential expression
    "differential_gene_test",
    "BEAM",
    "fit_model",
    "gen_smooth_curves",
    # clustering
    "cluster_cells",
    "cluster_genes",
    # plotting
    "plot_cell_trajectory",
    "plot_genes_in_pseudotime",
    "plot_genes_branched_heatmap",
    "plot_genes_branched_pseudotime",
    "plot_cell_clusters",
    "plot_genes_jitter",
    "plot_genes_violin",
    "plot_ordering_genes",
    "plot_pseudotime_heatmap",
    "plot_complex_cell_trajectory",
    "plot_multiple_branches_pseudotime",
    "plot_multiple_branches_heatmap",
    "plot_rho_delta",
    "plot_pc_variance_explained",
    # core algorithms & utilities
    "DDRTree",
    "cal_ABCs",
    "cal_ILRs",
]
