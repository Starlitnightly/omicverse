r"""
Monocle2-style trajectory analysis for AnnData.

Pure-Python re-implementation of Monocle 2 (Qiu et al. 2017), exposed as
a single `Monocle` class for convenient use within omicverse.

Examples
--------
>>> import omicverse as ov
>>> mono = ov.single.Monocle(adata)
>>> mono.preprocess()
>>> mono.select_ordering_genes()
>>> mono.reduce_dimension(max_components=2)
>>> mono.order_cells()
>>> mono.plot_trajectory(color_by='clusters')
>>> de_results = mono.differential_gene_test()
>>> beam_results = mono.BEAM(branch_point=1)
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData

# NOTE: `omicverse.single` is a core domain whose modules must not
# perform `from ..external import ...` at the module top level (see
# ``tests/architecture/test_no_top_level_external_imports.py``). The
# Monocle class below imports the monocle2_py backend once inside
# ``__init__`` and stores it on the instance as ``self._m2``.


class Monocle:
    """
    Monocle2-style single-cell trajectory analysis.

    Wraps a pure-Python implementation of Monocle 2 as a stateful analyzer
    operating on an AnnData object. All results are stored in the AnnData
    (``.obs``, ``.var``, ``.uns['monocle']``, ``.obsm``) so the usual scanpy
    workflow continues to work seamlessly.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells × genes). Expression matrix in
        ``adata.X`` should be raw/normalized counts (negative binomial model).

    Attributes
    ----------
    adata : AnnData
        The annotated data matrix with analysis results stored in-place.

    Examples
    --------
    Basic trajectory analysis:

    >>> mono = ov.single.Monocle(adata)
    >>> mono.preprocess()              # size factors + dispersions
    >>> mono.select_ordering_genes()   # high-variance gene selection
    >>> mono.reduce_dimension()        # DDRTree
    >>> mono.order_cells()             # assign pseudotime + State
    >>> mono.plot_trajectory(color_by='clusters')
    >>> mono.plot_genes_in_pseudotime(['Ins1', 'Gcg'])

    Differential expression along pseudotime:

    >>> de = mono.differential_gene_test()
    >>> beam = mono.BEAM(branch_point=1)
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, adata: AnnData):
        # Import the external backend lazily inside the method body.
        # The architecture test forbids this at module scope but
        # allows it inside functions/methods.
        from ..external import monocle2_py as _m2  # noqa: PLC0415
        self._m2 = _m2

        self.adata = adata
        self._preprocessed = False
        self._ordering_set = False
        self._reduced = False
        self._ordered = False

    def __repr__(self):
        status = []
        status.append(f"Monocle({self.adata.n_obs} cells × {self.adata.n_vars} genes)")
        if self._preprocessed:
            status.append("  preprocessed: ✓")
        if self._ordering_set:
            n_ord = int(self.adata.var.get('use_for_ordering', pd.Series([False])).sum())
            status.append(f"  ordering genes: {n_ord}")
        if self._reduced:
            method = self.adata.uns.get('monocle', {}).get('dim_reduce_type', 'unknown')
            status.append(f"  reduced: {method}")
        if self._ordered:
            pt = self.adata.obs.get('Pseudotime')
            if pt is not None:
                n_states = self.adata.obs['State'].nunique() if 'State' in self.adata.obs.columns else 0
                status.append(f"  ordered: pseudotime [{pt.min():.2f}, {pt.max():.2f}], "
                              f"{n_states} states")
        return "\n".join(status)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def detect_genes(self, min_expr: float = 0.1):
        """Detect genes expressed above a threshold."""
        self.adata = self._m2.detect_genes(self.adata, min_expr=min_expr)
        return self

    def estimate_size_factors(self, method: str = 'mean-geometric-mean-total',
                               round_exprs: bool = True):
        """Estimate size factors (matches Monocle2's default method)."""
        self.adata = self._m2.estimate_size_factors(
            self.adata, method=method, round_exprs=round_exprs,
        )
        return self

    def estimate_dispersions(self, min_cells_detected: int = 1, verbose: bool = False):
        """Estimate gene dispersions for the negative-binomial model."""
        self.adata = self._m2.estimate_dispersions(
            self.adata, min_cells_detected=min_cells_detected, verbose=verbose,
        )
        return self

    def preprocess(self, min_expr: float = 0.1, verbose: bool = False):
        """One-shot preprocessing: detect_genes + size factors + dispersions."""
        self.detect_genes(min_expr=min_expr)
        self.estimate_size_factors()
        self.estimate_dispersions(verbose=verbose)
        self._preprocessed = True
        return self

    def dispersion_table(self) -> pd.DataFrame:
        """Return the per-gene dispersion table as a DataFrame."""
        return self._m2.dispersion_table(self.adata)

    def relative2abs(self, method: str = 'num_genes',
                     expected_capture_rate: float = 0.25,
                     verbose: bool = False) -> AnnData:
        """Census normalization (TPM/FPKM → estimated absolute counts)."""
        return self._m2.relative2abs(
            self.adata, method=method,
            expected_capture_rate=expected_capture_rate, verbose=verbose,
        )

    # ------------------------------------------------------------------ #
    # Ordering gene selection
    # ------------------------------------------------------------------ #

    def select_ordering_genes(self, genes: Optional[List[str]] = None,
                               mean_expr_thresh: float = 0.1,
                               max_genes: Optional[int] = None):
        """
        Select genes used for trajectory inference.

        Parameters
        ----------
        genes : list of str or None
            If given, use these genes directly. Otherwise auto-select by
            ``dispersion_empirical > dispersion_fit`` and
            ``mean_expression >= mean_expr_thresh``.
        mean_expr_thresh : float
            Minimum mean expression to keep a gene.
        max_genes : int or None
            Cap the number of ordering genes (for large datasets).
        """
        if genes is None:
            disp = self._m2.dispersion_table(self.adata)
            mask = (
                (disp['mean_expression'] >= mean_expr_thresh) &
                (disp['dispersion_empirical'] >= disp['dispersion_fit'])
            )
            genes = disp[mask].index.tolist()

            if max_genes is not None and len(genes) > max_genes:
                ratio = (disp.loc[genes, 'dispersion_empirical']
                         / disp.loc[genes, 'dispersion_fit'])
                genes = ratio.sort_values(ascending=False).head(max_genes).index.tolist()

        self.adata = self._m2.set_ordering_filter(self.adata, genes)
        self._ordering_set = True
        return self

    def set_ordering_filter(self, genes: List[str]):
        """Explicitly set the list of ordering genes."""
        self.adata = self._m2.set_ordering_filter(self.adata, genes)
        self._ordering_set = True
        return self

    # ------------------------------------------------------------------ #
    # Dimension reduction & ordering
    # ------------------------------------------------------------------ #

    def reduce_dimension(self, max_components: int = 2,
                          reduction_method: str = 'DDRTree',
                          norm_method: str = 'log',
                          method: str = 'fast',
                          verbose: bool = False, **kwargs):
        """
        Reduce dimensionality and learn the principal graph.

        Parameters
        ----------
        max_components : int
            Number of dimensions to reduce to (2 is standard for visualization).
        reduction_method : {'DDRTree', 'tSNE', 'ICA'}
            Algorithm family to use for the reduction.
        norm_method : {'log', 'none'}
            Gene-expression normalisation applied before the reduction.
        method : {'fast', 'exact'}, default 'fast'
            DDRTree convergence mode (ignored for tSNE/ICA).

            * ``'fast'`` (default) — Reformulated update + sparse
              soft-assignment + cheap stopping criterion
              (``||ΔY||_F / ||Y||_F``).  About 3× faster per call;
              trajectory topology and pseudotime correlation with the
              exact mode are preserved (typically 0.99+), but absolute
              pseudotime values may shift slightly.
            * ``'exact'`` — Matches R Monocle 2 bitwise: evaluates the
              full objective (including the expensive ``||X − WZ||_2^2``
              term) on every iteration and terminates when it stops
              decreasing.  Pass this when you need bitwise R parity.
        **kwargs : additional DDRTree parameters
            ``ncenter``, ``lambda_param``, ``param_gamma``, ``sigma``,
            ``maxIter``, ``tol``.
        """
        self.adata = self._m2.reduce_dimension(
            self.adata, max_components=max_components,
            reduction_method=reduction_method,
            norm_method=norm_method, verbose=verbose,
            method=method, **kwargs,
        )
        self._reduced = True
        return self

    def order_cells(self, root_state=None, reverse: Optional[bool] = None,
                    root_by_column: Optional[str] = None,
                    root_by_value=None):
        """Order cells along the learned trajectory, assigning Pseudotime and State.

        Parameters
        ----------
        root_state : int or None
            Numeric state id to use as the trajectory root. If given,
            the cells in this state are used to locate the root tip on
            the Y-centre MST.
        reverse : bool or None
            If ``True``, flip the default diameter-endpoint choice
            (match R's ``orderCells(reverse=TRUE)``).
        root_by_column : str or None
            Name of a column in ``mono.adata.obs``. If given, the
            state with the most cells matching ``root_by_value`` in
            that column is auto-chosen as the root state. This
            replicates R Monocle 2's ``GM_state()`` tutorial helper —
            e.g. for HSMM::

                mono.order_cells()                 # first pass
                mono.order_cells(root_by_column='Hours',
                                 root_by_value=0)  # point root at 0h

            If ``root_by_value`` is ``None``, the minimum value in the
            column is used (so the earliest-observed cells become root).
        root_by_value
            Value within ``root_by_column`` that marks the progenitor
            population. Defaults to the column minimum.
        """
        # Auto-detect root_state from a metadata column if requested.
        # Requires a previous order_cells() call so that `State` exists.
        if root_by_column is not None:
            col = self.adata.obs.get(root_by_column)
            if col is None:
                raise KeyError(
                    f"root_by_column={root_by_column!r} not found in "
                    "adata.obs. Available: "
                    f"{list(self.adata.obs.columns)}"
                )
            if 'State' not in self.adata.obs.columns:
                # Run a first ordering so `State` exists
                self.adata = self._m2.order_cells(self.adata)
                self._ordered = True
            target = root_by_value if root_by_value is not None else col.min()
            mask = col == target
            if not mask.any():
                raise ValueError(
                    f"No cells matched {root_by_column}={target!r}"
                )
            counts = self.adata.obs.loc[mask, 'State'].value_counts()
            if counts.empty:
                raise ValueError("No State values under the selected mask")
            root_state = counts.idxmax()

        self.adata = self._m2.order_cells(
            self.adata, root_state=root_state, reverse=reverse,
        )
        self._ordered = True
        return self

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #

    def cluster_cells(self, method: str = 'leiden', k: int = 50,
                      resolution_parameter: float = 0.1, verbose: bool = False,
                      **kwargs):
        """Cluster cells (Leiden / Louvain / densityPeak / DDRTree)."""
        self.adata = self._m2.cluster_cells(
            self.adata, method=method, k=k,
            resolution_parameter=resolution_parameter, verbose=verbose, **kwargs,
        )
        return self

    @staticmethod
    def cluster_genes(expression_matrix, k: int, method: str = 'correlation'):
        """Cluster genes by their expression pattern."""
        from ..external import monocle2_py as _m2  # noqa: PLC0415
        return _m2.cluster_genes(expression_matrix, k, method=method)

    # ------------------------------------------------------------------ #
    # Differential expression
    # ------------------------------------------------------------------ #

    def differential_gene_test(self,
                                fullModelFormulaStr: str = "~sm.ns(Pseudotime, df=3)",
                                reducedModelFormulaStr: str = "~1",
                                relative_expr: bool = True,
                                cores: int = -1, verbose: bool = False) -> pd.DataFrame:
        """Pseudotime-dependent differential expression test."""
        return self._m2.differential_gene_test(
            self.adata,
            fullModelFormulaStr=fullModelFormulaStr,
            reducedModelFormulaStr=reducedModelFormulaStr,
            relative_expr=relative_expr, cores=cores, verbose=verbose,
        )

    def BEAM(self, branch_point: int = 1, branch_states=None,
              branch_labels=None,
              fullModelFormulaStr: str = "~sm.ns(Pseudotime, df=3)*Branch",
              reducedModelFormulaStr: str = "~sm.ns(Pseudotime, df=3)",
              cores: int = -1, verbose: bool = False) -> pd.DataFrame:
        """Branch Expression Analysis Modeling."""
        return self._m2.BEAM(
            self.adata, branch_point=branch_point, branch_states=branch_states,
            branch_labels=branch_labels,
            fullModelFormulaStr=fullModelFormulaStr,
            reducedModelFormulaStr=reducedModelFormulaStr,
            cores=cores, verbose=verbose,
        )

    def fit_model(self, modelFormulaStr: str = "~sm.ns(Pseudotime, df=3)",
                   relative_expr: bool = True, cores: int = 1):
        """Fit a GLM per gene (used internally by DE tests)."""
        return self._m2.fit_model(self.adata, modelFormulaStr=modelFormulaStr,
                              relative_expr=relative_expr, cores=cores)

    def gen_smooth_curves(self, new_data=None,
                           trend_formula: str = "~sm.ns(Pseudotime, df=3)",
                           relative_expr: bool = True, cores: int = 1):
        """Generate smoothed expression curves along pseudotime."""
        return self._m2.gen_smooth_curves(
            self.adata, new_data=new_data, trend_formula=trend_formula,
            relative_expr=relative_expr, cores=cores,
        )

    def cal_ABCs(self, branch_point: int = 1, **kwargs) -> pd.DataFrame:
        """Calculate Area Between Curves for branch-specific genes."""
        return self._m2.cal_ABCs(self.adata, branch_point=branch_point, **kwargs)

    def cal_ILRs(self, branch_point: int = 1, return_all: bool = False, **kwargs):
        """Calculate Intrinsic Log Ratios (per-gene lineage bias)."""
        return self._m2.cal_ILRs(
            self.adata, branch_point=branch_point, return_all=return_all, **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def plot_trajectory(self, color_by: str = 'State', **kwargs):
        """plot_cell_trajectory — main DDRTree trajectory plot."""
        return self._m2.plot_cell_trajectory(self.adata, color_by=color_by, **kwargs)

    # Alias matching Monocle2 R names
    plot_cell_trajectory = plot_trajectory

    def plot_complex_cell_trajectory(self, color_by: str = 'State', **kwargs):
        """Dendrogram-style trajectory layout (Pseudotime on Y-axis)."""
        return self._m2.plot_complex_cell_trajectory(self.adata, color_by=color_by, **kwargs)

    def plot_cell_clusters(self, color_by: str = 'Cluster', **kwargs):
        """Plot cells colored by cluster in reduced-dim space."""
        return self._m2.plot_cell_clusters(self.adata, color_by=color_by, **kwargs)

    def plot_genes_in_pseudotime(self, genes: List[str], **kwargs):
        """Gene expression vs pseudotime with smoothed curves."""
        return self._m2.plot_genes_in_pseudotime(self.adata, genes=genes, **kwargs)

    def plot_genes_branched_pseudotime(self, genes: List[str], branch_point: int = 1,
                                        **kwargs):
        """Gene expression split by branch."""
        return self._m2.plot_genes_branched_pseudotime(
            self.adata, genes=genes, branch_point=branch_point, **kwargs,
        )

    def plot_genes_branched_heatmap(self, branch_point: int = 1, **kwargs):
        """Heatmap of branch-specific gene expression."""
        return self._m2.plot_genes_branched_heatmap(
            self.adata, branch_point=branch_point, **kwargs,
        )

    def plot_multiple_branches_pseudotime(self, genes: List[str],
                                            branches: List, **kwargs):
        """Multi-branch gene expression curves."""
        return self._m2.plot_multiple_branches_pseudotime(
            self.adata, genes=genes, branches=branches, **kwargs,
        )

    def plot_multiple_branches_heatmap(self, branches: List, **kwargs):
        """Multi-branch expression heatmap."""
        return self._m2.plot_multiple_branches_heatmap(
            self.adata, branches=branches, **kwargs,
        )

    def plot_pseudotime_heatmap(self, genes: Optional[List[str]] = None, **kwargs):
        """Heatmap of gene expression sorted by pseudotime."""
        return self._m2.plot_pseudotime_heatmap(self.adata, genes=genes, **kwargs)

    def plot_genes_jitter(self, genes: List[str], grouping: str = 'State', **kwargs):
        """Jitter plot of gene expression by group."""
        return self._m2.plot_genes_jitter(self.adata, genes=genes, grouping=grouping, **kwargs)

    def plot_genes_violin(self, genes: List[str], grouping: str = 'State', **kwargs):
        """Violin plot of gene expression by group."""
        return self._m2.plot_genes_violin(self.adata, genes=genes, grouping=grouping, **kwargs)

    def plot_ordering_genes(self, **kwargs):
        """Dispersion vs mean-expression plot, highlighting ordering genes."""
        return self._m2.plot_ordering_genes(self.adata, **kwargs)

    def plot_pc_variance_explained(self, max_components: int = 50, **kwargs):
        """Plot variance explained by principal components."""
        return self._m2.plot_pc_variance_explained(
            self.adata, max_components=max_components, **kwargs,
        )

    def plot_rho_delta(self, **kwargs):
        """Plot rho vs delta for density-peak clustering."""
        return self._m2.plot_rho_delta(self.adata, **kwargs)

    # ------------------------------------------------------------------ #
    # Utility accessors
    # ------------------------------------------------------------------ #

    @property
    def pseudotime(self) -> Optional[pd.Series]:
        """Per-cell pseudotime (after order_cells)."""
        return self.adata.obs.get('Pseudotime')

    @property
    def state(self) -> Optional[pd.Series]:
        """Per-cell state (after order_cells)."""
        return self.adata.obs.get('State')

    @property
    def branch_points(self) -> list:
        """Branch-point vertex names in the learned tree."""
        return self.adata.uns.get('monocle', {}).get('branch_points', [])

    @property
    def Z(self) -> Optional[np.ndarray]:
        """Reduced-dim cell coordinates (dim × N)."""
        return self.adata.uns.get('monocle', {}).get('reducedDimS')

    @property
    def Y(self) -> Optional[np.ndarray]:
        """Tree-center coordinates (dim × K)."""
        return self.adata.uns.get('monocle', {}).get('reducedDimK')
