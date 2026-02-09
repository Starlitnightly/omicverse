# This file is a modified version of the scrublet.py file from scanpy.
# The only difference is that it allows for the use of GPU and removes scanpy dependency.
# The original file can be found at:
# https://github.com/scverse/scanpy/blob/main/scanpy/preprocessing/_scrublet/core.py
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING
from functools import wraps

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ..external.scrublet import Scrublet
from ..external.scrublet.helper_functions import (
    pipeline_mean_center,
    pipeline_normalize_variance,
    pipeline_zscore,
    pipeline_truncated_svd,
)
from .._settings import EMOJI, Colors, settings
from datetime import datetime

# Import utility functions from pp module
from ._compat import old_positionals
from ._scale import _get_obs_rep
from ._qc import filter_cells, filter_genes
from ._normalization import normalize_total, log1p
from ._highly_variable_genes import highly_variable_genes

if TYPE_CHECKING:
    from typing import Literal
    from numpy.random import RandomState
    _LegacyRandom = int | RandomState | None
    _Metric = str
    _MetricFn = callable

@old_positionals(
    "batch_key",
    "sim_doublet_ratio",
    "expected_doublet_rate",
    "stdev_doublet_rate",
    "synthetic_doublet_umi_subsampling",
    "knn_dist_metric",
    "normalize_variance",
    "log_transform",
    "mean_center",
    "n_prin_comps",
    "use_approx_neighbors",
    "get_doublet_neighbor_parents",
    "n_neighbors",
    "threshold",
    "verbose",
    "copy",
    "random_state",
)
def scrublet(
    adata: AnnData,
    adata_sim: AnnData | None = None,
    *,
    batch_key: str | None = None,
    sim_doublet_ratio: float = 2.0,
    expected_doublet_rate: float = 0.05,
    stdev_doublet_rate: float = 0.02,
    synthetic_doublet_umi_subsampling: float = 1.0,
    knn_dist_metric: _Metric | _MetricFn = "euclidean",
    normalize_variance: bool = True,
    log_transform: bool = False,
    mean_center: bool = True,
    n_prin_comps: int = 30,
    use_approx_neighbors: bool | None = None,
    get_doublet_neighbor_parents: bool = False,
    n_neighbors: int | None = None,
    threshold: float | None = None,
    verbose: bool = True,
    copy: bool = False,
    random_state: _LegacyRandom = 0,
    use_gpu: bool = False,
) -> AnnData | None:
    r"""Predict doublets using Scrublet with optional GPU acceleration.

    Predict cell doublets using a nearest-neighbor classifier of observed
    transcriptomes and simulated doublets. This implementation includes
    GPU acceleration options for improved performance on large datasets.

    Arguments:
        adata: Annotated data matrix of shape n_obs × n_vars
        adata_sim (AnnData): Pre-simulated doublets (default: None)
        batch_key (str): Column name for batch information (default: None)
        sim_doublet_ratio (float): Number of doublets to simulate relative to observed cells (default: 2.0)
        expected_doublet_rate (float): Estimated doublet rate for the experiment (default: 0.05)
        stdev_doublet_rate (float): Uncertainty in expected doublet rate (default: 0.02)
        synthetic_doublet_umi_subsampling (float): UMI sampling rate for synthetic doublets (default: 1.0)
        knn_dist_metric (str): Distance metric for nearest neighbors (default: 'euclidean')
        normalize_variance (bool): Whether to normalize gene variance (default: True)
        log_transform (bool): Whether to log-transform data prior to PCA (default: False)
        mean_center (bool): Whether to center data for PCA (default: True)
        n_prin_comps (int): Number of principal components for embedding (default: 30)
        use_approx_neighbors (bool): Use approximate nearest neighbor search (default: None)
        get_doublet_neighbor_parents (bool): Return parent transcriptomes for doublet neighbors (default: False)
        n_neighbors (int): Number of neighbors for KNN graph (default: None)
        threshold (float): Doublet score threshold for classification (default: None)
        verbose (bool): Whether to log progress updates (default: True)
        copy (bool): Return copy instead of modifying in place (default: False)
        random_state (int): Random seed for reproducibility (default: 0)
        use_gpu (bool): Whether to use GPU acceleration (default: False)

    Returns:
        adata with doublet predictions added if copy=False, otherwise returns modified copy
    """
    #if threshold is None and not find_spec("skimage"):  # pragma: no cover
        # Scrublet.call_doublets requires `skimage` with `threshold=None` but PCA
        # is called early, which is wasteful if there is not `skimage`
        #msg = "threshold is None and thus scrublet requires skimage, but skimage is not installed."
        #raise ValueError(msg)

    if copy:
        adata = adata.copy()

    print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Running Scrublet Doublet Detection:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Mode: {Colors.BOLD}{settings.mode}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Computing doublet prediction using Scrublet algorithm{Colors.ENDC}")
    from ._qc import _is_rust_backend
    is_rust = _is_rust_backend(adata)
    if is_rust:
        adata_obs = adata.to_memory()
    else:
        adata_obs = adata.copy()
    #adata_obs = adata.copy()

    def _run_scrublet(ad_obs: AnnData, ad_sim: AnnData | None = None):
        # With no adata_sim we assume the regular use case, starting with raw
        # counts and simulating doublets

        if ad_sim is None:
            print(f"   {Colors.GREEN}{EMOJI['start']} Filtering genes and cells...{Colors.ENDC}")
            filter_genes(ad_obs, min_cells=3)
            filter_cells(ad_obs, min_genes=3)

            # Doublet simulation will be based on the un-normalised counts, but on the
            # selection of genes following normalisation and variability filtering. So
            # we need to save the raw and subset at the same time.

            print(f"   {Colors.GREEN}{EMOJI['start']} Normalizing data and selecting highly variable genes...{Colors.ENDC}")
            ad_obs.layers["raw"] = ad_obs.X.copy()
            normalize_total(ad_obs)

            # HVG process needs log'd data.
            ad_obs.layers["log1p"] = ad_obs.X.copy()
            log1p(ad_obs, layer="log1p")
            highly_variable_genes(ad_obs, layer="log1p")
            del ad_obs.layers["log1p"]
            ad_obs = ad_obs[:, ad_obs.var["highly_variable"]].copy()

            # Simulate the doublets based on the raw expressions from the normalised
            # and filtered object.

            print(f"   {Colors.GREEN}{EMOJI['start']} Simulating synthetic doublets...{Colors.ENDC}")
            ad_sim = scrublet_simulate_doublets(
                ad_obs,
                layer="raw",
                sim_doublet_ratio=sim_doublet_ratio,
                synthetic_doublet_umi_subsampling=synthetic_doublet_umi_subsampling,
                random_seed=random_state,
            )
            del ad_obs.layers["raw"]
            if log_transform:
                log1p(ad_obs)
                log1p(ad_sim)

            # Now normalise simulated and observed in the same way

            print(f"   {Colors.GREEN}{EMOJI['start']} Normalizing observed and simulated data...{Colors.ENDC}")
            normalize_total(ad_obs, target_sum=1e6)
            normalize_total(ad_sim, target_sum=1e6)

        ad_obs = _scrublet_call_doublets(
            adata_obs=ad_obs,
            adata_sim=ad_sim,
            n_neighbors=n_neighbors,
            expected_doublet_rate=expected_doublet_rate,
            stdev_doublet_rate=stdev_doublet_rate,
            mean_center=mean_center,
            normalize_variance=normalize_variance,
            n_prin_comps=n_prin_comps,
            use_approx_neighbors=use_approx_neighbors,
            knn_dist_metric=knn_dist_metric,
            get_doublet_neighbor_parents=get_doublet_neighbor_parents,
            threshold=threshold,
            random_state=random_state,
            verbose=verbose,
            use_gpu=use_gpu,
        )

        return {"obs": ad_obs.obs, "uns": ad_obs.uns["scrublet"]}

    if batch_key is not None:
        if batch_key not in adata.obs.columns:
            msg = (
                "`batch_key` must be a column of .obs in the input AnnData object,"
                f"but {batch_key!r} is not in {adata.obs.keys()!r}."
            )
            raise ValueError(msg)

        # Run Scrublet independently on batches and return just the
        # scrublet-relevant parts of the objects to add to the input object

        batches = np.unique(adata.obs[batch_key])
        scrubbed = [
            _run_scrublet(
                adata_obs[adata_obs.obs[batch_key] == batch].copy(),
                adata_sim,
            )
            for batch in batches
        ]
        scrubbed_obs = pd.concat([scrub["obs"] for scrub in scrubbed])

        # Now reset the obs to get the scrublet scores

        adata.obs = scrubbed_obs.loc[adata.obs_names.values]

        # Save the .uns from each batch separately

        adata.uns["scrublet"] = {}
        adata.uns["scrublet"]["batches"] = dict(
            zip(batches, [scrub["uns"] for scrub in scrubbed])
        )

        # Record that we've done batched analysis, so e.g. the plotting
        # function knows what to do.

        adata.uns["scrublet"]["batched_by"] = batch_key

    else:
        scrubbed = _run_scrublet(adata_obs, adata_sim)

        # Copy outcomes to input object from our processed version

        adata.obs["doublet_score"] = scrubbed["obs"]["doublet_score"]
        adata.obs["predicted_doublet"] = scrubbed["obs"]["predicted_doublet"]
        adata.uns["scrublet"] = scrubbed["uns"]

    print(f"\n{Colors.GREEN}{EMOJI['done']} Scrublet Analysis Completed Successfully!{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Results added to AnnData object:{Colors.ENDC}")
    print(f"     {Colors.CYAN}• 'doublet_score': {Colors.BOLD}Doublet scores{Colors.ENDC}{Colors.CYAN} (adata.obs){Colors.ENDC}")
    print(f"     {Colors.CYAN}• 'predicted_doublet': {Colors.BOLD}Boolean predictions{Colors.ENDC}{Colors.CYAN} (adata.obs){Colors.ENDC}")
    print(f"     {Colors.CYAN}• 'scrublet': {Colors.BOLD}Parameters and metadata{Colors.ENDC}{Colors.CYAN} (adata.uns){Colors.ENDC}")

    return adata if copy else None


def _scrublet_call_doublets(
    adata_obs: AnnData,
    adata_sim: AnnData,
    *,
    n_neighbors: int | None = None,
    expected_doublet_rate: float = 0.05,
    stdev_doublet_rate: float = 0.02,
    mean_center: bool = True,
    normalize_variance: bool = True,
    n_prin_comps: int = 30,
    use_approx_neighbors: bool | None = None,
    knn_dist_metric: _Metric | _MetricFn = "euclidean",
    get_doublet_neighbor_parents: bool = False,
    threshold: float | None = None,
    random_state: _LegacyRandom = 0,
    verbose: bool = True,
    use_gpu: bool = False,
    ) -> AnnData:
    """Core function for predicting doublets using Scrublet :cite:p:`Wolock2019`.

    Predict cell doublets using a nearest-neighbor classifier of observed
    transcriptomes and simulated doublets.

    Parameters
    ----------
    adata_obs
        The annotated data matrix of shape ``n_obs`` × ``n_vars``. Rows
        correspond to cells and columns to genes. Should be normalised with
        :func:`~scanpy.pp.normalize_total` and filtered to include only highly
        variable genes.
    adata_sim
        Anndata object generated by
        :func:`~scanpy.pp.scrublet_simulate_doublets`, with same number of vars
        as adata_obs. This should have been built from adata_obs after
        filtering genes and cells and selcting highly-variable genes.
    n_neighbors
        Number of neighbors used to construct the KNN graph of observed
        transcriptomes and simulated doublets. If ``None``, this is
        automatically set to ``np.round(0.5 * np.sqrt(n_obs))``.
    expected_doublet_rate
        The estimated doublet rate for the experiment.
    stdev_doublet_rate
        Uncertainty in the expected doublet rate.
    mean_center
        If True, center the data such that each gene has a mean of 0.
        `sklearn.decomposition.PCA` will be used for dimensionality
        reduction.
    normalize_variance
        If True, normalize the data such that each gene has a variance of 1.
        `sklearn.decomposition.TruncatedSVD` will be used for dimensionality
        reduction, unless `mean_center` is True.
    n_prin_comps
        Number of principal components used to embed the transcriptomes prior
        to k-nearest-neighbor graph construction.
    use_approx_neighbors
        Use approximate nearest neighbor method (annoy) for the KNN
        classifier.
    knn_dist_metric
        Distance metric used when finding nearest neighbors. For list of
        valid values, see the documentation for annoy (if `use_approx_neighbors`
        is True) or sklearn.neighbors.NearestNeighbors (if `use_approx_neighbors`
        is False).
    get_doublet_neighbor_parents
        If True, return the parent transcriptomes that generated the
        doublet neighbors of each observed transcriptome. This information can
        be used to infer the cell states that generated a given
        doublet state.
    threshold
        Doublet score threshold for calling a transcriptome a doublet. If
        `None`, this is set automatically by looking for the minimum between
        the two modes of the `doublet_scores_sim_` histogram. It is best
        practice to check the threshold visually using the
        `doublet_scores_sim_` histogram and/or based on co-localization of
        predicted doublets in a 2-D embedding.
    random_state
        Initial state for doublet simulation and nearest neighbors.
    verbose
        If :data:`True`, log progress updates.

    Returns
    -------
    if ``copy=True`` it returns or else adds fields to ``adata``:

    ``.obs['doublet_score']``
        Doublet scores for each observed transcriptome

    ``.obs['predicted_doublets']``
        Boolean indicating predicted doublet status

    ``.uns['scrublet']['doublet_scores_sim']``
        Doublet scores for each simulated doublet transcriptome

    ``.uns['scrublet']['doublet_parents']``
        Pairs of ``.obs_names`` used to generate each simulated doublet transcriptome

    ``.uns['scrublet']['parameters']``
        Dictionary of Scrublet parameters

    """
    # Estimate n_neighbors if not provided, and create scrublet object.

    if n_neighbors is None:
        n_neighbors = int(round(0.5 * np.sqrt(adata_obs.shape[0])))

    # Note: Scrublet() will sparse adata_obs.X if it's not already, but this
    # matrix won't get used if we pre-set the normalised slots.

    scrub = Scrublet(
        adata_obs.X,
        n_neighbors=n_neighbors,
        expected_doublet_rate=expected_doublet_rate,
        stdev_doublet_rate=stdev_doublet_rate,
        random_state=random_state,
    )

    # Ensure normalised matrix sparseness as Scrublet does
    # https://github.com/swolock/scrublet/blob/67f8ecbad14e8e1aa9c89b43dac6638cebe38640/src/scrublet/scrublet.py#L100

    scrub._E_obs_norm = sparse.csc_matrix(adata_obs.X)
    scrub._E_sim_norm = sparse.csc_matrix(adata_sim.X)

    scrub.doublet_parents_ = adata_sim.obsm["doublet_parents"]

    # Call scrublet-specific preprocessing where specified

    if mean_center and normalize_variance:
        pipeline_zscore(scrub)
    elif mean_center:
        pipeline_mean_center(scrub)
    elif normalize_variance:
        pipeline_normalize_variance(scrub)

    # Do PCA. Scrublet fits to the observed matrix and decomposes both observed
    # and simulated based on that fit, so we'll just let it do its thing rather
    # than trying to use Scanpy's PCA wrapper of the same functions.

    if mean_center:
        print(f"   {Colors.GREEN}{EMOJI['start']} Embedding transcriptomes using PCA...{Colors.ENDC}")
        pca_torch(scrub, n_prin_comps=n_prin_comps, random_state=scrub.random_state,use_gpu=use_gpu)
        #pipeline.pca(scrub, n_prin_comps=n_prin_comps, random_state=scrub.random_state)
    else:
        print(f"   {Colors.GREEN}{EMOJI['start']} Embedding transcriptomes using Truncated SVD...{Colors.ENDC}")
        pipeline_truncated_svd(
            scrub, n_prin_comps=n_prin_comps, random_state=scrub.random_state
        )

    # Score the doublets
    print(f"   {Colors.GREEN}{EMOJI['start']} Calculating doublet scores...{Colors.ENDC}")
    scrub.calculate_doublet_scores(
        use_approx_neighbors=use_approx_neighbors,
        distance_metric=knn_dist_metric,
        get_doublet_neighbor_parents=get_doublet_neighbor_parents,
    )

    # Actually call doublets
    print(f"   {Colors.GREEN}{EMOJI['start']} Calling doublets with threshold detection...{Colors.ENDC}")
    scrub.call_doublets(threshold=threshold, verbose=False)  # Suppress scrublet's output
    
    # Display formatted scrublet results
    if hasattr(scrub, 'threshold_'):
        print(f"   {Colors.CYAN}📊 Automatic threshold: {Colors.BOLD}{scrub.threshold_:.3f}{Colors.ENDC}")
        
        # Calculate statistics
        n_doublets_detected = sum(scrub.predicted_doublets_)
        detection_rate = n_doublets_detected / len(scrub.predicted_doublets_) * 100
        print(f"   {Colors.CYAN}📈 Detected doublet rate: {Colors.BOLD}{detection_rate:.1f}%{Colors.ENDC}")
        
        # Show expected vs estimated
        expected_rate = expected_doublet_rate * 100
        if hasattr(scrub, 'doublet_scores_sim_') and len(scrub.doublet_scores_sim_) > 0:
            # Estimate detectable fraction (simplified calculation)
            sim_scores_above_threshold = sum(scrub.doublet_scores_sim_ > scrub.threshold_)
            detectable_fraction = sim_scores_above_threshold / len(scrub.doublet_scores_sim_) * 100
            estimated_rate = detection_rate / (detectable_fraction / 100) if detectable_fraction > 0 else detection_rate
            
            print(f"   {Colors.CYAN}🔍 Detectable doublet fraction: {Colors.BOLD}{detectable_fraction:.1f}%{Colors.ENDC}")
            print(f"   {Colors.BLUE}📊 Overall doublet rate comparison:{Colors.ENDC}")
            print(f"     {Colors.CYAN}• Expected: {Colors.BOLD}{expected_rate:.1f}%{Colors.ENDC}")
            print(f"     {Colors.CYAN}• Estimated: {Colors.BOLD}{estimated_rate:.1f}%{Colors.ENDC}")
    else:
        print(f"   {Colors.WARNING}⚠️ Could not determine automatic threshold - manual threshold may be needed{Colors.ENDC}")

    # Store results in AnnData for return

    adata_obs.obs["doublet_score"] = scrub.doublet_scores_obs_

    # Store doublet Scrublet metadata

    adata_obs.uns["scrublet"] = {
        "doublet_scores_sim": scrub.doublet_scores_sim_,
        "doublet_parents": adata_sim.obsm["doublet_parents"],
        "parameters": {
            "expected_doublet_rate": expected_doublet_rate,
            "sim_doublet_ratio": (
                adata_sim.uns.get("scrublet", {})
                .get("parameters", {})
                .get("sim_doublet_ratio", None)
            ),
            "n_neighbors": n_neighbors,
            "random_state": random_state,
        },
    }

    # If threshold hasn't been located successfully then we couldn't make any
    # predictions. The user will get a warning from Scrublet, but we need to
    # set the boolean so that any downstream filtering on
    # predicted_doublet=False doesn't incorrectly filter cells. The user can
    # still use this object to generate the plot and derive a threshold
    # manually.

    if hasattr(scrub, "threshold_"):
        adata_obs.uns["scrublet"]["threshold"] = scrub.threshold_
        adata_obs.obs["predicted_doublet"] = scrub.predicted_doublets_
    else:
        adata_obs.obs["predicted_doublet"] = False

    if get_doublet_neighbor_parents:
        adata_obs.uns["scrublet"]["doublet_neighbor_parents"] = (
            scrub.doublet_neighbor_parents_
        )

    return adata_obs


@old_positionals(
    "layer", "sim_doublet_ratio", "synthetic_doublet_umi_subsampling", "random_seed"
)
def scrublet_simulate_doublets(
    adata: AnnData,
    *,
    layer: str | None = None,
    sim_doublet_ratio: float = 2.0,
    synthetic_doublet_umi_subsampling: float = 1.0,
    random_seed: _LegacyRandom = 0,
) -> AnnData:
    r"""Simulate doublets by adding counts of random observed transcriptome pairs.

    Generate synthetic doublets by randomly selecting pairs of observed cells
    and combining their transcriptomes to create artificial doublet profiles
    for training the doublet detection classifier.

    Arguments:
        adata: Annotated data matrix of shape n_obs × n_vars
        layer (str): Layer containing raw values, or None to use .X (default: None)
        sim_doublet_ratio (float): Number of doublets to simulate relative to observed cells (default: 2.0)
        synthetic_doublet_umi_subsampling (float): UMI sampling rate for doublet creation (default: 1.0)
        random_seed (int): Random seed for reproducible doublet simulation (default: 0)

    Returns:
        adata: AnnData object containing simulated doublets with metadata
    """
    X = _get_obs_rep(adata, layer=layer)
    scrub = Scrublet(X, random_state=random_seed)

    scrub.simulate_doublets(
        sim_doublet_ratio=sim_doublet_ratio,
        synthetic_doublet_umi_subsampling=synthetic_doublet_umi_subsampling,
    )

    adata_sim = AnnData(scrub._E_sim)
    adata_sim.obs["n_counts"] = scrub._total_counts_sim
    adata_sim.obsm["doublet_parents"] = scrub.doublet_parents_
    adata_sim.uns["scrublet"] = {"parameters": {"sim_doublet_ratio": sim_doublet_ratio}}
    return adata_sim

def pca_torch(
    self: Scrublet,
    n_prin_comps: int = 50,
    *,
    random_state: _LegacyRandom = 0,
    svd_solver: Literal["auto", "full", "arpack", "randomized","gesvd", "gesvdj", "gesvda"] = "auto",
    use_gpu: bool = False,
) -> None:
    if self._E_sim_norm is None:
        msg = "_E_sim_norm is not set"
        raise RuntimeError(msg)

    if use_gpu:
        # For GPU mode, keep sparse matrices as-is since torch_pca supports them
        X_obs = self._E_obs_norm
        X_sim = self._E_sim_norm
        if svd_solver == "auto":
            svd_solver = "gesvd"
        
        import torch
        from .._settings import get_optimal_device, prepare_data_for_device
        device = get_optimal_device(prefer_gpu=True, verbose=True)
        
        # Use MLX for MPS devices (Apple Silicon optimization)
        if device.type == 'mps':
            try:
                from ._pca_mlx import MLXPCA, MockPCA
                print(f"   {Colors.GREEN}{EMOJI['gpu']} Using MLX PCA for Apple Silicon MPS acceleration in scrublet{Colors.ENDC}")
                print(f"   {Colors.GREEN}{EMOJI['gpu']} MLX PCA backend: Apple Silicon GPU acceleration (scrublet){Colors.ENDC}")
                
                # Create MLX PCA instance (use "metal" for MLX)
                mlx_pca = MLXPCA(n_components=n_prin_comps, device="metal")
                
                # Fit and transform
                X_obs_transformed = mlx_pca.fit_transform(X_obs)
                X_sim_transformed = mlx_pca.transform(X_sim)
                
                # Create a mock PCA object with sklearn-compatible interface
                pca = MockPCA(mlx_pca)
                # Set manifold directly since we already have transformed data
                self.set_manifold(X_obs_transformed, X_sim_transformed)
                
            except (ImportError, Exception) as e:
                print(f"   {EMOJI['warning']} {Colors.WARNING}MLX PCA failed in scrublet ({str(e)}), falling back to sklearn for MPS device{Colors.ENDC}")
                print(f"   {EMOJI['warning']} {Colors.WARNING}sklearn PCA backend: MPS device fallback (scrublet){Colors.ENDC}")
                # For MPS devices, fall back to sklearn instead of TorchDR
                from sklearn.decomposition import PCA
                
                pca = PCA(
                    n_components=n_prin_comps, random_state=random_state, svd_solver="arpack",
                ).fit(X_obs)
                self.set_manifold(pca.transform(X_obs), pca.transform(X_sim))
        else:
            # Use torch_pca for non-MPS GPU devices (CUDA, etc.)
            print(f"   {Colors.GREEN}{EMOJI['gpu']} Using torch_pca PCA for {device.type.upper()} GPU acceleration in scrublet{Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['gpu']} torch_pca PCA backend: {device.type.upper()} GPU acceleration (scrublet, supports sparse){Colors.ENDC}")

            try:
                from ..external.torch_pca import PCA
            except ImportError:
                raise ImportError("torch_pca is not available. Please check the installation.")

            # Prepare data for GPU compatibility (float32 requirement)
            X_obs = prepare_data_for_device(X_obs, device, verbose=True)
            X_sim = prepare_data_for_device(X_sim, device, verbose=True)

            # Print input data type information
            print(f"   {Colors.CYAN}📊 Scrublet PCA input data type - X_obs: {type(X_obs).__name__}, shape: {X_obs.shape}, dtype: {X_obs.dtype}{Colors.ENDC}")
            print(f"   {Colors.CYAN}📊 Scrublet PCA input data type - X_sim: {type(X_sim).__name__}, shape: {X_sim.shape}, dtype: {X_sim.dtype}{Colors.ENDC}")
            if sparse.issparse(X_obs):
                print(f"   {Colors.CYAN}📊 X_obs sparse density: {X_obs.nnz / (X_obs.shape[0] * X_obs.shape[1]) * 100:.2f}%{Colors.ENDC}")
            if sparse.issparse(X_sim):
                print(f"   {Colors.CYAN}📊 X_sim sparse density: {X_sim.nnz / (X_sim.shape[0] * X_sim.shape[1]) * 100:.2f}%{Colors.ENDC}")

            # Map svd_solver to torch_pca compatible values
            if svd_solver == "auto":
                svd_solver_mapped = "auto"
            elif svd_solver in ["gesvd", "gesvdj", "gesvda"]:
                svd_solver_mapped = "full"
            else:
                svd_solver_mapped = "auto"

            # torch_pca supports sparse matrices natively - no need to convert to dense!
            if sparse.issparse(X_obs):
                # For sparse matrices, use ARPACK solver
                pca = PCA(n_components=n_prin_comps, svd_solver='arpack', random_state=random_state)
                pca.fit(X_obs)
                X_obs_transformed = pca.transform(X_obs)
                X_sim_transformed = pca.transform(X_sim)

                # Convert to numpy if tensor
                if hasattr(X_obs_transformed, 'cpu'):
                    X_obs_transformed = X_obs_transformed.cpu().numpy()
                if hasattr(X_sim_transformed, 'cpu'):
                    X_sim_transformed = X_sim_transformed.cpu().numpy()
            else:
                # For dense arrays, convert to torch tensor and move to GPU
                X_obs_torch = torch.from_numpy(np.asarray(X_obs)).to(device)
                X_sim_torch = torch.from_numpy(np.asarray(X_sim)).to(device)

                pca = PCA(n_components=n_prin_comps, svd_solver=svd_solver_mapped, random_state=random_state)
                pca.fit(X_obs_torch)
                # Move PCA model to GPU
                pca.to(device)
                X_obs_transformed = pca.transform(X_obs_torch)
                X_sim_transformed = pca.transform(X_sim_torch)

                # Convert torch tensors back to numpy arrays
                if hasattr(X_obs_transformed, 'cpu'):
                    X_obs_transformed = X_obs_transformed.cpu().numpy()
                if hasattr(X_sim_transformed, 'cpu'):
                    X_sim_transformed = X_sim_transformed.cpu().numpy()

            self.set_manifold(X_obs_transformed, X_sim_transformed)
    else:
        # For CPU mode, convert sparse matrices to dense for sklearn PCA
        if sparse.issparse(self._E_obs_norm):
            X_obs = self._E_obs_norm.toarray()
        else:
            X_obs = np.asarray(self._E_obs_norm)

        if sparse.issparse(self._E_sim_norm):
            X_sim = self._E_sim_norm.toarray()
        else:
            X_sim = np.asarray(self._E_sim_norm)

        # Print input data type information for CPU mode
        print(f"   {Colors.CYAN}📊 Scrublet PCA input data type (CPU) - X_obs: {type(X_obs).__name__}, shape: {X_obs.shape}, dtype: {X_obs.dtype}{Colors.ENDC}")
        print(f"   {Colors.CYAN}📊 Scrublet PCA input data type (CPU) - X_sim: {type(X_sim).__name__}, shape: {X_sim.shape}, dtype: {X_sim.dtype}{Colors.ENDC}")

        from sklearn.decomposition import PCA
        if svd_solver == "auto":
            svd_solver = "arpack"
        pca = PCA(
            n_components=n_prin_comps, random_state=random_state, svd_solver=svd_solver,
        ).fit(X_obs)
        self.set_manifold(pca.transform(X_obs), pca.transform(X_sim))

