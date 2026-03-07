import time
import numpy as np

from scipy.sparse import issparse, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

from typing import List, Tuple

from ..utils._neighboors import eff_n_jobs, update_rep, W_from_rep

import logging
logger = logging.getLogger(__name__)



def calculate_normalized_affinity(
    W: csr_matrix
) -> Tuple[csr_matrix, np.array, np.array]:
    """
    Symmetrically normalize affinity graph for diffusion-map eigendecomposition.

    Parameters
    ----------
    W : csr_matrix
        Cell-cell affinity/connectivity matrix.

    Returns
    -------
    Tuple[csr_matrix, np.array, np.array]
        Normalized affinity matrix, degree vector, and square-root degree vector.
    """
    diag = W.sum(axis=1).A1
    diag_half = np.sqrt(diag)
    W_norm = W.tocoo(copy=True)
    W_norm.data /= diag_half[W_norm.row]
    W_norm.data /= diag_half[W_norm.col]
    W_norm = W_norm.tocsr()

    return W_norm, diag, diag_half


def calc_von_neumann_entropy(lambdas: List[float], t: float) -> float:
    """
    Compute von Neumann entropy of diffusion operator at diffusion time ``t``.

    Parameters
    ----------
    lambdas : List[float]
        Diffusion eigenvalues (excluding the trivial first component).
    t : float
        Diffusion time.

    Returns
    -------
    float
        Von Neumann entropy value used for knee-point search.
    """
    etas = 1.0 - lambdas ** t
    etas = etas / etas.sum()
    return entropy(etas)


def find_knee_point(x: List[float], y: List[float]) -> int:
    """
    Find elbow/knee index via maximum distance to end-point connecting line.

    Parameters
    ----------
    x : List[float]
        X coordinates (monotonic sequence).
    y : List[float]
        Y coordinates (curve values).

    Returns
    -------
    int
        Index of detected knee point.
    """
    p1 = np.array((x[0], y[0]))
    p2 = np.array((x[-1], y[-1]))
    length_p12 = np.linalg.norm(p2 - p1)

    max_dis = 0.0
    knee = 0
    for cand_knee in range(1, len(x) - 1):
        p3 = np.array((x[cand_knee], y[cand_knee]))
        dis = np.linalg.norm(np.cross(p2 - p1, p3 - p1)) / length_p12
        if max_dis < dis:
            max_dis = dis
            knee = cand_knee

    return knee


def calculate_diffusion_map(
    W: csr_matrix, n_components: int, solver: str, max_t: int, n_jobs: int, random_state: int,
) -> Tuple[np.array, np.array, np.array]:
    """
    Compute diffusion-map coordinates from an affinity graph.

    Parameters
    ----------
    W : csr_matrix
        Cell-cell affinity/connectivity matrix.
    n_components : int
        Number of eigen components to compute (including trivial first one
        before internal removal).
    solver : str
        Eigen solver, either ``'eigsh'`` or ``'randomized'``.
    max_t : int
        Maximum diffusion time for knee-point-based time selection. Use ``-1``
        to apply analytical scaling ``lambda/(1-lambda)``.
    n_jobs : int
        Number of threads used in linear algebra backend.
    random_state : int
        Random seed used by stochastic solvers/initialization.

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        ``(X_diffmap, evals, X_phi)`` where ``X_diffmap`` is pseudo-time-scaled
        diffusion embedding, ``evals`` are non-trivial eigenvalues, and
        ``X_phi`` are normalized eigenvectors.
    """
    from threadpoolctl import threadpool_limits
    assert issparse(W)

    nc, labels = connected_components(W, directed=True, connection="strong")
    logger.info("Calculating connected components is done.")

    assert nc == 1

    W_norm, diag, diag_half = calculate_normalized_affinity(W.astype(np.float64)) # use double precision to guarantee reproducibility
    logger.info("Calculating normalized affinity matrix is done.")

    n_jobs = eff_n_jobs(n_jobs)
    with threadpool_limits(limits = n_jobs):
        if solver == "eigsh":
            np.random.seed(random_state)
            v0 = np.random.uniform(-1.0, 1.0, W_norm.shape[0])
            Lambda, U = eigsh(W_norm, k=n_components, v0=v0)
            Lambda = Lambda[::-1]
            U = U[:, ::-1]
        else:
            assert solver == "randomized"
            U, S, VT = randomized_svd(
                W_norm, n_components=n_components, random_state=random_state
            )
            signs = np.sign((U * VT.transpose()).sum(axis=0))  # get eigenvalue signs
            Lambda = signs * S  # get eigenvalues

    # remove the first eigen value and vector
    Lambda = Lambda[1:]
    U = U[:, 1:]
    Phi = U / diag_half[:, np.newaxis]

    if max_t == -1:
        Lambda_new = Lambda / (1.0 - Lambda)
    else:
        # Find the knee point
        x = np.array(range(1, max_t + 1), dtype = float)
        y = np.array([calc_von_neumann_entropy(Lambda, t) for t in x])
        t = x[find_knee_point(x, y)]
        logger.info("Detected knee point at t = {:.0f}.".format(t))

        # U_df = U * Lambda #symmetric diffusion component
        Lambda_new = Lambda * ((1.0 - Lambda ** t) / (1.0 - Lambda))
    Phi_pt = Phi * Lambda_new  # asym pseudo component

    return Phi_pt, Lambda, Phi  # , U_df, W_norm


def diffmap(
    data,
    n_components: int = 100,
    rep: str = "pca",
    solver: str = "eigsh",
    max_t: float = 5000,
    n_jobs: int = -1,
    random_state: int = 0,
) -> None:
    """
    Compute diffusion-map embedding and store results in AnnData-like object.

    Parameters
    ----------
    data : AnnData-like
        Data object containing representation matrices and graph metadata.
    n_components : int, optional
        Number of diffusion components to compute.
    rep : str, optional
        Representation key used to build/connect graph (for example ``'pca'``).
    solver : str, optional
        Eigensolver method: ``'eigsh'`` or ``'randomized'``.
    max_t : float, optional
        Upper bound for automatic diffusion-time knee search.
    n_jobs : int, optional
        Number of compute threads. ``-1`` uses all available CPU cores.
    random_state : int, optional
        Random seed.

    Returns
    -------
    None
        Updates ``data.obsm['X_diffmap']``, ``data.obsm['X_phi']`` and
        ``data.uns['diffmap_evals']`` in place.

    Examples
    --------
    >>> pg.diffmap(data)
    """

    rep = update_rep(rep)
    Phi_pt, Lambda, Phi = calculate_diffusion_map(
        W_from_rep(data, rep),
        n_components=n_components,
        solver=solver,
        max_t = max_t,
        n_jobs = n_jobs,
        random_state=random_state,
    )

    data.obsm["X_diffmap"] = np.ascontiguousarray(Phi_pt, dtype=np.float32)
    data.uns["diffmap_evals"] = Lambda.astype(np.float32)
    data.obsm["X_phi"] = np.ascontiguousarray(Phi, dtype=np.float32)
    # data.uns['W_norm'] = W_norm
    # data.obsm['X_dmnorm'] = U_df

    # remove previous FLE calculations
    data.uns.pop("diffmap_knn_indices", None)
    data.uns.pop("diffmap_knn_distances", None)
    data.uns.pop("W_diffmap", None)
