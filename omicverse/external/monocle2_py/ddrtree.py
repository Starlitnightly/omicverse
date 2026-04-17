"""
DDRTree: Discriminative Dimensionality Reduction via Learning a Tree.

Pure Python/NumPy/SciPy implementation of the DDRTree algorithm,
faithfully ported from the R/C++ DDRTree package.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans


def _sqdist(a, b):
    """Compute squared distance matrix between columns of a and b.
    a: (D, N), b: (D, M) -> returns (N, M)
    """
    aa = np.sum(a ** 2, axis=0)
    bb = np.sum(b ** 2, axis=0)
    ab = a.T @ b
    dist = np.abs(aa[:, None] + bb[None, :] - 2 * ab)
    return dist


def _pca_projection_irlba_like(C, L):
    """Iterative top-L eigenvectors of symmetric matrix C — matches R's irlba.

    R's DDRTree C++ code calls pca_projection_R (which uses irlba) inside
    every iteration. The approximation introduces a small random-direction
    perturbation that helps DDRTree escape smooth local minima and
    discover branching structure that exact eigendecomposition can miss.
    """
    from scipy.stats import norm as _norm
    D = C.shape[0]
    if L >= min(C.shape):
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx[:L]]
    v0 = _norm.ppf(np.arange(1, D + 1) / (D + 1))
    v0 = v0 / np.linalg.norm(v0)
    try:
        eigenvalues, eigenvectors = eigsh(C, k=L, which='LM', v0=v0,
                                          maxiter=1000, tol=1e-5)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx]
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx[:L]]


def _pca_projection(C, L):
    """PCA projection: return top-L eigenvectors of symmetric matrix C.

    Uses exact np.linalg.eigh when C is small enough (more accurate than
    iterative eigsh), otherwise falls back to eigsh with R-style initial
    vector to match R's irlba.

    C: (D, D), L: int -> returns (D, L)
    """
    from scipy.stats import norm as _norm

    D = C.shape[0]

    # Use exact eigendecomposition for matrices that fit in memory
    # Memory: D*D*8 bytes, 4000x4000 = 128MB is fine
    if L >= min(C.shape) or D <= 5000:
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx[:L]]
    else:
        # Match R: initial_v <- qnorm(1:(ncol(C)+1) / (ncol(C)+1))[1:ncol(C)]
        v0 = _norm.ppf(np.arange(1, D + 1) / (D + 1))
        v0 = v0 / np.linalg.norm(v0)

        eigenvalues, eigenvectors = eigsh(C, k=L, which='LM', v0=v0,
                                          maxiter=1000)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx]


def _get_major_eigenvalue(C, L):
    """Get major eigenvalue of matrix C.
    R returns max(abs(irlba(C)$v)) which is max abs of right singular vectors.
    For our purposes, we compute the largest singular value.
    """
    D, N = C.shape
    if L >= min(D, N):
        return np.linalg.norm(C, ord=2) ** 2
    else:
        # Use small matrix trick: singular values of C (D×N) are sqrt of
        # eigenvalues of C^T C (N×N)
        if N <= D:
            small = C.T @ C  # N×N
            eigenvalues = np.linalg.eigvalsh(small)
            return np.max(np.abs(eigenvalues))
        else:
            small = C @ C.T  # D×D
            eigenvalues = np.linalg.eigvalsh(small)
            return np.max(np.abs(eigenvalues))


def DDRTree(X, dimensions=2, initial_method=None, maxIter=20, sigma=0.001,
            lambda_param=None, ncenter=None, param_gamma=10, tol=0.001,
            verbose=False, pca_method='irlba'):
    """
    Perform DDRTree dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray, shape (D, N)
        Input data matrix (genes x cells), already preprocessed.
    dimensions : int
        Number of dimensions to reduce to.
    initial_method : callable or None
        Custom initialization method. If None, uses PCA.
    maxIter : int
        Maximum number of iterations.
    sigma : float
        Bandwidth parameter for soft assignment.
    lambda_param : float or None
        Regularization parameter. If None, set to 5*N.
    ncenter : int or None
        Number of centers. If None, K=N (all cells are centers).
    param_gamma : float
        Gamma parameter controlling the trade-off.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict with keys:
        W : (D, dimensions) projection matrix
        Z : (dimensions, N) reduced coordinates of cells
        Y : (dimensions, K) center positions in reduced space
        stree : (K, K) sparse MST adjacency (weighted)
        objective_vals : list of objective function values
    """
    D, N = X.shape

    # PCA initialization for W
    W = _pca_projection(X @ X.T, dimensions)

    # Initialize Z
    if initial_method is None:
        Z = W.T @ X
    else:
        tmp = initial_method(X)
        Z = tmp[:, :dimensions].T

    # Initialize Y (centers)
    if ncenter is None:
        K = N
        Y = Z.copy()
    else:
        K = ncenter
        if K > Z.shape[1]:
            raise ValueError("ncenter must be <= number of cells")
        # Sample evenly spaced points as initial centers
        indices = np.linspace(0, Z.shape[1] - 1, K, dtype=int)
        centers = Z[:, indices].T
        kmeans = KMeans(n_clusters=K, init=centers, n_init=1, max_iter=100)
        kmeans.fit(Z.T)
        Y = kmeans.cluster_centers_.T

    if lambda_param is None:
        lambda_param = 5.0 * N

    # Main iterative optimization
    objective_vals = []
    B = np.zeros((K, K))

    for iteration in range(maxIter):
        if verbose:
            print(f"Iteration: {iteration}")

        # Step 1: Compute MST on Y
        distsqMU = _sqdist(Y, Y)

        # Build full distance matrix for MST computation
        # scipy MST works on dense or sparse, expects (K, K) with distances
        # Use scipy's minimum_spanning_tree which expects a CSR matrix
        dist_dense = distsqMU.copy()
        np.fill_diagonal(dist_dense, 0)
        mst_sparse = minimum_spanning_tree(csr_matrix(dist_dense))
        mst_array = mst_sparse.toarray()

        # Build symmetric adjacency B
        B = np.zeros((K, K))
        B[mst_array > 0] = 1
        B = B + B.T
        B[B > 0] = 1

        # Graph Laplacian
        L = np.diag(B.sum(axis=1)) - B

        # Step 2: Soft assignment of cells to centers
        distZY = _sqdist(Z, Y)  # (N, K)

        min_dist = distZY.min(axis=1, keepdims=True)  # (N, 1)
        tmp_distZY = distZY - min_dist  # (N, K)

        tmp_R = np.exp(-tmp_distZY / sigma)  # (N, K)
        R = tmp_R / tmp_R.sum(axis=1, keepdims=True)  # (N, K)

        # Gamma = diag(colSums(R))
        Gamma = np.diag(R.sum(axis=0))  # (K, K)

        # Termination check
        x1 = np.log(np.exp(-tmp_distZY / sigma).sum(axis=1))
        obj1 = -sigma * (x1 - min_dist.flatten() / sigma).sum()

        try:
            major_ev = _get_major_eigenvalue(X - W @ Z, dimensions)
        except Exception:
            major_ev = np.linalg.norm(X - W @ Z, ord=2)
        obj2 = major_ev ** 2
        obj2 += lambda_param * np.trace(Y @ L @ Y.T) + param_gamma * obj1
        objective_vals.append(obj2)

        if verbose:
            print(f"  Objective: {obj2:.6f}")

        if iteration >= 1:
            delta_obj = abs(objective_vals[-1] - objective_vals[-2])
            delta_obj /= abs(objective_vals[-2]) if objective_vals[-2] != 0 else 1.0
            if verbose:
                print(f"  delta_obj: {delta_obj:.8f}")
            if delta_obj < tol:
                if verbose:
                    print("Converged!")
                break

        # Step 3: Update W, Z, Y
        # tmp = solve(((gamma+1)/gamma) * (lambda/gamma * L + Gamma) - R^T R, R^T)
        A_mat = ((param_gamma + 1.0) / param_gamma) * (
            (lambda_param / param_gamma) * L + Gamma
        ) - R.T @ R

        try:
            # Use Cholesky if positive definite
            from scipy.linalg import cho_factor, cho_solve
            cho = cho_factor(A_mat)
            tmp_dense = cho_solve(cho, R.T).T  # (N, K)
        except np.linalg.LinAlgError:
            tmp_dense = np.linalg.solve(A_mat, R.T).T  # (N, K)

        # Q = (X + (X @ tmp_dense) @ R^T) / (gamma + 1)
        Q = (X + (X @ tmp_dense) @ R.T) / (param_gamma + 1.0)  # (D, N)

        # C = Q (in the optimized path)
        C = Q

        # W = pca_projection((Q X^T + X Q^T)/2, dimensions)
        # sym_mat = (Q X^T + X Q^T) / 2 is D×D but rank ≤ N
        # Use low-rank trick: if M = Q X^T, then M + M^T has same eigenvectors
        # as the N×N matrix (X^T Q + Q^T X) / 2, mapped back via X or Q.
        # Specifically: top eigvecs of M M^T can be found via N×N problem.
        #
        # Direct approach: form the N×N Gram matrices and solve exactly.
        # Let A = Q (D×N), B = X (D×N). M = A B^T.
        # (M+M^T)/2 v = λv  ⟺  (AB^T + BA^T)/2 v = λv
        # Multiply by B^T: B^T(AB^T + BA^T)v/2 = λ B^T v
        # Let u = B^T v (N×1): (B^T A)(B^T v) + (B^T B)(A^T v) = 2λ u  — coupled.
        #
        # Simpler: since rank(sym_mat) ≤ 2N, form [Q, X] (D×2N) and solve
        # the 2N×2N eigenvalue problem.
        QX = np.hstack([Q, X])  # (D, 2N)
        small = QX.T @ QX       # (2N, 2N)
        # sym_mat = (Q X^T + X Q^T)/2, so sym_mat @ QX = QX @ M_small
        # where M_small = [[Q^T Q, Q^T X], [X^T Q, X^T X]] ... not quite.
        # Actually: sym_mat = (Q X^T + X Q^T)/2
        # sym_mat @ v = λv means we need eigvecs of sym_mat.
        # Since sym_mat has rank ≤ N, its top eigvecs live in span(Q, X).
        # Write v = Q a + X b = [Q X] [a; b]
        # Then [Q X]^T sym_mat [Q X] [a;b] = λ [Q X]^T [Q X] [a;b]
        # This is a generalized eigenvalue problem of size 2N×2N.
        #
        # But even simpler: Q = (X + X @ tmp_dense @ R^T)/(γ+1)
        # So Q is in span(X), meaning sym_mat = (QX^T + XQ^T)/2 is rank ≤ N.
        # v = X c for some c. Then X^T sym_mat X c = λ X^T X c
        # X^T (QX^T + XQ^T)/2 X c = λ (X^T X) c
        # ((X^T Q)(X^T X) + (X^T X)(Q^T X))/2 c = λ (X^T X) c

        # sym_mat = (Q X^T + X Q^T)/2 — top eigvecs give new W.
        #
        # Two methods:
        #   'irlba'  (default, matches R): iterative eigsh — the approximation
        #            noise in each iteration acts as implicit regularization
        #            and helps DDRTree discover branching structure.
        #   'exact': np.linalg.eigh — faster and more precise but may
        #            converge to an oversmoothed solution (fewer branches).
        if pca_method == 'irlba':
            sym_mat = (Q @ X.T + X @ Q.T) / 2.0
            W = _pca_projection_irlba_like(sym_mat, dimensions)
        elif D <= N:
            sym_mat = (Q @ X.T + X @ Q.T) / 2.0
            evals_all, evecs_all = np.linalg.eigh(sym_mat)
            idx = np.argsort(evals_all)[::-1][:dimensions]
            W = evecs_all[:, idx]
        else:
            # Low-rank trick for D > N
            XtX = X.T @ X
            XtQ = X.T @ Q
            lhs = (XtQ @ XtX + XtX @ XtQ.T) / 2.0
            from scipy.linalg import eigh as eigh_gen
            try:
                reg = 1e-10 * np.trace(XtX) / N
                evals, evecs = eigh_gen(
                    lhs, XtX + reg * np.eye(N),
                    subset_by_index=[N - dimensions, N - 1])
                idx = np.argsort(evals)[::-1]
                evecs = evecs[:, idx]
                W = X @ evecs
                for col in range(dimensions):
                    norm_val = np.linalg.norm(W[:, col])
                    if norm_val > 0:
                        W[:, col] /= norm_val
            except np.linalg.LinAlgError:
                sym_mat = (Q @ X.T + X @ Q.T) / 2.0
                W = _pca_projection(sym_mat, dimensions)

        # Z = W^T @ C
        Z = W.T @ C

        # Y = solve(lambda/gamma * L + Gamma, (Z @ R)^T)^T
        A_Y = (lambda_param / param_gamma) * L + Gamma
        try:
            from scipy.linalg import cho_factor, cho_solve
            cho = cho_factor(A_Y)
            Y = cho_solve(cho, (Z @ R).T).T
        except np.linalg.LinAlgError:
            Y = np.linalg.solve(A_Y, (Z @ R).T).T

    # Build final MST with weights
    distsqMU = _sqdist(Y, Y)
    dist_dense = distsqMU.copy()
    np.fill_diagonal(dist_dense, 0)
    mst_sparse = minimum_spanning_tree(csr_matrix(dist_dense))
    # Make symmetric
    stree = mst_sparse + mst_sparse.T

    return {
        'W': W,
        'Z': Z,
        'Y': Y,
        'stree': stree,
        'objective_vals': objective_vals,
    }
