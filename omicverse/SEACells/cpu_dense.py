import copy

import numpy as np
import palantir
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

from . import build_graph


class SEACellsCPUDense:
    """Fast kernel archetypal analysis.

    Finds archetypes and weights given annotated data matrix.
    Modifies annotated data matrix in place to include SEACell assignments in ad.obs['SEACell'].
    """

    def __init__(
        self,
        ad,
        build_kernel_on: str,
        n_SEACells: int,
        verbose: bool = True,
        n_waypoint_eigs: int = 10,
        n_neighbors: int = 15,
        convergence_epsilon: float = 1e-3,
        l2_penalty: float = 0,
        max_franke_wolfe_iters: int = 50,
    ):
        """Fast kernel archetypal analysis.

        ad - anndata.AnnData.
        """
        print("Welcome to SEACells!")
        self.ad = ad
        self.build_kernel_on = build_kernel_on
        self.n_cells = ad.shape[0]

        if not isinstance(n_SEACells, int):
            try:
                n_SEACells = int(n_SEACells)
            except ValueError:
                raise ValueError(
                    f"The number of SEACells specified must be an integer type, not {type(n_SEACells)}"
                )

        self.k = n_SEACells

        self.n_waypoint_eigs = n_waypoint_eigs
        self.waypoint_proportion = 1
        self.n_neighbors = n_neighbors

        self.max_FW_iter = max_franke_wolfe_iters
        self.verbose = verbose
        self.l2_penalty = l2_penalty

        self.RSS_iters = []
        self.convergence_epsilon = convergence_epsilon
        self.convergence_threshold = None

        # Parameters to be initialized later in the model
        self.kernel_matrix = None
        self.K = None

        # Archetypes as list of cell indices
        self.archetypes = None

        self.A_ = None
        self.B_ = None
        self.B0 = None

        return

    def add_precomputed_kernel_matrix(self, K):
        """Compute kernel matrix."""
        assert K.shape == (self.n_cells, self.n_cells), (
            f"Dimension of kernel matrix must be n_cells = "
            f"({self.n_cells},{self.n_cells}), not {K.shape} "
        )
        self.kernel_matrix = K

        # Pre-compute dot product
        self.K = self.kernel_matrix @ self.kernel_matrix.T

    def construct_kernel_matrix(
        self, n_neighbors: int = None, graph_construction="union"
    ):
        """Construct kernel matrix."""
        # input to graph construction is PCA/SVD
        kernel_model = build_graph.SEACellGraph(
            self.ad, self.build_kernel_on, verbose=self.verbose
        )

        # K is a sparse matrix representing input to SEACell alg
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        M = kernel_model.rbf(n_neighbors, graph_construction=graph_construction)
        self.kernel_matrix = M

        # Pre-compute dot product
        self.K = self.kernel_matrix @ self.kernel_matrix.T

        return

    def initialize_archetypes(self):
        """Initialize B matrix which defines cells as SEACells.

        Selects waypoint_proportion from waypoint analysis, and the remainder by greedy selection.

        Modifies self.archetypes in-place with the indices of cells that are used as initialization for archetypes
        """
        k = self.k

        if self.waypoint_proportion > 0:
            waypoint_ix = self._get_waypoint_centers(k)
            waypoint_ix = np.random.choice(
                waypoint_ix,
                int(len(waypoint_ix) * self.waypoint_proportion),
                replace=False,
            )
            from_greedy = self.k - len(waypoint_ix)
            if self.verbose:
                print(
                    f"Selecting {len(waypoint_ix)} cells from waypoint initialization."
                )

        else:
            from_greedy = self.k

        greedy_ix = self._get_greedy_centers(n_mcs=from_greedy + 10)
        if self.verbose:
            print(f"Selecting {from_greedy} cells from greedy initialization.")

        if self.waypoint_proportion > 0:
            all_ix = np.hstack([waypoint_ix, greedy_ix])
        else:
            all_ix = np.hstack([greedy_ix])

        unique_ix, ind = np.unique(all_ix, return_index=True)
        all_ix = unique_ix[np.argsort(ind)][:k]
        self.archetypes = all_ix

    def initialize(self, initial_archetypes=None, initial_assignments=None):
        """Initalize SEACells assignment."""
        if self.K is None:
            raise RuntimeError(
                "Must first construct kernel matrix before initializing SEACells."
            )
        K = self.K
        # initialize B (update this to allow initialization from RRQR)
        n = K.shape[0]

        if initial_archetypes is not None:
            if self.verbose:
                print("Using provided list of initial archetypes")
            self.archetypes = initial_archetypes

        if self.archetypes is None:
            self.initialize_archetypes()
        self.k = len(self.archetypes)
        k = self.k

        # Construction of B matrix
        B0 = np.zeros((n, k))
        all_ix = self.archetypes
        idx1 = list(zip(all_ix, np.arange(k)))
        B0[tuple(zip(*idx1))] = 1
        self.B0 = B0
        B = self.B0.copy()

        if initial_assignments is not None:
            A0 = initial_assignments
            assert A0.shape == (
                k,
                n,
            ), f"Initial assignment matrix should be of shape (k={k} x n={n})"

        else:
            A0 = np.random.random((k, n))
            A0 /= A0.sum(0)
            if self.verbose:
                print("Randomly initialized A matrix.")

        self.A0 = A0
        A = self.A0.copy()
        A = self._updateA(B, A)

        self.A_ = A
        self.B_ = B

        # Create convergence threshold
        RSS = self.compute_RSS(A, B)
        self.RSS_iters.append(RSS)

        if self.convergence_threshold is None:
            self.convergence_threshold = self.convergence_epsilon * RSS
            if self.verbose:
                print(
                    f"Setting convergence threshold at {self.convergence_threshold:.5f}"
                )

    def _get_waypoint_centers(self, n_waypoints=None):
        """Initialize B matrix using waypoint analysis, as described in Palantir.

        From https://www.nature.com/articles/s41587-019-0068-4.

        :param n_waypoints: (int) number of SEACells to initialize using waypoint analysis. If None specified,
                        all SEACells initialized using this method.
        :return: B - (array) n_datapoints x n_SEACells matrix with initial SEACell definitions
        """
        if n_waypoints is None:
            k = self.k
        else:
            k = n_waypoints

        ad = self.ad

        if self.build_kernel_on == "X_pca":
            pca_components = pd.DataFrame(ad.obsm["X_pca"]).set_index(ad.obs_names)
        elif self.build_kernel_on == "X_svd":
            # Compute PCA components from ad object
            pca_components = pd.DataFrame(ad.obsm["X_svd"]).set_index(ad.obs_names)
        else:
            pca_components = pd.DataFrame(ad.obsm[self.build_kernel_on]).set_index(
                ad.obs_names
            )

        print(f"Building kernel on {self.build_kernel_on}")

        if self.verbose:
            print(
                f"Computing diffusion components from {self.build_kernel_on} for waypoint initialization ... "
            )

        dm_res = palantir.utils.run_diffusion_maps(
            pca_components, n_components=self.n_neighbors
        )
        dc_components = palantir.utils.determine_multiscale_space(
            dm_res, n_eigs=self.n_waypoint_eigs
        )
        if self.verbose:
            print("Done.")

        # Initialize SEACells via waypoint sampling
        if self.verbose:
            print("Sampling waypoints ...")
        waypoint_init = palantir.core._max_min_sampling(
            data=dc_components, num_waypoints=k
        )
        dc_components["iix"] = np.arange(len(dc_components))
        waypoint_ix = dc_components.loc[waypoint_init]["iix"].values
        if self.verbose:
            print("Done.")

        return waypoint_ix

    def _get_greedy_centers(self, n_mcs=None):
        """Initialize SEACells using fast greedy adaptive CSSP.

        From https://arxiv.org/pdf/1312.6838.pdf
        :param n_mcs: (int) number of SEACells to initialize using greedy selection. If None specified,
                        all SEACells initialized using this method.
        :return: B - (array) n_datapoints x n_SEACells matrix with initial SEACell definitions
        """
        K = self.K
        n = K.shape[0]

        if n_mcs is None:
            k = self.k
        else:
            k = n_mcs

        if self.verbose:
            print("Initializing residual matrix using greedy column selection")

        # precompute M.T * M
        # ATA = M.T @ M
        ATA = K

        if self.verbose:
            print("Initializing f and g...")

        f = np.array((ATA.multiply(ATA)).sum(axis=0)).ravel()
        # f = np.array((ATA * ATA).sum(axis=0)).ravel()
        g = np.array(ATA.diagonal()).ravel()

        d = np.zeros((k, n))
        omega = np.zeros((k, n))

        # keep track of selected indices
        centers = np.zeros(k, dtype=int)

        # sampling
        for j in tqdm(range(k)):
            score = f / g
            p = np.argmax(score)

            # print residuals
            np.sum(f)

            delta_term1 = ATA[:, p].toarray().squeeze()
            # print(delta_term1)
            delta_term2 = (
                np.multiply(omega[:, p].reshape(-1, 1), omega).sum(axis=0).squeeze()
            )
            delta = delta_term1 - delta_term2

            # some weird rounding errors
            delta[p] = np.max([0, delta[p]])

            o = delta / np.max([np.sqrt(delta[p]), 1e-6])
            omega_square_norm = np.linalg.norm(o) ** 2
            omega_hadamard = np.multiply(o, o)
            term1 = omega_square_norm * omega_hadamard

            # update f (term2)
            pl = np.zeros(n)
            for r in range(j):
                omega_r = omega[r, :]
                pl += np.dot(omega_r, o) * omega_r

            ATAo = (ATA @ o.reshape(-1, 1)).ravel()
            term2 = np.multiply(o, ATAo - pl)

            # update f
            f += -2.0 * term2 + term1

            # update g
            g += omega_hadamard

            # store omega and delta
            d[j, :] = delta
            omega[j, :] = o

            # add index
            centers[j] = int(p)

        return centers

    def _updateA(self, B, A_prev):
        """Given archetype matrix B and using kernel matrix K, compute assignment matrix A using gradient descent.

        :param B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
        :return: A: (array) k*n matrix (dense) defining weights used for assigning cells to SEACells
        """
        n, k = B.shape
        A = A_prev

        t = 0  # current iteration (determine multiplicative update)

        # precompute some gradient terms
        t2 = (self.K @ B).T
        t1 = t2 @ B

        # update rows of A for given number of iterations
        while t < self.max_FW_iter:
            # compute gradient (must convert matrix to ndarray)
            G = 2.0 * np.array(t1 @ A - t2) - self.l2_penalty * A

            # get argmins
            amins = np.argmin(G, axis=0)

            # loop free implementation
            e = np.zeros((k, n))
            e[amins, np.arange(n)] = 1.0

            A += 2.0 / (t + 2.0) * (e - A)
            t += 1

        return A

    def _updateB(self, A, B_prev):
        """Given assignment matrix A and using kernel matrix K, compute archetype matrix B.

        :param A: (array) k*n matrix (dense) defining weights used for assigning cells to SEACells
        :return: B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
        """
        K = self.K
        k, n = A.shape

        B = B_prev

        # keep track of error
        t = 0

        # precompute some terms
        t1 = A @ A.T
        t2 = K @ A.T

        # update rows of B for a given number of iterations
        while t < self.max_FW_iter:
            # compute gradient (need to convert np.matrix to np.array)
            G = 2.0 * np.array(K @ B @ t1 - t2)

            # get all argmins
            amins = np.argmin(G, axis=0)

            e = np.zeros((n, k))
            e[amins, np.arange(k)] = 1.0

            B += 2.0 / (t + 2.0) * (e - B)

            t += 1

        return B

    def compute_reconstruction(self, A=None, B=None):
        """Compute reconstructed data matrix using learned archetypes (SEACells) and assignments.

        :param A: (array) k*n matrix (dense) defining weights used for assigning cells to SEACells
                If None provided, self.A is used.
        :param B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
                If None provided, self.B is used.
        :return: array (n data points x data dimension) representing reconstruction of original data matrix
        """
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        if A is None or B is None:
            raise RuntimeError(
                "Either assignment matrix A or archetype matrix B is None."
            )
        return (self.kernel_matrix.dot(B)).dot(A)

    def compute_RSS(self, A=None, B=None):
        """Compute residual sum of squares error in difference between reconstruction and true data matrix.

        :param A: (array) k*n matrix (dense) defining weights used for assigning cells to SEACells
                If None provided, self.A is used.
        :param B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
        :param B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
                If None provided, self.B is used.
        :return:
            ||X-XBA||^2 - (float) square difference between true data and reconstruction.
        """
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        reconstruction = self.compute_reconstruction(A, B)
        return np.linalg.norm(self.kernel_matrix - reconstruction)

    def plot_convergence(self, save_as=None, show=True):
        """Plot behaviour of squared error over iterations.

        :param save_as: (str) name of file which figure is saved as. If None, no plot is saved.
        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.RSS_iters)
        plt.title("Reconstruction Error over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Squared Error")
        if save_as is not None:
            plt.savefig(save_as, dpi=150)
        if show:
            plt.show()
        plt.close()

    def step(self):
        """Perform one iteration of fitting to update A and B assignment matrices."""
        A = self.A_
        B = self.B_

        if self.K is None:
            raise RuntimeError(
                "Kernel matrix has not been computed. Run model.construct_kernel_matrix() first."
            )

        if A is None:
            raise RuntimeError(
                "Cell to SEACell assignment matrix has not been initialised. Run model.initialize() first."
            )

        if B is None:
            raise RuntimeError(
                "Archetype matrix has not been initialised. Run model.initialize() first."
            )

        A = self._updateA(B, A)
        B = self._updateB(A, B)

        self.RSS_iters.append(self.compute_RSS(A, B))

        self.A_ = A
        self.B_ = B

        # Label cells by SEACells assignment
        labels = self.get_hard_assignments()
        self.ad.obs["SEACell"] = labels["SEACell"]

        return

    def _fit(
        self,
        max_iter: int = 50,
        min_iter: int = 10,
        initial_archetypes=None,
        initial_assignments=None,
    ):
        """Compute archetypes and loadings given kernel matrix K.

        Iteratively updates A and B matrices until maximum number of iterations or convergence has been achieved.

        Modifies ad.obs in place to add 'SEACell' labels to cells.

        :param max_iter: (int) maximum number of iterations to update A and B matrices
        :param min_iter: (int) minimum number of iterations to update A and B matrices
        :param initial_archetypes: (list) indices of cells to use as initial archetypes

        """
        self.initialize(
            initial_archetypes=initial_archetypes,
            initial_assignments=initial_assignments,
        )

        converged = False
        n_iter = 0
        while (not converged and n_iter < max_iter) or n_iter < min_iter:
            n_iter += 1
            if n_iter == 1 or (n_iter) % 10 == 0:
                if self.verbose:
                    print(f"Starting iteration {n_iter}.")

            self.step()

            if n_iter == 1 or (n_iter) % 10 == 0:
                if self.verbose:
                    print(f"Completed iteration {n_iter}.")

            # Check for convergence
            if (
                np.abs(self.RSS_iters[-2] - self.RSS_iters[-1])
                < self.convergence_threshold
            ):
                if self.verbose:
                    print(f"Converged after {n_iter} iterations.")
                converged = True

        self.Z_ = self.B_.T @ self.K

        # Label cells by SEACells assignment
        labels = self.get_hard_assignments()
        self.ad.obs["SEACell"] = labels["SEACell"]

        if not converged:
            raise RuntimeWarning(
                "Warning: Algorithm has not converged - you may need to increase the maximum number of iterations"
            )
        return

    def fit(
        self,
        max_iter: int = 100,
        min_iter: int = 10,
        initial_archetypes=None,
        initial_assignments=None,
    ):
        """Wrapper to fit model.

        :param max_iter: (int) maximum number of iterations to update A and B matrices. Default: 100
        :param min_iter: (int) maximum number of iterations to update A and B matrices. Default: 10
        :param initial_archetypes: (list) indices of cells to use as initial archetypes
        """
        if max_iter < min_iter:
            raise ValueError(
                "The maximum number of iterations specified is lower than the minimum number of iterations specified."
            )
        self._fit(
            max_iter=max_iter,
            min_iter=min_iter,
            initial_archetypes=initial_archetypes,
            initial_assignments=initial_assignments,
        )

    def get_archetype_matrix(self):
        """Return k x n matrix of archetypes."""
        return self.Z_

    def get_soft_assignments(self):
        """Compute soft SEACells assignment."""
        archetype_labels = self.get_hard_archetypes()
        A = copy.deepcopy(self.A_.T)

        labels = []
        weights = []
        for _i in range(5):
            l = A.argmax(1)
            labels.append(archetype_labels[l])
            weights.append(A[np.arange(A.shape[0]), l])
            A[np.arange(A.shape[0]), l] = -1

        weights = np.vstack(weights).T
        labels = np.vstack(labels).T

        soft_labels = pd.DataFrame(labels)
        soft_labels.index = self.ad.obs_names

        return soft_labels, weights

    def get_hard_assignments(self):
        """Returns a dataframe with SEACell assignments under the column 'SEACell'.

        :return: pd.DataFrame with column 'SEACell'.
        """
        # Use argmax to get the index with the highest assignment weight

        df = pd.DataFrame({"SEACell": [f"SEACell-{i}" for i in self.A_.argmax(0)]})
        df.index = self.ad.obs_names
        df.index.name = "index"

        return df

    def get_archetypes(self):
        """TODO."""
        raise NotImplementedError

    def get_hard_archetypes(self):
        """Return the names of cells most strongly identified as archetypes."""
        return self.ad.obs_names[self.B_.argmax(0)]

    def save_assignments(self, outdir):
        """Save (sparse) assignment matrices to specified directory."""
        import os

        os.makedirs(outdir, exist_ok=True)

        A = csr_matrix(self.A_)
        B = csr_matrix(self.B_)

        A = A.T

        save_npz(outdir + "/kernel_matrix.npz", self.kernel_matrix)
        save_npz(outdir + "/A.npz", A)
        save_npz(outdir + "/B.npz", B)

        labels = self.get_hard_assignments()
        labels.to_csv(outdir + "/SEACells.csv")
