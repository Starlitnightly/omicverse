# optimization
# for parallelizing stuff
from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix
from tqdm.notebook import tqdm

# get number of cores for multiprocessing
NUM_CORES = cpu_count()

##########################################################
# Helper functions for parallelizing kernel construction
##########################################################


def kth_neighbor_distance(distances, k, i):
    """Returns distance to kth nearest neighbor.

    Distances: sparse CSR matrix
    k: kth nearest neighbor
    i: index of row
    .
    """
    # convert row to 1D array
    row_as_array = distances[i, :].toarray().ravel()

    # number of nonzero elements
    num_nonzero = np.sum(row_as_array > 0)

    # argsort
    kth_neighbor_idx = np.argsort(np.argsort(-row_as_array)) == num_nonzero - k
    return np.linalg.norm(row_as_array[kth_neighbor_idx])


def rbf_for_row(G, data, median_distances, i):
    """Helper function for computing radial basis function kernel for each row of the data matrix.

    :param G: (array) KNN graph representing nearest neighbour connections between cells
    :param data: (array) data matrix between which euclidean distances are computed for RBF
    :param median_distances: (array) radius for RBF - the median distance between cell and k nearest-neighbours
    :param i: (int) data row index for which RBF is calculated
    :return: sparse matrix containing computed RBF for row
    """
    # convert row to binary numpy array
    row_as_array = G[i, :].toarray().ravel()

    # compute distances ||x - y||^2 in PC/original X space
    numerator = np.sum(np.square(data[i, :] - data), axis=1, keepdims=False)

    # compute radii - median distance is distance to kth nearest neighbor
    denominator = median_distances[i] * median_distances

    # exp
    full_row = np.exp(-numerator / denominator)

    # masked row - to contain only indices captured by G matrix
    masked_row = np.multiply(full_row, row_as_array)

    return lil_matrix(masked_row)


##########################################################
# Archetypal Analysis Metacell Graph
##########################################################


class SEACellGraph:
    """SEACell graph class."""

    def __init__(self, ad, build_on="X_pca", n_cores: int = -1, verbose: bool = False):
        """SEACell graph class.

        :param ad: (anndata.AnnData) object containing data for which metacells are computed
        :param build_on: (str) key corresponding to matrix in ad.obsm which is used to compute kernel for metacells
                        Typically 'X_pca' for scRNA or 'X_svd' for scATAC
        :param n_cores: (int) number of cores for multiprocessing. If unspecified, computed automatically as
                        number of CPU cores
        :param verbose: (bool) whether or not to suppress verbose program logging
        """
        """Initialize model parameters"""
        # data parameters
        self.n, self.d = ad.obsm[build_on].shape

        # indices of each point
        self.indices = np.array(range(self.n))

        # save data
        self.ad = ad
        self.build_on = build_on

        self.knn_graph = None
        self.sym_graph = None

        # number of cores for parallelization
        if n_cores != -1:
            self.num_cores = n_cores
        else:
            self.num_cores = NUM_CORES

        self.M = None  # similarity matrix
        self.G = None  # graph
        self.T = None  # transition matrix

        # model params
        self.verbose = verbose

    ##############################################################
    # Methods related to kernel + sim matrix construction
    ##############################################################

    def rbf(self, k: int = 15, graph_construction="union"):
        """Initialize adaptive bandwith RBF kernel (as described in C-isomap).

        :param k: (int) number of nearest neighbors for RBF kernel
        :return: (sparse matrix) constructed RBF kernel
        """
        import scanpy as sc

        if self.verbose:
            print("Computing kNN graph using scanpy NN ...")

        # compute kNN and the distance from each point to its nearest neighbors
        sc.pp.neighbors(self.ad, use_rep=self.build_on, n_neighbors=k, knn=True)
        knn_graph_distances = self.ad.obsp["distances"]

        # Binarize distances to get connectivity
        knn_graph = knn_graph_distances.copy()
        knn_graph[knn_graph != 0] = 1
        # Include self as neighbour
        knn_graph.setdiag(1)

        self.knn_graph = knn_graph
        if self.verbose:
            print("Computing radius for adaptive bandwidth kernel...")

            # compute median distance for each point amongst k-nearest neighbors
        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            median = k // 2
            median_distances = parallel(
                delayed(kth_neighbor_distance)(knn_graph_distances, median, i)
                for i in tqdm(range(self.n))
            )

        # convert to numpy array
        median_distances = np.array(median_distances)

        if self.verbose:
            print("Making graph symmetric...")

        print(
            f"Parameter graph_construction = {graph_construction} being used to build KNN graph..."
        )
        if graph_construction == "union":
            sym_graph = (knn_graph + knn_graph.T > 0).astype(float)
        elif graph_construction in ["intersect", "intersection"]:
            knn_graph = (knn_graph > 0).astype(float)
            sym_graph = knn_graph.multiply(knn_graph.T)
        else:
            raise ValueError(
                f"Parameter graph_construction = {graph_construction} is not valid. \
             Please select `union` or `intersection`"
            )

        self.sym_graph = sym_graph
        if self.verbose:
            print("Computing RBF kernel...")

        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            similarity_matrix_rows = parallel(
                delayed(rbf_for_row)(
                    sym_graph, self.ad.obsm[self.build_on], median_distances, i
                )
                for i in tqdm(range(self.n))
            )

        if self.verbose:
            print("Building similarity LIL matrix...")

        similarity_matrix = lil_matrix((self.n, self.n))
        for i in tqdm(range(self.n)):
            similarity_matrix[i] = similarity_matrix_rows[i]

        if self.verbose:
            print("Constructing CSR matrix...")

        self.M = (similarity_matrix).tocsr()
        return self.M
