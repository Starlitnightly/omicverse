from __future__ import division
import scipy as s
import numpy as np
from .. import gpu_utils
from ..gp_utils import covar_to_corr
from .variational_nodes import *
from ..gp_utils import *

# TODO: for large number of groups avoid constructing Kmat if spectral decomp and only save x at the end (getParameters in SigmaNode)
class Kg_Node(Node):
    """
    Sigma node to model the covariance structure K_g across groups.
    This node constructs the group-group covariance and stores the ELBO-optimal hyperparamters
    KG = xx^T + diag(sigma)

    PARAMETERS
    ----------
    dim: dimensionality of the node (= number of latent factors x number of groups)
    rank: rank of the approximation of the group-group covariance term (x)
    sigma: initial value of the diagonal term
    """

    def __init__(self, dim, rank, sigma = 0.1, scale_to_cor = True, spectral_decomp = True):
        super().__init__(dim)
        self.G = dim[1]                                             # number of observations for covariate
        self.K = dim[0]                                             # number of latent processes
        self.rank = rank
        self.sigma = np.array([sigma] * self.K)
        self.scale_to_cor = scale_to_cor
        # initialize components in node
        self.compute4init(spectral_decomp = spectral_decomp)


    def compute4init(self, spectral_decomp = True):
        """
        Function to initialize kernel matrix
        """
        # initialize x by full connectedness of groups (matrix of 1's)
        self.x = np.sqrt(np.ones([self.K, self.rank, self.G]) * 1/self.rank)

        # initialise kernel matrix
        self.Kmat = np.zeros([self.K, self.G, self.G])

        # initialise spectral decomposition
        self.V = np.zeros([self.K, self.G, self.G])     # eigenvectors of kernel matrix
        self.D = np.zeros([self.K, self.G])             # eigenvalues of kernel matrix

        self.compute_kernel(spectral_decomp = spectral_decomp)

    def compute_kernel(self, spectral_decomp = True):
        """
        Function to compute kernel matrix for current hyperparameters (x, sigma)
        """
        for k in range(self.K):
            self.compute_kernel_k(k, spectral_decomp = spectral_decomp)


    def compute_kernel_k(self, k, spectral_decomp = True):
        # build kernel matrix based on low-rank approximation
        self.Kmat[k, :, :] = np.dot(self.x[k,:,:].transpose(), self.x[k,:,:]) + self.sigma[k] * np.eye(self.G)

        if self.scale_to_cor:
            self.Kmat[k, :, :] = covar_to_corr(self.Kmat[k, :, :])
        # compute spectral decomposition
        # Kg = VDV^T with V^T V = I
        if spectral_decomp:
            self.D[k, :], self.V[k, :, :] = s.linalg.eigh(self.Kmat[k, :, :])


    def removeFactors(self, idx, axis=1):
        self.updateDim(0, self.dim[0] - len(idx))
        self.K = self.K - 1

    def get_kernel_components_k(self, k):
        """
        Method to fetch components of kernel
        """
        return self.V[k, :, :], self.D[k, :]

    def set_parameters(self, x, sigma, k, spectral_decomp = True):
        """
        Method to set hyperparameters of kernel and recompute covariance matrices
        """
        self.set_x(x,k)
        self.set_sigma(sigma, k)
        self.compute_kernel_k(k, spectral_decomp)

    def set_x(self, x, k):
        self.x[k,:,:] = x

    def set_sigma(self, sigma, k):
        self.sigma[k] = sigma

    def get_x(self):
        return self.x

    def get_sigma(self):
        return self.sigma

    def get_Kmat(self):
        return self.Kmat