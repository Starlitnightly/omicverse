import numpy as np
import scipy as s
import numpy.linalg as linalg
import scipy.stats as stats
import sys
from .basic_distributions import Distribution

from ..utils import *
# from mofapy2.core import gpu_utils
# TODO: Enable GPU support

class MultivariateGaussian(Distribution):
    """
    Class to define multivariate Gaussian distribution.
    This class can store:
    - if axis_cov=1 : N multivariate Gaussian Distributions of dimensionality D each
       (each row of a N x D matrix is a multivariate Gaussian)
    - if axis_cov=0 : D multivariate Gaussian Distributions of dimensionality N each
       (each column of a N x D X matrix is a multivariate Gaussian)
    
    For example, used with the Z node (N x K) this node can be used to store K multivariate Gaussians with dimensonality N each (axis_cov = 0).
    
    Equations (for axis_cov=0) :
    p(X|Mu,Sigma) = 1/(2pi)^{N/2} * 1/(|Sigma|^0.5) * exp( -0.5*(X-Mu)^{T} Sigma^{-1} (X-Mu) )
    log p(X|Mu,Sigma) = -N/2*log(2pi) - 0.5*log(|Sigma|) - 0.5*(X-Mu)^{T} Sigma^{-1} (X-Mu)
    E[X] = Mu
    E[X^2] = E[X]E[X] + Cov[X] = MuMu + Sigma
    cov[X] = Sigma
    H[X] = 0.5*log(|Sigma|) + N/2*(1+log(2*pi))

    Dimensionalities :
    - X: (N,D)
    - Mu: (N,D)
    - Sigma: (N,D,D) if axis_cov=1 or (D,N,N) if axis_cov=0
    - E[X]: (N,D)
    - E[X^2]: (N,D,D) if axis_cov=1 or (D,N,N) if axis_cov=0
    
    Parameters:
    - dim: (N,D)
    - mean: This can be an (N,D)-array, a scalar to be used as a constant mean vector for all Gaussians or
    a vector to be used for each multivariate Gaussian distirbution (Should have length D if axis_cov =1 and N otherwise)
    - cov: either a array of dimensions (N,N) for axis_cov=0 or (D,D) for axis_cov=1 that will be used as covariance matrix in each of the D or N multivariate Gaussian distirbutions
    or a list of length N (if axis_cov=1) or D (if axis_cov=0) or a tensor of dim (N,D,D) (if axis_cov=1) or (D,N,N) (if axis_cov=0)
    - axis_cov indicating which elementof dim (N,D) to use for dimenionality of the covariance matrix. If 0, it gives rise to D multivariate Gaussians with dimension N ( and (N,N)-covariance).
    If 1, it gives rise to N multivariate Gaussians with dimension D ( and (D,D)-covariance).
    """
    def __init__(self, dim, mean, cov, axis_cov=1, E=None):
        Distribution.__init__(self, dim)

        # Check dimensions are correct
        assert len(dim) == 2, "You need to specify two dimensions for 'dim': (number of distributions, dimensionality) "
        assert not (dim[0]==1 and dim[1]==1), "A 1x1 multivariate normal reduces to a Univariate normal "
        assert ((axis_cov == 0)or(axis_cov == 1)), "Error : axis_cov is the index of the dimension (N, D) for the covariance matrix, either 0 (N,N) or 1 (D,D)"

        ## Initialise the parameters ##

        ### Initialise the mean
        # If 'mean' is a scalar, broadcast it to all dimensions
        if isinstance(mean,(int,float)): mean = np.ones( (dim[0],dim[1]) ) * mean
        
        # If 'mean' has dim (D,) and we have N distributions, repeat it to all N distributions (for axis_cov=1) and correspondingly for axis_cov=0
        if len(mean.shape)==1 and mean.shape[0]==dim[1] and axis_cov==1: mean = mean * np.ones( (dim[0],dim[1]) ) 
        if len(mean.shape)==1 and mean.shape[0]==dim[0] and axis_cov==0: mean = (mean * np.ones( (dim[1],dim[0]) )).transpose()
        
        # check 'mean' has the right dimensions
        assert mean.shape[0]==dim[0] and mean.shape[1]==dim[1], "The given mean could not be broadcasted into a matrix with shape (N,D) "

        ### Initialise the covariance
        
        if isinstance(cov,np.ndarray):
            # If 'cov' is a matrix and not a tensor, broadcast it along the zeroth axis
            if len(cov.shape) == 2:
                if axis_cov == 1 :
                    assert cov.shape == (dim[1],dim[1]), "If providing a 2d-array, the covariance has to be of dim (D,D)"
                    cov = [cov] * dim[0]
                else:
                    assert cov.shape == (dim[0],dim[0]), "If providing a 2d-array, the covariance has to be of dim (N,N)"
                    cov = [cov] * dim[1]
                cov = np.array(cov)
        # If 'cov' is a list transform it to a tensor
        elif isinstance(cov, list):
            if axis_cov == 1:
                assert cov[0].shape ==  (dim[1],dim[1]) and len(cov) == dim[0], "If providing a list, the covariance has to be a list of length N with arrays of dim (D,D)"
            else:
                assert cov[0].shape ==  (dim[0],dim[0]) and len(cov) == dim[1], "If providing a list, the covariance has to be a list of length D with arrays of dim (N,N)"
            cov = np.array(cov)
        else: 
            print("The covariance needs to be a list or an array.")
            sys.exit()
            
        # check 'cov' has the right dimensions
        if axis_cov == 1:
            assert (cov.shape == (dim[0], dim[1], dim[1])), "The covariance could not be broadcasted into a tensor with shape (N,D,D)."
        if axis_cov == 0:
            assert (cov.shape == (dim[1], dim[0], dim[0])), "The covariance could not be broadcasted into a tensor with shape (D,N,N)."
            
        # store parameters
        self.axis_cov = axis_cov
        self.params = {'mean' : mean, 'cov' : cov }

        # Initialise expectations
        if E is None:
            self.updateExpectations()
        else:
            self.expectations = { 'E' : E }

    def updateExpectations(self):
        # Update first and second moments (of marginal only)
        E = self.params['mean']

        # second moment here of the marginal components: given by E(X_n^2) = E(X_n)^2 + Var(X_n)
        E2 = s.empty( (self.dim[0],self.dim[1]))
        if self.axis_cov == 1:
            for i in range(self.dim[0]):
               E2[i, :] = E[i,:]**2 + np.diag(self.params['cov'][i,:,:])
        else:
           for i in range(self.dim[1]):
               E2[:, i] = E[:,i]**2 + np.diag(self.params['cov'][i,:,:])

        self.expectations = {'E': E, 'E2': E2, 'cov' : self.params['cov']}

    def loglik(self, x):
        assert x.shape == self.dim, "Problem with the dimensionalities"
        l = 0.

        if self.axis_cov == 1:
            D = self.dim[1]
            for n in range(self.dim[0]):
                qterm = (x[n,:]-self.params['mean'][n,:]).T.dot(linalg.det(self.params['cov'][n])).dot(x[n,:]-self.params['mean'][n,:])
                l += -0.5*D*np.log(2*np.pi) - 0.5*np.log(linalg.det(self.params['cov'][n])) - 0.5 * qterm
            # return np.sum( np.log(stats.multivariate_normal.pdf(x, mean=self.mean[n,:], cov=self.cov[n,:,:])) )

        else:
            N = self.dim[0]
            for d in range(self.dim[1]):
                qterm = (x[:, d] - self.params['mean'][:, d]).dot(linalg.det(self.params['cov'][d])).dot((x[:, d] - self.params['mean'][:, d]).T)
                l += -0.5 * N * np.log(2 * np.pi) - 0.5 * np.log(linalg.det(self.params['cov'][d])) - 0.5 * qterm

        return l

    def removeDimensions(self, axis, idx):
        # Method to remove undesired dimensions
        # - axis (int): axis from where to remove the elements
        # - idx (numpy array): indices of the elements to remove
        assert axis <= len(self.dim)
        assert np.all(idx < self.dim[axis])

        self.params["mean"] = np.delete(self.params["mean"], axis=axis, obj=idx)
        self.expectations["E"] = np.delete(self.expectations["E"], axis=axis, obj=idx)
        self.expectations["E2"] = np.delete(self.expectations["E2"], axis=axis, obj=idx)

        if self.axis_cov == 1: #cov has shape (N,D,D) for mean of shape (N,D)
            if axis == 0:
                self.params["cov"] = np.delete(self.params["cov"], axis=0, obj=idx)
            else:
                self.params["cov"] = np.delete(self.params["cov"], axis=1, obj=idx)
                self.params["cov"] = np.delete(self.params["cov"], axis=2, obj=idx)

        else: #cov has shape (D,N,N) for mean of shape (N,D)
            if axis == 0:
                self.params["cov"] = np.delete(self.params["cov"], axis=1, obj=idx)
                self.params["cov"] = np.delete(self.params["cov"], axis=2, obj=idx)
            else:
                self.params["cov"] = np.delete(self.params["cov"], axis=0, obj=idx)
        self.updateDim(axis=axis, new_dim=self.dim[axis] - len(idx))

    def sample(self):
        if axis_cov==1:
            samples = []
            for n in range(self.dim[0]):
                samples.append(np.random.multivariate_normal(self.params['mean'][n,:], self.params['cov'][n]))
            samples = np.array(samples)
        else:
            samples = []
            for d in range(self.dim[1]):
                samples.append(np.random.multivariate_normal(self.params['mean'][:,d], self.params['cov'][d]))
            samples = np.array(samples).T
        return samples

#    def entropy(self):
#         tmp = sum( [ logdet(self.cov[i,:,:]) for i in range(self.dim[0]) ] )
#         return ( 0.5*(tmp + (self.dim[0]*self.dim[1])*(1+np.log(2*pi)) ).sum() )


class MultivariateGaussian_reparam(Distribution):
    """
       Class to define multivariate Gaussian distribution with reparamtriation following Opper & Archambeau (2009)
\      Each distribution's mean and covariance are reparametrized as
       mu = K alpha
       Sigma = (K^(-1) + diag(lambda**2))^(-1)

       Dimensions:
       K : (D, N, N) if axis_cov = 0, else (N, D, D)
       alpha : (N, D)
       lambda: (N, D)
    """

    def __init__(self, dim, alpha, K, lamb, axis_cov=1, E =None):
        Distribution.__init__(self, dim)

        assert len(dim) == 2, "You need to specify two dimensions for 'dim': (number of distributions, dimensionality) "
        assert not (dim[0] == 1 and dim[1] == 1), "A 1x1 multivariate normal reduces to a Univariate normal "
        assert ((axis_cov == 0) or (
                    axis_cov == 1)), "Error : axis_cov is the index of the dimension (N, D) for the covariance matrix, either 0 (N,N) or 1 (D,D)"

        # Broadcast scalars to arrays for alpha and lambda
        if isinstance(alpha, (int, float)): alpha = np.ones((dim[0], dim[1])) * alpha
        if isinstance(lamb, (int, float)): lamb = np.ones((dim[0], dim[1])) * lamb

        # If 'alpha' has dim (D,) and we have N distributions, repeat it to all N distributions (for axis_cov=1) and correspondingly for axis_cov=0
        if len(alpha.shape) == 1 and alpha.shape[0] == dim[1] and axis_cov == 1: alpha = alpha * np.ones((dim[0], dim[1]))
        if len(alpha.shape) == 1 and alpha.shape[0] == dim[0] and axis_cov == 0: alpha = (
                    alpha * np.ones((dim[1], dim[0]))).transpose()
        if len(lamb.shape) == 1 and lamb.shape[0] == dim[1] and axis_cov == 1: lamb = lamb * np.ones(
            (dim[0], dim[1]))
        if len(lamb.shape) == 1 and lamb.shape[0] == dim[0] and axis_cov == 0: lamb = (
                lamb * np.ones((dim[1], dim[0]))).transpose()

        # Check dimensions of alpha and lamb are correct
        assert alpha.shape[0] == dim[0] and alpha.shape[1] == dim[1], "The given alpha could not be broadcasted into a matrix with shape (N,D) "
        assert lamb.shape[0] == dim[0] and lamb.shape[1] == dim[1], "The given lamb could not be broadcasted into a matrix with shape (N,D) "

        # check K has the right dimensions
        if isinstance(K, np.ndarray):
            # If 'K' is a matrix and not a tensor, broadcast it along the zeroth axis
            if len(K.shape) == 2:
                if axis_cov == 1:
                    assert K.shape == (
                    dim[1], dim[1]), "If providing a 2d-array, K has to be of dim (D,D)"
                    K = [K] * dim[0]
                else:
                    assert K.shape == (
                    dim[0], dim[0]), "If providing a 2d-array, K has to be of dim (N,N)"
                    K = [K] * dim[1]
                K = np.array(K)
        # If 'K' is a list transform it to a tensor
        elif isinstance(K, list):
            if axis_cov == 1:
                assert K[0].shape == (dim[1], dim[1]) and len(K) == dim[
                    0], "If providing a list, K has to be a list of length N with arrays of dim (D,D)"
            else:
                assert K[0].shape == (dim[0], dim[0]) and len(K) == dim[
                    1], "If providing a list, K has to be a list of length D with arrays of dim (N,N)"
            K = np.array(K)
        else:
            print("The input K needs to be a list or an array.")
            sys.exit()

        if axis_cov == 1:
            assert (K.shape == (
            dim[0], dim[1], dim[1])), "K could not be broadcasted into a tensor with shape (N,D,D)."
        if axis_cov == 0:
            assert (K.shape == (
            dim[1], dim[0], dim[0])), "K could not be broadcasted into a tensor with shape (D,N,N)."


        # Initialise the parameters
        self.axis_cov = axis_cov
        self.params = {'alpha': alpha, 'K': K, 'lamb' : lamb}

        # Initialise expectations
        if E is None:
            self.updateExpectations()
        else:
            self.expectations = { 'E' : E }

    def updateExpectations(self):
        # Method to calculate expectation (N,D) and variance of th marginals (N,D)

        # first moments
        E = s.empty((self.dim[0], self.dim[1]))
        if self.axis_cov == 0:
            for i in range(self.dim[1]):
                E[:, i] = self.params['K'][i, :, :].dot(self.params['alpha'][:, i])
        elif self.axis_cov == 1:
            for i in range(self.dim[0]):
                E[i,:] = self.params['K'][i, :, :].dot(self.params['alpha'][i, :].transpose())

        # second moment here of the marginal components: given by E(X_n^2) = E(X_n)^2 + Var(X_n)
        E2 = s.empty((self.dim[0],self.dim[1]))
        if self.axis_cov == 0:
           for i in range(self.dim[1]):
               A = np.diag(self.params['lamb'][:, i]).dot(self.params['K'][i, :, :]).dot(
                   np.diag(self.params['lamb'][:, i])) + np.eye(self.dim[0])
               Ainv = np.linalg.inv(A)
               Sigma_diag = (1 / self.params['lamb'][:,i]**2) * (1 - np.diag(Ainv))
               E2[:, i] = E[:, i] ** 2 + Sigma_diag

        elif self.axis_cov == 1:
            for i in range(self.dim[0]):
                A = np.diag(self.params['lamb'][i, :]).dot(self.params['K']).dot(
                    np.diag(self.params['lamb'][i, :])) + np.eye(self.dim[1])
                Ainv = np.linalg.inv(A)
                Sigma_diag = (1 / self.params['lamb'][i,:]**2) * (1 - np.diag(Ainv))
                E2[i, :] = E[i, :] ** 2 + Sigma_diag

        self.expectations = {'E': E, 'E2': E2}

    def removeDimensions(self, axis, idx):
        # Method to remove undesired dimensions
        # - axis (int): axis from where to remove the elements
        # - idx (numpy array): indices of the elements to remove
        assert axis <= len(self.dim)
        assert np.all(idx < self.dim[axis])

        self.params["alpha"] = np.delete(self.params["alpha"], axis=axis, obj=idx)
        self.expectations["E"] = np.delete(self.expectations["E"], axis=axis, obj=idx)
        self.expectations["E2"] = np.delete(self.expectations["E2"], axis=axis, obj=idx)

        if self.axis_cov == 1: #K has shape (N,D,D) for mean of shape (N,D)
            if axis == 0:
                self.params["K"] = np.delete(self.params["K"], axis=0, obj=idx)
            else:
                self.params["K"] = np.delete(self.params["K"], axis=1, obj=idx)
                self.params["K"] = np.delete(self.params["K"], axis=2, obj=idx)

        else: #K has shape (D,N,N) for mean of shape (N,D)
            if axis == 0:
                self.params["K"] = np.delete(self.params["K"], axis=1, obj=idx)
                self.params["k"] = np.delete(self.params["K"], axis=2, obj=idx)
            else:
                self.params["K"] = np.delete(self.params["K"], axis=0, obj=idx)
        self.updateDim(axis=axis, new_dim=self.dim[axis] - len(idx))
