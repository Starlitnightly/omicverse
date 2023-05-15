import numpy as np
import scipy as s
import scipy.stats as stats
from .basic_distributions import Distribution

from ... import config

class UnivariateGaussian(Distribution):
    """
    Class to define univariate Gaussian distributions

    Equations:
    Class for a univariate Gaussian distributed node
    p(x|mu,sigma^2) = 1/sqrt(2*pi*sigma^2) * exp(-0.5*(x-mu)^2/(sigma^2) )
    log p(x|mu,sigma^2) =
    E[x] = mu
    var[x] = sigma^2
    H[x] = 0.5*log(sigma^2) + 0.5*(1+log(2pi))

    """
    def __init__(self, dim, mean, var, E=None, E2=None):
        Distribution.__init__(self, dim)

        # Initialise parameters
        mean = np.ones(dim) * mean
        var = np.ones(dim) * var
        self.params = { 'mean':mean, 'var':var }

        # Initialise expectations
        self.expectations = {}
        if E is None:
            self.updateExpectations()
        else:
            self.expectations['E'] = np.ones(dim)*E

        if E2 is not None:
            self.expectations['E2'] = np.ones(dim)*E2

        # float64 -> float32
        if config.use_float32: self.to_float32()

        # Check that dimensionalities match
        self.CheckDimensionalities()

    def updateExpectations(self):
        # Update first and second moments using current parameters
        E = self.params['mean']
        E2 = E**2 + self.params['var']
        self.expectations = { 'E':E, 'E2':E2 }

    def density(self, x):
        assert x.shape == self.dim, "Problem with the dimensionalities"
        # print stats.norm.pdf(x, loc=self.mean, scale=np.sqrt(self.var))
        return np.sum( (1/np.sqrt(2*np.pi*self.params['var'])) * np.exp(-0.5*(x-self.params['mean'])**2/self.params['var']) )

    def loglik(self, x):
        assert x.shape == self.dim, "Problem with the dimensionalities"
        # return np.log(stats.norm.pdf(x, loc=self.mean, scale=np.sqrt(self.var)))
        return np.sum( -0.5*np.log(2*np.pi) - 0.5*np.log(self.params['var']) -0.5*(x-self.params['mean'])**2/self.params['var'] )

    def entropy(self):
        return np.sum( 0.5*np.log(self.params['var']) + 0.5*(1+np.log(2*np.pi)) )

    def sample(self):
        return np.random.normal(self.params['mean'], np.sqrt(self.params['var']))
