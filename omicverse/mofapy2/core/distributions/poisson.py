import numpy as np
import scipy as s
import scipy.stats as stats
from .basic_distributions import Distribution

from ..utils import *

class Poisson(Distribution):
    """
    Class to define Poisson distributions.

    Equations:
    p(x|theta) = theta**x * exp(-theta) * 1/theta!
    log p(x|a,b) = x*theta - theta - log(x!)
    E[x] = theta
    var[x] = theta
    """

    def __init__(self, dim, theta, E=None):
        Distribution.__init__(self, dim)

        # Initialise parameters
        theta = np.ones(dim) * theta
        self.params = { 'theta':theta }

        # Initialise expectations
        if E is None:
            self.updateExpectations()
        else:
            self.expectations = { 'E':np.ones(dim)*E }

        # Check that dimensionalities match
        self.CheckDimensionalities()

    def updateExpectations(self):
        E = self.params['theta']
        self.expectations = { 'E':E }

    def density(self, x):
        assert x.shape == self.dim, "Problem with the dimensionalities"
        assert x.dtype == int, "x has to be an integer array"
        theta = self.params['theta'].flatten()
        x = x.flatten()
        # return np.prod (stats.poisson.pmf(x,theta) )
        return np.prod( np.divide(theta**x * np.exp(-theta),s.misc.factorial(x)) )

    def loglik(self, x):
        assert x.shape == self.dim, "Problem with the dimensionalities"
        assert x.dtype == int, "x has to be an integer array"
        theta = self.params['theta'].flatten()
        x = x.flatten()
        # return np.log( np.prod (stats.poisson.pmf(x,theta) ))
        return np.sum( x*np.log(theta) - theta - np.log(s.misc.factorial(x)) )
