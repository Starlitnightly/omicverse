import numpy as np
import scipy as s
import scipy.special as special
import scipy.stats as stats
from .basic_distributions import Distribution

from ..utils import *

class Binomial(Distribution):
    """
    Class to define Binomial distributions

    Equations:
    p(x|N,theta) = binom(N,x) * theta**(x) * (1-theta)**(N-x)
    log p(x|N,theta) = log(binom(N,x)) + x*theta + (N-x)*(1-theta)
    E[x] = N*theta
    var[x] = N*theta*(1-theta)
    """
    def __init__(self, dim, N, theta, E=None):
        Distribution.__init__(self, dim)

        # Initialise parameters
        theta = np.ones(dim)*theta
        N = np.ones(dim)*N
        self.params = { 'theta':theta, 'N':N }

        # Initialise expectations
        if E is None:
            self.updateExpectations()
        else:
            E = np.ones(dim)*E
            self.expectations = { 'E':E }

        # Check that dimensionalities match
        self.CheckDimensionalities()

    def updateExpectations(self):
        E = self.params["N"] * self.params["N"]
        self.expectations = { 'E':E }

    def density(self, x):
        assert x.shape == self.dim, "Problem with the dimensionalities"
        assert x.dtype == int, "x has to be an integer array"
        # return np.prod( stats.binom.pmf(x, self.params["N"], self.theta) )
        return np.prod( special.binom(self.params["N"],x) * self.params["theta"]**x * (1-self.params["theta"])**(self.params["N"]-x) )

    def loglik(self, x):
        assert x.shape == self.dim, "Problem with the dimensionalities"
        assert x.dtype == int, "x has to be an integer array"
        # print np.sum (stats.binom.logpmf(x, self.params["N"], self.theta) )
        return np.sum( np.log(special.binom(self.params["N"],x)) + x*np.log(self.params["theta"]) + (self.params["N"]-x)*np.log(1-self.params["theta"]) )

    def sample(self):
        return np.random.binomial(self.params['N'], self.params['theta'])
