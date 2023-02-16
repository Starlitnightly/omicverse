import numpy as np
import scipy as s
from .basic_distributions import Distribution

from ... import config


class Bernoulli(Distribution):
    """
    Class to define Bernoulli distributions

    Equations:

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

        # float64 -> float32
        if config.use_float32: self.to_float32()

        # Check that dimensionalities match
        self.CheckDimensionalities()

    def updateExpectations(self):
        E = self.params['theta']
        self.expectations = { 'E':E }

    def density(self, x):
        assert x.shape == self.dim, "Problem with the dimensionalities"
        return np.prod( self.params['theta']**x * (1-self.params['theta'])**(1-x) )

    def loglik(self, x):
        assert x.shape == self.dim, "Problem with the dimensionalities"
        return np.sum( x*np.log(self.params['theta']) + (1-x)*np.log(1-self.params['theta']) )

    def sample(self):
        return np.random.binomial(1, self.params['theta'])
