"""
This module is used to define classes for statistical distributions

Each 'Distribution' class can store an arbitrary number of distributions of the same type, this is specified in the 'dim' argument

A 'Distribution' class has two main types of attributes: parameters and expectations. Both have to be defined when initialising a class.
Note that in some distributions (Gaussian mainly) a parameter is equal to an expectation. However, they are stored as separate
attributes and are not always necessarily equal due to E and M steps in the VB algorithm.

TO-DO:
- we should pass parametrs and expectations to Distribution() and perform sanity checks there
- Improve initialisation of Multivariate Gaussian
- Sanity checks on setter and getter functions
"""

import scipy as s
import numpy as np

# from mofapy2.config import settings
# from mofapy2.core.utils import *  # TODO prob not necessary ?

# General class for probability distributions
class Distribution(object):
    """ General class for a statistical distribution """
    def __init__(self, dim):
        self.dim = dim

    def density(self):
        """ General method to calculate density """
        pass
    def loglik(self):
        """ General method to calculate log likelihood """
        pass
    def sample(self):
        """ General method to sample from the distribution """
        pass
    def entropy(self):
        """ General method to calculate entropy """
        pass
    def updateExpectations(self):
        """ General method to update expectations """
        pass

    def getParameters(self):
        """ General getter function for parameters """
        return self.params

    def setParameters(self,**params):
        """ General setter function for parameters """
        self.params = params

    def getExpectation(self):
        """ General getter function for expectations """
        return self.expectations['E']

    def getExpectations(self):
        """ General setter function for expectations """
        return self.expectations

    def CheckDimensionalities(self):
        """ General method to do a sanity check on the dimensionalities """
        # p_dim = set(map(s.shape, self.params.values()))
        e_dim = set(map(s.shape, self.expectations.values()))
        # assert len(p_dim) == 1, "Parameters have different dimensionalities"
        assert len(e_dim) == 1, "Expectations have different dimensionalities"
        # assert e_dim == p_dim, "Parameters and Expectations have different dimensionality"

    def to_float32(self):
        """ Convert numpy arrays from float64 to float32 """
        for i in self.params.keys(): self.params[i] = self.params[i].astype(np.float32)
        for i in self.expectations.keys(): self.expectations[i] = self.expectations[i].astype(np.float32)


    def removeDimensions(self, axis, idx):
        """ General method to remove undesired dimensions

        PARAMETERS
        ----------
        axis: int
            axis from where to remove the elements
        idx: list or numpy array
            indices of the elements to remove
        """
        assert axis <= len(self.dim)
        assert np.all(idx < self.dim[axis])
        for k in self.params.keys(): self.params[k] = np.delete(self.params[k], idx, axis)
        for k in self.expectations.keys(): self.expectations[k] = np.delete(self.expectations[k], idx, axis)
        self.updateDim(axis=axis, new_dim=self.dim[axis]-len(idx))

    def updateDim(self, axis, new_dim):
        """ Method to update the dimensionality of a particular axis. 
        This method is a bit inefficient but we store dimensionalities with tuples and they cannot be modified

        PARAMETERS
        ----------
        axis: int
            axis to be updated
        new_dim: int
            updated dimensionality
        """
        dim = list(self.dim)
        dim[axis] = new_dim
        self.dim = tuple(dim)
