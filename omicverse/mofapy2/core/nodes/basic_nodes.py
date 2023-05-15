"""
Module to define general nodes in a Bayesian network

All nodes have two main attributes:
- dim: dimensionality of the node
- markov_blanket: the markov blanket of the node

"""

import scipy as s
import numpy as np

from ... import config


class Node(object):
    """General class for a node in a Bayesian network

    PARAMETERS
    ----------
    dim: tuple
        Dimensionality of the node
    """
    def __init__(self, dim):
        self.dim = dim

    def addMarkovBlanket(self, **kwargs):
        """Method to define the Markov blanket of the node"""
        if hasattr(self, 'markov_blanket'):
            for k,v in kwargs.items():
                if k in self.markov_blanket.keys():
                    print("Error: " + str(k) + " is already in the markov blanket of " + str(self))
                    exit(1)
                else:
                    self.markov_blanket[k] = v
        else:
            self.markov_blanket = kwargs

    def getMarkovBlanket(self):
        """ Method to return the Markov blanket of the node """
        return self.markov_blanket

    def update(self, ix=None, ro=1.):
        """ General method to update both parameters and expectations of the node """
        self.updateParameters(ix, ro)
        self.updateExpectations()

    def updateExpectations(self):
        """ General method to update the expectations of a node """
        pass

    def getDimensions(self):
        """ Method to return dimensionality of the node """
        return self.dim

    def getExpectations(self):
        """ General method to get all expectations of a node """
        pass

    def getExpectation(self):
        """ General method to get the first moment (expectation) of a node """
        pass

    def updateParameters(self, ix=None, ro=1.):
        """ General function to update parameters of the node """
        pass

    def getParameters(self):
        """ General function to get the parameters of the node """
        pass

    def updateDim(self, axis, new_dim):
        """ Method to update the dimensionality of a node
        PARAMETERS
        ----------
        axis:
        new_dim:
        """
        dim = list(self.dim)
        dim[axis] = new_dim
        self.dim = tuple(dim)

    def precompute(self, options=None):
        pass


class Constant_Node(Node):
    """ General class for a constant node in a Bayesian network
    Constant nodes do not have expectations or parameters but just values.
    However, for technical reasons the constant cvalues are defined as expectations
    """
    def __init__(self, dim, value):
        self.dim = dim
        if isinstance(value,(int,float)):
            if config.use_float32:
                self.value = value * np.ones(dim, dtype=np.float32)
            else:
                self.value = value * np.ones(dim, dtype=np.float64)
        else:
            assert value.shape == dim, "dimensionality mismatch"
            self.value = value

    def getValue(self):
        """ Method to return the values of the node """
        return self.value

    def getExpectation(self):
        """ Method to return the first moment of the node, which just points to the values """
        return self.getValue()

    def getExpectations(self):
        """ Method to return the expectations of the node, which just points to the values """
        return { 'E':self.getValue(), 'lnE':np.log(self.getValue()), 'E2':self.getValue()**2 }

    def removeFactors(self, idx, axis=None):
        if hasattr(self,"factors_axis"): axis = self.factors_axis
        if axis is not None:
            self.value = np.delete(self.value, idx, axis)
            self.updateDim(axis=axis, new_dim=self.dim[axis]-len(idx))

    def sample(self, distrib='P'):
        self.samp = self.value
        return self.samp
