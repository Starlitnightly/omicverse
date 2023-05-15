
from __future__ import division
import scipy as s
import numpy as np

from .basic_nodes import *
from ..distributions import *


"""
This module is used to define nodes that are inferred using the variational bayes framework.

All variational nodes share the property of having a lower bound associated with it.
We divide variational nodes into Constant and Unobserved.
The constant variational nodes are fixed and have no parameters or expectations
The unobserved variational nodes are learnt and have an associated P and Q distribution with corresponding expectations and parameters
"""

###########################################
## General classes for variational nodes ##
###########################################

class Variational_Node(Node):
    """
    Abstract class for a variational node in a Bayesian probabilistic model.
    Variational nodes can be observed (constant) or unobserved
    """
    def __init__(self, dim):
        Node.__init__(self,dim)

    def calculateELBO(self):
        # General method to calculate the ELBO of the node
        return 0.

###################################################################
## General classes for observed and unobserved variational nodes ##
###################################################################

class Constant_Variational_Node(Variational_Node,Constant_Node):
    """
    Abstract class for an observed/constant variational node in a Bayesian probabilistic model.
    """
    def __init__(self, dim, value):
        # SHOULD WE ALSO INITIALISE VARIATIONAL_NODE ..?
        Constant_Node.__init__(self, dim, value)

class Unobserved_Variational_Node(Variational_Node):
    """
    Abstract class for an unobserved variational node in a Bayesian probabilistic model.
    Unobserved variational nodes contain a prior P(X) and a variational Q(X) distribution,
    which will be stored as instances of Distribution() attributes .P and .Q, respectively.
    The distributions are in turn composed of parameters and expectations
    """
    def __init__(self, dim):
        Variational_Node.__init__(self, dim)
        self.P = None
        self.Q = None
    def updateExpectations(self, dist="Q"):
        # Method to update expectations of the node
        if dist == "Q": self.Q.updateExpectations()

    def getExpectation(self, dist="Q"):
        # Method to get the first moment (expectation) of the node
        if dist == "Q": expectations = self.Q.getExpectations()
        elif dist == "P": expectations = self.P.getExpectations()
        return expectations["E"]

    def getExpectations(self, dist="Q"):
        # Method to get all relevant moments of the node
        if dist == "Q": expectations = self.Q.getExpectations()
        elif dist == "P": expectations = self.P.getExpectations()
        return expectations

    def getParameters(self, dist="Q"):
        # Method to get all parameters of the node
        if dist == "Q": params = self.Q.getParameters()
        elif dist == "P": params = self.P.getParameters()
        return params

    def removeFactors(self, idx, axis=None):
        # Method to remove entire factors from the nodes

        if hasattr(self,"factors_axis"): axis = self.factors_axis
        if hasattr(self,"covariates"): self.covariates = self.covariates[np.arange(len(self.covariates)) != idx]
        if axis is not None:
            self.P.removeDimensions(axis=axis, idx=idx)
            self.Q.removeDimensions(axis=axis, idx=idx)
            self.updateDim(axis=axis, new_dim=self.dim[axis]-len(idx))

    def sample(self, distrib='P'):
        self.samp = self.P.sample()
        return self.samp


class Unobserved_Variational_Mixed_Node(Variational_Node):
    def __init__(self):
        pass

    def updateParameters(self, ix=None, ro=None):
        for node in self.nodes:
            node.updateParameters()

    def updateExpectations(self):
        for node in self.nodes:
            node.updateExpectations()

    def precompute(self, options=None):
        for node in self.nodes:
            node.precompute(options)

#######################################################
## Specific classes for unobserved variational nodes ##
#######################################################

class UnivariateGaussian_Unobserved_Variational_Node(Unobserved_Variational_Node):
    """
    Abstract class for a variational node where P(.) and Q(.)
    are both univariate Gaussian distributions.
    """
    def __init__(self, dim, pmean, pvar, qmean, qvar, qE=None, qE2=None):
	    # dim (2d tuple): dimensionality of the node
	    # pmean (nd array): the mean parameter of the P distribution
	    # qmean (nd array): the mean parameter of the Q distribution
	    # pvar (nd array): the variance parameter of the P distribution
	    # qvar (nd array): the variance parameter of the Q distribution
	    # qE (nd array): the initial first moment of the Q distribution
        Unobserved_Variational_Node.__init__(self, dim)
        # Initialise the P and Q distributions
        self.P = UnivariateGaussian(dim=dim, mean=pmean, var=pvar)
        self.Q = UnivariateGaussian(dim=dim, mean=qmean, var=qvar, E=qE, E2=qE2)

class MultivariateGaussian_Unobserved_Variational_Node(Unobserved_Variational_Node):
    """
    Abstract class for a variational node where P(.) and Q(.)
    are both multivariate Gaussian distributions.
    """
    def __init__(self, dim, pmean, pcov, qmean, qcov, qE=None, axis_cov=0):
        # dim (2d tuple): dimensionality of the node
        # pmean (nd array): the mean parameter of the P distribution
        # pcov (nd array): the covariance parameter of the P distribution
        # qmean (nd array): the mean parameter of the Q distribution
        # qcov (nd array): the covariance parameter of the Q distribution
        # qE (nd array): the initial first moment of the Q distribution
        Unobserved_Variational_Node.__init__(self, dim)

        # Initialise the P and Q distributions
        self.P = MultivariateGaussian(dim=dim, mean=pmean, cov=pcov, axis_cov=axis_cov)
        self.Q = MultivariateGaussian(dim=dim, mean=qmean, cov=qcov, E=qE, axis_cov=axis_cov)

class MultivariateGaussian_AO_Unobserved_Variational_Node(Unobserved_Variational_Node):
    """
    Abstract class for a variational node where P(.) and Q(.)
    are both multivariate Gaussian distributions using the Opper Archambeau reparamtrization for the variational.
    """
    def __init__(self, dim, pmean, pcov, qalpha, qlamb, qE=None, axis_cov=0):
        # dim (2d tuple): dimensionality of the node
        # pmean (nd array): the mean parameter of the P distribution
        # pcov (nd array): the covariance parameter of the P distribution
        # qalpha (nd array): the alpha parameter of the Q distribution (mean = K * alpha)
        # qlamb (nd array): the lambda parameter of the Q distribution (presicion = K^-1 + diag(lambda**2))
        # qE (nd array): the initial first moment of the Q distribution
        Unobserved_Variational_Node.__init__(self, dim)

        # Initialise the P and Q distributions
        self.P = MultivariateGaussian(dim=dim, mean=pmean, cov=pcov, axis_cov=axis_cov)
        self.Q = MultivariateGaussian_reparam(dim=dim, K=pcov, alpha=qalpha, lamb=qlamb, E=qE, axis_cov=0)


class UnivariateGaussian_Unobserved_Variational_Node_with_MultivariateGaussian_Prior(Unobserved_Variational_Node):
    """
    Abstract class for a variational node where P(.) is a multivariate Gaussian distribution
    and Q(.) is univariate Gaussian distribution.
    """
    def __init__(self, dim, pmean, pcov, qmean, qvar, axis_cov=0, qE=None, qE2=None):
	    # dim (2d tuple): dimensionality of the node
	    # pmean (nd array): the mean parameter of the P distribution
	    # qmean (nd array): the mean parameter of the Q distribution
	    # pcov (nd array): the covariance parameter of the P distribution
	    # qvar (nd array): the variance parameter of the Q distribution
	    # qE (nd array): the initial first moment of the Q distribution
        Unobserved_Variational_Node.__init__(self, dim)
        # Initialise the P and Q distributions
        self.P = MultivariateGaussian(dim=dim, axis_cov=axis_cov, mean=pmean, cov=pcov)
        self.Q = UnivariateGaussian(dim=dim, mean=qmean, var=qvar, E=qE, E2=qE2)

class Gamma_Unobserved_Variational_Node(Unobserved_Variational_Node):
    """
    Abstract class for a variational node where P(x) and Q(x) are both gamma distributions
    """
    def __init__(self, dim, pa, pb, qa, qb, qE=None, qlnE=None):
	    # dim (2d tuple): dimensionality of the node
	    # pa (nd array): the 'a' parameter of the P distribution
	    # qa (nd array): the 'b' parameter of the P distribution
	    # qa (nd array): the 'a' parameter of the Q distribution
	    # qb (nd array): the 'b' parameter of the Q distribution
	    # qE (nd array): the initial expectation of the Q distribution
        Unobserved_Variational_Node.__init__(self,dim)

        # Initialise the distributions
        self.P = Gamma(dim=dim, a=pa, b=pb)
        self.Q = Gamma(dim=dim, a=qa, b=qb, E=qE, lnE=qlnE)

class Bernoulli_Unobserved_Variational_Node(Unobserved_Variational_Node):
    """
    Abstract class for a variational node where P(.) and Q(.)
    are both bernoulli distributions.
    """
    def __init__(self, dim, ptheta, qtheta, qE=None):
	    # dim (2d tuple): dimensionality of the node
	    # ptheta (nd array): the 'theta' parameter of the P distribution
	    # qtheta (nd array): the 'theta' parameter of the Q distribution
	    # qE (nd array): initial first moment of the Q distribution
        Unobserved_Variational_Node.__init__(self,dim)

        # Initialise the distributions
        self.P = Bernoulli(dim=dim, theta=ptheta)
        self.Q = Bernoulli(dim=dim, theta=qtheta, E=qE)

class BernoulliGaussian_Unobserved_Variational_Node(Unobserved_Variational_Node):
    """
    Abstract class for a variational node where P(.) and Q(.)
    are joint gaussian-bernoulli distributions (see paper  Spike and Slab Variational Inference for
    Multi-Task and Multiple Kernel Learning by Titsias and Gredilla)
    """
    def __init__(self, dim,
        pmean_B0, pmean_B1, pvar_B0, pvar_B1, ptheta,
        qmean_B0, qmean_B1, qvar_B0, qvar_B1, qtheta, qEN_B0=None, qEN_B1=None, qEB=None):
	    # dim (2d tuple): dimensionality of the node
        # pmean (nd array): the mean parameter of the P distribution
        # pvar (nd array): the var parameter of the P distribution
        # ptheta (nd array): the theta parameter of the P distribution
        # qmean (nd array): the mean parameter of the Q distribution
        # qvar (nd array): the var parameter of the Q distribution
        # qtheta (nd array): the theta parameter of the Q distribution

        # Unobserved_Variational_Node.__init__(self,dim)
        Unobserved_Variational_Node.__init__(self,dim)

        # Initialise the P and Q distributions
        self.P = BernoulliGaussian(dim=dim, theta=ptheta, mean_B0=pmean_B0, var_B0=pvar_B0, mean_B1=pmean_B1, var_B1=pvar_B1)
        self.Q = BernoulliGaussian(dim=dim, theta=qtheta, mean_B0=qmean_B0, var_B0=qvar_B0, mean_B1=qmean_B1, var_B1=qvar_B1, EN_B0=qEN_B0, EN_B1=qEN_B1, EB=qEB)

class Beta_Unobserved_Variational_Node(Unobserved_Variational_Node):
    """
    Abstract class for a variational node where both P(.) and Q(.) are beta
    distributions
    """
    def __init__(self, dim, pa, pb, qa, qb, qE=None):
        # dim (2d tuple): dimensionality of the node
        # pa (nd array): the 'a' parameter of the P distribution
        # qa (nd array): the 'b' parameter of the P distribution
        # qa (nd array): the 'a' parameter of the Q distribution
        # qb (nd array): the 'b' parameter of the Q distribution
        # qE (nd array): the initial expectation of the Q distribution
        super().__init__(dim)

        # Initialise P and Q distributions
        self.P = Beta(dim, a=pa, b=pb)
        self.Q = Beta(dim, a=qa, b=qb, E=qE)
