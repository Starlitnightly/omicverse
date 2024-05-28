from __future__ import division
import numpy.ma as ma
import numpy as np
import scipy as s
import scipy.special as special

from ..utils import *
from .. import gpu_utils

# Import manually defined functions
from .variational_nodes import Gamma_Unobserved_Variational_Node

from ..distributions import *


class TauD_Node(Gamma_Unobserved_Variational_Node):
    def __init__(self, dim, pa, pb, qa, qb, groups, qE=None):
        super().__init__(dim=dim, pa=pa, pb=pb, qa=qa, qb=qb, qE=qE)

        self.groups = groups
        self.n_groups = len(np.unique(groups))

        assert self.n_groups == dim[0], "node dimension does not match number of groups"

    def precompute(self, options):
        """ Method to precompute some terms to speed up the calculations """

        # GPU mode
        gpu_utils.gpu_mode = options['gpu_mode']

        # Constant ELBO terms
        self.lbconst = np.sum(self.P.params['a']*np.log(self.P.params['b']) - special.gammaln(self.P.params['a']))

        # compute number of samples per group
        self.n_per_group = np.zeros(self.n_groups)
        for c in range(self.n_groups):
            self.n_per_group[c] = (self.groups == c).sum()

        self.mini_batch = None

    def getExpectations(self, expand=True):
        QExp = self.Q.getExpectations()
        if expand:
            expanded_E = QExp['E'][self.groups, :]
            expanded_lnE = QExp['lnE'][self.groups, :]
            return {'E': expanded_E, 'lnE': expanded_lnE}
        else:
            return QExp

    def getExpectation(self, expand=True):
        QExp = self.Q.getExpectation()
        if expand:
            return QExp[self.groups,:]
        else:
            return Qexp

    def define_mini_batch(self, ix):
        """ Method to define minibatch for the expectation """
        self.mini_batch = self.Q.getExpectation()[self.groups[ix],:]

    def get_mini_batch(self):
        if self.mini_batch is None:
            return self.getExpectation()
        else:
            return self.mini_batch

    def updateParameters(self, ix=None, ro=1.):
        """
        Public method to update the nodes parameters
        Optional arguments for stochastic updates are:
            - ix: list of indices of the minibatch
            - ro: step size of the natural gradient ascent
        """

        # Get expectations from other nodes
        Y = self.markov_blanket["Y"].get_mini_batch()
        mask = self.markov_blanket["Y"].getMask()
        Wtmp = self.markov_blanket["W"].getExpectations()
        Ztmp = self.markov_blanket["Z"].get_mini_batch()
        W, WW = Wtmp["E"], Wtmp["E2"]
        Z, ZZ = Ztmp["E"], Ztmp["E2"]

        # Collect parameters from the P distributions of this node
        P = self.P.getParameters()
        Pa, Pb = P['a'], P['b']

        # subset mini-batch
        if ix is None:
            groups = self.groups
        else:
            groups = self.groups[ix]

        # compute the updated parameters
        Qa, Qb = self._updateParameters(Y, W, WW, Z, ZZ, Pa, Pb, mask, ro, groups)

        self.Q.setParameters(a=Qa, b=Qb)

    def _updateParameters(self, Y, W, WW, Z, ZZ, Pa, Pb, mask, ro, groups):
        """ Hidden method to compute parameter updates """
        Q = self.Q.getParameters()
        Qa, Qb = Q['a'], Q['b']

        # Move matrices to the GPU
        Y_gpu = gpu_utils.array(Y)
        Z_gpu = gpu_utils.array(Z)
        W_gpu = gpu_utils.array(W).T

        # Calculate terms for the update (SPEED EFFICIENT, MEMORY INEFFICIENT FOR GPU)
        # ZW = Z_gpu.dot(W_gpu)
        # tmp = gpu_utils.asnumpy( gpu_utils.square(Y_gpu) \
        #     + gpu_utils.array(ZZ).dot(gpu_utils.array(WW.T)) \
        #     - gpu_utils.dot(gpu_utils.square(Z_gpu),gpu_utils.square(W_gpu)) + gpu_utils.square(ZW) \
        #     - 2*ZW*Y_gpu )
        # tmp[mask] = 0.

        # Calculate terms for the update (SPEED INEFFICIENT, MEMORY EFFICIENT FOR GPU)
        tmp = gpu_utils.asnumpy( gpu_utils.square(Y_gpu) \
            + gpu_utils.array(ZZ).dot(gpu_utils.array(WW.T)) \
            - gpu_utils.dot(gpu_utils.square(Z_gpu),gpu_utils.square(W_gpu)) + gpu_utils.square(Z_gpu.dot(W_gpu)) \
            - 2*Z_gpu.dot(W_gpu)*Y_gpu )
        tmp[mask] = 0.

        # Compute updates
        Qa *= (1-ro)
        Qb *= (1-ro)
        for g in range(self.n_groups):
            g_mask = (groups == g)

            n_batch = g_mask.sum()
            if n_batch == 0: continue

            # Calculate scaling coefficient for mini-batch
            coeff = self.n_per_group[g]/n_batch

            Qa[g,:] += ro * (Pa[g,:] + 0.5*coeff*(mask[g_mask,:].shape[0] - mask[g_mask,:].sum(axis=0)))
            Qb[g,:] += ro * (Pb[g,:] + 0.5*coeff*tmp[g_mask,:].sum(axis=0))

        return Qa, Qb

    def calculateELBO(self):
        """ Method to compute ELBO """
        
        # Collect parameters and expectations from current node
        P, Q = self.P.getParameters(), self.Q.getParameters()
        Pa, Pb, Qa, Qb = P['a'], P['b'], Q['a'], Q['b']
        QE, QlnE = self.Q.expectations['E'], self.Q.expectations['lnE']

        # Do the calculations
        lb_p = self.lbconst + np.sum((Pa-1.)*QlnE) - np.sum(Pb*QE)
        lb_q = np.sum(Qa*np.log(Qb)) + np.sum((Qa-1.)*QlnE) - np.sum(Qb*QE) - np.sum(special.gammaln(Qa))

        return lb_p - lb_q
