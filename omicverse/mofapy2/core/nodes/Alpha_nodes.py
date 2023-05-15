
from __future__ import division
import numpy.ma as ma
import numpy as np
import scipy as s
import scipy.special as special

# Import manually defined functions
from .variational_nodes import Gamma_Unobserved_Variational_Node

class AlphaW_Node(Gamma_Unobserved_Variational_Node):
    def __init__(self, dim, pa, pb, qa, qb, qE=None, qlnE=None):
        super().__init__(dim=dim, pa=pa, pb=pb, qa=qa, qb=qb, qE=qE, qlnE=qlnE)

    def precompute(self, options=None):
        """ Method to precompute some terms to speed up the calculations """
        self.factors_axis = 0

    def getExpectations(self, expand=False):
        QExp = self.Q.getExpectations()
        if expand:
            D = self.markov_blanket['W'].dim[0]
            expanded_E = np.repeat(QExp['E'][None, :], D, axis=0)
            expanded_lnE = np.repeat(QExp['lnE'][None, :], D, axis=0)
            return {'E': expanded_E, 'lnE': expanded_lnE}
        else:
            return QExp

    def getExpectation(self, expand=False):
        return self.getExpectations(expand)['E']

    def updateParameters(self, ix=None, ro=1.):
        """
        Public method to update the nodes parameters
        Optional arguments for stochastic updates are:
            - ix: list of indices of the minibatch
            - ro: step size of the natural gradient ascent
        """

        # NOTE Here we use a step of 1 because higher in the hierarchy means useless to decay the step size as W would converge anyway
        self._updateParameters()

    def _updateParameters(self):
        """ Hidden method to compute parameter updates """

        # Collect expectations from other nodes
        Wtmp = self.markov_blanket["W"].getExpectations()
        W  = Wtmp["E"]
        if "ENN" in Wtmp:
            WW = Wtmp["ENN"]
        else:
            WW = Wtmp["E2"]

        # Collect parameters from the P distribution of this node
        P = self.P.getParameters()
        Pa, Pb = P['a'], P['b']

        # Perform updates
        Qa = Pa + 0.5*W.shape[0]
        Qb = Pb + 0.5*WW.sum(axis=0)
        
        # Qa = Pa + 0.5*W.shape[0]*W.shape[1]
        # Qb = Pb + 0.5*WW.sum()

        self.Q.setParameters(a=Qa, b=Qb)

    def calculateELBO(self):
        """ Method to compute ELBO """

        # Collect parameters and expectations
        P,Q = self.P.getParameters(), self.Q.getParameters()
        Pa, Pb, Qa, Qb = P['a'], P['b'], Q['a'], Q['b']
        QE, QlnE = self.Q.getExpectations()['E'], self.Q.getExpectations()['lnE']

        # Do the calculations
        lb_p = (Pa*np.log(Pb)).sum() - special.gammaln(Pa).sum() + ((Pa-1.)*QlnE).sum() - (Pb*QE).sum()
        lb_q = (Qa*np.log(Qb)).sum() - special.gammaln(Qa).sum() + ((Qa-1.)*QlnE).sum() - (Qb*QE).sum()

        return lb_p - lb_q

class AlphaZ_Node(Gamma_Unobserved_Variational_Node):
    def __init__(self, dim, pa, pb, qa, qb, groups, qE=None, qlnE=None):
        super().__init__(dim=dim, pa=pa, pb=pb, qa=qa, qb=qb, qE=qE, qlnE=qlnE)
        
        self.groups = groups
        self.n_groups = len(np.unique(groups))
        assert self.n_groups == dim[0], "node dimension does not match number of groups"

        self.mini_batch = None

    def precompute(self, options=None):
        """ Method to precompute some terms to speed up the calculations """

        # Define axis of factors (to drop them)
        self.factors_axis = 1

        self.n_per_group = np.zeros(self.n_groups)
        for c in range(self.n_groups):
            self.n_per_group[c] = (self.groups == c).sum()

    def getExpectations(self, expand=False):
        QExp = self.Q.getExpectations()
        if expand:
            return {'E': QExp['E'][self.groups, :], 'lnE': QExp['lnE'][self.groups, :] }
        else:
            return {'E': QExp['E'], 'lnE': QExp['lnE']}

    def getExpectation(self, expand=False):
        return self.getExpectations(expand)['E']

    def define_mini_batch(self, ix):
        QExp = self.Q.getExpectations()
        self.mini_batch = QExp['E'][self.groups[ix], :]

    def get_mini_batch(self):
        if self.mini_batch is None:
            return self.getExpectation(expand=True)
        else:
            return self.mini_batch

    def updateParameters(self, ix=None, ro=1.):
        """
        Public method to update the nodes parameters
        Optional arguments for stochastic updates are:
            - ix: list of indices of the minibatch
            - ro: step size of the natural gradient ascent
        """
        Ztmp = self.markov_blanket["Z"].get_mini_batch()
        if 'ENN' in Ztmp:
            ZZ = Ztmp["ENN"]
        else:
            ZZ = Ztmp["E2"]

        # Collect parameters from the P distributions of this node
        P = self.P.getParameters()
        Pa, Pb = P['a'], P['b']

        # subset mini-batch
        if ix is None:
            groups = self.groups
        else:
            groups = self.groups[ix]

        # compute the updated parameters
        self._updateParameters(Pa, Pb, ZZ, groups, ro)

        # self.Q.setParameters(a=Qa, b=Qb)

    def _updateParameters(self, Pa, Pb, ZZ, groups, ro):
        """ Hidden method to compute parameter updates """

        Q = self.Q.getParameters()
        Q['a'] *= (1-ro)
        Q['b'] *= (1-ro)

        for c in range(self.n_groups):
            mask = (groups == c)

            # Compute anti-bias coefficient for stochastic inference
            n_batch = mask.sum()
            if n_batch == 0: continue
            coeff = self.n_per_group[c]/n_batch

            Q['a'][c,:] += ro * (Pa[c,:] + 0.5 * self.n_per_group[c])  # TODO should be precomputed
            Q['b'][c,:] += ro * (Pb[c,:] + 0.5 * coeff * ZZ[mask,:].sum(axis=0))

    def calculateELBO(self):
        """ Method to compute ELBO """
        
        # Collect parameters and expectations
        P,Q = self.P.getParameters(), self.Q.getParameters()
        Pa, Pb, Qa, Qb = P['a'], P['b'], Q['a'], Q['b']
        QE, QlnE = self.Q.getExpectations()['E'], self.Q.getExpectations()['lnE']

        # Do the calculations
        lb_p = (Pa*np.log(Pb)).sum() - special.gammaln(Pa).sum() + ((Pa-1.)*QlnE).sum() - (Pb*QE).sum()
        lb_q = (Qa*np.log(Qb)).sum() - special.gammaln(Qa).sum() + ((Qa-1.)*QlnE).sum() - (Qb*QE).sum()

        return lb_p - lb_q
