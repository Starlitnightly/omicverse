from __future__ import division
import numpy.ma as ma
import numpy as np
import scipy as s
import math

from ..utils import dotd
from .. import gpu_utils

# Import manually defined functions
from .variational_nodes import Constant_Variational_Node

class Y_Node(Constant_Variational_Node):
    def __init__(self, dim, value, groups):
        Constant_Variational_Node.__init__(self, dim, value)

        # Define groups
        assert len(groups) == dim[0]
        self.groups = groups
        self.n_groups = len(np.unique(groups))

        # Mask missing values
        self.mask = self.mask()

        self.mini_batch = None
        self.mini_mask = None

    def precompute(self, options=None):
        """ Method to precompute some terms to speed up the calculations """

        # Dimensionalities
        self.N = self.dim[0] - self.getMask().sum(axis=0)
        self.D = self.dim[1] - self.getMask().sum(axis=1)

        # GPU mode
        gpu_utils.gpu_mode = options['gpu_mode']

        # Do TauTrick to speed up ELBO computation?
        # Important: this assumes that the Tau update has been done prior to calculating elbo of Y
        self.TauTrick = True

        # Constant ELBO terms
        self.likconst = -0.5 * np.sum(self.N) * np.log(2.*np.pi)

    def mask(self):
        """ Method to mask missing observations """
        mask = np.isnan(self.value)
        self.value[mask] = 0.
        return mask

    def getMask(self, full=False):
        """ Get method for the mask """
        if full:
            return self.mask
        else:
            if self.mini_batch is None:
                return self.mask
            else:
                return self.mini_mask
            
    def define_mini_batch(self, ix):
        """ Method to define a mini-batch (only for stochastic inference) """
        self.mini_batch = self.value[ix,:]
        self.mini_mask = self.mask[ix,:]

    def get_mini_batch(self):
        """ Method to retrieve a mini-batch (only for stochastic inference) """
        if self.mini_batch is None:
            return self.getExpectation()
        else:
            return self.mini_batch

    def calculateELBO(self):
        """ Method to calculate evidence lower bound """
        mask = self.mask
        Tau = self.markov_blanket["Tau"].getExpectations(expand=False)
        elbo = self.likconst
        groups = self.markov_blanket["Tau"].groups

        if self.TauTrick: 
            tauQ_param = self.markov_blanket["Tau"].getParameters("Q")
            tauP_param = self.markov_blanket["Tau"].getParameters("P")
            for g in range(len(np.unique(groups))):
                idx = groups==g
                foo = (~mask[idx,:]).sum(axis=0)
                elbo += 0.5*(Tau["lnE"][g,:]*foo).sum() - np.dot(Tau["E"][g,:],(tauQ_param["b"][g,:] - tauP_param["b"][g,:]))

        else:
            Y = self.getExpectation()
            Wtmp = self.markov_blanket["W"].getExpectations()
            Ztmp = self.markov_blanket["Z"].getExpectations()
            W, WW = Wtmp["E"].T, Wtmp["E2"].T
            Z, ZZ = Ztmp["E"], Ztmp["E2"]

            tmp = np.square(Y) \
                + ZZ.dot(WW) \
                - np.dot(np.square(Z),np.square(W)) + np.square(Z.dot(W)) \
                - 2*Z.dot(W)*Y 
            tmp *= 0.5
            tmp[mask] = 0.
            
            for g in range(len(np.unique(groups))):
                idx = groups==g
                foo = (~mask[idx,:]).sum(axis=0)
                elbo += 0.5*(Tau["lnE"][g,:]*foo).sum() - (Tau["E"][g,:]*tmp[idx,:]).sum()
        return elbo

