from __future__ import division
import numpy.ma as ma
import numpy as np
import scipy as s
import math
from ..utils import *
from .. import gpu_utils
from time import time


# Import manually defined functions
from .variational_nodes import BernoulliGaussian_Unobserved_Variational_Node
from .variational_nodes import UnivariateGaussian_Unobserved_Variational_Node

class Z_Node(UnivariateGaussian_Unobserved_Variational_Node):
    def __init__(self, dim, pmean, pvar, qmean, qvar, qE=None, qE2=None, weight_views = False):
        super().__init__(dim=dim, pmean=pmean, pvar=pvar, qmean=qmean, qvar=qvar, qE=qE, qE2=qE2)

        self.mini_batch = None
        self.factors_axis = 1
        self.weight_views = weight_views

    def precompute(self, options):
        """ Method to precompute terms to speed up computation """
        gpu_utils.gpu_mode = options['gpu_mode']

    def removeFactors(self, idx, axis=1):
        """ Method to remove inactive factors """
        super(Z_Node, self).removeFactors(idx, axis)
        # self.dim[1] -= len(idx)

    def define_mini_batch(self, ix):
        """ Method to define minibatch for the expectation """
        QExp = self.Q.getExpectations()
        self.mini_batch = {'E': QExp["E"][ix,:], 'E2': QExp["E2"][ix,:]}

    def get_mini_batch(self):
        """ Method to fetch minibatch """
        if self.mini_batch is None:
            return self.getExpectations()
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
        W = self.markov_blanket["W"].getExpectations()
        Y = self.markov_blanket["Y"].get_mini_batch()
        tau = self.markov_blanket["Tau"].get_mini_batch()
        mask = [self.markov_blanket["Y"].nodes[m].getMask() for m in range(len(Y))]
        if "MuZ" in self.markov_blanket:
            Mu =  self.markov_blanket['MuZ'].get_mini_batch()
        else:
            Mu = self.P.getParameters()["mean"]
            if ix is not None: Mu = Mu[ix]

        if "AlphaZ" in self.markov_blanket:
            Alpha = self.markov_blanket['AlphaZ'].get_mini_batch()
        else:
            Alpha = 1./self.P.params['var']
            if ix is not None: Alpha = Alpha[ix,:]

        # Get parameters of current node
        Q = self.Q.getParameters()
        Qmean, Qvar = Q['mean'], Q['var']
        if ix is not None:
            self.mini_batch = {}
            Qmean = Qmean[ix,:]
            Qvar = Qvar[ix,:]

        # Compute updates
        par_up = self._updateParameters(Y, W, tau, Mu, Alpha, Qmean, Qvar, mask)

        # Update parameters
        if ix is None:
            Q['mean'] = par_up['Qmean']
            Q['var'] = par_up['Qvar']
        else:
            self.mini_batch['E'] = par_up['Qmean']
            self.mini_batch['E2'] = np.square(par_up['Qmean']) + par_up['Qvar']

            Q['mean'][ix,:] = par_up['Qmean']
            Q['var'][ix,:] = par_up['Qvar']

        self.Q.setParameters(mean=Q['mean'], var=Q['var'])  # NOTE should not be necessary but safer to keep for now

    def _updateParameters(self, Y, W, tau, Mu, Alpha, Qmean, Qvar, mask):
        """ Hidden method to compute parameter updates """
        # Speed analysis: the pre-computation part does not benefit from GPU, but the next updates doe

        N = Y[0].shape[0]  # this is different from self.N for minibatch
        M = len(Y)
        K = self.dim[1]

        # Masking
        for m in range(M):
            tau[m][mask[m]] = 0.

        weights = [1] * M       
        if self.weight_views and M > 1:
            total_w = np.asarray([Y[m].shape[1] for m in range(M)]).sum()
            weights = np.asarray([total_w / (M * Y[m].shape[1]) for m in range(M)])
            weights = weights / weights.sum() * M
            # weights = [(total_w-Y[m].shape[1])/total_w * M / (M-1) for m in range(M)]

        # Precompute terms to speed up GPU computation
        foo = gpu_utils.array(np.zeros((N,K)))
        precomputed_bar = gpu_utils.array(np.zeros((N,K)))
        for m in range(M):
            tau_gpu = gpu_utils.array(tau[m])
            foo += weights[m] * gpu_utils.dot(tau_gpu, gpu_utils.array(W[m]["E2"]))
            bar_tmp1 = gpu_utils.array(W[m]["E"])
            bar_tmp2 = tau_gpu * gpu_utils.array(Y[m])
            precomputed_bar += weights[m] * gpu_utils.dot(bar_tmp2, bar_tmp1)
        foo = gpu_utils.asnumpy(foo)

        # Calculate variational updates
        for k in range(K):
            bar = gpu_utils.array(np.zeros((N,)))
            tmp_cp1 = gpu_utils.array(Qmean[:, np.arange(K) != k])
            for m in range(M):
                tmp_cp2 = gpu_utils.array(W[m]["E"][:, np.arange(K) != k].T)

                bar_tmp1 = gpu_utils.array(W[m]["E"][:,k])
                bar_tmp2 = gpu_utils.array(tau[m])*(-gpu_utils.dot(tmp_cp1, tmp_cp2))

                bar += weights[m] * gpu_utils.dot(bar_tmp2, bar_tmp1)
            bar += precomputed_bar[:,k]
            bar = gpu_utils.asnumpy(bar)

            Qvar[:, k] = 1. / (Alpha[:, k] + foo[:,k])
            Qmean[:, k] = Qvar[:, k] * (bar + Alpha[:, k] * Mu[:, k])

        # Save updated parameters of the Q distribution
        return {'Qmean': Qmean, 'Qvar':Qvar}

    def calculateELBO(self):

        # Collect parameters and expectations of current node
        Qpar, Qexp = self.Q.getParameters(), self.Q.getExpectations()
        Qmean, Qvar = Qpar['mean'], Qpar['var']
        QE, QE2 = Qexp['E'], Qexp['E2']

        if "MuZ" in self.markov_blanket:
            PE, PE2 = self.markov_blanket['MuZ'].getExpectations()['E'], \
                      self.markov_blanket['MuZ'].getExpectations()['E2']
        else:
            PE, PE2 = self.P.getParameters()["mean"], np.zeros((self.dim[0], self.dim[1]))

        if 'AlphaZ' in self.markov_blanket:
            Alpha = self.markov_blanket['AlphaZ'].getExpectations(expand=True)
        else:
            Alpha = dict()
            Alpha['E'] = 1./self.P.params['var']
            Alpha['lnE'] = np.log(1./self.P.params['var'])

        # compute term from the exponential in the Gaussian
        tmp1 = 0.5 * QE2 - PE * QE + 0.5 * PE2
        tmp1 = -(tmp1 * Alpha['E']).sum()

        # compute term from the precision factor in front of the Gaussian
        tmp2 = 0.5 * Alpha["lnE"].sum()

        lb_p = tmp1 + tmp2
        lb_q = -(np.log(Qvar).sum() + self.dim[0] * self.dim[1]) / 2.

        return lb_p - lb_q

class SZ_Node(BernoulliGaussian_Unobserved_Variational_Node):
    def __init__(self, dim, pmean_T0, pmean_T1, pvar_T0, pvar_T1, ptheta, qmean_T0, qmean_T1, qvar_T0, qvar_T1, qtheta, qEZ_T0=None, qEZ_T1=None, qET=None, weight_views =False):
        super().__init__(dim, pmean_T0, pmean_T1, pvar_T0, pvar_T1, ptheta, qmean_T0, qmean_T1, qvar_T0, qvar_T1, qtheta, qEZ_T0, qEZ_T1, qET)

        self.mini_batch = None
        self.factors_axis = 1
        self.weight_views = weight_views

    def precompute(self, options):
        """ Method to precompute some terms to speed up the calculations """

        # GPU mode
        gpu_utils.gpu_mode = options['gpu_mode']

    def removeFactors(self, idx, axis=1):
        """ Method to remove inactive factors """
        super(SZ_Node, self).removeFactors(idx, axis)
        # self.dim[1] -= len(idx)

    def define_mini_batch(self, ix):
        """ Method to define minibatch for the expectation """
        QExp = self.Q.getExpectations()
        self.mini_batch = {
        'E': QExp["E"][ix,:], 
        'E2': QExp["E2"][ix,:],
        'EB': QExp["EB"][ix,:],
        'EN': QExp["EN"][ix,:],
        'ENN': QExp["ENN"][ix,:]
        }

    def get_mini_batch(self):
        if self.mini_batch is None:
            return self.getExpectations()
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
        W = self.markov_blanket["W"].getExpectations()
        Y = self.markov_blanket["Y"].get_mini_batch()
        tau = self.markov_blanket["Tau"].get_mini_batch()
        mask = [self.markov_blanket["Y"].nodes[m].getMask() for m in range(len(Y))]

        if "AlphaZ" in self.markov_blanket:
            Alpha = self.markov_blanket['AlphaZ'].get_mini_batch()
        else:
            Alpha = 1./self.P.params['var_B1']
            if ix is not None:
                Alpha = Alpha[ix,:]

        thetatmp = self.markov_blanket['ThetaZ'].get_mini_batch()
        theta_lnE, theta_lnEInv = thetatmp['lnE'], thetatmp['lnEInv']
        
        # Get expectations and parameters from current node
        Q = self.Q.getParameters()
        SZ = self.Q.getExpectations()["E"]
        Qmean_T1, Qvar_T1, Qtheta = Q['mean_B1'], Q['var_B1'], Q['theta']
        if ix is not None:
            self.mini_batch = {}
            Qmean_T1 = Qmean_T1[ix,:]
            Qvar_T1 = Qvar_T1[ix,:]
            Qtheta = Qtheta[ix,:]
            SZ = SZ[ix,:]


        # Compute the updates
        par_up = self._updateParameters(Y, W, tau, mask, Alpha, Qmean_T1, Qvar_T1, Qtheta, SZ, theta_lnE, theta_lnEInv)

        # Update the parameters (this is not very clean...)
        if ix is None:
            Q['mean_B1'] = par_up['mean_B1']
            Q['var_B1'] = par_up['var_B1']
            Q['theta'] = par_up['theta']
            Q['var_B0'] = 1. / Alpha
        else:
            Q['mean_B1'][ix,:] = par_up['mean_B1']
            Q['var_B1'][ix,:] = par_up['var_B1']
            Q['theta'][ix,:] = par_up['theta']
            Q['var_B0'][ix, :] = 1. / Alpha

            self.mini_batch['EB']  = par_up['theta']
            self.mini_batch['E']   = par_up['mean_B1'] * par_up['theta']
            self.mini_batch['E2']  = par_up['theta'] * (np.square(par_up['mean_B1']) + par_up['var_B1'])
            self.mini_batch['ENN'] = par_up['theta'] * (np.square(par_up['mean_B1']) + par_up['var_B1']) + \
                                     (1-par_up['theta']) * Q['var_B0'][ix, :]

        # self.Q.setParameters(mean_B0=np.zeros((self.dim[0], self.dim[1])), var_B0=Q['var_B0'],
        #                      mean_B1=Q['mean_B1'], var_B1=Q['var_B1'], theta=Q['theta'])  # NOTE should not be necessary but safer to keep for now
    def _updateParameters(self, Y, W, tau, mask, Alpha, Qmean_T1, Qvar_T1, Qtheta, SZ, theta_lnE, theta_lnEInv):
        """ Hidden method to compute parameter updates """

        # Mask matrices
        for m in range(len(Y)):
            tau[m][mask[m]] = 0.

        # Precompute terms to speed up GPU computation
        N = Qmean_T1.shape[0]
        M = len(Y)
        K = self.dim[1]

        weights = [1] * M       
        if self.weight_views and M > 1:
            total_w = np.asarray([Y[m].shape[1] for m in range(M)]).sum()
            # weights = [(total_w-Y[m].shape[1])/total_w * M / (M-1) for m in range(M)]
            weights = np.asarray([total_w / (M * Y[m].shape[1]) for m in range(M)])
            weights = weights / weights.sum() * M


        # term4_tmp1 = [ gpu_utils.array(Alpha[:,k]) for k in range(self.dim[1]) ]
        # term4_tmp2 = [ gpu_utils.array(Alpha[:,k]) for k in range(self.dim[1]) ]
        # term4_tmp3 = [ gpu_utils.array(Alpha[:,k]) for k in range(self.dim[1]) ]
        # for m in range(M):
        #     tau_gpu = gpu_utils.array(tau[m])
        #     Y_gpu = gpu_utils.array(Y[m])
        #     for k in range(K):
        #         Wk_gpu = gpu_utils.array(W[m]["E"][:,k])
        #         WWk_gpu = gpu_utils.array(W[m]["E2"][:,k])
        #         term4_tmp1[k] += gpu_utils.dot(tau_gpu*Y_gpu, Wk_gpu)
        #         term4_tmp3[k] += gpu_utils.dot(tau_gpu, WWk_gpu)
        # del tau_gpu, Y_gpu, Wk_gpu, WWk_gpu

        term4_tmp1 = gpu_utils.array( np.zeros((N,K))+Alpha )
        term4_tmp2 = gpu_utils.array( np.zeros((N,K))+Alpha )
        term4_tmp3 = gpu_utils.array( np.zeros((N,K))+Alpha ) 

        for m in range(M):
            tau_gpu = gpu_utils.array(tau[m])
            Y_gpu = gpu_utils.array(Y[m])
            W_gpu = gpu_utils.array(W[m]["E"])
            WW_gpu = gpu_utils.array(W[m]["E2"])
            term4_tmp1 +=  weights[m] * gpu_utils.dot(tau_gpu*Y_gpu, W_gpu)
            term4_tmp3 +=  weights[m] * gpu_utils.dot(tau_gpu, WW_gpu)
        del tau_gpu, Y_gpu, W_gpu, WW_gpu

        # Update each latent variable in turn (notice that the update of Z[,k] depends on the other values of Z!)
        for k in range(K):
            term1 = (theta_lnE - theta_lnEInv)[:, k]
            term2 = 0.5 * np.log(Alpha[:,k])

            for m in range(M):
                tau_gpu = gpu_utils.array(tau[m])
                Wk_gpu = gpu_utils.array(W[m]["E"][:,k])
                term4_tmp2_tmp = (tau_gpu * gpu_utils.dot(gpu_utils.array(SZ[:, np.arange(K) != k]),
                                (Wk_gpu * gpu_utils.array(W[m]["E"][:, np.arange(K) != k].T)))).sum(axis=1)
                term4_tmp2[:,k] +=  weights[m] * term4_tmp2_tmp
                del tau_gpu, Wk_gpu, term4_tmp2_tmp
            
            # term4_tmp3[k] += Alpha[:,k]
            term3 = gpu_utils.asnumpy( 0.5*gpu_utils.log(term4_tmp3[:,k]) )
            term4 = gpu_utils.asnumpy( 0.5*gpu_utils.divide(gpu_utils.square(term4_tmp1[:,k] - term4_tmp2[:,k]), term4_tmp3[:,k]) )

            # Update S
            # NOTE there could be some precision issues in T --> loads of 1s in result
            Qtheta[:, k] = 1. / (1. + np.exp(-(term1 + term2 - term3 + term4)))
            Qtheta[:,k] = np.nan_to_num(Qtheta[:,k])

            # Update Z
            Qvar_T1[:, k] = gpu_utils.asnumpy( 1. / term4_tmp3[:,k] )
            Qmean_T1[:, k] = Qvar_T1[:, k] * gpu_utils.asnumpy(term4_tmp1[:,k] - term4_tmp2[:,k])

            # Update Expectations for the next iteration
            SZ[:, k] = Qtheta[:, k] * Qmean_T1[:, k]

        return {'mean_B1': Qmean_T1, 'var_B1': Qvar_T1, 'theta': Qtheta}

    def calculateELBO(self):

        # Collect parameters and expectations
        Qpar, Qexp = self.Q.getParameters(), self.Q.getExpectations()
        T, ZZ = Qexp["EB"], Qexp["ENN"]
        Qvar = Qpar['var_B1']
        theta = self.markov_blanket['ThetaZ'].getExpectations(expand=True)

        # Get ARD sparsity or prior variance
        if "AlphaZ" in self.markov_blanket:
            alpha = self.markov_blanket['AlphaZ'].getExpectations(expand=True)
        else:
            alpha = dict()
            alpha['E'] = 1./self.P.params['var_B1']
            alpha['lnE'] = np.log(1./self.P.params['var_B1'])

        # Calculate ELBO for Z
        lb_pz = (alpha["lnE"].sum() - np.sum(alpha["E"] * ZZ)) / 2.
        lb_qz = -0.5 * self.dim[1] * self.dim[0] - 0.5 * (T * np.log(Qvar) + (1. - T) * np.log(1. / alpha["E"])).sum()
        lb_z = lb_pz - lb_qz

        # Calculate ELBO for T
        # T[T<1e-10] = 1e-10
        # T[T>0.9999999] = 0.9999999
        lb_pt = T * theta['lnE'] + (1. - T) * theta['lnEInv']
        lb_qt = T * np.log(T) + (1. - T) * np.log(1. - T)

        # Replace NAs (due to theta=1) with zeros
        lb_pt[np.isnan(lb_pt)] = 0.
        lb_qt[np.isnan(lb_qt)] = 0.
        
        lb_t = np.sum(lb_pt) - np.sum(lb_qt)

        return lb_z + lb_t
