"""
Module to define updates for non-conjugate matrix factorisation models.
Reference: 'Fast Variational Bayesian Inference for Non-Conjugate Matrix Factorisation models' by Seeger and Bouchard (2012)
PseudoY: general class for pseudodata
    PseudoY_seeger: general class for pseudodata using seeger aproach
        Poisson_PseudoY: Poisson likelihood
        Bernoulli_PseudoY: Bernoulli likelihood
        Binomial_PseudoY: Binomial likelihood (TO FINISH)
    PseudoY_Jaakkola: NOT IMPLEMENTED
        Bernoulli_PseudoY_Jaakkola: bernoulli likelihood for Jaakkola approach (see REF)
        Tau_Jaakkola:
"""

from __future__ import division
import scipy as s
import numpy.ma as ma
import numpy as np

from .variational_nodes import Unobserved_Variational_Node, Unobserved_Variational_Mixed_Node
from .basic_nodes import *
from .Y_nodes import Y_Node
from .Tau_nodes import TauD_Node

from .. import gpu_utils
from ..utils import sigmoid, lambdafn


##############################
## General pseudodata nodes ##
##############################

class PseudoY(Unobserved_Variational_Node):
    """ General class for pseudodata nodes """
    def __init__(self, dim, obs, groups, params=None, E=None):
        """
        PARAMETERS
        ----------
         dim (2d tuple): dimensionality of each view
         obs (ndarray): observed data
         params:
         E (ndarray): initial expected value of pseudodata
        """
        Unobserved_Variational_Node.__init__(self, dim)

        # Initialise observed data
        assert obs.shape == dim, "Problems with the dimensionalities"
        self.obs = obs

        # Set groups
        self.groups = groups

        # Initialise parameters
        if params is not None:
            assert type(self.params) == dict
            self.params = params
        else:
            self.params = {}

        # Create a boolean mask of the data to handle missing values
        self.mask = ma.getmask( ma.masked_invalid(self.obs) )
        self.obs[np.isnan(self.obs)] = 0.

        # Initialise expectation
        if E is not None:
            assert E.shape == dim, "Problems with the dimensionalities"
            E[self.mask] = 0.
        self.E = E

    def updateParameters(self, ix=None, ro=None):
        pass

    def getMask(self, full=True):
        # Currently always full as stochastic inference not implemented for Gaussian nodes
        #TODO: change when implementing SVI for non-Gaussian
        return self.mask

    def precompute(self, options=None):
        # Precompute some terms to speed up the calculations
        pass

    def updateExpectations(self):
        print("Error: expectation updates for pseudodata node depend on the type of likelihood. They have to be specified in a new class.")
        exit()

    def getExpectation(self, expand=True):
        return self.E

    def define_mini_batch(self, ix):
        """ Method to define a mini-batch (only for stochastic inference) """
        # self.mini_batch = self.E[ix,:]
        pass

    # TODO change: define a proper mini-batch !!
    def get_mini_batch(self, expand=True):
        return self.getExpectation(expand)

    def getExpectations(self, expand=True):
        return { 'E':self.getExpectation(expand) }

    # def getObservations(self):
    #     return self.obs

    def getValue(self):
        return self.obs

    def getParameters(self):
        return self.params

    def calculateELBO(self):
        print("Not implemented")
        exit()

##################
## Seeger nodes ##
##################

class PseudoY_Seeger(PseudoY):
    """ General class for pseudodata nodes using the seeger approach """
    def __init__(self, dim, obs, groups, params=None, E=None):
        # Inputs:
        #  dim (2d tuple): dimensionality of each view
        #  obs (ndarray): observed data
        #  E (ndarray): initial expected value of pseudodata
        PseudoY.__init__(self, dim=dim, obs=obs, groups=groups, params=params, E=E)

    def updateParameters(self, ix=None, ro=None):
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()
        # self.params["zeta"] = np.dot(Z,W.T)
        self.params["zeta"] = gpu_utils.dot( gpu_utils.array(Z),gpu_utils.array(W).T )

class Tau_Seeger(Constant_Node):
    """
    """
    def __init__(self, dim, value):
        Constant_Node.__init__(self, dim=dim, value=value)

    def getValue(self):
        return self.value

    def define_mini_batch(self, ix):
        """ Method to define a mini-batch (only for stochastic inference) """
        # self.mini_batch = self.value[ix,:]
        pass

    # TODO change: define a proper min-batch !!
    def get_mini_batch(self, expand=True):
        return self.getExpectation(expand)

    def getExpectation(self, expand=True):
        return self.getValue()

    def getExpectations(self, expand=True):
        return { 'E':self.getExpectation(expand) }

class Poisson_PseudoY(PseudoY_Seeger):
    """
    Class for a Poisson pseudodata node.
    Likelihood:
        p(y|x) \prop gamma(x) * e^{-gamma(x)}  (1)
    where gamma(x) is a rate function that is chosen to be convex and log-concave
    A simple choise for the rate function is e^{x} but this rate function is non-robust
    in the presence of outliers, so in Seeger et al they chose the function:
        gamma(x) = log(1+e^x)
    The data follows a Poisson distribution, but Followung Seeger et al the pseudodata Yhat_ij
    follows a normal distribution with mean E[W_{i,:}]*E[Z_{j,:}] and precision 'tau_j'
    where 'tau_j' is an upper bound of the second derivative of the loglikelihood:
        x_ij = sum_k^k w_{i,k}*z_{k,j}
        f_ij''(x_ij) <= tau_j for all i,j
    For the Poisson likelihood with rate function (1), the upper bound tau is calculated as follows:
        f_ij''(x_ij) = 0.25 + 0.17*ymax_j   where ymax_j = max(Y_{:,j})
    Pseudodata is updated as follows:
        yhat_ij = zeta_ij - f'(zeta_ij)/tau_j = ...
    The bound degrades with the presence of entries with large y_ij, so one should consider
    clipping overly large counts
    """
    def __init__(self, dim, obs, groups, params=None, E=None):
        # - dim (2d tuple): dimensionality of each view
        # - obs (ndarray): observed data
        # - E (ndarray): initial expected value of pseudodata
        PseudoY_Seeger.__init__(self, dim=dim, obs=obs, groups=groups, params=params, E=E)

        # Initialise the observed data
        assert np.all(s.mod(self.obs, 1) == 0), "Data must not contain float numbers, only integers"
        assert np.all(self.obs >= 0), "Data must not contain negative numbers"

    def precompute(self, options):
        self.updateParameters()
        self.updateExpectations()

    def ratefn(self, X):
        # Poisson rate function
        return np.log(1+np.exp(X)) + 0.0001

    def clip(self, threshold):
        # The local bound degrades with the presence of large values in the observed data, which should be clipped
        pass

    def updateExpectations(self):
        # Update the pseudodata
        tau = self.markov_blanket["Tau"].getValue()
        self.E = self.params["zeta"] - sigmoid(self.params["zeta"])*(1-self.obs/self.ratefn(self.params["zeta"])) / tau
        self.E[self.mask] = 0.

        # regress out feature-wise mean from the pseudodata
        self.means = self.E.mean(axis=0).data
        self.E -= self.means

    def calculateELBO(self):
        """ Compute Evidence Lower Bound """

        Wtmp = self.markov_blanket["W"].getExpectations()
        Ztmp = self.markov_blanket["Z"].getExpectations()
        W, WW = Wtmp["E"], Wtmp["E2"]
        Z, ZZ = Ztmp["E"], Ztmp["E2"]
        zeta = self.params["zeta"]
        tau = self.markov_blanket["Tau"].getValue()
        mask = self.getMask()

        # Precompute terms
        ZW = Z.dot(W.T)
        ZZWW = np.square(ZW) - np.dot(np.square(Z),np.square(W).T) + ZZ.dot(WW.T)

        # term1 = 0.5*tau*(ZW - zeta)**2
        term1 = 0.5*tau*(ZZWW - 2*ZW*zeta + np.square(zeta))
        term2 = (ZW - zeta)*(sigmoid(zeta)*(1.-self.obs/self.ratefn(zeta)))
        term3 = self.ratefn(zeta) - self.obs*np.log(self.ratefn(zeta))

        elbo = -(term1 + term2 + term3)
        elbo[mask] = 0.

        # I AM NOT SURE WHY NAs are generated...
        np.isnan(elbo).sum()
        elbo[np.isnan(elbo)] = 0.
        
        return elbo.sum()

class Bernoulli_PseudoY(PseudoY_Seeger):
    """
    Class for a Bernoulli (0,1 data) pseudodata node
    Likelihood:
        p(y|x) = (e^{yx}) / (1+e^x)  (1)
        f(x) = -log p(y|x) = log(1+e^x) - yx
    The second derivative is upper bounded by tau=0.25
    Folloiwng Seeger et al, the data follows a Bernoulli distribution but the pseudodata follows a
    normal distribution with mean E[W]*E[Z] and precision 'tau'
    IMPROVE EXPLANATION
    Pseudodata is updated as follows:
        yhat_ij = zeta_ij - f'(zeta_ij)/tau
                = zeta_ij - 4*(sigmoid(zeta_ij) - y_ij)
    """
    def __init__(self, dim, obs, groups, params=None, E=None):
        # - dim (2d tuple): dimensionality of each view
        # - obs (ndarray): observed data
        # - E (ndarray): initial expected value of pseudodata
        PseudoY_Seeger.__init__(self, dim=dim, obs=obs, groups=groups, params=params, E=E)

        # Initialise the observed data
        assert np.all( (self.obs==0) | (self.obs==1) ), "Data must be binary"

    def updateExpectations(self):
        # Update the pseudodata
        self.E = self.params["zeta"] - 4.*(sigmoid(self.params["zeta"]) - self.obs)

        # regress out feature-wise mean from the pseudodata
        self.means = self.E.mean(axis=0).data
        self.E -= self.means

    def calculateELBO(self):
        # Compute Lower Bound using the Bernoulli likelihood with observed data
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()
        mask = self.getMask()

        tmp = gpu_utils.asnumpy( gpu_utils.dot( gpu_utils.array(Z),gpu_utils.array(W).T ) )

        lb = self.obs*tmp - np.log(1.+np.exp(tmp))
        lb[mask] = 0.

        return lb.sum()


####################
## Jaakkola nodes ##
####################

class Tau_Jaakkola(Node):
    """
    Local Parameter that needs to be optimised in the Jaakkola approach.
    For more details see Supplementary Methods
    """
    def __init__(self, dim, value):
        Node.__init__(self, dim=dim)

        if isinstance(value,(int,float)):
            self.value = value * np.ones(dim)
        else:
            assert value.shape == dim, "Dimensionality mismatch"
            self.value = value

    def define_mini_batch(self, ix):
        """ Method to define a mini-batch (only for stochastic inference) """
        # self.mini_batch = self.E[ix,:]
        pass

    # TODO change: define a proper mini-batch !!
    def get_mini_batch(self, expand=True):
        return self.getExpectation(expand)
        
    def updateExpectations(self):
        self.value = 2*lambdafn(self.markov_blanket["Y"].getParameters()["zeta"])

    def getValue(self):
        return self.value

    def getExpectation(self, expand=True):
        return self.getValue()

    def getExpectations(self, expand=True):
        return { 'E':self.getValue(), 'lnE':np.log(self.getValue()) }

    def removeFactors(self, idx, axis=None):
        pass
class Bernoulli_PseudoY_Jaakkola(PseudoY):
    """
    Class for a Bernoulli pseudodata node using the Jaakkola approach:
    Likelihood:
        p(y|x) = (e^{yx}) / (1+e^x)
    Following Jaakola et al and intterpreting the bound as a liklihood on gaussian pseudodata
    leads to the folllowing updates
    Pseudodata is given by
            yhat_ij = (2*y_ij-1)/(4*lambadfn(xi_ij))
        where lambdafn(x)= tanh(x/2)/(4*x).
    Its conditional distribution is given by
            N((ZW)_ij, 1/(2 lambadfn(xi_ij)))
    Updates for the variational parameter xi_ij are given by
            sqrt(E((ZW)_ij^2))
    xi_ij in above notation is the same as zeta (variational parameter)
    NOTE: For this class to work the noise variance tau needs to be updated according to
        tau_ij <- 2*lambadfn(xi_ij)
    """
    def __init__(self, dim, obs, groups, params=None, E=None):
        PseudoY.__init__(self, dim=dim, obs=obs, groups=groups, params=params, E=E)

        # Initialise the observed data
        assert np.all( (self.obs==0) | (self.obs==1) ), "Data must be binary"

    def precompute(self, options):
        self.updateParameters(ix=None, ro=None)
        self.updateExpectations()

    def updateExpectations(self):
        self.E = (2.*self.obs - 1.)/(4.*lambdafn(self.params["zeta"]))

        # regress out feature-wise mean from the pseudodata
        self.means = self.E.mean(axis=0).data
        self.E -= self.means

    def updateParameters(self, ix=None, ro=None):
        Z = self.markov_blanket["Z"].getExpectations()
        W = self.markov_blanket["W"].getExpectations()
        self.params["zeta"] = np.sqrt(np.square(Z["E"].dot(W["E"].T)) - np.dot(np.square(Z["E"]), np.square(W["E"].T)) + np.dot(Z["E2"],W["E2"].T))

    def calculateELBO(self):
        # Compute Evidence Lower Bound using the lower bound to the likelihood
        Z = self.markov_blanket["Z"].getExpectation()
        Wtmp = self.markov_blanket["W"].getExpectations()
        Ztmp = self.markov_blanket["Z"].getExpectations()
        zeta = self.params["zeta"]
        SW, SWW = Wtmp["E"], Wtmp["E2"]
        Z, ZZ = Ztmp["E"], Ztmp["E2"]
        mask = self.getMask()

        # calculate E(Z)E(W)
        ZW = Z.dot(SW.T)
        ZW[mask] = 0.

        # Calculate E[(ZW_nd)^2]
        # this is equal to E[\sum_{k != k} z_k w_k z_k' w_k'] + E[\sum_{k} z_k^2 w_k^2]
        tmp1 = np.square(ZW) - np.dot(np.square(Z),np.square(SW).T) # this is for terms in k != k'
        tmp2 = ZZ.dot(SWW.T) # this is for terms in k = k'
        EZZWW = tmp1 + tmp2

        # calculate elbo terms
        term1 = 0.5 * ((2.*self.obs - 1.)*ZW - zeta)
        term2 = - np.log(1 + np.exp(-zeta))
        term3 = - 1/(4 * zeta) *  np.tanh(zeta/2.) * (EZZWW - zeta**2)

        lb = term1 + term2 + term3
        lb[mask] = 0.

        return lb.sum()

#-------------------------------------------------------------------------------
# Zero inflated data: mixed node implementation
#-------------------------------------------------------------------------------
# TODO in the data processing make sure that the data is centered around the zeros
# TODO create initialiser
# TODO build Markov Blanket in each sub-node. for the tau we need to give the right Y
class Zero_Inflated_PseudoY_Jaakkola(Unobserved_Variational_Mixed_Node):
    """
    Mixed node containing:
        - a normal Y node for non-zero data
        - a Bernoulli node for zero data
    Zeros are replaced by pseudo data as in Jaakola
    Non-Zero data remain as such
    getExpectations returns the merged matrices
    Appropriate wiring is done in the markov blanket so that tau jaakola sees the
    jaakola Y and normal tau sees the normal Y
    """
    def __init__(self, dim, obs, params=None, E=None):
        self.dim = dim
        
        self.all_obs = obs
        if type(self.all_obs) != ma.MaskedArray:
            self.all_obs = ma.masked_invalid(self.all_obs)

        # identify the zeros, nonzeros and nas and store their positions in masks
        self.zeros = (self.all_obs == 0)
        self.nonzeros = ~self.zeros # this masks includes the nas
        self.nas = ma.getmask(self.all_obs)
        self.mask = self.nas   # TODO mask to be used when calculating variance explained 

        self.sparsity = self.zeros.sum()/(self.zeros.sum() + self.nonzeros.sum())
        print('using zero inflated noise model with sparsity ', self.sparsity)

        # initialise the jaakola node with nas and non zeros masked
        obs_jaakola = obs.copy()  # TODO if obs is already a masked array should we update mask ?
        obs_jaakola[self.nonzeros] = np.nan
        # instead:?
        # self.notnas = np.logical_not(self.nas)
        # obs_jaakola[self.nonzeros & self.notnas] = 1 # TOCHECK: for non-zero values put 1 for observed
        self.jaakola_node = Bernoulli_PseudoY_Jaakkola(dim, obs_jaakola)

        # Initialise a y node where the zeros and nas are masked
        obs_normal = obs.copy()
        obs_normal[self.zeros]= np.nan  # nas are already masked so no need to
        self.normal_node = Y_Node(dim, obs_normal)

        self.nodes = [self.normal_node, self.jaakola_node]


    def addMarkovBlanket(self, **kwargs):
        self.jaakola_node.addMarkovBlanket(**kwargs)

        # NOTE here we make sure that the non-zero Y node sees the corresponding Tau
        if not hasattr(self.normal_node, 'markov_blanket'):
            self.normal_node.markov_blanket = {}

        for k,v in kwargs.items():
            if k in self.normal_node.markov_blanket.keys():
                print("Error: " + str(k) + " is already in the markov blanket of " + str(self.normal_node))
                exit(1)
            elif k == 'Tau':
                self.normal_node.markov_blanket[k] = v.tau_normal
            else:
                self.normal_node.markov_blanket[k] = v

    def getMask(self):
        return self.nas

    def updateParameters(self, ix=None, ro=None):
        self.jaakola_node.updateParameters(ix, ro)

    def updateExpectations(self):
        self.jaakola_node.updateExpectations()

    def getExpectation(self, expand=True):
        E = self.normal_node.getExpectation().copy()
        pseudo_y = self.jaakola_node.getExpectation() # TODO Is this in any ways comparable to E?
        E[self.zeros] = pseudo_y[self.zeros] 
        return E

    def get_mini_batch(self, expand=True):
        return self.getExpectation(expand=True)

    def calculateELBO(self):
        # as the values used by the jaakola node and the gamma node are rightly masked
        # we can just sum the two contributions (double check that this works ofc)
        # TODO Masking to 1 would change this!
        # TODO check masks
        tmp1 = self.jaakola_node.calculateELBO()
        tmp2 = self.normal_node.calculateELBO()

        return tmp1 + tmp2


class Zero_Inflated_Tau_Jaakkola(Unobserved_Variational_Mixed_Node):
    """
    Mixed node containing:
        - a jaakola tau
        - a normal tau
    Both nodes are initialised normally and the right wiring is done the markov blanket
    """

    def __init__(self, dim, value, pa, pb, qa, qb, groups, qE=None):
        # TODO what is the value in tau jaakola
        # initialiser for the two nodes initialise different members which are
        # all contained in the Zero_Inflated_Tau_Jaakkola node
        N = len(groups)
        self.tau_jaakola = Tau_Jaakkola((N, dim[1]), value)
        self.tau_normal  = TauD_Node(dim, pa, pb, qa, qb, groups, qE)

        self.nodes = [self.tau_jaakola, self.tau_normal]

    def addMarkovBlanket(self, **kwargs):
        # create a marov blanket for tau containing the zero inflated Y
        if not hasattr(self, 'markov_blanket'):
            self.markov_blanket = {}

        # create the markov blanket for the jaakola tau, wiring the corresonding Y
        if not hasattr(self.tau_jaakola, 'markov_blanket'):
            self.tau_jaakola.markov_blanket = {}

        for k,v in kwargs.items():
            if k in self.tau_jaakola.markov_blanket.keys():
                print("Error: " + str(k) + " is already in the markov blanket of " + str(self.tau_jaakola))
                exit(1)
            elif k == 'Y':
                self.tau_jaakola.markov_blanket[k] = v.jaakola_node
                self.markov_blanket[k] = v # put the 'mixed' Y node in the mb of self
            else:
                self.tau_jaakola.markov_blanket[k] = v

        # create the markov blanket for the normal tau, wiring the corressponding Y
        if not hasattr(self.tau_normal, 'markov_blanket'):
            self.tau_normal.markov_blanket = {}

        for k,v in kwargs.items():
            if k in self.tau_normal.markov_blanket.keys():
                print("Error: " + str(k) + " is already in the markov blanket of " + str(self.tau_normal))
                exit(1)
            elif k == 'Y':
                self.tau_normal.markov_blanket[k] = v.normal_node
            else:
                self.tau_normal.markov_blanket[k] = v

    def getExpectations(self, expand=True):
        # Get expectations from separate nodes
        E = self.tau_normal.getExpectations(expand=True)
        tau_jk = self.tau_jaakola.getExpectations(expand=True)

        # Merge
        zeros = self.markov_blanket['Y'].zeros
        E['E'][zeros], E['lnE'][zeros] = tau_jk['E'][zeros], tau_jk['lnE'][zeros]

        return E

    def getExpectation(self, expand=True):
        return self.getExpectations(expand)['E']

    def get_mini_batch(self, expand=True):
        return self.getExpectation(expand=True)

    def calculateELBO(self):
        # TODO make sure that tau_d masks the zeros
        return self.tau_normal.calculateELBO()
