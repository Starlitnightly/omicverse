"""
Module to build a bioFAM model
"""

from sys import path
from time import time,sleep
import numpy as np
import scipy as s
#from joblib import Parallel, delayed

from .init_model import initModel
from .utils import *

class buildModel(object):
    def __init__(self, data, dimensionalities, data_opts, model_opts, train_opts):
        self.data = data
        self.dim = dimensionalities
        self.data_opts = data_opts
        self.model_opts = model_opts
        self.train_opts = train_opts

    def createMarkovBlankets(self):
        """ Define the markov blankets """
        pass

    def createSchedule(self):
        """ Define the schedule of updates """
        pass

    def build_nodes(self):
        """ Build all nodes """
        pass

    def get_nodes(self):
        """ Get all nodes """
        return self.init_model.getNodes()




class buildBiofam(buildModel):
    def  __init__(self, *args, **kwargs): 
        super(buildBiofam, self).__init__(*args, **kwargs)

    def main(self):
        # create an instance of initModel
        self.init_model = initModel(
            dim = self.dim, 
            data = self.data, 
            lik = self.model_opts["likelihoods"], 
            groups = self.data_opts['samples_groups'],
            seed = self.train_opts['seed']
        )

        # Build all nodes
        self.build_nodes()

        # Define markov blankets
        self.createMarkovBlankets()

    def build_nodes(self):
        """ Method to build all nodes """

        # Build general nodes
        self.build_Z()
        self.build_W()
        self.build_Tau()
        self.build_Y() # important to keep Y last. NAs get replaced by 0 and can mislead PCA initialisation

        # define ARD sparsity per sample group (on Z)
        if self.model_opts['ard_factors']:
            self.build_AlphaZ()

        # define ARD sparsity per feature_group, or view (on W)
        if self.model_opts['ard_weights']:
            self.build_AlphaW()

        # define feature-wise spike and slab sparsity in Z
        if self.model_opts['spikeslab_factors']:
            self.build_ThetaZ()

        # define feature-wise spike and slab sparsity in W
        if self.model_opts['spikeslab_weights']:
            self.build_ThetaW()

    def build_Y(self):
        """ Build node Y for the observations """
        self.init_model.initY()

    def build_Z(self):
        """ Build node Z for the factors or latent variables """
        if self.model_opts['spikeslab_factors']:
            # self.init_model.initSZ(qmean_T1=0)
            # self.init_model.initSZ(qmean_T1="random")
            self.init_model.initSZ(qmean_T1="pca", Y=self.data, impute=True, weight_views = self.train_opts['weight_views'])
        else:
            # self.init_model.initZ(qmean=0)
            # self.init_model.initZ(qmean="random")
            self.init_model.initZ(
                qmean="pca", 
                Y=self.data, 
                impute=True,
                weight_views = self.train_opts['weight_views']
            )

    def build_W(self):
        """ Build node W for the weights """
        if self.model_opts['spikeslab_weights']:
            self.init_model.initSW(qmean_S1=0)
            # self.init_model.initSW(qmean_S1="random", Y=self.data)
            # self.init_model.initSW(qmean_S1="pca", Y=self.data)
        else:
            self.init_model.initW(qmean=0)
            # self.init_model.initW(qmean="random", Y=self.data)
            # self.init_model.initW(qmean="pca", Y=self.data)

    def build_Tau(self):
        # TODO sort out how to choose where to use Tau
        # initTau_qE = 100.
        initTau_qE = None
            
        self.init_model.initTau(qE=initTau_qE)

    def build_AlphaZ(self):
        """ Build node AlphaZ for the ARD prior on the factors """

        # ARD prior per sample group
        self.init_model.initAlphaZ()

    def build_AlphaW(self):
        """ Build node AlphaW for the ARD prior on the weights"""

        # ARD prior per factor and feature_group (view)
        self.init_model.initAlphaW()

    def build_ThetaZ(self):
        # TODO use mixed theta node instead when fixed in update and init -> should be ok then for intercept
        """ Build node ThetaZ for the Spike and Slab prior on the factors """

        # Initialise hyperparameters for the ThetaZ prior
        initTheta_a = 1.
        initTheta_b = 1.
        initTheta_qE = None
        # initTheta_qE = 1.

        self.init_model.initThetaZ(self.data_opts['samples_groups'], qa=initTheta_a, qb=initTheta_b, qE=initTheta_qE)

    def build_ThetaW(self):
        """ Build node ThetaW for the Spike and Slab prior on the weights """

        # Initialise hyperparameters for the ThetaW prior
        initTheta_a = 1.
        initTheta_b = 1.
        initTheta_qE = None
        # initTheta_qE = 1.

        self.init_model.initThetaW(qa=initTheta_a, qb=initTheta_b, qE=initTheta_qE)

    def createMarkovBlankets(self):
        """ Define the markov blankets """

        # Fetch all nodes
        nodes = self.get_nodes()

        # Define basic connections
        nodes['Y'].addMarkovBlanket(Z=nodes["Z"], W=nodes["W"], Tau=nodes["Tau"])
        nodes['Z'].addMarkovBlanket(Y=nodes["Y"], W=nodes["W"], Tau=nodes["Tau"])
        nodes['W'].addMarkovBlanket(Y=nodes["Y"], Z=nodes["Z"], Tau=nodes["Tau"])
        nodes['Tau'].addMarkovBlanket(Y=nodes["Y"], W=nodes["W"], Z=nodes["Z"])

        # Add ThetaZ in the markov blanket of Z and viceversa if Spike and Slab prior on Z
        if self.model_opts['spikeslab_factors']:
            nodes['Z'].addMarkovBlanket(ThetaZ=nodes["ThetaZ"])
            nodes["ThetaZ"].addMarkovBlanket(Z=nodes["Z"])

        # Add ThetaW in the markov blanket of W and viceversa if Spike and Slab prior on W
        if self.model_opts['spikeslab_weights']:
            nodes['W'].addMarkovBlanket(ThetaW=nodes["ThetaW"])
            nodes["ThetaW"].addMarkovBlanket(W=nodes["W"])

        # Add AlphaZ in the markov blanket of Z and viceversa if ARD prior on Z
        if self.model_opts['ard_factors']:
                nodes['AlphaZ'].addMarkovBlanket(Z=nodes['Z'])
                nodes['Z'].addMarkovBlanket(AlphaZ=nodes['AlphaZ'])

        # Add AlphaW in the markov blanket of W and viceversa if ARD prior on W
        if self.model_opts['ard_weights']:
            nodes['AlphaW'].addMarkovBlanket(W=nodes['W'])
            nodes['W'].addMarkovBlanket(AlphaW=nodes['AlphaW'])


class build_mofa_smooth(buildBiofam):
    def __init__(self, data, sample_cov, dimensionalities, data_opts, model_opts, train_opts, smooth_opts):
        super(build_mofa_smooth, self).__init__(data, dimensionalities, data_opts, model_opts, train_opts)
        self.sample_cov = sample_cov
        self.smooth_opts = smooth_opts

    def main(self):
        # create an instance of initModel
        self.init_model = initModel(
            dim = self.dim, 
            data = self.data, 
            lik = self.model_opts["likelihoods"], 
            groups = self.data_opts['samples_groups'],
            seed = self.train_opts['seed']
        )

        # Build all nodes
        self.build_nodes()

        # Define markov blankets
        self.createMarkovBlankets()

    def build_Z(self):
        """ Build node Z for the factors """
        self.init_model.initZ_smooth(
            qmean="pca", 
            Y=self.data, 
            impute=True,
            weight_views = self.train_opts['weight_views']
        )

    def build_ZgU(self):
        """ Build node for Z given U for the factors or latent variables conditioned on inducing points"""

        self.init_model.initZgU(
            qmean="pca", 
            Y=self.data, 
            impute=True,
            idx_inducing = self.smooth_opts['idx_inducing'],
            weight_views = self.train_opts['weight_views']
        )

    def build_U(self):
        """ Build node U for the inducing points of latent variable GP """

        # initialise U by PCA (no use of GP prior)
        self.init_model.initU(
            idx_inducing = self.smooth_opts['idx_inducing'],
            weight_views = self.train_opts['weight_views']
        )

    def build_Sigma(self):
        """ Build node Sigma for the GP prior on the factors """

        # TO-DO: assert on build that G >1 if warping
        # TO-DO: exclude idx_inducing when using warping

        # Sparse GPs
        if self.smooth_opts['sparseGP'] is True:
            # Warping
            if self.smooth_opts['warping'] is True:
                raise NotImplementedError
            # Non-Warping
            else:
                self.init_model.initSigma_sparse(
                    self.sample_cov,
                    start_opt = self.smooth_opts['start_opt'],
                    n_grid = self.smooth_opts['n_grid'],
                    idx_inducing = self.smooth_opts['idx_inducing'],
                    opt_freq = self.smooth_opts['opt_freq'],
                    model_groups = self.smooth_opts['model_groups']
                )

        # Non-sparse GPs
        else:
            # Warping
            if self.smooth_opts['warping'] is True:
                self.init_model.initSigma_warping(
                    sample_cov = self.sample_cov,
                    start_opt = self.smooth_opts['start_opt'],
                    n_grid = self.smooth_opts['n_grid'],
                    warping_freq = self.smooth_opts['warping_freq'],
                    warping_ref = self.smooth_opts['warping_ref'],
                    warping_open_begin = self.smooth_opts['warping_open_begin'],
                    warping_open_end = self.smooth_opts['warping_open_end'],
                    warping_groups = self.smooth_opts['warping_groups'],
                    opt_freq = self.smooth_opts['opt_freq'],
                    model_groups = self.smooth_opts['model_groups']#,
                    # use_gpytorch  = self.model_opts['use_gpytorch']
                )
            # Non-warping
            else:
                self.init_model.initSigma(
                    self.sample_cov,
                    start_opt = self.smooth_opts['start_opt'],
                    n_grid = self.smooth_opts['n_grid'],
                    # idx_inducing = self.smooth_opts['idx_inducing'],
                    # warping = self.smooth_opts['warping'],
                    # warping_freq = self.smooth_opts['warping_freq'],
                    # warping_ref = self.smooth_opts['warping_ref'],
                    # warping_open_begin = self.smooth_opts['warping_open_begin'],
                    # warping_open_end = self.smooth_opts['warping_open_end'],
                    opt_freq = self.smooth_opts['opt_freq'],
                    model_groups = self.smooth_opts['model_groups']#,
                    # use_gpytorch  = self.model_opts['use_gpytorch']
                )

    def build_nodes(self):
        """ Method to build all nodes """

        # smooth factors nodes
        if self.smooth_opts["sparseGP"]:
            self.build_U()
            self.build_ZgU()
        else:
            self.build_Z()

        # Build general nodes
        self.build_W()
        self.build_Tau()
        self.build_Y() # important to keep Y last. NAs get replaced by 0 and can mislead PCA initialisation

        # define ARD sparsity per feature_group, or view (on W)
        if self.model_opts['ard_weights']:
            self.build_AlphaW()

        # define Gaussian process prior on Z
        self.build_Sigma()

        # define feature-wise spike and slab sparsity in W
        if self.model_opts['spikeslab_weights']:
            self.build_ThetaW()

    def createMarkovBlankets(self):
        """ Define the markov blankets """

        # Fetch all nodes
        nodes = self.get_nodes()

        # Define basic connections
        nodes['Y'].addMarkovBlanket(Z=nodes["Z"], W=nodes["W"], Tau=nodes["Tau"])
        nodes['Z'].addMarkovBlanket(Y=nodes["Y"], W=nodes["W"], Tau=nodes["Tau"])
        nodes['W'].addMarkovBlanket(Y=nodes["Y"], Z=nodes["Z"], Tau=nodes["Tau"])
        nodes['Tau'].addMarkovBlanket(Y=nodes["Y"], W=nodes["W"], Z=nodes["Z"])

        
        # Add Sigma in the markov blanket of Z and viceversa if GP prior on Z
        if self.smooth_opts['sparseGP']:
            nodes['Z'].addMarkovBlanket(U=nodes["U"], Sigma = nodes['Sigma'])
            nodes["Sigma"].addMarkovBlanket(U=nodes["U"])
            nodes['U'].addMarkovBlanket(Z=nodes["Z"], Sigma = nodes['Sigma'], Y=nodes["Y"], W=nodes["W"], Tau=nodes["Tau"])
        else:
            nodes['Z'].addMarkovBlanket(Sigma=nodes["Sigma"])
            nodes["Sigma"].addMarkovBlanket(Z=nodes["Z"])

        # Add ThetaW in the markov blanket of W and viceversa if Spike and Slab prior on W
        if self.model_opts['spikeslab_weights']:
            nodes['W'].addMarkovBlanket(ThetaW=nodes["ThetaW"])
            nodes["ThetaW"].addMarkovBlanket(W=nodes["W"])


        # Add AlphaW in the markov blanket of W and viceversa if ARD prior on W
        if self.model_opts['ard_weights']:
            nodes['AlphaW'].addMarkovBlanket(W=nodes['W'])
            nodes['W'].addMarkovBlanket(AlphaW=nodes['AlphaW'])
