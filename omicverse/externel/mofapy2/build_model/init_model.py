"""
Module to initialise a bioFAM model
"""

import numpy as np
import scipy as s
import scipy.stats as stats
from sys import path
import sklearn.decomposition
from sklearn.impute import SimpleImputer

from ..core.nodes import *

class initModel(object):
    def __init__(self, dim, data, lik, groups, seed):
        """
        PARAMETERS
        dim: dictionary with keyworded dimensionalities:
            N for the number of samples
            M for the number of views
            K for the number of factors or latent variables,
            D for the number of features per view
        data: list of length M with numpy arrays of dimensionality (N,Dm)
        lik: list of strings with length M
            likelihood for each view, choose from ('gaussian','poisson','bernoulli')
        """

        # Set seed
        np.random.seed(seed)

        # Set groups
        assert len(groups) == dim["N"], 'sample groups labels do not match number of samples'
        self.groups = groups
        self.groups_ix = np.unique(groups, return_inverse=True)[1] # convert groups into integers from 0 to n_groups


        # Set data
        self.data = data

        # Set likelihoods
        self.lik = lik

        # Set dimensionalities
        self.N = dim["N"]
        self.K = dim["K"]
        self.M = dim["M"]
        self.D = dim["D"]
        self.G = dim["G"]
        # self.G = len(np.unique(groups))

        self.nodes = {}

    def initZ(self, pmean=0., pvar=1., qmean="random", qvar=1., qE=None, qE2=None, Y=None, impute=False, weight_views = False):
        """Method to initialise the latent variables

        PARAMETERS
        ----------
        pmean: mean of the prior distribution
        pvar: (co)variance of the prior distribution
        qmean: initialisation of the mean of the variational distribution
            "random" for a random initialisation sampled from a standard normal distribution
            "pca" for an initialisation based on PCA
            "orthogonal" for a random initialisation with orthogonal factors
        qvar: initial value of the (co)variance of the variational distribution
        qE: initial value of the expectation of the variational distribution
        qE2: initial value of the second moment of the variational distribution
        Y: matrix to run PCA on (when qmean="pca")
        impute: logical value if to perform imputation before running PCA,
            this is only applicable when qmean="pca" and missing values (np.NaN) are present in the data
        weight_views: logical whether to weight the ELBO
        """

        ## Initialise prior distribution (P)

        # mean
        pmean = np.ones((self.N, self.K)) * pmean

        ## Initialise variational distribution (Q)

        # variance
        qvar = np.ones((self.N, self.K)) * qvar

        # mean
        if qmean is not None:
            if isinstance(qmean, str):

                # Random initialisation
                if qmean == "random":
                    qmean = stats.norm.rvs(loc=0, scale=1, size=(self.N, self.K))

                # Random initialisation with orthogonal factors
                elif qmean == "orthogonal":
                    pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
                    pca.fit(stats.norm.rvs(loc=0, scale=1, size=(self.N, 9999)).T)
                    qmean = pca.components_.T

                # PCA initialisation
                elif qmean == "pca":
                    pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
                    Ytmp = np.concatenate(Y, axis=1)

                    if impute == True:
                        if np.any(np.isnan(Ytmp)):
                            imp = SimpleImputer(missing_values=np.NaN, strategy="mean")
                            imp.fit(Ytmp)
                            Ytmp = imp.transform(Ytmp)

                    pca.fit(Ytmp)
                    qmean = pca.transform(Ytmp)

                # scale factor values from -1 to 1 (per factor)
                qmean = 2.*(qmean - np.min(qmean,axis=0))/np.ptp(qmean,axis=0)-1

            elif isinstance(qmean, np.ndarray):
                assert qmean.shape == (self.N, self.K), "Wrong shape for the expectation of the Q distribution of Z"

            elif isinstance(qmean, (int, float)):
                qmean = np.ones((self.N, self.K)) * qmean

            else:
                print("Wrong initialisation for Z")
                exit()

        # Initialise the node
        self.nodes["Z"] = Z_Node(
            dim=(self.N, self.K),
            pmean=pmean, pvar=pvar,
            qmean=qmean, qvar=qvar,
            qE=qE, qE2=qE2, weight_views = weight_views
        )

    def initZ_smooth(self, pmean=0., pvar=1., qmean="random", qvar=1., qE=None, qE2=None, Y=None, impute=False, weight_views = False):
        """Method to initialise the latent variables

        PARAMETERS
        ----------
        pmean: mean of the prior distribution
        pvar: (co)variance of the prior distribution
        qmean: initialisation of the mean of the variational distribution
            "random" for a random initialisation sampled from a standard normal distribution
            "pca" for an initialisation based on PCA
            "orthogonal" for a random initialisation with orthogonal factors
        qvar: initial value of the (co)variance of the variational distribution
        qE: initial value of the expectation of the variational distribution
        qE2: initial value of the second moment of the variational distribution
        Y: matrix to run PCA on (when qmean="pca")
        impute: logical value if to perform imputation before running PCA,
            this is only applicable when qmean="pca" and missing values (np.NaN) are present in the data
        weight_views: logical whether to weight the ELBO
        """

        ## Initialise prior distribution (P)

        # mean
        pmean = np.ones((self.N, self.K)) * pmean
        if isinstance(pvar, (int,float)):
            pvar = np.array([np.eye(self.N)*pvar for k in range(self.K)])

        ## Initialise variational distribution (Q)

        # variance
        qvar = np.array([np.eye(self.N)*qvar for k in range(self.K)])

        # mean
        if qmean is not None:
            if isinstance(qmean, str):

                # Random initialisation
                if qmean == "random":
                    qmean = stats.norm.rvs(loc=0, scale=1, size=(self.N, self.K))

                # Random initialisation with orthogonal factors
                elif qmean == "orthogonal":
                    pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
                    pca.fit(stats.norm.rvs(loc=0, scale=1, size=(self.N, 9999)).T)
                    qmean = pca.components_.T

                # PCA initialisation
                elif qmean == "pca":
                    pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
                    Ytmp = np.concatenate(Y, axis=1)

                    if impute == True:
                        if np.any(np.isnan(Ytmp)):
                            imp = SimpleImputer(missing_values=np.NaN, strategy="mean")
                            imp.fit(Ytmp)
                            Ytmp = imp.transform(Ytmp)

                    pca.fit(Ytmp)
                    qmean = pca.transform(Ytmp)

                # scale factor values from -1 to 1 (per factor)
                qmean = 2.*(qmean - np.min(qmean,axis=0))/np.ptp(qmean,axis=0)-1

            elif isinstance(qmean, np.ndarray):
                assert qmean.shape == (self.N, self.K), "Wrong shape for the expectation of the Q distribution of Z"

            elif isinstance(qmean, (int, float)):
                qmean = np.ones((self.N, self.K)) * qmean

            else:
                print("Wrong initialisation for Z")
                exit()

        # Initialise the node
        self.nodes["Z"] = Z_GP_Node_mv(
            dim=(self.N, self.K),
            pmean=pmean, pcov=pvar,
            qmean=qmean, qcov=qvar,
            qE=qE, 
            weight_views = weight_views
        )
        # self.nodes["Z"] = Z_GP_Node(
        #     dim=(self.N, self.K),
        #     pmean=pmean, pcov=pvar,
        #     qmean=qmean, qvar=qvar,
        #     qE=qE, 
        #     weight_views = weight_views
        # )

    def initU(self, pmean=0., pvar=1., qmean=0, qvar=1., qE=None, qE2=None, Y = None, impute=True, idx_inducing = None, weight_views = False): # prior hanp.diagonal covariance here, ls optimiation in Sigma node
        """Method to initialise the inducing points

         PARAMETERS
         ----------
        pmean: mean of the prior distribution
        pvar: (co)variance of the prior distribution
         qmean: initialisation of the mean of the variational distribution
             "random" for a random initialisation sampled from a standard normal distribution
             "pca" for an initialisation based on PCA
             "orthogonal" for a random initialisation with orthogonal factors
         qvar: initial value of the variance of the variational distribution
         qE: initial value of the expectation of the variational distribution
         qE2: initial value of the second moment of the variational distribution
         Y: matrix to run PCA on (when qmean="pca")
         impute: logical value if to perform imputation before running PCA,
             this is only applicable when qmean="pca" and missing values (np.NaN) are present in the data
         GP_factors: logical whether to use a Z node with GP prior or not
         """

        assert idx_inducing is not  None, "Stop: U nodes is used without inducing points"

        Nu = len(idx_inducing)

        ## Initialise prior distribution (P)
        # mean
        pmean = np.ones((Nu, self.K)) * pmean
        if isinstance(pvar, (int, float)):
            pvar = np.array([np.eye(Nu) * pvar for k in range(self.K)])

        ## Initialise variational distribution (Q)
        qmean = np.ones((Nu, self.K)) * qmean

        # variance
        qvar = np.array([np.eye(Nu) * qvar for k in range(self.K)])

        # Initialise the node
        self.nodes["U"] = U_GP_Node_mv(
            dim=(Nu, self.K),
            pmean=pmean, pcov=pvar,
            qmean=qmean, qcov=qvar,
            qE=qE, idx_inducing = idx_inducing,
            weight_views = weight_views
        )

    def initZgU(self, pmean = 0, pvar =1., qmean = "random", qvar = 1., qE = None, qE2 = None, Y = None,
                impute = True, idx_inducing = None, weight_views = False):
        assert idx_inducing is not None, "Stop: ZgU nodes is used without inducing points"

        ## Initialise prior distribution (P)

        # mean
        pmean = np.ones((self.N, self.K)) * pmean

        ## Initialise variational distribution (Q)

        # variance
        qvar = np.ones((self.N, self.K)) * qvar

        # mean
        if qmean is not None:
            if isinstance(qmean, str):

                # Random initialisation
                if qmean == "random":
                    qmean = stats.norm.rvs(loc=0, scale=1, size=(self.N, self.K))

                # Random initialisation with orthogonal factors
                elif qmean == "orthogonal":
                    pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
                    pca.fit(stats.norm.rvs(loc=0, scale=1, size=(self.N, 9999)).T)
                    qmean = pca.components_.T

                # PCA initialisation
                elif qmean == "pca":
                    # whiten=True scales the principal components to match the prior N(0,1)
                    pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
                    Ytmp = np.concatenate(Y, axis=1)

                    if impute == True:
                        if np.any(np.isnan(Ytmp)):
                            imp = SimpleImputer(missing_values=np.NaN, strategy="mean")
                            imp.fit(Ytmp)
                            Ytmp = imp.transform(Ytmp)

                    pca.fit(Ytmp)
                    qmean = pca.transform(Ytmp)

                # scale factor values from -1 to 1 (per factor)
                qmean = 2.*(qmean - np.min(qmean,axis=0))/np.ptp(qmean,axis=0)-1

            elif isinstance(qmean, np.ndarray):
                assert qmean.shape == (self.N, self.K), "Wrong shape for the expectation of the Q distribution of Z"

            elif isinstance(qmean, (int, float)):
                qmean = np.ones((self.N, self.K)) * qmean

            else:
                print("Wrong initialisation for Z")
                exit()

        self.nodes["Z"] = ZgU_node(
            dim=(self.N, self.K),
            pmean=pmean, pvar=pvar,
            qmean=qmean, qvar=qvar,
            qE=qE, qE2=qE2,
            idx_inducing = idx_inducing,
            weight_views = weight_views
        )

    def initSigma(self, sample_cov, start_opt = 20, n_grid = 10,
        opt_freq = 10, model_groups = False, use_gpytorch = False):

        self.Sigma = Sigma_Node(
            dim=(self.K,), 
            sample_cov=sample_cov, 
            groups=self.groups,
            start_opt=start_opt, 
            n_grid=n_grid, 
            opt_freq=opt_freq, 
            model_groups = model_groups
        )
        self.nodes["Sigma"] = self.Sigma

    def initSigma_sparse(self, sample_cov, start_opt = 20, n_grid = 10, idx_inducing = None, 
        opt_freq = 10, model_groups = False, use_gpytorch = False):

        self.Sigma = Sigma_Node_sparse(
            dim=(self.K,), 
            sample_cov=sample_cov, 
            groups=self.groups,
            start_opt=start_opt, 
            n_grid=n_grid, 
            idx_inducing=idx_inducing, 
            opt_freq=opt_freq, 
            model_groups = model_groups
        )
        self.nodes["Sigma"] = self.Sigma

    def initSigma_warping(self, sample_cov, start_opt = 20, n_grid = 10, 
        warping_freq = 20, warping_ref = 0, warping_open_begin = True, warping_open_end =True, warping_groups = None,
        opt_freq = 10, model_groups = False, use_gpytorch = False):

        self.Sigma = Sigma_Node_warping(
            dim=(self.K,), 
            sample_cov=sample_cov, 
            groups=self.groups,
            start_opt=start_opt, 
            n_grid=n_grid, 
            warping_freq=warping_freq, 
            warping_ref=warping_ref,
            warping_open_begin=warping_open_begin, 
            warping_open_end=warping_open_end,
            warping_groups=warping_groups,
            opt_freq=opt_freq, 
            model_groups = model_groups
        )
        self.nodes["Sigma"] = self.Sigma

    def initSZ(self, pmean_T0=0., pmean_T1=0., pvar_T0=1., pvar_T1=1., ptheta=1., qmean_T0=0., qmean_T1="random", qvar_T0=1.,
        qvar_T1=1., qtheta=1., qEZ_T0=None, qEZ_T1=None, qET=None, Y=None, impute=False, weight_views = False):
        """Method to initialise sparse factors with a spike and slab prior

        PARAMETERS
        ----------
        (...)
        """

        # TODO:
        # - add different ways to initialise the expectations of the posterior: random, orthogonal, PCA


        ## Initialise prior distribution (P)

        ## Initialise variational distribution (Q)
        if isinstance(qmean_T1, str):

            if qmean_T1 == "random":
                qmean_T1 = stats.norm.rvs(loc=0, scale=1, size=(self.N, self.K))
            elif qmean_T1 == "pca":
                pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
                Ytmp = np.concatenate(Y, axis=1)

                if impute == True:
                    if np.any(np.isnan(Ytmp)):
                        imp = SimpleImputer(missing_values=np.NaN, strategy="mean")
                        imp.fit(Ytmp)
                        Ytmp = imp.transform(Ytmp)

                pca.fit(Ytmp)
                qmean_T1 = pca.transform(Ytmp)
            else:
                print("%s initialisation not implemented for Z" % qmean_T1)
                exit()

        elif isinstance(qmean_T1, np.ndarray):
            assert qmean_T1.shape == (self.N, self.K), "Wrong dimensionality"

        elif isinstance(qmean_T1, (int, float)):
            qmean_T1 *= np.ones((self.N, self.K))

        else:
            print("Wrong initialisation for Z")
            exit(1)

        self.nodes["Z"] = SZ_Node(
            dim=(self.N, self.K),

            ptheta=ptheta,
            pmean_T0=pmean_T0, pvar_T0=pvar_T0,
            pmean_T1=pmean_T1, pvar_T1=pvar_T1,

            qtheta=qtheta,
            qmean_T0=qmean_T0, qvar_T0=qvar_T0,
            qmean_T1=qmean_T1, qvar_T1=qvar_T1,

            qET=qET, qEZ_T0=qEZ_T0, qEZ_T1=qEZ_T1,
            weight_views = weight_views
        )

    def initW(self, pmean=0., pvar=1., qmean='random', qvar=1., qE=None, qE2=None, Y=None):
        """Method to initialise the weights

        PARAMETERS
        ----------
        pmean: mean of the prior distribution
        pvar: variance of the prior distribution
        qmean: initial value of the mean of the variational distribution
        qvar: initial value of the variance of the variational distribution
        qE: initial value of the expectation of the variational distribution
        qE2: initial value of the second moment of the variational distribution
        """

        W_list = [None] * self.M

        for m in range(self.M):

            ## Initialise prior distribution (P) ##

            # mean
            pmean_m = np.ones((self.D[m], self.K)) * pmean

            ## Initialise variational distribution (Q) ##

            # variance
            qvar_m = np.ones((self.D[m], self.K)) * qvar

            # mean
            if qmean is not None:
                if isinstance(qmean, str):

                    # Random initialisation
                    if qmean == "random":
                        qmean_m = stats.norm.rvs(loc=0, scale=1., size=(self.D[m], self.K))

                    elif qmean_S1 == "pca":
                        # print("Initialising weights with PCA solution")
                        pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
                        pca.fit(Y[m])
                        qmean_S1_tmp = pca.components_.T

                elif isinstance(qmean, np.ndarray):
                    assert qmean.shape == (
                    self.D[m], self.K), "Wrong shape for the expectation of the Q distribution of W"
                    qmean_m = qmean

                elif isinstance(qmean, (int, float)):
                    qmean_m = np.ones((self.D[m], self.K)) * qmean

                else:
                    print("Wrong initialisation for W")
                    exit()

            else:
                qmean_m = None

            # Initialise the node
            W_list[m] = W_Node(
                dim=(self.D[m], self.K),
                pmean=pmean_m, pvar=pvar,
                qmean=qmean_m, qvar=qvar_m,
                qE=qE, qE2=qE2
            )

        self.nodes["W"] = Multiview_Variational_Node(self.M, *W_list)

    def initSW(self, pmean_S0=0., pmean_S1=0., pvar_S0=1., pvar_S1=1., ptheta=1.,
        qmean_S0=0., qmean_S1='random', qvar_S0=1., qvar_S1=1., qtheta=1.,
        qEW_S0=None, qEW_S1=None, qES=None, Y=None):
        """Method to initialise sparse weights with a (reparametrised) spike and slab prior

        PARAMETERS
        ----------
        (...)
        """

        W_list = [None]*self.M

        for m in range(self.M):

            ## Initialise prior distribution (P)

            ## Initialise variational distribution (Q)
            if isinstance(qmean_S1,str):

                if qmean_S1 == "random":
                    qmean_S1_tmp = stats.norm.rvs(loc=0, scale=1., size=(self.D[m],self.K))
                elif qmean_S1 == "pca":
                    # if np.any(np.isnan(Y[m])):
                    #     print("Initialising weights with PCA solution, but data has missing values. Doing quick feature-wise mean imputation (just for the initialisation)... ")
                    #     # TO-DO: BE CAREFUL WITH NON-GAUSSIAN DATA...
                    #     from sklearn.impute import SimpleImputer
                    #     imp = SimpleImputer(missing_values=np.nan, strategy='mean') # using the mean along each column
                    #     imp.fit(Y[m])
                    #     imp.transform(Y[m])

                    pca = sklearn.decomposition.PCA(n_components=self.K, whiten=True)
                    pca.fit(Y[m])
                    qmean_S1_tmp = pca.components_.T
                    # qmean_S1_tmp /= np.nanstd(qmean_S1_tmp, axis=0) # Scale weights to unit variance
                else:
                    print("%s initialisation not implemented for W" % qmean_S1)
                    exit()

                # Scale weights to the variance of the view
                # if Y is not None:
                #     qmean_S1_tmp *= np.nanstd(Y[m]) 

            elif isinstance(qmean_S1,np.ndarray):
                assert qmean_S1.shape == (self.D[m],self.K), "Wrong dimensionality"

            elif isinstance(qmean_S1,(int,float)):
                qmean_S1_tmp = np.ones((self.D[m],self.K)) * qmean_S1

            else:
                print("Wrong initialisation for W")
                exit(1)

            W_list[m] = SW_Node(
                dim=(self.D[m],self.K),

                ptheta=ptheta,
                pmean_S0=pmean_S0, pvar_S0=pvar_S0,
                pmean_S1=pmean_S1, pvar_S1=pvar_S1,

                qtheta=qtheta,
                qmean_S0=qmean_S0, qvar_S0=qvar_S0,
                qmean_S1=qmean_S1_tmp, qvar_S1=qvar_S1,

                qES=qES, qEW_S0=qEW_S0, qEW_S1=qEW_S1
            )

        self.nodes["W"] = Multiview_Variational_Node(self.M, *W_list)

    def initAlphaZ(self, pa=1e-14, pb=1e-14, qa=1., qb=1., qE=None, qlnE=None):
        """Method to initialise the ARD prior on Z per sample group

        PARAMETERS
        ----------
        groups: dictionary
            a dictionariy with mapping between sample names (keys) and samples_groups (values)
        pa: float
            'a' parameter of the prior distribution
        pb :float
            'b' parameter of the prior distribution
        qb: float
            initialisation of the 'b' parameter of the variational distribution
        qE: float
            initial expectation of the variational distribution
        qlnE: float
            initial log expectation of the variational distribution
        """

        self.nodes["AlphaZ"] = AlphaZ_Node(
            dim=(self.G, self.K),
            pa=pa, pb=pb,
            qa=qa, qb=qb,
            groups=self.groups_ix,
            qE=qE, qlnE=qlnE
        )

    def initAlphaW(self, pa=1e-14, pb=1e-14, qa=1., qb=1., qE=None, qlnE=None):
        """Method to initialise the ARD prior on W

        PARAMETERS
        ----------
        pa: float
            'a' parameter of the prior distribution
        pb :float
            'b' parameter of the prior distribution
        qa: float
            initialisation of the 'b' parameter of the variational distribution
        qb: float
            initialisation of the 'b' parameter of the variational distribution
        qE: float
            initial expectation of the variational distribution
        qlnE: float
            initial log expectation of the variational distribution
        """

        alpha_list = [None]*self.M

        for m in range(self.M):
            alpha_list[m] = AlphaW_Node(dim=(self.K,), pa=pa, pb=pb, qa=qa, qb=qb, qE=qE, qlnE=qlnE)
        self.nodes["AlphaW"] = Multiview_Variational_Node(self.M, *alpha_list)

    def initTau(self, pa=1e-3, pb=1e-3, qa=1., qb=1., qE=None):
        """Method to initialise the precision of the noise

        PARAMETERS
        ----------
        pa: float
            'a' parameter of the prior distribution
        pb :float
            'b' parameter of the prior distribution
        qb: float
            initialisation of the 'b' parameter of the variational distribution
        qE: float
            initial expectation of the variational distribution
        """

        tau_list = [None]*self.M

        for m in range(self.M):

            # Poisson noise model for count data
            if self.lik[m] == "poisson":
                tmp = 0.25 + 0.17*np.nanmax(self.data[m],axis=0)
                tmp = np.repeat(tmp[None,:], self.N, axis=0)
                tau_list[m] = Tau_Seeger(dim=(self.N, self.D[m]), value=tmp)

            # Bernoulli noise model for binary data
            elif self.lik[m] == "bernoulli":
                # tau_list[m] = Constant_Node(dim=(self.D[m],), value=np.ones(self.D[m])*0.25)
                # tau_list[m] = Tau_Jaakkola(dim=(self.D[m],), value=0.25)
                tau_list[m] = Tau_Jaakkola(dim=((self.N, self.D[m])), value=1.)

            elif self.lik[m] == "zero_inflated":
                # contains parameters to initialise both normal and jaakola tau
                tau_list[m] = Zero_Inflated_Tau_Jaakkola(dim=((self.G, self.D[m])), value=1., pa=pa, pb=pb, qa=qa, qb=qb, groups=self.groups_ix, qE=qE)

            # Gaussian noise model for continuous data
            elif self.lik[m] == "gaussian":
                tau_list[m] = TauD_Node(dim=(self.G, self.D[m]), pa=pa, pb=pb, qa=qa, qb=qb, groups=self.groups_ix, qE=qE)

        self.nodes["Tau"] = Multiview_Mixed_Node(self.M, *tau_list)

    def initY(self):
        """Method to initialise the observations"""

        Y_list = [None]*self.M
        for m in range(self.M):
            if self.lik[m]=="gaussian":
                Y_list[m] = Y_Node(dim=(self.N,self.D[m]), value=self.data[m], groups=self.groups_ix)
            elif self.lik[m]=="poisson":
                Y_list[m] = Poisson_PseudoY(dim=(self.N,self.D[m]), obs=self.data[m], E=self.data[m], groups=self.groups_ix)
            elif self.lik[m]=="bernoulli":
                Y_list[m] =  Bernoulli_PseudoY_Jaakkola(dim=(self.N,self.D[m]), obs=self.data[m], E=self.data[m], groups=self.groups_ix)
            elif self.lik[m]=="zero_inflated":
                Y_list[m] = Zero_Inflated_PseudoY_Jaakkola(dim=(self.N,self.D[m]), obs=self.data[m], E=self.data[m])
        self.Y = Multiview_Mixed_Node(self.M, *Y_list)
        self.nodes["Y"] = self.Y

    def initThetaZ(self, pa=1., pb=1., qa=1., qb=1., qE=None):
        """Method to initialise the ARD prior on Z per sample group

        PARAMETERS
        ----------
        pa: float
            'a' parameter of the prior distribution
        pb :float
            'b' parameter of the prior distribution
        qb: float
            initialisation of the 'b' parameter of the variational distribution
        qE: float
            initial expectation of the variational distribution
        qlnE: float
            initial log expectation of the variational distribution
        K: number of factors for which we learn theta. If no argument is given, we'll just use the
            total number of factors
        """

        self.nodes["ThetaZ"] = ThetaZ_Node(
            dim=(self.G, self.K),
            pa=pa, pb=pb,
            qa=qa, qb=qb,
            groups=self.groups_ix,
            qE=qE)

    def initThetaW(self, pa=1., pb=1., qa=1., qb=1., qE=None):
        """Method to initialise the sparsity parameter of the spike and slab weights

        PARAMETERS
        ----------
         pa: float
            'a' parameter of the prior distribution
         pb :float
            'b' parameter of the prior distribution
         qb: float
            initialisation of the 'b' parameter of the variational distribution
         qE: float
            initial expectation of the variational distribution
        """
        Theta_list = [None] * self.M
        for m in range(self.M):
            Theta_list[m] = ThetaW_Node(dim=(self.K,), pa=pa, pb=pb, qa=qa, qb=qb, qE=qE)
        self.nodes["ThetaW"] = Multiview_Variational_Node(self.M, *Theta_list)

    def initExpectations(self, *nodes):
        """ Method to initialise all expectations """
        for node in nodes:
            self.nodes[node].updateExpectations()

    def getNodes(self):
        """ Get method to return the nodes """
        return self.nodes
        #return { k:v for (k,v) in self.nodes.items()}
