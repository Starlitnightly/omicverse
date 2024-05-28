import numpy as np
import scipy as s
from .basic_distributions import Distribution
from .bernoulli import Bernoulli
from .univariate_gaussian import UnivariateGaussian

from ..utils import *

class BernoulliGaussian(Distribution):
    """
    Class to define a Bernoulli-Gaussian distributions (for more information see Titsias and Gredilla, 2014)

    The best way to characterise a joint Bernoulli-Gaussian distribution P(N,B) is by considering
    its factorisation p(N|B)p(B) where B is a bernoulli distribution and N|B=0 and N|B=1 are normal distributions

    Equations:
    p(N,B) = Normal(N|mean,var) * Bernoulli(B|theta)
    FINISH EQUATIONS

    ROOM FOR IMPROVEMENT: i think the current code is inefficient because you have to keep track of the params
    and expectations in both the factorised distributions and the joint one. I think
    perhaps is better not to define new Bernoulli and UnivariateGaussian but work directly with the joint model
    """
    def __init__(self, dim, mean_B0, mean_B1, var_B0, var_B1, theta, EN_B0=None, EN_B1=None, EB=None):
        Distribution.__init__(self,dim)
        self.B = Bernoulli(dim=dim, theta=theta, E=EB)
        self.N_B0 = UnivariateGaussian(dim=dim, mean=mean_B0, var=var_B0, E=EN_B0)
        self.N_B1 = UnivariateGaussian(dim=dim, mean=mean_B1, var=var_B1, E=EN_B1)

        # Collect parameters
        self.params = { 'mean_B0':self.N_B0.params['mean'],
                        'mean_B1':self.N_B1.params['mean'],
                        'var_B0':self.N_B0.params['var'],
                        'var_B1':self.N_B1.params['var'],
                        'theta':self.B.params['theta'] }

        # Collect expectations
        self.updateExpectations()

    def getParameters(self):
        # Get function for parameters
        return self.params

    def setParameters(self,**params):
        # Setter function for parameters
        self.B.setParameters(theta=params['theta'])
        self.N_B0.setParameters(mean=params['mean_B0'], var=params['var_B0'])
        self.N_B1.setParameters(mean=params['mean_B1'], var=params['var_B1'])
        self.params = params

    def updateParameters(self):
        # Method to update the parameters of the joint distribution based on its constituent distributions
        self.params = { 'theta':self.B.params["theta"],
                        'mean_B0':self.N_B0.params["mean"], 'var_B0':self.N_B0.params["var"],
                        'mean_B1':self.N_B1.params["mean"], 'var_B1':self.N_B1.params["var"] }

    def updateExpectations(self):
        # Method to calculate the expectations based on the current estimates for the parameters

        # Update expectations of the constituent distributions
        self.B.updateExpectations()
        self.N_B0.updateExpectations()
        self.N_B1.updateExpectations()

        # Calculate expectations of the joint distribution
        EB = self.B.getExpectation()
        EN = self.N_B1.getExpectation()
        E = EB * EN
        # TODO double check the order here 
        E2 = EB * (np.square(EN) + self.params["var_B1"])
        ENN = EB*(np.square(EN)+self.params["var_B1"]) + (1-EB)*self.params["var_B0"]

        # Compute the expectation of X*X.T (where X=BN)
        # Work but not useful now !
        # TODO : remove the loop below
        #EXXT = np.zeros((self.dim[0], self.dim[1], self.dim[1]))
        #for n in range(self.dim[0]):
        #    EXXT[n, :, :] = np.dot(E[n, :].T, E[n, :])
        #    var_n = E2[n,:] - np.square(E[n,:])
        #    EXXT[n, :, :] += np.diag(var_n)
        #

        # Collect expectations
        self.expectations = {'E': E, 'EB': EB, 'EN': EN, 'E2': E2, 'ENN': ENN}
        #self.expectations = {'E':E, 'EB':EB, 'EN':EN, 'E2':E2, 'ENN':ENN, 'EXXT':EXXT }

    def removeDimensions(self, axis, idx):

        # Method to remove undesired dimensions
        # - axis (int): axis from where to remove the elements
        # - idx (numpy array): indices of the elements to remove
        assert axis <= len(self.dim)
        assert np.all(idx < self.dim[axis])
        self.B.removeDimensions(axis,idx)
        self.N_B0.removeDimensions(axis,idx)
        self.N_B1.removeDimensions(axis,idx)

        # TODO : check this add
        dim = list(self.dim)
        dim[1] -= len(idx)
        self.dim = tuple(dim)

        self.updateParameters()
        self.updateExpectations()

    def updateDim(self, axis, new_dim):
        # Function to update the dimensionality of a particular axis
        self.B.updateDim(axis,new_dim)
        self.N_B0.updateDim(axis,new_dim)
        self.N_B1.updateDim(axis,new_dim)
        dim = list(self.dim)
        dim[axis] = new_dim
        self.dim = tuple(dim)
