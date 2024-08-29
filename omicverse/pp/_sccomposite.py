import math
import operator
import random
import scipy
from scipy import sparse
import numpy as np

def select_stable(full, modality = "RNA", stable_number = None, criterion = "signal"):

    '''
    full is the data in np array format, either dense or sparse
    
    '''
    if sparse.issparse(full) is True:
        full = full.todense()

    if stable_number is None:
        if modality == "RNA":
            stable_number = 300
        elif modality == "ATAC":
            stable_number = 300
        elif modality == "ADT":
            stable_number = 15

    nonzero_proportion = np.array((full>0).sum(axis = 1))/full.shape[1]
    full = np.array(full)
    subset = full[np.squeeze(nonzero_proportion>0.5),]
    high_nonzero = full[np.squeeze(nonzero_proportion>0.5),]
    high_nonzero = np.log1p(high_nonzero)

    if high_nonzero.shape[0]<stable_number:
        print("Warning: too few stable features to provide reliable inference.")
        result = subset


    elif criterion == "signal":
        mean = np.mean(high_nonzero, axis = 1)
        std = np.std(high_nonzero, axis = 1)
        signal = mean/std

        #calculate the rank of the magnitude of signal for each cell
        order = (-signal).argsort()
        ranks = order.argsort()
        result = subset[ranks<stable_number]

    else:
        mean = np.mean(high_nonzero, axis = 1)
        #calculate the rank of the magnitude of mean for each cell
        order = (-mean).argsort()
        ranks = order.argsort()
        result = subset[ranks<stable_number]

    result = np.transpose(result)

    return result

#from stable_feature_selection import select_stable
import numpy as np
import pandas as pd
import scipy
from scipy.io import mmread
#import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy import special
from numpy import genfromtxt

#intialization: theta is based on estimated doublet proportion;
#alpha and beta is estimated by assuming a simple gamma distribution and
#estimated by method of moment

def initialization_rna(theta, data):

    '''
    theta is a numerical value
    data should be the count matrix in tensor form

    '''

    theta = torch.tensor(theta, device = dev)
    mu = torch.mean(data, dim=0)
    var = torch.var(data, dim=0)
    alpha = torch.square(mu)/var.to(dev)
    beta = mu/var.to(dev)

    theta = theta.requires_grad_()
    alpha = alpha.requires_grad_()
    beta = beta.requires_grad_()
    #print(torch.mean(alpha))

    return theta, alpha, beta

#log likelihood
def loglik_rna(n, theta, alpha, beta, decay, data):
    '''
    N is the user estimated maximum number of cells in a droplet
    theta, alpha, and beta are outputs from initialization function
    data should be the count matrix in tensor form
    
    '''
    poisson1 = torch.distributions.Poisson(theta)
    for k in range(n):
        gamma1 = torch.distributions.Gamma(alpha*(1+torch.tensor(k, device = dev)/\
        (1+torch.exp(-decay))), beta)
        if k ==0 :
            sum_k = gamma1.log_prob(data).exp()*\
            poisson1.log_prob(torch.tensor([k], device = dev)).exp()
        else:
            sum_k = sum_k.clone() + \
            gamma1.log_prob(data).exp()*poisson1.log_prob(torch.tensor([k], device = dev)).exp()

    l = sum_k.log().sum()
    return l

def MLE_rna(data, n=3, p=0.7, lr=0.001, tolerance = 15):
    '''
    p is a numerical value, initial guess of singlet rate; 
    this value doesn't have a big impact on parameter estimation
    data should be the count matrix in tensor form
    n is the user estimated maximum number of cells in a droplet
    tolerance controls how many steps we allow the loss not to improve
    '''
    x = data
    x.requires_grad_(False)  ## data

    decay = torch.tensor(0.0, device = dev)
    decay = decay.requires_grad_()

    theta = -math.log(p)
    theta, alpha, beta = initialization_rna(theta, data)
    alpha0 = 0.5*torch.mean(alpha).to('cpu').detach().item()
    parameters = [theta, alpha, beta, decay]


    # Defines a SGD optimizer to update the parameters
    #optimizer = optim.SGD(parameters, lr=learning_rate) # not working
    optimizer = optim.Adam(parameters, lr=lr)
    #optimizer = optim.Rprop(parameters, lr=learning_rate)
    nll_0 = -loglik_rna(n, theta, alpha, beta, decay, x)
    l = []
    singlet_rate = []


    for i in range(5000):
        nll = -loglik_rna(n, theta, alpha, beta, decay, x)
        if i % 200 == 0:
            l.append(nll.to('cpu').detach())
            singlet_rate.append(torch.tensor([-theta]).exp())
            if len(l) > 2:
                if (l[-2] - l[-1]) < 0.01*(l[0] - l[1]):
                    tolerance = tolerance - 1
        if tolerance == 0:
            break

        nll.backward()
        optimizer.step()
        optimizer.zero_grad()
        theta.data.clamp_(0.001, 1.0)
        decay.data.clamp_(-10, 10)
        alpha.data.clamp_(min = alpha0)


    return theta, alpha, beta, decay, l, singlet_rate

def log_joint_one_k_rna(data, theta, alpha, beta, decay, k0):
    '''
    k0 starts from 0, same interpretation as the k0 in the derivation
    '''
    alpha = alpha.to('cpu').detach().numpy()
    beta = beta.to('cpu').detach().numpy()
    data = data.to('cpu').numpy()
    theta = theta.to('cpu').detach().numpy()
    decay = decay.to('cpu').detach().numpy()
    alpha = alpha*(1+k0/(1+np.exp(-decay)))

    log_conditional = np.log(gamma.pdf(data, alpha, loc=0, scale=1/beta))
    sum_gene = np.sum(log_conditional, axis = 1)
    log_joint = sum_gene + np.log(poisson.pmf(k0, theta))

    return log_joint

def prob_k0_rna(data, theta, alpha, beta, decay, k0, k=3):
    # Check if float128 is supported, otherwise fall back to float64
    try:
        np.dtype('float128')
        float_type = np.float128
    except TypeError:
        float_type = np.float64

    log_joint_k0 = log_joint_one_k_rna(data, theta, alpha, beta, decay, k0)

    one_ks = np.ones((data.shape[0], k), dtype=float_type)
    for i in np.arange(k):
        one_ks[:, i] = log_joint_one_k_rna(data, theta, alpha, beta, decay, i)

    logsumexp_ks = special.logsumexp(one_ks, axis=1)
    log_prob = log_joint_k0 - logsumexp_ks
    log_prob = log_prob.astype(float_type)
    prob = np.exp(log_prob, dtype=float_type)


    return prob

def reliability_rna(data, theta, alpha, beta, decay, k=3):

    '''
    Evaluate the reliability of each cell and predict whether they will be single or double
    '''
    prob_singlet = prob_k0_rna(data, theta, alpha, beta, decay, 0, k)
    prob_doublet = 1-prob_singlet
    pred = np.where(prob_doublet > 0.5, True, False)

    alpha = alpha.to('cpu').detach().numpy()
    beta = beta.to('cpu').detach().numpy()
    data = data.to('cpu').numpy()
    theta = theta.to('cpu').detach().numpy()
    decay = decay.to('cpu').detach().numpy()

    one_ks = np.ones((data.shape[0], data.shape[1], k))
    for i in np.arange(k):
        alpha_k = alpha*(1+i/(1+np.exp(-decay)))
        one_ks[:,:,i] = np.log(gamma.pdf(data,  alpha_k, loc=0, scale=1/beta))




    reliability = 1 - (np.exp(one_ks[:,:,0]-special.logsumexp(one_ks, axis = 2)))
    #probability of doublets predicted by individual feature


    #if individual feature prediction result is the same as result by all features,
    #then record as 1. otherwise record as 0
    #then, calculate proportion of features that can individually provide correct prediction
    reliability[pred,:]=np.where(reliability[pred,:] > 0.5, 1, 0) #predicted doublets
    reliability[list(map(operator.not_, pred)),:]=\
    np.where(reliability[list(map(operator.not_, pred)),:] < 0.5, 1, 0)

    reliability = np.sum(reliability, axis = 1)/data.shape[1]

    result = np.zeros((2, data.shape[0]))
    result[0,:] = reliability
    result[1,:] = np.where(reliability <= 0.5, 1, 0)
    #flags the cells whose prediction is subject to outliers


    return result

def rna_fit_goodness(data, alpha, beta, theta, decay, k=3):

    '''
    Evaluate how well the RNA sequence data fit the theoretical model
    '''
    data = torch.round(data)
    data = data.int()
    data = data.to('cpu').numpy()
    alpha = alpha.to('cpu').detach().numpy()
    beta = beta.to('cpu').detach().numpy()
    theta = theta.to('cpu').detach().numpy()
    decay = decay.to('cpu').detach().numpy()

    empirical = \
    np.apply_along_axis(lambda x: np.bincount(x, minlength=np.max(data)+1), axis=0, arr=data)
    empirical_dist = empirical/data.shape[0]
    #empirical_dist #each column is the empirical distribution of a gene

    for i in range(empirical_dist.shape[0]-1):
        empirical_dist[i+1,] += empirical_dist[i,] #empirical cdf

    #calculate theoretical cdf below
    grid = np.expand_dims(np.arange(0, empirical_dist.shape[0], 1, dtype=int)+0.0001,axis=1)
    grid=np.repeat(grid, empirical_dist.shape[1], axis = 1)

    one_ks = np.ones((grid.shape[0], grid.shape[1], k))

    for i in np.arange(k):
        alpha_k = alpha*(1+i/(1+np.exp(-decay)))
        one_ks[:,:,i] = np.log(gamma.cdf(grid, alpha_k, loc=0, scale=1/beta))+\
        np.log(poisson.pmf(i, theta))

    logsumexp_ks = special.logsumexp(one_ks, axis = 2)
    theoretical_dist = np.exp(logsumexp_ks)

    diff = np.abs(theoretical_dist-empirical_dist)
    mean_ks = np.mean(np.amax(diff, axis = 0))


    if mean_ks > 0.33:
        print("The RNA modality goodness-of-fit score is less than 3; \
        The model may not fit the data well")

    return mean_ks


def composite_rna(adata, multiomics = False,
                  n=3, lr=0.001, p=0.7, stable_criterion = "signal",
                  stable_number = None, tolerance = 10):
    '''
    N: the maximum number of cells in a droplet that will be modeled
    lr: learning rate in the maximum likelihood estimation step; 
    Note that the recommanded learning rates for different modalities are different
    p: user estimated singlet proportion
    stable_criterion: By default, stable features are selected to have high signal-to-noise ratio.
    User may set it to "mean" so that 
    stable features are selected to have high mean expression level
    stable_number: number of stable features to be selected for modeling
    tolerance: Controls early stopping. Increasing this number may slightly 
    improve model performance but will significantly increase computing time
    
    '''
    global dev
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    if torch.cuda.is_available():
        print ("Cuda is available; Fitting the COMPOSITE model on the RNA modality")
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print(f"Found {torch.cuda.device_count()} GPUs available. Using GPU {device_id} \
        ({gpu_properties.name}) of compute capability {gpu_properties.major}.\
        {gpu_properties.minor} with "
        f"{gpu_properties.total_memory / 1e9:.1f}Gb total memory.\n")
    else:
        print ("Cuda is not available; Fitting the COMPOSITE model on RNA modality")

    rna_input = adata.X.T
    stable = select_stable(rna_input, modality = "RNA",\
    criterion = stable_criterion, stable_number = stable_number)
    stable = torch.tensor(stable, device = dev)
    stable = stable.double()
    stable = stable + torch.tensor([0.0001], device = dev)
    theta,alpha,beta,decay,loss, p = MLE_rna(stable, n=n, p=p, lr=lr, tolerance=tolerance)
    rna_fit = rna_fit_goodness(stable, alpha, beta, theta, decay, k=n)
    prob_singlet = prob_k0_rna(stable, theta, alpha, beta, decay, 0, k=n)
    prob_doublet = 1-prob_singlet
    doublet_classification = np.where(prob_doublet > 0.5, 1, 0)
    reliability_table = reliability_rna(stable, theta, alpha, beta, decay, k=n)
    rna_overall_weight = reliability_table[0,]/rna_fit
    data = {#'prob_doublet': prob_doublet,
            'doublet_classification': doublet_classification,
            'consistency':reliability_table[0,]}

    # Create DataFrame
    reliability_file = pd.DataFrame(data)
    reliability_file.index.name = 'index'
    reliability_file.reset_index(inplace=True)
    #reliability_file.to_csv(output_path,index=False)

    print("The RNA modality goodness-of-fit score is:",\
     1/rna_fit, "\n<3: poor fit \n3~5: moderate fit \n>5: good fit")

    if multiomics is False:
        return doublet_classification, reliability_table[0,]
    if multiomics is True:
        return prob_doublet, rna_overall_weight
