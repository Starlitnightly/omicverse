from __future__ import division
from time import sleep
from time import time
import numpy as np
import scipy as s
import pandas as pd
import numpy.ma as ma
import os
import sys
import h5py

from ..core.nodes import *

def mask_data(data, mask_fraction):
    """ Method to mask data values, mainly used to evaluate imputation

    PARAMETERS
    ----------
    data: ndarray
    mask_fraction: float with the fraction of values to mask (from 0 to 1)
    """

    D = data.shape[1]
    N = data.shape[0]

    mask = np.ones(N*D)
    mask[:int(round(N*D*mask_fraction))] = np.nan
    np.random.shuffle(mask)
    mask = np.reshape(mask, [N, D])
    data *= mask

    return data

def _gaussianise_vec(vec):
    # take ranks and scale to uniform
    vec = s.stats.rankdata(vec, 'dense').astype(float)
    vec /= (vec.max()+1.)

    # transform uniform to gaussian using probit
    vec_norm = np.sqrt(2.) * s.special.erfinv(2.*vec-1.)  # TODO to double check
    # phenotype_norm = np.reshape(phenotype_norm, [len(phenotype_norm), 1])

    return vec_norm

def gaussianise(Y_m, axis=0):
    # double check axis for pandas
    Y_norm = Y_m.apply(_gaussianise_vec, axis)

    return Y_norm

def process_data(data, likelihoods, data_opts, samples_groups):

    for m in range(len(data)):

        # For some wierd reason, when using reticulate from R, missing values are stored as -2147483648
        data[m][data[m] == -2147483648] = np.nan

        # Removing features with no variance
        var = data[m].std(axis=0)
        if np.any(var==0.):
            print("Warning: %d features(s) in view %d have zero variance, consider removing them before training the model...\n" % ((var==0.).sum(), m))
            sys.stdout.flush()

        # Check that there are no features full of missing values
        tmp = np.isnan(data[m]).mean(axis=0)
        if np.any(tmp==1.):
            print("Warning: %d features(s) in view %d are full of missing values, please consider removing them before training the model...\n" % ((tmp==0.).sum(), m))
            sys.stdout.flush()


        # Centering and scaling is only appropriate for gaussian data
        if likelihoods[m] in ["gaussian"]:

            # Center features per group
            if data_opts['center_groups']:
                for g in data_opts['groups_names']:
                    filt = [gp==g for gp in samples_groups]
                    data[m][filt,:] -= np.nanmean(data[m][filt,:],axis=0)

            # Scale views to unit variance
            if data_opts['scale_views']:
                data[m] /= np.nanstd(data[m])

            # Scale groups to unit variance
            if data_opts['scale_groups']:
                for g in data_opts['groups_names']:
                    filt = [gp==g for gp in samples_groups]
                    data[m][filt,:] /= np.nanstd(data[m][filt,:])

    return data

def guess_likelihoods(data):
    """
    Method to infer likelihoods from the data
    (Note groups are already concatenated when calling this function)
    """
    M = len(data)

    likelihoods = ["gaussian" for m in range(M)]
    for m in range(M):
        mask = ~np.isnan(data[m])
        if np.isin(data[m][mask],[0,1]).all():
            likelihoods[m] = "bernoulli"
        else:
            if np.all( (data[m][mask]%1)==0):
                likelihoods[m] = "poisson"  

    return likelihoods
