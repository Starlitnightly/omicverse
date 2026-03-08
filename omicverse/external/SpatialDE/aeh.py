import logging

import numpy as np
import pandas as pd
from scipy import optimize

from . import base


# Variational update functions

def Q_Z_expectation(mu, Y, s2e, N, C, G, pi=None):
    if pi is None:
        pi = np.ones(C) / C

    log_rho = np.log(pi[None, :]) \
              - 0.5 * N * np.log(s2e) \
              - 0.5 * np.sum((mu.T[None, :, :] - Y[:, None, :]) ** 2, 2) / s2e \
              - 0.5 * N * np.log(2 * np.pi)

    # Subtract max per row for numerical stability, and add offset from 0 for same reason.
    rho = np.exp(log_rho - log_rho.max(1)[:, None]) + 1e-12
    # Then evaluate softmax
    r = (rho.T / (rho.sum(1))).T
    
    return r


def Q_mu_k_expectation(Z_k, Y, K, s2e):
    y_k_tilde = np.dot(Z_k, Y) / s2e
    Sytk = np.dot(K, y_k_tilde)
    IpSDk = np.eye(K.shape[0]) + K * Z_k.sum() / s2e
    m_k = np.linalg.solve(IpSDk, Sytk)
    
    return m_k


def Q_mu_expectation(Z, Y, K, s2e):
    m = np.zeros((Y.shape[1], Z.shape[1]))

    y_k_tilde = np.dot(Z.T, Y).T / s2e

    for k in range(Z.shape[1]):
        m[:, k] = Q_mu_k_expectation(Z[:, k], Y, K, s2e)

    return m


# Log expectation functions for the ELBO

def ln_P_YZms(Y, Z, mu, s2e, pi=None):
    ''' Expecation of ln P(Y | Z, mu, s2e)
    '''
    G = Y.shape[0]
    N = Y.shape[1]
    C = Z.shape[1]
    if pi is None:
        pi = np.ones(C) / C
    
    log_rho = np.log(pi[None, :]) \
              - 0.5 * N * np.log(s2e) \
              - 0.5 * np.sum((mu.T[None, :, :] - Y[:, None, :]) ** 2, 2) / s2e \
              - 0.5 * N * np.log(2 * np.pi)
            
    return (Z * log_rho).sum()


def ln_P_mu(mu, K):
    ''' Expectation of ln P(mu)
    '''
    N = K.shape[0]
    C = mu.shape[1]
    ll = 0
    for k in range(C):
        ll = ll + np.linalg.det(K)
        ll = ll + mu[:, k].dot(np.linalg.solve(K, mu[:, k]))
        ll = ll + N * np.log(2 * np.pi)
        
    ll = -0.5 * ll
    
    return ll


def ln_P_Z(Z, pi=None):
    ''' Expectation of ln P(Z)
    '''
    C = Z.shape[1]
    if pi is None:
        pi = np.ones(C) / C
        
    return np.dot(Z, np.log(pi)).sum()


def ln_Q_mu(K, Z, s2e):
    ''' Expecation of ln Q(mu)
    '''
    N = K.shape[0]
    C = Z.shape[1]
    G_k = Z.sum(0)
    
    ll = 0
    U, S = base.factor(K)
    for k in range(C):
        ll = ll - (1. / S + G_k[k] / s2e).sum()
        ll = ll + N * np.log(2 * np.pi)
        
    
    ll = -0.5 * ll
    
    return ll


def ln_Q_Z(Z, r):
    ''' Expectation of ln Q(Z)
    '''
    return np.sum(Z * np.log(r))


# ELBO and ELBO objective

def ELBO(Y, r, m, s2e, K, K_0, s2e_0, pi=None):
    L = ln_P_YZms(Y, r, m, s2e, pi) + ln_P_Z(r, pi) + ln_P_mu(m, K) \
        - ln_Q_Z(r, r) - ln_Q_mu(K_0, r, s2e_0)
    
    return L


def make_elbojective(Y, r, m, X, K_0, s2e_0, pi=None):
    def elbojective(log_s2e):
        return -ELBO(Y, r, m, np.exp(log_s2e), K_0, K_0, s2e_0, pi)
    
    return elbojective


# Model fitting

def fit_patterns(X, Y, C, l, s2e_0=1.0, verbosity=0, maxiter=100, printerval=1, opt_interval=1, delta_elbo_threshold=1e-2):
    ''' Fit spatial patterns using Automatic Expression Histology.

    X : Spatial coordinates

    Y : Gene expression values

    C : The number of patterns

    l : The charancteristic length scale of the clusters

    Returns

    final_elbo : The final ELBO value.

    m : The posterior mean underlying expression values for each cluster.

    r : The posterior pattern assignment probabilities for each gene and pattern.

    s2e : The estimated noise parameter of the model
    '''
    # Set up constants
    G = Y.shape[0]
    N = Y.shape[1]
    eps = 1e-8 * np.eye(N)
    
    s2e = s2e_0
    
    K = base.SE_kernel(X, l) + eps
    
    # Randomly initialize
    r = np.random.uniform(size=(G, C))
    r = r / r.sum(0)
    
    pi = r.sum(0) / G

    m = np.random.normal(size=(N, C))
    
    elbo_0 = ELBO(Y, r, m, s2e, K, K, s2e, pi)
    elbo_1 = elbo_0

    if verbosity > 0:
        print('iter {}, ELBO: {:0.2e}'.format(0, elbo_1))

    if verbosity > 1:
        print()

    for i in range(maxiter):
        if (i % opt_interval == (opt_interval - 1)):
            elbojective = make_elbojective(Y, r, m, X, K, s2e, pi)
            
            o = optimize.minimize_scalar(elbojective)
            s2e = np.exp(o.x)
            
            
        r = Q_Z_expectation(m, Y, s2e, N, C, G, pi)
        m = Q_mu_expectation(r, Y, K, s2e)
        
        pi = r.sum(0) / G

        elbo_0 = elbo_1
        elbo_1 = ELBO(Y, r, m, s2e, K, K, s2e, pi)
        delta_elbo = np.abs(elbo_1 - elbo_0)

        if verbosity > 0 and (i % printerval == 0):
            print('iter {}, ELBO: {:0.2e}, delta_ELBO: {:0.2e}'.format(i + 1, elbo_1, delta_elbo))
            
            if verbosity > 1:
                print('ln(l): {:0.2f}, ln(s2e): {:.2f}'.format(np.log(l), np.log(s2e)))
                
            if verbosity > 2:
                line1 = 'P(Y | Z, mu, s2e): {:0.2e}, P(Z): {:0.2e}, P(mu): {:0.2e}' \
                        .format(ln_P_YZms(Y, r, m, s2e, pi), ln_P_Z(r, pi), ln_P_mu(m, K))
                line2 = 'Q(Z): {:0.2e}, Q(mu): {:0.2e}'.format(ln_Q_Z(r, r), ln_Q_mu(K, r, s2e))
                print(line1 + '\n' + line2)
            
            if verbosity > 1:
                print()
            
        if delta_elbo < delta_elbo_threshold:
            if verbosity > 0:
                print('Converged on iter {}'.format(i + 1))

            break
            
    else:
        print('Warning! ELBO dit not converge after {} iters!'.format(i + 1))

    final_elbo = ELBO(Y, r, m, s2e, K, K, s2e, pi)
        
    return final_elbo, m, r, s2e


def spatial_patterns(X, exp_mat, DE_mll_results, C, l, **kwargs):
    ''' Group spatially variable genes into spatial patterns using
    automatic expression histology (AEH).

    X : Spatial coordinates

    exp_mat : Expression matrix, appropriately normalised.

    DE_mll_results : Results table from SpatialDE, after filtering
        for significance level.

    C : The number of spatial patterns

    **kwards are passed on to the function fit_patterns()

    Returns

    pattern_results : A DataFrame with pattern membership information
        for each gene

    patterns : The posterior mean underlying expression for genes in
        given spatial patterns.
    '''
    Y = exp_mat[DE_mll_results['g']].values.T

    # This is important, we only care about co-expression, not absolute levels.
    Y = (Y.T - Y.mean(1)).T
    Y = (Y.T / Y.std(1)).T

    _, m, r, _ = fit_patterns(X, Y, C, l, **kwargs)

    cres = pd.DataFrame({'g': DE_mll_results['g'],
                         'pattern': r.argmax(1),
                         'membership': r.max(1)})

    m = pd.DataFrame.from_records(m)
    m.index = exp_mat.index

    return cres, m

