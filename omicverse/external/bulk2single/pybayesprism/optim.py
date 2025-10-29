import numpy as np
import multiprocessing
import xarray as xr 
from scipy.optimize import minimize
import pandas as pd

#from pybayesprism import process_input
from .process_input import norm_to_one
#from pybayesprism import references as rf
from .references import RefPhi, RefTumor


####################################################



def logsumexp(x):
    y = np.max(x)
    return y + np.log(np.sum(np.exp(x - y)))


def log_posterior_gamma(gamma_t, args):
    phi_t = args["phi_t"]
    phi_t_log = args["phi_t_log"]
    Z_gt_t = args["Z_gt_t"]
    Z_t_t = args["Z_t_t"]
    prior_num = args["prior_num"]

    x = phi_t_log + gamma_t
    psi_t_log = x - logsumexp(x)
    log_likelihood = np.nansum(Z_gt_t * psi_t_log)
    log_prior = np.sum(prior_num * gamma_t ** 2)
    log_posterior = log_likelihood + log_prior
    return -1 * log_posterior

def log_posterior_gamma_grad(gamma_t, args):
    phi_t = args["phi_t"]
    phi_t_log = args["phi_t_log"]
    Z_gt_t = args["Z_gt_t"]
    Z_t_t = args["Z_t_t"]
    prior_num = args["prior_num"]

    psi_t = transform_phi_t(phi_t, gamma_t)
    log_likelihood_grad = Z_gt_t - (Z_t_t * psi_t)
    log_prior_grad = 2 * prior_num * gamma_t
    log_posterior_grad = log_likelihood_grad + log_prior_grad
    return -1 * log_posterior_grad

def optimize_psi_multi(phi_t, pgi_t_log, Z_gt_t, Z_t_t, prior_num):
    args = {"phi_t" : phi_t,
            "phi_t_log" : pgi_t_log,
            "Z_gt_t" : Z_gt_t, 
            "Z_t_t" : Z_t_t,
            "prior_num": prior_num}
    res = minimize(fun = log_posterior_gamma, x0 = np.zeros(phi_t.shape[0]), 
                    args = args, method = "CG", jac = log_posterior_gamma_grad)
    value =  [log_posterior_gamma(x, args = args) for x in res.x] 
    return (res.x, value)


def optimize_psi(phi, Z_gt, prior_num, opt_control):
    phi_dim = (phi.index, phi.columns)
    phi = phi.to_numpy()
    phi_log = np.log(phi)
    Z_gt = Z_gt.to_numpy()
    Z_t = np.sum(Z_gt, axis = 0)

    print("running with " + str(opt_control['n.cores']) + " cores!")
    star_input = [(phi[i,:], phi_log[i,:], Z_gt[:,i], Z_t[i], prior_num) for i in range(phi.shape[0])]
    with multiprocessing.Pool(processes = opt_control['n.cores']) as pool:
        results = pool.starmap(optimize_psi_multi, star_input)

    opt_gamma = np.vstack([res for (res, val) in results])
    value = np.sum([val for (res, val) in results])
    
    opt_gamma[np.apply_along_axis(np.max, 1, np.abs(opt_gamma)) > 20, :] = 0
    
    phi = pd.DataFrame(phi, index = phi_dim[0], columns = phi_dim[1])
    psi = transform_phi(phi, opt_gamma)
    psi = pd.DataFrame(psi.to_numpy(), index = phi_dim[0], columns = phi_dim[1])

    return {"psi": psi, "value": value}



####################################################



def transform_phi_transpose(phi_transpose, gamma):
    psi = np.empty((phi_transpose.shape[1], phi_transpose.shape[0]))
    for t in range(psi.shape[0]):
        psi[t, :] = transform_phi_t(phi_transpose[:, t], gamma)
    psi = xr.DataArray(psi, 
                       corrds = [phi_transpose.coords[phi_transpose.dims[1]], 
                                 phi_transpose.coords[phi_transpose.dims[0]]],
                       dims=[phi_transpose.dims[1], phi_transpose.dims[0]])
    return psi

def log_mle_gamma(gamma, args):
    phi_transpose, phi_log_transpose, Z_tg, Z_t = args
    x = phi_log_transpose + gamma
    psi_log = x.T - np.apply_along_axis(logsumexp, 1, x)
    log_likelihood = np.sum(Z_tg * psi_log)
    return -log_likelihood

def log_mle_gamma_grad(gamma, args):
    phi_transpose, phi_log_transpose, Z_tg, Z_t = args
    psi = transform_phi_transpose(phi_transpose, gamma)
    log_likelihood_grad = np.sum(Z_tg - (Z_t * psi), axis = 0)
    return -log_likelihood_grad



def optimize_psi_oneGamma(phi, Z_gt, opt_control, optimizer = "Rcgmin"):
    opt_control["ncores"] = None
    
    if optimizer == "Rcgmin":
        method = "CG"
    elif optimizer == "BFGS":
        method = "BFGS"

    args = (phi.T, np.log(phi).T, Z_gt.T, np.sum(Z_gt, axis = 0))

    opt_res = minimize(fun = log_mle_gamma, 
                       x0 = np.zeros(phi.shape[1]), 
                       args = args, 
                       method = method, 
                       jac = log_mle_gamma_grad)
    
    opt_gamma = opt_res.x
    value = [log_mle_gamma(x, args) for x in opt_gamma]

    psi = transform_phi(phi, np.vstack([opt_gamma] * phi.shape[0]))
    psi = xr.DataArray(psi, 
                       corrds = [phi.coords[phi.dims[1]], 
                                 phi.coords[phi.dims[0]]],
                       dims=[phi.dims[1], phi.dims[0]])

    return {"psi": psi, "value": value, "gamma": opt_gamma}



####################################################



def transform_phi_t(phi_t, gamma_t):
    stabilizing_constant = np.max(gamma_t)
    gamma_stab = gamma_t - stabilizing_constant
    psi_t = phi_t * np.exp(gamma_stab)
    psi_t = psi_t / np.sum(psi_t)
    return psi_t


def transform_phi(phi, gamma):
    psi = pd.DataFrame(np.zeros_like(phi), index = phi.index, columns = phi.columns)
    for t in range(psi.shape[0]):
        psi.iloc[t, :] = transform_phi_t(phi.iloc[t, :], gamma[t, :])
    return psi


def get_MLE_psi_mal(Z_ng_mal, pseudo_min):
    row_sums = Z_ng_mal.sum(axis = 1)
    mle_psi_mal = Z_ng_mal.div(row_sums, axis = 0)
    mle_psi_mal_pseudo_min = norm_to_one(mle_psi_mal, pseudo_min)
    return mle_psi_mal_pseudo_min


def xr_to_pd(xr):
    assert xr.ndim == 2
    return pd.DataFrame(xr.to_numpy(), 
            index = list(xr.coords[xr.dims[0]].values),
            columns = list(xr.coords[xr.dims[1]].values))


def update_reference(Z, phi_prime, map, key, opt_control, optimizer=["MAP", "MLE"]):
    print("Update the reference matrix ...")
    
    sigma = opt_control["sigma"]
    opt_control.pop("sigma")
    
    optimizer = opt_control["optimizer"]
    opt_control.pop("optimizer")
    
    if key is None:
        Z_gt = np.sum(Z, axis = 0)
        Z_gt = xr_to_pd(Z_gt)

        if optimizer == "MAP":
            psi = optimize_psi(phi = phi_prime.phi, 
                               Z_gt = Z_gt, 
                               prior_num = -1 / (2 * sigma ** 2), 
                               opt_control = opt_control)["psi"]
        if optimizer == "MLE":
            psi = optimize_psi_oneGamma(phi = phi_prime.phi, 
                                        Z_gt = Z_gt, 
                                        opt_control = opt_control)["psi"]
        return RefPhi(psi, None)
    else:
        
        Z_ng_mal = Z.loc[:, :, key]

        Z_ng_mal = xr_to_pd(Z_ng_mal)

        psi_mal = get_MLE_psi_mal(Z_ng_mal, phi_prime.pseudo_min)

        cellType_env = [cell_type for cell_type in map.keys() if cell_type != key]
        Z_gt_env = xr_to_pd(np.sum(Z.loc[:, :, cellType_env], axis = 0))
        phi_env = phi_prime.phi.loc[cellType_env, :]
        
        if optimizer == "MAP":
            psi_env = optimize_psi(phi = phi_env, 
                                   Z_gt = Z_gt_env, 
                                   prior_num = -1 / (2 * sigma ** 2), 
                                   opt_control = opt_control)["psi"]

        if optimizer == "MLE":
            psi_env = optimize_psi_oneGamma(phi = phi_env, 
                                            Z_gt = Z_gt_env, 
                                            opt_control = opt_control)["psi"]
        return RefTumor(psi_mal, psi_env, key, None)


