import numpy as np
from scipy import sparse
from scipy.spatial import distance_matrix
from scipy.special import wrightomega
import matplotlib.pyplot as plt

def unot(a,
        b,
        C,
        eps_p,
        rho,
        eps_mu=None,
        eps_nu=None,
        sparse_mtx=False,
        solver="sinkhorn",
        nitermax=10000,
        stopthr=1e-8,
        verbose=False,
        momentum_dt=0.1,
        momentum_beta=0.0):
    """ The main function calling different algorithms.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) numpy.ndarray
        The cost matrix possibly with infinity entries.
    eps_p :  float
        The coefficient of entropy regularization for P.
    rho : float
        The coefficient of penalty for unmatched mass.
    eps_mu : float, defaults to eps_p
        The coefficient of entropy regularization for mu.
    eps_nu : float, defaults to eps_p
        The coefficient of entropy regularization for nu.
    sparse_mtx : boolean, defaults to False
        Whether using sparse matrix format. If True, C should be in coo_sparse format.
    solver : str, defaults to 'sinkhorn'
        The solver to use. Choose from 'sinkhorn' and 'momentum'.
    nitermax : int, defaults to 10000
        The maximum number of iterations.
    stopthr : float, defaults to 1e-8
        The relative error threshold for stopping.
    verbose : boolean, defaults to False
        Whether to print algorithm logs.
    momentum_dt : float, defaults to 1e-1
        Step size if momentum method is used.
    momentum_beta : float, defautls to 0
        The coefficient for the momentum term if momemtum method is used.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    # Return a zero matrix if either a or b is all zero
    nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
    if len(nzind_a) == 0 or len(nzind_b) == 0:
        if sparse_mtx:
            P = sparse.coo_matrix(([],([],[])), shape=(len(a), len(b)))
        else:
            P = np.zeros([len(a), len(b)], float)
        return P
    if solver == "sinkhorn" and max(abs(eps_p-eps_mu),abs(eps_p-eps_nu))>1e-8:
        print("To use Sinkhorn algorithm, set eps_p=eps_mu=eps_nu")
        exit()
    if solver == "sinkhorn" and not sparse_mtx:
        P = unot_sinkhorn_l1_dense(a,b,C,eps_p,rho, \
            nitermax=nitermax,stopthr=stopthr,verbose=verbose)
    elif solver == "sinkhorn" and sparse_mtx:
        P = unot_sinkhorn_l1_sparse(a,b,C,eps_p,rho, \
            nitermax=nitermax,stopthr=stopthr,verbose=verbose)
    elif solver == "momentum" and not sparse_mtx: 
        P = unot_momentum_l1_dense(a,b,C,eps_p,eps_mu,eps_nu,rho, \
            nitermax=nitermax,stopthr=stopthr,dt=momentum_dt, \
            beta=momentum_beta,precondition=True,verbose=verbose)
    elif solver == "momentum" and sparse_mtx:
        print("under construction")
        exit()
    return P

def unot_sinkhorn_l2_dense(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False):
    """ Solve the unnormalized optimal transport with l2 penalty in dense matrix format.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) numpy.ndarray
        The cost matrix possibly with infinity entries.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.

    Returns
    -------
    (ns,nt) numpy.ndarray
        The optimal transport matrix.
    """
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    r = np.zeros_like(a)
    s = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err > stopthr:
        fprev = f
        gprev = g
        # Iteration
        f = eps * np.log(a) \
            - eps * np.log( np.sum( np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps ), axis=1 ) \
            + np.exp( ( r + f ) / eps ) ) + f
        g = eps * np.log(b) \
            - eps * np.log( np.sum( np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps ), axis=0 ) \
            + np.exp( ( s + g ) / eps ) ) + g
        r = - eps * wrightomega( f/eps - np.log( eps/m ) ).real
        s = - eps * wrightomega( g/eps - np.log( eps/m ) ).real
        # Check relative error
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter = niter + 1
    if verbose:
        print('Number of iterations in unot:', niter)
    P = np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps )
    return P

def unot_sinkhorn_l1_dense(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False,output_fg=False):
    """ Solve the unnormalized optimal transport with l1 penalty in dense matrix format.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) numpy.ndarray
        The cost matrix possibly with infinity entries.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.

    Returns
    -------
    (ns,nt) numpy.ndarray
        The optimal transport matrix.
    """
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err >  stopthr:
        fprev = f
        gprev = g
        # Iteration
        f = eps * np.log(a) \
            - eps * np.log( np.sum( np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps ), axis=1 ) \
            + np.exp( ( -m + f ) / eps ) ) + f
        g = eps * np.log(b) \
            - eps * np.log( np.sum( np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps ), axis=0 ) \
            + np.exp( ( -m + g ) / eps ) ) + g
        # Check relative error
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter = niter + 1
    if verbose:
        print('Number of iterations in unot:', niter)
    P = np.exp( ( f.reshape(-1,1) + g.reshape(1,-1) - C ) / eps )
    if output_fg:
        return f,g
    else:
        return P

def unot_barycenter_sinkhorn_l1_dense(a,C,eps,m,w,nitermax=5):
    L = len(a)
    n = len(a[0])
    f,g = {},{}
    for k in range(L):
        f[k] = np.zeros([n],float)
        g[k] = np.zeros([n],float)

    for i in range(nitermax):
        for k in range(L):
            f[k] = eps * np.log(a[k]) \
                - eps * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=1 ) #) + f[k] #\
                + np.exp( ( -m + f[k] ) / eps ) ) + f[k]
        Lnu = np.zeros([n],float)
        for k in range(L):
            Lnu = Lnu + w[k] * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=0 ))
        for k in range(L):
            g[k] = eps * Lnu \
                - eps * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=0 ) #) + g[k] #\
                + np.exp( ( -m + g[k] ) / eps ) ) + g[k]
    return(np.exp(Lnu))

def regular_barycenter(a,C,eps,w,nitermax=10000):
    L = len(a)
    n = len(a[0])
    f,g = {},{}
    for k in range(L):
        f[k] = np.zeros([n],float)
        g[k] = np.zeros([n],float)

    for i in range(nitermax):
        for k in range(L):
            f[k] = eps * np.log(a[k]) \
                - eps * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=1 ) ) + f[k]
        Lnu = np.zeros([n],float)
        for k in range(L):
            Lnu = Lnu + w[k] * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=0 ))
        for k in range(L):
            g[k] = eps * Lnu \
                - eps * np.log( np.sum( np.exp( ( f[k].reshape(-1,1) + g[k].reshape(1,-1) - C ) / eps ), axis=0 ) ) + g[k]
    return(np.exp(Lnu))

def unot_sinkhorn_l2_sparse(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False):
    """ Solve the unnormalized optimal transport with l2 penalty in sparse matrix format.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) scipy.sparse.coo_matrix
        The cost matrix in coo sparse format. The entries exceeds the cost cutoff are omitted. The naturally zero entries should be explicitely included.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.

    Returns
    -------
    (ns,nt) scipy.sparse.coo_matrix
        The optimal transport matrix. The locations of entries should agree with C and there might by explicit zero entries.
    """
    tmp_K = C.copy()
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    r = np.zeros_like(a)
    s = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err > stopthr:
        fprev = f
        gprev = g
        # Iteration
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        f = eps * np.log(a) \
            - eps * np.log( np.sum( tmp_K, axis=1 ).A.reshape(-1) \
            + np.exp( ( r + f ) / eps ) ) + f
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        g = eps * np.log(b) \
            - eps * np.log( np.sum( tmp_K, axis=0 ).A.reshape(-1) \
            + np.exp( ( s + g ) / eps ) ) + g
        r = - eps * wrightomega( f/eps - np.log( eps/m ) ).real
        s = - eps * wrightomega( g/eps - np.log( eps/m ) ).real
        # Check relative error
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter = niter + 1
    if verbose:
        print('Number of iterations in unot:', niter)
    tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
    return tmp_K

def unot_sinkhorn_l1_sparse(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False):
    """ Solve the unnormalized optimal transport with l1 penalty in sparse matrix format.

    Parameters
    ----------
    a : (ns,) numpy.ndarray
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) scipy.sparse.coo_matrix
        The cost matrix in coo sparse format. The entries exceeds the cost cutoff are omitted. The naturally zero entries should be explicitely included.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.

    Returns
    -------
    (ns,nt) scipy.sparse.coo_matrix
        The optimal transport matrix. The locations of entries should agree with C and there might by explicit zero entries.
    """
    tmp_K = C.copy()
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    r = np.zeros_like(a)
    s = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err > stopthr:
        fprev = f
        gprev = g
        # Iteration
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        f = eps * np.log(a) \
            - eps * np.log( np.sum( tmp_K, axis=1 ).A.reshape(-1) \
            + np.exp( ( -m + f ) / eps ) ) + f
        tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
        g = eps * np.log(b) \
            - eps * np.log( np.sum( tmp_K, axis=0 ).A.reshape(-1) \
            + np.exp( ( -m + g ) / eps ) ) + g
        # Check relative error
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter = niter + 1

    if verbose:
        print('Number of iterations in unot:', niter)
    tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
    return tmp_K

def unot_nesterov_l2_dense(a,b,C,eps1,eps2,m,nitermax=10000,stopthr=1e-8):
    dt = 0.01
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    f_old = np.array(f)
    g_old = np.array(g)
    F_old = np.array(f)
    G_old = np.array(g)
    niter = 0
    err = 100
    def Qf(ff,gg,ee1,ee2,mm,aa,bb,CC):
        out = np.exp(ff/ee1) * np.sum(np.exp((gg.reshape(1,-1)-CC)/ee1), axis=1) \
            + ee2 * wrightomega( ff*mm/ee2 - np.log( ee2/mm ) ).real / mm - aa
        return out
    def Qg(ff,gg,ee1,ee2,mm,aa,bb,CC):
        out = np.exp(gg/ee1) * np.sum(np.exp((ff.reshape(-1,1)-CC)/ee1), axis=0) \
            + ee2 * wrightomega( gg*mm/ee2 - np.log( ee2/mm ) ).real / mm - bb
        return out
    while niter <= nitermax and err > stopthr:
        f = F_old - dt * Qf(F_old, G_old, eps1, eps2, m, a, b, C)
        F = f + float(niter)/(niter+3.0) * (f-f_old)
        g = G_old - dt * Qg(F_old, G_old, eps1, eps2, m, a, b, C)
        G = g + float(niter)/(niter+3.0) * (g-g_old)
        if niter % 10 == 0:
            err_f = abs(f - f_old).max() / max(abs(f).max(), abs(f_old).max(), 1.)
            err_g = abs(g - g_old).max() / max(abs(g).max(), abs(g_old).max(), 1.)
            err = 0.5 * (err_f + err_g)
        f_old[:] = f[:]; F_old[:] = F[:]
        g_old[:] = g[:]; G_old[:] = G[:]
        niter += 1
    P = np.exp((f.reshape(-1,1)+g.reshape(1,-1)-C)/eps1)
    return P

def unot_momentum_l2_dense(a,b,C,eps_p,eps_mu,eps_nu,m,nitermax=1e4,stopthr=1e-8,dt=0.01,beta=0.8):
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    f_old = np.array(f)
    g_old = np.array(g)
    F_old = np.array(f)
    G_old = np.array(g)
    niter = 0
    err = 100
    def Qf(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(ff/ee_p) * np.sum(np.exp((gg.reshape(1,-1)-CC)/ee_p), axis=1) \
            + ee_mu * wrightomega( ff/ee_mu - np.log( ee_mu/mm ) ).real / mm - aa
        return out
    def Qg(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(gg/ee_p) * np.sum(np.exp((ff.reshape(-1,1)-CC)/ee_p), axis=0) \
            + ee_nu * wrightomega( gg/ee_nu - np.log( ee_nu/mm ) ).real / mm - bb
        return out
    while niter <= nitermax and err > stopthr:
        F = beta * F_old + Qf(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        f = f_old - dt * F
        G = beta * G_old + Qg(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        g = g_old - dt * G
        f_old[:] = f[:]; F_old[:] = F[:]
        g_old[:] = g[:]; G_old[:] = G[:]
        niter += 1
    P = np.exp((f.reshape(-1,1)+g.reshape(1,-1)-C)/eps_p)
    return P

def unot_momentum_l1_dense(a,b,C,eps_p,eps_mu,eps_nu,m,nitermax=1e4,stopthr=1e-8,dt=0.01,beta=0.8,precondition=False,verbose=False):
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    if precondition:
        f,g = unot_sinkhorn_l1_dense(a,b,C,eps_p,m,output_fg=True)
    f_old = np.array(f)
    g_old = np.array(g)
    F_old = np.array(f)
    G_old = np.array(g)
    niter = 0
    err = 100
    def Qf(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(ff/ee_p) * np.sum(np.exp((gg.reshape(1,-1)-CC)/ee_p), axis=1) \
            + np.exp((ff-mm)/ee_mu) - aa
        return out
    def Qg(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(gg/ee_p) * np.sum(np.exp((ff.reshape(-1,1)-CC)/ee_p), axis=0) \
            + np.exp((gg-mm)/ee_nu) - bb
        return out
    while niter <= nitermax and err > stopthr:
        F = beta * F_old + Qf(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        f = f_old - dt * F
        G = beta * G_old + Qg(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        g = g_old - dt * G
        if niter % 10 == 0:
            err_f = abs(f - f_old).max() / max(abs(f).max(), abs(f_old).max(), 1.)
            err_g = abs(g - g_old).max() / max(abs(g).max(), abs(g_old).max(), 1.)
            err = 0.5 * (err_f + err_g)
        f_old[:] = f[:]; F_old[:] = F[:]
        g_old[:] = g[:]; G_old[:] = G[:]
        niter += 1
    P = np.exp((f.reshape(-1,1)+g.reshape(1,-1)-C)/eps_p)
    if verbose:
        print(niter)
    return P

def unot_momentum_l1_2end_dense(a,b,C,eps_p,eps_mu,eps_nu,m,nitermax=1e4,stopthr=1e-8,dt=0.01,beta=0.8):
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    f_old = np.array(f)
    g_old = np.array(g)
    F_old = np.array(f)
    G_old = np.array(g)
    niter = 0
    err = 100
    def Qf(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(ff/ee_p) * np.sum(np.exp((gg.reshape(1,-1)-CC)/ee_p), axis=1) \
            + aa / (np.exp(-(ff-mm)/ee_mu)+1) - aa
        return out
    def Qg(ff,gg,ee_p,ee_mu,ee_nu,mm,aa,bb,CC):
        out = np.exp(gg/ee_p) * np.sum(np.exp((ff.reshape(-1,1)-CC)/ee_p), axis=0) \
            + bb / (np.exp(-(gg-mm)/ee_nu)+1) - bb
        return out
    while niter <= nitermax and err > stopthr:
        F = beta * F_old + Qf(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        f = f_old - dt * F
        G = beta * G_old + Qg(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
        g = g_old - dt * G
        f_old[:] = f[:]; F_old[:] = F[:]
        g_old[:] = g[:]; G_old[:] = G[:]
        niter += 1
    P = np.exp((f.reshape(-1,1)+g.reshape(1,-1)-C)/eps_p)
    print(niter)
    return P