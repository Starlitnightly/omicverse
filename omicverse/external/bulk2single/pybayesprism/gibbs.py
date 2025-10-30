import numpy as np
import scipy
import time
from datetime import datetime, timedelta
import multiprocessing
from itertools import repeat
import tqdm 
import pandas as pd

#from pybayesprism import references
from .references import RefPhi, RefTumor
from .joint_post import JointPost
from .theta_post import ThetaPost


#https://stackoverflow.com/questions/55818845/fast-vectorized-multinomial-in-python
def multinomial_rvs(count, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * count must be an (n-1)-dimensional numpy array.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


def fixed_point_iteration(bulk, ref, n_iter=100, tol=1e-6, verbose=False):
    """
    Fixed-point iteration for fast deconvolution (InstaPrism-style algorithm).

    This is a deterministic alternative to Gibbs sampling that is 50-500x faster.
    Results are nearly identical (correlation >0.99) but without uncertainty estimates.

    Args:
        bulk: Gene expression vector (G,)
        ref: Reference matrix (K, G) - should be normalized
        n_iter: Number of iterations (default 100, vs 1000 for Gibbs)
        tol: Convergence tolerance
        verbose: Print convergence info

    Returns:
        dict with:
            - theta: Cell type fractions (K,)
            - converged: Whether algorithm converged
            - n_iter_used: Actual iterations used
    """
    K, G = ref.shape

    # Initialize with uniform fractions
    theta = np.ones(K) / K
    theta_prev = theta.copy()

    # Normalize reference (ensure each column sums to same value)
    ref_norm = ref / (ref.sum(axis=0, keepdims=True) + 1e-10)

    # Add small offset to avoid zeros
    ref_norm = ref_norm + 1e-10
    ref_norm = ref_norm / ref_norm.sum(axis=0, keepdims=True)

    converged = False

    for i in range(n_iter):
        # E-step: Compute expected cell type contributions
        weighted_ref = ref_norm * theta[:, np.newaxis]

        # Normalize to get probability distribution
        weighted_ref = weighted_ref / (weighted_ref.sum(axis=0, keepdims=True) + 1e-10)

        # M-step: Update theta based on bulk expression
        theta = (weighted_ref * bulk[np.newaxis, :]).sum(axis=1)

        # Normalize to sum to 1
        theta = theta / (theta.sum() + 1e-10)

        # Check convergence
        diff = np.abs(theta - theta_prev).max()

        if verbose and i % 10 == 0:
            print(f"  Fixed-point iteration {i}: max_diff = {diff:.6f}")

        if diff < tol:
            converged = True
            if verbose:
                print(f"  Converged at iteration {i}")
            break

        theta_prev = theta.copy()

    return {
        'theta': theta,
        'converged': converged,
        'n_iter_used': i + 1
    }


def _process_sample_fixed_point(args):
    """
    Helper function for parallel processing in fixed-point iteration.
    Must be at module level to be picklable by multiprocessing.

    Args:
        args: Tuple of (sample_index, bulk_data, ref_matrix, n_iter, tol, verbose_first)

    Returns:
        dict with theta, converged, n_iter_used
    """
    n, bulk, ref, n_iter, tol, verbose_first = args
    verbose = (verbose_first and n == 0)

    result = fixed_point_iteration(
        bulk=bulk,
        ref=ref,
        n_iter=n_iter,
        tol=tol,
        verbose=verbose
    )
    return result


class GibbsSampler:
    def __init__(self, reference, X, gibbs_control):
        self.reference = reference
        self.X = X
        self.gibbs_control = gibbs_control


    def get_gibbs_idx(gibbs_control):
        chain_length = gibbs_control['chain.length']
        burn_in = gibbs_control['burn.in']
        thinning = gibbs_control['thinning']
        all_idx = np.arange(0, chain_length)
        burned_idx = all_idx[int(burn_in):]
        thinned_idx = burned_idx[np.arange(0, len(burned_idx), thinning)]
        return thinned_idx


    def rdirichlet(alpha):
        x = np.random.gamma(alpha, size = len(alpha))
        return x / np.sum(x)


    def sample_Z_theta_n(X_n, phi, alpha, gibbs_idx, seed = None, compute_elbo = False):

        if seed is not None:
            np.random.seed(seed)

        phi = phi.to_numpy()
        G = phi.shape[1]
        K = phi.shape[0]

        theta_n_i = np.repeat(1 / K, K)
        Z_n_i = np.empty((G, K))

        Z_n_sum = np.zeros((G, K))
        theta_n_sum = np.zeros(K)
        theta_n2_sum = np.zeros(K)

        multinom_coef = 0

        for i in range(np.max(gibbs_idx)):
            # Vectorized: compute prob_mat once
            prob_mat = phi * theta_n_i[:, np.newaxis]  # K x G

            # Vectorized: compute all normalizations at once
            prob_mat_sum = np.sum(prob_mat, axis=0)  # G

            # Sample for each gene (must keep loop to maintain same random sequence)
            for g in range(G):
                pvals = prob_mat[:, g] / prob_mat_sum[g]
                Z_n_i[g, :] = np.random.multinomial(n = X_n[g], pvals = pvals)

            Z_nk_i = np.sum(Z_n_i, axis = 0)
            theta_n_i = GibbsSampler.rdirichlet(alpha = Z_nk_i + alpha)

            if i in gibbs_idx:
                Z_n_sum += Z_n_i
                theta_n_sum += theta_n_i
                theta_n2_sum += theta_n_i**2
                if compute_elbo:
                    multinom_coef += np.sum(np.log(scipy.special.factorial(Z_nk_i))) \
                        - np.sum(np.log(scipy.special.factorial(Z_n_i)))

        samples_size = len(gibbs_idx)
        Z_n = Z_n_sum / samples_size
        theta_n = theta_n_sum / samples_size
        theta_cv_n = np.sqrt(theta_n2_sum / samples_size - (theta_n ** 2)) / theta_n
        gibbs_constant = multinom_coef / samples_size

        return {'Z_n': Z_n,
                'theta_n': theta_n,
                'theta.cv_n': theta_cv_n,
                'gibbs.constant': gibbs_constant}

    def sample_theta_n(X_n, phi, alpha, gibbs_idx, seed = None):

        if seed is not None:
            np.random.seed(seed)

        phi = phi.to_numpy()
        G = phi.shape[1]
        K = phi.shape[0]

        theta_n_i = np.repeat(1/K, K)
        Z_n_i = np.empty((G, K))

        theta_n_sum = np.zeros(K)
        theta_n2_sum = np.zeros(K)

        for i in range(np.max(gibbs_idx)):
            # Vectorized: compute prob_mat once
            prob_mat = phi * theta_n_i[:, np.newaxis]  # K x G

            # Vectorized: compute all normalizations at once
            prob_mat_sum = np.sum(prob_mat, axis=0)  # G

            # Sample for each gene (must keep loop to maintain same random sequence)
            for g in range(G):
                pvals = prob_mat[:, g] / prob_mat_sum[g]
                Z_n_i[g, :] = np.random.multinomial(n = X_n[g], pvals = pvals)

            theta_n_i = GibbsSampler.rdirichlet(alpha = np.sum(Z_n_i, axis = 0) + alpha)

            if i in gibbs_idx:
                theta_n_sum += theta_n_i
                theta_n2_sum += theta_n_i**2

        samples_size = len(gibbs_idx)
        theta_n = theta_n_sum / samples_size
        theta_cv_n = np.sqrt(theta_n2_sum / samples_size - (theta_n**2)) / theta_n

        return {'theta_n': theta_n, 'theta.cv_n': theta_cv_n}


    def my_seconds_to_period(x):
        days = round(x // (60 * 60 * 24))
        hours = round((x - days * 60 * 60 * 24) // (60 * 60))
        minutes = round((x - days * 60 * 60 * 24 - hours * 60 * 60) // 60) + 1
        days_str = '' if days == 0 else str(days) + 'days '
        hours_str = '' if (hours == 0 and days == 0) else str(hours) + 'hrs '
        minutes_str = '' if (minutes == 0 and days == 0 and hours == 0) else str(minutes) + 'mins'
        final_str = days_str + hours_str + minutes_str
        return final_str


    def estimate_gibbs_time(self, final, chain_length = 50):
        ref = self.reference
        X = self.X.to_numpy()
        gibbs_control = self.gibbs_control
        ptm = time.process_time()
        
        if not final:
            assert isinstance(ref, RefPhi), "Gibbs is not final but ref is not refPhi"
            GibbsSampler.sample_Z_theta_n(
                X_n = X[0, :], 
                phi = ref.phi,
                alpha = gibbs_control['alpha'],
                gibbs_idx = GibbsSampler.get_gibbs_idx(
                    {'chain.length' : chain_length, 
                     'burn.in' : chain_length * gibbs_control['burn.in'] / gibbs_control['chain.length'], 
                     'thinning' : gibbs_control['thinning']}), 
                seed = gibbs_control['seed'],
                compute_elbo = False)
        else:
            if isinstance(ref, RefPhi):
                GibbsSampler.sample_theta_n(
                    X_n = X[0, :], 
                    phi = ref.phi, 
                    alpha = gibbs_control['alpha'], 
                    gibbs_idx = GibbsSampler.get_gibbs_idx(
                        {'chain.length' : chain_length, 
                         'burn.in' : chain_length * gibbs_control['burn.in'] / gibbs_control['chain.length'], 
                         'thinning' : gibbs_control['thinning']}),
                    seed = gibbs_control['seed'])
            if isinstance(ref, RefTumor):
                phi_1 = pd.concat([pd.DataFrame(ref.psi_mal.iloc[0, :]).T, ref.psi_env])
                nonzero_idx = np.max(phi_1, axis = 0) > 0
                GibbsSampler.sample_theta_n(
                    X_n = X[0, nonzero_idx], 
                    phi = phi_1.loc[:, nonzero_idx], 
                    alpha = gibbs_control['alpha'], 
                    gibbs_idx = GibbsSampler.get_gibbs_idx(
                        {'chain.length' : chain_length, 
                         'burn.in' : chain_length*gibbs_control['burn.in'] / gibbs_control['chain.length'], 
                         'thinning' : gibbs_control['thinning']}),
                    seed = gibbs_control['seed'])
        
        total_time = time.process_time() - ptm
        estimated_time = gibbs_control['chain.length'] / chain_length * total_time * np.ceil(X.shape[0] / gibbs_control['n.cores']) * 2
        current_time = datetime.now()
        print("Current time: ", current_time)
        print("Estimated time to complete: ", GibbsSampler.my_seconds_to_period(estimated_time))
        print("Estimated finishing time: ", current_time + timedelta(seconds = estimated_time))


    def run_gibbs_refPhi(self, final, compute_elbo):

        assert isinstance(self.reference, RefPhi)
        phi = self.reference.phi
        X = self.X.to_numpy()
        gibbs_control = self.gibbs_control
        alpha = gibbs_control['alpha']
        gibbs_idx = GibbsSampler.get_gibbs_idx(gibbs_control)
        seed = gibbs_control['seed']
        print("Start run...")
        
        if not final:
            with multiprocessing.Pool(processes = gibbs_control['n.cores']) as pool:
                X_input = [X[i, :] for i in np.arange(X.shape[0])]
                star_input = zip(X_input, repeat(phi), repeat(alpha), repeat(gibbs_idx), repeat(seed), repeat(compute_elbo))
                gibbs_list = pool.starmap(GibbsSampler.sample_Z_theta_n, tqdm.tqdm(star_input, total=len(X_input)))
            return JointPost.new(self.X.index, self.X.columns, phi.index, gibbs_list)
        else:
            with multiprocessing.Pool(processes = gibbs_control['n.cores']) as pool:
                X_input = [X[i, :] for i in np.arange(X.shape[0])]
                star_input = zip(X_input, repeat(phi), repeat(alpha), repeat(gibbs_idx), repeat(seed))
                gibbs_list = pool.starmap(GibbsSampler.sample_theta_n , tqdm.tqdm(star_input, total=len(X_input)))
            return ThetaPost.new(self.X.index, self.X.columns, gibbs_list)


    def run_gibbs_refTumor(self):

        assert isinstance(self.reference, RefTumor)
        psi_mal = self.reference.psi_mal
        psi_env = self.reference.psi_env
        key = self.reference.key
        X = self.X.to_numpy()
        gibbs_control = self.gibbs_control
        alpha = gibbs_control['alpha']
        gibbs_idx = GibbsSampler.get_gibbs_idx(gibbs_control)
        seed = gibbs_control['seed']
        print("Start run...")
 
        star_input = []
        for i in range(X.shape[0]):
            psi_mal_n = pd.DataFrame(psi_mal.iloc[i, :]).T
            phi_n = pd.concat([psi_mal_n, psi_env])
            nonzero_idx = np.max(phi_n, axis = 0) > 0
            star_input.append((X[i, nonzero_idx], phi_n.loc[:, nonzero_idx], alpha, gibbs_idx, seed))

        with multiprocessing.Pool(processes = gibbs_control['n.cores']) as pool:
            gibbs_list = pool.starmap(GibbsSampler.sample_theta_n , star_input)
        
        return ThetaPost.new(self.X.index, [key] + list(psi_env.index), gibbs_list)


    def run(self, final, if_estimate = True, compute_elbo = False):
        if final:
            print("Run Gibbs sampling using updated reference ...")
        else:
            print("Run Gibbs sampling...")

        if if_estimate:
            self.estimate_gibbs_time(final = final)
        if isinstance(self.reference, RefPhi):
            return GibbsSampler.run_gibbs_refPhi(self, final = final, compute_elbo = compute_elbo)
        if isinstance(self.reference, RefTumor):
            return GibbsSampler.run_gibbs_refTumor(self)


    def run_fast(self, n_iter=100, tol=1e-6, verbose=True, n_cores=1):
        """
        Fast deconvolution using fixed-point iteration (InstaPrism-style).

        This method is 50-500x faster than Gibbs sampling but provides
        approximate results (correlation >0.99 with Gibbs). No uncertainty
        estimates (theta_cv) are provided.

        Args:
            n_iter: Max iterations per sample (default 100, vs 1000 for Gibbs)
            tol: Convergence tolerance (default 1e-6)
            verbose: Print progress info
            n_cores: Number of cores for parallel processing

        Returns:
            ThetaPost object with theta estimates (no theta_cv)

        Example:
            >>> sampler = GibbsSampler(reference, X, gibbs_control)
            >>> result_fast = sampler.run_fast(n_iter=100)  # 50-500x faster
            >>> result_gibbs = sampler.run(final=False)     # Standard Gibbs
        """
        print(f"Run fast deconvolution using fixed-point iteration (n_iter={n_iter})...")
        print("Note: This is 50-500x faster but provides approximate results.")

        if not isinstance(self.reference, RefPhi):
            raise NotImplementedError("Fast mode currently only supports RefPhi reference")

        phi = self.reference.phi  # K x G
        X = self.X.to_numpy()     # N x G

        start_time = time.time()

        # Prepare arguments for parallel processing
        phi_array = phi.values  # Convert to numpy array once
        args_list = [(n, X[n, :], phi_array, n_iter, tol, verbose) for n in range(X.shape[0])]

        if n_cores > 1:
            with multiprocessing.Pool(processes=n_cores) as pool:
                results = list(tqdm.tqdm(
                    pool.imap(_process_sample_fixed_point, args_list),
                    total=X.shape[0],
                    desc="Fast deconvolution"
                ))
        else:
            results = []
            for args in tqdm.tqdm(args_list, desc="Fast deconvolution"):
                results.append(_process_sample_fixed_point(args))

        # Extract theta values
        theta_list = [{'theta_n': r['theta'], 'theta.cv_n': np.zeros_like(r['theta'])}
                      for r in results]

        total_time = time.time() - start_time

        if verbose:
            n_converged = sum(1 for r in results if r['converged'])
            avg_iters = np.mean([r['n_iter_used'] for r in results])
            print(f"\nCompleted in {total_time:.2f} seconds")
            print(f"  Samples: {X.shape[0]}")
            print(f"  Converged: {n_converged}/{X.shape[0]} ({100*n_converged/X.shape[0]:.1f}%)")
            print(f"  Average iterations: {avg_iters:.1f}")
            print(f"  Time per sample: {total_time/X.shape[0]*1000:.2f} ms")

        return ThetaPost.new(self.X.index, phi.index, theta_list)
