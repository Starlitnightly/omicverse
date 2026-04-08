"""MLX backend for Harmony on Apple Silicon (MPS).

This module provides a native MLX implementation that avoids PyTorch MPS
beta-stage instability. It is used automatically when ``device='mps'``
and MLX is installed.
"""

import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger("harmonypy")

try:
    from ..._settings import EMOJI, Colors
except ImportError:
    EMOJI = {'start': '🚀', 'done': '✅', 'warning': '⚠️', 'cpu': '🖥️', 'gpu': '🚀'}
    Colors = type('Colors', (), {
        'CYAN': '\033[96m', 'BOLD': '\033[1m', 'ENDC': '\033[0m',
        'GREEN': '\033[92m', 'WARNING': '\033[93m',
    })()

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def safe_entropy_mlx(x):
    y = x * mx.log(x)
    return mx.where(mx.isfinite(y), y, mx.zeros_like(y))


class HarmonyMLX:
    """Harmony batch correction using MLX (Apple Silicon GPU)."""

    def __init__(
        self, Z, Phi, Pr_b, sigma, theta, lamb, alpha, lambda_estimation,
        max_iter_harmony, max_iter_kmeans,
        epsilon_kmeans, epsilon_harmony, K, block_size, verbose,
        random_state,
    ):
        from sklearn.cluster import KMeans

        self.N = Z.shape[1]
        self.B = Phi.shape[0]
        self.d = Z.shape[0]
        self.K = K
        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony
        self.block_size = block_size
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose = verbose
        self.alpha = alpha
        self.lambda_estimation = lambda_estimation

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        # Convert to MLX arrays
        self._Z_orig = mx.array(Z.astype(np.float32))
        self._Z_corr = mx.array(Z.astype(np.float32))
        self._Z_cos = self._Z_orig / mx.linalg.norm(self._Z_orig, axis=0, keepdims=True)
        self._Phi = mx.array(Phi.astype(np.float32))
        self._Pr_b = mx.array(Pr_b.astype(np.float32))
        self._sigma = mx.array(sigma.astype(np.float32))
        self._theta = mx.array(theta.astype(np.float32))

        # Lambda
        self._lamb_np = lamb
        self._lamb = mx.array(np.diag(np.insert(lamb, 0, 0)).astype(np.float32)) \
            if not lambda_estimation else mx.zeros((1,), dtype=mx.float32)

        # Phi_moe with intercept
        ones = mx.ones((1, self.N), dtype=mx.float32)
        self._Phi_moe = mx.concatenate([ones, self._Phi], axis=0)

        # Allocate buffers
        self._R = mx.zeros((K, self.N), dtype=mx.float32)
        self._Y = mx.zeros((self.d, K), dtype=mx.float32)
        self._E = mx.zeros((K, self.B), dtype=mx.float32)
        self._O = mx.zeros((K, self.B), dtype=mx.float32)

        # Init cluster
        if verbose:
            print(f"{Colors.CYAN}    Initializing centroids (K={K}) ...{Colors.ENDC}", end=" ", flush=True)
        Z_cos_np = np.array(self._Z_cos)
        model = KMeans(n_clusters=K, init="k-means++", n_init=1,
                       max_iter=25, random_state=random_state)
        model.fit(Z_cos_np.T)
        self._Y = mx.array(model.cluster_centers_.T.astype(np.float32))
        self._Y = self._Y / mx.linalg.norm(self._Y, axis=0, keepdims=True)
        if verbose:
            print("done")

        self._dist_mat = 2 * (1 - mx.matmul(self._Y.T, self._Z_cos))
        self._R = -self._dist_mat / mx.expand_dims(self._sigma, 1)
        self._R = mx.exp(self._R)
        self._R = self._R / mx.sum(self._R, axis=0, keepdims=True)

        self._E = mx.outer(mx.sum(self._R, axis=1), self._Pr_b)
        self._O = mx.matmul(self._R, self._Phi.T)

        self._compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

        # Run
        self._harmonize(max_iter_harmony, verbose)

    # Properties returning numpy
    @property
    def Z_corr(self):
        return np.array(self._Z_corr).T

    @property
    def Z_orig(self):
        return np.array(self._Z_orig).T

    @property
    def R(self):
        return np.array(self._R).T

    @property
    def Y(self):
        return np.array(self._Y)

    @property
    def O(self):
        return np.array(self._O)

    @property
    def E(self):
        return np.array(self._E)

    def result(self):
        return np.array(self._Z_corr).T

    def _compute_objective(self):
        kmeans_error = float(mx.sum(self._R * self._dist_mat))
        _entropy = float(mx.sum(safe_entropy_mlx(self._R) * mx.expand_dims(self._sigma, 1)))
        x = self._R * mx.expand_dims(self._sigma, 1)
        y = mx.broadcast_to(mx.expand_dims(self._theta, 0), (self.K, self.B))
        z = mx.log((self._O + 1) / (self._E + 1))
        _cross_entropy = float(mx.sum(x * mx.matmul(y * z, self._Phi)))
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def _harmonize(self, iter_harmony, verbose):
        converged = False
        if verbose:
            print(f"{EMOJI['start']} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running Harmony integration...")

        pbar = tqdm(range(1, iter_harmony + 1), desc="Harmony iterations", disable=not verbose)
        for i in pbar:
            self._cluster()
            self._moe_correct_ridge()
            converged = self._check_convergence(1)
            if verbose:
                pbar.set_description(f"Harmony iteration {i}/{iter_harmony}")
            if converged:
                if verbose:
                    pbar.set_description(f"Harmony converged after {i} iterations")
                    print(f"\n{EMOJI['done']} Harmony converged after {i} iteration{'s' if i > 1 else ''}")
                break
        pbar.close()

        if verbose and not converged:
            print(f"{EMOJI['warning']} Harmony stopped before convergence after {iter_harmony} iterations")

    def _cluster(self):
        self._dist_mat = 2 * (1 - mx.matmul(self._Y.T, self._Z_cos))
        rounds = 0
        for i in range(self.max_iter_kmeans):
            self._Y = mx.matmul(self._Z_cos, self._R.T)
            self._Y = self._Y / mx.linalg.norm(self._Y, axis=0, keepdims=True)
            self._dist_mat = 2 * (1 - mx.matmul(self._Y.T, self._Z_cos))
            self._update_R()
            self._compute_objective()
            if i > self.window_size:
                if self._check_convergence(0):
                    rounds = i + 1
                    break
            rounds = i + 1
        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def _update_R(self):
        scale_dist = -self._dist_mat / mx.expand_dims(self._sigma, 1)
        scale_dist = mx.exp(scale_dist)
        scale_dist = scale_dist / mx.sum(scale_dist, axis=0, keepdims=True)

        update_order = np.random.permutation(self.N)
        n_blocks = int(np.ceil(1.0 / self.block_size))
        cells_per_block = int(self.N * self.block_size)

        R_np = np.array(self._R)
        scale_np = np.array(scale_dist)
        Phi_np = np.array(self._Phi)
        E_np = np.array(self._E)
        O_np = np.array(self._O)
        theta_np = np.array(self._theta)
        Pr_b_np = np.array(self._Pr_b)

        for blk in range(n_blocks):
            idx_min = blk * cells_per_block
            idx_max = self.N if blk == n_blocks - 1 else (blk + 1) * cells_per_block
            b = update_order[idx_min:idx_max]

            E_np -= np.outer(R_np[:, b].sum(1), Pr_b_np)
            O_np -= R_np[:, b] @ Phi_np[:, b].T

            ratio = np.clip(E_np / np.clip(O_np + E_np, 1e-8, None), 1e-8, 1.0)
            ratio_pow = np.empty_like(ratio)
            for c in range(ratio.shape[1]):
                ratio_pow[:, c] = np.power(ratio[:, c], theta_np[c])
            R_new = scale_np[:, b] * (ratio_pow @ Phi_np[:, b])
            R_new = R_new / np.clip(R_new.sum(0), 1e-8, None)

            E_np += np.outer(R_new.sum(1), Pr_b_np)
            O_np += R_new @ Phi_np[:, b].T
            R_np[:, b] = R_new

        self._R = mx.array(R_np.astype(np.float32))
        self._E = mx.array(E_np.astype(np.float32))
        self._O = mx.array(O_np.astype(np.float32))

    def _moe_correct_ridge(self):
        Z_orig = np.array(self._Z_orig)
        Z_corr = Z_orig.copy()
        Phi_moe = np.array(self._Phi_moe)
        R = np.array(self._R)
        lamb = np.diag(np.insert(self._lamb_np, 0, 0)).astype(np.float32)

        for k in range(self.K):
            Phi_Rk = Phi_moe * R[k, :]
            x = Phi_Rk @ Phi_moe.T + lamb
            try:
                x_inv = np.linalg.inv(x)
            except np.linalg.LinAlgError:
                x_inv = np.linalg.pinv(x)
            W = x_inv @ Phi_Rk @ Z_orig.T
            W[0, :] = 0
            Z_corr -= W.T @ Phi_Rk

        self._Z_corr = mx.array(Z_corr.astype(np.float32))
        self._Z_cos = self._Z_corr / mx.linalg.norm(self._Z_corr, axis=0, keepdims=True)

    def _check_convergence(self, i_type):
        if i_type == 0:
            if len(self.objective_kmeans) <= self.window_size + 1:
                return False
            w = self.window_size
            obj_old = sum(self.objective_kmeans[-w - 1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans
        if i_type == 1:
            if len(self.objective_harmony) < 2:
                return False
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony
        return True
