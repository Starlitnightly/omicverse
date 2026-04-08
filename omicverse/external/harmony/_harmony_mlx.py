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
        self._rng = np.random.default_rng(random_state)
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
        """R package formula, pure MLX operations."""
        scale_dist = mx.exp(-self._dist_mat / mx.expand_dims(self._sigma, 1))
        scale_dist = scale_dist / mx.sum(scale_dist, axis=0, keepdims=True)

        update_order = mx.array(self._rng.permutation(self.N))
        n_blocks = int(np.ceil(1.0 / self.block_size))
        cells_per_block = int(self.N * self.block_size)

        R_perm = self._R[:, update_order]
        scale_perm = scale_dist[:, update_order]
        Phi_perm = self._Phi[:, update_order]

        for blk in range(n_blocks):
            idx_min = blk * cells_per_block
            idx_max = self.N if blk == n_blocks - 1 else (blk + 1) * cells_per_block

            R_block = R_perm[:, idx_min:idx_max]
            scale_block = scale_perm[:, idx_min:idx_max]
            Phi_block = Phi_perm[:, idx_min:idx_max]

            # Remove cells from statistics
            self._E = self._E - mx.outer(mx.sum(R_block, axis=1), self._Pr_b)
            self._O = self._O - mx.matmul(R_block, Phi_block.T)

            # R package formula: ratio = E / (O + E)
            O_E = mx.clip(self._O + self._E, a_min=1e-8, a_max=None)
            ratio = mx.clip(self._E / O_E, a_min=1e-8, a_max=1.0)
            # Broadcast power: ratio^theta
            ratio_pow = mx.power(ratio, mx.expand_dims(self._theta, 0))
            R_new = scale_block * mx.matmul(ratio_pow, Phi_block)
            R_new = R_new / mx.clip(mx.sum(R_new, axis=0, keepdims=True), a_min=1e-8, a_max=None)

            # Put cells back
            self._E = self._E + mx.outer(mx.sum(R_new, axis=1), self._Pr_b)
            self._O = self._O + mx.matmul(R_new, Phi_block.T)
            R_perm[:, idx_min:idx_max] = R_new

        # Restore original order
        inv_order = mx.argsort(update_order)
        self._R = R_perm[:, inv_order]

    def _moe_correct_ridge(self):
        """Ridge regression correction, pure MLX operations."""
        self._Z_corr = mx.array(np.array(self._Z_orig))  # clone

        for k in range(self.K):
            # Dynamic lambda estimation (matches CPU/Torch backends)
            if self.lambda_estimation:
                lamb_vec = np.zeros(self.B + 1, dtype=np.float32)
                lamb_vec[1:] = np.array(self._E[k, :]) * self.alpha
            else:
                lamb_vec = np.insert(self._lamb_np, 0, 0).astype(np.float32)
            lamb_diag = mx.array(np.diag(lamb_vec))

            Phi_Rk = self._Phi_moe * self._R[k, :]
            cov = mx.matmul(Phi_Rk, self._Phi_moe.T) + lamb_diag
            try:
                inv_cov = mx.linalg.inv(cov)
            except (ValueError, RuntimeError):
                inv_cov = mx.array(np.linalg.pinv(np.array(cov)).astype(np.float32))
            W = mx.matmul(mx.matmul(inv_cov, Phi_Rk), self._Z_orig.T)
            # Zero out intercept row
            W = mx.concatenate([mx.zeros((1, W.shape[1])), W[1:, :]], axis=0)
            self._Z_corr = self._Z_corr - mx.matmul(W.T, Phi_Rk)

        self._Z_cos = self._Z_corr / mx.linalg.norm(self._Z_corr, axis=0, keepdims=True)

    def _check_convergence(self, i_type):
        if i_type == 0:
            if len(self.objective_kmeans) <= self.window_size + 1:
                return False
            w = self.window_size
            obj_old = sum(self.objective_kmeans[-w - 1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            if abs(obj_old) < 1e-10:
                return True
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans
        if i_type == 1:
            if len(self.objective_harmony) < 2:
                return False
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            if abs(obj_old) < 1e-10:
                return True
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony
        return True
