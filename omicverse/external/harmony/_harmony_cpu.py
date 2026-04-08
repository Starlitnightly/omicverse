"""Pure NumPy CPU backend for Harmony.

Avoids PyTorch dispatch overhead on CPU — faster for small-to-medium
datasets. Uses the R package formulas (same as upstream harmonypy v0.2.0).
"""

import numpy as np
import logging

from sklearn.cluster import KMeans

logger = logging.getLogger("harmonypy")


def _safe_entropy(x):
    y = x * np.log(x)
    y[~np.isfinite(y)] = 0.0
    return y


def _pow_by_col(A, T):
    """Element-wise power with per-column exponents."""
    result = np.empty_like(A)
    for c in range(A.shape[1]):
        result[:, c] = np.power(A[:, c], T[c])
    return result


class HarmonyCPU:
    """Harmony batch correction — pure NumPy, R-package-aligned formulas."""

    def __init__(
        self, Z, Phi, Pr_b, sigma, theta, lamb, alpha, lambda_estimation,
        max_iter_harmony, max_iter_kmeans,
        epsilon_kmeans, epsilon_harmony, K, block_size, verbose,
        random_state,
    ):
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
        self._lamb_vec = lamb.copy()

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        # Arrays
        Z = Z.astype(np.float32)
        self._Z_orig = Z.copy()
        self._Z_corr = Z.copy()
        self._Z_cos = Z / np.linalg.norm(Z, ord=2, axis=0, keepdims=True)
        self._Phi = Phi.astype(np.float32)
        self._Pr_b = Pr_b.astype(np.float32)
        self._sigma = sigma.astype(np.float32)
        self._theta = theta.astype(np.float32)

        # Phi_moe with intercept
        self._Phi_moe = np.vstack([np.ones((1, self.N), dtype=np.float32), self._Phi])

        # Batch index for fast ridge
        self._batch_index = []
        for b in range(self.B):
            self._batch_index.append(np.where(self._Phi[b, :] > 0)[0])

        # Buffers
        self._R = np.zeros((K, self.N), dtype=np.float32)
        self._Y = np.zeros((self.d, K), dtype=np.float32)
        self._E = np.zeros((K, self.B), dtype=np.float32)
        self._O = np.zeros((K, self.B), dtype=np.float32)

        # Init cluster
        logger.info("Computing initial centroids with sklearn.KMeans...")
        model = KMeans(n_clusters=K, init="k-means++", n_init=1,
                       max_iter=25, random_state=random_state)
        model.fit(self._Z_cos.T)
        self._Y = model.cluster_centers_.T.astype(np.float32)
        logger.info("KMeans initialization complete.")

        self._Y /= np.linalg.norm(self._Y, ord=2, axis=0, keepdims=True)
        self._dist_mat = 2.0 * (1.0 - self._Y.T @ self._Z_cos)
        self._R = np.exp(-self._dist_mat / self._sigma[:, None])
        self._R /= self._R.sum(axis=0, keepdims=True)
        self._E = np.outer(self._R.sum(axis=1), self._Pr_b)
        self._O = self._R @ self._Phi.T

        self._compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])
        self._harmonize(max_iter_harmony, verbose)

    # ── Properties ──────────────────────────────────────────────
    @property
    def Z_corr(self):
        return self._Z_corr.T

    @property
    def Z_orig(self):
        return self._Z_orig.T

    @property
    def Z_cos(self):
        return self._Z_cos.T

    @property
    def R(self):
        return self._R.T

    @property
    def Y(self):
        return self._Y

    @property
    def O(self):
        return self._O

    @property
    def E(self):
        return self._E

    def result(self):
        return self._Z_corr.T

    # ── Core algorithm ──────────────────────────────────────────

    def _compute_objective(self):
        norm_const = 2000.0 / self.N
        kmeans_error = np.sum(self._R * self._dist_mat)
        _entropy = np.sum(_safe_entropy(self._R) * self._sigma[:, None])
        # R package cross-entropy formula
        O_E_sum = np.clip(self._O + self._E, 1e-8, None)
        E_clipped = np.clip(self._E, 1e-8, None)
        ratio = O_E_sum / E_clipped
        theta_log = self._theta[None, :] * np.log(ratio)
        R_sigma = self._R * self._sigma[:, None]
        _cross_entropy = np.sum(R_sigma * (theta_log @ self._Phi))
        self.objective_kmeans.append((kmeans_error + _entropy + _cross_entropy) * norm_const)
        self.objective_kmeans_dist.append(kmeans_error * norm_const)
        self.objective_kmeans_entropy.append(_entropy * norm_const)
        self.objective_kmeans_cross.append(_cross_entropy * norm_const)

    def _harmonize(self, iter_harmony, verbose):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                logger.info(f"Iteration {i} of {iter_harmony}")
            self._cluster()
            self._moe_correct_ridge()
            converged = self._check_convergence(1)
            if converged:
                if verbose:
                    logger.info(f"Converged after {i} iteration{'s' if i > 1 else ''}")
                break
        if verbose and not converged:
            logger.info("Stopped before convergence")

    def _cluster(self):
        self._dist_mat = 2.0 * (1.0 - self._Y.T @ self._Z_cos)
        rounds = 0
        for i in range(self.max_iter_kmeans):
            self._Y = self._Z_cos @ self._R.T
            self._Y /= np.linalg.norm(self._Y, ord=2, axis=0, keepdims=True)
            self._dist_mat = 2.0 * (1.0 - self._Y.T @ self._Z_cos)
            self._update_R()
            self._compute_objective()
            if i > self.window_size and self._check_convergence(0):
                rounds = i + 1
                break
            rounds = i + 1
        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def _update_R(self):
        """R package formula: ratio = E / (O + E), raised to theta."""
        scale_dist = np.exp(-self._dist_mat / self._sigma[:, None])
        scale_dist /= scale_dist.sum(axis=0, keepdims=True)

        update_order = np.random.permutation(self.N)
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

            self._E -= np.outer(R_block.sum(axis=1), self._Pr_b)
            self._O -= R_block @ Phi_block.T

            # R package formula
            O_E = np.clip(self._O + self._E, 1e-8, None)
            ratio = np.clip(self._E / O_E, 1e-8, 1.0)
            ratio_pow = _pow_by_col(ratio, self._theta)
            R_new = scale_block * (ratio_pow @ Phi_block)
            R_new /= np.clip(R_new.sum(axis=0, keepdims=True), 1e-8, None)

            self._E += np.outer(R_new.sum(axis=1), self._Pr_b)
            self._O += R_new @ Phi_block.T
            R_perm[:, idx_min:idx_max] = R_new

        inv_order = np.argsort(update_order)
        self._R = R_perm[:, inv_order]

    def _moe_correct_ridge(self):
        self._Z_corr = self._Z_orig.copy()
        for k in range(self.K):
            if self.lambda_estimation:
                lamb_vec = np.zeros(self.B + 1, dtype=np.float32)
                lamb_vec[1:] = self._E[k, :] * self.alpha
            else:
                lamb_vec = self._lamb_vec

            Phi_Rk = self._Phi_moe * self._R[k, :]
            cov = Phi_Rk @ self._Phi_moe.T + np.diag(lamb_vec)
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)

            Z_tmp = self._Z_orig * self._R[k, :]
            W = inv_cov[:, 0:1] @ Z_tmp.sum(axis=1, keepdims=True).T
            for b in range(self.B):
                batch_sum = Z_tmp[:, self._batch_index[b]].sum(axis=1, keepdims=True)
                W += inv_cov[:, b + 1:b + 2] @ batch_sum.T
            W[0, :] = 0
            self._Z_corr -= W.T @ Phi_Rk

        self._Z_cos = self._Z_corr / np.linalg.norm(
            self._Z_corr, ord=2, axis=0, keepdims=True
        )

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
