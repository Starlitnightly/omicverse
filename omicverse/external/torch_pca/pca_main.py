"""Main module for PCA."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
# Inspired from https://github.com/scikit-learn (BSD-3-Clause License)
# Copyright (c) Scikit-learn developers. All Rights Reserved.
from math import log
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType
from scipy import sparse as sp
import warnings

from .ncompo import NComponentsType, find_ncomponents
from .svd import choose_svd_solver, randomized_svd, svd_flip

HIGH_DENSITY_AUTO_DENSE_THRESHOLD = 0.2
MAX_AUTO_DENSE_ELEMENTS = 250_000_000
AUTO_DENSE_COV_EIGH_MAX_FEATURES = 4096
AUTO_DENSE_CPU_MAX_BYTES = 2_500_000_000
AUTO_DENSE_CUDA_FREE_MEM_FRACTION = 0.4


class PCA:
    """Principal Component Analysis (PCA).

    Works with PyTorch tensors.
    API similar to sklearn.decomposition.PCA.

    Parameters
    ----------
    n_components: int | float | str | None, optional
        Number of components to keep.

        * If int, number of components to keep.
        * If float (should be between 0.0 and 1.0), the number of components
          to keep is determined by the cumulative percentage of variance
          explained by the components until the proportion is reached.
        * If "mle", the number of components is selected using Minka's MLE.
        * If None, all components are kept: n_components = min(n_samples, n_features).

        By default, n_components=None.

    svd_solver: str, optional
        One of {'auto', 'full', 'covariance_eigh', 'randomized', 'lobpcg', 'arpack'}

        * 'auto': the solver is selected automatically based on the shape and type of input.
          For sparse matrices, 'lobpcg' is selected. For dense tensors, selection follows
          the original heuristics.
        * 'full': Run exact full SVD with torch.linalg.svd (dense tensors only)
        * 'covariance_eigh': Compute the covariance matrix and take
          the eigenvalues decomposition with torch.linalg.eigh (dense tensors only).
          Most efficient for small n_features and large n_samples.
        * 'randomized': Compute the randomized SVD by the method of Halko et al (dense tensors only).
        * 'lobpcg': Compute top eigenpairs of covariance with torch.lobpcg.
          Works for dense and sparse tensors in pure PyTorch.
        * 'arpack': Deprecated compatibility alias; internally redirected to 'lobpcg'.

        By default, svd_solver='auto'.

    whiten : bool, optional
        If True, the components_ vectors are divided by sqrt(n_samples - 1)
        and scaled by the singular values to ensure uncorrelated outputs
        with unit component-wise variances.
        By default, False.

    iterated_power: int | str, optional
        Integer or 'auto'. Number of iterations for the power method
        computed by randomized SVD. Must be >= 0.
        Ignored if svd_solver!='randomized'. By default, 'auto'.
    n_oversamples : int, optional
        Additional number of random vectors to sample the
        range of input data in randomized solver to ensure proper
        conditioning.
        Ignored if svd_solver!='randomized'. By default, 10.
    power_iteration_normalizer : str, optional
        One of 'auto', 'QR', 'LU', 'none'.
        Power iteration normalizer for randomized SVD solver.
        Ignored if svd_solver!='randomized'. By default, 'auto.
    random_state : int | None, optional
        Seed of randomized SVD solver.
        Ignored if svd_solver!='randomized'. By default, None.
    """

    def __init__(
        self,
        n_components: NComponentsType = None,
        *,
        whiten: bool = False,
        svd_solver: str = "covariance_eigh",
        iterated_power: Union[str, int] = "auto",
        n_oversamples: int = 10,
        power_iteration_normalizer: str = "auto",
        random_state: Optional[int] = None,
    ):
        #: Principal axes in feature space.
        self.components_: Optional[Tensor] = None
        #: The amount of variance explained by each of the selected components.
        self.explained_variance_: Optional[Tensor] = None
        #: Percentage of variance explained by each of the selected components.
        self.explained_variance_ratio_: Optional[Tensor] = None
        #: Mean of the input data during fit.
        self.mean_: Optional[Tensor] = None
        #: Number of components to keep.
        self.n_components_: NComponentsType = n_components
        #: Number of features in the input data.
        self.n_features_in_: int = -1
        #: Number of samples seen during fit.
        self.n_samples_: int = -1
        #: The estimated noise covariance.
        self.noise_variance_: Optional[Tensor] = None
        #: Singular values corresponding to each of the selected components.
        self.singular_values_: Optional[Tensor] = None
        #: Whether the data is whitened or not.
        self.whiten: bool = whiten
        #: Solver to use for the PCA computation.
        self.svd_solver_: str = svd_solver
        # Randomized SVD parameters
        self.n_oversamples = n_oversamples
        self.iterated_power = iterated_power
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        # Internal tracking for sparse matrix support
        self._input_is_sparse: bool = False
        self._original_device: Optional[torch.device] = None
        self._original_dtype: Optional[torch.dtype] = None
        self._auto_densified_from_sparse: bool = False

        if self.svd_solver_ not in ["auto", "full", "covariance_eigh", "randomized", "lobpcg", "arpack"]:
            raise ValueError(
                "Unknown SVD solver. `svd_solver` should be one of "
                "'auto', 'full', 'covariance_eigh', 'randomized', 'lobpcg', 'arpack'."
            )

    def fit_transform(self, inputs: Union[Tensor, sp.csr_matrix, sp.csc_matrix], *, determinist: bool = True) -> Tensor:
        """Fit the PCA model and apply the dimensionality reduction.

        Parameters
        ----------
        inputs : Tensor or scipy.sparse matrix
            Input data of shape (n_samples, n_features).
            Can be a PyTorch tensor (dense) or scipy sparse matrix (csr_matrix, csc_matrix).
        determinist : bool, optional
            If True, the SVD solver is deterministic but the gradient
            cannot be computed through the PCA fit (the PCA transform is
            always differentiable though).
            If False, the SVD can be non-deterministic but the
            gradient can be computed through the PCA fit.
            By default, determinist=True.


        Returns
        -------
        transformed : Tensor
            Transformed data.
        """
        # Convert scipy sparse input once and reuse it for both fit and transform.
        if sp.issparse(inputs):
            from .sparse_utils import scipy_sparse_to_torch_sparse

            inputs = scipy_sparse_to_torch_sparse(
                inputs, device=torch.device("cpu"), dtype=torch.float32
            ).coalesce()

        self.fit(inputs, determinist=determinist)
        transformed = self.transform(inputs)
        return transformed

    def _compute_sparse_gram_matrix(self, inputs: Tensor) -> Tensor:
        """Compute X^T X for sparse inputs with robust fallback paths."""
        x_t = inputs.transpose(0, 1)
        try:
            # Fast path: sparse@sparse -> sparse (then densify, small n_features x n_features)
            return torch.sparse.mm(x_t, inputs).to_dense()
        except RuntimeError as err:
            err_msg = str(err).lower()

            # Older/newer torch builds may require dense mat2 for sparse.mm.
            if "dense" in err_msg or "strided" in err_msg:
                return torch.sparse.mm(x_t, inputs.to_dense())

            # CUDA sparse SpGEMM can fail with insufficient resources on large nnz.
            if any(tok in err_msg for tok in ("cusparse", "spgemm", "insufficient resources")):
                warnings.warn(
                    "Sparse GPU SpGEMM failed while building PCA Gram matrix; "
                    "falling back to CPU sparse multiplication in torch.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                x_t_cpu = x_t.cpu()
                inputs_cpu = inputs.cpu()
                try:
                    gram_cpu = torch.sparse.mm(x_t_cpu, inputs_cpu).to_dense()
                except RuntimeError as cpu_err:
                    cpu_err_msg = str(cpu_err).lower()
                    if "dense" not in cpu_err_msg and "strided" not in cpu_err_msg:
                        raise
                    gram_cpu = torch.sparse.mm(x_t_cpu, inputs_cpu.to_dense())
                return gram_cpu.to(device=inputs.device, dtype=inputs.dtype)

            raise

    def _maybe_auto_densify_sparse(self, inputs: Tensor) -> Tensor:
        """Auto-convert high-density sparse tensors to dense for better performance."""
        n_samples, n_features = inputs.shape
        total = int(n_samples) * int(n_features)
        if total <= 0 or total > MAX_AUTO_DENSE_ELEMENTS:
            return inputs

        # inputs is coalesced sparse COO in our pipeline
        nnz = int(inputs._nnz())
        density = nnz / total
        if density < HIGH_DENSITY_AUTO_DENSE_THRESHOLD:
            return inputs

        element_size = torch.empty((), dtype=inputs.dtype).element_size()
        dense_bytes = total * element_size
        if inputs.is_cuda:
            try:
                free_mem, _ = torch.cuda.mem_get_info(inputs.device)
            except Exception:
                return inputs
            if dense_bytes > int(free_mem * AUTO_DENSE_CUDA_FREE_MEM_FRACTION):
                return inputs
        elif dense_bytes > AUTO_DENSE_CPU_MAX_BYTES:
            return inputs

        warnings.warn(
            "High-density sparse input detected "
            f"(density={density * 100:.2f}%, shape={tuple(inputs.shape)}); "
            "converting to dense tensor for faster torch PCA path.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._input_is_sparse = False
        self._auto_densified_from_sparse = True
        return inputs.to_dense()

    def fit(self, inputs: Union[Tensor, sp.csr_matrix, sp.csc_matrix], *, determinist: bool = True) -> "PCA":
        """Fit the PCA model and return it.

        Parameters
        ----------
        inputs : Tensor or scipy.sparse matrix
            Input data of shape (n_samples, n_features).
            Can be a PyTorch tensor (dense) or scipy sparse matrix (csr_matrix, csc_matrix).
        determinist : bool, optional
            If True, the SVD solver is deterministic but the gradient
            cannot be computed through the PCA fit (the PCA transform is
            always differentiable though).
            If False, the SVD can be non-deterministic but the
            gradient can be computed through the PCA fit.
            By default, determinist=True.

        Returns
        -------
        PCA
            The PCA model fitted on the input data.
        """
        # Detect and validate input type, convert to torch tensor
        self._auto_densified_from_sparse = False
        if sp.issparse(inputs):
            self._input_is_sparse = True
            if not isinstance(inputs, (sp.csr_matrix, sp.csc_matrix)):
                raise ValueError(
                    f"Sparse input must be csr_matrix or csc_matrix, got {type(inputs)}"
                )
            from .sparse_utils import scipy_sparse_to_torch_sparse
            inputs = scipy_sparse_to_torch_sparse(
                inputs, device=torch.device("cpu"), dtype=torch.float32
            ).coalesce()
        elif isinstance(inputs, Tensor) and inputs.is_sparse:
            self._input_is_sparse = True
            inputs = inputs.coalesce()
            if inputs.dtype == torch.float16:
                inputs = inputs.to(torch.float32)
        else:
            self._input_is_sparse = False
            if self.svd_solver_ == "arpack":
                raise ValueError(
                    "ARPACK compatibility alias only works with sparse input. "
                    "Please use 'auto', 'full', 'covariance_eigh', 'randomized', or 'lobpcg'."
                )
            # Convert to tensor if needed
            if not isinstance(inputs, Tensor):
                inputs = torch.as_tensor(inputs)
            # Auto-cast to float32 because float16 is not supported
            if inputs.dtype == torch.float16:
                inputs = inputs.to(torch.float32)

        if self._input_is_sparse and isinstance(inputs, Tensor):
            inputs = self._maybe_auto_densify_sparse(inputs)

        if self.svd_solver_ == "auto":
            # For sparse inputs auto-densified due to high density, prefer
            # covariance_eigh when feature dimensionality is moderate.
            if (
                self._auto_densified_from_sparse
                and inputs.shape[-1] <= AUTO_DENSE_COV_EIGH_MAX_FEATURES
                and inputs.shape[-2] >= inputs.shape[-1]
            ):
                self.svd_solver_ = "covariance_eigh"
            else:
                self.svd_solver_ = choose_svd_solver(
                    inputs=inputs,
                    n_components=self.n_components_,
                    is_sparse=self._input_is_sparse,
                )
        elif self.svd_solver_ == "arpack":
            warnings.warn(
                "svd_solver='arpack' is deprecated in torch_pca sparse mode and "
                "has been replaced by pure PyTorch 'lobpcg'.",
                FutureWarning,
                stacklevel=2,
            )
            self.svd_solver_ = "lobpcg"

        if self._input_is_sparse and self.svd_solver_ != "lobpcg":
            raise ValueError(
                f"Sparse inputs only support 'auto', 'lobpcg', or 'arpack' compatibility alias. "
                f"Got '{self.svd_solver_}'."
            )

        # Compute mean and shape based on input type
        if self._input_is_sparse:
            # Sparse-friendly column mean: sum sparse columns then divide by n_samples.
            self.n_samples_, self.n_features_in_ = inputs.shape
            col_sums = torch.sparse.sum(inputs, dim=0).to_dense()
            self.mean_ = (col_sums / self.n_samples_).reshape(1, -1)
        else:
            self.mean_ = inputs.mean(dim=-2, keepdim=True)
            self.n_samples_, self.n_features_in_ = inputs.shape[-2:]

        if self.svd_solver_ == "lobpcg":
            max_lobpcg_components = min(self.n_samples_, self.n_features_in_) - 1
            if max_lobpcg_components < 1:
                raise ValueError(
                    "lobpcg requires at least 2 samples/features after centering."
                )
            if self.n_components_ is None:
                self.n_components_ = max_lobpcg_components
            elif isinstance(self.n_components_, str):
                raise ValueError(
                    f"LOBPCG solver does not support n_components='{self.n_components_}'. "
                    "Please specify an integer value."
                )
            elif isinstance(self.n_components_, float):
                raise ValueError(
                    "LOBPCG solver does not support float n_components (variance threshold). "
                    "Please specify an integer value."
                )
            elif self.n_components_ > max_lobpcg_components:
                raise ValueError(
                    f"LOBPCG requires n_components <= min(n_samples, n_features) - 1. "
                    f"Got n_components={self.n_components_}, max allowed is {max_lobpcg_components}"
                )
            self.n_components_ = int(self.n_components_)

            if self._input_is_sparse:
                gram = self._compute_sparse_gram_matrix(inputs)
                mean_vec = self.mean_.reshape(-1)
                covariance = (
                    gram - self.n_samples_ * torch.outer(mean_vec, mean_vec)
                ) / (self.n_samples_ - 1)
            else:
                inputs_centered = inputs - self.mean_
                covariance = (inputs_centered.T @ inputs_centered) / (self.n_samples_ - 1)

            covariance = 0.5 * (covariance + covariance.T)
            cov_size = covariance.shape[0]
            # CUDA LOBPCG can be unstable/slow for moderate-sized covariance matrices.
            # Prefer exact eigh in this regime while staying fully in PyTorch.
            prefer_eigh = covariance.is_cuda and cov_size <= 4096
            if prefer_eigh or self.n_components_ >= cov_size - 1:
                eigenvals, eigenvecs = torch.linalg.eigh(covariance)
                eigenvals = eigenvals[-self.n_components_:]
                eigenvecs = eigenvecs[:, -self.n_components_:]
            else:
                init_vecs = None
                if isinstance(self.random_state, int):
                    gen = torch.Generator(device=covariance.device)
                    gen.manual_seed(self.random_state)
                    init_vecs = torch.randn(
                        cov_size,
                        self.n_components_,
                        device=covariance.device,
                        dtype=covariance.dtype,
                        generator=gen,
                    )
                eigenvals, eigenvecs = torch.lobpcg(
                    covariance,
                    k=self.n_components_,
                    X=init_vecs,
                    largest=True,
                    method="ortho",
                )

            order = torch.argsort(eigenvals, descending=True)
            explained_variance = torch.clamp(eigenvals[order], min=0.0)
            components = eigenvecs[:, order].T
            coefs = torch.sqrt(explained_variance * (self.n_samples_ - 1))
            vh_mat = components
            u_mat = None
            total_var = torch.clamp(
                torch.trace(covariance),
                min=torch.finfo(covariance.dtype).eps,
            )
        elif self.svd_solver_ == "full":
            inputs_centered = inputs - self.mean_
            u_mat, coefs, vh_mat = torch.linalg.svd(  # pylint: disable=E1102
                inputs_centered,
                full_matrices=False,
            )
            explained_variance = coefs**2 / (inputs.shape[-2] - 1)
            total_var = torch.sum(explained_variance)
        elif self.svd_solver_ == "covariance_eigh":
            covariance = inputs.T @ inputs
            delta = self.n_samples_ * torch.transpose(self.mean_, -2, -1) * self.mean_
            covariance -= delta
            covariance /= self.n_samples_ - 1
            eigenvals, eigenvecs = torch.linalg.eigh(covariance)
            # Fix eventual numerical errors
            eigenvals[eigenvals < 0.0] = 0.0
            # Inverted indices
            idx = range(eigenvals.size(0) - 1, -1, -1)
            idx = torch.LongTensor(idx).to(eigenvals.device)
            explained_variance = eigenvals.index_select(0, idx)
            total_var = torch.sum(explained_variance)
            # Compute equivalent variables to full SVD output
            vh_mat = eigenvecs.T.index_select(0, idx)
            coefs = torch.sqrt(explained_variance * (self.n_samples_ - 1))
            u_mat = None
        elif self.svd_solver_ == "randomized":
            if self.n_components_ is None:
                self.n_components_ = min(inputs.shape[-2:])
            if (
                not isinstance(self.n_components_, int)
                or int(self.n_components_) != self.n_components_
            ):
                raise ValueError(
                    "Randomized SVD only supports integer number of components."
                    f"Found '{self.n_components_}'."
                )
            inputs_centered = inputs - self.mean_
            u_mat, coefs, vh_mat = randomized_svd(
                inputs=inputs_centered,
                n_components=self.n_components_,
                n_oversamples=self.n_oversamples,
                n_iter=self.iterated_power,
                power_iteration_normalizer=self.power_iteration_normalizer,
                random_state=self.random_state,
            )
            explained_variance = coefs**2 / (inputs.shape[-2] - 1)
            total_var = torch.sum(inputs_centered**2) / (self.n_samples_ - 1)

        if determinist:
            _, vh_mat = svd_flip(u_mat, vh_mat)  # pylint: disable=E0601
        explained_variance_ratio = explained_variance / total_var
        self.n_components_ = find_ncomponents(
            n_components=self.n_components_,
            inputs=inputs,
            n_samples=self.n_samples_,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
        )
        self.components_ = vh_mat[: self.n_components_]
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        self.singular_values_ = coefs[: self.n_components_]
        # Compute noise covariance using Probabilistic PCA model.
        if self.svd_solver_ == "lobpcg":
            remaining = self.n_features_in_ - self.n_components_
            if remaining > 0:
                residual = torch.clamp(
                    total_var - torch.sum(self.explained_variance_),
                    min=0.0,
                )
                self.noise_variance_ = residual / remaining
            else:
                self.noise_variance_ = torch.tensor(
                    0.0, device=explained_variance.device, dtype=explained_variance.dtype
                )
        elif self.n_components_ < explained_variance.shape[0]:
            self.noise_variance_ = torch.mean(explained_variance[self.n_components_ :])
        else:
            self.noise_variance_ = torch.tensor(
                0.0, device=explained_variance.device, dtype=explained_variance.dtype
            )
        return self

    def _check_fitted(self, method_name: str) -> None:
        """Check if the PCA model is fitted."""
        if self.components_ is None:
            raise ValueError(
                f"PCA not fitted when calling {method_name}. "
                "Please call `fit` or `fit_transform` first."
            )

    def transform(self, inputs: Union[Tensor, sp.csr_matrix, sp.csc_matrix], center: str = "fit") -> Tensor:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        inputs : Tensor or scipy.sparse matrix
            Input data of shape (n_samples, n_features).
        center : str
            One of 'fit', 'input' or 'none'.
            Precise how to center the data.

            * 'fit': center the data using the mean fitted during `fit` (default).
            * 'input': center the data using the mean of the input data.
            * 'none': do not center the data.

            By default, 'fit' (as sklearn PCA implementation)

        Returns
        -------
        transformed : Tensor
            Transformed data of shape (n_samples, n_components).
        """
        self._check_fitted("transform")
        assert self.components_ is not None  # for mypy
        assert self.mean_ is not None  # for mypy

        # Handle sparse input (scipy sparse or torch sparse)
        if sp.issparse(inputs):
            # Convert scipy sparse to torch sparse for GPU acceleration
            from .sparse_utils import scipy_sparse_to_torch_sparse

            # Convert to torch sparse on same device as components
            inputs_sparse = scipy_sparse_to_torch_sparse(
                inputs,
                device=self.components_.device,
                dtype=self.components_.dtype
            )

            # Use torch sparse matrix multiplication (GPU accelerated)
            # X @ components.T where X is sparse and components.T is dense
            transformed = torch.sparse.mm(inputs_sparse, self.components_.T)

            # Handle centering
            if center == "fit":
                # Subtract mean projection: mean @ components.T
                mean_projection = self.mean_ @ self.components_.T
                transformed = transformed - mean_projection
            elif center == "input":
                # Compute input mean and subtract its projection
                input_mean = (
                    torch.sparse.sum(inputs_sparse, dim=0).to_dense().reshape(1, -1)
                    / inputs_sparse.shape[0]
                )
                mean_projection = input_mean @ self.components_.T
                transformed = transformed - mean_projection
            elif center != "none":
                raise ValueError(
                    "Unknown centering, `center` argument should be "
                    "one of 'fit', 'input' or 'none'."
                )

        elif isinstance(inputs, Tensor) and inputs.is_sparse:
            # Already a torch sparse tensor
            # Use torch sparse matrix multiplication (GPU accelerated)
            transformed = torch.sparse.mm(inputs, self.components_.T)

            # Handle centering
            if center == "fit":
                mean_projection = self.mean_ @ self.components_.T
                transformed = transformed - mean_projection
            elif center == "input":
                input_mean = (
                    torch.sparse.sum(inputs, dim=0).to_dense().reshape(1, -1)
                    / inputs.shape[0]
                )
                mean_projection = input_mean @ self.components_.T
                transformed = transformed - mean_projection
            elif center != "none":
                raise ValueError(
                    "Unknown centering, `center` argument should be "
                    "one of 'fit', 'input' or 'none'."
                )

        else:
            # Original tensor path
            if not isinstance(inputs, Tensor):
                inputs = torch.as_tensor(inputs)

            components = (
                self.components_.to(torch.float16)
                if inputs.dtype == torch.float16
                else self.components_
            )
            mean = (
                self.mean_.to(torch.float16)
                if inputs.dtype == torch.float16
                else self.mean_
            )
            transformed = inputs @ components.T
            if center == "fit":
                transformed -= mean @ components.T
            elif center == "input":
                transformed -= inputs.mean(dim=-2, keepdim=True) @ components.T
            elif center != "none":
                raise ValueError(
                    "Unknown centering, `center` argument should be "
                    "one of 'fit', 'input' or 'none'."
                )

        if self.whiten:
            scale = torch.sqrt(self.explained_variance_)
            scale[scale < 1e-8] = 1e-8
            transformed /= scale
        return transformed

    def inverse_transform(self, inputs: Tensor) -> Tensor:
        """De-transform transformed data.

        Parameters
        ----------
        inputs : Tensor
            Transformed data of shape (n_samples, n_components).

        Returns
        -------
        de_transformed : Tensor
            De-transformed data of shape (n_samples, n_features)
            where n_features is the number of features in the input data
            before applying transform.
        """
        self._check_fitted("inverse_transform")
        assert self.components_ is not None  # for mypy
        de_transformed = inputs @ self.components_ + self.mean_
        return de_transformed

    def get_covariance(self) -> Tensor:
        """Compute data covariance with the generative model."""
        self._check_fitted("get_covariance")
        assert self.components_ is not None  # for mypy
        components, exp_variance_diff = self.get_exp_variance_diff()
        covariance = (components.T * exp_variance_diff) @ components
        covariance += self.noise_variance_ * torch.eye(components.shape[-1])
        return covariance

    def get_exp_variance_diff(self) -> Tuple[Tensor, Tensor]:
        """Get explained variance difference (from noise)."""
        assert self.noise_variance_ is not None  # for mypy
        components = self.components_
        explained_variance = self.explained_variance_
        if self.whiten:
            components = components * torch.sqrt(explained_variance)[:, None]
        exp_variance_diff = explained_variance - self.noise_variance_
        exp_variance_diff = torch.where(
            exp_variance_diff > 0,
            exp_variance_diff,
            torch.tensor(0.0),
        )
        return components, exp_variance_diff

    def get_precision(self) -> Tensor:
        """Compute data precision matrix with the generative model.

        It is the inverse the covariance matrix but the method is more
        efficient than computing it directly.
        """
        self._check_fitted("get_precision")
        assert self.noise_variance_ is not None  # for mypy
        assert self.components_ is not None  # for mypy
        n_features = self.components_.shape[-1]
        if self.n_components_ == 0:
            return torch.eye(n_features) / self.noise_variance_
        if self.noise_variance_ == 0.0:
            return torch.linalg.inv(self.get_covariance())
        components, exp_variance_diff = self.get_exp_variance_diff()
        precision = components @ components.T / self.noise_variance_
        precision += (1.0 / exp_variance_diff) * torch.eye(precision.shape[0])
        precision = components.T @ torch.linalg.inv(precision) @ components
        precision /= -(self.noise_variance_**2)
        precision += (1.0 / self.noise_variance_) * torch.eye(precision.shape[0])
        return precision

    def score_samples(self, inputs: Tensor) -> Tensor:
        """Compute score of each sample based on log-likelihood.

        Returns
        -------
        log_likelihood : Tensor
            Log-likelihood of each sample under the current model,
            of shape (n_samples,)
        """
        centered_inputs = inputs - self.mean_
        n_features = centered_inputs.shape[-1]
        precision = self.get_precision()
        log_likelihood = -0.5 * (
            n_features * log(2 * torch.pi)
            - torch.linalg.slogdet(precision)[1]
            + torch.sum((centered_inputs @ precision) * centered_inputs, dim=-1)
        )
        return log_likelihood

    def score(self, inputs: Tensor) -> Tensor:
        """Return the average score (log-likelihood) of all samples."""
        return self.score_samples(inputs).mean()

    @property
    def _n_features_out(self) -> int:
        """Number of transformed output features."""
        self._check_fitted("_n_features_out")
        assert self.components_ is not None  # for mypy
        return self.components_.shape[0]

    def to(self, *args: Any, **kwargs: Any) -> None:
        """Move the model to the specified device/dtype.

        Call the native PyTorch `.to()` method on all tensors, parameters
        and NN modules to move the model to the specified device and/or dtype.

        Parameters
        ----------
        args : Any
            Positional arguments to pass to the `.to()` method.
        kwargs : Any
            Keyword arguments to pass to the `.to()` method.
            They can be:
            device : DeviceLikeType
                Device to move the model to.
            dtype : torch.dtype
                Data type to move the model to.
            non_blocking : bool, optional
                If True, the operation will be non-blocking.
                By default, False.
            copy : bool, optional
            memory_format : torch.memory_format, optional

        Note
        ----
            By default, the parameters dtype and device are the same as
            the input data dtype and device during the fit.
            This method is used if want you to change the dtype and/or device
            of the model after the fit. For instance if you fit the model
            on GPU and want to make inference on CPU.

        Warning
        -------
            Require the model to be fitted first.
        """
        to_args = {}
        for arg in args:
            if isinstance(arg, torch.dtype):
                to_args["dtype"] = arg
            elif isinstance(arg, (str, torch.device, int)):
                to_args["device"] = arg
            else:
                raise ValueError(
                    "Unknown argument type in `args`, "
                    "should be one of `torch.dtype` or `torch.DeviceLikeType`."
                )
        to_args.update(kwargs)
        self._to(**to_args)

    def _to(
        self,
        device: Optional[DeviceLikeType] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        non_blocking: bool = False,
        **kwargs: dict,
    ) -> None:
        """Move the model to the specified device/dtype.

        Call the native PyTorch `.to()` method on all tensors, parameters
        and NN modules to move the model to the specified device and/or dtype.
        """
        if self.components_ is None:
            raise ValueError(
                "PCA not fitted when calling `.to()`. "
                "Please call `fit` or `fit_transform` first."
            )
        attr_list = list(
            filter(
                lambda x: not x.startswith("__"),
                dir(self),
            )
        )
        for attr_name in attr_list:
            attr_value = getattr(self, attr_name)
            if isinstance(
                attr_value, (torch.Tensor, torch.nn.Parameter, torch.nn.Module)
            ):
                setattr(
                    self,
                    attr_name,
                    attr_value.to(
                        device=device, dtype=dtype, non_blocking=non_blocking, **kwargs
                    ),
                )
