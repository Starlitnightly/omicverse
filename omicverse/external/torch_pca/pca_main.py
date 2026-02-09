"""Main module for PCA."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
# Inspired from https://github.com/scikit-learn (BSD-3-Clause License)
# Copyright (c) Scikit-learn developers. All Rights Reserved.
from math import log
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType

from .ncompo import NComponentsType, find_ncomponents
from .svd import choose_svd_solver, randomized_svd, svd_flip


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
        One of {'auto', 'full', 'covariance_eigh'}

        * 'auto': the solver is selected automatically based on the shape of the input.
        * 'full': Run exact full SVD with torch.linalg.svd
        * 'covariance_eigh': Compute the covariance matrix and take
          the eigenvalues decomposition with torch.linalg.eigh.
          Most efficient for small n_features and large n_samples.
        * 'randomized': Compute the randomized SVD by the method of Halko et al.

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
        svd_solver: str = "auto",
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

        if self.svd_solver_ not in ["auto", "full", "covariance_eigh", "randomized"]:
            raise ValueError(
                "Unknown SVD solver. `svd_solver` should be one of "
                "'auto', 'full', 'covariance_eigh', 'randomized'."
            )

    def fit_transform(self, inputs: Tensor, *, determinist: bool = True) -> Tensor:
        """Fit the PCA model and apply the dimensionality reduction.

        Parameters
        ----------
        inputs : Tensor
            Input data of shape (n_samples, n_features).
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
        self.fit(inputs, determinist=determinist)
        transformed = self.transform(inputs)
        return transformed

    def fit(self, inputs: Tensor, *, determinist: bool = True) -> "PCA":
        """Fit the PCA model and return it.

        Parameters
        ----------
        inputs : Tensor
            Input data of shape (n_samples, n_features).
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
        # Auto-cast to float32 because float16 is not supported
        if inputs.dtype == torch.float16:
            inputs = inputs.to(torch.float32)

        if self.svd_solver_ == "auto":
            self.svd_solver_ = choose_svd_solver(
                inputs=inputs,
                n_components=self.n_components_,
            )
        self.mean_ = inputs.mean(dim=-2, keepdim=True)
        self.n_samples_, self.n_features_in_ = inputs.shape[-2:]
        if self.svd_solver_ == "full":
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
        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        self.noise_variance_ = (
            torch.mean(explained_variance[self.n_components_ :])
            if self.n_components_ < min(inputs.shape[-2:])
            else torch.tensor(0.0)
        )
        return self

    def _check_fitted(self, method_name: str) -> None:
        """Check if the PCA model is fitted."""
        if self.components_ is None:
            raise ValueError(
                f"PCA not fitted when calling {method_name}. "
                "Please call `fit` or `fit_transform` first."
            )

    def transform(self, inputs: Tensor, center: str = "fit") -> Tensor:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        inputs : Tensor
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