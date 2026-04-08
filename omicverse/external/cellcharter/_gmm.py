from __future__ import annotations

import importlib.util
import logging
from dataclasses import asdict, dataclass
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture

from ._utils import AnyRandom

logger = logging.getLogger(__name__)


def _has_torchgmm() -> bool:
    return importlib.util.find_spec("torchgmm") is not None


@dataclass
class _ClusterConfig:
    n_clusters: int = 1
    covariance_type: str = "full"
    init_strategy: str = "kmeans"
    convergence_tolerance: float = 0.001
    covariance_regularization: float = 1e-06
    batch_size: int | None = None
    trainer_params: dict | None = None
    random_state: AnyRandom = 0
    backend: str = "auto"


class Cluster:
    """Minimal CellCharter-style GMM clusterer with optional torchgmm backend."""

    def __init__(
        self,
        n_clusters: int = 1,
        *,
        covariance_type: str = "full",
        init_strategy: str = "kmeans",
        convergence_tolerance: float = 0.001,
        covariance_regularization: float = 1e-06,
        batch_size: int | None = None,
        trainer_params: dict | None = None,
        random_state: AnyRandom = 0,
        backend: str = "auto",
    ):
        self.config = _ClusterConfig(
            n_clusters=n_clusters,
            covariance_type=covariance_type,
            init_strategy=init_strategy,
            convergence_tolerance=convergence_tolerance,
            covariance_regularization=covariance_regularization,
            batch_size=batch_size,
            trainer_params=trainer_params,
            random_state=random_state,
            backend=backend,
        )
        self.n_clusters = n_clusters
        self.model_: Any | None = None
        self.converged_: bool | None = None
        self.num_iter_: int | None = None
        self.nll_: float | None = None
        self.backend_: str | None = None
        self._resolved_reg_covar: float = float(covariance_regularization)

    def _resolve_backend(self) -> str:
        backend = self.config.backend
        if backend == "auto":
            return "torchgmm" if _has_torchgmm() else "sklearn"
        if backend == "torchgmm":
            if not _has_torchgmm():
                raise ImportError(
                    "CellCharter backend `torchgmm` was requested, but `torchgmm` is not installed."
                )
            return backend
        if backend == "sklearn":
            return backend
        raise ValueError("`backend` must be one of {'auto', 'torchgmm', 'sklearn'}.")

    def _fit_sklearn(self, X: np.ndarray) -> None:
        reg_covar = float(self.config.covariance_regularization)
        last_error = None
        while reg_covar <= 1.0:
            model = SklearnGaussianMixture(
                n_components=self.config.n_clusters,
                covariance_type=self.config.covariance_type,
                init_params=self.config.init_strategy,
                tol=self.config.convergence_tolerance,
                reg_covar=reg_covar,
                random_state=self.config.random_state,
            )
            try:
                model.fit(X)
            except ValueError as exc:
                last_error = exc
                if "ill-defined empirical covariance" not in str(exc):
                    raise
                reg_covar *= 10.0
                continue
            self.model_ = model
            self._resolved_reg_covar = reg_covar
            self.converged_ = bool(model.converged_)
            self.num_iter_ = int(model.n_iter_)
            self.nll_ = float(-model.score(X))
            return

        raise ValueError(
            "CellCharter clustering failed after increasing covariance regularization up to 1.0."
        ) from last_error

    def _fit_torchgmm(self, X: np.ndarray) -> None:
        import torch
        from torchgmm.bayes import GaussianMixture as TorchGaussianMixture

        class _TorchGMMAdapter(TorchGaussianMixture):
            def __init__(
                self,
                n_clusters: int = 1,
                *,
                covariance_type: str = "full",
                init_strategy: str = "kmeans",
                convergence_tolerance: float = 0.001,
                covariance_regularization: float = 1e-06,
                batch_size: int | None = None,
                trainer_params: dict | None = None,
                random_state: AnyRandom = 0,
            ):
                super().__init__(
                    num_components=n_clusters,
                    covariance_type=covariance_type,
                    init_strategy=init_strategy,
                    convergence_tolerance=convergence_tolerance,
                    covariance_regularization=covariance_regularization,
                    batch_size=batch_size,
                    trainer_params=trainer_params,
                )
                self.n_clusters = n_clusters
                self.random_state = random_state

            def fit(self, data):
                if sps.issparse(data):
                    raise ValueError(
                        "Sparse data is not supported. You may have forgotten to reduce the dimensionality of the data. "
                        "Otherwise, please convert the data to a dense format."
                    )
                return self._fit(data)

            def _fit(self, data):
                try:
                    return super().fit(data)
                except torch._C._LinAlgError as exc:
                    if self.covariance_regularization >= 1:
                        raise ValueError(
                            "Cholesky decomposition failed even with covariance regularization = 1. "
                            "The matrix may be singular."
                        ) from exc
                    self.covariance_regularization *= 10
                    logger.warning(
                        "Cholesky decomposition failed. Retrying with covariance regularization %s.",
                        self.covariance_regularization,
                    )
                    return self._fit(data)

            def predict(self, data):
                labels = super().predict(data)
                if hasattr(labels, "detach"):
                    labels = labels.detach().cpu().numpy()
                return labels

        if isinstance(self.config.random_state, (int, np.integer)):
            seed = int(self.config.random_state)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        model = _TorchGMMAdapter(
            n_clusters=self.config.n_clusters,
            covariance_type=self.config.covariance_type,
            init_strategy=self.config.init_strategy,
            convergence_tolerance=self.config.convergence_tolerance,
            covariance_regularization=self.config.covariance_regularization,
            batch_size=self.config.batch_size,
            trainer_params=self.config.trainer_params,
            random_state=self.config.random_state,
        )
        model.fit(X)
        self.model_ = model
        self._resolved_reg_covar = float(model.covariance_regularization)
        self.converged_ = bool(getattr(model, "converged_", True))
        self.num_iter_ = getattr(model, "num_iter_", None)
        self.nll_ = float(getattr(model, "nll_", np.nan))

    def fit(self, adata: ad.AnnData, use_rep: str = "X_cellcharter"):
        """Fit a Gaussian mixture on ``adata.obsm[use_rep]``."""
        X = adata.X if use_rep is None else adata.obsm[use_rep]
        if sps.issparse(X):
            raise ValueError(
                "Sparse data is not supported by CellCharter clustering. "
                "Please use a dense embedding such as `X_pca` or the aggregated `X_cellcharter` representation."
            )

        self.backend_ = self._resolve_backend()

        if self.backend_ == "torchgmm":
            X = np.asarray(X, dtype=np.float32)
            self._fit_torchgmm(X)
        else:
            X = np.asarray(X, dtype=np.float64)
            self._fit_sklearn(X)

        params = asdict(self.config)
        params["backend"] = self.backend_
        params["covariance_regularization"] = self._resolved_reg_covar
        adata.uns["_cellcharter"] = params
        return self

    def predict(self, adata: ad.AnnData, use_rep: str = "X_cellcharter") -> pd.Categorical:
        """Predict categorical cluster labels for ``adata``."""
        if self.model_ is None:
            raise RuntimeError("Call `fit` before `predict`.")

        X = adata.X if use_rep is None else adata.obsm[use_rep]
        if sps.issparse(X):
            raise ValueError(
                "Sparse data is not supported by CellCharter clustering. "
                "Please use a dense embedding such as `X_pca` or the aggregated `X_cellcharter` representation."
            )

        dtype = np.float32 if self.backend_ == "torchgmm" else np.float64
        labels = self.model_.predict(np.asarray(X, dtype=dtype))
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()
        else:
            labels = np.asarray(labels)
        labels = labels.astype(str)
        categories = np.arange(self.n_clusters).astype(str)
        return pd.Categorical(labels, categories=categories)
