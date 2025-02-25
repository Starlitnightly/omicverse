from __future__ import annotations

import logging

import numpy as np
import scanpy as sc
from pynndescent import PyNNDescentTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class KNN_SCANORAMA(BaseAlgorithm):
    """
    Compute KNN classifier after Scanorama integration.

    Parameters
    ----------
    batch_key : str, optional
        Key in adata.obs containing batch information. Default is "_batch_annotation".
    labels_key : str, optional
        Key in adata.obs containing cell-type labels. Default is "_labels_annotation".
    result_key : str, optional
        Key in adata.obs where predictions are stored.
        Default is "popv_knn_scanorama_prediction".
    embedding_key : str, optional
        Key in adata.obsm with the integrated embedding.
        Default is "X_pca_scanorama_popv".
    umap_key : str, optional
        Key in adata.obsm with the UMAP embedding.
        Default is "X_umap_scanorama_popv".
    method_kwargs : dict, optional
        Additional parameters for Scanorama integration.
    classifier_kwargs : dict, optional
        KNeighborsClassifier parameters.
        Default is {"weights": "uniform", "n_neighbors": 15}.
    embedding_kwargs : dict, optional
        Additional parameters for UMAP embedding. Default is {"min_dist": 0.1}.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_knn_scanorama_prediction",
        embedding_key: str | None = "X_pca_scanorama_popv",
        umap_key: str | None = "X_umap_scanorama_popv",
        method_kwargs: dict | None = None,
        classifier_kwargs: dict | None = None,
        embedding_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            embedding_key=embedding_key,
            umap_key=umap_key,
        )

        self.method_kwargs = method_kwargs or {}
        self.classifier_kwargs = {"weights": "uniform", "n_neighbors": 15}
        self.classifier_kwargs.update(classifier_kwargs or {})
        self.embedding_kwargs = {"min_dist": 0.1}
        self.embedding_kwargs.update(embedding_kwargs or {})

    def compute_integration(self, adata):
        """
        Integrate data using Scanorama.

        Parameters
        ----------
        adata : AnnData
            AnnData object. Integrated embedding is stored in adata.obsm[self.embedding_key].
        """
        logging.info("Integrating data with scanorama")
        # Sort and copy adata for integration
        sorted_idx = adata.obs.sort_values(self.batch_key).index
        tmp = adata[sorted_idx].copy()
        sc.external.pp.scanorama_integrate(
            tmp,
            key=self.batch_key,
            adjusted_basis=self.embedding_key,
            **self.method_kwargs,
        )
        adata.obsm[self.embedding_key] = tmp[adata.obs_names].obsm[self.embedding_key]

    def predict(self, adata):
        """
        Compute KNN classifier on Scanorama integrated data.

        Parameters
        ----------
        adata : AnnData
            AnnData object. Predictions are stored in adata.obs[self.result_key].
        """
        logging.info(f'Saving knn predictions to adata.obs["{self.result_key}"]')

        ref_idx = adata.obs["_labelled_train_indices"]
        train_X = adata[ref_idx].obsm[self.embedding_key]
        train_Y = adata.obs.loc[ref_idx, self.labels_key].cat.codes.to_numpy()

        knn = make_pipeline(
            PyNNDescentTransformer(
                n_neighbors=self.classifier_kwargs["n_neighbors"],
                n_jobs=settings.n_jobs,
            ),
            KNeighborsClassifier(
                metric="precomputed", weights=self.classifier_kwargs["weights"]
            ),
        )

        knn.fit(train_X, train_Y)
        knn_pred = knn.predict(adata.obsm[self.embedding_key])
        adata.obs[self.result_key] = adata.uns["label_categories"][knn_pred]

        if getattr(self, "return_probabilities", False):
            probs = knn.predict_proba(adata.obsm[self.embedding_key])
            adata.obs[f"{self.result_key}_probabilities"] = np.max(probs, axis=1)

    def compute_umap(self, adata):
        """
        Compute UMAP embedding of integrated data.

        Parameters
        ----------
        adata : AnnData
            AnnData object. UMAP embedding is stored in adata.obsm[self.umap_key].
        """
        if getattr(self, "compute_umap_embedding", False):
            logging.info(f'Saving UMAP embedding to adata.obsm["{self.umap_key}"]')
            transformer = "rapids" if settings.cuml else None
            sc.pp.neighbors(
                adata, use_rep=self.embedding_key, transformer=transformer
            )
            method = "rapids" if settings.cuml else "umap"
            umap_result = sc.tl.umap(adata, copy=True, method=method, **self.embedding_kwargs)
            adata.obsm[self.umap_key] = umap_result.obsm["X_umap"]
