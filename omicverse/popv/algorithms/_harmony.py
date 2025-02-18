from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from harmony import harmonize
from pynndescent import PyNNDescentTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class KNN_HARMONY(BaseAlgorithm):
    """
    Compute KNN classifier after Harmony integration.

    Parameters
    ----------
    batch_key : str or None
        Key in obs field of adata for batch information. Default "_batch_annotation".
    labels_key : str or None
        Key in obs field of adata for cell-type information. Default "_labels_annotation".
    result_key : str or None
        Key in obs where celltype annotation results are stored. Default "popv_knn_harmony_prediction".
    embedding_key : str or None
        Key in obsm for PCA-harmony embedding. Default "X_pca_harmony_popv".
    umap_key : str or None
        Key in obsm for UMAP embedding of integrated data. Default "X_umap_harmony_popv".
    method_kwargs : dict or None
        Additional parameters for HARMONY. Options at harmony.integrate_scanpy.
    classifier_dict : dict or None
        Dictionary with non-default values for KNN classifier. (n_neighbors, weights).
    embedding_kwargs : dict or None
        Dictionary with non-default values for UMAP embedding. Options at sc.tl.umap.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_knn_harmony_prediction",
        embedding_key: str | None = "X_pca_harmony_popv",
        umap_key: str | None = "X_umap_harmony_popv",
        method_kwargs: dict | None = None,
        classifier_dict: dict | None = None,
        embedding_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            embedding_key=embedding_key,
            umap_key=umap_key,
        )

        self.method_kwargs = {"dimred": 50, **(method_kwargs or {})}
        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15, **(classifier_dict or {})}
        self.embedding_kwargs = {"min_dist": 0.1, **(embedding_kwargs or {})}
        self.recompute_classifier = True

    def compute_integration(self, adata):
        logging.info("Integrating data with harmony")
        mode = adata.uns["_prediction_mode"]
        if mode == "inference" and self.embedding_key in adata.obsm and not settings.recompute_embeddings:
            self.recompute_classifier = False
            index_path = os.path.join(adata.uns["_save_path_trained_models"], "pynndescent_index.joblib")
            index = joblib.load(index_path)
            query_mask = adata.obs["_dataset"] == "query"
            ref_mask = adata.obs["_dataset"] == "ref"
            query_features = adata.obsm["X_pca"][query_mask, :]
            indices, _ = index.query(query_features.astype(np.float32), k=5)
            neighbor_values = adata.obsm[self.embedding_key][ref_mask][:, :][indices].astype(np.float32)
            adata.obsm[self.embedding_key][query_mask, :] = np.mean(neighbor_values, axis=1)
            adata.obsm[self.embedding_key] = adata.obsm[self.embedding_key].astype(np.float32)
        elif mode != "fast":
            adata.obsm[self.embedding_key] = harmonize(
                adata.obsm["X_pca"],
                adata.obs,
                batch_key=self.batch_key,
                use_gpu=settings.accelerator == "gpu",
            )
        else:
            raise ValueError(f"Prediction mode {mode} not supported for HARMONY")

    def predict(self, adata):
        """
        Predict celltypes using KNN on Harmony integrated embeddings.
        Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Saving knn on harmony results to adata.obs["{self.result_key}"]')
        mode = adata.uns["_prediction_mode"]

        if self.recompute_classifier:
            ref_idx = adata.obs["_labelled_train_indices"]
            train_X = adata[ref_idx].obsm[self.embedding_key].copy()
            train_Y = adata.obs.loc[ref_idx, self.labels_key].cat.codes.to_numpy()
            knn = make_pipeline(
                PyNNDescentTransformer(
                    n_neighbors=self.classifier_dict["n_neighbors"],
                    n_jobs=settings.n_jobs,
                ),
                KNeighborsClassifier(metric="precomputed", weights=self.classifier_dict["weights"]),
            )
            knn.fit(train_X, train_Y)
            if mode == "retrain" and adata.uns["_save_path_trained_models"]:
                save_path = os.path.join(adata.uns["_save_path_trained_models"], "harmony_knn_classifier.joblib")
                with open(save_path, "wb") as f:
                    joblib.dump(knn, f)
        else:
            load_path = os.path.join(adata.uns["_save_path_trained_models"], "harmony_knn_classifier.joblib")
            with open(load_path, "rb") as f:
                knn = joblib.load(f)

        # Save results
        relabel_mask = adata.obs["_predict_cells"] == "relabel"
        embedding = adata[relabel_mask].obsm[self.embedding_key]
        knn_pred = knn.predict(embedding)
        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[relabel_mask, self.result_key] = adata.uns["label_categories"][knn_pred]

        if self.return_probabilities:
            prob_key = f"{self.result_key}_probabilities"
            if prob_key not in adata.obs.columns:
                adata.obs[prob_key] = pd.Series(dtype="float64")
            adata.obs.loc[relabel_mask, prob_key] = np.max(embedding, axis=1)

    def compute_umap(self, adata):
        """
        Compute UMAP embedding of the integrated data.
        Results are stored in adata.obsm[self.umap_key].
        """
        if self.compute_umap_embedding:
            logging.info(f'Saving UMAP of harmony results to adata.obs["{self.umap_key}"]')
            transformer = "rapids" if settings.cuml else None
            sc.pp.neighbors(adata, use_rep=self.embedding_key, transformer=transformer)
            method = "rapids" if settings.cuml else "umap"
            # sc.tl.umap returns a copy when using copy=True.
            adata_umap = sc.tl.umap(adata, copy=True, method=method, **self.embedding_kwargs)
            adata.obsm[self.umap_key] = adata_umap.obsm["X_umap"]
