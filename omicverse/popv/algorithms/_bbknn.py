from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import scanpy as sc
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class KNN_BBKNN(BaseAlgorithm):
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_knn_bbknn_prediction",
        umap_key: str | None = "X_umap_bbknn_popv",
        method_kwargs: dict | None = None,
        classifier_kwargs: dict | None = None,
        embedding_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            umap_key=umap_key,
        )
        self.method_kwargs = {
            "metric": "euclidean",
            "approx": True,
            "n_pcs": 50,
            "neighbors_within_batch": 3,
            "use_annoy": False,
        }
        if method_kwargs:
            self.method_kwargs.update(method_kwargs)

        self.classifier_kwargs = {"weights": "uniform", "n_neighbors": 15}
        if classifier_kwargs:
            self.classifier_kwargs.update(classifier_kwargs)

        self.embedding_kwargs = {"min_dist": 0.1}
        if embedding_kwargs:
            self.embedding_kwargs.update(embedding_kwargs)

    def compute_integration(self, adata):
        """Compute BBKNN integration."""
        logging.info("Integrating data with bbknn")
        # Use the available umap key from the reference integration.
        ref_umap_key = "X_umap_bbknn"
        if (
            adata.uns.get("_prediction_mode") == "inference"
            and ref_umap_key in adata.obsm
            and not settings.recompute_embeddings
        ):
            # Cache masks for query and reference cells
            query_mask = adata.obs["_dataset"] == "query"
            ref_mask = adata.obs["_dataset"] == "ref"

            index_path = os.path.join(adata.uns["_save_path_trained_models"], "pynndescent_index.joblib")
            index = joblib.load(index_path)

            query_features = adata.obsm["X_pca"][query_mask].astype(np.float32)
            indices, _ = index.query(query_features, k=5)

            ref_umap = adata.obsm[ref_umap_key][ref_mask].astype(np.float32)
            neighbor_embedding = ref_umap[indices]
            # Compute mean embedding across neighbors and assign
            adata.obsm[self.umap_key][query_mask] = np.mean(neighbor_embedding, axis=1)
            adata.obsm[self.umap_key] = adata.obsm[self.umap_key].astype(np.float32)

            ref_probs = adata.obs[f"{self.result_key}_probabilities"][ref_mask].astype(np.float32)
            neighbor_probabilities = ref_probs[indices]
            adata.obs.loc[query_mask, f"{self.result_key}_probabilities"] = np.mean(
                neighbor_probabilities, axis=1
            )

            ref_prediction = adata.obs[f"{self.result_key}"][ref_mask]
            neighbor_prediction = ref_prediction.iloc[indices].to_numpy()
            mode_vals = mode(neighbor_prediction, axis=1).mode.ravel()
            adata.obs.loc[query_mask, f"{self.result_key}"] = mode_vals
        else:
            # If too many batches, reduce settings to avoid memory issues.
            batches = adata.obs[self.batch_key].unique()
            if len(batches) > 100:
                logging.warning("Using PyNNDescent instead of FAISS due to high number of batches.")
                self.method_kwargs["neighbors_within_batch"] = 1
                self.method_kwargs["pynndescent_n_neighbors"] = 10
                sc.external.pp.bbknn(
                    adata, batch_key=self.batch_key, use_faiss=False, use_rep="X_pca", **self.method_kwargs
                )
            else:
                sc.external.pp.bbknn(
                    adata, batch_key=self.batch_key, use_faiss=True, use_rep="X_pca", **self.method_kwargs
                )

    def predict(self, adata):
        """Predict celltypes using KNN based on BBKNN results."""
        logging.info(f'Saving knn on bbknn results to adata.obs["{self.result_key}"]')
        distances = adata.obsp["distances"]
        # Get indices of reference cells
        ref_indices = np.where(adata.obs["_labelled_train_indices"])[0]
        train_y = adata.obs.loc[adata.obs["_labelled_train_indices"], self.labels_key].cat.codes.to_numpy()

        train_distances = distances[ref_indices, :][:, ref_indices]
        test_distances = distances[:, :][:, ref_indices]

        # Adjust KNN neighbors if BBKNN returned a smaller graph
        train_counts = np.diff(train_distances.indptr).min()
        test_counts = np.diff(test_distances.indptr).min()
        smallest_graph = min(train_counts, test_counts)
        if smallest_graph < self.classifier_kwargs["n_neighbors"]:
            logging.warning(f"BBKNN found only {smallest_graph} neighbors. Reducing KNN n_neighbors.")
            self.classifier_kwargs["n_neighbors"] = smallest_graph

        knn = KNeighborsClassifier(metric="precomputed", **self.classifier_kwargs)
        knn.fit(train_distances, y=train_y)
        predicted_codes = knn.predict(test_distances)
        adata.obs[self.result_key] = adata.uns["label_categories"][predicted_codes]

        if self.return_probabilities:
            probabilities = knn.predict_proba(test_distances).max(axis=1)
            adata.obs[f"{self.result_key}_probabilities"] = probabilities

    def compute_umap(self, adata):
        """Compute UMAP embedding of integrated data."""
        if self.compute_umap_embedding:
            logging.info(f'Saving UMAP of bbknn results to adata.obsm["{self.umap_key}"]')
            if len(adata.obs[self.batch_key]) < 30 and settings.cuml:
                method = "rapids"
            else:
                logging.warning("Using UMAP instead of RAPIDS due to high number of batches.")
                method = "umap"
            # Compute UMAP embedding with a copy and assign the embedded coordinates directly.
            embedded = sc.tl.umap(adata, copy=True, method=method, **self.embedding_kwargs).obsm["X_umap"]
            adata.obsm[self.umap_key] = embedded
