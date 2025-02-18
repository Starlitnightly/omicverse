from __future__ import annotations

import logging
import os

import celltypist
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mode

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class CELLTYPIST(BaseAlgorithm):
    """
    Class to compute Celltypist classifier.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_celltypist_prediction",
        method_kwargs: dict | None = None,
        classifier_dict: dict | None = None,
    ) -> None:
        super().__init__(batch_key=batch_key, labels_key=labels_key, result_key=result_key)

        default_method_kwargs = {"check_expression": False, "n_jobs": 10, "max_iter": 500}
        self.method_kwargs = {**default_method_kwargs, **(method_kwargs or {})}

        default_classifier_dict = {"mode": "best match", "majority_voting": True}
        self.classifier_dict = {**default_classifier_dict, **(classifier_dict or {})}

    def predict(self, adata):
        logging.info(f'Saving celltypist results to adata.obs["{self.result_key}"]')
        prediction_mode = adata.uns.get("_prediction_mode")
        save_path = adata.uns.get("_save_path_trained_models")

        # Setup for over_clustering based on prediction mode
        if prediction_mode == "fast":
            self.classifier_dict["majority_voting"] = False
            over_clustering = None

        elif prediction_mode == "inference" and "over_clustering" in adata.obs and not settings.recompute_embeddings:
            index_file = os.path.join(save_path, "pynndescent_index.joblib")
            index = joblib.load(index_file)
            ref_mask = adata.obs["_dataset"] == "ref"
            query_mask = adata.obs["_dataset"] == "query"
            query_features = adata.obsm["X_pca"][query_mask, :].astype(np.float32)
            indices, _ = index.query(query_features, k=5)

            ref_over = adata.obs.loc[ref_mask, "over_clustering"].cat.codes.values
            neighbor_codes = ref_over[indices]
            majority = mode(neighbor_codes, axis=1).mode.flatten()
            categories = adata.obs["over_clustering"].cat.categories
            adata.obs.loc[query_mask, "over_clustering"] = categories[majority]

            over_clustering = adata.obs.loc[adata.obs["_predict_cells"] == "relabel", "over_clustering"]

        else:
            transformer = "rapids" if settings.cuml else None
            sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca", transformer=transformer)
            sc.tl.leiden(adata, resolution=25.0, key_added="over_clustering")
            over_clustering = adata.obs.loc[adata.obs["_predict_cells"] == "relabel", "over_clustering"]

        # Train model if needed
        if prediction_mode == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            if len(train_idx) > 100000 and not settings.cuml:
                self.method_kwargs["use_SGD"] = True
                self.method_kwargs["mini_batch"] = True

            train_adata = adata[train_idx].copy()
            model = celltypist.train(
                train_adata,
                self.labels_key,
                use_GPU=settings.cuml,
                **self.method_kwargs,
            )
            model_file = os.path.join(save_path, "celltypist.pkl")
            model.write(model_file)
        else:
            model_file = os.path.join(save_path, "celltypist.pkl")

        # Annotate using the trained model
        predict_mask = adata.obs["_predict_cells"] == "relabel"
        predictions = celltypist.annotate(
            adata[predict_mask],
            model=model_file,
            over_clustering=over_clustering,
            **self.classifier_dict,
        )

        # Choose the predicted label column
        out_column = (
            "majority_voting"
            if "majority_voting" in predictions.predicted_labels.columns
            else "predicted_labels"
        )

        # Initialize the result column if it doesn't exist
        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]

        adata.obs.loc[predict_mask, self.result_key] = predictions.predicted_labels[out_column]

        # Save probability values if required
        if self.return_probabilities:
            prob_col = f"{self.result_key}_probabilities"
            if prob_col not in adata.obs.columns:
                adata.obs[prob_col] = pd.Series(dtype="float64")
            adata.obs.loc[predict_mask, prob_col] = predictions.probability_matrix.max(axis=1).values
