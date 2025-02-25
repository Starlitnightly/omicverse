from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class Random_Forest(BaseAlgorithm):
    """
    Class to compute Random forest classifier.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
        Default is "_batch_annotation".
    labels_key
        Key in obs field of adata for cell-type information.
        Default is "_labels_annotation".
    layer_key
        Key in layers field of adata used for classification. By default uses 'X' (log1p10K).
    result_key
        Key in obs in which celltype annotation results are stored.
        Default is "popv_rf_prediction".
    enable_cuml
        Enable cuml, which currently doesn't support weighting. Default to popv.settings.cuml.
    classifier_dict
        Dictionary to supply non-default values for RF classifier. Options at :class:`sklearn.ensemble.RandomForestClassifier`.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layer_key: str | None = None,
        result_key: str | None = "popv_rf_prediction",
        enable_cuml: bool = settings.cuml,
        classifier_dict: dict | None = None,
    ) -> None:
        super().__init__(batch_key=batch_key, labels_key=labels_key, result_key=result_key)
        self.layer_key = layer_key
        # Set up default classifier parameters, then update from provided dictionary if any.
        self.classifier_dict = {
            "class_weight": "balanced_subsample",
            "max_features": 200,
            "n_jobs": settings.n_jobs,
        }
        if classifier_dict:
            self.classifier_dict.update(classifier_dict)
        self.enable_cuml = enable_cuml

    def predict(self, adata):
        """
        Predict celltypes using Random Forest.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Computing random forest classifier. Storing prediction in adata.obs["{self.result_key}"]')

        # Determine the test features based on layer_key.
        test_x = adata[adata.obs["_predict_cells"] == "relabel"].layers[self.layer_key] if self.layer_key else adata.X

        model_path = os.path.join(adata.uns["_save_path_trained_models"], "rf_classifier.joblib")

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_x = adata[train_idx].layers[self.layer_key] if self.layer_key else adata[train_idx].X
            train_y = adata.obs.loc[train_idx, self.labels_key].cat.codes.to_numpy()
            rf = RandomForestClassifier(**self.classifier_dict)
            rf.fit(train_x, train_y)
            with open(model_path, "wb") as f:
                joblib.dump(rf, f)
        else:
            with open(model_path, "rb") as f:
                rf = joblib.load(f)

        # Ensure the result column exists.
        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]

        # Write predictions.
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", self.result_key] = \
            adata.uns["label_categories"][rf.predict(test_x)]

        if self.return_probabilities:
            probability_key = f"{self.result_key}_probabilities"
            if probability_key not in adata.obs.columns:
                adata.obs[probability_key] = pd.Series(dtype="float64")
            adata.obs.loc[adata.obs["_predict_cells"] == "relabel", probability_key] = \
                np.max(rf.predict_proba(test_x), axis=1).astype(float)
