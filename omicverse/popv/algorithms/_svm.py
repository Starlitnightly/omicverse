from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class Support_Vector(BaseAlgorithm):
    """
    Compute a linear SVM classifier.

    Parameters
    ----------
    batch_key : str or None
        Key for batch information in adata.obs; default: "_batch_annotation".
    labels_key : str or None
        Key for cell-type information in adata.obs; default: "_labels_annotation".
    layer_key : str or None
        Key for the layer to use for classification. Defaults to None (using adata.X).
    result_key : str or None
        Key in adata.obs for storing predictions; default: "popv_svm_prediction".
    classifier_dict : dict or None
        Dictionary with non-default parameters for SVM. Defaults to:
        {'C': 1, 'max_iter': 5000, 'class_weight': 'balanced'}.
    train_both : bool
        Whether to train both cuml and sklearn classifiers.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layer_key: str | None = None,
        result_key: str | None = "popv_svm_prediction",
        classifier_dict: dict | None = None,
        train_both: bool = False,
    ) -> None:
        super().__init__(batch_key=batch_key, labels_key=labels_key, result_key=result_key)
        self.layer_key = layer_key
        self.classifier_dict = {"C": 1, "max_iter": 5000, "class_weight": "balanced"}
        if classifier_dict:
            self.classifier_dict.update(classifier_dict)
        self.train_both = train_both

    def _save_model(self, clf, filename: str) -> None:
        save_path = os.path.join(adata.uns["_save_path_trained_models"], filename)
        with open(save_path, "wb") as f:
            joblib.dump(clf, f)

    def _load_model(self, filename: str):
        load_path = os.path.join(adata.uns["_save_path_trained_models"], filename)
        with open(load_path, "rb") as f:
            return joblib.load(f)

    def predict(self, adata):
        """
        Predict celltypes using a linear SVM.

        Parameters
        ----------
        adata
            Anndata object. Predictions are stored in adata.obs[self.result_key].
        """
        logging.info(f'Computing support vector machine. Storing prediction in adata.obs["{self.result_key}"]')
        test_x = adata.layers[self.layer_key] if self.layer_key else adata.X

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_x = adata[train_idx].layers[self.layer_key] if self.layer_key else adata[train_idx].X
            train_y = adata.obs.loc[train_idx, self.labels_key].cat.codes.to_numpy()

            if settings.cuml:
                from cuml.svm import LinearSVC
                from sklearn.multiclass import OneVsRestClassifier

                # Enable probability support if requested
                self.classifier_dict["probability"] = self.return_probabilities
                from cuml.svm import LinearSVC as cumlLinearSVC  # ensure import aliasing
                clf_cuml = OneVsRestClassifier(cumlLinearSVC(**self.classifier_dict))
                train_x_dense = train_x.todense() if hasattr(train_x, "todense") else train_x
                clf_cuml.fit(train_x_dense, train_y)
                with open(
                    os.path.join(adata.uns["_save_path_trained_models"], "svm_classifier_cuml.joblib"), "wb"
                ) as f:
                    joblib.dump(clf_cuml, f)
                self.classifier_dict.pop("probability")
            if not settings.cuml or self.train_both:
                clf = CalibratedClassifierCV(svm.LinearSVC(**self.classifier_dict))
                clf.fit(train_x, train_y)
                with open(
                    os.path.join(adata.uns["_save_path_trained_models"], "svm_classifier.joblib"), "wb"
                ) as f:
                    joblib.dump(clf, f)

        # Set up result dataframe with appropriate columns
        columns = [self.result_key]
        if self.return_probabilities:
            columns.append(f"{self.result_key}_probabilities")

        result_df = pd.DataFrame(index=adata.obs_names, columns=columns, dtype=float)
        result_df[self.result_key] = result_df[self.result_key].astype("object")

        if settings.cuml:
            with open(
                os.path.join(adata.uns["_save_path_trained_models"], "svm_classifier_cuml.joblib"), "rb"
            ) as f:
                clf = joblib.load(f)
            shard_size = int(settings.shard_size)
            for i in range(0, adata.n_obs, shard_size):
                tmp_x = test_x[i : i + shard_size]
                names_x = adata.obs_names[i : i + shard_size]
                tmp_x_dense = tmp_x.todense() if hasattr(tmp_x, "todense") else tmp_x
                preds = clf.predict(tmp_x_dense).astype(int)
                result_df.loc[names_x, self.result_key] = adata.uns["label_categories"][preds]
                if self.return_probabilities:
                    probs = np.max(clf.predict_proba(tmp_x_dense), axis=1).astype(float)
                    result_df.loc[names_x, f"{self.result_key}_probabilities"] = probs
        else:
            with open(
                os.path.join(adata.uns["_save_path_trained_models"], "svm_classifier.joblib"), "rb"
            ) as f:
                clf = joblib.load(f)
            preds = clf.predict(test_x)
            result_df[self.result_key] = adata.uns["label_categories"][preds]
            if self.return_probabilities:
                result_df[f"{self.result_key}_probabilities"] = np.max(clf.predict_proba(test_x), axis=1)

        # Initialize missing columns in adata.obs if needed
        for col in columns:
            if col not in adata.obs.columns:
                adata.obs[col] = (
                    pd.Series(dtype="float64")
                    if "probabilities" in col
                    else adata.uns["unknown_celltype_label"]
                )

        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", result_df.columns] = result_df
