from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class XGboost(BaseAlgorithm):
    """
    Class to compute XGboost classifier.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
    labels_key
        Key in obs field of adata for cell-type information.
    layer_key
        Key in layers field of adata used for classification. By default uses 'X' (log1p10K).
    result_key
        Key in obs in which celltype annotation results are stored.
    classifier_dict
        Dictionary to supply non-default values for XGboost classifier.
        Options at :func:`xgboost.train`.
        Default is {'tree_method': 'hist', 'device': 'cuda' if settings.cuml else 'cpu', 'objective': 'multi:softprob'}.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layer_key: str | None = None,
        result_key: str | None = "popv_xgboost_prediction",
        classifier_dict: dict | None = None,
    ) -> None:
        super().__init__(batch_key=batch_key, labels_key=labels_key, result_key=result_key)
        self.layer_key = layer_key
        self.classifier_dict = {
            "tree_method": "hist",
            "device": "cuda" if settings.cuml else "cpu",
            "objective": "multi:softprob",
        }
        if classifier_dict:
            self.classifier_dict.update(classifier_dict)

    def predict(self, adata):
        """
        Predict celltypes using XGboost.

        Parameters
        ----------
        adata
            Anndata object. Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Computing XGboost classifier. Storing prediction in adata.obs["{self.result_key}"]')
        
        # Create a boolean mask for relabeling cells
        relabel_mask = adata.obs["_predict_cells"] == "relabel"
        subset = adata[relabel_mask]
        
        # Select layer if provided, otherwise use adata.X
        test_x = subset.layers[self.layer_key] if self.layer_key else subset.X
        test_y = subset.obs[self.labels_key].cat.codes.to_numpy()

        dtest = xgb.DMatrix(test_x, label=test_y)

        model_path = os.path.join(adata.uns["_save_path_trained_models"], "xgboost_classifier.model")
        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            training_data = adata[train_idx]
            train_x = training_data.layers[self.layer_key] if self.layer_key else training_data.X
            train_y = adata.obs.loc[train_idx, self.labels_key].cat.codes.to_numpy()
            dtrain = xgb.DMatrix(train_x, label=train_y)

            self.classifier_dict["num_class"] = len(adata.uns["label_categories"])
            bst = xgb.train(self.classifier_dict, dtrain, num_boost_round=300)
            bst.save_model(model_path)
        else:
            bst = xgb.Booster({"device": "cuda" if False else "cpu"})
            bst.load_model(model_path)

        output_probabilities = bst.predict(dtest)
        # Zero the probability for the unknown celltype
        unknown_idx = list(adata.uns["label_categories"]).index(adata.uns["unknown_celltype_label"])
        output_probabilities[:, unknown_idx] = 0.0

        # Map predictions back to cell-type labels
        predicted_labels = adata.uns["label_categories"][np.argmax(output_probabilities, axis=1)]
        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[relabel_mask, self.result_key] = predicted_labels

        # Optionally assign maximum prediction probability per cell
        if self.return_probabilities:
            prob_key = f"{self.result_key}_probabilities"
            if prob_key not in adata.obs.columns:
                adata.obs[prob_key] = pd.Series(dtype="float64")
            adata.obs.loc[relabel_mask, prob_key] = np.max(output_probabilities, axis=1).astype(float)
