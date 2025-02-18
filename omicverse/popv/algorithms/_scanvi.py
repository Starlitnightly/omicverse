from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scvi

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class SCANVI_POPV(BaseAlgorithm):
    """
    Class to compute a classifier using the scANVI model.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        save_folder: str | None = None,
        result_key: str | None = "popv_scanvi_prediction",
        embedding_key: str | None = "X_scanvi_popv",
        umap_key: str | None = "X_umap_scanvi_popv",
        model_kwargs: dict | None = None,
        classifier_kwargs: dict | None = None,
        embedding_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            embedding_key=embedding_key,
        )
        self.umap_key = umap_key
        self.save_folder = save_folder

        # Initialize parameters with defaults, update with provided kwargs
        self.model_kwargs = {
            "dropout_rate": 0.05,
            "dispersion": "gene",
            "n_layers": 3,
            "n_latent": 20,
            "gene_likelihood": "nb",
            "use_batch_norm": "none",
            "use_layer_norm": "both",
            "encode_covariates": True,
        }
        if model_kwargs:
            self.model_kwargs.update(model_kwargs)

        self.train_kwargs = {
            "max_epochs": 20,
            "batch_size": 512,
            "n_samples_per_label": 20,
            "accelerator": settings.accelerator,
            "plan_kwargs": {"n_epochs_kl_warmup": 20},
            "max_epochs_unsupervised": 20,
        }
        if train_kwargs:
            self.train_kwargs.update(train_kwargs)
        # Save separate unsupervised training epoch value and remove from train_kwargs
        self.max_epochs_unsupervised = self.train_kwargs.pop("max_epochs_unsupervised")
        self.max_epochs = self.train_kwargs.get("max_epochs", None)

        self.classifier_kwargs = {"n_layers": 3, "dropout_rate": 0.1}
        if classifier_kwargs:
            self.classifier_kwargs.update(classifier_kwargs)

        self.embedding_kwargs = {"min_dist": 0.3}
        if embedding_kwargs:
            self.embedding_kwargs.update(embedding_kwargs)

    def compute_integration(self, adata):
        """
        Compute scANVI model and integrate data.
        Resulting latent representation is stored in adata.obsm[self.embedding_key].
        """
        logging.info("Integrating data with scANVI")
        if adata.uns["_prediction_mode"] == "retrain":
            if adata.uns["_pretrained_scvi_path"]:
                scvi_model = scvi.model.SCVI.load(
                    os.path.join(adata.uns["_save_path_trained_models"], "scvi"),
                    adata=adata,
                )
            else:
                scvi.model.SCVI.setup_anndata(
                    adata,
                    batch_key=self.batch_key,
                    labels_key=self.labels_key,
                    layer="scvi_counts",
                )
                scvi_model = scvi.model.SCVI(adata, **self.model_kwargs)
                scvi_model.train(
                    max_epochs=self.max_epochs_unsupervised,
                    accelerator=settings.accelerator,
                    plan_kwargs={"n_epochs_kl_warmup": 20},
                )

            self.model = scvi.model.SCANVI.from_scvi_model(
                scvi_model,
                unlabeled_category=adata.uns["unknown_celltype_label"],
                classifier_parameters=self.classifier_kwargs,
            )
        else:
            query = adata[adata.obs["_predict_cells"] == "relabel"].copy()
            self.model = scvi.model.SCANVI.load_query_data(
                query,
                os.path.join(adata.uns["_save_path_trained_models"], "scanvi"),
                freeze_classifier=True,
            )

        # Adjust training configuration for "fast" prediction mode.
        if adata.uns["_prediction_mode"] == "fast":
            self.train_kwargs["max_epochs"] = 1

        self.model.train(**self.train_kwargs)

        if adata.uns["_prediction_mode"] == "retrain":
            self.model.save(
                os.path.join(adata.uns["_save_path_trained_models"], "scanvi"),
                save_anndata=False,
                overwrite=True,
            )

        latent_representation = self.model.get_latent_representation()
        relabel_mask = adata.obs["_predict_cells"] == "relabel"
        if self.embedding_key not in adata.obsm:
            # Initialize with the correct shape if not present.
            adata.obsm[self.embedding_key] = np.zeros((adata.n_obs, latent_representation.shape[1]))
        adata.obsm[self.embedding_key][relabel_mask, :] = latent_representation

    def predict(self, adata):
        """
        Predict cell types using scANVI.
        Results are stored in adata.obs[self.result_key].
        """
        logging.info(f'Saving scanvi label prediction to adata.obs["{self.result_key}"]')

        # Subset once to avoid redundant computation.
        relabel_adata = adata[adata.obs["_predict_cells"] == "relabel"]

        # Predict labels once and cache the result.
        predicted_labels = self.model.predict(relabel_adata)
        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]

        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", self.result_key] = predicted_labels

        if self.return_probabilities:
            # Calculate soft predictions only once.
            soft_preds = self.model.predict(relabel_adata, soft=True)
            probs = np.max(soft_preds, axis=1)
            prob_key = f"{self.result_key}_probabilities"
            if prob_key not in adata.obs.columns:
                adata.obs[prob_key] = pd.Series(dtype="float64")
            adata.obs.loc[adata.obs["_predict_cells"] == "relabel", prob_key] = probs

    def compute_umap(self, adata):
        """
        Compute the UMAP embedding of the integrated data.
        Result is stored in adata.obsm[self.umap_key].
        """
        transformer = "rapids" if settings.cuml else None
        sc.pp.neighbors(adata, use_rep=self.embedding_key, transformer=transformer)
        method = "rapids" if settings.cuml else "umap"
        umap_result = sc.tl.umap(adata, copy=True, method=method, **self.embedding_kwargs)
        adata.obsm[self.umap_key] = umap_result.obsm["X_umap"]
