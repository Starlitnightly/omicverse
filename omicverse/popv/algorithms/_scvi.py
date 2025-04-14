from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from pynndescent import PyNNDescentTransformer
from scvi.model import SCVI
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class KNN_SCVI(BaseAlgorithm):
    """
    Compute KNN classifier after scVI integration.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
    labels_key
        Key in obs field of adata for cell-type information.
    max_epochs
        Number of epochs for scvi training.
    result_key
        Key in obs where celltype annotation results are stored.
    embedding_key
        Key in obsm where latent dimensions are stored.
    umap_key
        Key in obsm where UMAP embedding is stored.
    model_kwargs
        Options passed to :class:`scvi.model.SCVI`.
    classifier_dict
        Options passed to :class:`sklearn.neighbors.KNeighborsClassifier`.
    embedding_kwargs
        Options passed to :func:`scanpy.tl.umap`.
    train_kwargs
        Options passed to :meth:`scvi.model.SCVI.train`.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        save_folder: str | None = None,
        result_key: str | None = "popv_knn_on_scvi_prediction",
        embedding_key: str | None = "X_scvi_popv",
        umap_key: str | None = "X_umap_scvi_popv",
        model_kwargs: dict | None = None,
        classifier_dict: dict | None = None,
        embedding_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            embedding_key=embedding_key,
            umap_key=umap_key,
        )
        # Set defaults and update with user-supplied options
        self.save_folder = save_folder

        self.model_kwargs = {
            "n_layers": 3,
            "n_latent": 20,
            "gene_likelihood": "nb",
            "use_batch_norm": "none",
            "use_layer_norm": "both",
            "encode_covariates": True,
        }
        if model_kwargs:
            self.model_kwargs.update(model_kwargs)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        if classifier_dict:
            self.classifier_dict.update(classifier_dict)

        self.train_kwargs = {
            "max_epochs": 20,
            "batch_size": 512,
            "accelerator": settings.accelerator,
            "plan_kwargs": {"n_epochs_kl_warmup": 20},
        }
        if train_kwargs:
            self.train_kwargs.update(train_kwargs)
        self.max_epochs = self.train_kwargs.get("max_epochs", None)

        self.embedding_kwargs = {"min_dist": 0.3}
        if embedding_kwargs:
            self.embedding_kwargs.update(embedding_kwargs)

    def compute_integration(self, adata):
        """
        Compute scVI integration and store latent representation in adata.obsm.
        """
        logging.info("Integrating data with scVI")
        if not adata.uns["_pretrained_scvi_path"]:
            SCVI.setup_anndata(
                adata,
                batch_key=self.batch_key,
                labels_key=self.labels_key,
                layer="scvi_counts",
            )
            model = SCVI(adata, **self.model_kwargs)
            logging.info("Training scvi offline.")
        else:
            query = adata[adata.obs["_predict_cells"] == "relabel"].copy()
            model = SCVI.load_query_data(query, adata.uns["_pretrained_scvi_path"])
            logging.info("Training scvi online.")

        if adata.uns["_prediction_mode"] == "fast":
            self.train_kwargs["max_epochs"] = 1
            model.train(**self.train_kwargs)
        else:
            if self.max_epochs is None:
                self.max_epochs = min(round((20000 / adata.n_obs) * 200), 200)
            model.train(**self.train_kwargs)
            if adata.uns["_save_path_trained_models"] and adata.uns["_prediction_mode"] == "retrain":
                save_path = os.path.join(adata.uns["_save_path_trained_models"], "scvi")
                adata.uns["_pretrained_scvi_path"] = save_path
                model.save(save_path, save_anndata=False, overwrite=True)

        latent_representation = model.get_latent_representation()
        relabel_idx = adata.obs["_predict_cells"] == "relabel"
        if self.embedding_key not in adata.obsm:
            adata.obsm[self.embedding_key] = np.zeros((adata.n_obs, latent_representation.shape[1]))
        adata.obsm[self.embedding_key][relabel_idx, :] = latent_representation

    def predict(self, adata):
        """
        Predict celltypes using KNN on the scVI embedding.
        """
        logging.info(f'Saving KNN on scvi results to adata.obs["{self.result_key}"]')
        knn_path = os.path.join(adata.uns["_save_path_trained_models"], "scvi_knn_classifier.joblib")

        if adata.uns["_prediction_mode"] == "retrain":
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
            with open(knn_path, "wb") as f:
                joblib.dump(knn, f)
        else:
            with open(knn_path, "rb") as f:
                knn = joblib.load(f)

        embedding = adata[adata.obs["_predict_cells"] == "relabel"].obsm[self.embedding_key]
        knn_pred = knn.predict(embedding)
        if self.result_key not in adata.obs.columns:
            adata.obs[self.result_key] = adata.uns["unknown_celltype_label"]
        adata.obs.loc[
            adata.obs["_predict_cells"] == "relabel", self.result_key
        ] = adata.uns["label_categories"][knn_pred]

        if self.return_probabilities:
            prob_key = f"{self.result_key}_probabilities"
            if prob_key not in adata.obs.columns:
                adata.obs[prob_key] = pd.Series(dtype="float64")
            adata.obs.loc[
                adata.obs["_predict_cells"] == "relabel", prob_key
            ] = np.max(knn.predict_proba(embedding), axis=1)

    def compute_umap(self, adata):
        """
        Compute UMAP embedding from the scVI representation.
        """
        if self.compute_umap_embedding:
            logging.info(f'Saving UMAP of scvi results to adata.obs["{self.umap_key}"]')
            transformer = "rapids" if settings.cuml else None
            sc.pp.neighbors(adata, use_rep=self.embedding_key, transformer=transformer)
            method = "rapids" if settings.cuml else "umap"
            adata_umap = sc.tl.umap(adata, copy=True, method=method, **self.embedding_kwargs)
            adata.obsm[self.umap_key] = adata_umap.obsm["X_umap"]
