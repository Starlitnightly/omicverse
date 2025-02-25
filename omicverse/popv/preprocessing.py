from __future__ import annotations

import json
import logging
import os
import warnings

import anndata
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as scp
import torch
from pynndescent import NNDescent
from scanpy._utils import check_nonnegative_integers

from popv import _utils


class Process_Query:
    """
    Processes the query and reference dataset in preparation for the annotation pipeline.

    Parameters
    ----------
    query_adata
        AnnData of query cells
    ref_adata
        AnnData of reference cells. Can contain only latent spaces if prediction_mode is not 'retrain'.
    ref_labels_key
        Key in obs field of reference AnnData with cell-type information
    ref_batch_key
        List of Keys (or None) in obs field of reference AnnData to
        use as batch covariate
    cl_obo_folder
        Folder containing the cell-type obo for OnClass, ontologies for OnClass and nlp embedding of cell-types.
        Passing a list will use element 1 as obo, element 2 as ontologies and element 3 as nlp embedding.
        Setting it to false will disable ontology use.
    query_batch_key
        Key in obs field of query adata for batch information.
    query_layer_key
        If not None, expects raw_count data in query_layer_key.
    ref_layer_key
        If not None, expects raw_count data in ref_layer_key.
    prediction_mode
        Execution mode of cell-type annotation.
        "retrain": Train all prediction models and saves them to disk if save_path_trained_models is not None.
        "inference": Classify all cells based on pretrained models.
        "fast": Fast inference using only query cells and single epoch in scArches.
    unknown_celltype_label
        Label for cells without a known cell-type.
    n_samples_per_label
        Reference AnnData will be subset to these amount of cells per cell-type to increase speed.
    pretrained_scvi_path
        If path is None, will train scVI from scratch. Else if
        pretrained_path is set and all the genes in the pretrained models are present
        in query adata, will train the scARCHES version of scVI and scANVI, resulting in
        faster training times.
    relabel_reference_cells
        If True, will relabel reference cells with cell-type information from query cells in inference mode.
    save_path_trained_models
        If mode=='retrain' saves models to this directory. Otherwise trained models are expected in this folder.
    hvg
        If Int, subsets data to n highly variable genes according to `sc.pp.highly_variable_genes`
    """

    def __init__(
        self,
        query_adata: anndata.AnnData,
        ref_adata: anndata.AnnData | bool,
        ref_labels_key: str,
        ref_batch_key: str,
        cl_obo_folder: list | str | bool,
        query_batch_key: str | None = None,
        query_layer_key: str | None = None,
        ref_layer_key: str | None = None,
        prediction_mode: str | None = "retrain",
        unknown_celltype_label: str = "unknown",
        n_samples_per_label: int | None = 300,
        save_path_trained_models: str = "tmp/",
        pretrained_scvi_path: str | None = None,
        relabel_reference_cells: bool = False,
        hvg: int | None = 4000,
    ) -> None:
        if ref_adata.X.sum() == 0:  # Minified object
            if prediction_mode == "retrain":
                ValueError("Reference dataset needs to contain gene expression to retrain models.")
            self.process_reference = False
        else:
            self.process_reference = True
            del ref_adata.obsm
            del ref_adata.uns
        del query_adata.obsm
        del query_adata.uns
        del query_adata.raw
        del ref_adata.raw
        if relabel_reference_cells:
            del ref_adata.uns["prediction_keys"]

        if cl_obo_folder is False:
            self.cl_obo_file = False
            self.cl_ontology_file = False
            self.nlp_emb_file = False
        elif isinstance(cl_obo_folder, list):
            self.cl_obo_file = cl_obo_folder[0]
            self.cl_ontology_file = cl_obo_folder[1]
            self.nlp_emb_file = cl_obo_folder[2]
        else:
            self.cl_obo_file = os.path.join(cl_obo_folder, "cl_popv.json")
            self.cl_ontology_file = os.path.join(cl_obo_folder, "cl.ontology")
            self.nlp_emb_file = os.path.join(cl_obo_folder, "cl.ontology.nlp.emb")
        if self.cl_obo_file:
            try:
                with open(self.cl_obo_file):
                    pass
            except FileNotFoundError as err:
                raise FileNotFoundError(f"{self.cl_obo_file} doesn't exist. Check that folder exists.") from err
        self.setup_dict = {
            "ref_labels_key": ref_labels_key,
            "ref_batch_key": ref_batch_key,
            "unknown_celltype_label": unknown_celltype_label,
        }
        self.ref_labels_key = ref_labels_key
        self.unknown_celltype_label = unknown_celltype_label
        self.n_samples_per_label = n_samples_per_label
        self.batch_key = {"reference": ref_batch_key, "query": query_batch_key}

        os.makedirs(save_path_trained_models, exist_ok=True)
        self.save_path_trained_models = save_path_trained_models
        self.pretrained_scvi_path = pretrained_scvi_path

        self.prediction_mode = prediction_mode
        json_path = os.path.join(save_path_trained_models, "preprocessing.json")
        if prediction_mode != "retrain" and not os.path.exists(json_path):
            raise ValueError(f"Configuration {json_path} doesn't exist. Set mode='retrain' to reprocess.")
        if prediction_mode != "retrain":
            with open(json_path) as f:
                data = json.load(f)
            from scvi.model.base._archesmixin import _pad_and_sort_query_anndata

            if not set(data["gene_names"]).issubset(set(query_adata.var_names)):
                _pad_and_sort_query_anndata(
                    query_adata,
                    reference_var_names=pd.Index(data["gene_names"]),
                    inplace=True,
                )
            self.label_categories = data["label_categories"]
            self.genes = data["gene_names"]
        else:
            self.label_categories = None
            self.genes = None

        if self.pretrained_scvi_path or self.prediction_mode != "retrain":
            if self.pretrained_scvi_path is None:
                self.pretrained_scvi_path = os.path.join(self.save_path_trained_models, "scvi")
            pretrained_scvi_genes = torch.load(
                os.path.join(self.pretrained_scvi_path, "model.pt"),
                map_location="cpu",
                weights_only=False,
            )["var_names"]
            if self.genes is not None and not np.array_equal(pretrained_scvi_genes, self.genes):
                warnings.warn(
                    "Pretrained scVI model and query dataset contain different genes. Retrain models or disable scVI.",
                    UserWarning,
                    stacklevel=2,
                )
            elif self.genes is None:
                self.genes = list(pretrained_scvi_genes)
        scanvi_path = os.path.join(self.save_path_trained_models, "scanvi/model.pt")
        if os.path.exists(scanvi_path) and self.prediction_mode != "retrain":
            pretrained_scanvi_genes = torch.load(
                scanvi_path,
                map_location="cpu",
                weights_only=False,
            )["var_names"]
            if not np.array_equal(pretrained_scanvi_genes, self.genes):
                warnings.warn(
                    "Pretrained scANVI model and query dataset contain different genes. Retrain models or disable scANVI.",
                    UserWarning,
                    stacklevel=2,
                )
        onclass_path = os.path.join(self.save_path_trained_models, "OnClass.npz")
        if os.path.exists(onclass_path) and prediction_mode != "retrain":
            onclass_model = np.load(
                onclass_path,
                allow_pickle=True,
            )
            if not np.array_equal(onclass_model["genes"], self.genes):
                warnings.warn(
                    "Pretrained scANVI model and query dataset contain different genes. Retrain models or disable scANVI.",
                    UserWarning,
                    stacklevel=2,
                )
        os.makedirs(self.save_path_trained_models, exist_ok=True)

        if self.genes is not None:
            if not set(self.genes).issubset(set(query_adata.var_names)):
                raise ValueError(
                    "Query dataset misses genes that were used for reference model training. Retrain reference model, set mode='retrain'"
                )
            if hvg is not None:
                raise ValueError("Highly variable gene selection is not available if using trained reference model.")
        else:
            gene_intersection = np.intersect1d(ref_adata.var_names, query_adata.var_names)
            if hvg is not None and len(gene_intersection) > hvg:
                expressed_genes, _ = sc.pp.filter_genes(ref_adata[:, gene_intersection], min_cells=200, inplace=False)
                subset_genes = gene_intersection[expressed_genes]
                if len(subset_genes) > hvg:
                    highly_variable_genes = sc.pp.highly_variable_genes(
                        ref_adata[:, subset_genes].copy(),
                        n_top_genes=hvg,
                        subset=False,
                        flavor="seurat_v3",
                        inplace=False,
                        # batch_key=ref_batch_key,
                        span=1.0,
                    )["highly_variable"]
                    self.genes = list(ref_adata[:, subset_genes].var_names[highly_variable_genes])
                else:
                    self.genes = list(subset_genes)
            else:
                self.genes = list(gene_intersection)

        if self.process_reference:
            self.ref_adata = self._setup_dataset(
                ref_adata,
                key="reference",
                layer_key=ref_layer_key,
                add_meta="_ref",
            )
            self._check_validity_anndata(self.ref_adata, "reference")
        else:
            self.ref_adata = ref_adata
        self.query_adata = self._setup_dataset(
            query_adata,
            key="query",
            add_meta="_query",
            layer_key=query_layer_key,
        )
        self._check_validity_anndata(self.query_adata, "query")

        self._preprocess()

    def _check_validity_anndata(self, adata, key):
        if not check_nonnegative_integers(adata.layers["scvi_counts"]):
            raise ValueError(f"Make sure input {key} adata contains raw_counts")
        if not len(set(adata.var_names)) == len(adata.var_names):
            raise ValueError(f"{key} dataset contains multiple genes with same gene name.")
        if adata.n_obs == 0:
            raise ValueError(f"{key} anndata has no cells.")
        if adata.n_vars == 0:
            raise ValueError(f"{key} anndata has no genes.")

    def _setup_dataset(self, adata, key, add_meta="", layer_key=None):
        if isinstance(self.batch_key[key], list):
            adata.obs["_batch_annotation"] = adata.obs[self.batch_key[key]].astype(str).sum(1).astype("category")
        elif isinstance(self.batch_key[key], str):
            adata.obs["_batch_annotation"] = adata.obs[self.batch_key[key]]
        else:
            adata.obs["_batch_annotation"] = self.unknown_celltype_label

        if layer_key is not None:
            adata.X = scp.csr_matrix(adata.layers[layer_key])
        del adata.layers
        adata = adata[:, self.genes].copy()

        zero_cell_names = adata[np.array(adata.X.sum(1) < 30).flatten()].obs_names
        adata.uns["Filtered_cells"] = list(zero_cell_names)
        sc.pp.filter_cells(adata, min_counts=30, inplace=True)
        if len(zero_cell_names) > 0:
            logging.warning(
                f"The following cells will be excluded from annotation because they have low expression:{zero_cell_names}."
            )

        adata.obs["_batch_annotation"] = adata.obs["_batch_annotation"].astype(str) + add_meta
        adata.obs["_dataset"] = key
        adata.obs["_batch_annotation"] = adata.obs["_batch_annotation"].astype("category")
        adata.layers["scvi_counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        adata.obs["_reference_labels_annotation"] = adata.obs.get(self.ref_labels_key, None)

        # subsample the reference cells used for training certain models
        if key == "reference":
            adata.obs["_labels_annotation"] = adata.obs[self.ref_labels_key]
            if self.n_samples_per_label is not None:
                adata.obs["_ref_subsample"] = False
                subsample_idx = _utils.subsample_dataset(
                    adata,
                    self.ref_labels_key,
                    n_samples_per_label=self.n_samples_per_label,
                    ignore_label=[self.unknown_celltype_label],
                )
                adata.obs.loc[subsample_idx, "_ref_subsample"] = True
            else:
                adata.obs["_ref_subsample"] = True
            adata.layers["scaled"] = adata.X.copy()
            sc.pp.scale(adata, max_value=10, layer="scaled", zero_center=False)
            sc.pp.pca(adata, layer="scaled", zero_center=False)
            reference_features = adata.obsm["X_pca"]
            index = NNDescent(reference_features, n_neighbors=30, metric="euclidean")
            index_path = os.path.join(self.save_path_trained_models, "pynndescent_index.joblib")
            joblib.dump(index, open(index_path, "wb"))
        else:
            adata.obs["_labels_annotation"] = self.unknown_celltype_label
            adata.obs["_ref_subsample"] = False
            adata.layers["scaled"] = adata.X.copy()
            adata.layers["scaled"] /= self.ref_adata.var["std"].values
            adata.layers["scaled"].data = np.clip(adata.layers["scaled"].data, -10, 10)
            adata.layers["scaled"] = adata.layers["scaled"].tocsr()
            adata.obsm["X_pca"] = np.array(adata.layers["scaled"] @ self.ref_adata.varm["PCs"])
        return adata

    def _preprocess(self):
        if self.prediction_mode == "fast":
            self.adata = self.query_adata
        else:
            obsm_dtype = {key: value.dtype for key, value in self.ref_adata.obsm.items()}
            self.adata = anndata.concat(
                (self.ref_adata, self.query_adata),
                axis=0,
                label="_dataset",
                keys=["ref", "query"],
                join="outer",
                fill_value=self.unknown_celltype_label,
                merge="first",
                uns_merge="first",
            )
            self.adata.obsm = {
                key: pd.DataFrame(value).apply(pd.to_numeric, errors="coerce").astype(obsm_dtype[key]).to_numpy()
                for key, value in self.adata.obsm.items()
                if key in obsm_dtype
            }
        del self.query_adata, self.ref_adata
        self.adata.obs["_labels_annotation"] = self.adata.obs["_labels_annotation"].fillna(self.unknown_celltype_label)
        self.adata.obs["_labelled_train_indices"] = np.logical_and(
            self.adata.obs["_dataset"] == "ref",
            self.adata.obs["_labels_annotation"] != self.unknown_celltype_label,
        )
        if self.prediction_mode == "retrain" or "prediction_keys" not in self.adata.uns:
            self.adata.obs["_predict_cells"] = "relabel"
        else:
            self.adata.obs["_predict_cells"] = "reference"
            self.adata.obs.loc[self.adata.obs["_dataset"] == "query", "_predict_cells"] = "relabel"

        batch_count = self.adata.obs["_batch_annotation"].value_counts()
        if batch_count.min() < 11:
            logging.warning(
                f"Batch size of {batch_count.min()} is small. This will lead to issues when using BBKNN. Removing small batches."
            )
            valid_batches = batch_count[batch_count >= 11].index
            self.adata = self.adata[self.adata.obs["_batch_annotation"].isin(valid_batches)].copy()

        self.adata.obs["_labels_annotation"] = self.adata.obs["_labels_annotation"].astype("category")
        # Store values as default for current popv in adata
        self.adata.uns["unknown_celltype_label"] = self.unknown_celltype_label
        if self.prediction_mode == "retrain":
            self.label_categories = list(self.adata.obs["_labels_annotation"].cat.categories)
        self.adata.uns["label_categories"] = np.array(self.label_categories)
        self.adata.uns["_pretrained_scvi_path"] = self.pretrained_scvi_path
        self.adata.uns["_save_path_trained_models"] = self.save_path_trained_models
        self.adata.uns["_prediction_mode"] = self.prediction_mode
        self.adata.uns["_setup_dict"] = self.setup_dict
        self.adata.uns["_cl_obo_file"] = self.cl_obo_file
        self.adata.uns["_cl_ontology_file"] = self.cl_ontology_file
        self.adata.uns["_nlp_emb_file"] = self.nlp_emb_file
        if "prediction_keys" in self.adata.uns:
            self.adata.uns["ref_prediction_keys"] = self.adata.uns["prediction_keys"]
        else:
            self.adata.uns["ref_prediction_keys"] = []
        self.adata.uns["prediction_keys"] = []

        # Store some settings for reference models to output directory when retraining models.
        if self.prediction_mode == "retrain":
            data = {
                "gene_names": self.genes,
                "label_categories": self.label_categories,
            }

            with open(os.path.join(self.save_path_trained_models, "preprocessing.json"), "w") as f:
                json.dump(data, f, indent=4)
