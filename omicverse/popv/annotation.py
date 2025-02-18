"""Helper function to execute cell-type annotation and accumulate results."""

from __future__ import annotations

import inspect
import logging
import os
import string
from collections import defaultdict
from dataclasses import dataclass, field

import anndata
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from popv import _utils, algorithms


@dataclass
class AlgorithmsNT:
    """Dataclass to store all available algorithms."""

    OUTDATED_ALGORITHMS: tuple[str, ...] = (
        "Random_Forest",
        "KNN_SCANORAMA",
    )
    FAST_ALGORITHMS: tuple[str, ...] = (
        "KNN_SCVI",
        "SCANVI_POPV",
        "Support_Vector",
        "XGboost",
        "ONCLASS",
        "CELLTYPIST",
    )
    CURRENT_ALGORITHMS: tuple[str, ...] = field(init=False)
    ALL_ALGORITHMS: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        self.CURRENT_ALGORITHMS = tuple(
            i[0]
            for i in inspect.getmembers(algorithms, inspect.isclass)
            if i[0] not in self.OUTDATED_ALGORITHMS and i[0] != "BaseAlgorithm"
        )
        self.ALL_ALGORITHMS = tuple(
            i[0] for i in inspect.getmembers(algorithms, inspect.isclass) if i[0] != "BaseAlgorithm"
        )


algorithms_nt = AlgorithmsNT()


def annotate_data(
    adata: anndata.AnnData,
    methods: list | None = None,
    save_path: str | None = None,
    methods_kwargs: dict | None = None,
) -> None:
    """
    Annotate an AnnData dataset preprocessed by :class:`popv.preprocessing.Process_Query` by using the annotation pipeline.

    Parameters
    ----------
    adata
        AnnData of query and reference cells. AnnData object after running :class:`popv.preprocessing.Process_Query`.
    methods_
        List of methods used for cell-type annotation. Defaults to all algorithms.
    save_path
        Path were annotated query data is saved. Defaults to None and is not saving data.
    methods_kwargs
        Dictionary, where keys are used methods and values contain non-default parameters.
        Default to empty-dictionary.
    """
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    methods = (
        methods
        if isinstance(methods, list)
        else (
            algorithms_nt.ALL_ALGORITHMS
            if methods == "all"
            else (
                algorithms_nt.FAST_ALGORITHMS
                if adata.uns["_prediction_mode"] == "fast"
                else algorithms_nt.CURRENT_ALGORITHMS
            )
        )
    )
    if adata.uns["ref_prediction_keys"] is not None and adata.uns["_prediction_mode"] == "inference":
        if not set(methods).issubset(adata.uns["ref_prediction_keys"]):
            missing_methods = set(methods) - set(adata.uns["ref_prediction_keys"])
            ValueError(
                f"Method {missing_methods} are not present in the reference data."
                "Use relabel_reference_cells=True in Process_Query or remove these methods."
            )
        adata.obs[adata.uns["ref_prediction_keys"]] = adata.obs[adata.uns["ref_prediction_keys"]].astype("object")

    if adata.uns["_cl_obo_file"] is False and "ONCLASS" in methods:
        methods = tuple(method for method in methods if method != "ONCLASS")

    methods_kwargs = methods_kwargs if methods_kwargs else {}

    all_prediction_keys = []
    all_prediction_keys_seen = []

    for method in tqdm(methods):
        current_method = getattr(algorithms, method)(**methods_kwargs.pop(method, {}))
        current_method.compute_integration(adata)
        current_method.predict(adata)
        current_method.compute_umap(adata)
        all_prediction_keys += [current_method.result_key]
        all_prediction_keys_seen += [current_method.seen_result_key]

    # Here we compute the consensus statistics
    logging.info(f"Using predictions {all_prediction_keys} for PopV consensus")
    adata.uns["prediction_keys"] = all_prediction_keys
    adata.uns["prediction_keys_seen"] = all_prediction_keys_seen
    adata.uns["methods"] = list(methods)
    adata.uns["method_kwargs"] = methods_kwargs
    compute_consensus(adata, all_prediction_keys_seen)
    # No ontology prediction if ontology is set to False.
    if adata.uns["_cl_obo_file"] is False:
        adata.obs[["popv_prediction", "popv_prediction_score"]] = adata.obs[
            ["popv_majority_vote_prediction", "popv_majority_vote_score"]
        ]
        adata.obs[["popv_parent"]] = adata.obs[["popv_majority_vote_prediction"]]
    else:
        ontology_vote_onclass(adata, all_prediction_keys)
        ontology_parent_onclass(adata, all_prediction_keys)

    if save_path is not None:
        prediction_save_path = os.path.join(save_path, "predictions.csv")
        adata[adata.obs._dataset == "query"].obs[
            [
                *all_prediction_keys,
                "popv_prediction",
                "popv_prediction_score",
                "popv_majority_vote_prediction",
                "popv_majority_vote_score",
                "popv_parent",
            ]
        ].to_csv(prediction_save_path)

        logging.info(f"Predictions saved to {prediction_save_path}")


def compute_consensus(adata: anndata.AnnData, prediction_keys: list) -> None:
    """
    Compute consensus prediction and statistics between all methods.

    Parameters
    ----------
    adata
        AnnData object
    prediction_keys
        Keys in adata.obs containing predicted cell_types.

    Returns
    -------
    Saves the consensus prediction in adata.obs['popv_majority_vote_prediction']
    Saves the consensus percentage between methods in adata.obs['popv_majority_vote_score']

    """
    consensus_prediction = adata.obs[prediction_keys].apply(_utils.majority_vote, axis=1)
    adata.obs["popv_majority_vote_prediction"] = consensus_prediction

    agreement = adata.obs[prediction_keys].apply(_utils.majority_count, axis=1)
    adata.obs["popv_majority_vote_score"] = agreement.values
    adata.obs["popv_majority_vote_score"] = adata.obs["popv_majority_vote_score"].astype("category")


def ontology_vote_onclass(
    adata: anndata.AnnData,
    prediction_keys: list,
    save_key: str | None = "popv_prediction",
):
    """
    Compute prediction using ontology aggregation method.

    Parameters
    ----------
    adata
        AnnData object
    prediction_keys
        Keys in adata.obs containing predicted cell_types.
    save_key
        Name of the field in adata.obs to store the consensus prediction.

    Returns
    -------
    Saves the consensus prediction in adata.obs[save_key]
    Saves the consensus percentage between methods in adata.obs[save_key + '_score']
    Saves the overlap in original prediction in
    """
    if adata.uns["_prediction_mode"] == "retrain":
        G = _utils.make_ontology_dag(adata.uns["_cl_obo_file"])
        if adata.uns["_save_path_trained_models"] is not None:
            joblib.dump(
                G,
                open(
                    os.path.join(adata.uns["_save_path_trained_models"], "obo_dag.joblib"),
                    "wb",
                ),
            )
    else:
        G = joblib.load(
            open(
                os.path.join(adata.uns["_save_path_trained_models"], "obo_dag.joblib"),
                "rb",
            )
        )

    cell_type_root_to_node = {}
    aggregate_ontology_pred = [None] * adata.n_obs
    depth = {"cell": 0}
    scores = [None] * adata.n_obs
    depths = [None] * adata.n_obs
    onclass_depth = [None] * adata.n_obs
    depth["cell"] = 0

    for ind, cell in enumerate(adata.obs.index):
        score = defaultdict(lambda: 0)
        score["cell"] = 0
        for pred_key in prediction_keys:
            cell_type = adata.obs[pred_key][cell]
            if not pd.isna(cell_type):
                if cell_type in cell_type_root_to_node:
                    root_to_node = cell_type_root_to_node[cell_type]
                else:
                    root_to_node = nx.descendants(G, cell_type)
                    cell_type_root_to_node[cell_type] = root_to_node
                    depth[cell_type] = len(nx.shortest_path(G, cell_type, "cell"))
                    for ancestor_cell_type in root_to_node:
                        depth[ancestor_cell_type] = len(nx.shortest_path(G, ancestor_cell_type, "cell"))
                if pred_key == "popv_onclass_prediction":
                    onclass_depth[ind] = depth[cell_type]
                    for ancestor_cell_type in root_to_node:
                        score[ancestor_cell_type] += 1
                score[cell_type] += 1
        # Find cell-type most present across all classifiers.
        # If tie then deepest in network.
        # If tie then last in alphabet, just to make it consistent across multiple cells.
        celltype_consensus = max(
            score,
            key=lambda k: (
                score[k],
                depth[k],
                26 - string.ascii_lowercase.index(cell_type[0].lower()),
            ),
        )
        aggregate_ontology_pred[ind] = celltype_consensus
        scores[ind] = score[celltype_consensus]
        depths[ind] = depth[celltype_consensus]
    adata.obs[save_key] = aggregate_ontology_pred
    adata.obs[f"{save_key}_score"] = scores
    adata.obs[f"{save_key}_depth"] = depths
    adata.obs[f"{save_key}_onclass_relative_depth"] = np.array(onclass_depth) - adata.obs[f"{save_key}_depth"]
    # Change numeric values to categoricals.
    adata.obs[[f"{save_key}_score", f"{save_key}_depth", f"{save_key}_onclass_relative_depth"]] = adata.obs[
        [f"{save_key}_score", f"{save_key}_depth", f"{save_key}_onclass_relative_depth"]
    ].astype("category")
    return adata


def ontology_parent_onclass(
    adata: anndata.AnnData,
    prediction_keys: list,
    save_key: str = "popv_parent",
    allowed_errors: int = 2,
):
    """
    Compute common parent consensus prediction using ontology accumulation.

    Parameters
    ----------
    adata
        AnnData object
    prediction_keys
        Keys in adata.obs containing predicted cell_types.
    save_key
        Name of the field in adata.obs to store the consensus prediction. Default to 'popv_parent'.
    allowed_errors
        How many misclassifications are allowed to find common ontology ancestor. Defaults to 2.

    Returns
    -------
    Saves the consensus prediction in adata.obs[save_key]
    Saves the consensus percentage between methods in adata.obs[save_key + '_score']
    Saves the overlap in original prediction in
    """
    if adata.uns["_prediction_mode"] == "retrain":
        G = _utils.make_ontology_dag(adata.uns["_cl_obo_file"])
        if adata.uns["_save_path_trained_models"] is not None:
            joblib.dump(
                G,
                open(
                    os.path.join(adata.uns["_save_path_trained_models"], "obo_dag.joblib"),
                    "wb",
                ),
            )
    else:
        G = joblib.load(
            open(
                os.path.join(adata.uns["_save_path_trained_models"], "obo_dag.joblib"),
                "rb",
            )
        )

    cell_type_root_to_node = {}
    aggregate_ontology_pred = []
    depth = {"cell": 0}
    for cell in adata.obs.index:
        score = defaultdict(lambda: 0)
        score_popv = defaultdict(lambda: 0)
        score["cell"] = 0
        for pred_key in prediction_keys:
            cell_type = adata.obs[pred_key][cell]
            if not pd.isna(cell_type):
                if cell_type in cell_type_root_to_node:
                    root_to_node = cell_type_root_to_node[cell_type]
                else:
                    root_to_node = nx.descendants(G, cell_type)
                    cell_type_root_to_node[cell_type] = root_to_node
                    depth[cell_type] = len(nx.shortest_path(G, cell_type, "cell"))
                    for ancestor_cell_type in root_to_node:
                        depth[ancestor_cell_type] = len(nx.shortest_path(G, ancestor_cell_type, "cell"))
                for ancestor_cell_type in list(root_to_node) + [cell_type]:
                    score[ancestor_cell_type] += 1
                score_popv[cell_type] += 1
        score = {key: min(len(prediction_keys) - allowed_errors, value) for key, value in score.items()}

        # Find ancestor most present and deepest across all classifiers.
        # If tie, then highest in original classifier.
        # If tie then last in alphabet, just to make it consistent across multiple cells.
        celltype_consensus = max(
            score,
            key=lambda k: (
                score[k],
                depth[k],
                score_popv[k],
                26 - string.ascii_lowercase.index(cell_type[0].lower()),
            ),
        )
        aggregate_ontology_pred.append(celltype_consensus)
    adata.obs[save_key] = aggregate_ontology_pred
    return adata
