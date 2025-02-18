from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix


def get_minified_adata(
    adata,
) -> AnnData:
    """Return a minified AnnData.

    Parameters
    ----------
    adata
        Original AnnData, of which we want to create a minified version.
    """
    adata = adata.copy()
    del adata.raw
    all_zeros = csr_matrix(adata.X.shape)
    X = all_zeros
    layers = {layer: all_zeros.copy() for layer in adata.layers}
    adata.X = X
    adata.layers = layers
    return adata


def create_ontology_nlp_emb(lbl2sent, output_path):
    """
    Create ontology embeddings using NLP and saves them to {output_path}/cl.ontology.nlp.emb.

    Parameters
    ----------
    lbl2sent
        Dictionary with label as key and description as value.
    output_path
        Path to save the embeddings.

    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-mpnet-base-v2")

    sentences = list(lbl2sent.values())
    sentence_embeddings = model.encode(sentences)

    sent2vec = {}
    for label, embedding in zip(lbl2sent.keys(), sentence_embeddings, strict=True):
        sent2vec[label] = embedding

    output_file = os.path.join(output_path, "cl.ontology.nlp.emb")
    with open(output_file, "w") as fout:
        for label, vec in sent2vec.items():
            fout.write(label + "\t" + "\t".join(map(str, vec)) + "\n")


def create_ontology_resources(cl_obo_file):
    """
    Create ontology resources.

    Parameters
    ----------
    cl_obo_file

    Returns
    -------
    g
        Graph object
    id2name
        Dictionary of ontology id to celltype names
    name2id
        Dictionary of celltype names to ontology id
    """
    import json

    # Read the taxrank ontology
    with open(cl_obo_file) as f:
        graph = json.load(f)["graphs"][0]
    output_path = Path(cl_obo_file).parent
    popv_dict = {}
    popv_dict["nodes"] = [entry for entry in graph["nodes"] if entry["type"] == "CLASS" and entry.get("lbl", False)]
    popv_dict["lbl_sentence"] = {
        entry[
            "lbl"
        ]: f"{entry['lbl']}: {entry.get('meta', {}).get('definition', {}).get('val', '')} {' '.join(entry.get('meta', {}).get('comments', []))}"
        for entry in popv_dict["nodes"]
    }
    popv_dict["id_2_lbl"] = {entry["id"]: entry["lbl"] for entry in popv_dict["nodes"]}
    popv_dict["lbl_2_id"] = {entry["lbl"]: entry["id"] for entry in popv_dict["nodes"]}
    popv_dict["edges"] = [
        i
        for i in graph["edges"]
        if i["sub"].split("/")[-1][0:2] == "CL" and i["obj"].split("/")[-1][0:2] == "CL" and i["pred"] == "is_a"
    ]
    popv_dict["ct_edges"] = [
        [popv_dict["id_2_lbl"][i["sub"]], popv_dict["id_2_lbl"][i["obj"]]] for i in popv_dict["edges"]
    ]
    create_ontology_nlp_emb(popv_dict["lbl_sentence"], output_path)

    with open(f"{output_path}/cl_popv.json", "w") as f:
        json.dump(popv_dict, f, indent=4)
    children_edge_celltype_df = pd.DataFrame(popv_dict["ct_edges"])
    children_edge_celltype_df.to_csv(f"{output_path}/cl.ontology", sep="\t", header=False, index=False)


def subsample_dataset(
    adata,
    labels_key,
    n_samples_per_label=100,
    ignore_label=None,
):
    """
    Subsamples dataset per label to n_samples_per_label.

    If a label has fewer than n_samples_per_label examples, then will use
    all the examples. For labels in ignore_label, they won't be included
    in the resulting subsampled dataset.

    Parameters
    ----------
    adata
        AnnData object
    labels_key
        Key in adata.obs for label information
    n_samples_per_label
        Maximum number of samples to use per label
    ignore_label
        List of labels to ignore (not subsample).

    Returns
    -------
    Returns list of obs_names corresponding to subsampled dataset

    """
    sample_idx = []
    labels_counts = dict(adata.obs[labels_key].value_counts())

    logging.info(f"Sampling {n_samples_per_label} cells per label")

    for label in ignore_label:
        labels_counts.pop(label, None)

    for label in labels_counts.keys():
        label_locs = np.where(adata.obs[labels_key] == label)[0]
        if labels_counts[label] < n_samples_per_label:
            sample_idx.append(label_locs)
        else:
            label_subset = np.random.choice(label_locs, n_samples_per_label, replace=False)
            sample_idx.append(label_subset)
    sample_idx = np.concatenate(sample_idx)
    return adata.obs_names[sample_idx]


def check_genes_is_subset(ref_genes, query_genes):
    """
    Check whether query_genes is a subset of ref_genes.

    Parameters
    ----------
    ref_genes
        List of reference genes
    query_genes
        List of query genes

    Returns
    -------
    is_subset
        True if it is a subset, False otherwise.

    """
    if len(set(query_genes)) != len(query_genes):
        logging.warning("Genes in query_dataset are not unique.")

    if set(ref_genes).issubset(set(query_genes)):
        logging.info("All ref genes are in query dataset. Can use pretrained models.")
        is_subset = True
    else:
        logging.info("Not all reference genes are in query dataset. Set 'prediction_mode' to 'retrain'.")
        is_subset = False
    return is_subset


def make_batch_covariate(adata, batch_keys, new_batch_key):
    """
    Combine all the batches in batch_keys into a single batch. Save result into adata.obs['_batch'].

    Parameters
    ----------
    adata
        Anndata object
    batch_keys
        List of keys in adat.obs corresponding to batches
    """
    adata.obs[new_batch_key] = adata.obs[batch_keys].astype(str).sum(1).astype("category")


def calculate_depths(g):
    """
    Calculate depth of each node in a network.

    Parameters
    ----------
    g
        Graph object to compute path_length.

    Returns
    -------
    depths
        Dictionary containing depth for each node

    """
    depths = {}

    for node in g.nodes():
        path = nx.shortest_path_length(g, node)
        if "cell" not in path:
            logging.warning("Celltype not in DAG: ", node)
        else:
            depth = path["cell"]
        depths[node] = depth

    return depths


def make_ontology_dag(cl_obo_file, lowercase=False):
    """
    Construct a graph with all cell-types.

    Parameters
    ----------
    cl_obo_file
        File with all ontology cell-types.

    Returns
    -------
    g
        Graph containing all cell-types
    """
    with open(cl_obo_file) as f:
        cell_ontology = json.load(f)
    g = nx.DiGraph()
    g.add_edges_from(cell_ontology["ct_edges"])

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError(f"Graph is not a Directed Acyclic Graph. {nx.find_cycle(g, orientation='original')}")

    if lowercase:
        mapping = {s: s.lower() for s in list(g.nodes)}
        g = nx.relabel_nodes(g, mapping)
    return g


def majority_vote(x):
    a, b = np.unique(x, return_counts=True)
    return a[np.argmax(b)]


def majority_count(x):
    _, b = np.unique(x, return_counts=True)
    return np.max(b)
