# coding=utf-8

import base64
import json
import os
import zlib
from collections import OrderedDict
from itertools import chain, islice, repeat
from multiprocessing import cpu_count
from operator import attrgetter
from typing import List, Mapping, Optional, Sequence, Union

import loompy as lp
import networkx as nx
import numpy as np
import pandas as pd
#from ctxcore.genesig import Regulon
from sklearn.manifold import TSNE

from ...single._aucell import aucell

from .binarization import binarize


def compress_encode(value):
    """
    Compress using ZLIB algorithm and encode the given value in base64.
    Taken from: https://github.com/aertslab/SCopeLoomPy/blob/5438da52c4bcf48f483a1cf378b1eaa788adefcb/src/scopeloompy/utils/__init__.py#L7
    """
    return base64.b64encode(zlib.compress(value.encode("ascii"))).decode("ascii")


def export2loom(
    ex_mtx: pd.DataFrame,
    regulons,
    out_fname: str,
    cell_annotations: Optional[Mapping[str, str]] = None,
    tree_structure: Sequence[str] = (),
    title: Optional[str] = None,
    nomenclature: str = "Unknown",
    num_workers: int = cpu_count(),
    embeddings: Mapping[str, pd.DataFrame] = {},
    auc_mtx=None,
    auc_thresholds=None,
    compress: bool = False,
):
    """
    Create a loom file for a single cell experiment to be used in SCope.
    :param ex_mtx: The expression matrix (n_cells x n_genes).
    :param regulons: A list of Regulons.
    :param cell_annotations: A dictionary that maps a cell ID to its corresponding cell type annotation.
    :param out_fname: The name of the file to create.
    :param tree_structure: A sequence of strings that defines the category tree structure. Needs to be a sequence of strings with three elements.
    :param title: The title for this loom file. If None than the basename of the filename is used as the title.
    :param nomenclature: The name of the genome.
    :param num_workers: The number of cores to use for AUCell regulon enrichment.
    :param embeddings: A dictionary that maps the name of an embedding to its representation as a pandas DataFrame with two columns: the first
    column is the first component of the projection for each cell followed by the second. The first mapping is the default embedding (use `collections.OrderedDict` to enforce this).
    :param compress: compress metadata (only when using SCope).
    """
    # Information on the general loom file format: http://linnarssonlab.org/loompy/format/index.html
    # Information on the SCope specific alterations: https://github.com/aertslab/SCope/wiki/Data-Format

    if cell_annotations is None:
        cell_annotations = dict(zip(ex_mtx.index, ["-"] * ex_mtx.shape[0]))

    if regulons[0].name.find(" ") == -1:
        print(
            "Regulon name does not seem to be compatible with SCOPE. It should include a space to allow selection of the TF.",
            "\nPlease run: \n regulons = [r.rename(r.name.replace('(+)',' ('+str(len(r))+'g)')) for r in regulons]",
            "\nor:\n regulons = [r.rename(r.name.replace('(',' (')) for r in regulons]",
        )

    # Calculate regulon enrichment per cell using AUCell.
    if auc_mtx is None:
        auc_mtx = aucell(
            ex_mtx, regulons, num_workers=num_workers
        )  # (n_cells x n_regulons)
        auc_mtx = auc_mtx.loc[ex_mtx.index]

    # Binarize matrix for AUC thresholds.
    if auc_thresholds is None:
        _, auc_thresholds = binarize(auc_mtx)

    # Create an embedding based on tSNE.
    # Name of columns should be "_X" and "_Y".
    if len(embeddings) == 0:
        embeddings = {
            "tSNE (default)": pd.DataFrame(
                data=TSNE().fit_transform(auc_mtx),
                index=ex_mtx.index,
                columns=["_X", "_Y"],
            )
        }  # (n_cells, 2)

    id2name = OrderedDict()
    embeddings_X = pd.DataFrame(index=ex_mtx.index)
    embeddings_Y = pd.DataFrame(index=ex_mtx.index)
    for idx, (name, df_embedding) in enumerate(embeddings.items()):
        if len(df_embedding.columns) != 2:
            raise Exception("The embedding should have two columns.")

        embedding_id = idx - 1  # Default embedding must have id == -1 for SCope.
        id2name[embedding_id] = name

        embedding = df_embedding.copy()
        embedding.columns = ["_X", "_Y"]
        embeddings_X = pd.merge(
            embeddings_X,
            embedding["_X"].to_frame().rename(columns={"_X": str(embedding_id)}),
            left_index=True,
            right_index=True,
        )
        embeddings_Y = pd.merge(
            embeddings_Y,
            embedding["_Y"].to_frame().rename(columns={"_Y": str(embedding_id)}),
            left_index=True,
            right_index=True,
        )

    # Calculate the number of genes per cell.
    ngenes = np.count_nonzero(ex_mtx, axis=1)

    # Encode genes in regulons as "binary" membership matrix.
    genes = np.array(ex_mtx.columns)
    n_genes = len(genes)
    n_regulons = len(regulons)
    data = np.zeros(shape=(n_genes, n_regulons), dtype=int)
    for idx, regulon in enumerate(regulons):
        data[:, idx] = np.isin(genes, regulon.genes).astype(int)
    regulon_assignment = pd.DataFrame(
        data=data, index=ex_mtx.columns, columns=list(map(attrgetter("name"), regulons))
    )

    # Encode cell type clusters.
    # The name of the column should match the identifier of the clustering.
    name2idx = dict(map(reversed, enumerate(sorted(set(cell_annotations.values())))))
    clusterings = (
        pd.DataFrame(data=ex_mtx.index.values, index=ex_mtx.index, columns=["0"])
        .replace(cell_annotations)
        .replace(name2idx)
    )

    # Create meta-data structure.
    def create_structure_array(df):
        # Create a numpy structured array
        return np.array(
            [tuple(row) for row in df.values],
            dtype=np.dtype(list(zip(df.columns, df.dtypes))),
        )

    default_embedding = next(iter(embeddings.values())).copy()
    default_embedding.columns = ["_X", "_Y"]
    column_attrs = {
        "CellID": ex_mtx.index.values.astype("str"),
        "nGene": ngenes,
        "Embedding": create_structure_array(default_embedding),
        "RegulonsAUC": create_structure_array(auc_mtx),
        "Clusterings": create_structure_array(clusterings),
        "ClusterID": clusterings.values,
        "Embeddings_X": create_structure_array(embeddings_X),
        "Embeddings_Y": create_structure_array(embeddings_Y),
    }
    row_attrs = {
        "Gene": ex_mtx.columns.values.astype("str"),
        "Regulons": create_structure_array(regulon_assignment),
    }

    def fetch_logo(context):
        for elem in context:
            if elem.endswith(".png"):
                return elem
        return ""

    name2logo = {reg.name: fetch_logo(reg.context) for reg in regulons}
    regulon_thresholds = [
        {
            "regulon": name,
            "defaultThresholdValue": (
                threshold if isinstance(threshold, float) else threshold[0]
            ),
            "defaultThresholdName": "gaussian_mixture_split",
            "allThresholds": {
                "gaussian_mixture_split": (
                    threshold if isinstance(threshold, float) else threshold[0]
                )
            },
            "motifData": name2logo.get(name, ""),
        }
        for name, threshold in auc_thresholds.items()
    ]

    general_attrs = {
        "title": os.path.splitext(os.path.basename(out_fname))[0]
        if title is None
        else title,
        "MetaData": json.dumps(
            {
                "embeddings": [
                    {"id": identifier, "name": name}
                    for identifier, name in id2name.items()
                ],
                "annotations": [{"name": "", "values": []}],
                "clusterings": [
                    {
                        "id": 0,
                        "group": "celltype",
                        "name": "Cell Type",
                        "clusters": [
                            {"id": idx, "description": name}
                            for name, idx in name2idx.items()
                        ],
                    }
                ],
                "regulonThresholds": regulon_thresholds,
            }
        ),
        "Genome": nomenclature,
    }

    # Add tree structure.
    # All three levels need to be supplied
    assert len(tree_structure) <= 3, ""
    general_attrs.update(
        ("SCopeTreeL{}".format(idx + 1), category)
        for idx, category in enumerate(
            list(islice(chain(tree_structure, repeat("")), 3))
        )
    )

    # Compress MetaData global attribute
    if compress:
        general_attrs["MetaData"] = compress_encode(value=general_attrs["MetaData"])

    # Create loom file for use with the SCope tool.
    # The loom file format opted for rows as genes to facilitate growth along the column axis (i.e add more cells)
    # PySCENIC chose a different orientation because of limitation set by the feather format: selectively reading
    # information from disk can only be achieved via column selection. For the ranking databases this is of utmost
    # importance.
    lp.create(
        filename=out_fname,
        layers=ex_mtx.T.values,
        row_attrs=row_attrs,
        col_attrs=column_attrs,
        file_attrs=general_attrs,
    )


# TODO: remove duplication with export2loom function!
def add_scenic_metadata(
    adata: "sc.AnnData",
    auc_mtx: pd.DataFrame,
    regulons = None,
    bin_rep: bool = False,
    copy: bool = False,
) -> "sc.AnnData":
    """
    Add AUCell values and regulon metadata to AnnData object.
    :param adata: The AnnData object.
    :param auc_mtx: The dataframe containing the AUCell values (#observations x #regulons).
    :param bin_rep: Also add binarized version of AUCell values as separate representation. This representation
    is stored as `adata.obsm['X_aucell_bin']`.
    :param copy: Return a copy instead of writing to adata.
    :
    """
    # To avoid dependency with scanpy package the type hinting intentionally uses string literals.
    # In addition, the assert statement to assess runtime type is also commented out.
    # assert isinstance(adata, sc.AnnData)
    assert isinstance(auc_mtx, pd.DataFrame)
    assert len(auc_mtx) == adata.n_obs

    REGULON_SUFFIX_PATTERN = "Regulon({})"

    result = adata.copy() if copy else adata

    # Add AUCell values as new representation (similar to a PCA). This facilitates the usage of
    # AUCell as initial dimensional reduction.
    result.obsm["X_aucell"] = auc_mtx.values.copy()
    if bin_rep:
        bin_mtx, _ = binarize(auc_mtx)
        result.obsm["X_aucell_bin"] = bin_mtx.values

    # Encode genes in regulons as "binary" membership matrix.
    if regulons is not None:
        genes = np.array(adata.var_names)
        data = np.zeros(shape=(adata.n_vars, len(regulons)), dtype=bool)
        for idx, regulon in enumerate(regulons):
            data[:, idx] = np.isin(genes, regulon.genes).astype(bool)
        regulon_assignment = pd.DataFrame(
            data=data,
            index=genes,
            columns=list(
                map(lambda r: REGULON_SUFFIX_PATTERN.format(r.name), regulons)
            ),
        )
        result.var = pd.merge(
            result.var,
            regulon_assignment,
            left_index=True,
            right_index=True,
            how="left",
        )

    # Add additional meta-data/information on the regulons.
    def fetch_logo(context):
        for elem in context:
            if elem.endswith(".png"):
                return elem
        return ""

    result.uns["aucell"] = {
        "regulon_names": auc_mtx.columns.map(
            lambda s: REGULON_SUFFIX_PATTERN.format(s)
        ).values,
        "regulon_motifs": np.array(
            [fetch_logo(reg.context) for reg in regulons]
            if regulons is not None
            else []
        ),
    }

    # Add the AUCell values also as annotations of observations. This way regulon activity can be
    # depicted on cellular scatterplots.
    mtx = auc_mtx.copy()
    mtx.columns = result.uns["aucell"]["regulon_names"]
    result.obs = pd.merge(
        result.obs, mtx, left_index=True, right_index=True, how="left"
    )

    return result


def export_regulons(regulons, fname: str) -> None:
    """
    Export regulons as GraphML.
    :param regulons: The sequence of regulons to export.
    :param fname: The name of the file to create.
    """
    graph = nx.DiGraph()
    for regulon in regulons:
        src_name = regulon.transcription_factor
        graph.add_node(src_name, group="transcription_factor")
        edge_type = "activating" if "activating" in regulon.context else "inhibiting"
        node_type = (
            "activated_target"
            if "activating" in regulon.context
            else "inhibited_target"
        )
        for dst_name, edge_strength in regulon.gene2weight.items():
            graph.add_node(dst_name, group=node_type, **regulon.context)
            graph.add_edge(
                src_name,
                dst_name,
                weight=edge_strength,
                interaction=edge_type,
                **regulon.context
            )
    nx.readwrite.write_graphml(graph, fname)
