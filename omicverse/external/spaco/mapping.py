from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import random
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .logging import logger_manager as lm
from .utils import lab_to_hex, matrix_distance


def map_graph(
    cluster_distance: pd.DataFrame,
    color_distance: pd.DataFrame,
    random_seed: int = 123,
    distance_metric: Literal["euclidean", "manhattan", "log", "mul_1"] = "mul_1",
    random_max_iter: int = 5000,
    verbose: bool = False,
) -> Dict[Any, str]:
    """
    Function to map the vertices between two graph.

    Args:
        cluster_distance (pd.DataFrame): a `pandas.DataFrame` with unique cluster names as `index` and `columns`,
            which contains a distance adjacent matrix for clusters, representing the dissimilarity between clusters.
        color_distance (pd.DataFrame): a `pandas.DataFrame` with unique colors (in hex) as `index` and `columns`,
            which contains a distance adjacent matrix for colors, representing the perceptual difference between colors.
        random_seed (int): random seed for heuristic solver.
        distance_metric (Literal[&quot;euclidean&quot;, &quot;manhattan&quot;, &quot;log&quot;], optional): metric used for matrix mapping. Defaults to "manhattan".
        verbose (bool, optional): output info.

    Returns:
        Dict[Any, str]: optimized color mapping for clusters, keys are cluster names, values are hex colors.
    """

    assert (
        cluster_distance.shape == color_distance.shape
    ), "Clusters and colors are not in the same size."

    random.seed(random_seed)
    
    color_shuffle_index = list(range(color_distance.shape[0]))
    shuffle_distance = 1e9
    
    for iseed in range(random_max_iter):
        color_shuffle_index_tmp = color_shuffle_index.copy()
        np.random.seed(iseed)
        np.random.shuffle(color_shuffle_index_tmp)
        color_distance_shuffle = np.transpose(
            color_distance.to_numpy()[color_shuffle_index_tmp]
        )[color_shuffle_index_tmp]
        shuffle_distance_tmp = matrix_distance(
            matrix_x=cluster_distance.to_numpy(),
            matrix_y=color_distance_shuffle,
            metric=distance_metric,
        )
        if (shuffle_distance_tmp < shuffle_distance):
            shuffle_distance = shuffle_distance_tmp.copy()
            color_shuffle_index = color_shuffle_index_tmp.copy()
    
    for siter in range(random_max_iter*2):
        color_shuffle_index_tmp = color_shuffle_index.copy()
        idx = random.sample(range(len(color_shuffle_index_tmp)), 2)
        tmp = color_shuffle_index_tmp[idx[1]]
        color_shuffle_index_tmp[idx[1]] = color_shuffle_index_tmp[idx[0]]
        color_shuffle_index_tmp[idx[0]] = tmp
        color_distance_shuffle = np.transpose(
            color_distance.to_numpy()[color_shuffle_index_tmp]
        )[color_shuffle_index_tmp]
        shuffle_distance_tmp = matrix_distance(
            matrix_x=cluster_distance.to_numpy(),
            matrix_y=color_distance_shuffle,
            metric=distance_metric,
        )
        if (shuffle_distance_tmp < shuffle_distance):
            shuffle_distance = shuffle_distance_tmp.copy()
            color_shuffle_index = color_shuffle_index_tmp.copy()

    
    # Return color mapping dictionary, sorted by keys
    color_mapping = dict(
        zip(
            cluster_distance.index,
            color_distance.index[color_shuffle_index],
        )
    )
    color_mapping = {k: color_mapping[k] for k in sorted(list(color_mapping.keys()))}

    return color_mapping


def embed_graph(
    cluster_distance: pd.DataFrame,
    transformation: Literal["mds", "umap"] = "umap",
    l_range: Tuple[float, float] = (30, 80),
    log_colors: bool = False,
    trim_fraction: float = 0.0125,
) -> Dict[Any, str]:
    """
    Function to embed the cluster distance graph into chosen colorspace, while keeping distance
    relationship. Currently only supports CIE Lab space. Proper colors are selected within whole
    colorspace based on the embedding of each cluster.

    Args:
        cluster_distance (pd.DataFrame):  a `pandas.DataFrame` with unique cluster names as `index` and `columns`,
            which contains a distance adjacent matrix for clusters, representing the dissimilarity between clusters.
        transformation (Literal[&quot;mds&quot;, &quot;umap&quot;], optional): method used for graph embedding. Defaults to "umap".
        l_range (Tuple[float, float], optional): value range for L channel in LAB colorspace. Defaults to (10,90).
        log_colors (bool, optional): whether to perform log-transformation for color embeddings. Defaults to False.
        trim_fraction (float, optional): quantile for trimming (value clipping). Defaults to 0.0125.

    Returns:
        Dict[Any, str]: auto-generated color mapping for clusters, keys are cluster names, values are hex colors.
    """

    # Embed clusters into 3-dimensional space
    lm.main_info(f"Calculating cluster embedding...", indent_level=3)
    if transformation == "mds":
        from sklearn.manifold import MDS

        model = MDS(
            n_components=3,
            dissimilarity="precomputed",
            random_state=123,
        )
    elif transformation == "umap":
        from umap import UMAP

        model = UMAP(
            n_components=3,
            metric="precomputed",
            random_state=123,
        )
    embedding = model.fit_transform(cluster_distance)

    # Rescale embedding to CIE Lab colorspace
    lm.main_info(f"Rescaling embedding to CIE Lab colorspace...", indent_level=3)
    embedding -= np.quantile(embedding, trim_fraction, axis=0)
    embedding[embedding < 0] = 0
    embedding /= np.quantile(embedding, 1 - trim_fraction)
    embedding[embedding > 1] = 1

    if log_colors:
        embedding = np.log10(embedding + max(np.quantile(embedding, 0.05), 1e-3))
        embedding -= np.min(embedding, axis=0)
        embedding /= np.max(embedding, axis=0)

    embedding[:, 0] *= l_range[1] - l_range[0]
    embedding[:, 0] += l_range[0]
    embedding[:, 1:3] -= 0.5
    embedding[:, 1:3] *= 200

    lm.main_info(f"Optimizing cluster color mapping...")
    color_mapping = dict(
        zip(
            cluster_distance.index,
            np.apply_along_axis(lab_to_hex, axis=1, arr=embedding),
        )
    )

    return color_mapping


def cluster_mapping_exp(
    adata: AnnData,
    adata_reference: AnnData,
    cluster_key: str,
    mapping_gene_set: List[str] = None,
) -> List:

    #assert False, "under development."  # TODO: implement here
    adata.X = adata.layers['normalize']

    adata_sub = adata[:, mapping_gene_set]

    adata_sub.X.shape

    ct = adata_sub.obs[cluster_key].unique()
    ct = list(ct.categories)

    adf = pd.DataFrame(columns=ct, index=adata_sub.var.index) 

    for i in ct:
        val = adata_sub[adata_sub.obs[cluster_key] == i].X.mean(axis=0)
        adf.loc[:,str(i)] = val.tolist()[0]
    
    
    """
    calculate mean expression percluster 2
    """
    adata_reference.X = adata_reference.layers['normalize']

    adata_reference_sub = adata_reference[:, mapping_gene_set]

    adata_reference_sub.X.shape

    rct = adata_reference_sub.obs[cluster_key].unique()
    rct = list(rct.categories)

    rdf = pd.DataFrame(columns=rct, index=adata_reference_sub.var.index) 

    for i in rct:
        val = adata_reference_sub[adata_reference_sub.obs[cluster_key] == i].X.mean(axis=0)
        rdf.loc[:,str(i)] = val.tolist()[0]
    
    """
    calculate correlation matrix
    """
    
    dfcor = pd.DataFrame(index=rdf.columns, columns=adf.columns)

    for m in adf.columns:
        for n in rdf.columns:
            x = adf.loc[:, m]
            y = rdf.loc[:, n]
            dfcor.loc[n,m] = scipy.stats.pearsonr(x,y)[0]
    
    cor_mat = dfcor.values
    
    
    cluster_label_mapping = adata.obs[cluster_key].to_list()
    cluster_label_reference = adata_reference.obs[cluster_key].to_list()
    mapping_label_list = adf.columns
    reference_label_list = rdf.columns
    

    # Greedy mapping to the largest similarity of each label
    relationship = {}
    index_not_mapped = np.ones(len(mapping_label_list)).astype(bool)
    cor_mat_backup = cor_mat.copy()
    while np.sum(cor_mat) != 0:
        reference_index, mapping_index = np.unravel_index(
            cor_mat.argmax(), cor_mat.shape
        )
        relationship[mapping_label_list[mapping_index]] = reference_label_list[
            reference_index
        ]
        # Clear mapped labels to avoid duplicated mapping
        cor_mat[reference_index, :] = 0
        cor_mat[:, mapping_index] = 0
        index_not_mapped[mapping_index] = False

    # Check if every label is mapped to a reference
    duplicate_map_label = np.ones(len(reference_label_list))
    for mapping_index, is_force_map in enumerate(index_not_mapped):
        if is_force_map:
            reference_index = cor_mat_backup[:, mapping_index].argmax()
            relationship[mapping_label_list[mapping_index]] = (
                reference_label_list[reference_index]
                + ".%d" % duplicate_map_label[reference_index]
            )
            duplicate_map_label[reference_index] += 1
            # Log: warning
            lm.main_warning(
                f"Mapping between cluster {mapping_label_list[mapping_index]} and cluster "
                + f"{reference_label_list[reference_index]} is not bijective.",
                indent_level=3,
            )
    mapped_cluster_label = np.frompyfunc(lambda x: relationship[x], 1, 1)(
        cluster_label_mapping
    )
    return mapped_cluster_label.tolist()


def cluster_mapping_iou(
    cluster_label_mapping: List,
    cluster_label_reference: List,
) -> List:
    """
    Function to map clusters between different clustering results based
    on cluster overlap (IOU).

    Args:
        cluster_label_mapping (List): cluster result for cells to be mapped.
        cluster_label_reference (List): cluster result for cells to be
            mapped to.

    Returns:
        List: mapping result of `cluster_label_mapping`.
    """

    def iou(i, j):
        I = np.sum((cluster_label_mapping == i) & (cluster_label_reference == j))
        U = np.sum((cluster_label_mapping == i) | (cluster_label_reference == j))
        return I / U

    # Cells should be identical between different runs.
    assert len(cluster_label_mapping) == len(cluster_label_reference)

    ufunc_iou = np.frompyfunc(iou, 2, 1)
    cluster_label_mapping = np.array(cluster_label_mapping).astype(str)
    cluster_label_reference = np.array(cluster_label_reference).astype(str)

    # Reference label types should be more than mapping label types
    mapping_label_list = np.unique(cluster_label_mapping)
    reference_label_list = np.unique(cluster_label_reference)
    assert len(mapping_label_list) <= len(reference_label_list)

    # Grid label lists for vectorized calculation
    mapping_vector_column = mapping_label_list.reshape(
        1, len(mapping_label_list)
    ).repeat(len(reference_label_list), axis=0)
    reference_vector_index = reference_label_list.reshape(
        len(reference_label_list), 1
    ).repeat(len(mapping_label_list), axis=1)
    # Calculate IOU matrix
    iou_matrix = ufunc_iou(mapping_vector_column, reference_vector_index).astype(
        np.float64
    )

    # Greedy mapping to the largest IOU of each label
    relationship = {}
    index_not_mapped = np.ones(len(mapping_label_list)).astype(bool)
    iou_matrix_backup = iou_matrix.copy()
    while np.sum(iou_matrix) != 0:
        reference_index, mapping_index = np.unravel_index(
            iou_matrix.argmax(), iou_matrix.shape
        )
        relationship[mapping_label_list[mapping_index]] = reference_label_list[
            reference_index
        ]
        # Clear mapped labels to avoid duplicated mapping
        iou_matrix[reference_index, :] = 0
        iou_matrix[:, mapping_index] = 0
        index_not_mapped[mapping_index] = False

    # Check if every label is mapped to a reference
    duplicate_map_label = np.ones(len(reference_label_list))
    for mapping_index, is_force_map in enumerate(index_not_mapped):
        if is_force_map:
            reference_index = iou_matrix_backup[:, mapping_index].argmax()
            relationship[mapping_label_list[mapping_index]] = (
                reference_label_list[reference_index]
                + ".%d" % duplicate_map_label[reference_index]
            )
            duplicate_map_label[reference_index] += 1
            # Log: warning
            lm.main_warning(
                f"Mapping between cluster {mapping_label_list[mapping_index]} and cluster "
                + f"{reference_label_list[reference_index]} is not bijective.",
                indent_level=3,
            )
    mapped_cluster_label = np.frompyfunc(lambda x: relationship[x], 1, 1)(
        cluster_label_mapping
    )
    return mapped_cluster_label.tolist()
