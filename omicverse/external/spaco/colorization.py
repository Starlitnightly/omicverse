from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .color import assign_color
from .distance import spatial_distance
from .logging import logger_manager as lm
from .mapping import cluster_mapping_exp, cluster_mapping_iou


def colorize(
    cell_coordinates,
    cell_labels,
    colorblind_type: Literal["none", "protanopia", "deuteranopia", "tritanopia", "general"],
    palette: List[str] = None,
    image_palette: np.ndarray = None,
    manual_mapping: Dict[Any, str] = None,
    neighbor_weight: float = 0.5,
    radius: float = 90,  # TODO: confirm default value
    n_neighbors: int = 16,  # TODO: confirm default value
    neighbor_kwargs: dict = {},
    mapping_kwargs: dict = {},
    embed_kwargs: dict = {},
) -> Dict[Any, str]:
    """
    Colorize cell clusters based on spatial distribution, so that spatially interlaced and
    spatially neighboring clusters are assigned with more perceptually different colors.

    Spaco2 provides 3 basic color mapping mode:
        1. Optimize the mapping of a pre-defined color palette.
            If `palette` is provided, Spaco2 will calculate a optimized mapping
            between clusters and pre-defined colors. This is for those who already have
            a decent palette either for publication or aesthetic purpose.
            User can fix the color of clusters using `manual_mapping`, manually set clusters
            will not participate in the calculation. Manually mapped colors should be
            different with those in `palette`.
            Note that the final visualization performance is affected by the diversity
            (distinguishability) of the pre-defined palette itself.
        2. Extract colors from image.
            When `palette` is not available or hard to collect, user can input an image
            with desired color theme as an `image_palette`. Spaco2 will try to extract
            discriminate colors while keeping aesthetic theme from the image, and then
            calculate the optimized mapping between clusters and extracted colors.
        3. Automatically generate colors within colorspace.
            Without any given `palette` or `image_palette`, Spaco2 will automatically
            draw colors from colorspace according to the spatial distribution of cell
            clusters. This is the most easy-to-use mode and provides good correlation
            between color difference and cluster neighborship, but does not guarantee
            aesthetic performance.

    Args:
        cell_coordinates: a list like object containing spatial coordinates for each cell.
        cell_labels: a list like object containing cluster labels for each cell.
        palette (List[str], optional): a list of colors (in hex). If given, `image_palette`
            will be ignored. See `Mode 1` above. Defaults to None.
        image_palette (np.ndarray, optional): an image in numpy array format. Should be a
            typical RGB image of shape (x, y, 3). Ignored if `palette` is given. See `Mode 2`
            above. Defaults to None.
        manual_mapping (Dict[Any, str], optional): a `dict` for manual color mapping. Keys are
            cluster names. Values are manually assigned colors (in hex). Only works if `palette`
            or `image_palette` is provided (See `Mode 1` and `Mode 2`). Defaults to None.
        neighbor_weight: neighbor weight to calculate cell neighborhood. Defaults to 0.5.
        radius (float, optional): radius used to calculate cell neighborhood. Defaults to 90.
        n_neighbors (int, optional): k for KNN neighbor detection. Defaults to 16.
        neighbor_kwargs (dict, optional): arguments passed to `spatial_distance` function.
            Defaults to {}.
        mapping_kwargs (dict, optional): arguments passed to `map_graph` function.
            Defaults to {}.
        embed_kwargs (dict, optional): arguments passed to `embed_graph` function.
            Defaults to {}.

    Returns:
        Dict[Any, str]: optimized color mapping for clusters, keys are cluster names, values are hex colors.
    """

    if manual_mapping is not None:
        if palette is None and image_palette is None:
            lm.main_warning(
                f"Palette not provided, ignoring manual mapping to avoid color duplication with auto-generated ones."
            )
            manual_mapping = {}
        else:
            lm.main_info(
                f"Using manual color mapping for {list(manual_mapping.keys())}."
            )
            lm.main_info(
                f"Note that manual colors should be different with any color in `palette` or `image_palette`."
            )
            # Exclude cells with manually given color
            cell_coordinates = cell_coordinates[
                ~np.isin(cell_labels, list(manual_mapping.keys()))
            ]
            cell_labels = cell_labels[
                ~np.isin(cell_labels, list(manual_mapping.keys()))
            ]
    else:
        manual_mapping = {}

    if palette is not None:
        assert len(np.unique(cell_labels)) <= len(
            palette
        ), f"Palette not sufficient for {len(np.unique(cell_labels))} cell types."

    # Construct cluster spatial distance matrix based on cell neighborhood
    lm.main_info(f"Calculating cluster distance graph...")
    cluster_distance_matrix = spatial_distance(
        cell_coordinates=cell_coordinates,
        cell_labels=cell_labels,
        neighbor_weight=neighbor_weight,
        radius=radius,
        n_neighbors=n_neighbors,
        **neighbor_kwargs,
    )

    # Calculate color mapping
    color_mapping = assign_color(
        cluster_distance_matrix=cluster_distance_matrix,
        palette=palette,
        image_palette=image_palette,
        colorblind_type=colorblind_type,
        mapping_kwargs=mapping_kwargs,
        embed_kwargs=embed_kwargs,
    )

    # Restore manual colors, reorder color mapping by cluster names.
    color_mapping = {**color_mapping, **manual_mapping}
    color_mapping = {k: color_mapping[k] for k in sorted(list(color_mapping.keys()))}

    return color_mapping


def colorize_mutiple_slices(
    adatas: List[AnnData],
    cluster_key: str,
    colorblind_type: Literal["none", "protanopia", "deuteranopia", "tritanopia", "general"],
    slice_mapping: Literal["expression", "annotation"] = "annotation",
    mapping_gene_set: List[str] = None,
    spatial_key: str = "spatial",
    palette: Optional[List[str]] = None,
    image_palette=None,
    manual_mapping: Optional[dict] = None,
    neighbor_weight: float = 0.5,
    radius: float = 90,  # TODO: confirm default value
    n_neighbors: int = 16,  # TODO: confirm default value
    neighbor_kwargs: dict = {},
    mapping_kwargs: dict = {},
    embed_kwargs: dict = {},
) -> Dict[Any, str]:
    """
    Colorize cell clusters for mutiple datasets (tissue slices) based on spatial distribution.

    This is a multi-slices version of `spaco2.colorize`. Spaco2 allow colorization for multiple
    related, similar datasets, for example, continuous slices of spatial transcriptomic data,
    and calculate a global optimized color mapping for all dataset.

    Spaco2 provides two ways to map clusters in different slices:
        1. If `slice_mapping` is "expression", Spaco2 will perform mapping based on gene expression
            profile similarity between clusters, using genes in `mapping_gene_set` if available, or
            using highly variable genes by default.
        2. `slice_mapping` can be provided directly with "annotation", which means `cluster_key` in
            each `adata` is comparable by exact values.

    See docString for `spaco2.colorize` for more information.

    Args:
        adatas (List[AnnData]): a list of AnnData.
        cluster_key (str): key for cluster labels in `adata.obs`.
        slice_mapping (Literal[&quot;expression&quot;, &quot;annotation&quot;], optional):
            mode used to map clusters between slices. See docString above.
        mapping_gene_set (List[str], optional): gene set used to map clusters accordding to
            expression profile similarity. Defaults to None.
        spatial_key (str, optional): key for cell spatial coordinates in `adata.obsm`.
            Defaults to "spatial".
        palette (List[str], optional): a list of colors (in hex). If given, `image_palette`
            will be ignored. See `Mode 1` above. Defaults to None.
        image_palette (np.ndarray, optional): an image in numpy array format. Should be a
            typical RGB image of shape (x, y, 3). Ignored if `palette` is given. See `Mode 2`
            above. Defaults to None.
        manual_mapping (Dict[Any, str], optional): a `dict` for manual color mapping. Keys are
            cluster names. Values are manually assigned colors (in hex). Only works if `palette`
            or `image_palette` is provided (See `Mode 1` and `Mode 2`). Defaults to None.
        neighbor_weight: neighbor weight to calculate cell neighborhood. Defaults to 0.5.
        radius (float, optional): radius used to calculate cell neighborhood. Defaults to 90.
        n_neighbors (int, optional): k for KNN neighbor detection. Defaults to 16.
        neighbor_kwargs (dict, optional): arguments passed to `spatial_distance` function.
            Defaults to {}.
        mapping_kwargs (dict, optional): arguments passed to `map_graph` function.
            Defaults to {}.
        embed_kwargs (dict, optional): arguments passed to `embed_graph` function.
            Defaults to {}.

    Returns:
        Dict[Any, str]: optimized color mapping for clusters, keys are cluster names, values are hex colors.
    """

    if slice_mapping == "expression":
        assert (
            len(adatas) == 2
        ), "Currently spaco2 only support expression based mapping between 2 slices."

        # Map clusters between slices using expression similarity
        lm.main_info(f"Mapping clusters between slices using expression similarity...")
        cluster_counts = [len(np.unique(adata.obs[cluster_key])) for adata in adatas]
        base_adata_index = np.argmax(cluster_counts)

        adatas[base_adata_index].obs[cluster_key + "_spaco2"] = (
            adatas[base_adata_index].obs[cluster_key].astype("category")
        )
        for i in range(len(adatas)):
            if i == base_adata_index:
                continue
            lm.main_info(f"Mapping slice {i} to slice {base_adata_index}...")
            adatas[i].obs[cluster_key + "_spaco2"] = cluster_mapping_exp(
                adata=adatas[i],
                adata_reference=adatas[base_adata_index],
                mapping_gene_set=mapping_gene_set,
                cluster_key=cluster_key,
            )
            adatas[i].obs[cluster_key + "_spaco2"] = (
                adatas[i].obs[cluster_key + "_spaco2"].astype("category")
            )
        # After cluster alignment, just do "annotation" mode colorization
        lm.main_info_insert_adata_obs(f"'{cluster_key}_spaco2'", indent_level=2)
        lm.main_info(
            f"Mapped cluster name added to `adata.obs['{cluster_key}_spaco2']`. Result color mapping will base on new cluster name.",
            indent_level=2,
        )
        cluster_key = cluster_key + "_spaco2"

    if manual_mapping is not None:
        if slice_mapping != "annotation":
            lm.main_warning(
                f"Manual color mapping for multiple slices is only supported with 'annotation' mode. Ignoring..."
            )
            manual_mapping = {}
        elif palette is None and image_palette is None:
            lm.main_warning(
                f"Palette not provided, ignoring manual mapping to avoid color duplication with auto-generated ones."
            )
            manual_mapping = {}
        else:
            lm.main_info(
                f"Using manual color mapping for {list(manual_mapping.keys())}."
            )
            lm.main_info(
                f"Note that manual colors should be different with any color in `palette` or `image_palette`."
            )
            # Exclude cells with manually given color
    else:
        manual_mapping = {}

    excluded_clusters = list(manual_mapping.keys())

    # Calculate cluster distance matrix for each slice and merge
    cluster_distance_matrix_merged = pd.DataFrame()
    for i in range(len(adatas)):
        lm.main_info(f"Calculating cluster distance graph for slice {i}... ")
        cluster_distance_matrix_tmp = spatial_distance(
            cell_coordinates=adatas[i].obsm[spatial_key][
                ~adatas[i].obs[cluster_key].isin(excluded_clusters)
            ],
            cell_labels=adatas[i].obs[cluster_key][
                ~adatas[i].obs[cluster_key].isin(excluded_clusters)
            ],
            neighbor_weight=neighbor_weight,
            radius=radius,
            n_neighbors=n_neighbors,
            **neighbor_kwargs,
        )
        cluster_distance_matrix_tmp = cluster_distance_matrix_tmp.stack().reset_index()
        cluster_distance_matrix_tmp.columns = ["v1", "v2", "dist"]
        cluster_distance_matrix_merged = pd.concat(
            [cluster_distance_matrix_merged, cluster_distance_matrix_tmp]
        )

    lm.main_info(f"Merging cluster distance graph... ")
    cluster_distance_matrix_merged = (
        cluster_distance_matrix_merged.groupby(by=["v1", "v2"])
        .agg(sum)
        .unstack()
        .fillna(0)
    )
    cluster_distance_matrix_merged.columns = [
        i[1] for i in cluster_distance_matrix_merged.columns
    ]
    cluster_distance_matrix_merged.index.name = None

    if palette is not None:
        assert len(cluster_distance_matrix_merged) <= len(
            palette
        ), f"Palette not sufficient for {len(cluster_distance_matrix_merged)} cell types."

    # Calculate color mapping
    color_mapping = assign_color(
        cluster_distance_matrix=cluster_distance_matrix_merged,
        palette=palette,
        image_palette=image_palette,
        colorblind_type=colorblind_type,
        mapping_kwargs=mapping_kwargs,
        embed_kwargs=embed_kwargs,
    )

    # Restore manual colors, reorder color mapping by cluster names.
    color_mapping = {**color_mapping, **manual_mapping}
    color_mapping = {k: color_mapping[k] for k in sorted(list(color_mapping.keys()))}

    return color_mapping


def colorize_mutiple_runs(
    adata: AnnData,
    cluster_keys: List[str],
    colorblind_type: Literal["none", "protanopia", "deuteranopia", "tritanopia", "general"],
    spatial_key: str = "spatial",
    palette: Optional[List[str]] = None,
    image_palette=None,
    neighbor_weight: float = 0.5,
    radius: float = 90,  # TODO: confirm default value
    n_neighbors: int = 16,  # TODO: confirm default value
    neighbor_kwargs: dict = {},
    mapping_kwargs: dict = {},
    embed_kwargs: dict = {},
) -> Dict[Any, str]:
    """
    Colorize cell clusters considering mutiple clustering runs (results) based on spatial distribution.

    This is a multi-runs version of `spaco2.colorize`. Spaco2 allow calculating a global optimized
    color mapping among multiple clustering results, while providing a comparable visualization between
    these results.

    See docString for `spaco2.colorize` for more information.

    Args:
        adata (AnnData): AnnData object.
        cluster_keys (List[str]): keys for different clustering runs in `adata.obs`.
        spatial_key (str, optional): key for cell spatial coordinates in `adata.obsm`.
            Defaults to "spatial".
        palette (List[str], optional): a list of colors (in hex). If given, `image_palette`
            will be ignored. See `Mode 1` above. Defaults to None.
        image_palette (np.ndarray, optional): an image in numpy array format. Should be a
            typical RGB image of shape (x, y, 3). Ignored if `palette` is given. See `Mode 2`
            above. Defaults to None.
        neighbor_weight: neighbor weight to calculate cell neighborhood. Defaults to 0.5.
        radius (float, optional): radius used to calculate cell neighborhood. Defaults to 90.
        n_neighbors (int, optional): k for KNN neighbor detection. Defaults to 16.
        neighbor_kwargs (dict, optional): arguments passed to `spatial_distance` function.
            Defaults to {}.
        mapping_kwargs (dict, optional): arguments passed to `map_graph` function.
            Defaults to {}.
        embed_kwargs (dict, optional): arguments passed to `embed_graph` function.
            Defaults to {}.

    Returns:
        Dict[Any, str]: optimized color mapping for clusters, keys are cluster names, values are hex colors.
    """

    # Map clusters between slices using expression similarity
    lm.main_info(f"Mapping clusters between runs...")
    cluster_counts = [
        len(np.unique(adata.obs[cluster_key])) for cluster_key in cluster_keys
    ]
    base_cluster_key_index = np.argmax(cluster_counts)

    adata.obs[cluster_keys[base_cluster_key_index] + "_spaco2"] = adata.obs[
        cluster_keys[base_cluster_key_index]
    ].astype("category")
    lm.main_info_insert_adata_obs(
        f"'{cluster_keys[base_cluster_key_index]}_spaco2'", indent_level=3
    )
    for i in range(len(cluster_keys)):
        if i == base_cluster_key_index:
            continue
        lm.main_info(
            f"Mapping run {i} to run {base_cluster_key_index}...", indent_level=2
        )
        adata.obs[cluster_keys[i] + "_spaco2"] = cluster_mapping_iou(
            cluster_label_mapping=adata.obs[cluster_keys[i]].to_list(),
            cluster_label_reference=adata.obs[
                cluster_keys[base_cluster_key_index]
            ].to_list(),
        )
        adata.obs[cluster_keys[i] + "_spaco2"] = adata.obs[
            cluster_keys[i] + "_spaco2"
        ].astype("category")
        lm.main_info_insert_adata_obs(f"'{cluster_keys[i]}_spaco2'", indent_level=3)

    lm.main_info(
        f"Mapped cluster name added to `adata.obs['***_spaco2']`. Result color mapping will base on new cluster name.",
        indent_level=2,
    )
    cluster_keys = [cluster_key + "_spaco2" for cluster_key in cluster_keys]

    # Calculate cluster distance matrix for each slice and merge
    cluster_distance_matrix_merged = pd.DataFrame()
    for i in range(len(cluster_keys)):
        lm.main_info(f"Calculating cluster distance graph for run {i}... ")
        cluster_distance_matrix_tmp = spatial_distance(
            cell_coordinates=adata.obsm[spatial_key],
            cell_labels=adata.obs[cluster_keys[i]],
            neighbor_weight=neighbor_weight,
            radius=radius,
            n_neighbors=n_neighbors,
            **neighbor_kwargs,
        )
        cluster_distance_matrix_tmp = cluster_distance_matrix_tmp.stack().reset_index()
        cluster_distance_matrix_tmp.columns = ["v1", "v2", "dist"]
        cluster_distance_matrix_merged = pd.concat(
            [cluster_distance_matrix_merged, cluster_distance_matrix_tmp]
        )

    lm.main_info(f"Merging cluster distance graph... ")
    cluster_distance_matrix_merged = (
        cluster_distance_matrix_merged.groupby(by=["v1", "v2"])
        .agg(sum)
        .unstack()
        .fillna(0)
    )
    cluster_distance_matrix_merged.columns = [
        i[1] for i in cluster_distance_matrix_merged.columns
    ]
    cluster_distance_matrix_merged.index.name = None

    if palette is not None:
        assert len(cluster_distance_matrix_merged) <= len(
            palette
        ), f"Palette not sufficient for {len(cluster_distance_matrix_merged)} cell types."

    # Calculate color mapping
    color_mapping = assign_color(
        cluster_distance_matrix=cluster_distance_matrix_merged,
        palette=palette,
        image_palette=image_palette,
        colorblind_type=colorblind_type,
        mapping_kwargs=mapping_kwargs,
        embed_kwargs=embed_kwargs,
    )

    # Restore manual colors, reorder color mapping by cluster names.
    color_mapping = {k: color_mapping[k] for k in sorted(list(color_mapping.keys()))}

    return color_mapping
