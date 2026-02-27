from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .logging import logger_manager as lm
from .utils import color_difference_rgb, simulate_cvd


def spatial_distance(  # TODO: optimize neighbor calculation
    cell_coordinates,
    cell_labels,
    neighbor_weight: float = 0.5,
    radius: float = 90,  # TODO: comfirm default value
    n_neighbors: int = 16,
    n_cells: int = 3,  # TODO: why n_cells
) -> pd.DataFrame:
    """
    Function to calculate spatial interlacement distance graph for cell clusters, where
    we define the interlacement distance as the number of neighboring cells between two
    cluster.

    Args:
        cell_coordinates: a list like object containing spatial coordinates for each cell.
        cell_labels: a list like object containing cluster labels for each cell.
        cell_weigth: cell weight to calculate cell neighborhood. Defaults to 0.5.
        radius (float, optional): radius used to calculate cell neighborhood. Defaults to 90.
        n_neighbors (int, optional): k for KNN neighbor detection. Defaults to 16.
        n_cells (int, optional): only calculate neighborhood with more than `n_cells`. Defaults to 3.

    Returns:
        pd.DataFrame: a `pandas.DataFrame` with unique cluster names as `index` and `columns`,
            which contains interlacement distance between clusters.
    """

    unique_clusters = np.unique(cell_labels)
    cluster_index = {unique_clusters[i]: i for i in range(len(unique_clusters))}

    # Calculate neighborhoods for all cells
    lm.main_info(f"Calculating cell neighborhood...", indent_level=2)
    tree = KDTree(cell_coordinates, leaf_size=2)
    neighbor_distance_knn, neighbor_index_knn = tree.query(
        cell_coordinates,
        k=n_neighbors,
    )
    # assert len(cell_coordinates) == len(neighbor_distance_knn) == len(neighbor_index_knn)

    # Intersection between knn neighbors and radius neighbors
    lm.main_info(f"Filtering out neighborhood outliers...", indent_level=2)
    neighbor_index_filtered = []
    neighbor_distance_filtered = []
    for i in range(len(cell_coordinates)):
        # filter by radius is equalized to intersection
        neighbor_index_filtered_i = neighbor_index_knn[i][
            neighbor_distance_knn[i] <= radius
        ]
        neighbor_distance_filtered_i = neighbor_distance_knn[i][
            neighbor_distance_knn[i] <= radius
        ]
        # filter banished cell
        if np.sum(cell_labels[neighbor_index_filtered_i] == cell_labels[i]) < n_cells:
            # keep an empty network with only cell i itself
            neighbor_index_filtered_i = np.array([i])
            neighbor_distance_filtered_i = np.array([0])

        neighbor_index_filtered.append(neighbor_index_filtered_i)
        neighbor_distance_filtered.append(neighbor_distance_filtered_i)

    neighbor_index_filtered = np.array(neighbor_index_filtered, dtype=object)
    neighbor_distance_filtered = np.array(neighbor_distance_filtered, dtype=object)

    # Calculate score matrix
    lm.main_info(f"Calculating cluster interlacement score...", indent_level=2)
    score_matrix = np.zeros(
        [len(unique_clusters), len(unique_clusters)], dtype=np.float64
    )
    for cell_i in range(len(neighbor_index_filtered)):
        size_n_i = len(neighbor_index_filtered[cell_i])
        if size_n_i == 0:
            continue
        cell_cluster_i = cluster_index[cell_labels[cell_i]]
        for j in range(1, size_n_i):
            cell_cluster_j = cluster_index[
                cell_labels[neighbor_index_filtered[cell_i][j]]
            ]
            inversed_euclidean = 1 / neighbor_distance_filtered[cell_i][j]
            score_matrix[cell_cluster_i][cell_cluster_j] += (
                inversed_euclidean / size_n_i
            )
    # Keep maximum between score_matrix[x][y] and score_matrix[y][x], set diagonal to zero
    for x in range(len(unique_clusters)):
        for y in range(len(unique_clusters)):
            if x == y:
                score_matrix[x][y] = 0
            else:
                score_matrix[x][y] = max(score_matrix[x][y], score_matrix[y][x])

    cluster_interlace_matrix = score_matrix
    
    lm.main_info(f"Constructing cluster interlacement graph...", indent_level=2)
    cluster_interlace_matrix = pd.DataFrame(cluster_interlace_matrix)
    cluster_interlace_matrix.index = unique_clusters
    cluster_interlace_matrix.columns = unique_clusters

    return cluster_interlace_matrix


def perceptual_distance(
    colors: List[str], 
    colorblind_type: Literal["none", "protanopia", "deuteranopia", "tritanopia", "general"],
) -> pd.DataFrame:
    """
    Function to calculate color perceptual difference matrix.
    See `color_difference_rgb` for details.

    Args:
        colors (List[str]): a list of colors (in hex).

    Returns:
        pd.DataFrame: a `pandas.DataFrame` with unique colors (in hex) as `index` and `columns`,
            which contains perceptual distance between colors.
    """
    difference_matrix = np.zeros([len(colors), len(colors)], dtype=np.float32)

    if colorblind_type!="none":
        lm.main_info(f"Calculating color perceptual distance under {colorblind_type}...", indent_level=2)
        
        if colorblind_type=="general":
            for i in range(len(colors)):
                for j in range(len(colors)):
                    difference_matrix[i][j] += color_difference_rgb(colors[i], colors[j]) / 4
            for cb_t in ["protanopia", "deuteranopia", "tritanopia"]:
                color_cvd = simulate_cvd(colors, colorblind_type=cb_t)
                # Calculate difference between cvd colors
                for i in range(len(color_cvd)):
                    for j in range(len(color_cvd)):
                        difference_matrix[i][j] += color_difference_rgb(color_cvd[i], color_cvd[j]) / 4
        else:
            color_cvd = simulate_cvd(colors, colorblind_type=colorblind_type)
            # Calculate difference between cvd colors
            for i in range(len(color_cvd)):
                for j in range(len(color_cvd)):
                    difference_matrix[i][j] = color_difference_rgb(color_cvd[i], color_cvd[j])
    else:
        # Calculate difference between colors
        lm.main_info(f"Calculating color perceptual distance...", indent_level=2)
        for i in range(len(colors)):
            for j in range(len(colors)):
                difference_matrix[i][j] = color_difference_rgb(colors[i], colors[j])

    #difference_matrix = difference_matrix / np.sum(difference_matrix) * 1000

    min_difference = 1e9
    for i in range(len(colors)):
        for j in range(len(colors)):
            if i==j:
                continue
            min_difference = min(min_difference, difference_matrix[i][j])

    lm.main_info(f"Constructing color distance graph...", indent_level=2)
    difference_matrix = pd.DataFrame(difference_matrix)
    difference_matrix.index = colors
    difference_matrix.columns = colors

    lm.main_info(f"Difference of the most similar pair in the palette is {min_difference:.2f}", indent_level=2)

    return difference_matrix
