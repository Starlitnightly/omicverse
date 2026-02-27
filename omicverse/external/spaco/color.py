from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd

from .distance import perceptual_distance
from .logging import logger_manager as lm
from .mapping import embed_graph, map_graph
from .utils import extract_palette


def assign_color(
    cluster_distance_matrix: pd.DataFrame,
    colorblind_type: Literal["none", "protanopia", "deuteranopia", "tritanopia", "general"],
    palette: List[str] = None,
    image_palette: np.ndarray = None,
    mapping_kwargs: dict = {},
    embed_kwargs: dict = {},
) -> Dict[Any, str]:
    """
    Core color mapping function for Spaco2.

    Spaco2 provides 3 basic color mapping mode in this function:
        1. Optimize the mapping of a pre-defined color palette.
            If `palette` is provided, this function will calculate a optimized mapping
            between clusters and pre-defined colors. This is for those who already have
            a decent palette either for publication or aesthetic purpose.
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
        cluster_distance (pd.DataFrame): a `pandas.DataFrame` with unique cluster names as
            `index` and `columns`, which contains a distance adjacent matrix for clusters,
            representing the dissimilarity between clusters.
        palette (List[str], optional): a list of colors (in hex). If given, `image_palette`
            will be ignored. See `Mode 1` above. Defaults to None.
        image_palette (np.ndarray, optional): an image in numpy array format. Should be a
            typical RGB image of shape (x, y, 3). Ignored if `palette` is given. See `Mode 2`
            above. Defaults to None.
        mapping_kwargs (dict, optional): arguments passed to `map_graph` function.
            Defaults to {}.
        embed_kwargs (dict, optional): arguments passed to `embed_graph` function.
            Defaults to {}.

    Returns:
        Dict[Any, str]: optimized color mapping for clusters, keys are cluster names, values are hex colors.
    """

    # Auto-generate a palette if not provided
    if palette is None:
        lm.main_info(f"`palette` not provided.")
        if image_palette is None:
            # Mode 3
            lm.main_info(
                f"Auto-generating colors from CIE Lab colorspace...", indent_level=2
            )
            color_mapping = embed_graph(
                cluster_distance=cluster_distance_matrix,
                **embed_kwargs,
            )

            color_mapping = {
                k: color_mapping[k] for k in sorted(list(color_mapping.keys()))
            }
            return color_mapping
        else:
            # Mode 2
            lm.main_info(f"Using `image palette`...", indent_level=2)
            lm.main_info(
                f"Drawing appropriate colors from provided image...", indent_level=2
            )
            palette = extract_palette(
                reference_image=image_palette, n_colors=len(cluster_distance_matrix), colorblind_type=colorblind_type,
            )

    # Construct color perceptual distance matrix
    lm.main_info(f"Calculating color distance graph...")
    color_distance_matrix = perceptual_distance(colors=palette, colorblind_type=colorblind_type,) + 1e-5

    # Map clusters and colors via graph
    lm.main_info(f"Optimizing color mapping...")
    color_mapping = map_graph(
        cluster_distance=cluster_distance_matrix,
        color_distance=color_distance_matrix,
        **mapping_kwargs,
    )

    return color_mapping
