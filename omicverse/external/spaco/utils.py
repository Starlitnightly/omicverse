import random
from typing import List, Tuple, Union

import numpy as np
from colormath.color_conversions import Lab_to_XYZ, XYZ_to_Luv, convert_color
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from pyciede2000 import ciede2000
from scipy.spatial import ConvexHull
from skimage import color

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .logging import logger_manager as lm


def matrix_distance(
    matrix_x: np.ndarray,
    matrix_y: np.ndarray,
    metric: Literal["euclidean", "manhattan", "log", "mul_1"] = "manhattan",
) -> float:
    """
    Calculate the distance between two matrices.

    Args:
        matrix_x (np.ndarray): matrix x.
        matrix_y (np.ndarray): matrix y.
        metric (Literal[&quot;euclidean&quot;, &quot;manhattan&quot;, &quot;log&quot;], optional): metric used to calculate distance. Defaults to "manhattan".

    Returns:
        float: the distance between two matrices.
    """

    assert matrix_x.shape == matrix_y.shape, "Matrices should be of same size"

    if metric.lower() == "euclidean":
        return np.linalg.norm(matrix_x - matrix_y)
    elif metric.lower() == "manhattan":
        return np.sum(abs(matrix_x - matrix_y))
    elif metric.lower() == "log":
        return np.sum(matrix_x * np.log(matrix_y))
    elif metric.lower() == "mul_1":
        return 1000 / np.sum(matrix_x*matrix_y)


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """
    Convert hex string to RGB value (0~255).

    Args:
        hex_code (str): hex string representing a RGB color.

    Returns:
        Tuple[int,int,int]: integer values for RGB channels.
    """

    hex_code = hex_code.lstrip("#")
    assert (
        len(hex_code) == 6
    ), "Currently only support vanilla RGB hex color without alpha."

    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_code: Union[Tuple, List]) -> str:
    """
    Convert RGB value (0~255) to hex string.

    Args:
        rgb_code (Union[Tuple, List]): RGB channel value (scaled to 0~255).

    Returns:
        str: color hex string.
    """

    return sRGBColor(
        rgb_code[0], rgb_code[1], rgb_code[2], is_upscaled=True
    ).get_rgb_hex()


def lab_to_hex(lab_code: Union[Tuple, List]) -> str:
    """
    Convert CIE Lab color value to hex string.
    see `Lab color space <http://en.wikipedia.org/wiki/Lab_color_space>` on Wikipedia.

    Args:
        lab_code (Union[Tuple, List]): CIE Lab color value.

    Returns:
        str: color hex string.
    """

    # Convert Lab to sRGB
    lab_object = LabColor(lab_code[0], lab_code[1], lab_code[2], illuminant="d65")
    xyz_object = convert_color(lab_object, XYZColor)
    rgb_object = convert_color(xyz_object, sRGBColor)

    # Clip to legal RGB color
    rgb_object.rgb_r = min(rgb_object.rgb_r, 1)
    rgb_object.rgb_g = min(rgb_object.rgb_g, 1)
    rgb_object.rgb_b = min(rgb_object.rgb_b, 1)
    rgb_object.rgb_r = max(rgb_object.rgb_r, 0)
    rgb_object.rgb_g = max(rgb_object.rgb_g, 0)
    rgb_object.rgb_b = max(rgb_object.rgb_b, 0)

    return rgb_object.get_rgb_hex()


def rgb_to_lms_img(img: np.ndarray):
    """
    Convert RGB image matrix (0~255) to lms image matrix.
    """
    ## Implementation from https://gitlab.com/FloatFlow
    lms_matrix = np.array(
        [[0.3904725 , 0.54990437, 0.00890159],
        [0.07092586, 0.96310739, 0.00135809],
        [0.02314268, 0.12801221, 0.93605194]]
        )
    return np.tensordot(img, lms_matrix, axes=([2], [1]))


def lms_to_rgb_img(img: np.ndarray):
    """
    Convert lms image matrix to RGB image matrix.
    """
    ## Implementation from https://gitlab.com/FloatFlow
    rgb_matrix = np.array(
        [[ 2.85831110e+00, -1.62870796e+00, -2.48186967e-02],
        [-2.10434776e-01,  1.15841493e+00,  3.20463334e-04],
        [-4.18895045e-02, -1.18154333e-01,  1.06888657e+00]]
        )
    rgb_img = np.tensordot(img, rgb_matrix, axes=([2], [1]))
    rgb_img[rgb_img < 0] = 0
    rgb_img[rgb_img > 255] = 255
    return rgb_img


def color_difference_rgb(color_x: str, color_y: str) -> float:
    """
    Calculate the perceptual difference between colors.
    #TODO: Add a Wiki of this calculation

    Args:
        color_x (str): hex string for color x.
        color_y (str): hex string for color y.

    Returns:
        float: the perceptual difference between two colors.
    """

    rgb_x = hex_to_rgb(color_x)
    rgb_y = hex_to_rgb(color_y)

    rgb_r = (rgb_x[0] + rgb_y[0]) / 2

    return (
        (2 + rgb_r / 256) * (rgb_x[0] - rgb_y[0]) ** 2
        + 4 * (rgb_x[1] - rgb_y[1]) ** 2
        + (2 + (255 - rgb_r) / 256) * (rgb_x[2] - rgb_y[2]) ** 2
    ) ** 0.5


def _get_bin_color(bin_number: int) -> Tuple[float, float, float]:
    """
    Revert bin number in `extract_palette` function to Lab values.

    Args:
        bin_number (int): numbered bin color.

    Returns:
        Tuple[float, float, float]: Lab values for the centroid color of this bin.
    """
    l = bin_number // 400 * 5 + 2.5
    bin_number = bin_number % 400

    a = bin_number // 20 * 12.75 + 6.375 - 127
    bin_number = bin_number % 20

    b = bin_number * 12.75 + 6.375 - 127

    return (l, a, b)


def _palette_min_distance(palette: np.ndarray) -> float:
    """
    Calculate the minimal distance within a Lab palette.

    Args:
        palette (np.ndarray): a Lab palette.

    Returns:
        float: the min distance.
    """

    n_palette = palette.shape[0]
    distance_matrix = np.ones(shape=[n_palette, n_palette]) * 1e10
    for i in range(n_palette):
        for j in range(i, n_palette):
            if i == j:
                continue
            else:
                lab_distance = ciede2000(lab1=palette[i], lab2=palette[j])["delta_E_00"]
                distance_matrix[i, j] = lab_distance
                distance_matrix[j, i] = lab_distance

    return distance_matrix.ravel().min()


def _farthest_point_sample(
    color_set: np.ndarray, current_palette: np.ndarray, n_sample=3
) -> List:
    """
    Sample colors from a color set using FPS.

    Args:
        color_set (np.ndarray): color set to sample from.
        current_palette (np.ndarray): reference colors for farthest sample.
        n_sample (int, optional): number of colors to sample. Defaults to 3.

    Returns:
        List: sampled colors.
    """

    n_colors = color_set.shape[0]
    n_palette = current_palette.shape[0]
    sample_colors = []

    distance = np.ones(shape=[n_colors, n_palette]) * 1e10

    for i, color_i in enumerate(color_set):
        for j, palette_j in enumerate(current_palette):
            distance[i, j] = ciede2000(lab1=color_i, lab2=palette_j)["delta_E_00"]

    l_distance: np.ndarray = distance.min(axis=1)
    for k in range(n_sample):
        this_index = l_distance.argmax()
        this_choose = color_set[this_index]
        sample_colors.append(this_index)

        r_distance = (
            np.ones(
                shape=[
                    n_colors,
                ]
            )
            * 1e10
        )
        for i, point in enumerate(color_set):
            r_distance[i] = ciede2000(lab1=this_choose, lab2=point)["delta_E_00"]

        l_distance = np.vstack([l_distance, r_distance]).min(axis=0)

    return sample_colors


def _color_score(
    lab_color: tuple, color_count: int, palette: np.ndarray, wn: float
) -> float:
    """
    Score color replacement.

    """

    l, a, b = lab_color

    # na
    n = color_count

    # da
    e_distance = (
        np.ones(
            shape=[
                palette.shape[0],
            ]
        )
        * 1e10
    )
    for i, point in enumerate(palette):
        e_distance[i] = ciede2000(lab1=(l, a, b), lab2=point)["delta_E_00"]
    dist_e2000 = e_distance.min()

    # Sa
    # Calculate Luv color, drop L
    Lab_2_u_UF = np.frompyfunc(
        lambda l, a, b: XYZ_to_Luv(Lab_to_XYZ(LabColor(l, a, b))).get_value_tuple()[1],
        3,
        1,
    )
    Lab_2_v_UF = np.frompyfunc(
        lambda l, a, b: XYZ_to_Luv(Lab_to_XYZ(LabColor(l, a, b))).get_value_tuple()[2],
        3,
        1,
    )

    color_u = Lab_2_u_UF(l, a, b)
    color_v = Lab_2_v_UF(l, a, b)
    color_uv = np.array([color_u, color_v])

    palette_u = Lab_2_u_UF(palette[:, 0], palette[:, 1], palette[:, 2])
    palette_v = Lab_2_v_UF(palette[:, 0], palette[:, 1], palette[:, 2])
    palette_uv = np.vstack([palette_u, palette_v]).T

    convex_hull_palette = ConvexHull(palette_uv)
    distance_to_convex_hull = np.max(
        np.dot(convex_hull_palette.equations[:, :-1], color_uv.T).T
        + convex_hull_palette.equations[:, -1],
        axis=-1,
    )

    # S(ca)
    ws = 0.15
    return ws * distance_to_convex_hull + wn * n + dist_e2000


def simulate_cvd(
    palette_hex,
    colorblind_type: Literal["protanopia", "deuteranopia", "tritanopia"],
):
    ## Modified from https://gitlab.com/FloatFlow
    palette_rgb_img = np.array([[hex_to_rgb(i) for i in palette_hex]])
    palette_lms_img = rgb_to_lms_img(palette_rgb_img)
    if colorblind_type.lower() in ['protanopia']:
        sim_matrix = np.array([[0, 0.90822864, 0.008192], [0, 1, 0], [0, 0, 1]], dtype=np.float16)
    elif colorblind_type.lower() in ['deuteranopia']:
        sim_matrix =  np.array([[1, 0, 0], [1.10104433,  0, -0.00901975], [0, 0, 1]], dtype=np.float16)
    elif colorblind_type.lower() in ['tritanopia']:
        sim_matrix = np.array([[1, 0, 0], [0, 1, 0], [-0.15773032,  1.19465634, 0]], dtype=np.float16)
    else:
        raise ValueError('{} is an unrecognized colorblindness type.'.format(colorblind_type))
    palette_lms_img = np.tensordot(palette_lms_img, sim_matrix, axes=([2], [1]))
    palette_rgb_img = lms_to_rgb_img(palette_lms_img)
    
    return [rgb_to_hex(i) for i in palette_rgb_img[0]]


def extract_palette(
    reference_image: np.ndarray,
    n_colors: int,
    colorblind_type: Literal["none", "protanopia", "deuteranopia", "tritanopia", "general"],
    l_range: Tuple[float, float] = (20, 85),
    trim_percentile: float = 0.03,
    max_iteration=20,
    verbose=False,
) -> List[str]:
    """
    Extract palette from image. Implemented based on Zheng et al.'s work.

    Reference:
    Zheng, Qian, et al. "Image-guided color mapping for categorical data visualization."
    Computational Visual Media 8.4 (2022): 613-629.

    Args:
        reference_image (np.ndarray): _description_
        n_colors (int): _description_

    Returns:
        List[str]: _description_
    """

    lab_image = color.rgb2lab(reference_image)

    # Make L, A, B into 20 bins each.
    lm.main_info(f"Extracting color bins...", indent_level=3)
    bin_index_l = (lab_image[:, :, 0] / 5).astype(int)
    bin_index_a = (lambda x: (x + 127) / 2.55 / 5)(lab_image[:, :, 1]).astype(int)
    bin_index_b = (lambda x: (x + 127) / 2.55 / 5)(lab_image[:, :, 2]).astype(int)
    bin_index_l[bin_index_l == 20] = 19
    bin_index_a[bin_index_a == 20] = 19
    bin_index_b[bin_index_b == 20] = 19

    # Get bin freqency
    numbered_bin_colors = bin_index_l * 400 + bin_index_a * 20 + bin_index_b
    bin_color_set, bin_color_count = np.unique(
        numbered_bin_colors.ravel(), return_counts=True
    )

    # Truncate color set by l_range, filter out infrequent colors
    filter = (
        (bin_color_set > l_range[0] / 5 * 400)
        & (bin_color_set < l_range[1] / 5 * 400)
        & (bin_color_count > np.quantile(bin_color_count, trim_percentile))
    )
    bin_color_set = bin_color_set[filter]
    bin_color_count = bin_color_count[filter]

    lab_color_set = np.frompyfunc(_get_bin_color, 1, 1)(bin_color_set)

    # Initiate palette, chosen by freqency and distinction
    lm.main_info(f"Initiating palette...", indent_level=3)
    sigma = 80
    palette = {}
    for i in range(n_colors):
        selected_index = bin_color_count.argmax()
        palette[i] = lab_color_set[selected_index]
        # Update freqency to punish chosen colors and similar ones
        for j in range(len(bin_color_count)):
            if j == selected_index:
                bin_color_count[j] = -1
            else:
                lab_distance = ciede2000(
                    lab1=lab_color_set[selected_index], lab2=lab_color_set[i]
                )["delta_E_00"]
                if colorblind_type!="none":
                    if colorblind_type=="general":
                        lab_distance = lab_distance / 4
                        for cb_t in ["protanopia", "deuteranopia", "tritanopia"]:
                            color_cvd = simulate_cvd(
                                np.array([lab_to_hex(lab_color_set[selected_index]),lab_to_hex(lab_color_set[i])]),
                                colorblind_type=cb_t,
                            )
                            lab_distance += color_difference_rgb(color_cvd[0], color_cvd[1]) / 4
                    else:
                        color_cvd = simulate_cvd(
                            np.array([lab_to_hex(lab_color_set[selected_index]),lab_to_hex(lab_color_set[i])]),
                            colorblind_type=colorblind_type,
                        )
                        lab_distance = color_difference_rgb(color_cvd[0], color_cvd[1])
                bin_color_count[j] *= 1 - np.exp(-np.power(lab_distance / sigma, 2))

    palette = np.array(list(palette.values()))

    # Optimize palette distance
    lm.main_info(f"Optimizing extracted palette...", indent_level=3)
    palette_distance = np.zeros(max_iteration)
    for i in range(max_iteration):
        # Update palette distance
        palette_distance[i] = _palette_min_distance(palette)
        if verbose:
            lm.main_info(f"Palette distance: {palette_distance[i]}", indent_level=4)
        if i >= 3 and palette_distance[i - 3 : i].std() < 1e-6:
            break

        # Replace color using starting from ones with less frequency
        for j in range(n_colors - 1, -1, -1):
            sample_index = _farthest_point_sample(
                lab_color_set,
                palette[np.arange(n_colors) != j],
                n_sample=3,
            )
            sample_score = (
                np.ones(
                    shape=[
                        3,
                    ]
                )
                * 1e10
            )
            for k, idx in enumerate(sample_index):
                sample_score[k] = _color_score(
                    lab_color=lab_color_set[idx],
                    color_count=bin_color_count[idx],
                    palette=palette[np.arange(n_colors) != j],
                    wn=reference_image.size[0] * reference_image.size[1] * 0.0003,
                )
            palette[j] = lab_color_set[sample_index[sample_score.argmax()]]

    return list(np.apply_along_axis(lab_to_hex, axis=1, arr=palette))
