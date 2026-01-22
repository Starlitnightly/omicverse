#copy from squidpy.pl._spatial.py
#remove some functions that are not needed


from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from copy import copy
from functools import partial
from numbers import Number
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, TypeAlias, Union


import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import colors, patheffects, rcParams
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import Collection, PatchCollection
from matplotlib.colors import (
    ColorConverter,
    Colormap,
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon, Rectangle

from pandas import CategoricalDtype
from scanpy import logging as logg
from scanpy._settings import settings as sc_settings
from scanpy.plotting._tools.scatterplots import _add_categorical_legend

def _get_vector_friendly():
    """Get the vector_friendly setting from omicverse plot settings."""
    try:
        from ..utils._plot import _vector_friendly
        return _vector_friendly
    except ImportError:
        try:
            return sc_settings._vector_friendly
        except AttributeError:
            return True  # Default fallback


import itertools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Palette_t: TypeAlias = str | ListedColormap | None
_Normalize: TypeAlias = Normalize | Sequence[Normalize]
_SeqStr: TypeAlias = str | Sequence[str]
_SeqFloat: TypeAlias = float | Sequence[float]
_CoordTuple: TypeAlias = tuple[int, int, int, int]
_FontWeight: TypeAlias = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize: TypeAlias = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]


# named tuples
class FigParams(NamedTuple):
    """Figure params."""

    fig: Figure
    ax: Axes
    axs: Sequence[Axes] | None
    iter_panels: tuple[Sequence[Any], Sequence[Any]]
    title: _SeqStr | None
    ax_labels: Sequence[str]
    frameon: bool | None


class CmapParams(NamedTuple):
    """Cmap params."""

    cmap: Colormap
    img_cmap: Colormap
    norm: Normalize


class OutlineParams(NamedTuple):
    """Outline params."""

    outline: bool
    gap_size: float
    gap_color: np.ndarray | str
    bg_size: float
    bg_color: np.ndarray | str


class ScalebarParams(NamedTuple):
    """Scalebar params."""

    scalebar_dx: Sequence[float] | None
    scalebar_units: _SeqStr | None


class ColorParams(NamedTuple):
    """Color params."""

    shape: str | None
    color: Sequence[str | None]
    groups: Sequence[str] | None
    alpha: float
    img_alpha: float
    use_raw: bool


class SpatialParams(NamedTuple):
    """Color params."""

    library_id: Sequence[str]
    scale_factor: Sequence[float]
    size: Sequence[float]
    img: Sequence[np.ndarray] | tuple[None, ...]
    segment: Sequence[np.ndarray] | tuple[None, ...]
    cell_id: Sequence[np.ndarray] | tuple[None, ...]


to_hex = partial(colors.to_hex, keep_alpha=True)


def _get_library_id(
    adata: AnnData,
    shape: str | None,
    spatial_key: str = "spatial",
    library_id: Sequence[str] | None = None,
    library_key: str | None = None,
) -> Sequence[str]:
    r"""
    Get library IDs for spatial data processing.
    
    Args:
        adata: AnnData object
        shape: str, optional (default=None)
            Shape parameter for spatial data
        spatial_key: str, optional (default="spatial")
            Key for spatial information in adata.uns
        library_id: Sequence[str], optional (default=None)
            Specific library IDs to use
        library_key: str, optional (default=None)
            Key in adata.obs for library information
    
    Returns:
        Sequence[str]: List of library IDs
    """
    from squidpy._constants._pkg_constants import Key
    from squidpy.pl._utils import _assert_value_in_obs
    
    if shape is not None:
        library_id = Key.uns.library_id(adata, spatial_key, library_id, return_all=True)
        if library_id is None:
            raise ValueError(f"Could not fetch `library_id`, check that `spatial_key: {spatial_key}` is correct.")
        return library_id
    if library_key is not None:
        if library_key not in adata.obs:
            raise KeyError(f"`library_key: {library_key}` not in `adata.obs`.")
        if library_id is None:
            library_id = adata.obs[library_key].cat.categories.tolist()
        _assert_value_in_obs(adata, key=library_key, val=library_id)
        if isinstance(library_id, str):
            library_id = [library_id]
        return library_id
    if library_id is None:
        logg.warning("Please specify a valid `library_id` or set it permanently in `adata.uns['spatial']`")
        library_id = [""]  # dummy value to maintain logic of number of plots (nplots=library_id*color)
    elif isinstance(library_id, list):  # get library_id from arg
        pass
    elif isinstance(library_id, str):
        library_id = [library_id]
    else:
        raise TypeError(f"Invalid `library_id`: {library_id}.")
    return library_id


def _get_image(
    adata: AnnData,
    library_id: Sequence[str],
    spatial_key: str = "spatial",
    img: bool | Sequence[np.ndarray] | None = None,
    img_res_key: str | None = None,
    img_channel: int | list[int] | None = None,
    img_cmap: Colormap | str | None = None,
) -> Sequence[np.ndarray] | tuple[None, ...]:
    from squidpy._constants._pkg_constants import Key
    from squidpy.pl._utils import _to_grayscale
    import dask.array as da

    if isinstance(img, list | np.ndarray | da.Array):
        img = _get_list(img, _type=(np.ndarray, da.Array), ref_len=len(library_id), name="img")
    else:
        image_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.image_key, library_id)
        if img_res_key is None:
            img_res_key = _get_unique_map(image_mapping)[0]
        elif img_res_key not in _get_unique_map(image_mapping):
            raise KeyError(
                f"Image key: `{img_res_key}` does not exist. Available image keys: `{image_mapping.values()}`"
            )
        img = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][img_res_key] for i in library_id]

    if img_channel is None:
        img = [im[..., :3] for im in img]
    elif isinstance(img_channel, int):
        img = [im[..., [img_channel]] for im in img]
    elif isinstance(img_channel, list):
        img = [im[..., img_channel] for im in img]
    else:
        raise TypeError(f"Expected image channel to be either `int` or `None`, found `{type(img_channel).__name__}`.")

    if img_cmap == "gray":
        img = [_to_grayscale(im) for im in img]
    return img


def _get_segment(
    adata: AnnData,
    library_id: Sequence[str],
    seg_cell_id: str | None = None,
    library_key: str | None = None,
    seg: Sequence[np.ndarray] | bool | None = None,
    seg_key: str | None = None,
) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | tuple[tuple[None, ...], tuple[None, ...]]:
    from squidpy._constants._pkg_constants import Key
    import dask.array as da
    
    if seg_cell_id not in adata.obs:
        raise ValueError(f"Cell id `{seg_cell_id!r}` not found in `adata.obs`.")
    cell_id_vec = adata.obs[seg_cell_id].values

    if library_key not in adata.obs:
        raise ValueError(f"Library key `{library_key}` not found in `adata.obs`.")
    if not np.issubdtype(cell_id_vec.dtype, np.integer):
        raise ValueError(f"Invalid type `{cell_id_vec.dtype}` for `adata.obs[{seg_cell_id!r}]`.")
    cell_id_vec = [cell_id_vec[adata.obs[library_key] == lib] for lib in library_id]

    if isinstance(seg, list | np.ndarray | da.Array):
        img_seg = _get_list(seg, _type=(np.ndarray, da.Array), ref_len=len(library_id), name="img_seg")
    else:
        img_seg = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][seg_key] for i in library_id]
    return img_seg, cell_id_vec


def _get_scalefactor_size(
    adata: AnnData,
    library_id: Sequence[str],
    spatial_key: str = "spatial",
    img_res_key: str | None = None,
    scale_factor: _SeqFloat | None = None,
    size: _SeqFloat | None = None,
    size_key: str | None = "spot_diameter_fullres",
) -> tuple[Sequence[float], Sequence[float]]:
    from squidpy._constants._pkg_constants import Key
    
    try:
        scalefactor_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.scalefactor_key, library_id)
        scalefactors = _get_unique_map(scalefactor_mapping)
    except KeyError as e:
        scalefactors = None
        logg.debug(f"Setting `scalefactors={scalefactors}`, reason: `{e}`")

    if scalefactors is not None and img_res_key is not None:
        if scale_factor is None:  # get intersection of scale_factor and match to img_res_key
            scale_factor_key = [i for i in scalefactors if img_res_key in i]
            if not len(scale_factor_key):
                raise ValueError(f"No `scale_factor` found that could match `img_res_key`: {img_res_key}.")
            _scale_factor_key = scale_factor_key[0]  # get first scale_factor
            scale_factor = [
                adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][_scale_factor_key] for i in library_id
            ]
        else:  # handle case where scale_factor is float or list
            scale_factor = _get_list(scale_factor, _type=float, ref_len=len(library_id), name="scale_factor")

        if size_key not in scalefactors and size is None:
            raise ValueError(
                f"Specified `size_key: {size_key}` does not exist and size is `None`, "
                f"available keys are: `{scalefactors}`. Specify a valid `size_key` or `size`."
            )
        if size is None:
            size = 1.0
        size = _get_list(size, _type=Number, ref_len=len(library_id), name="size")
        if not (len(size) == len(library_id) == len(scale_factor)):
            raise ValueError("Len of `size`, `library_id` and `scale_factor` do not match.")
        size = [
            adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][size_key] * s * sf * 0.5
            for i, s, sf in zip(library_id, size, scale_factor, strict=False)
        ]
        return scale_factor, size

    scale_factor = 1.0 if scale_factor is None else scale_factor
    scale_factor = _get_list(scale_factor, _type=float, ref_len=len(library_id), name="scale_factor")

    size = 120000 / adata.shape[0] if size is None else size
    size = _get_list(size, _type=Number, ref_len=len(library_id), name="size")
    return scale_factor, size


def _image_spatial_attrs(
    adata: AnnData,
    shape: str | None = None,
    spatial_key: str = "spatial",
    library_id: Sequence[str] | None = None,
    library_key: str | None = None,
    img: bool | Sequence[np.ndarray] | None = None,
    img_res_key: str | None = "hires",
    img_channel: int | list[int] | None = None,
    seg: Sequence[np.ndarray] | bool | None = None,
    seg_key: str | None = None,
    cell_id_key: str | None = None,
    scale_factor: _SeqFloat | None = None,
    size: _SeqFloat | None = None,
    size_key: str | None = "spot_diameter_fullres",
    img_cmap: Colormap | str | None = None,
) -> SpatialParams:
    def truthy(img: bool | np.ndarray | Sequence[np.ndarray] | None) -> bool:
        if img is None or img is False:
            return False
        return img is True or len(img)  # type: ignore

    library_id = _get_library_id(
        adata=adata,
        shape=shape,
        spatial_key=spatial_key,
        library_id=library_id,
        library_key=library_key,
    )
    if len(library_id) > 1 and library_key is None:
        raise ValueError(
            f"Found `library_id: `{library_id} but no `library_key` was specified. Please specify `library_key`."
        )

    scale_factor, size = _get_scalefactor_size(
        adata=adata,
        spatial_key=spatial_key,
        library_id=library_id,
        img_res_key=img_res_key,
        scale_factor=scale_factor,
        size=size,
        size_key=size_key,
    )

    if (truthy(img) and truthy(seg)) or (truthy(img) and shape is not None):
        _img = _get_image(
            adata=adata,
            spatial_key=spatial_key,
            library_id=library_id,
            img=img,
            img_res_key=img_res_key,
            img_channel=img_channel,
            img_cmap=img_cmap,
        )
    else:
        _img = (None,) * len(library_id)

    if truthy(seg):
        _seg, _cell_vec = _get_segment(
            adata=adata,
            library_id=library_id,
            seg_cell_id=cell_id_key,
            library_key=library_key,
            seg=seg,
            seg_key=seg_key,
        )
    else:
        _seg = (None,) * len(library_id)
        _cell_vec = (None,) * len(library_id)

    return SpatialParams(library_id, scale_factor, size, _img, _seg, _cell_vec)


def _set_coords_crops(
    adata: AnnData,
    spatial_params: SpatialParams,
    spatial_key: str,
    crop_coord: Sequence[_CoordTuple] | _CoordTuple | None = None,
) -> tuple[list[np.ndarray], list[Any] | list[None]]:
    from squidpy.im._coords import CropCoords
    
    if crop_coord is None:
        crops = [None] * len(spatial_params.library_id)
    else:
        crop_coord = _get_list(
            crop_coord,
            _type=tuple,
            ref_len=len(spatial_params.library_id),
            name="crop_coord",
        )
        crops = [CropCoords(*cr) * sf for cr, sf in zip(crop_coord, spatial_params.scale_factor, strict=False)]  # type: ignore[misc]

    coords = adata.obsm[spatial_key]
    return [coords * sf for sf in spatial_params.scale_factor], crops  # TODO(giovp): refactor with _subs


def _subs(
    adata: AnnData,
    coords: np.ndarray,
    img: np.ndarray | None = None,
    library_key: str | None = None,
    library_id: str | None = None,
    crop_coords: Any | None = None,
    groups_key: str | None = None,
    groups: Sequence[Any] | None = None,
) -> AnnData:
    from squidpy.im._coords import CropCoords
    
    def assert_notempty(adata: AnnData, *, msg: str) -> AnnData:
        if not adata.n_obs:
            raise ValueError(f"Empty AnnData, reason: {msg}.")
        return adata

    def subset_by_key(
        adata: AnnData,
        coords: np.ndarray,
        key: str | None,
        values: Sequence[Any] | None,
    ) -> tuple[AnnData, np.ndarray]:
        if key is None or values is None:
            return adata, coords
        if key not in adata.obs or not isinstance(adata.obs[key].dtype, CategoricalDtype):
            return adata, coords
        try:
            mask = adata.obs[key].isin(values).values
            msg = f"None of `adata.obs[{key}]` are in `{values}`"
            return assert_notempty(adata[mask], msg=msg), coords[mask]
        except KeyError:
            raise KeyError(f"Unable to find `{key!r}` in `adata.obs`.") from None

    def subset_by_coords(
        adata: AnnData,
        coords: np.ndarray,
        img: np.ndarray | None,
        crop_coords: Any | None,
    ) -> tuple[AnnData, np.ndarray, np.ndarray | None]:
        if crop_coords is None:
            return adata, coords, img

        mask = (
            (coords[:, 0] >= crop_coords.x0)
            & (coords[:, 0] <= crop_coords.x1)
            & (coords[:, 1] >= crop_coords.y0)
            & (coords[:, 1] <= crop_coords.y1)
        )
        adata = assert_notempty(adata[mask, :], msg=f"Invalid crop coordinates `{crop_coords}`")
        coords = coords[mask]
        coords[:, 0] -= crop_coords.x0
        coords[:, 1] -= crop_coords.y0
        if img is not None:
            img = img[crop_coords.slice]
        return adata, coords, img

    adata, coords, img = subset_by_coords(adata, coords=coords, img=img, crop_coords=crop_coords)
    adata, coords = subset_by_key(adata, coords=coords, key=library_key, values=[library_id])
    adata, coords = subset_by_key(adata, coords=coords, key=groups_key, values=groups)
    return adata, coords, img


def _get_unique_map(dic: Mapping[str, Any]) -> Sequence[Any]:
    """Get intersection of dict values."""
    return sorted(set.intersection(*map(set, dic.values())))


def _get_list(
    var: Any,
    _type: type[Any] | tuple[type[Any], ...],
    ref_len: int | None = None,
    name: str | None = None,
) -> list[Any]:
    if isinstance(var, _type):
        return [var] if ref_len is None else ([var] * ref_len)
    if isinstance(var, list):
        if ref_len is not None and ref_len != len(var):
            raise ValueError(
                f"Variable: `{name}` has length: {len(var)}, which is not equal to reference length: {ref_len}."
            )
        for v in var:
            if not isinstance(v, _type):
                raise ValueError(f"Variable: `{name}` has invalid type: {type(v)}, expected: {_type}.")
        return var

    raise ValueError(f"Can't make a list from variable: `{var}`")


def _set_color_source_vec(
    adata: AnnData,
    value_to_plot: str | None,
    use_raw: bool | None = None,
    alt_var: str | None = None,
    layer: str | None = None,
    groups: _SeqStr | None = None,
    palette: Palette_t = None,
    na_color: str | tuple[float, ...] | None = None,
    alpha: float = 1.0,
) -> tuple[np.ndarray | pd.Series | None, np.ndarray, bool]:
    from squidpy.pl._color_utils import _get_palette
    
    if value_to_plot is None:
        color = np.full(adata.n_obs, to_hex(na_color))
        return color, color, False

    if alt_var is not None and value_to_plot not in adata.obs and value_to_plot not in adata.var_names:
        value_to_plot = adata.var_names[adata.var[alt_var] == value_to_plot][0]
    if use_raw and value_to_plot not in adata.obs:
        color_source_vector = adata.raw.obs_vector(value_to_plot)
    else:
        color_source_vector = adata.obs_vector(value_to_plot, layer=layer)

    if not isinstance(color_source_vector.dtype, CategoricalDtype):
        return None, color_source_vector, False

    color_source_vector = pd.Categorical(color_source_vector)  # convert, e.g., `pd.Series`
    categories = color_source_vector.categories
    if groups is not None:
        color_source_vector = color_source_vector.remove_categories(categories.difference(groups))

    color_map = _get_palette(
        adata,
        cluster_key=value_to_plot,
        categories=categories,
        palette=palette,
        alpha=alpha,
    )
    if color_map is None:
        raise ValueError("Unable to create color palette.")
    # do not rename categories, as colors need not be unique
    color_vector = color_source_vector.map(color_map, na_action=None)
    if color_vector.isna().any():
        color_vector = color_vector.add_categories([to_hex(na_color)])
        color_vector = color_vector.fillna(to_hex(na_color))

    return color_source_vector, color_vector, True


def _shaped_scatter(
    x: np.ndarray,
    y: np.ndarray,
    s: float,
    c: np.ndarray | str,
    shape: str | None = "circle",
    norm: _Normalize | None = None,
    **kwargs: Any,
) -> PatchCollection:
    """
    Get shapes for scatter plot.

    Adapted from `here <https://gist.github.com/syrte/592a062c562cd2a98a83>`_.
    This code is under `The BSD 3-Clause License <http://opensource.org/licenses/BSD-3-Clause>`_.
    """
    from squidpy._constants._constants import ScatterShape
    
    if shape is None:
        shape = "circle"
    
    if shape == "circle":
        patches = [Circle((x, y), radius=s) for x, y, s in np.broadcast(x, y, s)]
    elif shape == "square":
        patches = [Rectangle((x - s, y - s), width=2 * s, height=2 * s) for x, y, s in np.broadcast(x, y, s)]
    elif shape == "hex":
        n = 6
        r = s / (2 * np.sin(np.pi / n))
        polys = np.stack([_make_poly(x, y, r, n, i) for i in range(n)], 1).swapaxes(0, 2)
        patches = [Polygon(p, closed=False) for p in polys]
    else:
        raise NotImplementedError(f"Shape `{shape}` is not yet implemented.")
    collection = PatchCollection(patches, snap=False, **kwargs)

    if isinstance(c, np.ndarray) and np.issubdtype(c.dtype, np.number):
        collection.set_array(np.ma.masked_invalid(c).ravel())
        collection.set_norm(norm)
    else:
        alpha = ColorConverter().to_rgba_array(c)[..., -1]
        collection.set_facecolor(c)
        collection.set_alpha(alpha)

    return collection


def _make_poly(x: np.ndarray, y: np.ndarray, r: float, n: int, i: int) -> tuple[np.ndarray, np.ndarray]:
    x_i = x + r * np.sin((np.pi / n) * (1 + 2 * i))
    y_i = y + r * np.cos((np.pi / n) * (1 + 2 * i))
    return x_i, y_i


def _plot_edges(
    adata: AnnData,
    coords: np.ndarray,
    connectivity_key: str,
    ax: Axes,
    edges_width: float = 0.1,
    edges_color: str | Sequence[float] | Sequence[str] = "grey",
    **kwargs: Any,
) -> None:
    from networkx import Graph
    from networkx.drawing import draw_networkx_edges

    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"Unable to find `connectivity_key: {connectivity_key}` in `adata.obsp`. Please set `connectivity_key`."
        )

    g = Graph(adata.obsp[connectivity_key])
    if not len(g.edges):
        return None
    edge_collection = draw_networkx_edges(
        g,
        coords,
        width=edges_width,
        edge_color=edges_color,
        arrows=False,
        ax=ax,
        **kwargs,
    )
    edge_collection.set_rasterized(_get_vector_friendly())
    ax.add_collection(edge_collection)


def _get_title_axlabels(
    title: _SeqStr | None, axis_label: _SeqStr | None, spatial_key: str, n_plots: int
) -> tuple[_SeqStr | None, Sequence[str]]:
    if title is None:
        title = None

    elif isinstance(title, tuple | list) and len(title) != n_plots:
        raise ValueError(f"Expected `{n_plots}` titles, found `{len(title)}`.")
    elif isinstance(title, str):
        title = [title] * n_plots
    axis_label = spatial_key if axis_label is None else axis_label
    if isinstance(axis_label, list):
        if len(axis_label) != 2:
            raise ValueError(f"Expected axis labels to be of length `2`, found `{len(axis_label)}`.")
        axis_labels = axis_label
    elif isinstance(axis_label, str):
        axis_labels = [axis_label + str(x + 1) for x in range(2)]
    else:
        raise TypeError(f"Expected axis labels to be of type `list` or `str`, found `{type(axis_label).__name__}`.")

    return title, axis_labels


def _get_scalebar(
    scalebar_dx: _SeqFloat | None = None,
    scalebar_units: _SeqStr | None = None,
    len_lib: int | None = None,
) -> tuple[Sequence[float] | None, Sequence[str] | None]:
    if scalebar_dx is not None:
        _scalebar_dx = _get_list(scalebar_dx, _type=float, ref_len=len_lib, name="scalebar_dx")
        scalebar_units = "um" if scalebar_units is None else scalebar_units
        _scalebar_units = _get_list(scalebar_units, _type=str, ref_len=len_lib, name="scalebar_units")
    else:
        _scalebar_dx = None
        _scalebar_units = None

    return _scalebar_dx, _scalebar_units


def _decorate_axs(
    ax: Axes,
    cax: PatchCollection,
    lib_count: int,
    fig_params: FigParams,
    adata: AnnData,
    coords: np.ndarray,
    value_to_plot: str,
    color_source_vector: pd.Series[CategoricalDtype] | None,
    img: np.ndarray | None = None,
    img_cmap: str | None = None,
    img_alpha: float | None = None,
    palette: Palette_t = None,
    alpha: float = 1.0,
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_loc: str | None = "right margin",
    legend_fontoutline: int | None = None,
    na_color: str | tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    na_in_legend: bool = True,
    colorbar: bool = True,
    scalebar_dx: Sequence[float] | None = None,
    scalebar_units: Sequence[str] | None = None,
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Axes:
   
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(fig_params.ax_labels[0])
    ax.set_ylabel(fig_params.ax_labels[1])
    ax.autoscale_view()  # needed when plotting points but no image

    if value_to_plot is not None:
        # if only dots were plotted without an associated value
        # there is not need to plot a legend or a colorbar

        if legend_fontoutline is not None:
            path_effect = [patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")]
        else:
            path_effect = []

        # Adding legends
        if color_source_vector is not None and isinstance(color_source_vector.dtype, CategoricalDtype):
            clusters = color_source_vector.categories
            palette = _get_palette(
                adata,
                cluster_key=value_to_plot,
                categories=clusters,
                palette=palette,
                alpha=alpha,
            )
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=palette,
                scatter_array=coords,
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=[na_color],
                na_in_legend=na_in_legend,
                multi_panel=fig_params.axs is not None,
            )
        elif colorbar:
            # TODO: na_in_legend should have some effect here
            plt.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30)

    if img is not None:
        ax.imshow(img, cmap=img_cmap, alpha=img_alpha)
    else:
        ax.set_aspect("equal")
        ax.invert_yaxis()

    if isinstance(scalebar_dx, list) and isinstance(scalebar_units, list):
        from matplotlib_scalebar.scalebar import ScaleBar
        scalebar = ScaleBar(scalebar_dx[lib_count], units=scalebar_units[lib_count], **scalebar_kwargs)
        ax.add_artist(scalebar)

    return ax


def _map_color_seg(
    seg: np.ndarray,
    cell_id: np.ndarray,
    color_vector: np.ndarray | pd.Categorical,
    color_source_vector: pd.Categorical,
    cmap_params: CmapParams,
    seg_erosionpx: int | None = None,
    seg_boundaries: bool = False,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),
) -> np.ndarray:
    cell_id = np.array(cell_id)
    from skimage.morphology import erosion, square
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries
    from skimage.util import map_array

    if isinstance(color_vector, pd.Categorical):
        if isinstance(na_color, tuple) and len(na_color) == 4 and np.any(color_source_vector.isna()):
            cell_id[color_source_vector.isna()] = 0
        val_im: np.ndarray = map_array(seg, cell_id, color_vector.codes + 1)
        cols = colors.to_rgba_array(color_vector.categories)
        
        # Check if we have alpha variation (transparency gradients)
        has_alpha_variation = len(cols) > 0 and not np.allclose(cols[:, 3], cols[0, 3])
        use_label2rgb = not has_alpha_variation
    else:
        val_im = map_array(seg, cell_id, cell_id)  # replace with same seg id to remove missing segs
        
        # For continuous data, we need to map the actual color_vector values through the colormap
        # Don't pre-convert to colors here - let the plotting function handle it
        try:
            # Test if the colormap has alpha variation by sampling it
            test_values = np.linspace(0, 1, 10)
            test_colors = cmap_params.cmap(test_values)
            has_alpha_variation = not np.allclose(test_colors[:, 3], test_colors[0, 3])
        except (IndexError, TypeError):
            has_alpha_variation = False
        
        if has_alpha_variation:
            # For transparent gradients, we need to apply colormap to the actual values
            # Map each cell_id to its corresponding color_vector value
            cols = cmap_params.cmap(cmap_params.norm(color_vector))
        else:
            # For opaque colormaps, use the standard approach
            try:
                cols = cmap_params.cmap(cmap_params.norm(color_vector))
            except TypeError:
                assert all(colors.is_color_like(c) for c in color_vector), "Not all values are color-like."
                cols = colors.to_rgba_array(color_vector)
        
        use_label2rgb = not has_alpha_variation

    if seg_erosionpx is not None:
        val_im[val_im == erosion(val_im, square(seg_erosionpx))] = 0

    if use_label2rgb:
        # Use the original approach for opaque colors
        seg_im: np.ndarray = label2rgb(
            label=val_im,
            colors=cols,
            bg_label=0,
            bg_color=(1, 1, 1),  # transparency doesn't really work
        )
        
        if seg_boundaries:
            seg_bound: np.ndarray = np.clip(seg_im - find_boundaries(seg)[:, :, None], 0, 1)
            seg_bound = np.dstack((seg_bound, np.where(val_im > 0, 1, 0)))  # add transparency here
            return seg_bound
        seg_im = np.dstack((seg_im, np.where(val_im > 0, 1, 0)))  # add transparency here
        return seg_im
    else:
        # Handle transparent gradient colormaps with vectorized operations
        height, width = seg.shape
        seg_im = np.zeros((height, width, 4), dtype=np.float32)
        
        if isinstance(color_vector, pd.Categorical):
            # For categorical data, use vectorized approach
            # Create a color lookup table
            color_lut = np.zeros((len(cols) + 1, 4), dtype=np.float32)  # +1 for background
            color_lut[1:] = cols  # Skip index 0 (background)
            
            # Vectorized color mapping
            seg_im = color_lut[val_im]
        else:
            # For continuous data, use vectorized approach
            # Create a mapping from cell_id to color index efficiently
            if len(cell_id) > 0 and len(cols) > 0:
                # Create a dictionary for fast lookup, then vectorize
                valid_mask = cell_id > 0
                valid_cell_ids = cell_id[valid_mask]
                valid_colors = cols[valid_mask] if len(cols) == len(cell_id) else cols[:len(valid_cell_ids)]
                
                if len(valid_cell_ids) > 0:
                    # For efficiency, check if cell_ids are dense (consecutive)
                    min_id, max_id = int(np.min(valid_cell_ids)), int(np.max(valid_cell_ids))
                    id_range = max_id - min_id + 1
                    
                    if id_range <= len(valid_cell_ids) * 2:  # Dense case - use lookup table
                        color_lut = np.zeros((max_id + 1, 4), dtype=np.float32)
                        # Vectorized assignment using advanced indexing
                        color_lut[valid_cell_ids.astype(int)] = valid_colors[:len(valid_cell_ids)]
                        
                        # Vectorized color assignment
                        mask = val_im > 0
                        seg_im[mask] = color_lut[val_im[mask]]
                    else:  # Sparse case - use map_array for efficiency
                        # Create a dense mapping array
                        dense_ids = np.arange(len(valid_cell_ids))
                        
                        # Create lookup table for dense indices
                        dense_color_lut = np.zeros((len(valid_cell_ids) + 1, 4), dtype=np.float32)
                        dense_color_lut[1:] = valid_colors[:len(valid_cell_ids)]
                        
                        # Map sparse cell_ids to dense indices
                        dense_val_im = map_array(val_im, valid_cell_ids, dense_ids + 1)
                        
                        # Vectorized color assignment
                        seg_im = dense_color_lut[dense_val_im]
        
        if seg_boundaries:
            # Apply boundary effect while preserving alpha
            boundaries = find_boundaries(seg)
            # Only darken RGB channels, preserve alpha
            boundary_mask = boundaries[:, :, np.newaxis]
            seg_im[:, :, :3] = np.clip(seg_im[:, :, :3] - boundary_mask * 0.3, 0, 1)
        
        return seg_im


def _prepare_args_plot(
    adata: AnnData,
    shape: str | None = None,
    color: Sequence[str | None] | str | None = None,
    groups: _SeqStr | None = None,
    img_alpha: float | None = None,
    alpha: float = 1.0,
    use_raw: bool | None = None,
    layer: str | None = None,
    palette: Palette_t = None,
) -> ColorParams:
    from squidpy.pl._color_utils import _maybe_set_colors
    
    img_alpha = 1.0 if img_alpha is None else img_alpha

    # make colors and groups as list
    groups = [groups] if isinstance(groups, str) else groups
    if isinstance(color, list) and not len(color):
        color = None
    color = [color] if isinstance(color, str) or color is None else color

    # set palette if missing
    for c in color:
        if c is not None and c in adata.obs and isinstance(adata.obs[c].dtype, CategoricalDtype):
            _maybe_set_colors(source=adata, target=adata, key=c, palette=palette)

    # check raw
    if use_raw is None:
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            f"Cannot use both a layer and the raw representation. Got passed: use_raw={use_raw}, layer={layer}."
        )
    if adata.raw is None and use_raw:
        raise ValueError(f"`use_raw={use_raw}` but AnnData object does not have raw.")

    return ColorParams(shape, color, groups, alpha, img_alpha, use_raw)


def _prepare_params_plot(
    color_params: ColorParams,
    spatial_params: SpatialParams,
    spatial_key: str = "spatial",
    wspace: float | None = None,
    hspace: float = 0.25,
    ncols: int = 4,
    cmap: Colormap | str | None = None,
    norm: _Normalize | None = None,
    library_first: bool = True,
    img_cmap: Colormap | str | None = None,
    frameon: bool | None = None,
    na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    title: _SeqStr | None = None,
    axis_label: _SeqStr | None = None,
    scalebar_dx: _SeqFloat | None = None,
    scalebar_units: _SeqStr | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    fig: Figure | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    **kwargs: Any,
) -> tuple[FigParams, CmapParams, ScalebarParams, Any]:
    if library_first:
        iter_panels: tuple[range | Sequence[str | None], range | Sequence[str | None]] = (
            range(len(spatial_params.library_id)),
            color_params.color,
        )
    else:
        iter_panels = (color_params.color, range(len(spatial_params.library_id)))
    num_panels = len(list(itertools.product(*iter_panels)))

    wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02 if wspace is None else wspace
    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    dpi = rcParams["figure.dpi"] if dpi is None else dpi
    if num_panels > 1 and ax is None:
        fig, grid = _panel_grid(
            num_panels=num_panels,
            hspace=hspace,
            wspace=wspace,
            ncols=ncols,
            dpi=dpi,
            figsize=figsize,
        )
        axs: Sequence[Axes] | None = [plt.subplot(grid[c]) for c in range(num_panels)]
    elif num_panels > 1 and ax is not None:
        if len(ax) != num_panels:
            raise ValueError(f"Len of `ax`: {len(ax)} is not equal to number of panels: {num_panels}.")
        if fig is None:
            raise ValueError(
                f"Invalid value of `fig`: {fig}. If a list of `Axes` is passed, a `Figure` must also be specified."
            )
        axs = ax
    else:
        axs = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    # set cmap and norm
    if cmap is None:
        cmap = plt.rcParams["image.cmap"]
    if isinstance(cmap, str):
        cmap = plt.colormaps[cmap]
    cmap.set_bad("lightgray" if na_color is None else na_color)

    if isinstance(norm, Normalize):
        pass
    elif vcenter is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)

    # set title and axis labels
    title, ax_labels = _get_title_axlabels(title, axis_label, spatial_key, num_panels)

    # set scalebar
    if scalebar_dx is not None:
        scalebar_dx, scalebar_units = _get_scalebar(scalebar_dx, scalebar_units, len(spatial_params.library_id))

    fig_params = FigParams(fig, ax, axs, iter_panels, title, ax_labels, frameon)
    cmap_params = CmapParams(cmap, img_cmap, norm)
    scalebar_params = ScalebarParams(scalebar_dx, scalebar_units)

    return fig_params, cmap_params, scalebar_params, kwargs


def _panel_grid(
    num_panels: int,
    hspace: float,
    wspace: float,
    ncols: int,
    figsize: tuple[float, float],
    dpi: int | None = None,
) -> tuple[Figure, GridSpec]:
    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)

    fig = plt.figure(
        figsize=(figsize[0] * n_panels_x * (1 + wspace), figsize[1] * n_panels_y),
        dpi=dpi,
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs


def _set_ax_title(fig_params: FigParams, count: int, value_to_plot: str | None = None) -> Axes:
    ax = fig_params.axs[count] if fig_params.axs is not None else fig_params.ax
    if not (sc_settings._frameon if fig_params.frameon is None else fig_params.frameon):
        ax.axis("off")

    if fig_params.title is None:
        ax.set_title(value_to_plot)
    else:
        ax.set_title(fig_params.title[count])
    return ax


def _set_outline(
    size: float,
    outline: bool = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    **kwargs: Any,
) -> tuple[OutlineParams, Any]:
    bg_width, gap_width = outline_width
    point = np.sqrt(size)
    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
    # the default black and white colors can be changes using the contour_config parameter
    bg_color, gap_color = outline_color

    if outline:
        kwargs.pop("edgecolor", None)  # remove edge from kwargs if present
        kwargs.pop("alpha", None)  # remove alpha from kwargs if present

    return OutlineParams(outline, gap_size, gap_color, bg_size, bg_color), kwargs


def _plot_scatter(
    coords: np.ndarray,
    ax: Axes,
    outline_params: OutlineParams,
    cmap_params: CmapParams,
    color_params: ColorParams,
    size: float,
    color_vector: np.ndarray,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),  # TODO(giovp): remove?
    **kwargs: Any,
) -> tuple[Axes, Collection | PatchCollection]:
    if color_params.shape is not None:
        scatter = partial(_shaped_scatter, shape=color_params.shape, alpha=color_params.alpha)
    else:
        scatter = partial(ax.scatter, marker=".", alpha=color_params.alpha, plotnonfinite=True)

    # prevents reusing vmin/vmax when sharing a norm
    norm = copy(cmap_params.norm)
    if outline_params.outline:
        _cax = scatter(
            coords[:, 0],
            coords[:, 1],
            s=outline_params.bg_size,
            c=outline_params.bg_color,
            rasterized=_get_vector_friendly(),
            cmap=cmap_params.cmap,
            norm=norm,
            **kwargs,
        )
        ax.add_collection(_cax)
        _cax = scatter(
            coords[:, 0],
            coords[:, 1],
            s=outline_params.gap_size,
            c=outline_params.gap_color,
            rasterized=_get_vector_friendly(),
            cmap=cmap_params.cmap,
            norm=norm,
            **kwargs,
        )
        ax.add_collection(_cax)
    _cax = scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.array(color_vector),
        s=size,
        rasterized=_get_vector_friendly(),
        cmap=cmap_params.cmap,
        norm=norm,
        **kwargs,
    )
    cax = ax.add_collection(_cax)

    return ax, cax


def _plot_segment(
    seg: np.ndarray,
    cell_id: np.ndarray,
    color_vector: np.ndarray | pd.Series[CategoricalDtype],
    color_source_vector: pd.Series[CategoricalDtype],
    ax: Axes,
    cmap_params: CmapParams,
    color_params: ColorParams,
    categorical: bool,
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),
    **kwargs: Any,
) -> tuple[Axes, Collection]:
    # Check if the colormap has alpha variation (transparency gradients)
    has_alpha_variation = False
    is_custom_transparent_cmap = False
    from skimage.morphology import erosion, square
    from skimage.segmentation import find_boundaries
    from skimage.util import map_array
    
    if not categorical:
        try:
            test_values = np.linspace(0, 1, 10)
            test_colors = cmap_params.cmap(test_values)
            has_alpha_variation = not np.allclose(test_colors[:, 3], test_colors[0, 3])
            
            # Check if this is a custom transparent colormap (has alpha variation)
            # vs a standard matplotlib colormap
            is_custom_transparent_cmap = has_alpha_variation
        except (IndexError, TypeError):
            has_alpha_variation = False
    
    if has_alpha_variation and not categorical:
        # Handle transparent gradient colormaps specially
        # Create a value image that maps directly to the color_vector values
        cell_id_array = np.array(cell_id)
        val_im = map_array(seg, cell_id_array, cell_id_array)
        
        if seg_contourpx is not None:
            val_im[val_im == erosion(val_im, square(seg_contourpx))] = 0
        
        # Create a mapping from cell_id to color_vector value
        if len(cell_id) > 0 and len(color_vector) > 0:
            # Create lookup table for values
            max_cell_id = int(np.max(cell_id_array)) if len(cell_id_array) > 0 else 0
            if max_cell_id > 0:
                value_lut = np.full(max_cell_id + 1, np.nan, dtype=np.float32)
                
                # Vectorized mapping from cell_id to color_vector values
                valid_mask = (cell_id_array > 0) & (cell_id_array <= max_cell_id)
                if np.any(valid_mask):
                    valid_cell_ids = cell_id_array[valid_mask].astype(int)
                    valid_values = color_vector[valid_mask] if len(color_vector) == len(cell_id_array) else color_vector[:np.sum(valid_mask)]
                    value_lut[valid_cell_ids] = valid_values
                
                # Create the value image
                value_img = np.full_like(val_im, np.nan, dtype=np.float32)
                mask = val_im > 0
                value_img[mask] = value_lut[val_im[mask]]
                
                # Ensure we have valid data range for the colormap
                valid_data = value_img[~np.isnan(value_img)]
                if len(valid_data) > 0:
                    # Check if normalization needs adjustment
                    vmin, vmax = np.min(valid_data), np.max(valid_data)
                    
                    # Fix normalization if vmin/vmax are None
                    norm_to_use = copy(cmap_params.norm)
                    if norm_to_use.vmin is None or norm_to_use.vmax is None:
                        norm_to_use.vmin = vmin
                        norm_to_use.vmax = vmax
                    
                    # If all values are the same, create a small artificial range
                    if np.allclose(valid_data, valid_data[0]):
                        if vmin == vmax:
                            # Create a small range to show at least some variation
                            range_val = max(abs(vmin) * 0.01, 0.01)  # 1% of value or 0.01 minimum
                            value_img[mask] = np.linspace(vmin, vmin + range_val, np.sum(mask))
                            norm_to_use.vmax = vmin + range_val
                else:
                    norm_to_use = cmap_params.norm
                
                # Apply boundaries if needed
                if seg_outline:
                    boundaries = find_boundaries(seg)
                    # Reduce values at boundaries to create darker edges
                    boundary_mask = boundaries & mask
                    if hasattr(cmap_params.norm, 'vmin') and hasattr(cmap_params.norm, 'vmax'):
                        vmin, vmax = cmap_params.norm.vmin, cmap_params.norm.vmax
                        if vmin is not None and vmax is not None:
                            range_val = vmax - vmin
                            value_img[boundary_mask] = np.maximum(
                                value_img[boundary_mask] - 0.3 * range_val, 
                                vmin
                            )
                
                # Use matplotlib's imshow with the colormap - this preserves colorbar compatibility
                _cax = ax.imshow(
                    value_img,
                    cmap=cmap_params.cmap,
                    norm=norm_to_use,
                    alpha=color_params.alpha,  # Apply alpha parameter correctly
                    origin="lower",
                    zorder=3,
                    rasterized=True,
                    **kwargs,
                )
                cax = ax.add_image(_cax)
                return ax, cax
    
    # For standard matplotlib colormaps (like 'Reds') and other cases
    if not categorical and not is_custom_transparent_cmap:
        # Handle standard colormaps by creating a value image that preserves colormap/norm
        cell_id_array = np.array(cell_id)
        val_im = map_array(seg, cell_id_array, cell_id_array)
        
        if seg_contourpx is not None:
            val_im[val_im == erosion(val_im, square(seg_contourpx))] = 0
        
        # Create a mapping from cell_id to color_vector value
        if len(cell_id) > 0 and len(color_vector) > 0:
            max_cell_id = int(np.max(cell_id_array)) if len(cell_id_array) > 0 else 0
            if max_cell_id > 0:
                value_lut = np.full(max_cell_id + 1, np.nan, dtype=np.float32)
                
                # Vectorized mapping from cell_id to color_vector values
                valid_mask = (cell_id_array > 0) & (cell_id_array <= max_cell_id)
                if np.any(valid_mask):
                    valid_cell_ids = cell_id_array[valid_mask].astype(int)
                    valid_values = color_vector[valid_mask] if len(color_vector) == len(cell_id_array) else color_vector[:np.sum(valid_mask)]
                    value_lut[valid_cell_ids] = valid_values
                
                # Create the value image
                value_img = np.full_like(val_im, np.nan, dtype=np.float32)
                mask = val_im > 0
                value_img[mask] = value_lut[val_im[mask]]
                
                # Ensure we have valid data range for the colormap
                valid_data = value_img[~np.isnan(value_img)]
                if len(valid_data) > 0:
                    # Fix normalization if vmin/vmax are None
                    norm_to_use = copy(cmap_params.norm)
                    if norm_to_use.vmin is None or norm_to_use.vmax is None:
                        vmin, vmax = np.min(valid_data), np.max(valid_data)
                        norm_to_use.vmin = vmin
                        norm_to_use.vmax = vmax
                else:
                    norm_to_use = cmap_params.norm
                
                # Apply boundaries if needed
                if seg_outline:
                    boundaries = find_boundaries(seg)
                    # Create a mask for boundaries
                    boundary_mask = boundaries & mask
                    if len(valid_data) > 0:
                        # Reduce values at boundaries to create darker edges
                        vmin, vmax = np.min(valid_data), np.max(valid_data)
                        range_val = vmax - vmin if vmax > vmin else 1.0
                        value_img[boundary_mask] = np.maximum(
                            value_img[boundary_mask] - 0.3 * range_val, 
                            vmin
                        )
                
                # Use matplotlib's imshow with the colormap - this preserves colorbar compatibility
                _cax = ax.imshow(
                    value_img,
                    cmap=cmap_params.cmap,
                    norm=norm_to_use,
                    alpha=color_params.alpha,
                    origin="lower",
                    zorder=3,
                    rasterized=True,
                    **kwargs,
                )
                cax = ax.add_image(_cax)
                return ax, cax
    
    # Fall back to the original approach for categorical data and custom transparent colormaps
    img = _map_color_seg(
        seg=seg,
        cell_id=cell_id,
        color_vector=color_vector,
        color_source_vector=color_source_vector,
        cmap_params=cmap_params,
        seg_erosionpx=seg_contourpx,
        seg_boundaries=seg_outline,
        na_color=na_color,
    )

    # Check if the returned image is RGBA (has alpha channel)
    # If so, treat it as categorical to avoid applying additional colormap/normalization
    has_alpha_channel = img.shape[-1] == 4
    is_categorical_image = categorical or has_alpha_channel
    
    # For RGBA images from transparent gradients, apply the alpha parameter by multiplying
    if has_alpha_channel and color_params.alpha != 1.0:
        img = img.copy()  # Don't modify the original
        img[..., 3] *= color_params.alpha

    _cax = ax.imshow(
        img,
        rasterized=True,
        cmap=None if is_categorical_image else cmap_params.cmap,
        norm=None if is_categorical_image else cmap_params.norm,
        alpha=None if has_alpha_channel else color_params.alpha,
        origin="lower",
        zorder=3,
        **kwargs,
    )
    cax = ax.add_image(_cax)

    return ax, cax


def _spatial_plot(
    adata: AnnData,
    shape: str | None = None,
    color: str | Sequence[str | None] | None = None,
    groups: _SeqStr | None = None,
    library_id: _SeqStr | None = None,
    library_key: str | None = None,
    spatial_key: str = "spatial",
    # image
    img: bool | Sequence[np.ndarray] | None = True,
    img_res_key: str | None = "hires",
    img_alpha: float | None = None,
    img_cmap: Colormap | str | None = None,
    img_channel: int | list[int] | None = None,
    # segment
    seg: bool | Sequence[np.ndarray] | None = None,
    seg_key: str | None = "segmentation",
    seg_cell_id: str | None = None,
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    # features
    use_raw: bool | None = None,
    layer: str | None = None,
    alt_var: str | None = None,
    # size, coords, cmap, palette
    size: _SeqFloat | None = None,
    size_key: str | None = "spot_diameter_fullres",
    scale_factor: _SeqFloat | None = None,
    crop_coord: _CoordTuple | Sequence[_CoordTuple] | None = None,
    cmap: str | Colormap | None = None,
    palette: Palette_t = None,
    alpha: float = 1.0,
    norm: _Normalize | None = None,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),
    # edges
    connectivity_key: str | None = None,
    edges_width: float = 1.0,
    edges_color: str | Sequence[str] | Sequence[float] = "grey",
    # panels
    library_first: bool = True,
    frameon: bool | None = None,
    wspace: float | None = None,
    hspace: float = 0.25,
    ncols: int = 4,
    # outline
    outline: bool = False,
    outline_color: tuple[str, str] = ("black", "white"),
    outline_width: tuple[float, float] = (0.3, 0.05),
    # legend
    legend_loc: str | None = "right margin",
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_fontoutline: int | None = None,
    legend_na: bool = True,
    colorbar: bool = True,
    colorbar_position: str = "bottom",
    colorbar_grid: tuple[int, int] | None = None,
    colorbar_tick_size: int = 10,
    colorbar_title_size: int = 12,
    colorbar_width: float | None = None,
    colorbar_height: float | None = None,
    colorbar_spacing: dict[str, float] | None = None,
    # scalebar
    scalebar_dx: _SeqFloat | None = None,
    scalebar_units: _SeqStr | None = None,
    # title and axis
    title: _SeqStr | None = None,
    axis_label: _SeqStr | None = None,
    fig: Figure | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    return_ax: bool = False,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    # kwargs
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> Axes | Sequence[Axes] | None:
    """
    Plot spatial omics data.

    Use ``library_id`` to select the image. If multiple ``library_id`` are available, use ``library_key`` in
    :attr:`anndata.AnnData.obs` to plot the subsets.
    Use ``crop_coord`` to crop the spatial plot based on coordinate boundaries.

    This function has few key assumptions about how coordinates and libraries are handled:

        - The arguments ``library_key`` and ``library_id`` control which dataset is plotted.
          If multiple libraries are present, specifying solely ``library_key`` will suffice, and all unique libraries
          will be plotted sequentially. To select specific libraries, use the ``library_id`` argument.
        - The argument ``color`` controls which features in obs/var are plotted. They are plotted for all
          available/specified libraries. The argument ``groups`` can be used to select categories to be plotted.
          This is valid only for categorical features in :attr:`anndata.AnnData.obs`.
        - If multiple ``library_id`` are available, arguments such as ``size`` and ``crop_coord`` accept lists to
          selectively customize different ``library_id`` plots. This requires that the length of such lists matches
          the number of unique libraries in the dataset.
        - Coordinates are in the pixel space of the source image, so an equal aspect ratio is assumed.
        - The origin *(0, 0)* is on the top left, as is common convention with image data.
        - The plotted points (dots) do not have a real "size" but it is relative to their coordinate/pixel space.
          This does not hold if no image is plotted, then the size corresponds to points size passed to
          :meth:`matplotlib.axes.Axes.scatter`.

    If :attr:`anndata.AnnData.uns` ``['spatial']`` is present, use ``img_key``, ``seg_key`` and
    ``size_key`` arguments to find values for ``img``, ``seg`` and ``size``.
    Alternatively, these values can be passed directly via ``img``.

    Args:
        %(adata)s
        %(shape)s
        %(color)s
        %(groups)s
        %(library_id)s
        %(library_key)s
        %(spatial_key)s
        %(plotting_image)s
        %(plotting_segment)s
        %(plotting_features)s
        %(groups)s
        palette
        Categorical colormap for the clusters.
        norm
        Colormap normalization, see :class:`matplotlib.colors.Normalize`.
            na_color
            Color to be used for NAs values, if present.
            size
            Size of the scatter point/shape. In case of segmentation it represents the
            scaling factor for shape (accessed via ``size_key``).
            size_key
        Key of pixel size of shapes to be plotted, stored in :attr:`anndata.AnnData.uns`.
            scale_factor
            Scaling factor used to map from coordinate space to pixel space.
            Found by default if ``library_id`` can be resolved. Otherwise, defaults to 1.
            crop_coord
            Coordinates to use for cropping the image (left, right, top, bottom).
            These coordinates are expected to be in pixel space and will be transformed by ``scale_factor``.
            connectivity_key
            Key for neighbors graph to plot.

    Returns:
        %(plotting_returns)s
    """
    from squidpy._docs import d
    from squidpy.gr._utils import _assert_spatial_basis
    from squidpy.pl._utils import sanitize_anndata, save_fig
    
    sanitize_anndata(adata)
    _assert_spatial_basis(adata, spatial_key)

    scalebar_kwargs = dict(scalebar_kwargs)
    edges_kwargs = dict(edges_kwargs)

    color_params = _prepare_args_plot(
        adata=adata,
        shape=shape,
        color=color,
        groups=groups,
        alpha=alpha,
        img_alpha=img_alpha,
        use_raw=use_raw,
        layer=layer,
        palette=palette,
    )

    spatial_params = _image_spatial_attrs(
        adata=adata,
        shape=shape,
        spatial_key=spatial_key,
        library_id=library_id,
        library_key=library_key,
        img=img,
        img_res_key=img_res_key,
        img_channel=img_channel,
        seg=seg,
        seg_key=seg_key,
        cell_id_key=seg_cell_id,
        scale_factor=scale_factor,
        size=size,
        size_key=size_key,
        img_cmap=img_cmap,
    )

    coords, crops = _set_coords_crops(
        adata=adata,
        spatial_params=spatial_params,
        spatial_key=spatial_key,
        crop_coord=crop_coord,
    )

    fig_params, cmap_params, scalebar_params, kwargs = _prepare_params_plot(
        color_params=color_params,
        spatial_params=spatial_params,
        spatial_key=spatial_key,
        wspace=wspace,
        hspace=hspace,
        ncols=ncols,
        cmap=cmap,
        norm=norm,
        library_first=library_first,
        img_cmap=img_cmap,
        frameon=frameon,
        na_color=na_color,
        title=title,
        axis_label=axis_label,
        scalebar_dx=scalebar_dx,
        scalebar_units=scalebar_units,
        dpi=dpi,
        figsize=figsize,
        fig=fig,
        ax=ax,
        **kwargs,
    )

    for count, (_lib_count, value_to_plot) in enumerate(itertools.product(*fig_params.iter_panels)):
        if not library_first:
            _lib_count, value_to_plot = value_to_plot, _lib_count

        _size = spatial_params.size[_lib_count]
        _img = spatial_params.img[_lib_count]
        _seg = spatial_params.segment[_lib_count]
        _cell_id = spatial_params.cell_id[_lib_count]
        _crops = crops[_lib_count]
        _lib = spatial_params.library_id[_lib_count]
        _coords = coords[_lib_count]  # TODO: do we want to order points? for now no, skip
        adata_sub, coords_sub, image_sub = _subs(
            adata,
            _coords,
            _img,
            library_key=library_key,
            library_id=_lib,
            crop_coords=_crops,
            groups_key=value_to_plot,
            groups=color_params.groups,
        )
        color_source_vector, color_vector, categorical = _set_color_source_vec(
            adata_sub,
            value_to_plot,
            layer=layer,
            use_raw=color_params.use_raw,
            alt_var=alt_var,
            groups=color_params.groups,
            palette=palette,
            na_color=na_color,
            alpha=color_params.alpha,
        )

        # set frame and title
        ax = _set_ax_title(fig_params, count, value_to_plot)

        # plot edges and arrows if needed. Do it here cause otherwise image is on top.
        if connectivity_key is not None:
            _plot_edges(
                adata_sub,
                coords_sub,
                connectivity_key,
                ax=ax,
                edges_width=edges_width,
                edges_color=edges_color,
                **edges_kwargs,
            )

        if _seg is None and _cell_id is None:
            outline_params, kwargs = _set_outline(
                size=_size,
                outline=outline,
                outline_width=outline_width,
                outline_color=outline_color,
                **kwargs,
            )

            ax, cax = _plot_scatter(
                coords=coords_sub,
                ax=ax,
                outline_params=outline_params,
                cmap_params=cmap_params,
                color_params=color_params,
                size=_size,
                color_vector=color_vector,
                na_color=na_color,
                **kwargs,
            )
        elif _seg is not None and _cell_id is not None:
            ax, cax = _plot_segment(
                seg=_seg,
                cell_id=_cell_id,
                color_vector=color_vector,
                color_source_vector=color_source_vector,
                seg_contourpx=seg_contourpx,
                seg_outline=seg_outline,
                na_color=na_color,
                ax=ax,
                cmap_params=cmap_params,
                color_params=color_params,
                categorical=categorical,
                **kwargs,
            )

        # Handle axis decoration manually (essential parts from _decorate_axs)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.autoscale_view()  # needed when plotting points but no image
        
        # Handle image display and axis orientation (core logic from _decorate_axs)
        if image_sub is not None:
            ax.imshow(image_sub, cmap=img_cmap, alpha=img_alpha)
        else:
            ax.set_aspect("equal")
            ax.invert_yaxis()
        
        # Set title
        if title is None:
            title = f"Overlay: {', '.join(color)}"
        ax.set_title(title)

        # Add scalebar if specified
        if scalebar_dx is not None:
            from matplotlib_scalebar.scalebar import ScaleBar
            scalebar_dx, scalebar_units = _get_scalebar(scalebar_dx, scalebar_units, 1)
            if scalebar_dx and scalebar_units:
                scalebar = ScaleBar(scalebar_dx[0], units=scalebar_units[0], **scalebar_kwargs)
                ax.add_artist(scalebar)

    if fig_params.fig is not None and save is not None:
        save_fig(fig_params.fig, path=save)

    if return_ax:
        return fig_params.ax if fig_params.axs is None else fig_params.axs


def _wrap_signature(wrapper: Callable[[Any], Any]) -> Callable[[Any], Any]:
    import inspect

    name = wrapper.__name__
    params = inspect.signature(_spatial_plot).parameters.copy()
    wrapper_sig = inspect.signature(wrapper)
    wrapper_params = wrapper_sig.parameters.copy()

    if name == "spatial_scatter":
        params_remove = [
            "seg",
            "seg_cell_id",
            "seg_key",
            "seg_contourpx",
            "seg_outline",
        ]
        wrapper_remove = ["shape"]
    elif name == "spatial_segment":
        params_remove = ["shape", "size", "size_key", "scale_factor"]
        wrapper_remove = [
            "seg_cell_id",
            "seg",
            "seg_key",
            "seg_contourpx",
            "seg_outline",
        ]
    else:
        raise NotImplementedError(f"Docstring interpolation not implemented for `{name}`.")

    for key in params_remove:
        params.pop(key)
    for key in wrapper_remove:
        wrapper_params.pop(key)

    params.update(wrapper_params)
    annotations = {k: v.annotation for k, v in params.items() if v.annotation != inspect.Parameter.empty}
    if wrapper_sig.return_annotation is not inspect.Signature.empty:
        annotations["return"] = wrapper_sig.return_annotation

    wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        list(params.values()), return_annotation=wrapper_sig.return_annotation
    )
    wrapper.__annotations__ = annotations

    return wrapper


def spatial_scatter(
    adata: AnnData,
    shape: str | None = "circle",
    **kwargs: Any,
) -> Axes | Sequence[Axes] | None:
    """
    Plot spatial omics data with data overlayed on top.

    The plotted shapes (circles, squares or hexagons) have a real "size" with respect to their
    coordinate space, which can be specified via the ``size`` or ``size_key`` argument.

        - Use ``img_key`` to display the image in the background.
        - Use ``library_id`` to select the image. By default, ``'hires'`` is attempted.
        - Use ``img_alpha``, ``img_cmap`` and ``img_channel`` to control how it is displayed.
        - Use ``size`` to scale the size of the shapes plotted on top.

    If no image is plotted, it defaults to a scatter plot, see :meth:`matplotlib.axes.Axes.scatter`.

    %(spatial_plot.summary_ext)s

    .. seealso::
        - :func:`squidpy.pl.spatial_segment` on how to plot spatial data with segmentation masks on top.

    Args:
        %(adata)s
        %(shape)s
        %(color)s
        %(groups)s
        %(library_id)s
        %(library_key)s
        %(spatial_key)s
        %(plotting_image)s
        %(plotting_features)s

    Returns:
        %(spatial_plot.returns)s
    """
    from squidpy._docs import d
    
    return _spatial_plot(adata, shape=shape, seg=None, seg_key=None, **kwargs)


def spatial_segment(
    adata: AnnData,
    seg_cell_id: str,
    seg: bool | Sequence[np.ndarray] | None = True,
    seg_key: str = "segmentation",
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    **kwargs: Any,
) -> Axes | Sequence[Axes] | None:
    """
    Plot spatial omics data with segmentation masks on top.

    Argument ``seg_cell_id`` in :attr:`anndata.AnnData.obs` controls unique segmentation mask's ids to be plotted.
    By default, ``'segmentation'``, ``seg_key`` for the segmentation and ``'hires'`` for the image is attempted.

        - Use ``seg_key`` to display the image in the background.
        - Use ``seg_contourpx`` or ``seg_outline`` to control how the segmentation mask is displayed.

    %(spatial_plot.summary_ext)s

    .. seealso::
        - :func:`squidpy.pl.spatial_scatter` on how to plot spatial data with overlayed data on top.

    Args:
        %(adata)s
        %(plotting_segment)s
        %(color)s
        %(groups)s
        %(library_id)s
        %(library_key)s
        %(spatial_key)s
        %(plotting_image)s
        %(plotting_features)s

    Returns:
        %(spatial_plot.returns)s
    """
    from squidpy._docs import d
    
    return _spatial_plot(
        adata,
        seg=seg,
        seg_key=seg_key,
        seg_cell_id=seg_cell_id,
        seg_contourpx=seg_contourpx,
        seg_outline=seg_outline,
        **kwargs,
    )


def spatial_segment_overlay(
    adata: AnnData,
    seg_cell_id: str,
    color: str | Sequence[str],
    seg: bool | Sequence[np.ndarray] | None = True,
    seg_key: str = "segmentation",
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    alpha: float = 0.5,
    cmaps: str | Sequence[str] | None = None,
    library_id: _SeqStr | None = None,
    library_key: str | None = None,
    spatial_key: str = "spatial",  
    img: bool | Sequence[np.ndarray] | None = True,
    img_res_key: str | None = "hires",
    img_alpha: float | None = None,
    img_cmap: Colormap | str | None = None,
    img_channel: int | list[int] | None = None,
    use_raw: bool | None = None,
    layer: str | None = None,
    alt_var: str | None = None,
    groups: _SeqStr | None = None,
    palette: Palette_t = None,
    norm: _Normalize | None = None,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),
    size: _SeqFloat | None = None,
    size_key: str | None = "spot_diameter_fullres",
    scale_factor: _SeqFloat | None = None,
    crop_coord: _CoordTuple | Sequence[_CoordTuple] | None = None,
    connectivity_key: str | None = None,
    edges_width: float = 1.0,
    edges_color: str | Sequence[str] | Sequence[float] = "grey",
    frameon: bool | None = None,
    legend_loc: str | None = "right margin",
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_fontoutline: int | None = None,
    legend_na: bool = True,
    colorbar: bool = True,
    colorbar_position: str = "bottom",
    colorbar_grid: tuple[int, int] | None = None,
    colorbar_tick_size: int = 10,
    colorbar_title_size: int = 12,
    colorbar_width: float | None = None,
    colorbar_height: float | None = None,
    colorbar_spacing: dict[str, float] | None = None,
    scalebar_dx: _SeqFloat | None = None,
    scalebar_units: _SeqStr | None = None,
    title: _SeqStr | None = None,
    axis_label: _SeqStr | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    return_ax: bool = False,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> Axes | None:
    """
    Plot multiple genes overlaid on the same spatial segmentation plot.
    
    This function allows visualization of multiple genes in the same spatial context,
    using different colors and transparency to show expression overlap and co-localization.
    
    Args:
        %(adata)s
        seg_cell_id
        Key in :attr:`anndata.AnnData.obs` that contains unique cell IDs for segmentation.
            color
            Gene names or features to plot. Can be a single gene or list of genes.
            seg
            Segmentation mask. If `True`, uses segmentation from `adata.uns['spatial']`.
            seg_key
            Key for segmentation mask in `adata.uns['spatial'][library_id]['images']`.
            seg_contourpx
            Contour width in pixels. If specified, segmentation boundaries will be eroded.
            seg_outline
            If `True`, show segmentation boundaries.
            alpha
            Transparency level for gene expression overlay (0-1).
            cmaps
            Colormap(s) for each gene. If single string, uses same colormap for all genes.
            If list, should match length of `color` parameter.
            %(library_id)s
            %(library_key)s  
            %(spatial_key)s
            %(plotting_image)s
            %(plotting_features)s
            groups
            Categories to plot for categorical features.
            palette
            Color palette for categorical features.
            norm
            Colormap normalization.
            na_color
            Color for NA/missing values.
            colorbar
            Whether to show colorbars for each gene.
            colorbar_position
        Position of colorbars: 'bottom', 'right', or 'none'.
            colorbar_grid
            Grid layout for colorbars as (rows, cols). If None, auto-determined.
            colorbar_tick_size
            Font size for colorbar tick labels.
            colorbar_title_size
            Font size for colorbar titles.
            colorbar_width
            Width of individual colorbars. If None, uses default values.
            colorbar_height
            Height of individual colorbars. If None, uses default values.
            colorbar_spacing
        Dictionary with spacing parameters: 'hspace' (vertical gaps), 'wspace' (horizontal gaps).
            If None, uses default spacing.
            title
            Plot title.
            ax
            Matplotlib axes object to plot on.
            return_ax
            Whether to return the axes object.
            figsize
            Figure size (width, height).
            save
            Path to save the figure.
    
    Returns:
        If `return_ax` is `True`, returns matplotlib axes object, otherwise `None`.
    
    Examples:
        >>> import squidpy as sq
        >>> # Overlay two genes with different colors
        >>> sq.pl.spatial_segment_overlay(
        ...     adata, 
        ...     seg_cell_id='cell_id',
        ...     color=['gene1', 'gene2'],
        ...     cmaps=['Reds', 'Blues'],
        ...     alpha=0.7
        ... )
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.gridspec import GridSpec
    from squidpy.pl._utils import sanitize_anndata, save_fig
    import matplotlib as mpl
    from squidpy.gr._utils import _assert_spatial_basis
    
    sanitize_anndata(adata)
    _assert_spatial_basis(adata, spatial_key)
    
    # Ensure color is a list
    if isinstance(color, str):
        color = [color]
    
    # Handle colormaps
    if cmaps is None:
        # Default colors for overlay: Red, Green, Blue, Cyan, Magenta, Yellow
        default_colors = ['#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#FFFF00']
        cmaps = []
        for i, _ in enumerate(color):
            base_color = default_colors[i % len(default_colors)]
            base_rgb = to_rgb(base_color)
            cmap_colors = [
                base_rgb + (0.0,),  # Transparent
                base_rgb + (1.0,)   # Opaque
            ]
            cmap = LinearSegmentedColormap.from_list(f'overlay_{i}', cmap_colors, N=100)
            cmaps.append(cmap)
    elif isinstance(cmaps, str):
        cmaps = [cmaps] * len(color)
    
    # Create figure and axes with colorbar layout
    if ax is None:
        figsize = figsize or (10, 8)
        
        if colorbar and colorbar_position != "none":
            # Set default colorbar dimensions and spacing
            if colorbar_spacing is None:
                colorbar_spacing = {}
            
            # Determine colorbar grid layout
            if colorbar_grid is None:
                if colorbar_position == "bottom":
                    if len(color) <= 3:
                        colorbar_grid = (1, len(color))
                    else:
                        n_rows = (len(color) + 2) // 3  # Round up division
                        colorbar_grid = (n_rows, 3)
                elif colorbar_position == "right":
                    colorbar_grid = (len(color), 1)
            
            # Create GridSpec layout with customizable dimensions
            if colorbar_position == "bottom":
                # Default dimensions for bottom colorbars
                default_width = 0.2 if colorbar_width is None else colorbar_width
                default_height = 0.08 if colorbar_height is None else colorbar_height
                default_hspace = colorbar_spacing.get('hspace', 0.4)
                default_wspace = colorbar_spacing.get('wspace', 0.3)
                
                gs = GridSpec(
                    nrows=colorbar_grid[0] + 1,
                    ncols=max(colorbar_grid[1], 1) + 2,
                    width_ratios=[0.1, *[default_width] * max(colorbar_grid[1], 1), 0.1],
                    height_ratios=[1, *[default_height] * colorbar_grid[0]],
                    hspace=default_hspace,
                    wspace=default_wspace,
                    figure=plt.figure(figsize=figsize, dpi=dpi)
                )
                fig = gs.figure
                ax = fig.add_subplot(gs[0, :])
                ax.grid(False)
            elif colorbar_position == "right":
                # Default dimensions for right colorbars
                default_width = 0.15 if colorbar_width is None else colorbar_width
                default_height = 0.2 if colorbar_height is None else colorbar_height
                default_hspace = colorbar_spacing.get('hspace', 0.3)
                default_wspace = colorbar_spacing.get('wspace', 0.1)
                
                gs = GridSpec(
                    nrows=max(colorbar_grid[0], 1) + 2,
                    ncols=colorbar_grid[1] + 1,
                    width_ratios=[1, *[default_width] * colorbar_grid[1]],
                    height_ratios=[0.1, *[default_height] * max(colorbar_grid[0], 1), 0.1],
                    hspace=default_hspace,
                    wspace=default_wspace,
                    figure=plt.figure(figsize=figsize, dpi=dpi)
                )
                fig = gs.figure
                ax = fig.add_subplot(gs[:, 0])
                ax.grid(False)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.grid(False)
            gs = None  # No GridSpec when no colorbars
    else:
        fig = ax.figure if ax.figure is not None else fig
        gs = None  # No GridSpec when external ax is provided
    
    # Get spatial parameters once
    spatial_params = _image_spatial_attrs(
        adata=adata,
        spatial_key=spatial_key,
        library_id=library_id,
        library_key=library_key,
        img=img,
        img_res_key=img_res_key,
        img_channel=img_channel,
        seg=seg, 
        seg_key=seg_key,
        cell_id_key=seg_cell_id,
        scale_factor=scale_factor,
        size=size,
        size_key=size_key,
        img_cmap=img_cmap,
    )
    
    # Get coordinates and crops
    coords, crops = _set_coords_crops(
        adata=adata,
        spatial_params=spatial_params,
        spatial_key=spatial_key,
        crop_coord=crop_coord,
    )
    
    # Use first library for now (could be extended for multiple libraries)
    _lib_count = 0
    _size = spatial_params.size[_lib_count]
    _img = spatial_params.img[_lib_count]
    _seg = spatial_params.segment[_lib_count]
    _cell_id = spatial_params.cell_id[_lib_count]
    _crops = crops[_lib_count]
    _lib = spatial_params.library_id[_lib_count]
    _coords = coords[_lib_count]
    
    # Subset data
    adata_sub, coords_sub, image_sub = _subs(
        adata,
        _coords,
        _img,
        library_key=library_key,
        library_id=_lib,
        crop_coords=_crops,
    )
    
    # Store colorbar information for each gene
    colorbar_info = []
    
    # Plot each gene as an overlay
    for i, (gene, cmap) in enumerate(zip(color, cmaps, strict=False)):
        # Get color vector for this gene
        color_source_vector, color_vector, categorical = _set_color_source_vec(
            adata_sub,
            gene,
            layer=layer,
            use_raw=use_raw,
            alt_var=alt_var,
            groups=groups,
            palette=palette,
            na_color=na_color,
            alpha=alpha,
        )
        
        if _seg is not None and _cell_id is not None:
            # Create colormap parameters
            from matplotlib.colors import Normalize
            norm_to_use = norm or Normalize()
            
            # Fix normalization if needed
            if not categorical and (norm_to_use.vmin is None or norm_to_use.vmax is None):
                valid_values = color_vector[~pd.isna(color_vector)] if hasattr(color_vector, 'isna') else color_vector[~np.isnan(color_vector)]
                if len(valid_values) > 0:
                    norm_to_use.vmin = np.min(valid_values)
                    norm_to_use.vmax = np.max(valid_values)
            
            cmap_params = CmapParams(cmap, img_cmap, norm_to_use)
            color_params = ColorParams(None, [gene], groups, alpha, img_alpha or 1.0, use_raw or False)
            
            # Store colorbar information
            if colorbar and not categorical:
                colorbar_info.append({
                    'gene': gene,
                    'cmap': cmap,
                    'norm': norm_to_use,
                    'values': color_vector
                })
            
            # Plot this gene's segmentation
            ax, cax = _plot_segment(
                seg=_seg,
                cell_id=_cell_id,
                color_vector=color_vector,
                color_source_vector=color_source_vector,
                seg_contourpx=seg_contourpx,
                seg_outline=seg_outline,
                na_color=na_color,
                ax=ax,
                cmap_params=cmap_params,
                color_params=color_params,
                categorical=categorical,
                **kwargs,
            )
    
    # Handle axis decoration manually (essential parts from _decorate_axs)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.autoscale_view()  # needed when plotting points but no image
    
    # Handle image display and axis orientation (core logic from _decorate_axs)
    if image_sub is not None:
        ax.imshow(image_sub, cmap=img_cmap, alpha=img_alpha)
    else:
        ax.set_aspect("equal")
        ax.invert_yaxis()
    
    # Set title
    if title is None:
        title = f"Overlay: {', '.join(color)}"
    ax.set_title(title)
    
    # Add colorbars
    if colorbar and colorbar_position != "none" and colorbar_info:
        # Create colorbar axes
        cbar_axes = []
        n_genes = len(colorbar_info)
        
        if colorbar_position == "bottom":
            for row in range(colorbar_grid[0]):
                for col in range(colorbar_grid[1]):
                    idx = row * colorbar_grid[1] + col
                    if idx < n_genes:
                        cbar_ax = fig.add_subplot(gs[row + 1, col + 1])
                        cbar_axes.append(cbar_ax)
        elif colorbar_position == "right":
            for row in range(min(colorbar_grid[0], n_genes)):
                cbar_ax = fig.add_subplot(gs[row + 1, 1])
                cbar_axes.append(cbar_ax)
        
        # Create colorbars
        for i, (cbar_info, cbar_ax) in enumerate(zip(colorbar_info, cbar_axes)):
            if i < len(cbar_axes):
                # Calculate ticks
                vmin, vmax = cbar_info['norm'].vmin, cbar_info['norm'].vmax
                if vmin is not None and vmax is not None:
                    ticks = [vmin, (vmin + vmax) / 2, vmax]
                    if vmax > 10:
                        ticks = [int(t) for t in ticks]
                    else:
                        ticks = [round(t, 2) for t in ticks]
                    
                    # Create colorbar
                    orientation = "horizontal" if colorbar_position == "bottom" else "vertical"
                    cbar = fig.colorbar(
                        mpl.cm.ScalarMappable(norm=cbar_info['norm'], cmap=cbar_info['cmap']),
                        cax=cbar_ax,
                        orientation=orientation,
                        extend="both",
                        ticks=ticks
                    )
                    
                    # Style colorbar
                    cbar.ax.tick_params(labelsize=colorbar_tick_size)
                    cbar.ax.set_title(cbar_info['gene'], size=colorbar_title_size, pad=10)
    
    # Add scalebar if specified
    if scalebar_dx is not None:
        from matplotlib_scalebar.scalebar import ScaleBar
        scalebar_dx, scalebar_units = _get_scalebar(scalebar_dx, scalebar_units, 1)
        if scalebar_dx and scalebar_units:
            scalebar = ScaleBar(scalebar_dx[0], units=scalebar_units[0], **scalebar_kwargs)
            ax.add_artist(scalebar)
    
    if save is not None:
        save_fig(fig, path=save)
        
    if return_ax:
        return ax