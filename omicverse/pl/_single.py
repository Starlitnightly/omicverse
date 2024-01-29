from ..utils import embedding
import collections.abc as cabc
from copy import copy
from numbers import Integral
from itertools import combinations, product
import matplotlib
from typing import (
    Collection,
    Union,
    Optional,
    Sequence,
    Any,
    Mapping,
    List,
    Tuple,
    Literal,
)
from warnings import warn

import numpy as np
import pandas as pd
from anndata import AnnData
from cycler import Cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.api.types import is_categorical_dtype
from matplotlib import pyplot as pl, colors, colormaps
from matplotlib import rcParams
from matplotlib import patheffects
from matplotlib.colors import Colormap, Normalize
from functools import partial

from scanpy.plotting import _utils 

from scanpy.plotting._utils import (
    _IGraphLayout,
    _FontWeight,
    _FontSize,
    ColorLike,
    VBound,
    circles,
    check_projection,
    check_colornorm,
)

def embedding(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: Union[str, Sequence[float], Sequence[str]] = 'grey',
    neighbors_key: Optional[str] = None,
    arrows: bool = False,
    arrows_kwds: Optional[Mapping[str, Any]] = None,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    dimensions: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    layer: Optional[str] = None,
    projection: Literal['2d', '3d'] = '2d',
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = 'bold',
    legend_loc: str = 'right margin',
    legend_fontoutline: Optional[int] = None,
    colorbar_loc: Optional[str] = "right",
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ('black', 'white'),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    marker: Union[str, Sequence[str]] = '.',
    **kwargs,
) -> Union[Figure, Axes, None]:
    """\
    Scatter plot for user specified embedding basis (e.g. umap, pca, etc)

    Arguments:
        adata: Annotated data matrix.
        basis: Name of the `obsm` basis to use.
        
    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    """

    return embedding(adata=adata, basis=basis, color=color, 
                     gene_symbols=gene_symbols, use_raw=use_raw, 
                     sort_order=sort_order, edges=edges, 
                     edges_width=edges_width, edges_color=edges_color, 
                     neighbors_key=neighbors_key, arrows=arrows, 
                     arrows_kwds=arrows_kwds, groups=groups, 
                     components=components, dimensions=dimensions, 
                     layer=layer, projection=projection, scale_factor=scale_factor,
                       color_map=color_map, cmap=cmap, palette=palette, 
                       na_color=na_color, na_in_legend=na_in_legend, 
                       size=size, frameon=frameon, legend_fontsize=legend_fontsize, 
                       legend_fontweight=legend_fontweight, legend_loc=legend_loc, 
                       legend_fontoutline=legend_fontoutline, colorbar_loc=colorbar_loc, 
                       vmax=vmax, vmin=vmin, vcenter=vcenter, norm=norm, 
                       add_outline=add_outline, outline_width=outline_width, 
                       outline_color=outline_color, ncols=ncols, hspace=hspace,
                         wspace=wspace, title=title, show=show, save=save, ax=ax,
                           return_fig=return_fig, marker=marker, **kwargs)