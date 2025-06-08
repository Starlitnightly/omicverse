"""
Utility functions for plotting in Palantir
"""

from typing import Optional, Union, Dict, List, Tuple, Any, Callable
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextlib
import logging


@contextlib.contextmanager
def no_mellon_log_messages():
    # Import mellon locally to avoid JAX fork warnings in other parts of the code
    import mellon
    current_level = mellon.logger.level
    mellon.logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        mellon.logger.setLevel(current_level)

def _scatter_with_colorbar(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    colorbar_label: Optional[str] = None,
    s: float = 5,
    cmap: Union[str, matplotlib.colors.Colormap] = "viridis",
    norm: Optional[Normalize] = None,
    alpha: float = 1.0,
    **kwargs,
) -> Tuple[Axes, matplotlib.colorbar.Colorbar]:
    """Helper function to create scatter plot with colorbar.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    x : np.ndarray
        X-coordinates for scatter plot.
    y : np.ndarray
        Y-coordinates for scatter plot.
    c : np.ndarray
        Values for color mapping.
    colorbar_label : str, optional
        Label for the colorbar. Default is None.
    s : float, optional
        Size of scatter points. Default is 5.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the scatter plot. Default is 'viridis'.
    norm : Normalize, optional
        Normalization for colormap. Default is None.
    alpha : float, optional
        Transparency of scatter points. Default is 1.0.
    **kwargs : dict
        Additional keyword arguments to pass to plt.scatter.

    Returns
    -------
    Tuple[Axes, matplotlib.colorbar.Colorbar]
        The axes object and the colorbar object.
    """
    sc = ax.scatter(x, y, c=c, s=s, cmap=cmap, norm=norm, alpha=alpha, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax, orientation="vertical")
    if colorbar_label:
        cbar.set_label(colorbar_label)
    return ax, cbar


def _highlight_cells(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    deselected_color: str = "lightgray",
    selected_color: str = "crimson",
    s_selected: float = 10,
    s_deselected: float = 3,
    alpha_deselected: float = 0.5,
    alpha_selected: float = 1.0,
    **kwargs,
) -> Axes:
    """Helper function to highlight cells in scatter plot based on mask.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    x : np.ndarray
        X-coordinates for scatter plot.
    y : np.ndarray
        Y-coordinates for scatter plot.
    mask : np.ndarray
        Boolean mask for selecting cells to highlight.
    deselected_color : str, optional
        Color for non-highlighted cells. Default is "lightgray".
    selected_color : str, optional
        Color for highlighted cells. Default is "crimson".
    s_selected : float, optional
        Size of highlighted scatter points. Default is 10.
    s_deselected : float, optional
        Size of non-highlighted scatter points. Default is 3.
    alpha_deselected : float, optional
        Transparency of non-highlighted cells. Default is 0.5.
    alpha_selected : float, optional
        Transparency of highlighted cells. Default is 1.0.
    **kwargs : dict
        Additional keyword arguments to pass to plt.scatter.

    Returns
    -------
    Axes
        The modified axes object.
    """
    ax.scatter(
        x[~mask],
        y[~mask],
        c=deselected_color,
        s=s_deselected,
        alpha=alpha_deselected,
        label="Other Cells",
        **kwargs,
    )
    ax.scatter(
        x[mask],
        y[mask],
        c=selected_color,
        s=s_selected,
        alpha=alpha_selected,
        label="Selected Cells",
        **kwargs,
    )
    return ax


def _add_legend(
    ax: Axes,
    handles: Optional[List] = None,
    labels: Optional[List[str]] = None,
    loc: str = "best",
    title: Optional[str] = None,
    **kwargs,
) -> matplotlib.legend.Legend:
    """Helper function to add legend to plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to add legend to.
    handles : List, optional
        List of artists (lines, patches) to be added to the legend. Default is None.
    labels : List[str], optional
        List of labels for the legend. Default is None.
    loc : str, optional
        Location of the legend. Default is "best".
    title : str, optional
        Title for the legend. Default is None.
    **kwargs : dict
        Additional keyword arguments to pass to ax.legend().

    Returns
    -------
    matplotlib.legend.Legend
        The legend object.
    """
    if handles is not None and labels is not None:
        legend = ax.legend(handles, labels, loc=loc, title=title, **kwargs)
    else:
        legend = ax.legend(loc=loc, title=title, **kwargs)
    return legend


def _setup_axes(
    figsize: Tuple[float, float] = (6, 6),
    ax: Optional[Axes] = None,
    fig: Optional[plt.Figure] = None,
    **kwargs,
) -> Tuple[plt.Figure, Axes]:
    """Helper function to set up figure and axes for plotting.

    Parameters
    ----------
    figsize : Tuple[float, float], optional
        Size of the figure (width, height) in inches. Default is (6, 6).
    ax : Axes, optional
        Existing axes to plot on. Default is None.
    fig : Figure, optional
        Existing figure to plot on. Default is None.
    **kwargs : dict
        Additional keyword arguments to pass to plt.subplots().

    Returns
    -------
    Tuple[plt.Figure, Axes]
        The figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
    elif fig is None:
        fig = ax.figure
    return fig, ax


def _get_palantir_fates_colors(
    ad,
    fate_names: List[str],
    palantir_fates_colors: Optional[Union[List[str], Dict[str, str]]] = None
) -> Dict[str, str]:
    """
    Generate or update the mapping from branch names to colors.
    
    This utility checks if ad.uns already contains predefined colors.
    Then, if the `palantir_fates_colors` parameter is provided, its values are merged
    (with user-specified colors taking precedence). For any missing branch the function 
    generates a new color ensuring that no color is used twice.
    
    Parameters
    ----------
    ad : AnnData
        The annotated data object from which .uns will be checked.
    fate_names : list of str
        List of branch (fate) names.
    palantir_fates_colors : dict or list or None, optional
        If a dict, keys should be branch names with a color for each.
        If a list, its order is assumed to correspond to fate_names.
        If None, only the predefined colors (if any) and generated defaults are used.

    Returns
    -------
    dict
        Mapping from branch names to colors.
    """
    # Get any predefined colors stored in ad.uns.
    predefined = {}
    if "palantir_fates_colors" in ad.uns:
        predefined = ad.uns["palantir_fates_colors"]
    
    # Process user-provided colors from argument.
    provided = {}
    if palantir_fates_colors is not None:
        if isinstance(palantir_fates_colors, dict):
            provided = palantir_fates_colors
        elif isinstance(palantir_fates_colors, list):
            if len(palantir_fates_colors) < len(fate_names):
                raise ValueError("Provided color list length is less than the number of branch names.")
            provided = {name: clr for name, clr in zip(fate_names, palantir_fates_colors)}
        else:
            raise TypeError("palantir_fates_colors must be a dict, list, or None.")
    
    # Merge: user-provided takes precedence, then predefined.
    mapping = {}
    for branch in fate_names:
        if branch in provided:
            mapping[branch] = provided[branch]
        elif branch in predefined:
            mapping[branch] = predefined[branch]
    
    # Collect already used colors to exclude duplicates.
    used_colors = set(mapping.values())
    
    # Generate colors for missing branches.
    missing = [branch for branch in fate_names if branch not in mapping]
    if missing:
        # Get the default color cycle.
        default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
            'color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        )
        # Create a generator that skips colors already used.
        def color_generator(exclude):
            for clr in default_cycle:
                if clr not in exclude:
                    yield clr
            hex_digits = np.array(list("0123456789ABCDEF"))
            # If default cycle is exhausted, generate random colors.
            while True:
                new_color = "#" + "".join(np.random.choice(hex_digits, size=6))
                if new_color not in exclude:
                    yield new_color

        gen = color_generator(used_colors)
        for branch in missing:
            new_color = next(gen)
            mapping[branch] = new_color
            used_colors.add(new_color)
    
    return mapping


def _plot_arrows(x, y, n=5, ax=None, arrowprops=dict(), arrow_zorder=2, head_offset=0.0, **kwargs):
    """
    Helper function to plot arrows on a trajectory line.
    
    The new 'head_offset' parameter (as a fraction of the segment length)
    moves the arrow head slightly forward.
    
    Parameters
    ----------
    x, y : array-like
        Coordinates of the trajectory points.
    n : int, optional
        Number of arrows to plot. Defaults to 5.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    arrowprops : dict, optional
        Properties for the arrow style.
    arrow_zorder : int, optional
        zorder level for both the line and arrow annotations.
    head_offset : float, optional
        Fraction of the segment length to move the arrow head forward.
    **kwargs :
        Extra keyword arguments passed to the plot function.
    
    Returns
    -------
    matplotlib.axes.Axes
        The axis with the arrows plotted.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    default_kwargs = {"color": "black", "zorder": arrow_zorder}
    default_kwargs.update(kwargs)
    
    # Plot the trajectory line.
    ax.plot(x, y, **default_kwargs)
    
    if n <= 0:
        return ax
    
    default_arrowprops = dict(arrowstyle="->", lw=1, mutation_scale=20)
    default_arrowprops["color"] = default_kwargs.get("color", "black")
    default_arrowprops.update(arrowprops)
    
    total_points = len(x)
    section_length = total_points // n
    
    for i in range(n):
        idx = total_points - i * section_length
        if idx < 2:
            break
        # Compute the vector from the previous point to the arrow head.
        dx = x[idx - 1] - x[idx - 2]
        dy = y[idx - 1] - y[idx - 2]
        norm = (dx**2 + dy**2) ** 0.5
        # Compute the forward offset.
        if norm != 0:
            offset_dx = head_offset * dx / norm
            offset_dy = head_offset * dy / norm
        else:
            offset_dx = offset_dy = 0
        # Adjust the arrow head coordinates.
        target = (x[idx - 1] + offset_dx, y[idx - 1] + offset_dy)
        
        ax.annotate(
            "",
            xy=target,
            xytext=(x[idx - 2], y[idx - 2]),
            arrowprops=default_arrowprops,
            zorder=arrow_zorder,
        )
    return ax