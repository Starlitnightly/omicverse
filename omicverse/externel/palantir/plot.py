from typing import Union, Optional, List, Tuple, Dict, Literal, Sequence
import collections.abc as cabc
import warnings
from copy import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import scanpy as sc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.colors import Normalize, Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scanpy.plotting._tools.scatterplots import (
    _get_color_source_vector,
    _color_vector,
    _get_vboundnorm,
    _get_palette,
    _FontSize,
    _FontWeight,
    VBound,
)
from scanpy.plotting._utils import check_colornorm


from .presults import PResults
from .validation import (
    _validate_obsm_key,
    _validate_varm_key,
    _validate_gene_trend_input,
)
from . import config


class FigureGrid:
    """
    Generates a grid of axes for plotting
    axes can be iterated over or selected by number. e.g.:
    >>> # iterate over axes and plot some nonsense
    >>> fig = FigureGrid(4, max_cols=2)
    >>> for i, ax in enumerate(fig):
    >>>     plt.plot(np.arange(10) * i)
    >>> # select axis using indexing
    >>> ax3 = fig[3]
    >>> ax3.set_title("I'm axis 3")
    """

    # Figure Grid is favorable for displaying multiple graphs side by side.

    def __init__(self, n: int, max_cols=3, scale=3):
        """
        :param n: number of axes to generate
        :param max_cols: maximum number of axes in a given row
        """

        self.n = n
        self.nrows = int(np.ceil(n / max_cols))
        self.ncols = int(min((max_cols, n)))
        figsize = self.ncols * scale, self.nrows * scale

        # create figure
        self.gs = plt.GridSpec(nrows=self.nrows, ncols=self.ncols)
        self.figure = plt.figure(figsize=figsize)

        # create axes
        self.axes = {}
        for i in range(n):
            row = int(i // self.ncols)
            col = int(i % self.ncols)
            self.axes[i] = plt.subplot(self.gs[row, col])

    def __getitem__(self, item):
        return self.axes[item]

    def __iter__(self):
        for i in range(self.n):
            yield self[i]


def get_fig(fig=None, ax=None):
    """fills in any missing axis or figure with the currently active one
    :param ax: matplotlib Axis object
    :param fig: matplotlib Figure object
    """
    if not fig:
        fig = plt.figure()
    if not ax:
        ax = plt.gca()
    return fig, ax


def density_2d(x, y):
    """return x and y and their density z, sorted by their density (smallest to largest)
    :param x:
    :param y:
    :return:
    """
    xy = np.vstack([np.ravel(x), np.ravel(y)])
    z = gaussian_kde(xy)(xy)
    i = np.argsort(z)
    return np.ravel(x)[i], np.ravel(y)[i], np.arcsinh(z[i])


def plot_molecules_per_cell_and_gene(data, fig=None, ax=None):
    height = 4
    width = 12
    fig = plt.figure(figsize=[width, height])
    gs = plt.GridSpec(1, 3)
    colsum = np.log10(data.sum(axis=0))
    rowsum = np.log10(data.sum(axis=1))
    for i in range(3):
        ax = plt.subplot(gs[0, i])

        if i == 0:
            n, bins, patches = ax.hist(rowsum, bins="auto")
            plt.xlabel("Molecules per cell (log10 scale)")
        elif i == 1:
            temp = np.log10(data.astype(bool).sum(axis=0))
            n, bins, patches = ax.hist(temp, bins="auto")
            plt.xlabel("Nonzero cells per gene (log10 scale)")
        else:
            n, bins, patches = ax.hist(colsum, bins="auto")
            plt.xlabel("Molecules per gene (log10 scale)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        ax.tick_params(axis="x", labelsize=8)

    return fig, ax


def cell_types(tsne, clusters, cluster_colors=None, n_cols=5):
    """Plot cell clusters on the tSNE map
    :param tsne: tSNE map
    :param clusters: Results of the determine_cell_clusters function
    """

    # Cluster colors
    if cluster_colors is None:
        set2_colors = matplotlib.colormaps["hsv"](np.linspace(0, 1, len(set(clusters))))
        cluster_colors = pd.Series(
            [matplotlib.colors.rgb2hex(rgba) for rgba in set2_colors],
            index=set(clusters),
        )
    n_clusters = len(cluster_colors)

    # Cell types
    fig = FigureGrid(n_clusters, n_cols)
    for ax, cluster in zip(fig, cluster_colors.index):
        ax.scatter(tsne.loc[:, "x"], tsne.loc[:, "y"], s=3, color="lightgrey")
        cells = clusters.index[clusters == cluster]
        ax.scatter(
            tsne.loc[cells, "x"],
            tsne.loc[cells, "y"],
            s=5,
            color=cluster_colors[cluster],
        )
        ax.set_axis_off()
        ax.set_title(cluster, fontsize=10)

    return fig.figure, fig.axes


def highlight_cells_on_umap(
    data: Union[sc.AnnData, pd.DataFrame],
    cells: Union[List[str], Dict[str, str], pd.Series, pd.Index, np.ndarray, str],
    annotation_offset: float = 0.03,
    s: float = 1,
    s_highlighted: float = 10,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    embedding_basis: str = "X_umap",
) -> Tuple[plt.Figure, plt.Axes]:
    """
     Highlights and annotates specific cells on a UMAP plot.

     Parameters
     ----------
     data : Union[sc.AnnData, pd.DataFrame]
         Either a Scanpy AnnData object or a DataFrame of UMAP coordinates.
     cells : Union[List[str], Dict[str, str], pd.Series, pd.Index, np.ndarray, str]
         Cells to highlight on the UMAP. Can be provided as:
             - a list, dict, or pd.Series: used as cell names (values in dict/Series used as annotations).
             - a pd.Index: cell identifiers matching those in the data's index will be highlighted.
             - a boolean array-like: used as a mask to select cells from the data's index.
             - a string: used to retrieve a boolean mask from the AnnData's .obs attribute.
    annotation_offset : float, optional
         Offset for the annotations in proportion to the data range. Default is 0.03.
     s : float, optional
         Size of the points in the scatter plot. Default is 1.
     s_highlighted : float, optional
         Size of the points in the highlighted cells. Default is 10.
     fig : Optional[plt.Figure], optional
         Matplotlib Figure object. If None, a new figure is created. Default is None.
     ax : Optional[plt.Axes], optional
         Matplotlib Axes object. If None, a new Axes object is created. Default is None.
     embedding_basis : str, optional
         The key to retrieve UMAP results from the AnnData object. Default is 'X_umap'.

     Returns
     -------
     fig : plt.Figure
         A matplotlib Figure object containing the UMAP plot.
     ax : plt.Axes
         The corresponding Axes object.

     Raises
     ------
     KeyError
         If 'embedding_basis' is not found in 'data.obsm'.
     TypeError
         If 'cells' is neither list, dict nor pd.Series.
    """
    if isinstance(data, sc.AnnData):
        if embedding_basis not in data.obsm:
            raise KeyError(f"'{embedding_basis}' not found in .obsm.")
        umap = pd.DataFrame(
            data.obsm[embedding_basis], index=data.obs_names, columns=["x", "y"]
        )
    elif isinstance(data, pd.DataFrame):
        umap = data.copy()
    else:
        raise TypeError("'data' should be either sc.AnnData or pd.DataFrame.")

    if not isinstance(cells, (pd.Series, np.ndarray, pd.Index, list)):
        if isinstance(cells, str):
            if cells not in data.obs.columns:
                raise KeyError(f"'{cells}' not found in .obs.")
            mask = data.obs[cells].astype(bool).values
            cells = {cell: "" for cell in data.obs[mask].index}
        elif not isinstance(cells, dict):
            raise TypeError(
                "'cells' should be either list, dict, pd.Series, pd.Index, string "
                "(as column in .obs), or a boolean array-like."
            )
    elif isinstance(data, sc.AnnData) and len(cells) == data.n_obs:
        try:
            cells = data.obs_names[cells]
        except IndexError:
            raise ValueError(
                "Using 'cells' as boolean index since len(cells) == ad.n_obs but failed."
            )
        cells = {cell: "" for cell in cells}
    elif isinstance(cells, (list, np.ndarray, pd.Index)):
        cells = {cell: "" for cell in cells}
    elif isinstance(cells, pd.Series):
        cells = dict(cells)
    elif isinstance(cells, pd.Index):
        cells = {cell: "" for cell in cells}
    else:
        raise TypeError(
            "'cells' should be either list, dict, pd.Series, pd.Index, string "
            "(as column in .obs), or a boolean array-like."
        )

    xpad, ypad = (umap.max() - umap.min()) * annotation_offset

    fig, ax = get_fig(fig=fig, ax=ax)
    ax.scatter(umap["x"], umap["y"], s=s, color=config.DESELECTED_COLOR)

    for cell, annotation in cells.items():
        if cell in umap.index:
            x, y = umap.loc[cell, ["x", "y"]]
            ax.scatter(x, y, c=config.SELECTED_COLOR, s=s_highlighted)
            if annotation:
                ax.annotate(annotation, (x, y), (x + xpad, y + ypad), "data")
    ax.set_axis_off()

    return fig, ax


def plot_tsne_by_cell_sizes(data, tsne, fig=None, ax=None, vmin=None, vmax=None):
    """Plot tSNE projections of the data with cells colored by molecule counts
    :param data: Expression data, DataFrame-like
    :param tsne: tSNE coordinates, DataFrame-like
    :param fig: matplotlib Figure object, optional
    :param ax: matplotlib Axis object, optional
    :param vmin: Minimum molecule count for plotting, optional
    :param vmax: Maximum molecule count for plotting, optional
    """

    sizes = data.sum(axis=1)
    fig, ax = get_fig(fig, ax)
    scatter = ax.scatter(tsne["x"], tsne["y"], s=3, c=sizes, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    plt.colorbar(scatter, ax=ax)
    return fig, ax


def plot_gene_expression(
    data: pd.DataFrame,
    tsne: pd.DataFrame,
    genes: List[str],
    plot_scale: bool = False,
    n_cols: int = 5,
    percentile: int = 0,
    **kwargs,
) -> plt.Figure:
    """
    Plot gene expression overlaid on tSNE maps.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame where each column represents a gene and each row a cell.
    tsne : pd.DataFrame
        A DataFrame containing tSNE coordinates for each cell.
    genes : List[str]
        List of gene symbols to plot on tSNE.
    plot_scale : bool, optional
        If True, include a color scale bar in the plot. Defaults to False.
    n_cols : int, optional
        Number of columns in the subplot grid. Defaults to 5.
    percentile : int, optional
        Percentile to cut off extreme values in gene expression for plotting.
        Defaults to 0, meaning no cutoff.
    kwargs : dict, optional
        Additional keyword arguments to pass to the scatter plot function.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the gene expression.

    Raises
    ------
    ValueError
        If none of the genes in the list are found in the data.
    """

    not_in_dataframe = set(genes).difference(data.columns)
    if not_in_dataframe:
        if len(not_in_dataframe) < len(genes):
            print(
                "The following genes were either not observed in the experiment, "
                "or the wrong gene symbol was used: {!r}".format(not_in_dataframe)
            )
        else:
            raise ValueError(
                "None of the listed genes were observed in the experiment, or the "
                "wrong symbols were used."
            )

    # remove genes missing from experiment
    genes = pd.Series(genes)[pd.Series(genes).isin(data.columns)]

    # Plot
    cells = data.index.intersection(tsne.index)
    fig = FigureGrid(len(genes), n_cols)

    for g, ax in zip(genes, fig):
        # Data
        c = data.loc[cells, g]
        vmin = np.percentile(c[~np.isnan(c)], percentile)
        vmax = np.percentile(c[~np.isnan(c)], 100 - percentile)
        default_args = {
            "vmin": vmin,
            "vmax": vmax,
            "s": 3,
        }
        default_args.update(kwargs)
        ax.scatter(tsne["x"], tsne["y"], s=3, color="lightgrey")
        ax.scatter(
            tsne.loc[cells, "x"],
            tsne.loc[cells, "y"],
            c=c,
            **default_args,
        )
        ax.set_axis_off()
        ax.set_title(g)

        if plot_scale:
            normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cax, _ = matplotlib.colorbar.make_axes(ax)
            kaw = dict()
            if "cmap" in kwargs.keys():
                kaw["cmap"] = kwargs["cmap"]
            matplotlib.colorbar.ColorbarBase(cax, norm=normalize, **kaw)

    return fig.figure, fig.axes


def plot_diffusion_components(
    data: Union[sc.AnnData, pd.DataFrame],
    dm_res: Optional[Union[pd.DataFrame, str]] = "DM_EigenVectors",
    embedding_basis: str = "X_umap",
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize diffusion components on tSNE or UMAP plots.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        An AnnData object from Scanpy or a DataFrame that contains tSNE or UMAP results.
    dm_res : Optional[Union[pd.DataFrame, str]], optional
        A DataFrame that contains the diffusion map results or a string key to access diffusion map
        results from the obsm of the AnnData object. Default is 'DM_Eigenvectors'.
    embedding_basis : str, optional
        Key to access UMAP results from the obsm of the AnnData object. Defaults to 'X_umap'.
    kwargs : dict, optional
        Additional keyword arguments to pass to the scatter plot function.

    Returns
    -------
    Tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]
        A tuple containing the matplotlib Figure and Axes objects representing the diffusion component plot.

    Raises
    ------
    KeyError
        If `embedding_basis` or `dm_res` is not found in .obsm when `data` is an AnnData object.
    """

    # Retrieve the embedding data
    if isinstance(data, sc.AnnData):
        if embedding_basis not in data.obsm:
            raise KeyError(f"'{embedding_basis}' not found in .obsm.")
        embedding_data = pd.DataFrame(data.obsm[embedding_basis], index=data.obs_names)
        if isinstance(dm_res, str):
            if dm_res not in data.obsm:
                raise KeyError(f"'{dm_res}' not found in .obsm.")
            dm_res = {
                "EigenVectors": pd.DataFrame(data.obsm[dm_res], index=data.obs_names)
            }
    else:
        embedding_data = data

    default_args = {
        "edgecolors": "none",
        "s": 3,
    }
    default_args.update(kwargs)

    fig = FigureGrid(dm_res["EigenVectors"].shape[1], 5)

    for i, ax in enumerate(fig):
        ax.scatter(
            embedding_data.iloc[:, 0],
            embedding_data.iloc[:, 1],
            c=dm_res["EigenVectors"].loc[embedding_data.index, i],
            **default_args,
        )
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_aspect("equal")
        ax.set_title(f"Component {i}", fontsize=10)
        ax.set_axis_off()

    return fig.figure, fig.axes


def plot_palantir_results(
    data: Union[sc.AnnData, pd.DataFrame],
    pr_res: Optional[PResults] = None,
    embedding_basis: str = "X_umap",
    pseudo_time_key: str = "palantir_pseudotime",
    entropy_key: str = "palantir_entropy",
    fate_prob_key: str = "palantir_fate_probabilities",
    **kwargs,
):
    """
    Plot Palantir results on t-SNE or UMAP plots.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of tSNE or UMAP results.
    pr_res : Optional[PResults]
        Optional PResults object containing Palantir results. If None, results are expected to be found in the provided AnnData object.
    embedding_basis : str, optional
        The key to retrieve UMAP results from the AnnData object. Defaults to 'X_umap'.
    pseudo_time_key : str, optional
        Key to access the pseudotime from obs of the AnnData object. Default is 'palantir_pseudotime'.
    entropy_key : str, optional
        Key to access the entropy from obs of the AnnData object. Default is 'palantir_entropy'.
    fate_prob_key : str, optional
        Key to access the fate probabilities from obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
    **kwargs
        Additional keyword arguments passed to `ax.scatter`.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the Palantir results.
    """
    if isinstance(data, sc.AnnData):
        if embedding_basis not in data.obsm:
            raise KeyError(f"'{embedding_basis}' not found in .obsm.")
        embedding_data = pd.DataFrame(data.obsm[embedding_basis], index=data.obs_names)
        if pr_res is None:
            if (
                pseudo_time_key not in data.obs
                or entropy_key not in data.obs
                or fate_prob_key not in data.obsm
            ):
                raise KeyError("Required Palantir results not found in .obs or .obsm.")
            obsm_pobs, _ = _validate_obsm_key(data, fate_prob_key)
            obsm_pobs = data.obsm[fate_prob_key]
            pr_res = PResults(
                data.obs[pseudo_time_key],
                data.obs[entropy_key],
                obsm_pobs,
                None,
            )
    else:
        embedding_data = data

    n_branches = pr_res.branch_probs.shape[1]
    n_cols = 6
    n_rows = int(np.ceil(n_branches / n_cols))
    plt.figure(figsize=[2 * n_cols, 2 * (n_rows + 2)])
    gs = plt.GridSpec(
        n_rows + 2, n_cols, height_ratios=np.append([0.75, 0.75], np.repeat(1, n_rows))
    )

    def scatter_with_colorbar(ax, x, y, c, **kwargs):
        sc = ax.scatter(x, y, c=c, **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax, orientation="vertical")

    # Pseudotime
    ax = plt.subplot(gs[0:2, 1:3])
    scatter_with_colorbar(
        ax,
        embedding_data.iloc[:, 0],
        embedding_data.iloc[:, 1],
        pr_res.pseudotime[embedding_data.index],
        **kwargs,
    )
    ax.set_axis_off()
    ax.set_title("Pseudotime")

    # Entropy
    ax = plt.subplot(gs[0:2, 3:5])
    scatter_with_colorbar(
        ax,
        embedding_data.iloc[:, 0],
        embedding_data.iloc[:, 1],
        pr_res.entropy[embedding_data.index],
        **kwargs,
    )
    ax.set_axis_off()
    ax.set_title("Entropy")

    # Branch probabilities
    for i, branch in enumerate(pr_res.branch_probs.columns):
        ax = plt.subplot(gs[i // n_cols + 2, i % n_cols])
        scatter_with_colorbar(
            ax,
            embedding_data.iloc[:, 0],
            embedding_data.iloc[:, 1],
            pr_res.branch_probs[branch][embedding_data.index],
            **kwargs,
        )
        ax.set_axis_off()
        ax.set_title(branch, fontsize=10)

    plt.tight_layout()
    return plt.gcf()


def plot_terminal_state_probs(
    data: Union[sc.AnnData, pd.DataFrame],
    cells: List[str],
    pr_res: Optional[PResults] = None,
    fate_prob_key: str = "palantir_fate_probabilities",
    **kwargs,
):
    """Function to plot barplot for probabilities for each cell in the list

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of fate probabilities.
    cells : List[str]
        List of cell for which the barplots need to be plotted.
    pr_res : Optional[PResults]
        Optional PResults object containing Palantir results. If None, results are expected to be found in the provided AnnData object.
    fate_prob_key : str, optional
        Key to access the fate probabilities from obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
    kwargs : dict, optional
        Additional keyword arguments to pass to the matplotlib.pyplot.bar function.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the cell fate probabilities.
    """
    if isinstance(data, sc.AnnData):
        if pr_res is None:
            branch_probs, _ = _validate_obsm_key(data, fate_prob_key)
    else:
        if pr_res is None:
            raise ValueError("pr_res must be provided when data is a DataFrame.")
        branch_probs = pr_res.branch_probs

    n_cols = 5
    n_rows = int(np.ceil(len(cells) / n_cols))
    if len(cells) < n_cols:
        n_cols = len(cells)
    fig = plt.figure(figsize=[3 * n_cols, 3 * n_rows])

    # Branch colors
    set1_colors = matplotlib.colormaps["Set1"](range(branch_probs.shape[1]))
    cluster_colors = [matplotlib.colors.rgb2hex(rgba) for rgba in set1_colors]
    branch_colors = pd.Series(
        cluster_colors,
        index=branch_probs.columns,
    )

    for i, cell in enumerate(cells):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # Probs
        x = branch_probs.columns.values
        height = branch_probs.loc[cell, :].values

        # Plot
        ax.bar(x, height, color=branch_colors, **kwargs)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([0, 1], fontsize=8)
        ax.set_title(cell, fontsize=10)

    return fig


def plot_branch_selection(
    ad: sc.AnnData,
    pseudo_time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "palantir_fate_probabilities",
    masks_key: str = "branch_masks",
    fates: Optional[Union[List[str], str]] = None,
    embedding_basis: str = "X_umap",
    figsize: Tuple[float, float] = (15, 5),
    **kwargs,
):
    """
    Plot cells along specific fates of pseudotime ordering and the UMAP embedding.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix. The pseudotime and fate probabilities should be stored under the keys provided.
    pseudo_time_key : str, optional
        Key to access the pseudotime from obs of the AnnData object. Default is 'palantir_pseudotime'.
    fate_prob_key : str, optional
        Key to access the fate probabilities from obsm of the AnnData object.
        Default is 'palantir_fate_probabilities'.
    masks_key : str, optional
        Key to access the branch cell selection masks from obsm of the AnnData object.
        Default is 'branch_masks'.
    fates : Union[List[str], str]
        The fates to be plotted. If not specified, all fates will be plotted.
    embedding_basis : str, optional
        Key to access the UMAP embedding from obsm of the AnnData object. Default is 'X_umap'.
    figsize : Tuple[float, float], optional
        Width and height of each subplot in inches. The total height of the figure is determined by
        multiplying the height by the number of fates. Default is (15, 5).
    **kwargs
        Additional arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the branch selections.

    """
    if pseudo_time_key not in ad.obs:
        raise KeyError(f"{pseudo_time_key} not found in ad.obs")

    fate_probs, fate_probs_names = _validate_obsm_key(ad, fate_prob_key)
    fate_mask, fate_mask_names = _validate_obsm_key(ad, masks_key)

    if embedding_basis not in ad.obsm:
        raise KeyError(f"{embedding_basis} not found in ad.obsm")

    fate_names = set(fate_probs_names).intersection(fate_mask_names)
    if len(fate_names) == 0:
        raise ValueError(
            f"No agreeing fate names found for .obsm['{fate_prob_key}'] and .obsm['{masks_key}']."
        )
    n_fates = len(fate_names)
    if n_fates < len(fate_probs_names) or n_fates < len(fate_probs_names):
        warnings.warn(
            f"Found only {n_fates} fates in the intersection of "
            f"{len(fate_probs_names)} fate-probability fates in .obsm['{fate_prob_key}'] fates) "
            f"and {len(fate_mask_names)} fate-mask fates in .obsm['{masks_key}']."
        )
    if isinstance(fates, str):
        fates = [fates]
    elif fates is None:
        fates = fate_names
    else:
        for b in fates:
            if b not in fate_names:
                raise (
                    f"Specified fate name '{b}' is not in the available set of "
                    + ", ".join(fate_names)
                )

    pt = ad.obs[pseudo_time_key]
    umap = ad.obsm[embedding_basis]

    figsize = figsize[0], figsize[1] * len(fates)
    fig, axes = plt.subplots(len(fates), 2, figsize=figsize, width_ratios=[2, 1])

    for i, fate in enumerate(fates):
        if axes.ndim == 1:
            ax1, ax2 = axes
        else:
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]
        mask = fate_mask[fate].astype(bool)

        # plot cells along pseudotime
        ax1.scatter(
            pt[~mask],
            fate_probs.loc[~mask, fate],
            c=config.DESELECTED_COLOR,
            label="Other Cells",
            **kwargs,
        )
        ax1.scatter(
            pt[mask],
            fate_probs.loc[mask, fate],
            c=config.SELECTED_COLOR,
            label="Selected Cells",
            **kwargs,
        )
        ax1.set_title(f"Branch: {fate}")
        ax1.set_xlabel("Pseudotime")
        ax1.set_ylabel(f"{fate}-Fate Probability")
        ax1.legend()

        # plot UMAP
        ax2.scatter(
            umap[~mask, 0],
            umap[~mask, 1],
            c=config.DESELECTED_COLOR,
            label="Other Cells",
            **kwargs,
        )
        ax2.scatter(
            umap[mask, 0],
            umap[mask, 1],
            c=config.SELECTED_COLOR,
            label="Selected Cells",
            **kwargs,
        )
        ax2.set_title(f"Branch: {fate}")
        ax2.axis("off")

    plt.tight_layout()

    return fig


def plot_gene_trends_legacy(gene_trends, genes=None):
    """Plot the gene trends: each gene is plotted in a different panel
    :param: gene_trends: Results of the compute_marker_trends function
    """

    # Branches and genes
    branches = list(gene_trends.keys())
    set2_colors = matplotlib.colormaps["Set2"](range(len(branches)))
    colors = pd.Series(
        [matplotlib.colors.rgb2hex(rgba) for rgba in set2_colors],
        index=branches,
    )
    if genes is None:
        genes = gene_trends[branches[0]]["trends"].index

    # Set up figure
    fig = plt.figure(figsize=[7, 3 * len(genes)])
    for i, gene in enumerate(genes):
        ax = fig.add_subplot(len(genes), 1, i + 1)
        for branch in branches:
            trends = gene_trends[branch]["trends"]
            def_stds = pd.DataFrame(0, index=trends.index, columns=trends.columns)
            stds = gene_trends[branch].get("std", def_stds)
            ax.plot(
                trends.columns.astype(float),
                trends.loc[gene, :],
                color=colors[branch],
                label=branch,
            )
            ax.set_xticks([0, 1])
            ax.fill_between(
                trends.columns.astype(float),
                trends.loc[gene, :] - stds.loc[gene, :],
                trends.loc[gene, :] + stds.loc[gene, :],
                alpha=0.1,
                color=colors[branch],
            )
            ax.set_title(gene)
        # Add legend
        if i == 0:
            ax.legend()

    return fig


def plot_gene_trends(
    data: Union[Dict, sc.AnnData],
    genes: Optional[List[str]] = None,
    gene_trend_key: str = "gene_trends",
    branch_names: Union[str, List] = "branch_masks",
) -> plt.Figure:
    """
    Plot the gene trends for each gene across different panels.

    Parameters
    ----------
    data : Union[Dict, sc.AnnData]
        An AnnData object or a dictionary that contains the gene trends.
    genes : Optional[List[str]], optional
        A list of genes to plot. If not provided, all genes will be plotted. Default is None.
    gene_trend_key : str, optional
        The key to access gene trends in the varm of the AnnData object. Default is 'gene_trends'.
    branch_names : Union[str, List[str]], optional
        Key to retrieve branch names from the AnnData object, or a list of branch names. If a string is provided,
        it will be treated as a key in either AnnData.uns or AnnData.obsm. For AnnData.obsm, the column names will
        be treated as branch names. If it cannot be found in AnnData.obsm and AnnData.uns then branch_names + "_columns"
        will be looked up in AnnData.uns.
        Default is 'branch_masks'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.

    Raises
    ------
    KeyError
        If 'branch_names' as a string is not found in either .uns or .obsm, or if 'gene_trend_key + "_" + branch_name'
        is not found in .varm.
    ValueError
        If 'data' is not an AnnData object and not a dictionary.
    """

    gene_trends = _validate_gene_trend_input(data, gene_trend_key, branch_names)

    # Branches and genes
    branches = list(gene_trends.keys())
    set2_colors = matplotlib.colormaps["Set2"](range(len(branches)))
    colors = pd.Series(
        [matplotlib.colors.rgb2hex(rgba) for rgba in set2_colors],
        index=branches,
    )

    if genes is None:
        genes = gene_trends[branches[0]]["trends"].index

    # Set up figure
    fig = plt.figure(figsize=[7, 3 * len(genes)])
    for i, gene in enumerate(genes):
        ax = fig.add_subplot(len(genes), 1, i + 1)
        for branch in branches:
            trends = gene_trends[branch]["trends"]
            ax.plot(
                trends.columns.astype(float),
                trends.loc[gene, :],
                color=colors[branch],
                label=branch,
            )
            ax.set_xticks([0, 1])
            ax.set_title(gene)

        # Add legend
        if i == 0:
            ax.legend()

    return fig


def _process_mask(ad: sc.AnnData, masks_key: str, branch_name: str):
    """
    Processes the mask string to obtain mask indices.

    Parameters
    ----------
    ad : sc.AnnData
        The annotated data matrix
    masks_key : str
        The mask string

    Returns
    -------
    np.ndarray
        Boolean array for masking
    """

    if masks_key in ad.obs:
        return ad.obs_vector(masks_key).astype(bool)

    fate_mask, fate_mask_names = _validate_obsm_key(ad, masks_key)

    try:
        mask = fate_mask[branch_name]
    except KeyError:
        raise ValueError(
            f"Fate '{branch_name}' not found in {branch_name} "
            f"in ad.osbm or {branch_name}_columns in ad.uns"
        )

    return mask.astype(bool).values


def prepare_color_vector(
    ad: sc.AnnData,
    color: str,
    mask: Optional[np.ndarray] = None,
    layer: Optional[str] = None,
    palette: Optional[Union[str, Sequence[str]]] = None,
    na_color: str = "lightgray",
):
    """
    Prepare the color vector for plotting.

    Parameters
    ----------
    ad : sc.AnnData
        The annotated data matrix
    color : str
        The color parameter
    mask : np.ndarray, optional
        Boolean mask array
    layer : str, optional
        The data layer to use
    palette : str or Sequence[str], optional
        The color palette to use
    na_color : str, optional
        The color for NA values

    Returns
    -------
    Tuple
        Color vector and a flag indicating whether it's categorical
    """

    color_source_vector = _get_color_source_vector(ad, color, layer=layer)
    color_vector, categorical = _color_vector(
        ad, color, color_source_vector, palette=palette, na_color=na_color
    )
    if mask is not None:
        color_vector = color_vector[mask]

    return color_source_vector, color_vector, categorical


def _add_categorical_legend(
    ax,
    color_source_vector,
    palette: dict,
    legend_anchor: Tuple[float, float],
    legend_fontweight,
    legend_fontsize,
    legend_fontoutline,
    na_color,
    na_in_legend: bool,
):
    """Add a legend to the passed Axes."""
    if na_in_legend and pd.isnull(color_source_vector).any():
        if "NA" in color_source_vector:
            raise NotImplementedError(
                "No fallback for null labels has been defined if NA already in categories."
            )
        color_source_vector = color_source_vector.add_categories("NA").fillna("NA")
        palette = palette.copy()
        palette["NA"] = na_color
    if color_source_vector.dtype == bool:
        cats = pd.Categorical(color_source_vector.astype(str)).categories
    else:
        cats = color_source_vector.categories

    for label in cats:
        ax.scatter([], [], c=palette[label], label=label)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=legend_anchor,
        ncol=(1 if len(cats) <= 14 else 2 if len(cats) <= 30 else 3),
        fontsize=legend_fontsize,
    )


def plot_stats(
    ad: sc.AnnData,
    x: str,
    y: str,
    color: str = None,
    ax: Optional[plt.Axes] = None,
    na_color: str = "lightgray",
    color_layer: Optional[str] = None,
    x_layer: Optional[str] = None,
    y_layer: Optional[str] = None,
    branch_name: Optional[str] = None,
    masks_key: str = "branch_masks",
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = "bold",
    legend_fontoutline: Optional[int] = None,
    legend_anchor: Tuple[float, float] = (1.1, 0.5),
    color_bar_bounds: list = [1.1, 0.3, 0.02, 0.4],
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], None] = None,
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    nticks: int = 3,
    figsize: Tuple[float, float] = (6, 6),
    **kwargs,
):
    """
    This function visualizes a scatter plot of cells over pseudotime.
    The y-position indicates either a gene expression
    or any column from .obs or use a different position_layer, like
    "MAGIC_imputed_data". The color follows similar rules and behaves like the color
    parameter in scanpy.pl.embedding, but only accepts a single value instead of a list.

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix of shape n_obs x n_vars. Rows correspond
        to cells and columns to genes.
    x : str
        Similar to color but used for x-position.
    y : str
        Similar to color but used for y-position.
    color : str, optional
        Defines the color to be used for the plot, similar to the color in
        scanpy.pl.embedding. If not provided, the default is None.
    ax : Axes, optional
        A matplotlib axes object.
    na_color : str, optional
        The color to be used for 'NA' values. Default is "lightgray".
    color_layer : str, optional
        Specifies the data layer to use for color in the plot.
        If not provided, the .X layer is used.
    x_layer : str, optional
        Specifies the data layer to use for x-position in the plot.
        If not provided, the .X layer is used.
    y_layer : str, optional
        Specifies the data layer to use for y-position in the plot.
        If not provided, the .X layer is used.
    branch_name : str, optional
        Specifies the branch to subset the plot to. The default is None.
    masks_key : str, optional
        Key for accessing the branch cell selection masks from obsm of the AnnData object.
        Default is 'branch_masks'.
    legend_fontsize : Union[int, float, _FontSize, None], optional
        Specifies the font size for the legend. Default is None.
    legend_fontweight : Union[int, _FontWeight], optional
        Specifies the font weight for the legend. Default is 'bold'.
    legend_fontoutline : int, optional
        Specifies the font outline for the legend. Default is None.
    legend_anchor: Tuple[float, float] = (1.1, 0.5),
        Defines the position of the legend. The argument will be passed to the
        bbox_to_anchor parameter of ax.legend() method. The default is (1.1, 0.5).
    color_bar_bounds : list, optional
        Specifies the bounds for the color bar. Defaults to [1, 0.4, 0.01, 0.2].
    cmap : Union[Colormap, str, None]
        A colormap instance or registered colormap name to color the scatter plot.
    palette : Union[str, Sequence[str], Cycler, None], optional
        Colors to use for plotting categorical annotation groups.
        The palette can be a valid matplotlib.colors.ListedColormap name
        ('viridis', 'Set2', etc), a cycler.Cycler object, or a sequence of
        matplotlib colors like ['red', 'blue', 'green'].
    vmax : float or array-like or None
        Defines the lower limit of the color scale with values smaller than
        vmin sharing the same color. It can be a number, percentile string ('pN'),
        function returning a desired value from the plot values, or None for
        automatic selection. For multiple plots, a list of vmin can be specified.
    vmin : float or array-like or None
        Sets the upper limit of the color scale, with the same format and behavior as vmin.
    vcenter : float or array-like or None
        Sets the center of the color scale, useful for diverging colormaps.
        It follows the same format and rules as vmin and vmax. Example:
        sc.pl.umap(adata, color='TREM2', vcenter='p50', cmap='RdBu_r').
    norm : matplotlib.colors.Normalize or None
        The normalizing object which scales data, typically into the interval [0, 1].
        If provided, vmax, vmin, and vcenter are ignored.
    nticks: int, optional
        The number of ticks to be displayed on both the x and y axes. The matplotlib's
        Axes.locator_params method is used for setting this property. Default is 3.
    figsize : Tuple[float, float], optional
        Width and height of each subplot in inches. Default is (12, 4).

    Returns
    -------
    fig, ax : figure and axis elements of the plot.

    Raises
    ------
    TypeError
        If input parameters are not of the expected type.
    ValueError
        If input parameters do not have the expected values.
    """

    if not isinstance(ad, sc.AnnData):
        raise TypeError("Expected ad to be an instance of sc.AnnData")
    if branch_name is not None and not isinstance(branch_name, str):
        raise TypeError("Expected branch_name to be a str or None")
    if color is not None and not isinstance(color, str):
        raise TypeError("Expected color to be a str")
    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError("Expected ax to be a matplotlib Axes instance")

    if branch_name is not None:
        mask = (
            _process_mask(ad, masks_key, branch_name)
            if isinstance(masks_key, str)
            else masks_key
        )
    else:
        mask = None

    x_pos = _get_color_source_vector(ad, x, layer=x_layer)
    y_pos = _get_color_source_vector(ad, y, layer=y_layer)
    if mask is not None:
        x_pos = x_pos[mask]
        y_pos = y_pos[mask]

    color_source_vector, color_vector, categorical = prepare_color_vector(
        ad, color, mask, layer=color_layer, palette=palette, na_color=na_color
    )

    scatter_kwargs = {
        "edgecolor": "none",
        "plotnonfinite": True,
    }
    scatter_kwargs.update(kwargs)

    default_cmap = matplotlib.colormaps["viridis"]
    cmap = copy(matplotlib.colormaps.get(cmap, default_cmap))
    cmap.set_bad(na_color)

    na_color = matplotlib.colors.to_hex(na_color, keep_alpha=True)

    if isinstance(vmax, str) or not isinstance(vmax, cabc.Sequence):
        vmax = [vmax]
    if isinstance(vmin, str) or not isinstance(vmin, cabc.Sequence):
        vmin = [vmin]
    if isinstance(vcenter, str) or not isinstance(vcenter, cabc.Sequence):
        vcenter = [vcenter]
    if isinstance(norm, Normalize) or not isinstance(norm, cabc.Sequence):
        norm = [norm]

    if not categorical and color is not None:
        vmin_float, vmax_float, vcenter_float, norm_obj = _get_vboundnorm(
            vmin, vmax, vcenter, norm, 0, color_vector
        )
        normalize = check_colornorm(
            vmin_float,
            vmax_float,
            vcenter_float,
            norm_obj,
        )
        scatter_kwargs["norm"] = normalize
        scatter_kwargs["cmap"] = cmap

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    points = ax.scatter(
        x_pos,
        y_pos,
        marker=".",
        c=color_vector,
        **scatter_kwargs,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if branch_name is not None:
        ax.set_title(branch_name)
    ax.set_zorder(0)
    ax.set_facecolor("none")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        ax.locator_params(axis="both", nbins=nticks)

    if categorical or color_vector.dtype == bool:
        _add_categorical_legend(
            ax,
            color_source_vector,
            palette=_get_palette(ad, color),
            legend_anchor=legend_anchor,
            legend_fontweight=legend_fontweight,
            legend_fontsize=legend_fontsize,
            legend_fontoutline=None,
            na_color=na_color,
            na_in_legend=True,
        )
    elif color_bar_bounds is not None and color is not None:
        cax = ax.inset_axes(color_bar_bounds)
        cb = plt.colorbar(points, cax=cax)
        cb.set_label(color)

    return fig, ax


def plot_branch(
    ad: sc.AnnData,
    branch_name: str,
    position: str,
    color: str = None,
    masks_key: str = "branch_masks",
    ax: Optional[plt.Axes] = None,
    pseudo_time_key: str = "palantir_pseudotime",
    na_color: str = "lightgray",
    color_layer: Optional[str] = None,
    position_layer: Optional[str] = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = "bold",
    legend_fontoutline: Optional[int] = None,
    legend_anchor: Tuple[float, float] = (1.1, 0.5),
    color_bar_bounds: list = [1.1, 0.3, 0.02, 0.4],
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], None] = None,
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    nticks: int = 3,
    figsize: Tuple[float, float] = (12, 4),
    **kwargs,
):
    """
    This function visualizes a scatter plot of cells over pseudotime.
    The y-position indicates either a gene expression
    or any column from .obs or use a different position_layer, like
    "MAGIC_imputed_data". The color follows similar rules and behaves like the color
    parameter in scanpy.pl.embedding, but only accepts a single value instead of a list.

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix of shape n_obs x n_vars. Rows correspond
        to cells and columns to genes.
    branch_name : str
        Specifies the branch to plot the trend for.
    position : str
        Similar to color but used for y-position.
    color : str, optional
        Defines the color to be used for the plot, similar to the color in
        scanpy.pl.embedding. If not provided, the default is None.
    masks_key : str, optional
        Key for accessing the branch cell selection masks from obsm of the AnnData object.
        Default is 'branch_masks'.
    ax : Axes, optional
        A matplotlib axes object.
    pseudo_time_key : str, optional
        Specifies the pseudotime key to be used for the plot.
        The default is "palantir_pseudotime".
    na_color : str, optional
        The color to be used for 'NA' values. Default is "lightgray".
    color_layer : str, optional
        Specifies the data layer to use for color in the plot.
        If not provided, the .X layer is used.
    position_layer : str, optional
        Specifies the data layer to use for y-position in the plot.
        If not provided, the .X layer is used.
    legend_fontsize : Union[int, float, _FontSize, None], optional
        Specifies the font size for the legend. Default is None.
    legend_fontweight : Union[int, _FontWeight], optional
        Specifies the font weight for the legend. Default is 'bold'.
    legend_fontoutline : int, optional
        Specifies the font outline for the legend. Default is None.
    legend_anchor: Tuple[float, float] = (1.1, 0.5),
        Defines the position of the legend. The argument will be passed to the
        bbox_to_anchor parameter of ax.legend() method. The default is (1.1, 0.5).
    color_bar_bounds : list, optional
        Specifies the bounds for the color bar. Defaults to [1, 0.4, 0.01, 0.2].
    cmap : Union[Colormap, str, None]
        A colormap instance or registered colormap name to color the scatter plot.
    palette : Union[str, Sequence[str], Cycler, None], optional
        Colors to use for plotting categorical annotation groups.
        The palette can be a valid matplotlib.colors.ListedColormap name
        ('viridis', 'Set2', etc), a cycler.Cycler object, or a sequence of
        matplotlib colors like ['red', 'blue', 'green'].
    vmax : float or array-like or None
        Defines the lower limit of the color scale with values smaller than
        vmin sharing the same color. It can be a number, percentile string ('pN'),
        function returning a desired value from the plot values, or None for
        automatic selection. For multiple plots, a list of vmin can be specified.
    vmin : float or array-like or None
        Sets the upper limit of the color scale, with the same format and behavior as vmin.
    vcenter : float or array-like or None
        Sets the center of the color scale, useful for diverging colormaps.
        It follows the same format and rules as vmin and vmax. Example:
        sc.pl.umap(adata, color='TREM2', vcenter='p50', cmap='RdBu_r').
    norm : matplotlib.colors.Normalize or None
        The normalizing object which scales data, typically into the interval [0, 1].
        If provided, vmax, vmin, and vcenter are ignored.
    nticks: int, optional
        The number of ticks to be displayed on both the x and y axes. The matplotlib's
        Axes.locator_params method is used for setting this property. Default is 3.
    figsize : Tuple[float, float], optional
        Width and height of each subplot in inches. Default is (12, 4).

    Returns
    -------
    fig, ax : figure and axis elements of the plot.

    Raises
    ------
    TypeError
        If input parameters are not of the expected type.
    ValueError
        If input parameters do not have the expected values.
    """
    if not isinstance(branch_name, str):
        raise TypeError("Expected branch_name to be a str")
    if not isinstance(pseudo_time_key, str):
        raise TypeError("Expected pseudo_time_key to be a str")

    fig, ax = plot_stats(
        ad=ad,
        x=pseudo_time_key,
        y=position,
        color=color,
        ax=ax,
        na_color=na_color,
        color_layer=color_layer,
        x_layer=None,
        y_layer=position_layer,
        branch_name=branch_name,
        masks_key=masks_key,
        legend_fontsize=legend_fontsize,
        legend_fontweight=legend_fontweight,
        legend_fontoutline=legend_fontoutline,
        legend_anchor=legend_anchor,
        color_bar_bounds=color_bar_bounds,
        cmap=cmap,
        palette=palette,
        vmax=vmax,
        vmin=vmin,
        vcenter=vcenter,
        norm=norm,
        nticks=nticks,
        figsize=figsize,
        **kwargs,
    )
    ax.set_xlabel("Pseudotime")
    return fig, ax


def plot_trend(
    ad: sc.AnnData,
    branch_name: str,
    gene: str,
    color: str = None,
    masks_key: str = "branch_masks",
    gene_trend_key: str = "gene_trends",
    ax: Optional[plt.Axes] = None,
    pseudo_time_key: str = "palantir_pseudotime",
    na_color: str = "lightgray",
    position: str = None,
    color_layer: Optional[str] = None,
    position_layer: Optional[str] = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = "bold",
    legend_fontoutline: Optional[int] = None,
    legend_anchor: Tuple[float, float] = (1.1, 0.5),
    color_bar_bounds: list = [1.1, 0.3, 0.02, 0.4],
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], None] = None,
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    nticks: int = 3,
    figsize: Tuple[float, float] = (12, 4),
    **kwargs,
):
    """
    This function visualizes a trend graph for a specific gene's expression over pseudotime.
    It plots the trend of a single gene for a chosen branch and additionally overlays a
    scatter plot of cells from the same branch. The y-position indicates the gene expression
    and can be configured to take any column from .obs or use a different layer, like
    "MAGIC_imputed_data". The color follows similar rules and behaves like the color
    parameter in scanpy.pl.embedding, but only accepts a single value instead of a list.

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix of shape n_obs x n_vars. Rows correspond
        to cells and columns to genes.
    gene : str
        Specifies the gene to be plotted.
    branch_name : str
        Specifies the branch to plot the trend for.
    color : str, optional
        Defines the color to be used for the plot, similar to the color in
        scanpy.pl.embedding. If not provided, the default is None.
    masks_key : str, optional
        Key for accessing the branch cell selection masks from obsm of the AnnData object.
        Set to None to disable the scatter plot.
        Default is 'branch_masks'.
    gene_trend_key : str, optional
        Key for accessing gene trends in the AnnData object's varm. Default is 'gene_trends'.
    ax : Axes, optional
        A matplotlib axes object.
    pseudo_time_key : str, optional
        Specifies the pseudotime key to be used for the plot.
        The default is "palantir_pseudotime".
    na_color : str, optional
        The color to be used for 'NA' values. Default is "lightgray".
    position : str, optional
        Similar to color but used for y-position. If None, the default is gene.
    color_layer : str, optional
        Specifies the data layer to use for color in the plot.
        If not provided, the .X layer is used.
    position_layer : str, optional
        Specifies the data layer to use for y-position in the plot.
        If not provided, the .X layer is used.
    legend_fontsize : Union[int, float, _FontSize, None], optional
        Specifies the font size for the legend. Default is None.
    legend_fontweight : Union[int, _FontWeight], optional
        Specifies the font weight for the legend. Default is 'bold'.
    legend_fontoutline : int, optional
        Specifies the font outline for the legend. Default is None.
    legend_anchor: Tuple[float, float] = (1.1, 0.5),
        Defines the position of the legend. The argument will be passed to the
        bbox_to_anchor parameter of ax.legend() method. The default is (1.1, 0.5).
    color_bar_bounds : list, optional
        Specifies the bounds for the color bar. Defaults to [1, 0.4, 0.01, 0.2].
    cmap : Union[Colormap, str, None]
        A colormap instance or registered colormap name to color the scatter plot.
    palette : Union[str, Sequence[str], Cycler, None], optional
        Colors to use for plotting categorical annotation groups.
        The palette can be a valid matplotlib.colors.ListedColormap name
        ('viridis', 'Set2', etc), a cycler.Cycler object, or a sequence of
        matplotlib colors like ['red', 'blue', 'green'].
    vmax : float or array-like or None
        Defines the lower limit of the color scale with values smaller than
        vmin sharing the same color. It can be a number, percentile string ('pN'),
        function returning a desired value from the plot values, or None for
        automatic selection. For multiple plots, a list of vmin can be specified.
    vmin : float or array-like or None
        Sets the upper limit of the color scale, with the same format and behavior as vmin.
    vcenter : float or array-like or None
        Sets the center of the color scale, useful for diverging colormaps.
        It follows the same format and rules as vmin and vmax. Example:
        sc.pl.umap(adata, color='TREM2', vcenter='p50', cmap='RdBu_r').
    norm : matplotlib.colors.Normalize or None
        The normalizing object which scales data, typically into the interval [0, 1].
        If provided, vmax, vmin, and vcenter are ignored.
    nticks: int, optional
        The number of ticks to be displayed on both the x and y axes. The matplotlib's
        Axes.locator_params method is used for setting this property. Default is 3.
    figsize : Tuple[float, float], optional
        Width and height of each subplot in inches. Default is (12, 4).

    Returns
    -------
    fig, ax : figure and axis elements of the plot.

    Raises
    ------
    TypeError
        If input parameters are not of the expected type.
    ValueError
        If input parameters do not have the expected values.
    """

    if not isinstance(ad, sc.AnnData):
        raise TypeError("Expected ad to be an instance of sc.AnnData")
    if not isinstance(gene, str):
        raise TypeError("Expected gene to be a str")
    if not isinstance(branch_name, str):
        raise TypeError("Expected branch_name to be a str")
    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError("Expected ax to be a matplotlib Axes instance")

    gene_trends = _validate_gene_trend_input(ad, gene_trend_key, [branch_name])
    if position is None:
        position = gene

    trends = gene_trends[branch_name]["trends"]
    pseudotime_grid = trends.columns.astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.plot(
        pseudotime_grid,
        trends.loc[gene, :],
        color=config.SELECTED_COLOR,
    )
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel(f"{gene} trend")
    ax.locator_params(axis="both", nbins=nticks)
    ax.set_zorder(1)
    ax.set_facecolor("none")

    if masks_key is not None:
        ax2 = ax.twinx()
        plot_branch(
            ad,
            branch_name=branch_name,
            position=position,
            color=color,
            masks_key=masks_key,
            ax=ax2,
            pseudo_time_key=pseudo_time_key,
            na_color=na_color,
            color_layer=color_layer,
            position_layer=position_layer,
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            legend_fontoutline=legend_fontoutline,
            legend_anchor=legend_anchor,
            color_bar_bounds=color_bar_bounds,
            cmap=cmap,
            palette=palette,
            vmax=vmax,
            vmin=vmin,
            vcenter=vcenter,
            norm=norm,
            nticks=nticks,
            **kwargs,
        )
    else:
        ax.set_title(branch_name)

    return fig, ax


def _scale(
    mat: pd.DataFrame,
    scaling: Optional[Literal["none", "z-score", "quantile", "percent"]] = None,
) -> pd.DataFrame:
    """
    Scale the given matrix based on the scaling method provided.

    Parameters
    ----------
    mat : pd.DataFrame
        A pandas DataFrame to be scaled.
    scaling : Optional[Literal["none", "z-score", "quantile", "percent"]], optional
        The scaling method. Options are:
        - "none" : returns the original matrix.
        - "z-score" : standardizes the matrix to have 0 mean and 1 variance.
        - "quantile" : scales the matrix to have values between 0 and 1.
        - "percent" : scales the matrix to represent percentages of the max value in the row.
        If None or "none", the original matrix will be returned. Default is None.

    Returns
    -------
    pd.DataFrame
        The scaled pandas DataFrame.
    """
    if scaling in [None, "none", "None"]:
        return mat
    elif scaling == "z-score":
        return pd.DataFrame(
            StandardScaler().fit_transform(mat.T).T,
            index=mat.index,
            columns=mat.columns,
        )
    elif scaling == "quantile":
        return mat.rank(axis=1) / mat.shape[1]
    elif scaling == "percent":
        return mat.div(mat.max(axis=1), axis=0)
    else:
        raise ValueError("Invalid scaling method.")


def plot_gene_trend_heatmaps(
    data: Union[sc.AnnData, Dict],
    genes: Optional[List[str]] = None,
    gene_trend_key: str = "gene_trends",
    branch_names: Union[str, List[str]] = "branch_masks",
    scaling: Optional[Literal["none", "z-score", "quantile", "percent"]] = "z-score",
    basefigsize: Tuple[int, int] = (7, 0.7),
    cbkwargs: Dict = dict(),
    **kwargs,
) -> plt.Figure:
    """
    Plot the gene trends on heatmaps: a heatmap is generated for each branch.

    Parameters
    ----------
    data : Union[sc.AnnData, Dict]
        AnnData object or dictionary of gene trends.
    genes : Optional[List[str]], optional
        List of genes to include in the plot. If None, all genes are included.
        Default is None.
    gene_trend_key : str, optional
        Key to access gene trends in the AnnData object's varm. Default is 'gene_trends'.
    branch_names : Union[str, List[str]], optional
        Key to access branch names from AnnData object or list of branch names. If a string is provided,
        it is assumed to be a key in AnnData.uns. Default is 'branch_masks_columns'.
    scaling : Optional[Literal["none", "z-score", "quantile", "percent"]], optional
        Scaling method to apply on the gene trends. Options are:
        - "none" : no scaling is applied.
        - "z-score" : standardizes the data to have 0 mean and 1 variance.
        - "quantile" : scales the data to have values between 0 and 1.
        - "percent" : scales the data to represent percentages of the max value in the row.
        Default is 'z-score'.
    basefigsize : Tuple[int, int], optional
        Base width and height in inches of the figure. The actual height of the figure is calculated
        based on the number of genes and branches. Default base size is (7, 0.7).
    cbkwargs : dict
        Additional keyword arguments for matplotlib.pyplot.colorbar.
    **kwargs : dict
        Additional keyword arguments for matplotlib.pyplot.matshow.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object of the plot.
    """
    gene_trends = _validate_gene_trend_input(data, gene_trend_key, branch_names)

    default_kwargs = {
        "cmap": matplotlib.rcParams["image.cmap"],
        "aspect": 50,
    }
    default_kwargs.update(kwargs)
    default_cbkwargs = {
        "shrink": 0.9,
        "drawedges": False,
    }
    default_cbkwargs.update(cbkwargs)

    # Get the branch names
    branches = list(gene_trends.keys())
    if genes is None:
        genes = gene_trends[branches[0]]["trends"].index

    height = basefigsize[1] * len(genes) * len(branches)
    figsize = (basefigsize[0], height)

    fig = plt.figure(figsize=figsize)
    for i, branch in enumerate(branches):
        ax = fig.add_subplot(len(branches), 1, i + 1)

        mat = gene_trends[branch]["trends"].loc[genes, :]
        mat = _scale(mat, scaling)
        cbd = ax.matshow(mat, **default_kwargs)
        ax.set_xticks([])
        ax.set_yticks(range(len(genes)), genes)
        ax.set_frame_on(False)
        ax.set_title(branch, fontsize=12)
        cb = plt.colorbar(cbd, ax=ax, **default_cbkwargs)
        cb.outline.set_visible(False)

    return fig


def plot_gene_trend_clusters(
    data: Union[sc.AnnData, pd.DataFrame],
    branch_name: str = "",
    clusters: Optional[Union[pd.Series, str]] = None,
    gene_trend_key: Optional[str] = "gene_trends",
) -> plt.Figure:
    """
    Visualize gene trend clusters.

    This function accepts either a Scanpy AnnData object or a DataFrame of gene
    expression trends, along with a Series of clusters or a key to clusters in
    AnnData object's var. It creates a plot representing gene trend clusters.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        AnnData object or DataFrame of gene expression trends.
    branch_name : str, optional
        Name of the branch for which to plot gene trends.
        It is added to the gene_trend_key when accessing gene trends from varm.
        Defaults to "".
    clusters : Union[pd.Series, str], optional
        Series of clusters indexed by gene names or string key to access clusters
        from the AnnData object's var. If data is a DataFrame, clusters should be
        a Series. Default key is 'gene_trend_key + "_" + branch_name + "_clusters"'.
    gene_trend_key : str, optional
        Key to access gene trends in the AnnData object's varm. Default is 'gene_trends'.

    Returns
    -------
    matplotlib.pyplot.Figure
        Matplotlib Figure object representing the plot of the gene trend clusters.

    Raises
    ------
    KeyError
        If `gene_trend_key` is None when `data` is an AnnData object.
    """
    # Process inputs and standardize trends
    if isinstance(data, sc.AnnData):
        if gene_trend_key is None:
            raise KeyError(
                "Must provide a gene_trend_key when data is an AnnData object."
            )

        trends, pseudotimes = _validate_varm_key(
            data, gene_trend_key + "_" + branch_name
        )

        if clusters is None:
            clusters = gene_trend_key + "_clusters"
        if isinstance(clusters, str):
            clusters = data.var[clusters]
    else:
        trends = data

    trends = pd.DataFrame(
        StandardScaler().fit_transform(trends.T).T,
        index=trends.index,
        columns=trends.columns.astype(float),
    )

    # Obtain unique clusters and prepare figure
    cluster_labels = (
        clusters.cat.categories
        if pd.api.types.is_categorical_dtype(clusters)
        else set(clusters)
    )
    n_rows = int(np.ceil(len(cluster_labels) / 3))
    fig = plt.figure(figsize=[5.5 * 3, 2.5 * n_rows])

    # Plot each cluster
    for i, c in enumerate(cluster_labels):
        ax = fig.add_subplot(n_rows, 3, i + 1)
        cluster_trends = trends.loc[clusters.index[clusters == c], :]
        means = cluster_trends.mean()
        std = cluster_trends.std()

        ax.plot(
            cluster_trends.columns,
            cluster_trends.T,
            linewidth=0.5,
            color=config.DESELECTED_COLOR,
        )
        ax.plot(means.index, means, color=config.SELECTED_COLOR)
        ax.plot(
            means.index,
            means - std,
            linestyle="--",
            color=config.SELECTED_COLOR,
            linewidth=0.75,
        )
        ax.plot(
            means.index,
            means + std,
            linestyle="--",
            color=config.SELECTED_COLOR,
            linewidth=0.75,
        )

        ax.set_title(f"Cluster {c}", fontsize=12)
        ax.tick_params("both", length=2, width=1, which="major")
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
        ax.set_xticklabels([])

    return fig


def gene_score_histogram(
    ad: sc.AnnData,
    score_key: str,
    genes: Optional[List[str]] = None,
    bins: int = 100,
    quantile: Optional[float] = 0.95,
    extra_offset_fraction: float = 0.1,
    anno_min_diff_fraction: float = 0.05,
) -> plt.Figure:
    """
    Draw a histogram of gene scores with percentile line and annotations for specific genes.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix.
    score_key : str
        The key in `ad.var` data frame for the gene score.
    genes : Optional[List[str]], default=None
        List of genes to be annotated. If None, no genes are annotated.
    bins : int, default=100
        The number of bins for the histogram.
    quantile : Optional[float], default=0.95
        Quantile line to draw on the histogram. If None, no line is drawn.
    extra_offset_fraction : float, default=0.1
        Fraction of max height to use as extra offset for annotation.
    anno_min_diff_fraction : float, default=0.05
        Fraction of the range of the scores to be used as minimum difference for annotation.

    Returns
    -------
    fig : matplotlib Figure
        Figure object with the histogram.

    Raises
    ------
    ValueError
        If input parameters are not as expected.
    """
    if not isinstance(ad, sc.AnnData):
        raise ValueError("Input data should be of type sc.AnnData.")
    if score_key not in ad.var.columns:
        raise ValueError(f"Score key {score_key} not found in ad.var columns.")
    scores = ad.var[score_key]

    if genes is not None:
        if not all(gene in scores for gene in genes):
            raise ValueError("All genes must be present in the scores.")

    fig, ax = plt.subplots(figsize=(10, 6))
    n_markers = len(genes) if genes is not None else 0

    heights, bins, _ = ax.hist(scores, bins=bins, zorder=-n_markers - 2)

    if quantile is not None:
        if quantile < 0 or quantile > 1:
            raise ValueError("Quantile should be a float between 0 and 1.")
        ax.vlines(
            np.quantile(scores, quantile),
            0,
            np.max(heights),
            alpha=0.5,
            color="red",
            label=f"{quantile:.0%} percentile",
        )

    ax.legend()
    ax.set_xlabel(f"{score_key} score")
    ax.set_ylabel("# of genes")

    ax.spines[["right", "top"]].set_visible(False)
    plt.locator_params(axis="x", nbins=3)

    if genes is None:
        return fig

    previous_value = -np.inf
    extra_offset = extra_offset_fraction * np.max(heights)
    min_diff = anno_min_diff_fraction * (np.max(bins) - np.min(bins))
    marks = scores[genes].sort_values()
    ranks = scores.rank(ascending=False)
    for k, (highlight_gene, value) in enumerate(marks.items()):
        hl_rank = int(ranks[highlight_gene])
        i = np.searchsorted(bins, value)
        text_offset = -np.inf if value - previous_value > min_diff else previous_value
        previous_value = value
        height = heights[i - 1]
        text_offset = max(text_offset + extra_offset, height + 1.8 * extra_offset)
        txt = ax.annotate(
            f"{highlight_gene} #{hl_rank}",
            (value, height),
            (value, text_offset),
            arrowprops=dict(facecolor="black", width=1, alpha=0.5),
            rotation=90,
            horizontalalignment="center",
            zorder=-k,
        )
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=2, foreground="w", alpha=0.8)]
        )

    return fig


def plot_trajectory(
    ad: sc.AnnData,
    branch: str,
    ax: Optional[plt.Axes] = None,
    pseudo_time_key: str = "palantir_pseudotime",
    masks_key: str = "branch_masks",
    embedding_basis: str = "X_umap",
    cell_color: str = "branch_selection",
    smoothness: float = 1.0,
    n_arrows: int = 5,
    arrowprops: Optional[dict] = dict(),
    scanpy_kwargs: Optional[dict] = dict(),
    figsize: Tuple[float, float] = (5, 5),
    **kwargs,
):
    """
    Plot a trajectory on the UMAP embedding of single-cell data.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix. Pseudotime and fate probabilities should be stored under provided keys.
    branch : str
        Branch/fate to plot the trajectory for.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, a new figure is created.
    pseudo_time_key : str, optional
        Key for pseudotime data in `ad.obs`. Defaults to 'palantir_pseudotime'.
    masks_key : str, optional
        Key for branch cell selection masks in `ad.obsm`. Defaults to 'branch_masks'.
    embedding_basis : str, optional
        Key for UMAP embedding in `ad.obsm`. Defaults to 'X_umap'.
    cell_color : str or None, optional
        Coloring strategy for UMAP plot. 'branch_selection' highlights cells in the branch.
        If None, no coloring is applied. Defaults to 'branch_selection'.
    smoothness : float, optional
        Smoothness of fitted trajectory. Higher value means smoother. Defaults to 1.
    n_arrows : int, optional
        Number of arrows to plot. Defaults to 5.
    arrowprops : dict, optional
        Properties for the arrowstyle. If None, defaults to black arrow with lw=1.
    scanpy_kwargs : dict, optional
        Keyword arguments for the scanpy.pl.emebdding function to plot the cells.
    figsize : Tuple[float, float], optional
        Size of the plot in inches, as (width, height). Defaults to (5, 5).
    **kwargs
        Extra keyword arguments are passed to the plot function for trajectory lines.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the trajectory plot.
    """
    if pseudo_time_key not in ad.obs:
        raise KeyError(f"{pseudo_time_key} not found in ad.obs")

    fate_mask, fate_mask_names = _validate_obsm_key(ad, masks_key)

    if embedding_basis not in ad.obsm:
        raise KeyError(f"{embedding_basis} not found in ad.obsm")

    if branch not in fate_mask_names:
        raise (
            f"Specified branch name '{branch}' is not in the available set of "
            + ", ".join(fate_mask_names)
        )

    default_kwargs = {"color": "black"}
    default_kwargs.update(kwargs)

    pt = ad.obs[pseudo_time_key]
    umap = ad.obsm[embedding_basis]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    mask = fate_mask[branch].astype(bool)

    pseudotime = pt[mask]
    pseudotime_grid = np.linspace(np.min(pseudotime), np.max(pseudotime), 200)
    ls = (
        smoothness
        * np.sqrt(np.sum((np.max(umap, axis=0) - np.min(umap, axis=0)) ** 2))
        / 20
    )
    import mellon
    umap_est = mellon.FunctionEstimator(ls=ls, sigma=ls, n_landmarks=50)
    umap_trajectory = umap_est.fit_predict(pseudotime, umap[mask, :], pseudotime_grid)

    # plot UMAP
    if cell_color == "branch_selection":
        ax.scatter(
            umap[~mask, 0],
            umap[~mask, 1],
            c=config.DESELECTED_COLOR,
            label="Other Cells",
        )
        ax.scatter(
            umap[mask, 0],
            umap[mask, 1],
            c=config.SELECTED_COLOR,
            label="Selected Cells",
        )
    elif cell_color is not None:
        b = embedding_basis[2:] if embedding_basis.startswith("X_") else embedding_basis
        sc.pl.embedding(ad, b, color=cell_color, ax=ax, show=False, **scanpy_kwargs)

    # plot trajectory
    _plot_arrows(
        umap_trajectory[:, 0],
        umap_trajectory[:, 1],
        n=n_arrows,
        ax=ax,
        arrowprops=arrowprops,
        **default_kwargs,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Branch: {branch}")
    ax.axis("off")

    return ax


def _plot_arrows(x, y, n=5, ax=None, arrowprops=dict(), **kwargs):
    """
    Helper function to plot arrows on a trajectory line.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the trajectory points.
    n : int, optional
        Number of arrows to plot. Defaults to 5.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, a new figure is created.
    arrowprops : dict, optional
        Properties for the arrowstyle. If None, defaults to black arrow with lw=1.
    **kwargs
        Extra keyword arguments are passed to the plot function.

    Returns
    -------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()

    default_kwargs = {"color": "black"}
    default_kwargs.update(kwargs)

    ax.plot(x, y, **default_kwargs)

    if n <= 0:
        return ax

    default_arrowprops = dict(arrowstyle="->", lw=1)
    default_arrowprops["color"] = default_kwargs.get("color", "black")
    default_arrowprops.update(arrowprops)

    # Calculate the length of each subsection
    total_points = len(x)
    section_length = total_points // n

    for i in range(n):
        idx = total_points - i * section_length
        if idx < 2:
            break
        # Add arrowhead at the last point of the subsection on ax
        ax.annotate(
            "",
            xy=(x[idx - 1], y[idx - 1]),
            xytext=(x[idx - 2], y[idx - 2]),
            arrowprops=default_arrowprops,
        )
    return ax
