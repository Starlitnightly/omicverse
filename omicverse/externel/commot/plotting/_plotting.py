from typing import Optional, Union
import sys
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import plotly
import seaborn as sns
from scipy import sparse
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from .._utils import plot_cluster_signaling_chord
from .._utils import plot_cluster_signaling_network
from .._utils import plot_cluster_signaling_network_multipair
from .._utils import plot_cell_signaling
from .._utils import plot_cell_signaling_compare
from .._utils import get_cmap_qualitative
from .._utils import leiden_clustering

def plot_cell_communication(
    adata: anndata.AnnData,
    database_name: str = None,
    pathway_name: str = None,
    lr_pair = None,
    keys = None,
    plot_method: str = "cell",
    background: str = "summary",
    background_legend: bool=False,
    clustering: str = None,
    summary: str = "sender",
    cmap: str = "coolwarm",
    cluster_cmap: dict = None,
    pos_idx: np.ndarray = np.array([0,1],int),
    ndsize: float = 1,
    scale: float = 1.0,
    normalize_v: bool = False,
    normalize_v_quantile: float = 0.95,
    arrow_color: str = "#333333",
    grid_density: float = 1.0,
    grid_knn: int = None,
    grid_scale: float = 1.0,
    grid_thresh: float = 1.0,
    grid_width: float = 0.005,
    stream_density: float = 1.0,
    stream_linewidth: float = 1,
    stream_cutoff_perc: float = 5,
    filename: str = None,
    ax: Optional[mpl.axes.Axes] = None
):
    """
    Plot spatial directions of cell-cell communication.
    
    .. image:: cell_communication.png
        :width: 500pt

    The cell-cell communication should have been computed by the function :func:`commot.tl.spatial_communication`.
    The cell-cell communication direction should have been computed by the function :func:`commot.tl.communication_direction`.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        Name of the ligand-receptor interaction database. 
    pathway_name
        Name of the signaling pathway to be plotted. Will be used only when ``lr_pair`` and ``keys`` are None.
        If none of ``pathway_name``, ``lr_pair``, and ``keys`` are given, the total signaling through all pairs will be plotted.
    lr_pair
        A tuple of ligand name and receptor name. If given, ``pathway_name`` will be ignored. Will be ignored if ``keys`` is given.
    keys
        A list of keys for example 'ligA-recA' for a LR pair or 'pathwayX' for a signaling pathway 'pathwayX'. 
        If given, pathway_name and lr_pair will be ignored.
        If more than one is given, the average will be plotted.
    plot_method
        'cell' plot vectors on individual cells. 
        'grid' plot interpolated vectors on regular grids.
        'stream' streamline plot.
    background
        'summary': scatter plot with color representing total sent or received signal.
        'image': the image in Visium data.
        'cluster': scatter plot with color representing cell clusters.
    background_legend
        Whether to include the background legend when background is set to `summary` or `cluster`.
    clustering
        The key for clustering result. Needed if background is set to `cluster`.
        For example, if ``clustering=='leiden'``, the clustering result should be available in ``.obs['leiden']``.
    summary
        If background is set to 'summary', the numerical value to plot for background.
        'sender': node color represents sender weight.
        'receiver': node color represents receiver weight.
    cmap
        matplotlib colormap name for node summary if numerical (background set to 'summary'), e.g., 'coolwarm'.
        plotly colormap name for node color if summary is (background set to 'cluster'). e.g., 'Alphabet'.
    cluster_cmap
        A dictionary that maps cluster names to colors when setting background to 'cluster'. If given, ``cmap`` will be ignored.
    pos_idx
        The coordinates to use for plotting (2D plot).
    ndsize
        The node size of the spots.
    scale
        The scale parameter passed to the matplotlib quiver function :func:`matplotlib.pyplot.quiver` for vector field plots.
        The smaller the value, the longer the arrows.
    normalize_v
        Whether the normalize the vector field to uniform lengths to highlight the directions without showing magnitudes.
    normalize_v_quantile
        The vector length quantile to use to normalize the vector field.
    arrow_color
        The color of the arrows.
    grid_density
        The density of grid if ``plot_method=='grid'``.
    grid_knn
        If ``plot_method=='grid'``, the number of nearest neighbors to interpolate the signaling directions from spots to grid points.
    grid_scale
        The scale parameter (relative to grid size) for the kernel function of mapping directions of spots to grid points.
    grid_thresh
        The threshold of interpolation weights determining whether to include a grid point. A smaller value gives a tighter cover of the tissue by the grid points.
    grid_width
        The value passed to the ``width`` parameter of the function :func:`matplotlib.pyplot.quiver` when ``plot_method=='grid'``.
    stream_density
        The density of stream lines passed to the ``density`` parameter of the function :func:`matplotlib.pyplot.streamplot` when ``plot_method=='stream'``.
    stream_linewidth
        The width of stream lines passed to the ``linewidth`` parameter of the function :func:`matplotlib.pyplot.streamplot` when ``plot_method=='stream'``.
    stream_cutoff_perc
        The quantile cutoff to ignore the weak vectors. Default to 5 that the vectors shorter than the 5% quantile will not be plotted.
    filename
        If given, save to the filename. For example 'ccc_direction.pdf'.
    ax
        An existing matplotlib ax (`matplotlib.axis.Axis`).

    Returns
    -------
    ax : matplotlib.axis.Axis
        The matplotlib ax object of the plot.

    """

    if not keys is None:
        ncell = adata.shape[0]
        V = np.zeros([ncell, 2], float)
        signal_sum = np.zeros([ncell], float)
        for key in keys:
            if summary == 'sender':
                V = V + adata.obsm['commot_sender_vf-'+database_name+'-'+key][:,pos_idx]
                signal_sum = signal_sum + adata.obsm['commot-'+database_name+"-sum-sender"]['s-'+key]
            elif summary == 'receiver':
                V = V + adata.obsm['commot_receiver_vf-'+database_name+'-'+key][:,pos_idx]
                signal_sum = signal_sum + adata.obsm['commot-'+database_name+"-sum-receiver"]['r-'+key]
        V = V / float( len( keys ) )
        signal_sum = signal_sum / float( len( keys ) )
    elif keys is None:
        if not lr_pair is None:
            vf_name = database_name+'-'+lr_pair[0]+'-'+lr_pair[1]
            sum_name = lr_pair[0]+'-'+lr_pair[1]
        elif not pathway_name is None:
            vf_name = database_name+'-'+pathway_name
            sum_name = pathway_name
        else:
            vf_name = database_name+'-total-total'
            sum_name = 'total-total'
        if summary == 'sender':
            V = adata.obsm['commot_sender_vf-'+vf_name][:,pos_idx]
            signal_sum = adata.obsm['commot-'+database_name+"-sum-sender"]['s-'+sum_name]
        elif summary == 'receiver':
            V = adata.obsm['commot_receiver_vf-'+vf_name][:,pos_idx]
            signal_sum = adata.obsm['commot-'+database_name+"-sum-receiver"]['r-'+sum_name]

    if background=='cluster' and not cmap in ['Plotly','Light24','Dark24','Alphabet']:
        cmap='Alphabet'
    if ax is None:
        fig, ax = plt.subplots()
    if normalize_v:
        V = V / np.quantile(np.linalg.norm(V, axis=1), normalize_v_quantile)
    plot_cell_signaling(
        adata.obsm["spatial"][:,pos_idx],
        V,
        signal_sum,
        cmap = cmap,
        cluster_cmap = cluster_cmap,
        plot_method = plot_method,
        background = background,
        clustering = clustering,
        background_legend = background_legend,
        adata = adata,
        summary = summary,
        scale = scale,
        ndsize = ndsize,
        filename = filename,
        arrow_color = arrow_color,
        grid_density = grid_density,
        grid_knn = grid_knn,
        grid_scale = grid_scale,
        grid_thresh = grid_thresh,
        grid_width = grid_width,
        stream_density = stream_density,
        stream_linewidth = stream_linewidth,
        stream_cutoff_perc = stream_cutoff_perc,
        ax = ax,
        # fig = fig,
    )
    return ax

def plot_cluster_communication_network(
    adata: anndata.AnnData,
    uns_names: list = None,
    clustering: str = None,
    quantile_cutoff: float = 0.99,
    p_value_cutoff: float = 0.05,
    self_communication_off: bool = False,
    filename: str = None,
    nx_node_size: float = 0.2,
    nx_node_cmap: str = "Plotly",
    nx_node_cluster_cmap: dict = None,
    nx_pos_idx: np.ndarray = np.array([0,1],int),
    nx_node_pos: str = "cluster",
    nx_edge_width_lb_quantile: float = 0.05,
    nx_edge_width_ub_quantile: float = 0.95,
    nx_edge_width_min: float = 1,
    nx_edge_width_max: float = 4,
    nx_edge_color: Union[str, np.ndarray] = "node",
    nx_edge_colormap = cm.Greys,
    nx_bg_pos: bool = True,
    nx_bg_color: str = "lavender",
    nx_bg_ndsize: float = 0.05,
):
    """
    Plot cluster-cluster communication as network.

    .. image:: cluster_communication.png
        :width: 500pt


    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    uns_names
        A list of keys for the cluster-level CCC stored in ``.uns`` data slot.
        When more than one is given, the average score will be used.
        For example, ``'commot_cluster-leiden-cellchat-Fgf7-Fgfr1'`` will plot the CCC inferred 
        with 'cellchat' LR database through 'Fgf7-Fgfr1' LR pair and for clustering stored in ``.obs['leiden']``.
    clustering
        Name of the clustering. The cluster labels should be available in ``.obs[clustering]``.
    quantile_cutoff
        The quantile cutoff for including an edge. Set to 1 to disable this criterion.
        The quantile_cutoff and p_value_cutoff works in the "or" logic to avoid missing
        significant signaling connections.
    p_value_cutoff
        The cutoff of p-value to include an edge.
    self_communication_off
        Whether to exclude self communications in the visualization.
    filename
        Filename for saving the figure. Set the name to end with '.pdf' or 'png'
        to specify format.
    nx_node_size
        Size of node representing clusters.
    nx_node_cmap
        The discrete color map to use for clusters. Choices: 
        'Plotly', 'Alphabet', 'Light24', 'Dark24'. Recommend to use 'Plotly'
        for ten clusters or fewer and 'Alphabet' for 10-24 clusters.
    nx_pos_idx
        Coordinates to use for the 2D plot.
    nx_node_pos
        'cluster', the predicted spatial location of clusters will be used. 
        If setting to 'cluster', run the function :func:`cluster_position` first to set the cluster positions.
        If None, the 'dot' layout from Graphviz package will be used.
    nx_edge_width_lb_quantile
        The quantile of communication connections to set for the lower bound of edge
        width.
    nx_edge_width_ub_quantile
        The quantile of communication connections to set for the upper bound of edge
        width.
    nx_edge_width_min
        Minimum width for plotted edges.
    nx_edge_width_max
        Maximum width for plotted edges.
    nx_edge_color
        If 'node', the color of an edge will be the same as the source node.
        If an array of numbers between [0,1], the nx_edge_colormap will be used
        to determine the edge colors.
    nx_edge_colormap
        The color map to use when nx_edge_color is an array of weights.
    nx_bg_pos
        Whether to plot the cells/positions as spatial background.
        Set to False when not using the spatial layout of clusters.
    nx_bg_color
        Color of the spatial background.
    nx_bg_ndsize
        Node size of the spatial background.

    """
    
    X_tmp = adata.uns[uns_names[0]]['communication_matrix'].copy()
    labels = list( X_tmp.columns.values )
    X = np.zeros_like(X_tmp.values, float)
    for i in range(len(uns_names)):
        X_tmp = adata.uns[uns_names[i]]['communication_matrix'].values.copy()
        p_values_tmp = adata.uns[uns_names[i]]['communication_pvalue'].values.copy()
        if not quantile_cutoff is None:
            cutoff = np.quantile(X_tmp.reshape(-1), quantile_cutoff)
        else:
            cutoff = np.inf
        tmp_mask = ( X_tmp < cutoff ) * ( p_values_tmp > p_value_cutoff )
        X_tmp[tmp_mask] = 0
        X = X + X_tmp
    X = X / len(uns_names)
    if self_communication_off:
        for i in range(X.shape[0]):
            X[i,i] = 0

    if nx_node_pos == "cluster":
        node_pos = [adata.uns["cluster_pos-"+clustering][labels[i]] for i in range(len(labels)) ]
        node_pos = np.array(node_pos)
        node_pos = node_pos[:, nx_pos_idx]
        lx = np.max(node_pos[:,0])-np.min(node_pos[:,0])
        ly = np.max(node_pos[:,1])-np.min(node_pos[:,1])
        pos_scale = max(lx, ly)
        node_pos = node_pos / pos_scale * 8.0
    else:
        node_pos = None
    if nx_bg_pos:
        background_pos = adata.obsm["spatial"][:,nx_pos_idx]
        background_pos = background_pos / pos_scale * 8.0
    else:
        background_pos = None
    plot_cluster_signaling_network(X,
        labels = labels,
        filename = filename,
        node_size = nx_node_size,
        node_colormap = nx_node_cmap,
        node_cluster_colormap = nx_node_cluster_cmap,
        node_pos = node_pos,
        edge_width_lb_quantile = nx_edge_width_lb_quantile,
        edge_width_ub_quantile = nx_edge_width_ub_quantile,
        edge_width_min = nx_edge_width_min,
        edge_width_max = nx_edge_width_max,
        edge_color = nx_edge_color,
        edge_colormap = nx_edge_colormap,
        background_pos = background_pos,
        background_ndcolor = nx_bg_color,
        background_ndsize = nx_bg_ndsize
    )


    legend_elements = []
    if nx_node_cluster_cmap is None:
        cluster_cmap = get_cmap_qualitative(nx_node_cmap)
        for i in range(len(labels)):
            legend_elements.append(Line2D([0],[0], marker='o',color='w', markerfacecolor=cluster_cmap[i], label=labels[i], markersize=10))
    elif not nx_node_cluster_cmap is None:
        for i in range(len(labels)):
            legend_elements.append(Line2D([0],[0], marker='o',color='w', markerfacecolor=nx_node_cluster_cmap[labels[i]], label=labels[i], markersize=10))
    
    fig, ax = plt.subplots()
    tmp_filename,tmp_type = filename.split('.')
    ax.legend(handles=legend_elements, loc='center')
    ax.axis('off')
    fig.savefig(tmp_filename+"_cluster_legend."+tmp_type, bbox_inches='tight')


    

def plot_communication_dependent_genes(
    df_deg: pd.DataFrame,
    df_yhat: pd.DataFrame,
    show_gene_names: bool = True,
    top_ngene_per_cluster: int = -1,
    colormap: str = 'magma',
    cluster_colormap: str = 'Plotly',
    font_scale: float = 1.4,
    filename = None,
    return_genes = False
):
    """
    Plot smoothed gene expression of the detected communication-dependent genes.
    Takes input from the function :func:`commot.tl.communication_deg_clustering`.

    .. image:: communication_deg.png
        :width: 500pt

    Parameters
    ----------
    df_deg
        A data frame where each row is a gene and 
        the columns should include 'waldStat', 'pvalue', 'cluster'.
        Output of ``tl.communication_deg_clustering``
    df_yhat
        A data frame where each row is the smoothed expression of a gene.
        Output of ``tl.communication_deg_clustering``.
    show_gene_names
        Whether to plot the gene names.
    top_ngene_per_cluster
        If non-negative, plot the top_ngene_per_cluster genes 
        with highest wald statistics.
    colormap
        The colormap for the heatmap. Choose from available colormaps from ``seaborn``.
    cluster_colormap
        The qualitative colormap for annotating gene cluster labels.
        Choose from 'Plotly', 'Alphabet', 'Light24', 'Dark24'.
    font_scale
        Font size.
    filename
        Filename for saving the figure. Set the name to end with '.pdf' or 'png'
        to specify format.
    return_genes
        Whether to return the list of plotted genes.
    
    Returns
    -------
    genes
        Returns the gene list being plotted if return_genes is True.
    """
    cmap = get_cmap_qualitative(cluster_colormap)
    wald_stats = df_deg['waldStat'].values
    pvalue = df_deg['pvalue'].values
    labels = np.array( df_deg['cluster'].values, int)
    nlabel = np.max(labels)+1
    yhat_mat = df_yhat.values
    peak_locs = []
    for i in range(nlabel):
        tmp_idx = np.where(labels==i)[0]
        tmp_y = yhat_mat[tmp_idx,:]
        peak_locs.append(np.mean(np.argmax(tmp_y, axis=1)))
    cluster_order = np.argsort(peak_locs)
    idx = np.array([])
    row_colors = []
    for i in cluster_order:
        tmp_idx = np.where(labels==i)[0]
        tmp_order = np.argsort(-wald_stats[tmp_idx])
        if top_ngene_per_cluster >= 0:
            top_ngene = min(len(tmp_idx), top_ngene_per_cluster)
        else:
            top_ngene = len(tmp_idx)
        idx = np.concatenate((idx, tmp_idx[tmp_order][:top_ngene]))
        for j in range(top_ngene):
            row_colors.append(cmap[i % len(cmap)])

    sns.set(font_scale=font_scale)
    g = sns.clustermap(df_yhat.iloc[idx], 
        row_cluster=False, 
        col_cluster=False, 
        row_colors=row_colors,
        cmap = colormap,
        xticklabels = False,
        yticklabels = show_gene_names,
        linewidths=0)
    g.cax.set_position([.1, .2, .03, .45])
    plt.savefig(filename, dpi=300)

    if return_genes:
        return list( df_deg.iloc[idx].index )

def plot_communication_impact(
    df_impact: pd.DataFrame,
    summary: str = None,
    show_gene_names: str = True,
    show_comm_names: str = True,
    top_ngene: int = -1,
    top_ncomm: int = -1,
    colormap: str = 'rocket',
    font_scale: float = 1.4,
    filename: str = None,
    cluster_knn: str = 5,
    cluster_res: float = 0.5,
    cluster_colormap: str = "Plotly",
    linewidth = 0.0,
    vmin = 0.0,
    vmax = 1.0
):
    """
    Plot communication impact obtained by running the function :func:`commot.tl.communication_impact`.

    .. image:: communication_impact.png
        :width: 300pt

    Parameters
    ----------
    df_impact
        The output from ``tl.communication_impact``.
    summary
        If 'receiver', the received signals are plotted as rows. 
        If 'sender', the sent signals are plotted as rows.
        If None, both are plotted.
    show_gene_names
        Whether to plot gene names as x ticks.
    show_comm_names
        Whether to plot communication names as y ticks.
    top_ngene
        The number of most impacted genes to plot as columns.
        If -1, all genes in ``df_impact`` are plotted.
    top_ncomm
        The number of communications with most impacts to plot as rows.
        If -1, all communications in ``df_impact`` are plotted.
    colormap
        The colormap for the heatmap. Choose from available colormaps from ``seaborn``.
    font_scale
        Font size.
    filename
        Filename for saving the figure. Set the name to end with '.pdf' or 'png'
        to specify format.
    cluster_knn
        Number of nearest neighbors when clustering the rows and columns.
    cluster_res
        The resolution paratemeter when running leiden clustering.
    cluster_colormap
        The qualitative colormap for annotating gene cluster labels.
        Choose from 'Plotly', 'Alphabet', 'Light24', 'Dark24'.

    """
    index_names = list( df_impact.index )
    tmp_idx = []
    if summary == 'receiver':
        for i in range(len(index_names)):
            index_name = index_names[i]
            tmp_n = min(len(index_name), 8)
            if index_name[0] == 'r':
                tmp_idx.append(i)
    elif summary == 'sender':
        for i in range(len(index_names)):
            index_name = index_names[i]
            tmp_n = min(len(index_name), 6)
            if index_name[0] == 's':
                tmp_idx.append(i)
    elif summary is None:
        tmp_idx = [i for i in range(len(index_names))]
    tmp_idx = np.array(tmp_idx, int)
    df_plot = df_impact.iloc[tmp_idx]
    
    mat = df_plot.values
    sum_gene = np.sum(np.abs(mat), axis=0)
    sum_comm = np.sum(np.abs(mat), axis=1)
    if top_ngene == -1:
        top_ngene = mat.shape[1]
    else:
        top_ngene = min(top_ngene, mat.shape[1])
    if top_ncomm == -1:
        top_ncomm = mat.shape[0]
    else:
        top_ncomm = min(top_ncomm, mat.shape[0])
    row_idx = np.argsort(-sum_comm)[:top_ncomm]
    col_idx = np.argsort(-sum_gene)[:top_ngene]

    df_plot = ( df_plot.iloc[row_idx,:] ).iloc[:,col_idx]
    mat = df_plot.values
    cmap = get_cmap_qualitative(cluster_colormap)
    if mat.shape[1] > 10:
        mat_pca = PCA(n_components=np.min([10,mat.shape[1],mat.shape[0]]), svd_solver='full').fit_transform(mat)
    else:
        mat_pca = mat
    D = distance_matrix(mat_pca, mat_pca)
    labels = leiden_clustering(D, k=cluster_knn, resolution=cluster_res)
    row_idx, row_colors = reorder(labels, -np.abs(mat.sum(axis=1)), -np.abs(mat.sum(axis=1)), cmap)
    if mat.shape[0] > 10:
        mat_pca = PCA(n_components=np.min([10,mat.shape[1],mat.shape[0]]), svd_solver='full').fit_transform(mat.T)
    else:
        mat_pca = mat.T
    D = distance_matrix(mat_pca, mat_pca)
    labels = leiden_clustering(D, k=cluster_knn, resolution=cluster_res)
    col_idx, col_colors = reorder(labels, -np.abs(mat.sum(axis=0)), -np.abs(mat.sum(axis=0)), cmap)

    sns.set(font_scale=font_scale)
    g = sns.clustermap( ( df_plot.iloc[row_idx,:] ).iloc[:,col_idx], 
        row_cluster = False, 
        col_cluster  =False, 
        row_colors = row_colors,
        col_colors = col_colors,
        cmap = colormap,
        xticklabels = show_gene_names,
        yticklabels = show_comm_names,
        linewidths = linewidth,
        square = True,
        vmin = vmin,
        vmax = vmax)
    g.cax.set_position([0.01, .2, .03, .45])
    plt.savefig(filename, dpi=300)


def reorder(labels, cofactor_cluster, cofactor_sample, cmap):
    nlabels = np.max(labels) + 1
    cofactor = []
    for i in range(nlabels):
        tmp_idx = np.where(labels==i)[0]
        cofactor.append(cofactor_cluster[tmp_idx].mean())
    cluster_order = np.argsort(cofactor)
    idx = np.array([])
    colors = []
    for i in cluster_order:
        tmp_idx = np.where(labels==i)[0]
        tmp_order = np.argsort(cofactor_sample[tmp_idx])
        idx = np.concatenate((idx, tmp_idx[tmp_order]))
        for j in range(len(tmp_idx)):
            colors.append(cmap[i % len(cmap)])
    return np.array(idx, int), colors


class pvalueNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False):
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        left = np.log10(self.vmax); right = np.log10(self.vmin)
        value_log10 = np.log10(value)
        y = (value_log10 - left) / (right - left)
        return y


def plot_cluster_communication_dotplot(
    adata: anndata.AnnData,
    database_name: str = None,
    pathway_name: str = None,
    clustering: str = None,
    keys = None,
    p_value_cutoff: float = 0.05,
    p_value_vmin: float = 1e-3,
    size_max = 20,
    size_min = 10,
    vmax_quantile = 0.99,
    vmin_quantile = 0.0,
    cmap = 'cool',
    filename = None,
    font_scale = 0.5,
    top_nclus = -1,
    top_ncomm = -1,
    cluster_x = False,
    cluster_y = False,
    cluster_knn = 5,
    cluster_res = 1.0
):
    """
    Plot cluster-cluster communication through multiple ligand-receptor pairs as dotplot.

    .. image:: cluster_communication_dotplot.png
        :width: 300pt

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var`` after running ``tl.spatial_communication``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        The name of the LR database.
    pathway_name
        Name of the a signaling pathway or a list of several signaling pathways. 
        If given, all LR pairs of this signaling pathway(s) will be plotted.
        If ``keys`` is specified, ``pathway_name`` will be ignored.
    clustering
        Name of the clustering.        
    keys
        A list of keys for example 'ligA-recA' for a LR pair or 'pathwayX' for a signaling pathway 'pathwayX'. 
        If given, ``pathway_name`` will be ignored.
    p_value_cutoff
        Cutoff for being considered significant.
    p_value_vmin
        The lower bound of p-value corresponding to the biggest dot size.
    size_max
        Size of biggest dot (corresponding to p-value <= p_value_min).
    size_min
        Size of smallest dot (corresponding to p-value = p_value_cutoff).
    vmax_quantile
        The quantile of cluster-cluster communication weights for setting vmax of colormap.
    vmin_quantile
        The quantile of cluster-cluster communication weights for setting vmin of colormap.
    cmap
        The colormap for the nodes.
    filename
        Filename for saving the figure. Set the name to end with '.pdf' or 'png'
        to specify format.
    font_scale
        Font size.
    top_nclus
        If not -1, the top number of cluster-cluster pairs with the highest total communication
        weight among the signaling pathways are plotted.
    top_ncomm
        If not -1, the top number of ligand-recepter pairs with the highest total communication
        weight among the cluster-cluster pairs are plotted.
    cluster_x
        Whether to reorder the cluster-cluster pairs according to their patterns among
        the ligand-receptor pairs.
    cluster_y
        Whether to reorder the ligand-receptor pairs according to their patterns among
        the cluster-cluster pairs.
    cluster_knn
        The k value of knn graph for clustering if cluster_x or cluster_y is True.
    cluster_res
        The resolution of leiden clustering algorithm if cluster_x or cluster_y is True.

    """
    if keys is None:
        keys = []
        if isinstance(pathway_name, str):
            pathways = [pathway_name]
        elif isinstance(pathway_name, list):
            pathways = pathway_name
        df_ligrec = adata.uns['commot-%s-info' % database_name]['df_ligrec']
        for i in range(df_ligrec.shape[0]):
            if df_ligrec.iloc[i][2] in pathways:
                keys.append(df_ligrec.iloc[i][0]+'-'+df_ligrec.iloc[i][1])
        keys.extend( pathways )
    
    X_tmp = adata.uns['commot_cluster-'+clustering+'-'+database_name+'-'+keys[0]]['communication_matrix'].copy()
    labels = list( X_tmp.columns.values )
    name_matrix = np.empty([len(labels), len(labels)], object)
    for i in range(len(labels)):
        for j in range(len(labels)):
            name_matrix[i,j] = labels[i]+'->'+labels[j]
    x_names = name_matrix.flatten()

    y_names = []
    S = np.empty([len(x_names), len(keys)], float)
    P = np.empty([len(x_names), len(keys)], float)
    for ikey in range(len(keys)):
        key = keys[ikey]
        y_names.append(key)
        S_tmp = adata.uns['commot_cluster-%s-%s-%s' % (clustering, database_name, key)]['communication_matrix'].values.copy()
        P_tmp = adata.uns['commot_cluster-%s-%s-%s' % (clustering, database_name, key)]['communication_pvalue'].values.copy()
        S[:,ikey] = S_tmp.flatten()[:]
        P[:,ikey] = P_tmp.flatten()[:]
    y_names = np.array( y_names, str )

    P_mask = P <= p_value_cutoff
    P_mask_row = P_mask.sum(axis=1).astype(bool)
    P_mask_col = P_mask.sum(axis=0).astype(bool)

    P = P[P_mask_row,:][:,P_mask_col]
    S = S[P_mask_row,:][:,P_mask_col]
    x_names = x_names[P_mask_row]
    y_names = y_names[P_mask_col]

    if top_nclus > 0:
        tmp_idx = np.argsort(-S.sum(axis=1))[:top_nclus]
        P = P[tmp_idx,:]; S = S[tmp_idx,:]; x_names = x_names[tmp_idx]
    if top_ncomm > 0:
        tmp_idx = np.argsort(-S.sum(axis=0))[:top_ncomm]
        P = P[:,tmp_idx]; S = S[:,tmp_idx]; y_names = y_names[tmp_idx]

    if cluster_x:
        mat = S
        dummy_cmap = ['r']
        if mat.shape[1] > 10:
            mat_pca = PCA(n_components=np.min([10,mat.shape[1],mat.shape[0]]), svd_solver='full').fit_transform(mat)
        else:
            mat_pca = mat
        D = distance_matrix(mat_pca, mat_pca)
        labels = leiden_clustering(D, k=cluster_knn, resolution=cluster_res)
        x_idx, _ = reorder(labels, -np.abs(mat.sum(axis=1)), -np.abs(mat.sum(axis=1)), dummy_cmap)
        P = P[x_idx,:]; S = S[x_idx,:]; x_names = x_names[x_idx]
    if cluster_y:
        mat = S
        dummy_cmap = ['r']
        if mat.shape[0] > 10:
            mat_pca = PCA(n_components=np.min([10,mat.shape[1],mat.shape[0]]), svd_solver='full').fit_transform(mat.T)
        else:
            mat_pca = mat.T
        D = distance_matrix(mat_pca, mat_pca)
        labels = leiden_clustering(D, k=cluster_knn, resolution=cluster_res)
        y_idx, _ = reorder(labels, -np.abs(mat.sum(axis=0)), -np.abs(mat.sum(axis=0)), dummy_cmap)
        P = P[:,y_idx]; S = S[:,y_idx]; y_names = y_names[y_idx]
        
    vmax = np.quantile(S, vmax_quantile)
    vmin = np.quantile(S, vmin_quantile)

    data_plot = []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i,j] <= p_value_cutoff:
                data_plot.append([x_names[i], y_names[j], S[i,j], max(P[i,j], p_value_vmin)])
    df_plot = pd.DataFrame(data=data_plot, columns=['x','y','color_col','size_col'])
        
    sns.set_theme(style="whitegrid", font_scale=font_scale)
    g = sns.relplot(
        data=df_plot,
        x="x", y="y", hue="color_col", size="size_col",
        palette=cmap, hue_norm=(vmin, vmax), edgecolor=".7", legend='brief',
        height=10, sizes=(size_min, size_max), size_norm=pvalueNormalize(vmin=p_value_vmin, vmax=p_value_cutoff),
    )
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")
    plt.savefig(filename, dpi=500, bbox_inches = 'tight')

    return
