from typing import Tuple, Optional, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from pygam import LinearGAM, s

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.patheffects as PathEffects
import seaborn as sns

from .gam import fit_velo_peak
from .metrics import cross_boundary_correctness
from .utils import flatten, uniform_downsample_cells
from .kernel_density_smooth import kde2d, kde2d_to_mean_and_sigma


# Cell cycle genes are adapted from `dynamo`.
G1S_genes_human = [
        "ARGLU1",
        "BRD7",
        "CDC6",
        "CLSPN",
        "ESD",
        "GINS2",
        "GMNN",
        "LUC7L3",
        "MCM5",
        "MCM6",
        "NASP",
        "PCNA",
        "PNN",
        "SLBP",
        "SRSF7",
        "SSR3",
        "ZRANB2"]
S_genes_human = [
        "ASF1B",
        "CALM2",
        "CDC45",
        "CDCA5",
        "CENPM",
        "DHFR",
        "EZH2",
        "FEN1",
        "HIST1H2AC",
        "HIST1H4C",
        "NEAT1",
        "PKMYT1",
        "PRIM1",
        "RFC2",
        "RPA2",
        "RRM2",
        "RSRC2",
        "SRSF5",
        "SVIP",
        "TOP2A",
        "TYMS",
        "UBE2T",
        "ZWINT"]
G2M_genes_human = [
        "AURKB",
        "BUB3",
        "CCNA2",
        "CCNF",
        "CDCA2",
        "CDCA3",
        "CDCA8",
        "CDK1",
        "CKAP2",
        "DCAF7",
        "HMGB2",
        "HN1",
        "KIF5B",
        "KIF20B",
        "KIF22",
        "KIF23",
        "KIFC1",
        "KPNA2",
        "LBR",
        "MAD2L1",
        "MALAT1",
        "MND1",
        "NDC80",
        "NUCKS1",
        "NUSAP1",
        "PIF1",
        "PSMD11",
        "PSRC1",
        "SMC4",
        "TIMP1",
        "TMEM99",
        "TOP2A",
        "TUBB",
        "TUBB4B",
        "VPS25"]
M_genes_human = [
        "ANP32B",
        "ANP32E",
        "ARL6IP1",
        "AURKA",
        "BIRC5",
        "BUB1",
        "CCNA2",
        "CCNB2",
        "CDC20",
        "CDC27",
        "CDC42EP1",
        "CDCA3",
        "CENPA",
        "CENPE",
        "CENPF",
        "CKAP2",
        "CKAP5",
        "CKS1B",
        "CKS2",
        "DEPDC1",
        "DLGAP5",
        "DNAJA1",
        "DNAJB1",
        "GRK6",
        "GTSE1",
        "HMG20B",
        "HMGB3",
        "HMMR",
        "HN1",
        "HSPA8",
        "KIF2C",
        "KIF5B",
        "KIF20B",
        "LBR",
        "MKI67",
        "MZT1",
        "NUF2",
        "NUSAP1",
        "PBK",
        "PLK1",
        "PRR11",
        "PSMG3",
        "PWP1",
        "RAD51C",
        "RBM8A",
        "RNF126",
        "RNPS1",
        "RRP1",
        "SFPQ",
        "SGOL2",
        "SMARCB1",
        "SRSF3",
        "TACC3",
        "THRAP3",
        "TPX2",
        "TUBB4B",
        "UBE2D3",
        "USP16",
        "WIBG",
        "YWHAH",
        "ZNF207"]
MG1_genes_human = [
        "AMD1",
        "ANP32E",
        "CBX3",
        "CDC42",
        "CNIH4",
        "CWC15",
        "DKC1",
        "DNAJB6",
        "DYNLL1",
        "EIF4E",
        "FXR1",
        "GRPEL1",
        "GSPT1",
        "HMG20B",
        "HSPA8",
        "ILF2",
        "KIF5B",
        "KPNB1",
        "LARP1",
        "LYAR",
        "MORF4L2",
        "MRPL19",
        "MRPS2",
        "MRPS18B",
        "NUCKS1",
        "PRC1",
        "PTMS",
        "PTTG1",
        "RAN",
        "RHEB",
        "RPL13A",
        "SRSF3",
        "SYNCRIP",
        "TAF9",
        "TMEM138",
        "TOP1",
        "TROAP",
        "UBE2D3",
        "ZNF593"]

def polar_plot(adata, phase_key='phase', layer='velocity_gv', show_names=False, show_markers=True, show=False, **kwargs):
    """
    Polar plot adapted from VeloCycle(https://github.com/lamanno-epfl/velocycle/blob/main/velocycle/plots.py) with some modifications.
    """
    gene_names = np.array([a for a in adata.var_names])
    markers = G1S_genes_human + S_genes_human + G2M_genes_human + M_genes_human + MG1_genes_human
    phases_list = [G1S_genes_human, S_genes_human, G2M_genes_human, M_genes_human, MG1_genes_human, [i for i in gene_names if i.upper() not in markers]]
    gs = []
    gradient = []
    for i in range(len(phases_list)):
        for j in range(len(phases_list[i])):
            if phases_list[i][j] not in gene_names:
                continue
            else:
                gs.append(phases_list[i][j])
                gradient.append(i)

    color_gradient_map = pd.DataFrame({'Gene': gs,  'Color': gradient}).set_index('Gene').to_dict()['Color']
    colored_gradient = pd.Series(gs).map(color_gradient_map)
    gs = np.array(gs)
    
    for i,j in zip(gs, colored_gradient):
        if np.isnan(j):
            print(i)
         
    # rescale phase for polar plot
    phase = adata.obs[phase_key].values
    phase -= phase.min()
    phase = phase / phase.max() * 2 * np.pi

    gam_kwargs = {
        'n_splines': 7,
        'spline_order': 3,
    }
    gam_kwargs.update(kwargs)

    df_peak = fit_velo_peak(
        adata, 
        genes=gs, 
        tkey=phase_key, 
        layer=layer, 
        log_norm=True, 
        max_iter=1000, 
        **gam_kwargs
    )
    
    # Extract the peak phase and magnitude values.
    angle = df_peak['phase'].values
    magnitude = df_peak['magnitude'].values
    
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(projection='polar')
    
    # First: only plot dots with a color assignment
    angle_subset = angle[~np.isnan(colored_gradient.values)]
    r_subset = magnitude[~np.isnan(colored_gradient.values)]
    color_subset = colored_gradient.values[~np.isnan(colored_gradient.values)]
    
    # Remove genes with negative velo
    angle_subset = angle_subset[r_subset>=0]
    color_subset = color_subset[r_subset>=0]
    gene_names_subset = gs[r_subset>=0]
    r_subset = r_subset[r_subset>=0]
    
    # Plot all genes in phases list
    cell_cycle_cmap = {0:'firebrick', 1:'orange', 2:'yellowgreen', 3:'teal', 4:'royalblue', 5: 'black'} 
    ax.scatter(angle_subset, r_subset, c=[cell_cycle_cmap[i] for i in color_subset], s=50, alpha=0.2, edgecolor='none', rasterized=True)
    
    angle_subset_markers = angle_subset[color_subset!=5]
    r_subset_markers = r_subset[color_subset!=5]
    gene_names_subset_markers = gene_names_subset[color_subset!=5]
    color_subset_markers = color_subset[color_subset!=5]
    
    ax.scatter(angle_subset_markers, r_subset_markers, c=[cell_cycle_cmap[i] for i in color_subset_markers], s=80, alpha=1, edgecolor='none',rasterized=True)
    
    # Annotate genes
    if show_markers:
        for (i, txt), c in zip(enumerate(gene_names_subset_markers), colored_gradient.values):
            ix = np.where(np.array(gene_names_subset_markers)==txt)[0][0]
            ax.annotate(txt[0]+txt[1:].upper(), (angle_subset_markers[ix], r_subset_markers[ix]+0.02))

    if show_names:
        for (i, txt), c in zip(enumerate(gs), colored_gradient.values):
            ix = np.where(gs==txt)[0][0]
            ax.annotate(txt[0]+txt[1:].upper(), (angle[ix], magnitude[ix]+0.02))

    plt.xlim(0, 2*np.pi)
    plt.ylim(-1, )
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],["0", "π/2", "π", "3π/2", "2π"])
    plt.tight_layout()
    if show:
        plt.show();
    else: 
        return ax


def cbc_heatmap(
        adata,
        xkey:str='Ms',
        vkey:str='velocity',
        cluster_key:str='clusters', 
        basis:str='pca',
        neighbor_key:str='neighbors',
        vector:str='velocity', 
        corr_func='cosine',
        cmap = 'bwr',
        annot: bool = True,
        custom_order: list = None,
        ):
    """
    Visualize the CBC score of every possible transitions between clusters using heatmap. 
    """
    clusters = adata.obs[cluster_key].unique()
    rows, cols = clusters.to_list(), clusters.to_list()
    cluster_edges = []
    for i in clusters:
        for j in clusters:
            if i == j:
                continue
            else:
                cluster_edges.append((i, j))

    scores, _ = cross_boundary_correctness(
        adata, xkey=xkey, vkey=vkey, cluster_key=cluster_key, cluster_edges=cluster_edges,
        basis=basis, neighbor_key=neighbor_key, vector=vector, corr_func=corr_func, return_raw=False)

    df = pd.DataFrame(index=rows, columns=cols)
    for row in rows:
        for col in cols:
            if row == col:
                df.loc[row, col] = np.nan
            else:
                df.loc[row, col] = scores[(row, col)]

    df = df.apply(pd.to_numeric)
    if custom_order:
        df = df.reindex(index=custom_order, columns=custom_order)
        
    abs_max = np.nanmax(np.abs(df.values))

    sns.set_style("whitegrid")
    sns.heatmap(df, cmap=cmap, annot=annot, center=0, vmin=-abs_max, vmax=abs_max)
    plt.show()


def scatter_plot_mm(adata,
                 genes,
                 vukey,
                 vskey,
                 vckey,
                 by='cus',
                 color_by='celltype',
                 n_cols=5,
                 axis_on=True,
                 frame_on=True,
                 velocity_arrows=False,
                 downsample=1,
                 figsize=None,
                 pointsize=2,
                 cmap='coolwarm',
                 view_3d_elev=None,
                 view_3d_azim=None,
                 full_name=False
                 ):
    """Gene scatter plot.

    This function plots phase portraits of the specified plane.
    Modification of Multivelo

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    genes: `str`,  list of `str`
        List of genes to plot.
    by: `str` (default: `cus`)
        Plot unspliced-spliced plane if `us`. Plot chromatin-unspliced plane
        if `cu`.
        Plot 3D phase portraits if `cus`.
    color_by: `str` (default: `state`)
        Color by the cell labels like leiden, louvain, celltype, etc.
        The color field must be present in `.uns`, which can be pre-computed 
        with `scanpy.pl.scatter`.
        When `by=='us'`, `color_by` can also be `c`, which displays the log
        accessibility on U-S phase portraits.
    n_cols: `int` (default: 5)
        Number of columns to plot on each row.
    axis_on: `bool` (default: `True`)
        Whether to show axis labels.
    frame_on: `bool` (default: `True`)
        Whether to show plot frames.
    title_more_info: `bool` (default: `False`)
        Whether to display model, direction, and likelihood information for
        the gene in title.
    velocity_arrows: `bool` (default: `False`)
        Whether to show velocity arrows of cells on the phase portraits.
    downsample: `int` (default: 1)
        How much to downsample the cells. The remaining number will be
        `1/downsample` of original.
    figsize: `tuple` (default: `None`)
        Total figure size.
    pointsize: `float` (default: 2)
        Point size for scatter plots.
    markersize: `float` (default: 5)
        Point size for switch time points.
    linewidth: `float` (default: 2)
        Line width for connected anchors.
    cmap: `str` (default: `coolwarm`)
        Color map for log accessibilities or other continuous color keys when
        plotting on U-S plane.
    view_3d_elev: `float` (default: `None`)
        Matplotlib 3D plot `elev` argument. `elev=90` is the same as U-S plane,
        and `elev=0` is the same as C-U plane.
    view_3d_azim: `float` (default: `None`)
        Matplotlib 3D plot `azim` argument. `azim=270` is the same as U-S
        plane, and `azim=0` is the same as C-U plane.
    full_name: `bool` (default: `False`)
        Show full names for chromatin, unspliced, and spliced rather than
        using abbreviated terms c, u, and s.
    """
    if by not in ['us', 'cu', 'cus']:
        raise ValueError("'by' argument must be one of ['us', 'cu', 'cus']")
    if by == 'us' and color_by == 'c':
        cell_annot = None
    elif color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        cell_annot = None
        colors = adata.obs[color_by].values
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) \
            and color_by+'_colors' in adata.uns.keys():
        cell_annot = adata.obs[color_by].cat.categories
        if isinstance(adata.uns[f'{color_by}_colors'], dict):
            colors = list(adata.uns[f'{color_by}_colors'].values())
        else:
            colors = adata.uns[f'{color_by}_colors']
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata'
                         ' obs, and the colors of categories can be found in'
                         ' adata uns.')

    downsample = np.clip(int(downsample), 1, 10)
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        print(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        return
    if gn < n_cols:
        n_cols = gn
    if by == 'cus':
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                figsize=(3.2*n_cols, 2.7*(-(-gn // n_cols)))
                                if figsize is None else figsize,
                                subplot_kw={'projection': '3d'})
    else:
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                figsize=(2.7*n_cols, 2.4*(-(-gn // n_cols)))
                                if figsize is None else figsize)
    fig.patch.set_facecolor('white')
    count = 0
    for gene in genes:
        u = adata[:, gene].layers['Mu'].copy() if 'Mu' in adata.layers \
            else adata[:, gene].layers['unspliced'].copy()
        s = adata[:, gene].layers['Ms'].copy() if 'Ms' in adata.layers \
            else adata[:, gene].layers['spliced'].copy()
        u = u.A if sp.issparse(u) else u
        s = s.A if sp.issparse(s) else s
        u, s = np.ravel(u), np.ravel(s)
        if 'ATAC' not in adata.layers.keys() and \
                'Mc' not in adata.layers.keys():
            raise ValueError('Cannot find ATAC data in adata layers.')
        elif 'ATAC' in adata.layers.keys():
            c = adata[:, gene].layers['ATAC'].copy()
            c = c.A if sp.issparse(c) else c
            c = np.ravel(c)
        elif 'Mc' in adata.layers.keys():
            c = adata[:, gene].layers['Mc'].copy()
            c = c.A if sp.issparse(c) else c
            c = np.ravel(c)

        if velocity_arrows:
            if vukey in adata.layers.keys():
                vu = adata[:, gene].layers[vukey].copy()
            else:
                vu = np.zeros(adata.n_obs)
            max_u = np.max([np.max(u), 1e-6])
            u /= max_u
            vu = np.ravel(vu)
            vu /= np.max([np.max(np.abs(vu)), 1e-6])
            if vskey in adata.layers.keys():
                vs = adata[:, gene].layers[vskey].copy()
            else:
                raise ValueError(f'Splicing velocity {vskey} can not be found in adata layers.')
            max_s = np.max([np.max(s), 1e-6])
            s /= max_s
            vs = np.ravel(vs)
            vs /= np.max([np.max(np.abs(vs)), 1e-6])
            if vckey in adata.layers.keys():
                vc = adata[:, gene].layers[vckey].copy()
                max_c = np.max([np.max(c), 1e-6])
                c /= max_c
                vc = np.ravel(vc)
                vc /= np.max([np.max(np.abs(vc)), 1e-6])

        row = count // n_cols
        col = count % n_cols
        ax = axs[row, col]
        if cell_annot is not None:
            for i in range(len(cell_annot)):
                filt = adata.obs[color_by] == cell_annot[i]
                filt = np.ravel(filt)
                if by == 'us':
                    if velocity_arrows:
                        ax.quiver(s[filt][::downsample], u[filt][::downsample],
                                  vs[filt][::downsample],
                                  vu[filt][::downsample], color=colors[i],
                                  alpha=0.5, scale_units='xy', scale=10,
                                  width=0.005, headwidth=4, headaxislength=5.5)
                    else:
                        ax.scatter(s[filt][::downsample],
                                   u[filt][::downsample], s=pointsize,
                                   c=colors[i], alpha=0.7)
                elif by == 'cu':
                    if velocity_arrows:
                        ax.quiver(u[filt][::downsample],
                                  c[filt][::downsample],
                                  vu[filt][::downsample],
                                  vc[filt][::downsample], color=colors[i],
                                  alpha=0.5, scale_units='xy', scale=10,
                                  width=0.005, headwidth=4, headaxislength=5.5)
                    else:
                        ax.scatter(u[filt][::downsample],
                                   c[filt][::downsample], s=pointsize,
                                   c=colors[i], alpha=0.7)
                else:
                    if velocity_arrows:
                        ax.quiver(s[filt][::downsample],
                                  u[filt][::downsample], c[filt][::downsample],
                                  vs[filt][::downsample],
                                  vu[filt][::downsample],
                                  vc[filt][::downsample],
                                  color=colors[i], alpha=0.4, length=0.1,
                                  arrow_length_ratio=0.5, normalize=True)
                    else:
                        ax.scatter(s[filt][::downsample],
                                   u[filt][::downsample],
                                   c[filt][::downsample], s=pointsize,
                                   c=colors[i], alpha=0.7)
        elif color_by == 'c':
            outlier = 99.8
            non_zero = (u > 0) & (s > 0) & (c > 0)
            non_outlier = u < np.percentile(u, outlier)
            non_outlier &= s < np.percentile(s, outlier)
            non_outlier &= c < np.percentile(c, outlier)
            c -= np.min(c)
            c /= np.max(c)
            if velocity_arrows:
                ax.quiver(s[non_zero & non_outlier][::downsample],
                          u[non_zero & non_outlier][::downsample],
                          vs[non_zero & non_outlier][::downsample],
                          vu[non_zero & non_outlier][::downsample],
                          np.log1p(c[non_zero & non_outlier][::downsample]),
                          alpha=0.5,
                          scale_units='xy', scale=10, width=0.005,
                          headwidth=4, headaxislength=5.5, cmap=cmap)
            else:
                ax.scatter(s[non_zero & non_outlier][::downsample],
                           u[non_zero & non_outlier][::downsample],
                           s=pointsize,
                           c=np.log1p(c[non_zero & non_outlier][::downsample]),
                           alpha=0.8, cmap=cmap)
        else:
            if by == 'us':
                if velocity_arrows:
                    ax.quiver(s[::downsample], u[::downsample],
                              vs[::downsample], vu[::downsample],
                              colors[::downsample], alpha=0.5,
                              scale_units='xy', scale=10, width=0.005,
                              headwidth=4, headaxislength=5.5, cmap=cmap)
                else:
                    ax.scatter(s[::downsample], u[::downsample], s=pointsize,
                               c=colors[::downsample], alpha=0.7, cmap=cmap)
            elif by == 'cu':
                if velocity_arrows:
                    ax.quiver(u[::downsample], c[::downsample],
                              vu[::downsample], vc[::downsample],
                              colors[::downsample], alpha=0.5,
                              scale_units='xy', scale=10, width=0.005,
                              headwidth=4, headaxislength=5.5, cmap=cmap)
                else:
                    ax.scatter(u[::downsample], c[::downsample], s=pointsize,
                               c=colors[::downsample], alpha=0.7, cmap=cmap)
            else:
                if velocity_arrows:
                    ax.quiver(s[::downsample], u[::downsample],
                              c[::downsample], vs[::downsample],
                              vu[::downsample], vc[::downsample],
                              colors[::downsample], alpha=0.4, length=0.1,
                              arrow_length_ratio=0.5, normalize=True,
                              cmap=cmap)
                else:
                    ax.scatter(s[::downsample], u[::downsample],
                               c[::downsample], s=pointsize,
                               c=colors[::downsample], alpha=0.7, cmap=cmap)

        if by == 'cus' and \
                (view_3d_elev is not None or view_3d_azim is not None):
            # US: elev=90, azim=270. CU: elev=0, azim=0.
            ax.view_init(elev=view_3d_elev, azim=view_3d_azim)
        title = gene
        ax.set_title(f'{title}', fontsize=11)
        if by == 'us':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
        elif by == 'cu':
            ax.set_xlabel('unspliced' if full_name else 'u')
            ax.set_ylabel('chromatin' if full_name else 'c')
        elif by == 'cus':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
            ax.set_zlabel('chromatin' if full_name else 'c')
        if by in ['us', 'cu']:
            if not axis_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if not frame_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
        elif by == 'cus':
            if not axis_on:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.zaxis.set_ticklabels([])
            if not frame_on:
                ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.xaxis._axinfo['tick']['inward_factor'] = 0
                ax.xaxis._axinfo['tick']['outward_factor'] = 0
                ax.yaxis._axinfo['tick']['inward_factor'] = 0
                ax.yaxis._axinfo['tick']['outward_factor'] = 0
                ax.zaxis._axinfo['tick']['inward_factor'] = 0
                ax.zaxis._axinfo['tick']['outward_factor'] = 0
        count += 1
    for i in range(col+1, n_cols):
        fig.delaxes(axs[row, i])
    fig.tight_layout()


def scatter_twogenes_3d(adata,
                 x,
                 y,
                 z,
                 xkey='M_sc',
                 vkey='velocity_sc',
                 color_by='celltype',
                 n_cols=5,
                 axis_on=True,
                 frame_on=True,
                 velocity_arrows=False,
                 downsample=1,
                 figsize=None,
                 pointsize=2,
                 alpha=0.4,
                 cmap='coolwarm',
                 show_legend=False, 
                 view_3d_elev=None,
                 view_3d_azim=None
                 ):
    """Gene scatter plot.
    """
    if isinstance(x, str):
        x = [x]
    if isinstance(y, str):
        y = [y]
    if isinstance(z, str):
        z = [z]
    if xkey not in adata.layers.keys() or vkey not in adata.layers.keys():
        raise ValueError('Provided xkey or vkey not found in adata.layers!')
    if color_by == 'vz':
        cell_annot = None
    elif color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        cell_annot = None
        colors = adata.obs[color_by].values
        colors -= np.min(colors)
        colors /= np.max(colors)
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) \
            and color_by+'_colors' in adata.uns.keys():
        cell_annot = adata.obs[color_by].cat.categories
        if isinstance(adata.uns[f'{color_by}_colors'], dict):
            colors = list(adata.uns[f'{color_by}_colors'].values())
        else:
            colors = adata.uns[f'{color_by}_colors']
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata'
                         ' obs, and the colors of categories can be found in'
                         ' adata uns.')

    downsample = np.clip(int(downsample), 1, 10)
    genes = np.array(list(set(x + y + z)))
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        print(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(x)
    if gn < n_cols:
        n_cols = gn
    if color_by == 'vz':
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                figsize=(3*n_cols, 2.4*(-(-gn // n_cols)))
                                if figsize is None else figsize)
    else:
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                figsize=(3.8*n_cols, 2.7*(-(-gn // n_cols)))
                                if figsize is None else figsize,
                                subplot_kw={'projection': '3d'})
    point_size = 500.0 / np.sqrt(adata.shape[0]) if pointsize is None else 500.0 / np.sqrt(adata.shape[0]) * pointsize
    point_size = 4 * point_size

    fig.patch.set_facecolor('white')
    count = 0
    for a, b, c in zip(x, y, z):
        xa = adata[:, a].layers[xkey].copy()
        xb = adata[:, b].layers[xkey].copy()
        xc = adata[:, c].layers[xkey].copy()
        xa, xb, xc = np.ravel(xa), np.ravel(xb), np.ravel(xc)

        va = adata[:, a].layers[vkey].copy()
        vb = adata[:, b].layers[vkey].copy()
        vc = adata[:, c].layers[vkey].copy()
        max_a = np.max([np.max(xa), 1e-6])
        xa /= max_a
        va = np.ravel(va)
        va /= np.max([np.max(np.abs(va)), 1e-6])
        max_b = np.max([np.max(xb), 1e-6])
        xb /= max_b
        vb = np.ravel(vb)
        vb /= np.max([np.max(np.abs(vb)), 1e-6])
        max_c = np.max([np.max(xc), 1e-6])
        xc /= max_c
        vc = np.ravel(vc)
        vc /= np.max([np.max(np.abs(vc)), 1e-6])
        if color_by == 'vz':
            colors = vc

        cur_df = pd.DataFrame({"x": xa, "y": xb, "z": xc, 
                               "dx": va, "dy": vb, "dz": vc})

        row = count // n_cols
        col = count % n_cols
        ax = axs[row, col]
        if cell_annot is not None:
            for i in range(len(cell_annot)):
                filt = adata.obs[color_by] == cell_annot[i]
                filt = np.ravel(filt)
                if velocity_arrows:
                    ax.quiver(xa[filt][::downsample],
                              xb[filt][::downsample], 
                              xc[filt][::downsample],
                              va[filt][::downsample],
                              vb[filt][::downsample],
                              vc[filt][::downsample],
                              color=colors[i], alpha=alpha, length=0.1,
                              arrow_length_ratio=0.3, normalize=True)
                else:
                    ax.scatter(xa[filt][::downsample],
                               xb[filt][::downsample],
                               xc[filt][::downsample], s=pointsize,
                               c=colors[i], alpha=alpha)

        elif color_by == 'vz':
            vc_max = np.percentile(np.abs(vc).max(), 95)
            if velocity_arrows:
                ax.quiver(xa[::downsample],
                          xb[::downsample],
                          va[::downsample],
                          vb[::downsample],
                          color=colors[::downsample],
                          alpha=alpha,
                          scale_units='xy', scale=10, width=0.005,
                          headwidth=4, headaxislength=5.5, cmap=cmap)
            else:
                ax.scatter(xa[::downsample],
                           xb[::downsample],
                           s=pointsize,
                           c=colors,
                           alpha=alpha, cmap=cmap,
                           vmin=-vc_max, vmax=vc_max)
        else:
            if velocity_arrows:
                ax.quiver(xa[::downsample],
                            xb[::downsample], 
                            xc[::downsample],
                            va[::downsample],
                            vb[::downsample],
                            vc[::downsample],
                            color=colors[::downsample], alpha=alpha, length=0.1,
                            arrow_length_ratio=0.3, normalize=True,
                            cmap=cmap)
            else:
                ax.scatter(xa[::downsample],
                            xb[::downsample],
                            xc[::downsample], s=pointsize,
                            c=colors[::downsample], alpha=alpha, cmap=cmap)

        if show_legend:
            # plt.subplots_adjust(right=0.95)
            if color_by == 'vz' or (color_by in adata.obs and is_numeric_dtype(adata.obs[color_by])):
                # Create a scalar mappable for numeric data
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors)))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.1, aspect=20)
            elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]):
                # Create a color bar with discrete colors for categorical data
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(colors)
                bounds = np.arange(len(cell_annot)+1)
                norm = plt.Normalize(vmin=bounds.min(), vmax=bounds.max())
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array(bounds)
                cbar = fig.colorbar(sm, ticks=np.arange(len(cell_annot)), ax=ax, orientation='vertical', pad=0.1, aspect=20)
                cbar.set_ticklabels(cell_annot)

        if (view_3d_elev is not None or view_3d_azim is not None):
            # US: elev=90, azim=270. CU: elev=0, azim=0.
            ax.view_init(elev=view_3d_elev, azim=view_3d_azim)
        title = f'{a}_{b}_{c}'
        ax.set_title(title, fontsize=11)
        if color_by == 'vz':
            if not axis_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            else:
                ax.set_xlabel(a)
                ax.set_ylabel(b)
                ax.set_xticklabels(ax.get_yticklabels(), rotation=90, ha='right')
            if not frame_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
        else:
            if not axis_on:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.zaxis.set_ticklabels([])
            else:
                ax.set_xlabel(a)
                ax.set_ylabel(b)
                ax.set_zlabel(c)
                ax.set_xticklabels(ax.get_zticklabels(), rotation=90, ha='right')

            if not frame_on:
                ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.xaxis._axinfo['tick']['inward_factor'] = 0
                ax.xaxis._axinfo['tick']['outward_factor'] = 0
                ax.yaxis._axinfo['tick']['inward_factor'] = 0
                ax.yaxis._axinfo['tick']['outward_factor'] = 0
                ax.zaxis._axinfo['tick']['inward_factor'] = 0
                ax.zaxis._axinfo['tick']['outward_factor'] = 0
        count += 1
    for i in range(col+1, n_cols):
        fig.delaxes(axs[row, i])
    fig.tight_layout()


def plot_velocity_phase(
    adata,
    genes,
    s_layer: str = 'M_s',
    u_layer: str = 'M_u',
    vs_layer: str = 'velocity_S',
    vu_layer: str = 'velocity_U',
    smooth: bool = True,
    iteration: int = 5,
    beta: float = 0.1,
    color: str = 'celltype',
    downsample: float = None,
    scale: float = 1.0,
    alpha: float = 0.4,
    quiver_alpha: float = 0.3,
    quiver_color: str = 'black',
    show: bool = True,
    cmap='plasma', pointsize=1, figsize=None, ncols=3, dpi=100):
    if isinstance(genes, str):
        genes = [genes]

    if isinstance(genes, str):
        genes = [genes]
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        print(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        raise ValueError('genes not found in adata.var_names')
    if gn < ncols:
        ncols = gn

    cell_annot = None
    if color in adata.obs and is_numeric_dtype(adata.obs[color]):
        colors = adata.obs[color].values
    elif color in adata.obs and is_categorical_dtype(adata.obs[color])\
        and color+'_colors' in adata.uns.keys():
        cell_annot = adata.obs[color].cat.categories
        if isinstance(adata.uns[f'{color}_colors'], dict):
            colors = list(adata.uns[f'{color}_colors'].values())
        elif isinstance(adata.uns[f'{color}_colors'], list):
            colors = adata.uns[f'{color}_colors']
        elif isinstance(adata.uns[f'{color}_colors'], np.ndarray):
            colors = adata.uns[f'{color}_colors'].tolist()
        else:
            raise ValueError(f'Unsupported adata.uns[{color}_colors] object')
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata'
                         ' obs, and the colors of categories can be found in'
                         ' adata uns.')

    nrows = -(-gn // ncols)
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(6*ncols, 4*(-(-gn // ncols)))
                            if figsize is None else figsize,
                            tight_layout=True,
                            dpi=dpi)

    axs = np.reshape(axs, (nrows, ncols))
    logging.info("Plotting trends")

    cnt = 0
    for cnt, gene in tqdm(
        enumerate(genes),
        total=gn,
        desc="Plotting velocity in phase diagram",):
        spliced = flatten(adata[:, gene].layers[s_layer])
        unspliced = flatten(adata[:, gene].layers[u_layer])
        vs = flatten(adata[:, gene].layers[vs_layer])
        vu = flatten(adata[:, gene].layers[vu_layer])

        X_org = np.column_stack((spliced, unspliced)) 
        X = X_org.copy()
        V = np.column_stack((vs, vu))        


        if smooth:
            nbrs_idx = adata.uns['neighbors']['indices']
            prev_score = V
            cur_score = np.zeros(prev_score.shape)
            for _ in range(iteration):
                for i in range(len(prev_score)):
                    vi = prev_score[nbrs_idx[i]]
                    cur_score[i] = (beta * vi[0]) + ((1 - beta) * vi[1:].mean(axis=0))
                prev_score = cur_score
            V = cur_score

        n_total = X.shape[0]
        if downsample is not None:
            if 0 < downsample <= 1:
                target_num = int(n_total * downsample)
            else:
                target_num = int(downsample)
            selected_idx = uniform_downsample_cells(X, target_num)
            X = X[selected_idx]
            V = V[selected_idx]

        row = cnt // ncols
        col = cnt % ncols
        ax = axs[row, col]

        if cell_annot is not None:
            for j in range(len(cell_annot)):
                filt = adata.obs[color] == cell_annot[j]
                filt = np.ravel(filt)
                ax.scatter(X_org[filt, 0], X_org[filt, 1], c=colors[j], s=pointsize, alpha=alpha)
        else:
            ax.scatter(X_org[:, 0], X_org[:, 1], c=colors, s=pointsize, alpha=alpha, cmap=cmap, edgecolor='none')
        ax.quiver(
            X[:, 0], X[:, 1], 
            V[:, 0], V[:, 1],
            angles='xy', scale_units='xy', 
            scale=scale, color=quiver_color, alpha=quiver_alpha)
        
        ax.set_xlabel('Spliced')
        ax.set_ylabel('Unspliced')
        ax.set_title(f'{gene}')

    if show:
        plt.tight_layout()
        plt.show()
    else:
        return ax
    

def gene_score_histogram(
    adata,
    score_key: str,
    genes: Optional[List[str]] = None,
    bins: int = 100,
    quantile: Optional[float] = 0.95,
    extra_offset_fraction: float = 0.1,
    anno_min_diff_fraction: float = 0.05,
) -> plt.Figure:
    """
    Draw a histogram of gene scores with percentile line and annotations for specific genes.
    Adapted from Palantir (https://github.com/dpeerlab/Palantir/blob/master/src/palantir/plot.py#L1803)

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    score_key : str
        The key in `adata.var` data frame for the gene score.
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
    if score_key not in adata.var.columns:
        raise ValueError(f"Score key {score_key} not found in ad.var columns.")
    scores = adata.var[score_key]

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
    

def gene_trend(adata, 
               genes, 
               tkey:str, 
               layer:str='Ms', 
               n_x_grid: int = 100,
               max_iter: int = 2000,
               return_gam_result: bool = False,
               zero_indicator: bool = False,
               sharey: bool = False,
               hide_trend: bool = False,
               hide_cells: bool = False,
               hide_interval: bool = False,
               set_label: bool = True,
               same_plot: bool = False,
               color=None, cmap='plasma', pointsize=1, figsize=None, ncols=5, dpi=100, scatter_kwargs=None, **kwargs):
    """ Plot gene expression or velocity trends along trajectory using Generalized Addictive Model. """
    if tkey not in adata.obs.keys():
        raise ValueError(f'{tkey} not found in adata.obs')
    t = adata.obs[tkey]
    
    if same_plot:
        hide_cells = True
        logging.info('Setting `hide_cells` to True because of plotting trends in the same plot.')
    
    if isinstance(genes, str):
        genes = [genes]
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        print(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        raise ValueError('genes not found in adata.var_names')
    if gn < ncols:
        ncols = gn

    cell_annot = None
    if color in adata.obs and is_numeric_dtype(adata.obs[color]):
        colors = adata.obs[color].values
    elif color in adata.obs and is_categorical_dtype(adata.obs[color])\
        and color+'_colors' in adata.uns.keys():
        cell_annot = adata.obs[color].cat.categories
        if isinstance(adata.uns[f'{color}_colors'], dict):
            colors = list(adata.uns[f'{color}_colors'].values())
        elif isinstance(adata.uns[f'{color}_colors'], list):
            colors = adata.uns[f'{color}_colors']
        elif isinstance(adata.uns[f'{color}_colors'], np.ndarray):
            colors = adata.uns[f'{color}_colors'].tolist()
        else:
            raise ValueError(f'Unsupported adata.uns[{color}_colors] object')
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata'
                         ' obs, and the colors of categories can be found in'
                         ' adata uns.')

    nrows = -(-gn // ncols)
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(6*ncols, 4*(-(-gn // ncols)))
                            if figsize is None else figsize,
                            sharex=True, 
                            sharey=sharey,
                            tight_layout=True,
                            dpi=dpi)

    fig.patch.set_facecolor('white')
    axs = np.reshape(axs, (nrows, ncols))
    logging.info("Plotting trends")

    gam_kwargs = {
        'n_splines': 6,
        'spline_order': 3
    }
    gam_kwargs.update(kwargs)
    gam_results = np.zeros((len(genes), n_x_grid))

    cnt = 0
    for i, gene in tqdm(
        enumerate(genes),
        total=gn,
        desc="Fitting trends using GAM",):
        x = adata[:, gene].layers[layer]
        x = x.A.flatten() if sp.issparse(x) else x.flatten()

        ### GAM fitting
        term  =s(
            0,
            **gam_kwargs)
        gam = LinearGAM(term, max_iter=max_iter, verbose=False).fit(t, x)
        tx = gam.generate_X_grid(term=0, n=n_x_grid)
        row = i // ncols
        col = i % ncols
        ax = axs[row, col] if not same_plot else axs

        if not hide_trend:
            ax.plot(tx[:, 0], gam.predict(tx))
            if not hide_interval:
                ci = gam.confidence_intervals(tx, width=0.95)
                lower_bound, upper_bound = ci[:, 0], ci[:, 1]
                ax.fill_between(tx[:, 0], lower_bound, upper_bound, color='#cabad7', alpha=0.5)
                ax.plot(tx[:, 0], gam.confidence_intervals(tx, width=0.95), c='#cabad7', ls='--')
        if not hide_cells:
            if cell_annot is not None:
                for j in range(len(cell_annot)):
                    filt = adata.obs[color] == cell_annot[j]
                    filt = np.ravel(filt)
                    ax.scatter(t[filt], x[filt], c=colors[j], s=pointsize, alpha=0.5)
            else:
                ax.scatter(t, x, c=colors, s=pointsize, alpha=0.5, cmap=cmap)
        if set_label:
            ax.set_ylabel(gene)

        if zero_indicator:
            ax.axhline(y=0, color='red', linestyle='--')

        if return_gam_result:
            gam_results[i] = gam.predict(tx)
        
        cnt += 1

    if set_label:
        fig.text(0.5, 0.01, tkey, ha='center') 
    for i in range(col+1, ncols):
        fig.delaxes(axs[row, i])
    fig.tight_layout()
    
    if return_gam_result:
        return gam_results
    else:
        return


def response(
    adata,
    pairs_mat: np.ndarray,
    xkey: Optional[str] = 'M_sc',
    ykey: Optional[str] = 'jacobian',
    hide_cells: bool = True,
    annot_key: str = 'celltype',
    downsampling: int = 3,
    log: bool = True,
    drop_zero_cells: bool = True,
    cell_idx: Optional[pd.Index] = None,
    perc: Optional[tuple] = None,
    grid_num: int = 25,
    kde_backend: Literal['fixbdw', 'scipy', 'statsmodels'] = 'statsmodels',
    integral_rule: Literal['trapz', 'simps'] = 'trapz',
    plot_integration_uncertainty: bool = False,
    n_row: int = 1,
    n_col: Optional[int] = None,
    cmap: Union[str, Colormap, None] = None,
    curve_style: str = "c-",
    zero_indicator: bool = True,
    hide_mean: bool = False,
    hide_trend: bool = False,
    figsize: Tuple[float, float] = (6, 4),
    show: bool = True,
    return_integration: bool = False,
    **kwargs,
    ):
    """ Fiting response curve using GAM """
    from scipy import integrate
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    try:
        from dynamo.vectorfield.utils import get_jacobian
    except ImportError:
        raise ImportError(
            "If you want to show jacobian analysis in plotting function, you need to install `dynamo` "
            "package via `pip install dynamo-release` see more details at https://dynamo-release.readthedocs.io/en/latest/,")
    
    if not set([xkey, ykey]) <= set(adata.layers.keys()).union(set(["jacobian"])):
        raise ValueError(
            f"adata.layers doesn't have {xkey, ykey} layers. Please specify the correct layers or "
            "perform relevant preprocessing and vector field analyses first."
        )
    
    if integral_rule == 'trapz':
        rule = integrate.cumulative_trapezoid
    elif integral_rule == 'simps': 
        rule = integrate.simpson
    else:
        raise ValueError("Integral rule only support `trapz` and `simps` implemented by scipy.") 

    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "response", ["#000000", "#000000", "#000000", "#800080", "#FF0000", "#FFFF00"]
        )
    inset_dict = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "50%",  # height : 50%
        "loc": "lower left",
        "bbox_to_anchor": (1.0125, 0.0, 1, 1),
        "borderpad": 0,
    }

    if not hide_cells:
        if annot_key in adata.obs and is_categorical_dtype(adata.obs[annot_key]) \
                and annot_key+'_colors' in adata.uns.keys():
            cell_annot = adata.obs[annot_key].cat.categories
            if isinstance(adata.uns[f'{annot_key}_colors'], dict):
                colors = list(adata.uns[f'{annot_key}_colors'].values())
            else:
                colors = adata.uns[f'{annot_key}_colors']

    all_genes_in_pair = np.unique(pairs_mat)
    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise ValueError(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )

    flat_res = pd.DataFrame(columns=["x", "y", "den", "type"])
    xy = pd.DataFrame()
    # extract information from dynamo output in adata object
    id = 0
    for _, gene_pairs in enumerate(pairs_mat):
        f_ini_ind = (grid_num**2) * id

        gene_pair_name = gene_pairs[0] + "->" + gene_pairs[1]

        if xkey.startswith("jacobian"):
            J_df = get_jacobian(
                adata,
                gene_pairs[0],
                gene_pairs[1],
            )
            jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"
            x = flatten(J_df[jkey])
        else:
            x = flatten(adata[:, gene_pairs[0]].layers[xkey])

        if ykey.startswith("jacobian"):
            J_df = get_jacobian(
                adata,
                gene_pairs[0],
                gene_pairs[1],
            )
            jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"
            y_ori = flatten(J_df[jkey])
        else:
            y_ori = flatten(adata[:, gene_pairs[1]].layers[ykey])

        if drop_zero_cells:
            finite = np.isfinite(x + y_ori)
            nonzero = np.abs(x) + np.abs(y_ori) > 0
            valid_ids = np.logical_and(finite, nonzero)
        else:
            valid_ids = np.isfinite(x + y_ori)

        if cell_idx is not None:
            # subset cells for cell type-specific visualization
            subset_idx = np.zeros(adata.n_obs)
            idx = adata.obs_names.get_indexer(cell_idx)
            subset_idx[idx] = True
            valid_ids = np.logical_and(valid_ids, subset_idx)
        
        if perc is not None:
            # filter out outliers
            lb = np.percentile(x, perc[0])
            ub = np.percentile(x, perc[1])
            valid_ids = np.logical_and(valid_ids, np.logical_and(x>lb, x<ub))

        x, y_ori = x[valid_ids], y_ori[valid_ids]

        if log:
            x, y_ori = x if sum(x < 0) else np.log(np.array(x) + 1), y_ori if sum(y_ori) < 0 else np.log(
                np.array(y_ori) + 1)

        y = y_ori

        # den_res[0, 0] is at the lower bottom; dens[1, 4]: is the 2nd on x-axis and 5th on y-axis
        x_meshgrid, y_meshgrid, den_res = kde2d(
            x, y, n=[grid_num, grid_num], lims=[min(x), max(x), min(y), max(y)], 
            backend=kde_backend
        )
        den_res = np.array(den_res)

        den_x = np.sum(den_res, axis=1)  # condition on each input x, sum over y

        for i in range(len(x_meshgrid)):
            tmp = den_res[i] / den_x[i]  # condition on each input x, normalize over y
            tmp = den_res[i]
            max_val = max(tmp)
            min_val = min(tmp)

            rescaled_val = (tmp - min_val) / (max_val - min_val)
            res_row = pd.DataFrame(
                {
                    "x": x_meshgrid[i],
                    "y": y_meshgrid,
                    "den": rescaled_val,
                    "type": gene_pair_name,
                },
                index=[i * len(x_meshgrid) + np.arange(len(y_meshgrid)) + f_ini_ind],
            )

            flat_res = pd.concat([flat_res, res_row])

        cur_data = pd.DataFrame({"x": x, "y": y, "type": gene_pair_name})
        xy = pd.concat([xy, cur_data], axis=0)
        id = id + 1

    gene_pairs_num = len(flat_res.type.unique())

    # plot jacobian results and fitting curve
    n_col = -(-gene_pairs_num // n_row) if n_col is None else n_col

    if n_row * n_col < gene_pairs_num:
        raise ValueError("The number of row or column specified is less than the gene pairs")
    figsize = (figsize[0] * n_col, figsize[1] * n_row) if figsize is not None else (4 * n_col, 4 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize, sharex=False, sharey=False, squeeze=False)

    def scale_func(x, X, grid_num):
        return grid_num * (x - np.min(X)) / (np.max(X) - np.min(X))

    fit_integration = pd.DataFrame(columns=["x", "y", "mean", "ci_lower", "ci_upper", "regulators", "effectors"])

    for x, flat_res_type in enumerate(flat_res.type.unique()):
        gene_pairs = flat_res_type.split("->")

        flat_res_subset = flat_res[flat_res["type"] == flat_res_type]
        xy_subset = xy[xy["type"] == flat_res_type]

        x_val, y_val = flat_res_subset["x"], flat_res_subset["y"]

        i, j = x % n_row, x // n_row

        values = flat_res_subset["den"].values.reshape(grid_num, grid_num).T

        axins = inset_axes(axes[i, j], bbox_transform=axes[i, j].transAxes, **inset_dict)

        ext_lim = (min(x_val), max(x_val), min(y_val), max(y_val))
        im = axes[i, j].imshow(
            values,
            interpolation="mitchell",
            origin="lower",
            cmap=cmap)
        
        cb = fig.colorbar(im, cax=axins)
        cb.set_alpha(1)
        cb.draw_all()
        cb.locator = MaxNLocator(nbins=3, integer=False)
        cb.update_ticks()

        # closest_x_ind = np.array([np.searchsorted(x_meshgrid, i) for i in xy_subset["x"].values])
        # closest_y_ind = np.array([np.searchsorted(y_meshgrid, i) for i in xy_subset["y"].values])
        # valid_ids = np.logical_and(closest_x_ind < grid_num, closest_y_ind < grid_num)
        # axes[i, j].scatter(closest_x_ind[valid_ids], closest_y_ind[valid_ids], color="gray", alpha=0.1, s=1)

        if xkey.startswith("jacobian"):
            axes[i, j].set_xlabel(r"$\partial f_{%s} / {\partial x_{%s}$" % (gene_pairs[1], gene_pairs[0]))
        else:
            axes[i, j].set_xlabel(gene_pairs[0] + rf" (${xkey}$)")
        if ykey.startswith("jacobian"):
            axes[i, j].set_ylabel(r"$\partial f_{%s} / \partial x_{%s}$" % (gene_pairs[1], gene_pairs[0]))
            axes[i, j].title.set_text(r"$\rho(\partial f_{%s} / \partial x_{%s})$" % (gene_pairs[1], gene_pairs[0]))
        else:
            axes[i, j].set_ylabel(gene_pairs[1] + rf" (${ykey}$)")
            axes[i, j].title.set_text(rf"$\rho_{{{gene_pairs[1]}}}$ (${ykey}$)")

        xlabels = list(np.linspace(ext_lim[0], ext_lim[1], 5))
        ylabels = list(np.linspace(ext_lim[2], ext_lim[3], 5))

        # zero indicator
        if zero_indicator:
            axes[i, j].plot(
                scale_func([np.min(xlabels), np.max(xlabels)], xlabels, grid_num),
                scale_func(np.zeros(2), ylabels, grid_num),
                'w--',
                linewidth=2.0)

        # curve fiting using pygam
        gam_kwargs = {
            'n_splines': 6,
            'spline_order': 3
        }
        gam_kwargs.update(kwargs)
        if ykey.startswith("jacobian"):
            logging.info("Fitting response curve using GAM...")
            x_grid, y_mean, y_sigm = kde2d_to_mean_and_sigma(
                np.array(x_val), np.array(y_val), flat_res_subset["den"].values)
            y_mean[np.isnan(y_mean)] = 0
            term  =s(
                0,
                **gam_kwargs)
            # w = (1/y_sigm - (1/y_sigm).min()) / ((1/y_sigm).max()-(1/y_sigm).min())
            gam = LinearGAM(term, max_iter=1000, verbose=False).fit(x_grid, y_mean)
            if not hide_mean:
                axes[i, j].plot(
                    scale_func(x_grid, xlabels, grid_num), 
                    scale_func(y_mean, ylabels, grid_num), 
                    "c*"
                )
            if not hide_trend:
                axes[i, j].plot(
                    scale_func(x_grid, xlabels, grid_num), 
                    scale_func(gam.predict(x_grid), ylabels, grid_num), 
                    curve_style
                )

            # Integrate the fitted curve using the integration rule
            int_grid = np.linspace(x_grid.min(), x_grid.max(), 100)
            prediction = gam.predict(int_grid)
            confidence_intervals = gam.confidence_intervals(int_grid, width=0.95)
            integral = rule(prediction, int_grid, initial=0)
            mean_integral = integral
            ci_lower = np.zeros_like(integral)
            ci_upper = np.zeros_like(integral)

            # Perform Monte Carlo simulations to calculate confident interval
            if plot_integration_uncertainty: 
                logging.info('Simulation integral interval using Monte Carlo method...')
                n_simulations = 1000
                simulated_integrals = []
                for _ in range(n_simulations):
                    # Simulate the fitted values within their confidence intervals
                    simulated_values = np.random.uniform(confidence_intervals[:, 0], confidence_intervals[:, 1])
                    # Integrate the simulated values
                    simulated_integral = rule(simulated_values, int_grid, initial=0)
                    simulated_integrals.append(simulated_integral)
                simulated_integrals = np.array(simulated_integrals)
                mean_integral = np.mean(simulated_integrals, axis=0)
                ci_lower = np.percentile(simulated_integrals, 2.5, axis=0)
                ci_upper = np.percentile(simulated_integrals, 97.5, axis=0)

            tmp_integration = pd.DataFrame({
                "x": int_grid, "y": integral, "mean": mean_integral, "ci_lower": ci_lower, "ci_upper": ci_upper, 
                "regulators": gene_pairs[0], "effectors": gene_pairs[1]})
            fit_integration = pd.concat([fit_integration, tmp_integration])

        if not hide_cells:
            y_for_cells = y_meshgrid[-2]
            for type_i in range(len(cell_annot)):
                filt = adata[valid_ids].obs[annot_key] == cell_annot[type_i]
                filt = np.ravel(filt)
                tmp_x = xy_subset["x"].values[filt] # NOTE: xy already subset by valid_ids
                tmp_y = np.full_like(tmp_x, y_for_cells)
                axes[i, j].scatter(scale_func(tmp_x[::downsampling], xlabels, grid_num), tmp_y[::downsampling], s=12, color=colors[type_i], alpha=0.8)

        # set the x/y ticks
        inds = np.linspace(0, grid_num - 1, 5, endpoint=True)
        axes[i, j].set_xticks(inds)
        axes[i, j].set_yticks(inds)

        if ext_lim[1] < 1e-3:
            xlabels = ["{:.3e}".format(i) for i in xlabels]
        else:
            xlabels = [np.round(i, 2) for i in xlabels]
        if ext_lim[3] < 1e-3:
            ylabels = ["{:.3e}".format(i) for i in ylabels]
        else:
            ylabels = [np.round(i, 2) for i in ylabels]

        if ext_lim[1] < 1e-3:
            axes[i, j].set_xticklabels(xlabels, rotation=30, ha="right")
        else:
            axes[i, j].set_xticklabels(xlabels)

        axes[i, j].set_yticklabels(ylabels)

    plt.subplots_adjust(left=0.1, right=1, top=0.80, bottom=0.1, wspace=0.1)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return axes
    
    if return_integration:
        return fit_integration