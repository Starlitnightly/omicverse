from ..utils._scatterplot import _embedding
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

import matplotlib.patches as mpatches
from scipy.stats import kruskal
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
import matplotlib.pyplot as plt
import scanpy as sc

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

    return _embedding(adata=adata, basis=basis, color=color, 
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



def cellproportion(adata:AnnData,celltype_clusters:str,groupby:str,
                       groupby_li=None,figsize:tuple=(4,6),
                       ticks_fontsize:int=12,labels_fontsize:int=12,ax=None,
                       legend:bool=False):
    """
    Plot cell proportion of each cell type in each visual cluster.

    Arguments:
        adata: AnnData object.
        celltype_clusters: Cell type clusters.
        groupby: Visual clusters.
        groupby_li: Visual cluster list.
        figsize: Figure size.
        ticks_fontsize: Ticks fontsize.
        labels_fontsize: Labels fontsize.
        legend: Whether to show legend.
    
    
    """

    b=pd.DataFrame(columns=['cell_type','value','Week'])
    visual_clusters=groupby
    visual_li=groupby_li
    if visual_li==None:
        adata.obs[visual_clusters]=adata.obs[visual_clusters].astype('category')
        visual_li=adata.obs[visual_clusters].cat.categories
    
    for i in visual_li:
        b1=pd.DataFrame()
        test=adata.obs.loc[adata.obs[visual_clusters]==i,celltype_clusters].value_counts()
        b1['cell_type']=test.index
        b1['value']=test.values/test.sum()
        b1['Week']=i.replace('Retinoblastoma_','')
        b=pd.concat([b,b1])
    
    plt_data2=adata.obs[celltype_clusters].value_counts()
    plot_data2_color_dict=dict(zip(adata.obs[celltype_clusters].cat.categories,adata.uns['{}_colors'.format(celltype_clusters)]))
    plt_data3=adata.obs[visual_clusters].value_counts()
    plot_data3_color_dict=dict(zip([i.replace('Retinoblastoma_','') for i in adata.obs[visual_clusters].cat.categories],adata.uns['{}_colors'.format(visual_clusters)]))
    b['cell_type_color'] = b['cell_type'].map(plot_data2_color_dict)
    b['stage_color']=b['Week'].map(plot_data3_color_dict)
    
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
    #用ax控制图片
    #sns.set_theme(style="whitegrid")
    #sns.set_theme(style="ticks")
    n=0
    all_celltype=adata.obs[celltype_clusters].cat.categories
    for i in all_celltype:
        if n==0:
            test1=b[b['cell_type']==i]
            ax.bar(x=test1['Week'],height=test1['value'],width=0.8,color=list(set(test1['cell_type_color']))[0], label=i)
            bottoms=test1['value'].values
        else:
            test2=b[b['cell_type']==i]
            ax.bar(x=test2['Week'],height=test2['value'],bottom=bottoms,width=0.8,color=list(set(test2['cell_type_color']))[0], label=i)
            test1=test2
            bottoms+=test1['value'].values
        n+=1
    if legend!=False:
        plt.legend(bbox_to_anchor=(1.05, -0.05), loc=3, borderaxespad=0,fontsize=10)
    
    plt.grid(False)
    
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 设置左边和下边的坐标刻度为透明色
    #ax.yaxis.tick_left()
    #ax.xaxis.tick_bottom()
    #ax.xaxis.set_tick_params(color='none')
    #ax.yaxis.set_tick_params(color='none')

    # 设置左边和下边的坐标轴线为独立的线段
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    plt.xticks(fontsize=ticks_fontsize,rotation=90)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel(groupby,fontsize=labels_fontsize)
    plt.ylabel('Cells per Stage',fontsize=labels_fontsize)
    #fig.tight_layout()
    if ax==None:
        return fig,ax
    


def embedding_celltype(adata:AnnData,figsize:tuple=(6,4),basis:str='umap',
                            celltype_key:str='major_celltype',title:str=None,
                            celltype_range:tuple=(2,9),
                            embedding_range:tuple=(3,10),
                            xlim:int=-1000)->tuple:
    """
    Plot embedding with celltype color by omicverse

    Arguments:
        adata: AnnData object  
        figsize: figure size
        basis: embedding method
        celltype_key: celltype key in adata.obs
        title: figure title
        celltype_range: celltype range to plot
        embedding_range: embedding range to plot
        xlim: x axis limit

    Returns:
        fig : figure and axis
        ax: axis
    
    """

    adata.obs[celltype_key]=adata.obs[celltype_key].astype('category')
    cell_num_pd=pd.DataFrame(adata.obs[celltype_key].value_counts())
    if '{}_colors'.format(celltype_key) in adata.uns.keys():
        cell_color_dict=dict(zip(adata.obs[celltype_key].cat.categories.tolist(),
                        adata.uns['{}_colors'.format(celltype_key)]))
    else:
        if len(adata.obs[celltype_key].cat.categories)>28:
            cell_color_dict=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
        else:
            cell_color_dict=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))

    if figsize==None:
        if len(adata.obs[celltype_key].cat.categories)<10:
            fig = plt.figure(figsize=(6,4))
        else:
            print('The number of cell types is too large, please set the figsize parameter')
            return
    else:
        fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(10, 10)
    ax1 = fig.add_subplot(grid[:, embedding_range[0]:embedding_range[1]])       # 占据第一行的所有列
    ax2 = fig.add_subplot(grid[celltype_range[0]:celltype_range[1], :2]) 
    # 定义子图的大小和位置
         # 占据第二行的前两列
    #ax3 = fig.add_subplot(grid[1:, 2])      # 占据第二行及以后的最后一列
    #ax4 = fig.add_subplot(grid[2, 0])       # 占据最后一行的第一列
    #ax5 = fig.add_subplot(grid[2, 1])       # 占据最后一行的第二列

    sc.pl.embedding(
        adata,
        basis=basis,
        color=[celltype_key],
        title='',
        frameon=False,
        #wspace=0.65,
        ncols=3,
        ax=ax1,
        legend_loc=False,
        show=False
    )

    for idx,cell in zip(range(cell_num_pd.shape[0]),
                        adata.obs[celltype_key].cat.categories):
        ax2.scatter(100,
                cell,c=cell_color_dict[cell],s=50)
        ax2.plot((100,cell_num_pd.loc[cell,celltype_key]),(idx,idx),
                c=cell_color_dict[cell],lw=4)
        ax2.text(100,idx+0.2,
                cell+'('+str("{:,}".format(cell_num_pd.loc[cell,celltype_key]))+')',fontsize=11)
    ax2.set_xlim(xlim,cell_num_pd.iloc[1].values[0]) 
    ax2.text(xlim,idx+1,title,fontsize=12)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.axis('off')

    return fig,[ax1,ax2]


def ConvexHull(adata:AnnData,basis:str,cluster_key:str,
                    hull_cluster:str,ax,color=None,alpha:float=0.2):
    """
    Plot the ConvexHull for a cluster in embedding

    Arguments:
        adata: AnnData object
        basis: embedding method in adata.obsm
        cluster_key: cluster key in adata.obs
        hull_cluster: cluster to plot for ConvexHull
        ax: axes
        color: color for ConvexHull
        alpha: alpha for ConvexHull

    Returns:
        ax: axes
    
    """
    from scipy.spatial import ConvexHull
    adata.obs[cluster_key]=adata.obs[cluster_key].astype('category')
    if '{}_colors'.format(cluster_key) in adata.uns.keys():
        print('{}_colors'.format(cluster_key))
        type_color_all=dict(zip(adata.obs[cluster_key].cat.categories,adata.uns['{}_colors'.format(cluster_key)]))
    else:
        if len(adata.obs[cluster_key].cat.categories)>28:
            type_color_all=dict(zip(adata.obs[cluster_key].cat.categories,sc.pl.palettes.default_102))
        else:
            type_color_all=dict(zip(adata.obs[cluster_key].cat.categories,sc.pl.palettes.zeileis_28))
    
    #color_dict=dict(zip(adata.obs[cluster_key].cat.categories,adata.uns[f'{cluster_key}_colors']))
    points=adata[adata.obs[cluster_key]==hull_cluster].obsm[basis]
    hull = ConvexHull(points)
    vert = np.append(hull.vertices, hull.vertices[0])  # close the polygon by appending the first point at the end
    if color==None:
        ax.plot(points[vert, 0], points[vert, 1], '--', c=type_color_all[hull_cluster])
        ax.fill(points[vert, 0], points[vert, 1], c=type_color_all[hull_cluster], alpha=alpha)
    else:
        ax.plot(points[vert, 0], points[vert, 1], '--', c=color)
        ax.fill(points[vert, 0], points[vert, 1], c=color, alpha=alpha)
    return ax


def embedding_adjust(
    adata, groupby, exclude=(), 
    basis='X_umap',ax=None, adjust_kwargs=None, text_kwargs=None
):
    """ 
    Get locations of cluster median . Borrowed from scanpy github forum.
    """
    if adjust_kwargs is None:
        adjust_kwargs = {"text_from_points": False}
    if text_kwargs is None:
        text_kwargs = {}

    medians = {}

    for g, g_idx in adata.obs.groupby(groupby).groups.items():
        if g in exclude:
            continue
        medians[g] = np.median(adata[g_idx].obsm[basis], axis=0)

    if ax is None:
        texts = [
            plt.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()
        ]
    else:
        texts = [ax.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()]
    from adjustText import adjust_text
    adjust_text(texts, **adjust_kwargs)


def embedding_density(adata,basis,groupby,target_clusters,**kwargs):
    if 'X_' in basis:
        basis1=basis.split('_')[1]
    sc.tl.embedding_density(adata,
                       basis=basis1,
                       groupby=groupby,
                       key_added='temp_density')
    adata.obs.loc[adata.obs[groupby]!=target_clusters,'temp_density']=0
    return embedding(adata,
                  basis=basis,
                  color=['temp_density'],
                    title=target_clusters,
                   **kwargs
                 )

def bardotplot(adata,groupby,color,figsize=(8,3),return_values=False,
               fontsize=12,xlabel='',ylabel='',xticks_rotation=90,ax=None,
               bar_kwargs=None,scatter_kwargs=None):
    
    if bar_kwargs is None:
        bar_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    
    var_ticks=False
    obs_ticks=False
    plot_text_=color
    if (plot_text_ in adata.var_names):
        adata1=adata
        var_ticks=True
    elif plot_text_ in adata.obs.columns:
        adata1=adata
        obs_ticks=True
    elif (adata.raw!=None) and (plot_text_ in adata.raw.var_names):
        adata1=adata1.raw.to_adata()
        var_ticks=True
    else:
        print(f'Please check the `{color}` key in adata.obs or adata.var')
        return
    adata1.obs[groupby]=adata1.obs[groupby].astype('category')
    
    if var_ticks==True:
        plot_data=pd.DataFrame()
        max_len=0
        for group in adata1.obs[groupby].cat.categories:
            if max_len<len(adata1[adata1.obs[groupby]==group,plot_text_].to_df().values.reshape(-1)):
                max_len=len(adata1[adata1.obs[groupby]==group,plot_text_].to_df().values.reshape(-1))
        for group in adata1.obs[groupby].cat.categories:
            t_data1=list(adata1[adata1.obs[groupby]==group,plot_text_].to_df().values.reshape(-1))
            while len(t_data1)<max_len:
                t_data1.append(np.nan)
            plot_data[group]=t_data1
    elif obs_ticks==True:
        plot_data=pd.DataFrame()
        max_len=0
        for group in adata1.obs[groupby].cat.categories:
            if max_len<len(adata1.obs.loc[adata1.obs[groupby]==group,plot_text_].values.reshape(-1)):
                max_len=len(adata1.obs.loc[adata1.obs[groupby]==group,plot_text_].values.reshape(-1))
        for group in adata1.obs[groupby].cat.categories:
            t_data1=list(adata1.obs.loc[adata1.obs[groupby]==group,plot_text_].values.reshape(-1))
            while len(t_data1)<max_len:
                t_data1.append(np.nan)
            plot_data[group]=t_data1
    
    if return_values==True:
        return plot_data
    
    
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
        
    xbar = np.arange(len(plot_data.columns.to_numpy()))
    #color_list_dot=[ov.utils.green_color[0],ov.utils.green_color[1],ov.utils.red_color[0],'#EC9DC5','#5BC23D']
    #color_list_dot=adata.uns['clusters_colors']
    if '{}_colors'.format(groupby) in adata.uns.keys():
        color_list_dot=adata.uns['{}_colors'.format(groupby)]
    else:
        if len(adata.obs[groupby].cat.categories)>28:
            color_list_dot=sc.pl.palettes.default_102
        else:
            color_list_dot=sc.pl.palettes.zeileis_28
            
    plt.bar(x=plot_data.columns, 
            height=plot_data.describe().loc['mean'], 
            yerr=plot_data.sem(), 
            color=color_list_dot, 
            zorder=1, #fill=False,
            edgecolor=color_list_dot,
            error_kw={'elinewidth': None, 'capthick': None},**bar_kwargs)
    bw=0.4
    for cols in range(len(plot_data.columns.to_numpy())):
        # get markers from here https://matplotlib.org/3.1.1/api/markers_api.html
        plt.scatter(x=np.linspace(xbar[cols]-bw/2, xbar[cols]+bw/2, int(plot_data.describe().loc['count'][cols])),
                   y=plot_data[plot_data.columns[cols]].dropna(), 
                    color=color_list_dot[cols], zorder=1, 
                   **scatter_kwargs)


    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    plt.xticks(rotation=xticks_rotation,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize+1)
    plt.ylabel(ylabel,fontsize=fontsize+1)
    plt.title(plot_text_,fontsize=fontsize+1)
    if ax==None:
        return fig,ax
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kruskal

def single_group_boxplot(adata,
                         groupby: str = '',
                         color: str = '',
                         type_color_dict: dict = None,
                         title: str = '',
                         ylabel: str = '',
                         kruskal_test: bool = False,
                         figsize: tuple = (4, 4),
                         x_ticks_plot: bool = False,
                         legend_plot: bool = True,
                         bbox_to_anchor: tuple = (1, 0.55),
                         save: bool = False,
                         point_number: int = 5,
                         save_pathway: str = '',
                         sort: bool = True,
                         scatter_kwargs: dict = None,
                         ax = None,
                         fontsize = 12,
                        ):
    """
    adata (AnnData object): The data object containing the information for plotting.
    groupby (str): The variable used for grouping the data.
    color (str): The variable used for coloring the data points.
    type_color_dict (dict): A dictionary mapping group categories to specific colors.
    title (str): The title for the plot.
    ylabel (str): The label for the y-axis.
    kruskal_test (bool): Whether to perform a Kruskal-Wallis test and display the p-value on the plot.
    figsize (tuple): The size of the plot figure in inches (width, height).
    x_ticks_plot (bool): Whether to display x-axis tick labels.
    legend_plot (bool): Whether to display a legend for the groups.
    bbox_to_anchor (tuple): The position of the legend bbox (x, y) in axes coordinates.
    save (bool): Whether to save the plot to a file.
    point_number (int): The number of data points to be plotted for each group.
    save_pathway (str): The file path for saving the plot (if save is True).
    sort (bool): Whether to sort the groups based on their mean values.
    scatter_kwargs (dict): Additional keyword arguments for customizing the scatter plot.
    ax (matplotlib.axes.Axes): A pre-existing axes object for plotting (optional).
    
    Example:
    ov.pl.single_group_boxplot(adata,groupby='clusters',
             color='Sox_aucell',
             type_color_dict=dict(zip(pd.Categorical(adata.obs['clusters']).categories, adata.uns['clusters_colors'])),
             x_ticks_plot=True,
             figsize=(5,4),
             kruskal_test=True,
             ylabel='Sox_aucell',
             legend_plot=False,
             bbox_to_anchor=(1,1),
             title='Expression',
             scatter_kwargs={'alpha':0.8,'s':10,'marker':'o'},
             point_number=15,
             sort=False,
             save=False,
             )
    """

    if scatter_kwargs is None:
        scatter_kwargs = {}

    # Create an empty dictionary to store results
    plot_data = {}

    var_ticks = False
    obs_ticks = False
    plot_text_ = color
    if (plot_text_ in adata.var_names):
        adata1 = adata
        var_ticks = True
    elif plot_text_ in adata.obs.columns:
        adata1 = adata
        obs_ticks = True
    elif (adata.raw is not None) and (plot_text_ in adata.raw.var_names):
        adata1 = adata1.raw.to_adata()
        var_ticks = True
    else:
        print(f'Please check the `{color}` key in adata.obs or adata.var')
        return
    adata1.obs[groupby] = adata1.obs[groupby].astype('category')

    if var_ticks == True:
        adata1.obs[color] = adata1[:, plot_text_].to_df().values

    # Categorize by groups
    for group in set(adata1.obs[groupby]):
        plot_data[group] = np.array(adata1.obs.loc[adata1.obs[groupby] == group, color].tolist())

    if sort == True:
        sorted_keys = sorted(plot_data.keys(), key=lambda k: np.mean(plot_data[k]))
        sorted_plot_data = {key: plot_data[key] for key in sorted_keys}
        plot_data = sorted_plot_data

        sorted_colors = [type_color_dict[key] for key in sorted_keys]
        sc_color = sorted_colors
    else:
        sc_color = [type_color_dict[key] for key in plot_data.keys()]

    shake_dict = {}

    for group in set(adata1.obs[groupby]):
        data_list = []
        gene_data = adata1.obs.loc[adata1.obs[groupby] == group, color].tolist()
        if len(gene_data) > point_number:
            bootstrap_data = np.random.choice(gene_data, size=point_number, replace=False)
        else:
            bootstrap_data = gene_data
        shake_dict[group] = np.array(bootstrap_data)

    # Set figure size
    fig, ax = plt.subplots(figsize=figsize)

    # Plot boxplots
    width = 0.8
    ticks = np.arange(len(plot_data))
    positions = np.arange(len(ticks))

    for num, (hue_data, hue_color) in enumerate(zip(plot_data.keys(), sc_color)):
        position = positions[num]
        b1 = ax.boxplot(plot_data[hue_data],
                        positions=[position],
                        sym='',
                        widths=width,
                        patch_artist=True
                        )
        plt.scatter(np.random.normal(position, 0.12, point_number),
                    shake_dict[hue_data],
                    c=hue_color, zorder=1, **scatter_kwargs)
        box = b1['boxes'][0]
        light_hue_color = tuple((min(1, c + 0.5 * (1 - c))) for c in plt.cm.colors.to_rgb(hue_color))
        box.set(facecolor=light_hue_color, edgecolor=hue_color, linewidth=2)
        plt.setp(b1['whiskers'], color=hue_color, linewidth=2)
        plt.setp(b1['caps'], color=hue_color, linewidth=2)
        plt.setp(b1['medians'], color=hue_color, linewidth=3)

    # Axis labels and title
    

    if x_ticks_plot == True:
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_data.keys(), rotation=90, fontsize=fontsize)
    else:
        ax.set_xticklabels([])

    yticks = ax.get_yticks()
    ax.set_title(title, fontsize=fontsize+1,)
    plt.ylabel(ylabel, fontsize=fontsize+1, )
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    if legend_plot == True:
        labels = list(plot_data.keys())
        patches = [mpatches.Patch(color=sc_color[i], label="{:s}".format(labels[i])) for i in range(len(plot_data))]
        ax.legend(handles=patches, bbox_to_anchor=bbox_to_anchor, ncol=1, fontsize=fontsize)

    if kruskal_test == True:
        data_list = [plot_data[key] for key in plot_data]
        statistic, p_value = kruskal(*data_list)

        if p_value < 0.0001:
            formatted_p_value = "{:.2e}".format(p_value)
        else:
            formatted_p_value = "{:.4f}".format(p_value)
        if p_value < 2.2e-16:
            formatted_p_value = 2.2e-16
            text = f"Kruskal-Wallis: P < {formatted_p_value}"
        else:
            text = f"Kruskal-Wallis: P = {formatted_p_value}"
        plt.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize, fontweight='bold', verticalalignment='top',
                 bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'))

    if save == True:
        plt.savefig(save_pathway, dpi=300, bbox_inches='tight')

    if ax is None:
        return fig, ax
