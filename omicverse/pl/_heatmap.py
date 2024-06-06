import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import CategoricalDtype, is_numeric_dtype
import seaborn as sns
import scanpy as sc
import pandas as pd
from ..utils import plotset

pycomplexheatmap_install=False

def check_pycomplexheatmap():
    """
        
    """
    global pycomplexheatmap_install
    try:
        import PyComplexHeatmap as pch
        pycomplexheatmap_install=True
        print('PyComplexHeatmap have been install version:',pch.__version__)
    except ImportError:
        raise ImportError(
            'Please install the tangram: `pip install PyComplexHeatmap`.'
            )
    
def complexheatmap(adata,
                       groupby ='',
                       figsize =(6,10),
                       layer: str = None,
                       use_raw: bool = False,
                       var_names = None,
                       gene_symbols = None,
                       standard_scale:str = None,
                       col_color_bars:dict = None,
                       col_color_labels:dict = None,
                       left_color_bars:dict = None,
                       left_color_labels:dict = None,
                       right_color_bars:dict = None,
                       right_color_labels:dict = None,
                       marker_genes_dict:dict = None,
                       index_name:str = '',
                       value_name:str = '',
                       cmap:str = 'parula',
                       xlabel:str = None,
                       ylabel:str = None,
                       label:str = '',
                       save:bool = False,
                       save_pathway:str = '',
                       legend_gap:int = 7,
                       legend_hpad:int = 0,
                       show:bool = False,
                       left_add_text:bool = False,
                       col_split_gap:int = 1,
                       row_split_gap:int = 1,
                       col_height:int = 4,
                       left_height:int = 4,
                       right_height:int = 4,
                       col_cluster:bool = False,
                       row_cluster:bool = False,
                       row_split = None,
                       col_split = None,
                       legend:bool = True,
                       plot_legend:bool = True,
                       right_fontsize:int = 12,
                        ):
    """
    Generate a complex heatmap from single-cell RNA-seq data.

    Parameters:
    - adata (AnnData): Annotated data object containing single-cell RNA-seq data.
    - groupby (str, optional): Grouping variable for the heatmap. Default is ''.
    - figsize (tuple, optional): Figure size. Default is (6, 10).
    - layer (str, optional): Data layer to use. Default is None.
    - use_raw (bool, optional): Whether to use the raw data. Default is False.
    - var_names (list or None, optional): List of genes to include in the heatmap. Default is None.
    - gene_symbols (None, optional): Not used in the function.
    - standard_scale (str, optional): Method for standardizing values. Options: 'obs', 'var', None. Default is None.
    - col_color_bars (dict, optional): Dictionary mapping columns types to colors.
    - col_color_labels (dict, optional): Dictionary mapping column labels to colors.
    - left_color_bars (dict, optional): Dictionary mapping left types to colors.
    - left_color_labels (dict, optional): Dictionary mapping left labels to colors.
    - right_color_bars (dict, optional): Dictionary mapping right types to colors.
    - right_color_labels (dict, optional): Dictionary mapping right labels to colors.
    - marker_genes_dict (dict, optional): Dictionary mapping cell types to marker genes.
    - index_name (str, optional): Name for the index column in the melted DataFrame. Default is ''.
    - value_name (str, optional): Name for the value column in the melted DataFrame. Default is ''.
    - cmap (str, optional): Colormap for the heatmap. Default is 'parula'.
    - xlabel (str, optional): X-axis label. Default is ''.
    - ylabel (str, optional): Y-axis label. Default is ''.
    - label (str, optional): Label for the plot. Default is ''.
    - save (bool, optional): Whether to save the plot. Default is False.
    - save_pathway (str, optional): File path for saving the plot. Default is ''.
    - legend_gap (int, optional): Gap between legend items. Default is 7.
    - legend_hpad (int, optional): Horizontal space between the heatmap and legend, default is 2 [mm].
    - show (bool, optional): Whether to display the plot. Default is False.

    Returns:
    - None: Displays the complex heatmap.
    """
    check_pycomplexheatmap()
    global pycomplexheatmap_install
    if pycomplexheatmap_install==True:
        global_imports("PyComplexHeatmap","pch")
    
    #sns.set_style('white')
    plotset()
    if layer is not None:
        use_raw = False
    if use_raw == True:
        adata = adata.raw.to_adata()
        use_raw = False
        
    if var_names == None:
        var_names = adata.var_names
    if isinstance(var_names, str):
        var_names = [var_names]
    groupby_copy = groupby
    
    groupby_index = None
    if groupby is not None:
        if isinstance(groupby, str):
            # if not a list, turn into a list
            groupby = [groupby]
        for group in groupby:
            if group not in list(adata.obs_keys()) + [adata.obs.index.name]:
                if adata.obs.index.name is not None:
                    msg = f' or index name "{adata.obs.index.name}"'
                else:
                    msg = ""
                raise ValueError("groupby has to be a valid observation."
                                 f"Given {group}, is not in observations: {adata.obs_keys()}" + msg)
            if group in adata.obs.keys() and group == adata.obs.index.name:
                raise ValueError(f"Given group {group} is both and index and a column level, "
                                 "which is ambiguous.")
            if group == adata.obs.index.name:
                groupby_index = group
    if groupby_index is not None:
        # obs_tidy contains adata.obs.index
        # and does not need to be given
        groupby = groupby.copy()  # copy to not modify user passed parameter
        groupby.remove(groupby_index)

    if col_color_bars == None:
        print('Error, please input col_color before run this function.')
    if col_color_labels == None:
        print('Error, please input col_color before run this function.')

    if marker_genes_dict == None:
        print('Error, please input marker_genes_dict before run this function.')
    
    keys = list(groupby) + list(np.unique(var_names))
    obs_tidy = sc.get.obs_df(adata, keys=keys, layer=layer, use_raw=use_raw, gene_symbols=gene_symbols)
    assert np.all(np.array(keys) == np.array(obs_tidy.columns))

    if groupby_index is not None:
        # reset index to treat all columns the same way.
        obs_tidy.reset_index(inplace=True)
        groupby.append(groupby_index)

    if groupby is None:
        categorical = pd.Series(np.repeat("", len(obs_tidy))).astype("category")
    elif len(groupby) == 1 and is_numeric_dtype(obs_tidy[groupby[0]]):
        # if the groupby column is not categorical, turn it into one
        # by subdividing into  `num_categories` categories
        categorical = pd.cut(obs_tidy[groupby[0]], num_categories)
    elif len(groupby) == 1:
        categorical = obs_tidy[groupby[0]].astype("category")
        categorical.name = groupby[0]
    else:
        # join the groupby values  using "_" to make a new 'category'
        categorical = obs_tidy[groupby].apply("_".join, axis=1).astype("category")
        categorical.name = "_".join(groupby)

        # preserve category order
        from itertools import product

        order = {
            "_".join(k): idx
            for idx, k in enumerate(
                product(*(obs_tidy[g].cat.categories for g in groupby))
            )
        }
        categorical = categorical.cat.reorder_categories(
            sorted(categorical.cat.categories, key=lambda x: order[x])
        )
    obs_tidy = obs_tidy[var_names].set_index(categorical)
    categories = obs_tidy.index.categories
    obs_tidy = obs_tidy.groupby(groupby).mean()


    
    if standard_scale == "obs":
        obs_tidy = obs_tidy.sub(obs_tidy.min(1), axis=0)
        obs_tidy = obs_tidy.div(obs_tidy.max(1), axis=0).fillna(0)
    elif standard_scale == "var":
        obs_tidy -= obs_tidy.min(0)
        obs_tidy = (obs_tidy / obs_tidy.max(0)).fillna(0)
    elif standard_scale is None:
        pass


    if right_color_bars==None:
        # colorbar of gene
        gene_color_dict = {}
        for cell_type, genes in marker_genes_dict.items():
            cell_type_color = [color for color, category in zip(adata.uns[groupby_copy+'_colors'], adata.obs[groupby_copy].cat.categories) if category == cell_type][0]
            for gene in genes:
                gene_color_dict[gene] = cell_type_color
        right_color_bars = gene_color_dict

    if right_color_labels==None:
        right_color_labels = right_color_bars


    # col
    df_col = obs_tidy.copy()
    df_col[groupby_copy] = df_col.index
    col_ha = pch.HeatmapAnnotation(label=pch.anno_label(df_col[groupby_copy],merge=True,rotation=90,extend=True,
                                            colors=col_color_bars,adjust_color=True,luminance=0.75,
                                            relpos=(0.5,0)), #fontsize=10
                           Celltype=pch.anno_simple(df_col[groupby_copy],colors=col_color_labels,height=col_height), #legend_kws={'fontsize':4}
                           verbose=1,axis=1,plot=False)

    
    # dict to Dataframe
    marker_genes_df = pd.DataFrame.from_dict(marker_genes_dict, orient='index')
    # Dataframe transpose
    marker_genes_df = marker_genes_df.transpose()
    melted_df = marker_genes_df.melt(var_name=index_name, value_name=value_name).dropna()
    melted_df.index = melted_df.loc[:,value_name]
    df_row = melted_df
    del melted_df

    if left_color_labels == None:
        left_ha = pch.HeatmapAnnotation(
                           Marker_Gene=pch.anno_simple(df_row[index_name],legend=True,
                                             colors=left_color_bars,add_text=left_add_text,height=left_height),
                           verbose=1,axis=0,plot_legend=False,plot=False)
    else:   
        left_ha = pch.HeatmapAnnotation(
                           label=pch.anno_label(df_row[index_name],merge=True,extend=False,
                                            colors=left_color_labels,adjust_color=True,luminance=0.75,
                                              relpos=(1,0.5)),
                           Marker_Gene=pch.anno_simple(df_row[index_name],legend=True,
                                             colors=left_color_bars,add_text=left_add_text,height=left_height),
                           verbose=1,axis=0,plot_legend=False,plot=False)

    if right_color_labels == None:
        right_ha = pch.HeatmapAnnotation(
                           Group=pch.anno_simple(df_row[index_name],legend=True,
                                             colors=right_color_bars,height=right_height),
                           verbose=1,axis=0,plot_legend=False,label_kws=dict(visible=False),plot=False,)
    else:
        right_ha = pch.HeatmapAnnotation(
                           Group=pch.anno_simple(df_row[index_name],legend=True,
                                             colors=right_color_bars,height=right_height),
                           label=pch.anno_label(df_row[value_name],merge=True,extend=True,
                                            colors=right_color_labels,adjust_color=True,luminance=0.75,
                                            relpos=(0,0.5),fontsize=right_fontsize), #fontsize=10
                           verbose=1,axis=0,plot_legend=False,label_kws=dict(visible=False),plot=False)        



    if row_split!=None:
        row_split_copy = df_row.loc[:,index_name]
    else:
        row_split_copy = row_split
    if col_split!=None:
        col_split_copy = df_col.loc[:,index_name]
    else:
        col_split_copy = col_split


    plt.figure(figsize=figsize)
    obs_copy = obs_tidy.copy().loc[df_col.index.tolist(),df_row.index.tolist()]
    cm = pch.ClusterMapPlotter(data=obs_copy.T,
                       top_annotation=col_ha, 
                       left_annotation=left_ha,
                       right_annotation=right_ha,
                       row_cluster=row_cluster,col_cluster=col_cluster,
                       label=label, row_dendrogram=False,legend_gap=legend_gap,
                       row_split=row_split_copy,col_split=col_split_copy,
                       col_split_gap=col_split_gap,
                       row_split_gap=row_split_gap,
                       row_split_order=list(marker_genes_dict.keys()),
                       # col_split_order=df_row.Group.unique().tolist(),
                       cmap=cmap,rasterized=True,
                       xlabel=xlabel, legend_hpad=legend_hpad,
                       ylabel=ylabel,
                       xlabel_kws=dict(color='black', fontsize=14, labelpad=0),
                       legend=legend,
                       plot_legend=plot_legend,
                       # ylabel_kws=dict(color='black', fontsize=14, labelpad=0),
                          )
    
    #plt.savefig("Loyfer2023_heatmap.pdf",bbox_inches='tight')
    if save ==True:
        plt.savefig(save_pathway,bbox_inches='tight',dpi=300)
    if show ==True:
        plt.show()
    return cm

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)