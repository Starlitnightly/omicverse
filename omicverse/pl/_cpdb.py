import anndata
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import random
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from ..single import cpdb_exact_target,cpdb_exact_source
from ._cpdbviz import CellChatViz
from ._palette import palette_28,palette_56,palette_112

def cpdb_network(adata:anndata.AnnData,interaction_edges:pd.DataFrame,
                      celltype_key:str,nodecolor_dict=None,count_min=50,
                       source_cells=None,target_cells=None,
                      edgeswidth_scale:int=1,nodesize_scale:int=1,
                      figsize:tuple=(4,4),title:str='',
                      fontsize:int=12,ax=None,
                     return_graph:bool=False):
    r"""
    Create a circular network plot of CellPhoneDB cell-cell interactions.
    
    Args:
        adata: Annotated data object with cell type information
        interaction_edges: DataFrame with SOURCE, TARGET, and COUNT columns
        celltype_key: Column name for cell type annotation
        nodecolor_dict: Custom color mapping for cell types (None, uses default)
        count_min: Minimum interaction count threshold (50)
        source_cells: List of source cell types to include (None, uses all)
        target_cells: List of target cell types to include (None, uses all)
        edgeswidth_scale: Scale factor for edge widths (1)
        nodesize_scale: Scale factor for node sizes (1)
        figsize: Figure dimensions as (width, height) ((4,4))
        title: Plot title ('')
        fontsize: Font size for labels (12)
        ax: Existing matplotlib axes object (None)
        return_graph: Whether to return NetworkX graph object (False)
        
    Returns:
        ax: matplotlib.axes.Axes object or NetworkX graph if return_graph=True
    """
    G=nx.DiGraph()
    for i in interaction_edges.index:
        if interaction_edges.loc[i,'COUNT']>count_min:
            G.add_edge(interaction_edges.loc[i,'SOURCE'],
                       interaction_edges.loc[i,'TARGET'],
                       weight=interaction_edges.loc[i,'COUNT'],)
        else:
            G.add_edge(interaction_edges.loc[i,'SOURCE'],
                       interaction_edges.loc[i,'TARGET'],
                       weight=0,)

    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,palette_112))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,palette_28))

    G_nodes_dict={}
    links = []
    for i in G.edges:
        if i[0] not in G_nodes_dict.keys():
            G_nodes_dict[i[0]]=0
        if i[1] not in G_nodes_dict.keys():
            G_nodes_dict[i[1]]=0
        links.append({"source": i[0], "target": i[1]})
        weight=G.get_edge_data(i[0],i[1])['weight']
        G_nodes_dict[i[0]]+=weight
        G_nodes_dict[i[1]]+=weight

    edge_li=[]
    for u,v in G.edges:
        if G.get_edge_data(u, v)['weight']>0:
            if source_cells==None and target_cells==None:
                edge_li.append((u,v))
            elif source_cells!=None and target_cells==None:
                if u in source_cells:
                    edge_li.append((u,v))
            elif source_cells==None and target_cells!=None:
                if v in target_cells:
                    edge_li.append((u,v))
            else:
                if u in source_cells and v in target_cells:
                    edge_li.append((u,v))


    import matplotlib.pyplot as plt
    import numpy as np
    #edgeswidth_scale=10
    #nodesize_scale=5
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize) 
    else:
        ax=ax
    #pos = nx.spring_layout(G, scale=pos_scale, k=(pos_size)/np.sqrt(G.order()))
    pos = nx.circular_layout(G)
    p=dict(G.nodes)
    
    nodesize=np.array([G_nodes_dict[u] for u in G.nodes()])/nodesize_scale
    nodecolos=[type_color_all[u] for u in G.nodes()]
    nx.draw_networkx_nodes(G, pos, nodelist=p,node_size=nodesize,node_color=nodecolos)
    
    edgewidth = np.array([G.get_edge_data(u, v)['weight'] for u, v in edge_li])
    edgewidth=np.log10(edgewidth+1)/edgeswidth_scale
    edgecolos=[type_color_all[u] for u,o in edge_li]
    nx.draw_networkx_edges(G, pos,width=edgewidth,edge_color=edgecolos,edgelist=edge_li)
    plt.grid(False)
    plt.axis("off")
    
    pos1=dict()
    #for i in pos.keys():
    #    pos1[i]=np.array([-1000,-1000])
    for i in G.nodes:
        pos1[i]=pos[i]
    from adjustText import adjust_text
    import adjustText
    from matplotlib import patheffects
    texts=[ax.text(pos1[i][0], 
               pos1[i][1],
               i,
               fontdict={'size':fontsize,'weight':'normal','color':'black'},
                path_effects=[patheffects.withStroke(linewidth=2, foreground='w')]
               ) for i in G.nodes if 'ENSG' not in i]
    if adjustText.__version__<='0.8':
        adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
    else:
        adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    arrowprops=dict(arrowstyle='->', color='black'))
        
    plt.title(title,fontsize=fontsize+1)

    if return_graph==True:
        return G
    else:
        return ax
    
def cpdb_chord(adata:anndata.AnnData,interaction_edges:pd.DataFrame,
                      celltype_key:str,count_min=50,nodecolor_dict=None,
                      fontsize=12,padding=80,radius=100,save='chord.svg',
                      rotation=0,bg_color = "#ffffff",bg_transparancy = 1.0):
    r"""
    Create a chord diagram visualization of CellPhoneDB interactions.
    
    Args:
        adata: Annotated data object with cell type information
        interaction_edges: DataFrame with SOURCE, TARGET, and COUNT columns
        celltype_key: Column name for cell type annotation
        count_min: Minimum interaction count threshold (50)
        nodecolor_dict: Custom color mapping for cell types (None, uses default)
        fontsize: Font size for labels (12)
        padding: Padding around chord diagram (80)
        radius: Radius of the chord diagram (100)
        save: File path to save SVG output ('chord.svg')
        rotation: Rotation angle for the diagram (0)
        bg_color: Background color ('#ffffff')
        bg_transparancy: Background transparency (1.0)
        
    Returns:
        fig: OpenChord figure object
    """
    import itertools
    import openchord as ocd
    data=interaction_edges.loc[interaction_edges['COUNT']>count_min].iloc[:,:2]
    data = list(itertools.chain.from_iterable((i, i[::-1]) for i in data.values))
    matrix = pd.pivot_table(
        pd.DataFrame(data), index=0, columns=1, aggfunc="size", fill_value=0
    ).values.tolist()
    # 获取所有唯一的名称
    unique_names = sorted(set(itertools.chain.from_iterable(data)))

    # 将矩阵转换为 DataFrame，并使用 unique_names 作为行和列的标签
    matrix_df = pd.DataFrame(matrix, index=unique_names, columns=unique_names)

    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,palette_112))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,palette_28))
    
    fig=ocd.Chord(matrix, unique_names,radius=radius)
    fig.colormap=[type_color_all[u] for u in unique_names]
    fig.font_size=fontsize
    fig.padding = padding
    fig.rotation = rotation
    fig.bg_color = bg_color
    fig.bg_transparancy = bg_transparancy
    if save!=None:
        fig.save_svg(save)
    return fig

def cpdb_heatmap(adata:anndata.AnnData,interaction_edges:pd.DataFrame,
                      celltype_key:str,nodecolor_dict=None,ax=None,
                      source_cells=None,target_cells=None,
                      figsize=(3,3),fontsize=11,rotate=False,legend=True,
                      legend_kws={'fontsize':8,'bbox_to_anchor':(5, -0.5),'loc':'center left',},
                      return_table=False,**kwargs):
    r"""
    Create a dot heatmap of CellPhoneDB interaction counts between cell types.
    
    Args:
        adata: Annotated data object with cell type information
        interaction_edges: DataFrame with SOURCE, TARGET, and COUNT columns
        celltype_key: Column name for cell type annotation
        nodecolor_dict: Custom color mapping for cell types (None, uses default)
        ax: Existing matplotlib axes object (None)
        source_cells: List of source cell types to include (None, uses all)
        target_cells: List of target cell types to include (None, uses all)
        figsize: Figure dimensions as (width, height) ((3,3))
        fontsize: Font size for labels (11)
        rotate: Whether to rotate the heatmap layout (False)
        legend: Whether to show legend (True)
        legend_kws: Legend keyword arguments ({'fontsize':8,'bbox_to_anchor':(5, -0.5),'loc':'center left',})
        return_table: Whether to return data table instead (False)
        **kwargs: Additional arguments passed to DotClustermapPlotter
        
    Returns:
        ax: matplotlib.axes.Axes object or DataFrame if return_table=True
    """
    
    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,palette_112))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,palette_28))

    from PyComplexHeatmap import DotClustermapPlotter,HeatmapAnnotation,anno_simple,anno_label,AnnotationBase
    
    

    
    corr_mat=interaction_edges.copy()
    if source_cells!=None and target_cells==None:
        corr_mat=corr_mat.loc[corr_mat['SOURCE'].isin(source_cells)]
    elif source_cells==None and target_cells!=None:
        corr_mat=corr_mat.loc[corr_mat['TARGET'].isin(target_cells)]
    elif source_cells!=None and target_cells!=None:
        corr_mat=corr_mat.loc[corr_mat['TARGET'].isin(source_cells)]
        corr_mat=corr_mat.loc[corr_mat['SOURCE'].isin(target_cells)]
    else:
        pass

    #if rotate==True:


    df_row=corr_mat['SOURCE'].drop_duplicates().to_frame()
    df_row['Celltype']=df_row.SOURCE
    df_row.set_index('SOURCE',inplace=True)

    df_col=corr_mat['TARGET'].drop_duplicates().to_frame()
    df_col['Celltype']=df_col.TARGET
    df_col.set_index('TARGET',inplace=True)

    import matplotlib.pyplot as plt
    #from pycomplexheatmap import DotClustermapPlotter, HeatmapAnnotation, anno_simple
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize) 
    else:
        ax=ax

    # 定义行和列的注释
    if rotate!=True:
        row_ha = HeatmapAnnotation(
            TARGET=anno_simple(
                df_row.Celltype,
                colors=[type_color_all[i] for i in df_row.Celltype],
                add_text=False,
                text_kws={'color': 'black', 'rotation': 0,'fontsize':10},
                legend=False  # 设置为 True 以显示行的图例
            ),
            legend_gap=7,
            axis=1,
            verbose=0,
            #label_side='left',
            label_kws={'rotation': 90, 'horizontalalignment': 'right','fontsize':0}
        )
    else:
        row_ha = HeatmapAnnotation(
            TARGET=anno_simple(
                df_row.Celltype,
                colors=[type_color_all[i] for i in df_row.Celltype],
                add_text=False,
                text_kws={'color': 'black', 'rotation': 0,'fontsize':10},
                legend=False  # 设置为 True 以显示行的图例
            ),
            legend_gap=7,
            axis=0,
            verbose=0,
            #label_side='left',
            label_kws={'rotation': 90, 'horizontalalignment': 'right','fontsize':0}
        )
    if rotate!=True:
        col_ha = HeatmapAnnotation(
            SOURCE=anno_simple(
                df_col.Celltype,
                colors=[type_color_all[i] for i in df_col.Celltype],  # 修正列的颜色引用
                legend=False,
                add_text=False
            ),
            verbose=0,
            #label_side='top',
            label_kws={'horizontalalignment': 'right','fontsize':0},
            legend_kws={'ncols': 1},  # 调整图例的列数为1
            legend=False,
            legend_hpad=7,
            legend_vpad=5,
            axis=0
        )
    else:
        col_ha = HeatmapAnnotation(
            SOURCE=anno_simple(
                df_col.Celltype,
                colors=[type_color_all[i] for i in df_col.Celltype],  # 修正列的颜色引用
                legend=False,
                add_text=False
            ),
            verbose=0,
            #label_side='left',
            label_kws={'horizontalalignment': 'right','fontsize':0},
            legend_kws={'ncols': 1},  # 调整图例的列数为1
            legend=False,
            legend_hpad=7,
            legend_vpad=5,
            axis=1
        )
    # 绘制热图
    import PyComplexHeatmap as pch
    if pch.__version__>'1.7':
        hue_arg=None
    else:
        hue_arg='SOURCE'
    if rotate==True:
        cm = DotClustermapPlotter(
        corr_mat,
        y='SOURCE',
        x='TARGET',
        value='COUNT',
        hue=hue_arg,
        legend_gap=7,
        top_annotation=col_ha,
        left_annotation=row_ha,
        c='COUNT',
        s='COUNT',
        cmap='Reds',
        vmin=0,
        show_rownames=False,
        show_colnames=False,
        row_dendrogram=False,
        col_names_side='left',
        legend=legend,
        **kwargs
        #cmap_legend_kws={'font':12},
    )
    else:
        cm = DotClustermapPlotter(
            corr_mat,
            x='SOURCE',
            y='TARGET',
            value='COUNT',
            hue=hue_arg,
            legend_gap=7,
            top_annotation=row_ha,
            left_annotation=col_ha,
            c='COUNT',
            s='COUNT',
            cmap='Reds',
            vmin=0,
            show_rownames=False,
            show_colnames=False,
            row_dendrogram=False,
            col_names_side='top',
            legend=legend,
            **kwargs
            #cmap_legend_kws={'font':12},
        )

    # 调整网格设置
    cm.ax_heatmap.grid(which='minor', color='gray', linestyle='--', alpha=0.5)
    cm.ax_heatmap.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    cm.cmap_legend_kws={'ncols': 1}
    #cm.ax_heatmap.set_ylabel('TARGET',fontsize=12)

    # 修改颜色条的标注字体大小
    #cbar = cm.ax_heatmap.colorbar(cm.im, ax=cm.ax_heatmap)  # 获取颜色条对象
    #cbar.ax.tick_params(labelsize=12)  # 修改颜色条的 tick 字体大小，例如 12
    test=plt.gcf().axes
    if rotate!=True:
        for ax in plt.gcf().axes:
            if hasattr(ax, 'get_ylabel'):
                if ax.get_ylabel() == 'COUNT':  # 假设 colorbar 有一个特定的标签
                    cbar = ax
                    cbar.tick_params(labelsize=fontsize)
                    cbar.set_ylabel('COUNT',fontsize=fontsize)
                if ax.get_xlabel() == 'SOURCE':
                    ax.xaxis.set_label_position('top') 
                    #ax.xaxis.tick_top() 
                    ax.set_ylabel('Target',fontsize=fontsize)
                if ax.get_ylabel() == 'TARGET':
                    ax.xaxis.set_label_position('top') 
                    #ax.xaxis.tick_top() 
                    ax.set_xlabel('Source',fontsize=fontsize)
            ax.grid(False)
    else:
        for ax in plt.gcf().axes:
            if hasattr(ax, 'get_ylabel'):
                if ax.get_ylabel() == 'COUNT':  # 假设 colorbar 有一个特定的标签
                    cbar = ax
                    cbar.tick_params(labelsize=fontsize)
                    cbar.set_ylabel('COUNT',fontsize=fontsize)
                if ax.get_ylabel() == 'SOURCE':
                    ax.xaxis.set_label_position('top') 
                    #ax.xaxis.tick_top() 
                    ax.set_xlabel('Target',fontsize=fontsize)
                if ax.get_xlabel() == 'TARGET':
                    ax.xaxis.set_label_position('top') 
                    #ax.xaxis.tick_top() 
                    ax.set_ylabel('Source',fontsize=fontsize)
            ax.grid(False)

    plt.grid(False)
    plt.tight_layout()  # 调整布局以适应所有组件

    handles = [plt.Line2D([0], [0], color=type_color_all[cell], lw=4) for cell in type_color_all.keys()]

    # 设置图例的标签
    labels = type_color_all.keys()

    # 在现有图像的右侧添加图例
    plt.legend(handles, labels, 
            borderaxespad=1, handletextpad=0.5, labelspacing=0.2,**legend_kws)


    plt.tight_layout()
    if return_table==True:
        return corr_mat
    else:
        return ax

def cpdb_interacting_heatmap(adata,
                             celltype_key,
                             means,
                             pvalues,
                             source_cells,
                             target_cells,
                             min_means=3,
                             nodecolor_dict=None,
                             ax=None,
                             figsize=(2,6),
                             fontsize=12,
                             plot_secret=True,
                             return_table=False,
                             transpose=False):
    r"""
    Create a detailed heatmap of specific interacting pairs with significance.
    
    Args:
        adata: Annotated data object with cell type information
        celltype_key: Column name for cell type annotation
        means: CellPhoneDB means DataFrame with interaction data
        pvalues: CellPhoneDB pvalues DataFrame with significance data
        source_cells: List of source cell types
        target_cells: List of target cell types
        min_means: Minimum mean expression threshold (3)
        nodecolor_dict: Custom color mapping for cell types (None, uses default)
        ax: Existing matplotlib axes object (None)
        figsize: Figure dimensions as (width, height) ((2,6))
        fontsize: Font size for labels (12)
        plot_secret: Whether to show secreted/non-secreted annotation (True)
        return_table: Whether to return data table instead (False)
        transpose: Whether to transpose the heatmap layout (False)
        
    Returns:
        ax: matplotlib.axes.Axes object or DataFrame if return_table=True
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 生成颜色字典：优先使用传入的 nodecolor_dict，否则构建新的
    if nodecolor_dict is not None:
        type_color_all = nodecolor_dict
    else:
        key_colors = '{}_colors'.format(celltype_key)
        if key_colors in adata.uns:
            type_color_all = dict(zip(adata.obs[celltype_key].cat.categories,
                                      adata.uns[key_colors]))
        else:
            categories = adata.obs[celltype_key].cat.categories
            if len(categories) > 28:
                type_color_all = dict(zip(categories, palette_112))
            else:
                type_color_all = dict(zip(categories, palette_28))
                
    # 筛选 source 与 target 细胞，剔除无效的交互记录
    sub_means = cpdb_exact_target(means, target_cells)
    sub_means = cpdb_exact_source(sub_means, source_cells)
    sub_means = sub_means.loc[sub_means['gene_a'].notnull()]
    sub_means = sub_means.loc[sub_means['gene_b'].notnull()]

    # 构造交互对对应的分泌状态字典
    secreted_dict = dict(zip(sub_means['interacting_pair'],
                             ['Secreted' if status else 'Non' for status in sub_means['secreted']]))

    # 提取均值信息矩阵，新矩阵的新索引为交互对名称
    new_df = sub_means.iloc[:, 10:].copy()
    new_df.index = sub_means['interacting_pair'].tolist()
    # 只保留 sum > min_means 的交互对
    cor = new_df.loc[new_df.sum(axis=1) > min_means]

    # 处理 p 值矩阵：重新对齐行列，保留至少一个显著 p 值的行
    sub_p = pvalues.copy()
    sub_p.index = sub_p['interacting_pair']
    sub_p = sub_p.loc[cor.index, cor.columns]
    significant_rows = (sub_p < 0.05).any(axis=1)
    cor = cor.loc[significant_rows]
    sub_p = sub_p.loc[significant_rows]

    # 将矩阵转换为长格式数据，便于后续绘图使用
    sub_p_mat = sub_p.stack().reset_index(name="pvalue")
    corr_mat = cor.stack().reset_index(name="means")
    corr_mat['-logp'] = -np.log10(sub_p_mat["pvalue"] + 0.001)

    # 构造行和列的附加信息数据框，行信息来自交互对
    df_row = corr_mat['level_0'].drop_duplicates().to_frame()
    df_row['RowGroup'] = df_row['level_0'].apply(lambda x: x.split('|')[0])
    df_row['Secreted'] = df_row['level_0'].map(secreted_dict)
    df_row.set_index('level_0', inplace=True)

    # 列信息取自 stacking 后 level_1，对应的交互pair含有 "Source|Target"
    df_col = corr_mat['level_1'].drop_duplicates().to_frame()
    df_col['Source'] = df_col['level_1'].apply(lambda x: x.split('|')[0])
    df_col['Target'] = df_col['level_1'].apply(lambda x: x.split('|')[1])
    df_col.set_index('level_1', inplace=True)

    from PyComplexHeatmap import DotClustermapPlotter, HeatmapAnnotation, anno_simple

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 针对颜色注释参数，我们传入类别→颜色的字典，保证顺序与映射正确
    source_colors = {cat: type_color_all[cat] for cat in df_col['Source'].unique()}
    target_colors = {cat: type_color_all[cat] for cat in df_col['Target'].unique()}

    if transpose:
        # 转置模式：x 和 y 数据字段交换，同时对应的注释也交换位置
        # 原来的行信息 (df_row, Secreted)用于顶部注释
        if plot_secret:
            top_annotation = HeatmapAnnotation(
                Secreted=anno_simple(df_row['Secreted'],
                                     cmap='Set1',
                                     text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
                                     legend=True,
                                     add_text=False),
                verbose=0,
                label_side='left',
                label_kws={'horizontalalignment': 'right', 'fontsize': fontsize}
            )
        else:
            top_annotation = None
        # 原来的列信息 (df_col, Source & Target)用于左侧注释
        left_annotation = HeatmapAnnotation(
            Source=anno_simple(df_col['Source'],
                               colors=source_colors,
                               text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
                               legend=True,
                               add_text=False),
            Target=anno_simple(df_col['Target'],
                               colors=target_colors,
                               text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
                               legend=True,
                               add_text=False),
            verbose=0,
            axis=0,
            label_side='bottom',
            label_kws={'horizontalalignment': 'right', 'fontsize': fontsize}
        )
        # 注意：转置后，原行数据（df_row）对应 x 轴，原列数据（df_col）对应 y 轴
        cm = DotClustermapPlotter(corr_mat,
                                  x='level_0',
                                  y='level_1',
                                  value='means',
                                  c='means',
                                  s='-logp',
                                  cmap='Reds',
                                  vmin=0,
                                  top_annotation=top_annotation,
                                  left_annotation=left_annotation,
                                  row_dendrogram=True,
                                  show_rownames=True,
                                  show_colnames=True)
    else:
        # 默认模式：保持原始 x= level_1, y= level_0 的对应关系
        col_ha = HeatmapAnnotation(
            Source=anno_simple(df_col['Source'],
                               colors=source_colors,
                               text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
                               legend=True,
                               add_text=False),
            Target=anno_simple(df_col['Target'],
                               colors=target_colors,
                               text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
                               legend=True,
                               add_text=False),
            verbose=0,
            label_side='left',
            label_kws={'horizontalalignment': 'right', 'fontsize': fontsize}
        )
        if plot_secret:
            row_ha = HeatmapAnnotation(
                Secreted=anno_simple(df_row['Secreted'],
                                     cmap='Set1',
                                     text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
                                     legend=True,
                                     add_text=False),
                verbose=0,
                axis=0,
                label_kws={'horizontalalignment': 'left', 'fontsize': fontsize}
            )
        else:
            row_ha = None
        cm = DotClustermapPlotter(corr_mat,
                                  x='level_1',
                                  y='level_0',
                                  value='means',
                                  c='means',
                                  s='-logp',
                                  cmap='Reds',
                                  vmin=0,
                                  top_annotation=col_ha,
                                  left_annotation=row_ha,
                                  row_dendrogram=True,
                                  show_rownames=True,
                                  show_colnames=True)

    # 设置热图网格和 colorbar 外观
    cm.ax_heatmap.grid(which='minor', color='gray', linestyle='--')

    # 遍历所有 axes 控件，调整 colorbar 的标签字体
    for curr_ax in plt.gcf().axes:
        if hasattr(curr_ax, 'get_ylabel') and curr_ax.get_ylabel() == 'means':
            curr_ax.tick_params(labelsize=fontsize)
            curr_ax.set_ylabel('means', fontsize=fontsize)
        curr_ax.grid(False)
        curr_ax.tick_params(labelsize=fontsize)

    # 使用 tick_params 统一设置热图 x 和 y 轴 tick 标签的字体大小，解决转置情况下字体设置失效的问题
    cm.ax_heatmap.tick_params(axis='x', labelsize=fontsize)
    cm.ax_heatmap.tick_params(axis='y', labelsize=fontsize)

    for ax in plt.gcf().axes:
        leg = ax.get_legend()
        #print(leg)
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(fontsize) # 这里设为你想要的字体大小

    if return_table:
        return cor
    else:
        return cm.ax_heatmap


def cpdb_group_heatmap(adata,
                       
                            celltype_key,
                            means,
                            #pvalues,
                            source_cells,
                            target_cells,
                            min_means=3,
                            nodecolor_dict=None,
                            ax=None,
                            figsize=(2,6),
                            fontsize=12,
                            plot_secret=True,
                      cmap={'Target':'Blues','Source':'Reds'},
                      return_table=False,):
    r"""
    Create a grouped heatmap showing ligand and receptor expression by cell type.
    
    Args:
        adata: Annotated data object with expression and cell type data
        celltype_key: Column name for cell type annotation
        means: CellPhoneDB means DataFrame with interaction data
        source_cells: List of source cell types
        target_cells: List of target cell types
        min_means: Minimum mean expression threshold (3)
        nodecolor_dict: Custom color mapping for cell types (None, uses default)
        ax: Existing matplotlib axes object (None)
        figsize: Figure dimensions as (width, height) ((2,6))
        fontsize: Font size for labels (12)
        plot_secret: Whether to show secreted/non-secreted annotation (True)
        cmap: Colormap dictionary for Target and Source ({'Target':'Blues','Source':'Reds'})
        return_table: Whether to return data table instead (False)
        
    Returns:
        ax: matplotlib.axes.Axes object or DataFrame if return_table=True
    """

    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,palette_112))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,palette_28))
    
    sub_means=cpdb_exact_target(means,target_cells)
    sub_means=cpdb_exact_source(sub_means,source_cells)

    
    sub_means=sub_means.loc[~sub_means['gene_a'].isnull()]
    sub_means=sub_means.loc[~sub_means['gene_b'].isnull()]

    secreted_dict=dict(zip(sub_means['interacting_pair'],
                       ['Secreted' if i==True else 'Non' for i in sub_means['secreted']]
                  ))

    new=sub_means.iloc[:,10:]
    new.index=sub_means['interacting_pair'].tolist()
    cor=new.loc[new.sum(axis=1)[new.sum(axis=1)>min_means].index]
    '''
    sub_p=pvalues
    sub_p.index=sub_p['interacting_pair']
    sub_p=sub_p.loc[cor.index,cor.columns]
    sub_p_mat=sub_p.stack().reset_index(name="pvalue")
    '''

    source_gene_pd=pd.DataFrame(index=cor.index)
    for source_cell in source_cells:
        source_gene_pd[source_cell]=adata[adata.obs[celltype_key]==source_cell,
        [i.split('_')[0] for i in cor.index]].to_df().mean().values
    
    target_gene_pd=pd.DataFrame(index=cor.index)
    for target_cell in target_cells:
        target_gene_pd[target_cell]=adata[adata.obs[celltype_key]==target_cell,
        [i.split('_')[1] for i in cor.index]].to_df().mean().values

    cor=pd.concat([source_gene_pd,target_gene_pd],axis=1)

    

    corr_mat = cor.stack().reset_index(name="means")
    #corr_mat['-logp']=-np.log10(sub_p_mat['pvalue']+0.001)
    type_dict=dict(zip(source_cells+target_cells,
                      ['Source' for i in source_cells]+['Target' for i in target_cells]))
    corr_mat['Type']=corr_mat['level_1'].map(type_dict)
    #corr_mat.head()

    df_row=corr_mat['level_0'].drop_duplicates().to_frame()
    df_row['RowGroup']=df_row.level_0.apply(lambda x:x.split('|')[0])
    df_row['Secreted']=df_row.level_0.map(secreted_dict)
    df_row.set_index('level_0',inplace=True)
    
    df_col=corr_mat['level_1'].drop_duplicates().to_frame()
    df_col['Type']='Target'
    df_col.loc[df_col['level_1'].isin(source_cells),'Type']='Source'
    df_col.set_index('level_1',inplace=True)


    from PyComplexHeatmap import DotClustermapPlotter,HeatmapAnnotation,anno_simple,anno_label,AnnotationBase
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize) 
    else:
        ax=ax
    
    
    
    col_ha = HeatmapAnnotation(Type=anno_simple(df_col.Type,cmap='Set2',
                                              #colors=list(set([type_color_all[i] for i in  df_col.Source.drop_duplicates()])),
                                              text_kws={'color': 'black', 'rotation': 0,'fontsize':8},
                                              legend=True,add_text=False),
                           verbose=0,label_side='left',
                           label_kws={'horizontalalignment':'right','fontsize':10})
    if plot_secret==True:
        row_ha = HeatmapAnnotation(Source=anno_simple(df_row.Secreted,cmap='Set1',
                                              #colors=list(set([type_color_all[i] for i in  df_col.Source.drop_duplicates()])),
                                              text_kws={'color': 'black', 'rotation': 0,'fontsize':8},
                                              legend=True,add_text=False),
                           verbose=0,#label_side='top',
                           axis=0,
                           label_kws={'horizontalalignment':'left','fontsize':0})
    else:
        row_ha=None
    
    cm = DotClustermapPlotter(corr_mat,x='level_1',y='level_0',value='means',
              c='means',s='means',#cmap='Reds',
              vmin=0,
              hue='Type',
              top_annotation=col_ha,left_annotation=row_ha,
              row_dendrogram=True,
              cmap=cmap,
              col_split=df_col.Type,col_split_gap=1,
              show_rownames=True,show_colnames=True,)
    cm.ax_heatmap.grid(which='minor',color='gray',linestyle='--')
    tesr=plt.gcf().axes
    for ax in plt.gcf().axes:
        if hasattr(ax, 'get_ylabel'):
            if ax.get_ylabel() == 'means':  # 假设 colorbar 有一个特定的标签
                cbar = ax
                cbar.tick_params(labelsize=fontsize)
                cbar.set_ylabel('means',fontsize=fontsize)
        ax.grid(False)
    if plot_secret==True:
        tesr[8].set_xticklabels(tesr[8].get_xticklabels(),fontsize=fontsize)
        tesr[9].set_xticklabels(tesr[9].get_xticklabels(),fontsize=fontsize)
        tesr[9].set_yticklabels(tesr[9].get_yticklabels(),fontsize=fontsize)
    else:
        tesr[6].set_xticklabels(tesr[6].get_xticklabels(),fontsize=fontsize)
        tesr[7].set_xticklabels(tesr[7].get_xticklabels(),fontsize=fontsize)
        tesr[7].set_yticklabels(tesr[7].get_yticklabels(),fontsize=fontsize)
    if return_table==True:
        return cor
    else:
        return ax
    




def cpdb_interacting_network(adata,
                             celltype_key,
                             means,
                             source_cells,
                             target_cells,
                             means_min=1,
                             means_sum_min=1,        
                             nodecolor_dict=None,
                             ax=None,
                             figsize=(6,6),
                             fontsize=10,
                             return_graph=False):
    r"""
    Create a detailed network showing ligand-receptor interactions between cell types.
    
    Args:
        adata: Annotated data object with cell type information
        celltype_key: Column name for cell type annotation
        means: CellPhoneDB means DataFrame with interaction data
        source_cells: List of source cell types
        target_cells: List of target cell types
        means_min: Minimum interaction strength threshold (1)
        means_sum_min: Minimum sum threshold for individual interactions (1)
        nodecolor_dict: Custom color mapping for cell types (None, uses default)
        ax: Existing matplotlib axes object (None)
        figsize: Figure dimensions as (width, height) ((6,6))
        fontsize: Font size for node labels (10)
        return_graph: Whether to return NetworkX graph object (False)
        
    Returns:
        ax: matplotlib.axes.Axes object or NetworkX graph if return_graph=True
    """
    # Determine node colors based on provided dictionary or defaults from adata
    from adjustText import adjust_text
    import re
    import networkx as nx
    
    if nodecolor_dict:
        type_color_all = nodecolor_dict
    else:
        color_key = f"{celltype_key}_colors"
        categories = adata.obs[celltype_key].cat.categories
        if color_key in adata.uns:
            type_color_all = dict(zip(categories, adata.uns[color_key]))
        else:
            palette = palette_112 if len(categories) > 28 else palette_28
            type_color_all = dict(zip(categories, palette))

    # Create a directed graph
    G = nx.DiGraph()

    # Filter the means DataFrame based on target and source cells
    sub_means = cpdb_exact_target(means, target_cells)
    sub_means = cpdb_exact_source(sub_means, source_cells)
    
    # Remove rows with null values in 'gene_a' or 'gene_b'
    sub_means = sub_means.dropna(subset=['gene_a', 'gene_b'])

    # Initialize a dictionary to store interactions
    nx_dict = {}

    for source_cell in source_cells:
        for target_cell in target_cells:
            key = f"{source_cell}|{target_cell}"
            nx_dict[key] = []

            # Find receptors related to the source and target cells
            escaped_str = re.escape(key)
            receptor_names = sub_means.columns[sub_means.columns.str.contains(escaped_str)].tolist()
            receptor_sub = sub_means[sub_means.columns[:10].tolist() + receptor_names]

            # Filter interactions based on the threshold values
            for j in receptor_sub.index:
                if receptor_sub.loc[j, receptor_names].sum() > means_min:
                    for rece in receptor_names:
                        if receptor_sub.loc[j, rece] > means_sum_min:
                            nx_dict[key].append(receptor_sub.loc[j, 'gene_b'])
                            G.add_edge(source_cell, f'L:{receptor_sub.loc[j, "gene_a"]}')
                            G.add_edge(f'L:{receptor_sub.loc[j, "gene_a"]}', f'R:{receptor_sub.loc[j, "gene_b"]}')
                            G.add_edge(f'R:{receptor_sub.loc[j, "gene_b"]}', rece.split('|')[1])

            # Remove duplicate interactions
            nx_dict[key] = list(set(nx_dict[key]))

    # Set colors for ligand and receptor nodes
    color_dict = type_color_all
    color_dict['ligand'] = '#a51616'
    color_dict['receptor'] = '#c2c2c2'

    # Assign colors to nodes based on their type
    node_colors = [
        color_dict.get(node, 
                       color_dict['ligand'] if 'L:' in node 
                       else color_dict['receptor'])
        for node in G.nodes()
    ]

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Use graphviz layout for positioning nodes
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi")

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edge_color='#c2c2c2')

    # Add labels to the nodes
    texts = [
        ax.text(pos[node][0], pos[node][1], node,
                fontdict={'size': fontsize, 'weight': 'bold', 'color': 'black'})
        for node in G.nodes() if 'ENSG' not in node
    ]
    adjust_text(texts, only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                arrowprops=dict(arrowstyle='-', color='black'))

    # Remove axes
    ax.axis("off")
    if return_graph:
        return G
    else:
        return ax
    



def curved_line(x0, y0, x1, y1, eps=0.8, pointn=30):
    r"""
    Generate points for a curved line between two coordinates using Bezier curves.
    
    Args:
        x0: Starting x coordinate
        y0: Starting y coordinate
        x1: Ending x coordinate
        y1: Ending y coordinate
        eps: Curve control parameter (0.8)
        pointn: Number of points along the curve (30)
        
    Returns:
        x: Array of x coordinates along the curve
        y: Array of y coordinates along the curve
    """
    import bezier
    x2 = (x0 + x1) / 2.0 + 0.1 ** (eps + abs(x0 - x1)) * (-1) ** (random.randint(1, 4))
    y2 = (y0 + y1) / 2.0 + 0.1 ** (eps + abs(y0 - y1)) * (-1) ** (random.randint(1, 4))
    nodes = np.asfortranarray([
        [x0, x2, x1],
        [y0, y2, y1]
    ])
    curve = bezier.Curve(nodes, degree=2)
    s_vals = np.linspace(0.0, 1.0, pointn)
    data = curve.evaluate_multi(s_vals)
    x = data[0]
    y = data[1]
    return x, y

def curved_graph(_graph, pos=None, eps=0.2, pointn=30, 
                 linewidth=2, alpha=0.3, color_dict=None):
    r"""
    Draw a network graph with curved edges and arrows.
    
    Args:
        _graph: NetworkX graph object
        pos: Node position dictionary (None)
        eps: Curve control parameter (0.2)
        pointn: Number of points along each curve (30)
        linewidth: Width of edge lines (2)
        alpha: Transparency of edges (0.3)
        color_dict: Color mapping for edges (None)
        
    Returns:
        None: Draws on current matplotlib axes
    """
    ax = plt.gca()
    for u, v in _graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        x, y = curved_line(x0, y0, x1, y1, eps=eps, pointn=pointn)
        
        # 使用颜色字典为边着色
        color = color_dict[u] if color_dict and u in color_dict else 'k'
        
        # 绘制曲线部分
        segments = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([segments[:-1], segments[1:]], axis=1)
        lc = LineCollection(segments, linewidths=linewidth, alpha=alpha, color=color)
        ax.add_collection(lc)
        
        # 在曲线的最后添加箭头
        arrow = FancyArrowPatch((x[-2], y[-2]), (x[-1], y[-1]),
                                connectionstyle="arc3,rad=0.2",
                                arrowstyle='-|>', mutation_scale=10,
                                color=color, linewidth=linewidth, alpha=1)
        ax.add_patch(arrow)
        
    ax.autoscale_view()


def plot_curve_network(G: nx.Graph, G_type_dict: dict, G_color_dict: dict, pos_type: str = 'spring', pos_dim: int = 2,
                       figsize: tuple = (4, 4), pos_scale: int = 10, pos_k=None, pos_alpha: float = 0.4,
                       node_size: int = 50, node_alpha: float = 0.6, node_linewidths: int = 1,
                       plot_node=None, plot_node_num: int = 20, node_shape=None,
                       label_verticalalignment: str = 'center_baseline', label_fontsize: int = 12,
                       label_fontfamily: str = 'Arial', label_fontweight: str = 'bold', label_bbox=None,
                       legend_bbox: tuple = (0.7, 0.05), legend_ncol: int = 3, legend_fontsize: int = 12,
                       legend_fontweight: str = 'bold', curve_awarg=None):
    r"""
    Create a network plot with curved edges and customizable node styling.
    
    Args:
        G: NetworkX graph object
        G_type_dict: Dictionary mapping nodes to types
        G_color_dict: Dictionary mapping nodes to colors
        pos_type: Layout algorithm - 'spring', 'kamada_kawai', or custom positions ('spring')
        pos_dim: Dimensionality for layout algorithm (2)
        figsize: Figure dimensions as (width, height) ((4, 4))
        pos_scale: Scale factor for node positions (10)
        pos_k: Optimal distance parameter for spring layout (None)
        pos_alpha: Transparency for position calculation (0.4)
        node_size: Base size for nodes (50)
        node_alpha: Transparency for nodes (0.6)
        node_linewidths: Width of node borders (1)
        plot_node: Specific nodes to label (None, uses top degree nodes)
        plot_node_num: Number of top nodes to label (20)
        node_shape: Dictionary mapping node types to shapes (None)
        label_verticalalignment: Vertical alignment for labels ('center_baseline')
        label_fontsize: Font size for node labels (12)
        label_fontfamily: Font family for labels ('Arial')
        label_fontweight: Font weight for labels ('bold')
        label_bbox: Bounding box properties for labels (None)
        legend_bbox: Legend position as (x, y) ((0.7, 0.05))
        legend_ncol: Number of legend columns (3)
        legend_fontsize: Legend font size (12)
        legend_fontweight: Legend font weight ('bold')
        curve_awarg: Additional arguments for curved edges (None)
        
    Returns:
        fig: matplotlib.figure.Figure object
        ax: matplotlib.axes.Axes object
    """
    from adjustText import adjust_text
    fig, ax = plt.subplots(figsize=figsize)

    # Determine node positions
    if pos_type == 'spring':
        pos = nx.spring_layout(G, scale=pos_scale, k=pos_k)
    elif pos_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, dim=pos_dim, scale=pos_scale)
    else:
        pos = pos_type

    degree_dict = dict(G.degree(G.nodes()))

    # Update color and type dictionaries
    G_color_dict = {node: G_color_dict[node] for node in G.nodes}
    G_type_dict = {node: G_type_dict[node] for node in G.nodes}

    # Draw nodes with different shapes if specified
    if node_shape is not None:
        for node_type, shape in node_shape.items():
            node_list = [node for node in G.nodes if G_type_dict[node] == node_type]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=node_list,
                node_size=[degree_dict[v] * node_size for v in node_list],
                node_color=[G_color_dict[v] for v in node_list],
                alpha=node_alpha,
                linewidths=node_linewidths,
                node_shape=shape  # Set node shape
            )
    else:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(G_color_dict.keys()),
            node_size=[degree_dict[v] * node_size for v in G],
            node_color=list(G_color_dict.values()),
            alpha=node_alpha,
            linewidths=node_linewidths,
        )

    # Determine hub genes
    if plot_node is not None:
        hub_gene = plot_node
    else:
        hub_gene = [i[0] for i in sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:plot_node_num]]

    # Draw curved edges if specified
    if curve_awarg is not None:
        curved_graph(G, pos, color_dict=G_color_dict, **curve_awarg)
    else:
        curved_graph(G, pos, color_dict=G_color_dict)

    pos1 = {i: pos[i] for i in hub_gene}

    # Add labels to hub genes
    texts = [ax.text(pos1[i][0], pos1[i][1], i, fontdict={'size': label_fontsize, 'weight': label_fontweight, 'color': 'black'})
             for i in hub_gene if 'ENSG' not in i]
    import adjustText
    if adjustText.__version__<='0.8':
        adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
    else:
        adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    arrowprops=dict(arrowstyle='->', color='red'))

    ax.axis("off")

    # Prepare legend
    t = pd.DataFrame(index=G_type_dict.keys())
    t['gene_type_dict'] = G_type_dict
    t['gene_color_dict'] = G_color_dict
    type_color_dict = {i: t.loc[t['gene_type_dict'] == i, 'gene_color_dict'].values[0] for i in t['gene_type_dict'].value_counts().index}

    patches = [mpatches.Patch(color=type_color_dict[i], label="{:s}".format(i)) for i in type_color_dict.keys()]

    #plt.legend(handles=patches, bbox_to_anchor=legend_bbox, ncol=legend_ncol, fontsize=legend_fontsize)
    #leg = plt.gca().get_legend()
    #ltext = leg.get_texts()
    #plt.setp(ltext, fontsize=legend_fontsize, fontweight=legend_fontweight)

    return fig, ax