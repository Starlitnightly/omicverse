import anndata
import pandas as pd
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from ..single import cpdb_exact_target,cpdb_exact_source

def cpdb_network(adata:anndata.AnnData,interaction_edges:pd.DataFrame,
                      celltype_key:str,nodecolor_dict=None,counts_min=50,
                       source_cells=None,target_cells=None,
                      edgeswidth_scale:int=1,nodesize_scale:int=1,
                      figsize:tuple=(4,4),title:str='',
                      fontsize:int=12,ax=None,
                     return_graph:bool=False):
    G=nx.DiGraph()
    for i in interaction_edges.index:
        if interaction_edges.loc[i,'COUNT']>counts_min:
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
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))

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
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))
    
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
                      figsize=(3,3),fontsize=11,rotate=False,
                      legend_kws={'fontsize':8,'bbox_to_anchor':(5, -0.5),'loc':'center left',},return_table=False,**kwargs):
    
    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))

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
    if pch.__version>'1.7':
        hue_arg=None:
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
        legend=True,
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
            legend=True,
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
                            return_table=False,):

    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))
    
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

    sub_p=pvalues
    sub_p.index=sub_p['interacting_pair']
    sub_p=sub_p.loc[cor.index,cor.columns]
    sub_p_mat=sub_p.stack().reset_index(name="pvalue")

    corr_mat = cor.stack().reset_index(name="means")
    corr_mat['-logp']=-np.log10(sub_p_mat['pvalue']+0.001)

    df_row=corr_mat['level_0'].drop_duplicates().to_frame()
    df_row['RowGroup']=df_row.level_0.apply(lambda x:x.split('|')[0])
    df_row['Secreted']=df_row.level_0.map(secreted_dict)
    df_row.set_index('level_0',inplace=True)

    df_col=corr_mat['level_1'].drop_duplicates().to_frame()
    df_col['Source']=df_col.level_1.apply(lambda x:x.split('|')[0])
    df_col['Target']=df_col.level_1.apply(lambda x:x.split('|')[1])
    df_col.set_index('level_1',inplace=True)


    from PyComplexHeatmap import DotClustermapPlotter,HeatmapAnnotation,anno_simple,anno_label,AnnotationBase
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize) 
    else:
        ax=ax
    
    
    
    col_ha = HeatmapAnnotation(Source=anno_simple(df_col.Source,
                                                  colors=list(set([type_color_all[i] for i in  df_col.Source.drop_duplicates()])),
                                                  text_kws={'color': 'black', 'rotation': 0,'fontsize':fontsize},
                                                  legend=True,add_text=False,),
                               Target=anno_simple(df_col.Target,
                                                  colors=list(set([type_color_all[i] for i in  df_col.Target.drop_duplicates()])),
                                                  text_kws={'color': 'black', 'rotation': 0,'fontsize':fontsize},
                                                  legend=True,add_text=False),
                               verbose=0,label_side='left',
                               label_kws={'horizontalalignment':'right','fontsize':fontsize})
    if plot_secret==True:
        row_ha = HeatmapAnnotation(Secreted=anno_simple(df_row.Secreted,cmap='Set1',
                                                  #colors=list(set([type_color_all[i] for i in  df_col.Source.drop_duplicates()])),
                                                  text_kws={'color': 'black', 'rotation': 0,'fontsize':8},
                                                  legend=True,add_text=False),
                               verbose=0,#label_side='top',
                               axis=0,
                               label_kws={'horizontalalignment':'left','fontsize':0})
    else:
        row_ha=None
    
    cm = DotClustermapPlotter(corr_mat,x='level_1',y='level_0',value='means',
                  c='means',s='-logp',cmap='Reds',vmin=0,
                  top_annotation=col_ha,left_annotation=row_ha,
                  row_dendrogram=True,
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
        tesr[8].set_yticklabels(tesr[8].get_yticklabels(),fontsize=fontsize)
    else:
        tesr[6].set_xticklabels(tesr[6].get_xticklabels(),fontsize=fontsize)
        tesr[6].set_yticklabels(tesr[6].get_yticklabels(),fontsize=fontsize)
   
    if return_table==True:
        return cor
    else:
        return ax


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

    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))
    
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
    """
    Creates and visualizes a network of cell-cell interactions.

    Parameters:
    adata : AnnData
        AnnData object containing cell type and associated data.
    celltype_key : str
        Column name for cell types.
    means : DataFrame
        DataFrame containing interaction strengths.
    source_cells : list
        List of source cell types.
    target_cells : list
        List of target cell types.
    means_min : float, optional
        Minimum threshold for interaction strength (default is 1).
    means_sum_min : float, optional
        Minimum threshold for the sum of individual interactions (default is 1).
    nodecolor_dict : dict, optional
        Dictionary mapping cell types to colors (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes object for the plot (default is None).
    figsize : tuple, optional
        Size of the figure (default is (6, 6)).
    fontsize : int, optional
        Font size for node labels (default is 10).

    Returns:
    ax : matplotlib.axes.Axes
        Axes object with the drawn network.
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
            palette = sc.pl.palettes.default_102 if len(categories) > 28 else sc.pl.palettes.zeileis_28
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
