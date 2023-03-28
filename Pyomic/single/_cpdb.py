r"""
The downanlysis of cellphonedb
"""

import ktplotspy as kpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.patches as mpatches

def cpdb_network_cal(adata,pvals,celltype_key):
    
    cpdb_dict=kpy.plot_cpdb_heatmap(
        adata = adata,
        pvals = pvals,
        celltype_key = celltype_key,
        figsize = (1,1),
        title = "",
        symmetrical = False,
        return_tables=True,
    )
    return cpdb_dict


def cpdb_plot_network(adata,interaction_edges,celltype_key,nodecolor_dict=None,
                      edgeswidth_scale=10,nodesize_scale=1,
                      pos_scale=1,pos_size=10,figsize=(5,5),title='',
                      legend_ncol=3,legend_bbox=(1,0.2),legend_fontsize=10,
                     return_graph=False):
    
    #set Digraph of cellphonedb
    G=nx.DiGraph()
    for i in interaction_edges.index:
        G.add_edge(interaction_edges.loc[i,'SOURCE'],
                   interaction_edges.loc[i,'TARGET'],
                   weight=interaction_edges.loc[i,'COUNT'],)
    
    #set celltypekey's color
    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))
    
    #set G_nodes_dict
    nodes=[]
    G_degree=dict(G.degree(G.nodes()))


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
        
    #plot
    fig, ax = plt.subplots(figsize=figsize) 
    pos = nx.spring_layout(G, scale=pos_scale, k=(pos_size)/np.sqrt(G.order()))
    p=dict(G.nodes)

    nodesize=np.array([G_nodes_dict[u] for u in G.nodes()])/nodesize_scale
    nodecolos=[type_color_all[u.split('_')[-1]] for u in G.nodes()]
    nx.draw_networkx_nodes(G, pos, nodelist=p,node_size=nodesize,node_color=nodecolos)

    edgewidth = np.array([G.get_edge_data(u, v)['weight'] for u, v in G.edges()])/edgeswidth_scale
    nx.draw_networkx_edges(G, pos,width=edgewidth)


    #label_options = {"ec": "white", "fc": "white", "alpha": 0.6}
    #nx.draw_networkx_labels(G, pos, font_size=10,) #bbox=label_options)
    plt.grid(False)
    plt.axis("off")
    plt.xlim(-2,2)
    plt.ylim(-2,1.5)

    labels = adata.obs[celltype_key].cat.categories
    #用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
    color = [type_color_all[u.split('_')[-1]] for u in labels]
    patches = [mpatches.Patch(color=type_color_all[u.split('_')[-1]], label=u) for u in labels ] 

    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.title(title)
    plt.legend(handles=patches,
               bbox_to_anchor=legend_bbox, 
               ncol=legend_ncol,
               fontsize=legend_fontsize)
    if return_graph==True:
        return G
    else:
        return ax
    #return {'Graph':G,'ax':ax}

def cpdb_plot_interaction(adata,cell_type1,cell_type2,means,pvals,celltype_key,genes=None,
                         keep_significant_only=True,figsize = (4,8),title="",
                         max_size=1,highlight_size = 0.75,standard_scale = True,cmap_name='viridis',
                         ytickslabel_fontsize=8,xtickslabel_fontsize=8,title_fontsize=10):
    
    fig=kpy.plot_cpdb(
        adata = adata,
        cell_type1 = cell_type1,
        cell_type2 = cell_type2, 
        means = means,
        pvals = pvals,
        celltype_key = celltype_key,
        genes = genes,
        keep_significant_only=keep_significant_only,
        figsize = figsize,
        title = "",
        max_size = max_size,
        highlight_size = highlight_size,
        standard_scale = standard_scale,
        cmap_name=cmap_name
    ).draw()
    
    #ytickslabels
    labels=fig.get_axes()[0].yaxis.get_ticklabels()
    plt.setp(labels, fontsize=ytickslabel_fontsize)

    #xtickslabels
    labels=fig.get_axes()[0].xaxis.get_ticklabels()
    plt.setp(labels, fontsize=xtickslabel_fontsize)

    fig.get_axes()[0].set_title(title,fontsize=title_fontsize)
    
    return fig.get_axes()[0]

def cpdb_interaction_filtered(adata,cell_type1,cell_type2,means,pvals,celltype_key,genes=None,
                         keep_significant_only=True,figsize = (0,0),title="",
                         max_size=1,highlight_size = 0.75,standard_scale = True,cmap_name='viridis',
                         ytickslabel_fontsize=8,xtickslabel_fontsize=8,title_fontsize=10):
    
    res=kpy.plot_cpdb(
        adata = adata,
        cell_type1 = cell_type1,
        cell_type2 = cell_type2, 
        means = means,
        pvals = pvals,
        celltype_key = celltype_key,
        genes = genes,
        keep_significant_only=keep_significant_only,
        figsize = figsize,
        title = "",
        max_size = max_size,
        highlight_size = highlight_size,
        standard_scale = standard_scale,
        cmap_name=cmap_name,
        return_table=True
    )

    return list(set(res['interaction_group']))