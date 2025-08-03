import json
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
import seaborn as sns


from scipy.stats import pearsonr, gaussian_kde
from sklearn.neighbors import KNeighborsRegressor



def get_z_umap(qz_m):
    import umap
    fit = umap.UMAP(n_neighbors=45, min_dist=0.5)
    u = fit.fit_transform(qz_m)
    return u


def plot_type_all(model, adata, proportions, figsize=(4, 4),dpi=80):
    u = get_z_umap(adata.obsm['qz_m'])
    qc_m = adata.obsm["qc_m"]
    group_c = np.argmax(qc_m,axis=1)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmaps = ['Blues','Greens','Reds','Oranges','Purples']
    for i in range(proportions.shape[1]):
        plt.scatter(u[group_c==i,0],u[group_c==i,1],s=1,c = qc_m[group_c==i,i], cmap=cmaps[i])

    # project the model's u on the umap
    knr = KNeighborsRegressor(10)
    knr.fit(adata.obsm["qz_m"], u)
    qu_umap = knr.predict(adata.uns['qu'])
    ax.scatter(*qu_umap.T, c='yellow', edgecolors='black',
               s=np.exp(model.qs_logm.cpu().detach()).sum(1)**1/2)

    plt.legend(proportions.columns,loc='right', bbox_to_anchor=(2.2,0.5),)
    plt.axis('off')
    return fig, ax
    
    
def get_corr_map(inference_outputs,  proportions,
                 dpi=80,figsize=(3,3)):
    qc_m_n = inference_outputs["qc_m"].detach().cpu().numpy()
    corr_map_qcm = np.zeros([qc_m_n.shape[1],qc_m_n.shape[1]])

    for i in range(corr_map_qcm.shape[0]):
        for j in range(corr_map_qcm.shape[0]):
            corr_map_qcm[i, j], _ = pearsonr(qc_m_n[:,i], proportions.iloc[:, j])
              

    plt.figure(dpi=dpi,figsize=figsize)
    ax = sns.heatmap(corr_map_qcm.T, annot=True,
                     cmap='RdBu_r',vmax=1,vmin=-1,
                     cbar_kws={'label': 'Cell type proportion corr.'}
                    )
    plt.xticks(np.array(range(qc_m_n.shape[1]))+0.5,labels=proportions.columns,rotation=90)
    plt.yticks(np.array(range(qc_m_n.shape[1]))+0.5,labels=proportions.columns,rotation=0)
    plt.xlabel('Estimated proportion')
    plt.ylabel('Ground truth proportion')
    return ax



def display_reconst(
    df_true,
    df_pred,
    density=False,
    marker_genes=None,
    sample_rate=0.1,
    size=(3, 3),
    spot_size=1,
    title=None,
    x_label='',
    y_label='',
    x_min=0,
    x_max=10,
    y_min=0,
    y_max=10,
    dpi=80,
):
    """
    Scatter plot - raw gexp vs. reconstructed gexp
    """
    assert 0 < sample_rate <= 1, \
        "Invalid downsampling rate for reconstruct scatter plot: {}".format(sample_rate)

    if marker_genes is not None:
        marker_genes = set(marker_genes)

    df_true_sample = df_true.sample(frac=sample_rate, random_state=0)
    df_pred_sample = df_pred.loc[df_true_sample.index]

    plt.rcParams["figure.figsize"] = size
    plt.figure(dpi=dpi)
    ax = plt.gca()

    xx = df_true_sample.T.to_numpy().flatten()
    yy = df_pred_sample.T.to_numpy().flatten()

    if density:
        for gene in df_true_sample.columns:
            try:
                gene_true = df_true_sample[gene].values
                gene_pred = df_pred_sample[gene].values
                gexp_stacked = np.vstack([df_true_sample[gene].values, df_pred_sample[gene].values])

                z = gaussian_kde(gexp_stacked)(gexp_stacked)
                ax.scatter(gene_true, gene_pred, c=z, s=spot_size, alpha=0.5)
            except np.linalg.LinAlgError as e:
                pass

    elif marker_genes is not None:
        color_dict = {True: 'red', False: 'green'}
        gene_colors = np.vectorize(
            lambda x: color_dict[x in marker_genes]
        )(df_true_sample.columns)
        colors = np.repeat(gene_colors, df_true_sample.shape[0])

        ax.scatter(xx, yy, c=colors, s=spot_size, alpha=0.5)

    else:
        ax.scatter(xx, yy, s=spot_size, alpha=0.5)

    min_val = min(xx.min(), yy.min())
    max_val = max(xx.max(), yy.max())
    #ax.set_xlim(min_val, 400)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.set_ylim(min_val, 400)

    plt.suptitle(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axis('equal')
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return ax
    

def gene_mean_vs_inferred_prop(inference_outputs, visium_args,idx,
                                dpi=80,figsize=(2,2)):
    
    sig_mean_n_df = pd.DataFrame(
    np.array(visium_args.sig_mean_norm)/(np.sum(np.array(visium_args.sig_mean_norm),axis=1,keepdims=True)+1e-5),
    columns=visium_args.sig_mean_norm.columns,
    index=visium_args.sig_mean_norm.index
    )

    qc_m = inference_outputs["qc_m"].detach().cpu().numpy()
    
    figs,ax=plt.subplots(1,1,dpi=dpi,figsize=figsize)
    
    v1 = sig_mean_n_df.iloc[:,idx].values
    v2 = qc_m[:,idx]
    
    v_stacked = np.vstack([v1, v2])
    den = gaussian_kde(v_stacked)(v_stacked)
    
    ax.scatter(v1,v2,c=den,s=1,cmap='jet',vmax=den.max()/3)
    
    ax.set_aspect('equal')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axis('equal')
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.title(visium_args.gene_sig.columns[idx])
    plt.xlim([v1.min()-0.1,v1.max()+0.1])
    plt.ylim([v2.min()-0.1,v2.max()+0.1])
    #plt.xticks(np.arange(0,1.1,0.5))
    #plt.yticks(np.arange(0,1.1,0.5))
    plt.xlabel('Gene signature mean')
    plt.ylabel('Predicted proportions')
    return ax


def plot_stacked_prop(results, category_names,dpi=80,figsize=(2,2)):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.

    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('rainbow')(
        np.linspace(0.15, 0.85, data.shape[1]))
    #category_colors = np.array(['b','g','r','oragne','purple'])
    fig, ax = plt.subplots(figsize=figsize,dpi=dpi)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.6,label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'black' #if r * g * b < 0.5 else 'darkgrey'
        #for y, (x, c) in enumerate(zip(xcenters, widths)):
        #    ax.text(x, y, str(round(c,2)), ha='center', va='center',
        #            color=text_color)
    #ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
    #          loc='right', fontsize='small')
    ax.legend(category_names,loc='right',bbox_to_anchor=(2, 0.5))
    return fig, ax


def plot_density(results, category_names,
                 dpi=80,figsize=(2,2)):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.

    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    category_colors = plt.get_cmap('RdBu_r')(
        np.linspace(0.15, 0.85, data.shape[1]))
    fig, ax = plt.subplots(figsize=figsize,dpi=dpi)
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = 0
        ax.barh(labels, widths, left=starts, height=0.6,label=colname, color=color)

        r, g, b, _ = color
    return fig, ax


def get_factor_dist(sample_ids,file_path):
    qc_p_dist = {}
    # Opening JSON file
    for sample_id in sample_ids:
        print(sample_id)
        f = open(file_path+sample_id+'_factor.json','r')
        data = json.load(f)   
        qc_p_dist[sample_id] = data['qc_m']
        f.close()
    return qc_p_dist


def get_adata(sample_ids, data_folder):
    adata_sample_all = []
    map_info_all = []
    adata_image_all = []
    for sample_id in sample_ids:
        print('loading...',sample_id)
        if (sample_id.startswith('MBC'))|(sample_id.startswith('CT')):

            adata_sample = sc.read_visium(path=os.path.join(data_folder, sample_id),library_id =  sample_id)
            adata_sample.var_names_make_unique()
            adata_sample.obs['sample']=sample_id
            adata_sample.obs['sample_type']='MBC'
            #adata_sample.obs_names  = adata_sample.obs_names+'-'+sample_id
            #adata_sample.obs_names  = adata_sample.obs_names+'_'+sample_id
            if '_index' in adata_sample.var.columns:
                adata_sample.var_names=adata_sample.var['_index']
        
        else:
            adata_sample = sc.read_h5ad(os.path.join(data_folder,sample_id, sample_id+'.h5ad'))
            adata_sample.var_names_make_unique()
            adata_sample.obs['sample']=sample_id
            adata_sample.obs['sample_type']='TNBC'
            #adata_sample.obs_names  = adata_sample.obs_names+'-'+sample_id
            #adata_sample.obs_names  = adata_sample.obs_names+'_'+sample_id
            if '_index' in adata_sample.var.columns:
                adata_sample.var_names=adata_sample.var['_index']
        from .utils import get_simu_map_info,preprocess_img
        if data_folder =='simu_data':
            
            map_info = get_simu_map_info(umap_df)
        else:
            adata_image,map_info = preprocess_img(data_folder,sample_id,adata_sample.obs.index,hchannal=False)
        
        adata_sample.obs_names  = adata_sample.obs_names+'-'+sample_id
        map_info.index = map_info.index+'-'+sample_id
        adata_sample_all.append(adata_sample)  
        map_info_all.append(map_info)  
        adata_image_all.append(adata_image)  
    return adata_sample_all,map_info_all,adata_image_all

def get_Moran(W, X):
    N = W.shape[0]
    term1 = N / W.sum().sum()
    x_m = X.mean()
    term2 = np.matmul(np.matmul(np.diag(X-x_m),W),np.diag(X-x_m))
    term3 = term2.sum().sum()
    term4 = ((X-x_m)**2).sum()
    term5 = term1 * term3 / term4
    return term5

def get_LISA(W, X):
    lisa_score = np.zeros(X.shape)
    N = W.shape[0]
    x_m = X.mean()
    term1 = X-x_m 
    term2 = ((X-x_m)**2).sum()
    for i in range(term1.shape[0]):
        #term3 = np.zeros(X.shape)
        term3 = (W[i,:]*(X-x_m)).sum()
        #for j in range(W.shape[0]):
        #    term3[j]=W[i,j]*(X[j]-x_m)
        #term3 = term3.sum()
        lisa_score[i] = np.sign(X[i]-x_m) * N * (X[i]-x_m) * term3 / term2
        #lisa_score[i] =   N * (X[i]-x_m) * term3 / term2
        
    return lisa_score

def get_SCI(W, X, Y):
    
    N = W.shape[0]
    term1 = N / (2*W.sum().sum())

    x_m = X.mean()
    y_m = Y.mean()
    term2 = np.matmul(np.matmul(np.diag(X-x_m),W),np.diag(Y-y_m))
    term3 = term2.sum().sum()

    term4 = np.sqrt(((X-x_m)**2).sum()) * np.sqrt(((Y-y_m)**2).sum())

    term5 = term1 * term3 / term4

    return term5

def get_cormtx(sample_id, hub_num ):
    # TODO: get_cormtx
    prop_i = proportions_df[ids_df['source']==sample_id][cluster_df['cluster']==hub_num]
    loc_i = np.array(map_info_all.loc[prop_i.index].loc[:,['array_col','array_row',]])
    W = np.zeros([loc_i.shape[0],loc_i.shape[0]])

    cor_matrix = np.zeros([gene_sig.shape[1],gene_sig.shape[1]])
    for i in range(loc_i.shape[0]):
        for j in range(i,loc_i.shape[0]):
            if np.sqrt((loc_i[i,0]-loc_i[j,0])**2+(loc_i[i,1]-loc_i[j,1])**2)<=3:
                W[i,j] = 1
                W[j,i] = 1
        #indices = vor.regions[vor.point_region[i]]
        #neighbor_i = np.concatenate([vor.ridge_points[np.where(vor.ridge_points[:,0] == i)],np.flip(vor.ridge_points[np.where(vor.ridge_points[:,1] == i)],axis=1)],axis=0)[:,1]
        #W[i,neighbor_i]=1
        #W[neighbor_i,i]=1
    print('spots in hub ',hub_num, '= ',prop_i.shape[0])
    if prop_i.shape[0]>1:
        for i in range(gene_sig.shape[1]):
            for j in range(i+1,gene_sig.shape[1]):
                    cor_matrix[i,j]=get_SCI(W, np.array(prop_i.iloc[:,i]), np.array(prop_i.iloc[:,j]))
                    cor_matrix[j,i]=cor_matrix[i,j]
    return cor_matrix

def get_hub_cormtx(sample_ids, hub_num):
    cor_matrix = np.zeros([gene_sig.shape[1],gene_sig.shape[1]])
    for sample_id in sample_ids:
        print(sample_id)
        cor_matrix = cor_matrix + get_cormtx(sample_id = sample_id, hub_num=hub_num)
        #print(cor_matrix)
    cor_matrix = cor_matrix/len(sample_ids)
    #cor_matrix = pd.DataFrame(cor_matrix)
    return cor_matrix

def create_corr_network_5(G, node_size_list,corr_direction, min_correlation,
                          dpi=80,figsize=(2,2)):
    ##Creates a copy of the graph
    H = G.copy()
    
    ##Checks all the edges and removes some based on corr_direction
    for stock1, stock2, weight in G.edges(data=True):
        #print(weight)
        ##if we only want to see the positive correlations we then delete the edges with weight smaller than 0        
        if corr_direction == "positive":
            ####it adds a minimum value for correlation. 
            ####If correlation weaker than the min, then it deletes the edge
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                H.remove_edge(stock1, stock2)
        ##this part runs if the corr_direction is negative and removes edges with weights equal or largen than 0
        else:
            ####it adds a minimum value for correlation. 
            ####If correlation weaker than the min, then it deletes the edge
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)
                
    
    #crates a list for edges and for the weights
    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())
    
    
    ### increases the value of weights, so that they are more visible in the graph
    #weights = tuple([(0.5+abs(x))**1 for x in weights])
    weights = tuple([x*2 for x in weights])
    #print(len(weights))
    #####calculates the degree of each node
    d = nx.degree(H)
    #print(d)
    #####creates list of nodes and a list their degrees that will be used later for their sizes
    nodelist, node_sizes = zip(*dict(d).items())
    #import sys, networkx as nx, matplotlib.pyplot as plt

    # Create a list of 10 nodes numbered [0, 9]
    #nodes = range(10)
    node_sizes = []
    labels = {}
    for n in nodelist:
            node_sizes.append( node_size_list[n] )
            labels[n] = 1 * n

    # Node sizes: [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    # Connect each node to its successor
    #edges = [ (i, i+1) for i in range(len(nodes)-1) ]

    # Create the graph and draw it with the node labels
    #g = nx.Graph()
    #g.add_nodes_from(nodes)
    #g.add_edges_from(edges)

    #nx.draw_random(g, node_size = node_sizes, labels=labels, with_labels=True)    
    #plt.show()

    #positions
    positions=nx.circular_layout(H)
    #print(positions)
    
    #Figure size
    plt.figure(figsize=figsize,dpi=dpi)

    #draws nodes,
    #options = {"edgecolors": "tab:gray", "alpha": 0.9}
    nx.draw_networkx_nodes(H,positions,
                           #node_color='#DA70D6',
                           nodelist=nodelist,
                           #####the node size will be now based on its degree
                           node_color=_colors['leiden_colors'][hub_num],# 'lightgreen',#pink, 'lightblue',#'#FFACB7',lightgreen B19CD9。#FFACB7 brown
                           alpha = 0.8,
                           node_size=tuple([x**1 for x in node_sizes]),
                           #**options
                           )
    
    #Styling for labels
    nx.draw_networkx_labels(H, positions, font_size=4, 
                            font_family='sans-serif')
    
    ###edge colors based on weight direction
    if corr_direction == "positive":
        edge_colour = plt.cm.GnBu#PiYG_r#RdBu_r#Spectral_r#GnBu#RdPu#PuRd#Blues#PuRd#GnBu OrRd
    else:
        edge_colour = plt.cm.PuRd
        
    #draws the edges
    print(min(weights))
    print(max(weights))

    nx.draw_networkx_edges(H, positions, edgelist=edges,style='solid',
                          ###adds width=weights and edge_color = weights 
                          ###so that edges are based on the weight parameter 
                          ###edge_cmap is for the color scale based on the weight
                          ### edge_vmin and edge_vmax assign the min and max weights for the width
                          width=weights, edge_color = weights, edge_cmap = edge_colour,
                           edge_vmin = 0,#min(weights),#0.55,#min(weights), 
                           edge_vmax= 0.7,#max(weights),#0.6,#max(weights)
                           #edge_vmin = min(weights),#0.55,#min(weights), 
                           #edge_vmax= max(weights),#0.6,#max(weights)
                           )

    # displays the graph without axis
    plt.axis('off')
    #plt.legend(['r','r'])
    #saves image
    #plt.savefig("part5" + corr_direction + ".png", format="PNG")
    #plt.show() 
