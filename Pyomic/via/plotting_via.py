
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from scipy.stats import norm as normal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance
from scipy.sparse import csr_matrix, csgraph, find
import math
import pandas as pd
import numpy as np
from numpy import ndarray
from scipy.sparse import issparse, spmatrix
import hnswlib
import time
import matplotlib
import igraph as ig
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.cm as cm
import pygam as pg
from sklearn.preprocessing import normalize
from typing import Optional, Union
#from utils_via import *
from .utils_via import * #
import random
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

def geodesic_distance(data:ndarray, knn:int=10, root:int=0, mst_mode:bool=False, cluster_labels:ndarray=None):
    n_samples = data.shape[0]
    #make knn graph on low dimensional data "data"
    knn_struct = construct_knn_utils(data, knn=knn)
    neighbors, distances = knn_struct.knn_query(data, k=knn)
    msk = np.full_like(distances, True, dtype=np.bool_)
    #https://igraph.org/python/versions/0.10.1/tutorials/shortest_paths/shortest_paths.html
    # Remove self-loops
    msk &= (neighbors != np.arange(neighbors.shape[0])[:, np.newaxis])
    rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(neighbors)])[msk]
    cols = neighbors[msk]
    weights = distances[msk] #we keep the distances as the weights here will actually be edge distances
    result = csr_matrix((weights, (rows, cols)), shape=(len(neighbors), len(neighbors)), dtype=np.float32)

    if mst_mode:
        print(f'MST geodesic mode')
        from scipy.sparse.csgraph import minimum_spanning_tree
        MST_ = minimum_spanning_tree(result)
        result = result+MST_
    result.eliminate_zeros()
    sources, targets = result.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))

    G = ig.Graph(edgelist, edge_attrs={'weight': result.data.tolist()})
    if cluster_labels is not None:
        graph = ig.VertexClustering(G, membership=cluster_labels).cluster_graph(combine_edges='sum')

        graph = recompute_weights(graph, Counter(cluster_labels)) #returns csr matrix

        weights = graph.data / (np.std(graph.data))

        edges = list(zip(*graph.nonzero()))

        G= ig.Graph(edges, edge_attrs={'weight': weights})
        root = cluster_labels[root]
    #get shortest distance from root to each point
    geo_distance_list = []
    print(f'start computing shortest paths')
    for i in range(G.vcount()):
        if cluster_labels is None:
            if i%1000==0: print(f'{datetime.now()}\t{i} out of {n_samples} complete')
        shortest_path = G.get_shortest_paths(root,to=i, weights=G.es["weight"],  output="epath")

        if len(shortest_path[0]) > 0:
            # Add up the weights across all edges on the shortest path
            distance = 0
            for e in shortest_path[0]:
                distance += G.es[e]["weight"]
            geo_distance_list.append(distance)
            #print("Shortest weighted distance is: ", distance)
        else: geo_distance_list.append(0)
    return geo_distance_list

def corr_geodesic_distance_lowdim(embedding, knn=10, time_labels:list=[],root:int=0, saveto='/home/shobi/Trajectory/Datasets/geodesic_distance.csv', mst_mode:bool=False,cluster_labels:ndarray=None):

    geodesic_dist = geodesic_distance(embedding, knn=knn, root=root, mst_mode = mst_mode,cluster_labels=cluster_labels)
    df_ = pd.DataFrame()

    df_['true_time'] =time_labels
    if cluster_labels is not None:
        df_['cluster_labels'] = cluster_labels
        df_ = df_.sort_values(['cluster_labels'], ascending=True).groupby('cluster_labels').mean()
        print('df_groupby', df_.head())
    df_['geo'] = geodesic_dist
    df_['geo'] = df_['geo'].fillna(0)

    correlation = df_['geo'].corr(df_['true_time'])
    print(f'{datetime.now()}\tcorrelation geo 2d and true time, {correlation}')
    df_.to_csv(saveto)
    return correlation


def make_edgebundle_milestone(embedding:ndarray=None, sc_graph=None, via_object=None, sc_pt:list = None, initial_bandwidth=0.03, decay=0.7, n_milestones:int=None, milestone_labels:list=[], sc_labels_numeric:list=None, weighted:bool=True, global_visual_pruning:float=0.5, terminal_cluster_list:list=[], single_cell_lineage_prob:ndarray=None, random_state:int=0):
    '''
    Perform Edgebundling of edges in a milestone level to return a hammer bundle of milestone-level edges. This is more granular than the original parc-clusters but less granular than single-cell level and hence also less computationally expensive
    requires some type of embedding (n_samples x 2) to be available

    :param embedding: optional (not required if via_object is provided) embedding single cell. also looks nice when done on via_mds as more streamlined continuous diffused graph structure. Umap is a but "clustery"
    :param graph: optional (not required if via_object is provided) igraph single cell graph level
    :param via_object: via_object (best way to run this function by simply providing via_object)
    :param sc_graph: igraph graph set as the via attribute self.ig_full_graph (affinity graph)
    :param initial_bandwidth: increasing bw increases merging of minor edges
    :param decay: increasing decay increases merging of minor edges #https://datashader.org/user_guide/Networks.html
    :param milestone_labels: default list=[]. Usually autocomputed. but can provide as single-cell level labels (clusters, groups, which function as milestone groupings of the single cells)
    :param sc_labels_numeric: default is None which automatically chooses via_object's pseudotime or time_series_labels (when available). otherwise set to a list of numerical values representing some sequential/chronological information
    :param terminal_cluster_list: default list [] and automatically uses all terminal clusters. otherwise set to any of the terminal cluster numbers within a list
    :return: dictionary containing keys: hb_dict['hammerbundle'] = hb hammerbundle class with hb.x and hb.y containing the coords
                hb_dict['milestone_embedding'] dataframe with 'x' and 'y' columns for each milestone and hb_dict['edges'] dataframe with columns ['source','target'] milestone for each each and ['cluster_pop'], hb_dict['sc_milestone_labels'] is a list of milestone label for each single cell

    '''
    if embedding is None:
        if via_object is not None: embedding = via_object.embedding

    if sc_graph is None:
        if via_object is not None: sc_graph =via_object.ig_full_graph
    if embedding is None:
        if via_object is None:
            print(f'{datetime.now()}\tERROR: Please provide via_object')
            return
        else:
            print(f'{datetime.now()}\tWARNING: VIA will now autocompute an embedding. It would be better to precompute an embedding using embedding = via_umap() or via_mds() and setting this as the embedding attribute via_object = embedding.')
            embedding = via_mds(via_object=via_object, random_seed=random_state)
    n_samples = embedding.shape[0]
    if n_milestones is None:
        n_milestones = min(n_samples,max(100, int(0.01*n_samples)))
    #milestone_indices = random.sample(range(n_samples), n_milestones)  # this is sampling without replacement
    if len(milestone_labels)==0:
        print(f'{datetime.now()}\tStart finding milestones')
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_milestones, random_state=random_state).fit(embedding)
        milestone_labels = kmeans.labels_.flatten().tolist()
        print(f'{datetime.now()}\tEnd milestones')
        #plt.scatter(embedding[:, 0], embedding[:, 1], c=milestone_labels, cmap='tab20', s=1, alpha=0.3)
        #plt.show()
    if sc_labels_numeric is None:
        if via_object is not None:
            sc_labels_numeric = via_object.time_series_labels
        else: print(f'{datetime.now()}\tWill use via-pseudotime for edges, otherwise consider providing a list of numeric labels (single cell level) or via_object')
    if sc_pt is None:
        sc_pt =via_object.single_cell_pt_markov
    '''
    numeric_val_of_milestone = []
    if len(sc_labels_numeric)>0:
        for cluster_i in set(milestone_labels):
            loc_cluster_i = np.where(np.asarray(milestone_labels)==cluster_i)[0]
            majority_ = func_mode(list(np.asarray(sc_labels_numeric)[loc_cluster_i]))
            numeric_val_of_milestone.append(majority_)
    '''
    vertex_milestone_graph = ig.VertexClustering(sc_graph, membership=milestone_labels).cluster_graph(combine_edges='sum')

    print(f'{datetime.now()}\tRecompute weights')
    vertex_milestone_graph = recompute_weights(vertex_milestone_graph, Counter(milestone_labels))
    print(f'{datetime.now()}\tpruning milestone graph based on recomputed weights')
    #was at 0.1 global_pruning for 2000+ milestones
    edgeweights_pruned_milestoneclustergraph, edges_pruned_milestoneclustergraph, comp_labels = pruning_clustergraph(vertex_milestone_graph,
                                                                                                   global_pruning_std=global_visual_pruning,
                                                                                                   preserve_disconnected=True,
                                                                                                   preserve_disconnected_after_pruning=False, do_max_outgoing=False)

    print(f'{datetime.now()}\tregenerate igraph on pruned edges')
    vertex_milestone_graph = ig.Graph(edges_pruned_milestoneclustergraph,
                                edge_attrs={'weight': edgeweights_pruned_milestoneclustergraph}).simplify(combine_edges='sum')
    vertex_milestone_csrgraph = get_sparse_from_igraph(vertex_milestone_graph, weight_attr='weight')

    weights_for_layout = np.asarray(vertex_milestone_csrgraph.data)
    # clip weights to prevent distorted visual scale
    weights_for_layout = np.clip(weights_for_layout, np.percentile(weights_for_layout, 20),
                                 np.percentile(weights_for_layout,
                                               80))  # want to clip the weights used to get the layout
    #print('weights for layout', (weights_for_layout))
    #print('weights for layout std', np.std(weights_for_layout))

    weights_for_layout = weights_for_layout/np.std(weights_for_layout)
    #print('weights for layout post-std', weights_for_layout)
    #print(f'{datetime.now()}\tregenerate igraph after clipping')
    vertex_milestone_graph = ig.Graph(list(zip(*vertex_milestone_csrgraph.nonzero())), edge_attrs={'weight': list(weights_for_layout)})

    #layout = vertex_milestone_graph.layout_fruchterman_reingold()
    #embedding = np.asarray(layout.coords)

    #print(f'{datetime.now()}\tmake node dataframe')
    data_node = [node for node in range(embedding.shape[0])]
    nodes = pd.DataFrame(data_node, columns=['id'])
    nodes.set_index('id', inplace=True)
    nodes['x'] = embedding[:, 0]
    nodes['y'] = embedding[:, 1]
    nodes['pt'] = sc_pt
    if via_object is not None:
        terminal_cluster_list = via_object.terminal_clusters
        single_cell_lineage_prob = via_object.single_cell_bp#_rownormed does not make a huge difference whether or not rownorming is applied.
    if (len(terminal_cluster_list)>0) and (single_cell_lineage_prob is not None):
        for i, c_i in enumerate(terminal_cluster_list):
            nodes['sc_lineage_probability_'+str(c_i)] = single_cell_lineage_prob[:,i]
    if sc_labels_numeric is not None:
        print(f'{datetime.now()}\tSetting numeric label as time_series_labels or other sequential metadata for coloring edges')
        nodes['numeric label'] = sc_labels_numeric
    else:
        print(f'{datetime.now()}\tSetting numeric label as single cell pseudotime for coloring edges')
        nodes['numeric label'] = sc_pt

    nodes['kmeans'] = milestone_labels
    group_pop = []
    for i in sorted(set(milestone_labels)):
        group_pop.append(milestone_labels.count(i))

    nodes_mean = nodes.groupby('kmeans').mean()
    nodes_mean['cluster population'] = group_pop

    edges = pd.DataFrame([e.tuple for e in vertex_milestone_graph.es], columns=['source', 'target'])

    edges['weight0'] = vertex_milestone_graph.es['weight']
    edges = edges[edges['source'] != edges['target']]

    # seems to work better when allowing the bundling to occur on unweighted representation and later using length of segments to color code significance
    if weighted ==True: edges['weight'] = edges['weight0']#1  # [1/i for i in edges['weight0']]np.where((edges['source_cluster'] != edges['target_cluster']) , 1,0.1)#[1/i for i in edges['weight0']]#
    else: edges['weight'] = 1
    print(f'{datetime.now()}\tMaking smooth edges')
    hb = hammer_bundle(nodes_mean, edges, weight='weight', initial_bandwidth=initial_bandwidth,
                       decay=decay)  # default bw=0.05, dec=0.7
    # hb.x and hb.y contain all the x and y coords of the points that make up the edge lines.
    # each new line segment is separated by a nan value
    # https://datashader.org/_modules/datashader/bundling.html#hammer_bundle
    #nodes_mean contains the averaged 'x' and 'y' milestone locations based on the embedding
    hb_dict = {}
    hb_dict['hammerbundle'] = hb
    hb_dict['milestone_embedding'] = nodes_mean
    hb_dict['edges'] = edges[['source','target']]
    hb_dict['sc_milestone_labels'] = milestone_labels

    return hb_dict

def plot_gene_trend_heatmaps(via_object, df_gene_exp:pd.DataFrame, marker_lineages:list = [], 
                             fontsize:int=8,cmap:str='viridis', normalize:bool=True, ytick_labelrotation:int = 0, 
                             fig_width: int = 7):
    '''

    Plot the gene trends on heatmap: a heatmap is generated for each lineage (identified by terminal cluster number). Default selects all lineages

    :param via_object:
    :param df_gene_exp: pandas DataFrame single-cell level expression [cells x genes]
    :param marker_lineages: list default = None and plots all detected all lineages. Optionally provide a list of integers corresponding to the cluster number of terminal cell fates
    :param fontsize: int default = 8
    :param cmap: str default = 'viridis'
    :param normalize: bool = True
    :param ytick_labelrotation: int default = 0
    :return: fig and list of axes
    '''
    import seaborn as sns

    if len(marker_lineages) ==0: marker_lineages = via_object.terminal_clusters
    dict_trends = get_gene_trend(via_object=via_object, marker_lineages=marker_lineages, df_gene_exp=df_gene_exp)
    branches = list(dict_trends.keys())
    genes = dict_trends[branches[0]]['trends'].index
    height = len(genes) * len(branches)
    # Standardize the matrix (standardization along each gene. Since SS function scales the columns, we first transpose the df)
    #  Set up plot
    fig = plt.figure(figsize=[fig_width, height])
    ax_list = []
    for i, branch in enumerate(branches):
        ax = fig.add_subplot(len(branches), 1, i + 1)
        df_trends=dict_trends[branch]['trends']
        # normalize each genes (feature)
        if normalize==True:
            df_trends = pd.DataFrame(
            StandardScaler().fit_transform(df_trends.T).T,
            index=df_trends.index,
            columns=df_trends.columns)

        ax.set_title('Lineage: ' + str(branch) + '-' + str(dict_trends[branch]['name']), fontsize=int(fontsize*1.3))
        #sns.set(size=fontsize)  # set fontsize 2
        b=sns.heatmap(df_trends,yticklabels=True, xticklabels=False, cmap = cmap)
        b.tick_params(labelsize=fontsize,labelrotation=ytick_labelrotation)
        b.figure.axes[-1].tick_params(labelsize=fontsize)
        ax_list.append(ax)
    b.set_xlabel("pseudotime", fontsize=int(fontsize*1.3))
    return fig, ax_list

def plot_scatter(embedding:ndarray, labels:list, cmap='rainbow', s=5, alpha=0.3, edgecolors='None',title:str='', text_labels:bool=True, color_dict=None, categorical:bool=None, via_object=None,sc_index_terminal_states:list=None,true_labels:list=[]):
    '''
    General scatter plotting tool for numeric and categorical labels on the single-cell level

    :param embedding: ndarray n_samples x 2
    :param labels: list single cell labels list of number or strings
    :param cmap: str default = 'rainbow'
    :param s: int size of scatter dot
    :param alpha: float with 0 transparent to 1 opaque default =0.3
    :param edgecolors:
    :param title: str
    :param text_labels: bool default =True
    :param categorical: bool default =True
    :param via_object:
    :param sc_index_terminal_states: list of integers corresponding to one cell in each of the terminal states
    :param color_dict: {'true_label_group_1': #COLOR,'true_label_group_2': #COLOR2,....} where the dictionary keys correspond to the provided labels
    :param true_labels: list of single cell labels used to annotate the terminal states
    :return: matplotlib pyplot fig, ax
    '''
    fig, ax = plt.subplots()
    if (isinstance(labels[0], str)) == True:
        categorical = True
    else:
        categorical = False
    ax.set_facecolor('white')
    if color_dict is not None:
        for key in color_dict:
            loc_key = np.where(np.asarray(labels) == key)[0]
            ax.scatter(embedding[loc_key, 0], embedding[loc_key, 1], color=color_dict[key], label=key, s=s,
                       alpha=alpha, edgecolors=edgecolors)
            x_mean = embedding[loc_key, 0].mean()
            y_mean = embedding[loc_key, 1].mean()
            if text_labels == True: ax.text(x_mean, y_mean, key, style='italic', fontsize=10, color="black")

    elif categorical==True:
        color_dict = {}
        for index, value in enumerate(set(labels)):
            color_dict[value] = index
        palette = cm.get_cmap(cmap, len(color_dict.keys()))
        cmap_ = palette(range(len(color_dict.keys())))

        for key in color_dict:
            loc_key = np.where(np.asarray(labels) == key)[0]

            ax.scatter(embedding[loc_key, 0], embedding[loc_key, 1], color=cmap_[color_dict[key]], label=key, s=s,
                       alpha=alpha, edgecolors=edgecolors)
            x_mean = embedding[loc_key, 0].mean()
            y_mean = embedding[loc_key, 1].mean()
            if text_labels==True: ax.text(x_mean, y_mean, key, style='italic', fontsize=10, color="black")
    else:
        im=ax.scatter(embedding[:, 0], embedding[:, 1], c=labels,
                        cmap=cmap, s=3, alpha=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical',
                   label='pseudotime')
        if via_object is not None:
            tsi_list = []
            for tsi in via_object.terminal_clusters:
                loc_i = np.where(np.asarray(via_object.labels) == tsi)[0]
                val_pt = [via_object.single_cell_pt_markov[i] for i in loc_i]

                th_pt = np.percentile(val_pt, 50)  # 50
                loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
                temp = np.mean(via_object.data[loc_i], axis=0)
                labelsq, distances = via_object.knn_struct.knn_query(temp, k=1)

                tsi_list.append(labelsq[0][0])

            ax.set_title('root:' + str(via_object.root_user[0]) + 'knn' + str(via_object.knn) + 'Ncomp' + str(via_object.ncomp))
            for i in tsi_list:
                # print(i, ' has traj and cell type', self.df_annot.loc[i, ['Main_trajectory', 'Main_cell_type']])
                ax.text(embedding[i, 0], embedding[i, 1], str(true_labels[i])+'_Cell'+str(i))
                ax.scatter(embedding[i, 0], embedding[i, 1], c='black', s=10)
    if (via_object is None) & (sc_index_terminal_states is not None):
        for i in sc_index_terminal_states:
            ax.text(embedding[i, 0], embedding[i, 1], str(true_labels[i])+'_Cell'+str(i))
            ax.scatter(embedding[i, 0], embedding[i, 1], c='black', s=10)

    if len(title)==0: ax.set_title(label='scatter plot', color= 'blue')
    else:    ax.set_title(label=title, color= 'blue')
    ax.legend(fontsize=6, frameon=False)
    ax.grid(False)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    # Hide grid lines
    ax.grid(False)
    fig.patch.set_visible(False)
    return fig, ax

def _make_knn_embeddedspace(embedding):
    # knn struct built in the embedded space to be used for drawing the lineage trajectories onto the 2D plot
    knn = hnswlib.Index(space='l2', dim=embedding.shape[1])
    knn.init_index(max_elements=embedding.shape[0], ef_construction=200, M=16)
    knn.add_items(embedding)
    knn.set_ef(50)
    return knn

def via_forcelayout(X_pca, viagraph_full: csr_matrix=None, k: int = 10,
             n_milestones=2000, time_series_labels: list = [],
            knn_seq: int = 5, saveto='', random_seed:int = 0) -> ndarray:
    '''
    Compute force directed layout. #TODO not complete

    :param X_pca:
    :param viagraph_full: optional. if calling before via, then None. if calling after or from within via, then we can use the via-graph to reinforce the layout
    :param k:
    :param random_seed:
    :param t_diffusion:
    :param n_milestones:
    :param time_series_labels:
    :param knn_seq:
    :return: ndarray
    '''
    # use the csr_full_graph from via and subsample it.
    # but this results in a very fragmented graph because the subsampling index is too small a fraction of the total number of possible edges.
    # only works if you take a high enough percentage of the original samples

    print(f"{datetime.now()}\tCommencing Force Layout")
    np.random.seed(random_seed)
    milestone_indices = random.sample(range(X_pca.shape[0]), n_milestones)  # this is sampling without replacement
    if viagraph_full is not None:

        milestone_knn = viagraph_full[milestone_indices]  #
        milestone_knn = milestone_knn[:, milestone_indices]
        milestone_knn = normalize(milestone_knn, axis=1)



    knn_struct = construct_knn_utils(X_pca[milestone_indices, :], knn=k)
    # we need to add the new knn (milestone_knn_new) built on the subsampled indices to ensure connectivity. o/w graph is fragmented if only relying on the subsampled graph
    if time_series_labels is None: time_series_labels = []
    if len(time_series_labels) >= 1: time_series_labels = np.array(time_series_labels)[milestone_indices].tolist()
    milestone_knn_new = affinity_milestone_knn(data=X_pca[milestone_indices, :], knn_struct=knn_struct, k=k,
                                               time_series_labels=time_series_labels, knn_seq=knn_seq)

    print('milestone knn new', milestone_knn_new.shape, milestone_knn_new.data[0:10])
    if viagraph_full is None:milestone_knn = milestone_knn_new
    else: milestone_knn = milestone_knn + milestone_knn_new
    print('final reinforced milestone knn', milestone_knn.shape, 'number of nonzero edges', len(milestone_knn.data))

    print('force layout')
    g_layout = ig.Graph(list(zip(*milestone_knn.nonzero())))  # , edge_attrs={'weight': weights_for_layout})
    layout = g_layout.layout_fruchterman_reingold()
    force_layout = np.asarray(layout.coords)
    #compute knn used to estimate the embedding values of the full sample set based on embedding values computed just for a milestone subset of the full sample
    neighbor_array, distance_array = knn_struct.knn_query(X_pca, k=k)
    print('shape of ', X_pca.shape, neighbor_array.shape)
    row_mean = np.mean(distance_array, axis=1)
    row_var = np.var(distance_array, axis=1)
    row_znormed_dist_array = -(distance_array - row_mean[:, np.newaxis]) / row_var[:, np.newaxis]
    # when k is very small, then you can get very large affinities due to var being ~0
    row_znormed_dist_array = np.nan_to_num(row_znormed_dist_array, copy=True, nan=1, posinf=1, neginf=1)
    row_znormed_dist_array[row_znormed_dist_array > 10] = 0
    affinity_array = np.exp(row_znormed_dist_array)
    affinity_array = normalize(affinity_array, norm='l1', axis=1)  # row stoch

    row_list = []
    n_neighbors = neighbor_array.shape[1]
    n_cells = neighbor_array.shape[0]
    print('ncells and neighs', n_cells, n_neighbors)

    row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))

    col_list = neighbor_array.flatten().tolist()

    list_affinity = affinity_array.flatten().tolist()

    csr_knn = csr_matrix((list_affinity, (row_list, col_list)),
                         shape=(n_cells, len(milestone_indices)))  # n_samples*n_milestones

    milestone_force = csr_matrix(force_layout)  ##TODO remove this we are just testing force layout
    full_force = csr_knn * milestone_force  # is a matrix
    full_force = np.asarray(full_force.todense())


    plt.scatter(full_force[:, 0].tolist(), full_force[:, 1].tolist(), s=1, alpha=0.3, c='red')
    plt.title('full mds')
    plt.show()
    full_force = np.reshape(full_force, (n_cells, 2))
    if len(saveto)>0:
        U_df = pd.DataFrame(full_force)
        U_df.to_csv(saveto)
    return full_force

def via_mds(via_object=None, X_pca:ndarray=None, viagraph_full: csr_matrix=None, k: int = 15,
            random_seed: int = 0, diffusion_op: int = 1, n_milestones=2000, time_series_labels: list = [],
            knn_seq: int = 5, k_project_milestones:int = 3, t_difference:int=2, saveto='', embedding_type:str='mds', double_diffusion:bool=False,neighbors_distances:ndarray=None) -> ndarray:
    '''

    Fast computation of a 2D embedding
    FOR EXAMPLE:
    v0.embedding = via.via_mds(via_object = v0)
    via.plot_scatter(embedding = v0.embedding, labels = v0.true_labels)

    :param via_object:
    :param X_pca: dimension reduced (only if via_object is not passed)
    :param viagraph_full: optional. if calling before or without via, then None and a milestone graph will be computed. if calling after or from within via, then we can use the via-graph to reinforce the layout of the milestone graph
    :param k: number of knn for the via_mds reinforcement graph on milestones. default =15. integers 5-20 are reasonable
    :param random_seed: randomseed integer
    :param t_diffusion: default integer value = 1 with higher values generate more smoothing
    :param n_milestones: number of milestones used to generate the initial embedding
    :param time_series_labels: numerical values in list form representing some sequentual information
    :param knn_seq: if time-series data is available, this will augment the knn with sequential neighbors (2-10 are reasonable values) default =5
    :param embedding_type: default = 'mds' or set to 'umap'
    :param double_diffusion: default is False. To achieve sharper strokes/lineages, set to True
    :param k_project_milestones: number of milestones in the milestone-knngraph used to compute the single-cell projection
    :param n_iterations: number of iterations to run
    :param neighbors_distances: array of distances of each neighbor for each cell (n_cells x knn) used when called from within via.run() for autocompute via-mds
    :return: numpy array of size n_samples x 2
    '''

    # use the csr_full_graph from via and subsample it.
    # but this results in a very fragmented graph because the subsampling index is too small a fraction of the total number of possible edges.
    # only works if you take a high enough percentage of the original samples
    #however, omitting the integration of csr_full_graph also compromises the ability of the embedding to better reflect the underlying trajectory in terms of global structure

    print(f"{datetime.now()}\tCommencing Via-MDS")
    if via_object is not None:
        if X_pca is None: X_pca = via_object.data
        if viagraph_full is None: viagraph_full=via_object.csr_full_graph
    n_samples = X_pca.shape[0]
    final_full_mds = np.zeros((n_samples, 2))



    if n_milestones is None:
        n_milestones = min(n_samples,max(2000, int(0.01*n_samples)))
    if n_milestones > n_samples:
        n_milestones = min(n_samples, max(2000, int(0.01 * n_samples)))
        print(f"{datetime.now()}\tResetting n_milestones to {n_milestones} as n_samples > original n_milestones")

    '''
    if n_milestones < n_samples:
        if via_object is not None: milestone_indices = density_sampling(neighbors_distances= via_object.full_neighbor_array, desired_samples = n_milestones)
        else: milestone_indices = density_sampling(neighbors_distances= neighbors_distances, desired_samples = n_milestones)
        print(f'number of milestone indices from density sampling {milestone_indices.shape}')
        print('exp=True, dens sampling')
    '''
    np.random.seed(random_seed)
    print('exp=True, randomsampling') #in the fourth run, exp is =False
    milestone_indices = random.sample(range(X_pca.shape[0]), n_milestones) #this is sampling without replacement

    if viagraph_full is not None:
        milestone_knn = viagraph_full[milestone_indices]  #
        milestone_knn = milestone_knn[:, milestone_indices]
        milestone_knn = normalize(milestone_knn, axis=1) #using these effectively emphasises the edges that are pass an even more stringent requirement on Nearest neighbors (since they are selected from the full set of cells, rather than a subset of milestones)

    knn_struct = construct_knn_utils(X_pca[milestone_indices, :], knn=k)
    # we need to add the new knn (milestone_knn_new) built on the subsampled indices to ensure connectivity. o/w graph is fragmented if only relying on the subsampled graph
    if time_series_labels is None: time_series_labels = []
    if len(time_series_labels) >= 1: time_series_labels = np.array(time_series_labels)[milestone_indices].tolist()
    milestone_knn_new = affinity_milestone_knn(data=X_pca[milestone_indices, :], knn_struct=knn_struct, k=k,
                                               time_series_labels=time_series_labels, knn_seq=knn_seq, t_difference=t_difference)


    if viagraph_full is None:milestone_knn = milestone_knn_new
    else: milestone_knn = milestone_knn + milestone_knn_new


    # build a knn to project the input n_samples based on milestone knn

    neighbor_array, distance_array = knn_struct.knn_query(X_pca, k=k_project_milestones) #[n_samples x n_milestones]

    row_mean = np.mean(distance_array, axis=1)
    row_var = np.var(distance_array, axis=1)
    row_znormed_dist_array = -(distance_array - row_mean[:, np.newaxis]) / row_var[:, np.newaxis]
    # when k is very small, then you can get very large affinities due to var being ~0
    row_znormed_dist_array = np.nan_to_num(row_znormed_dist_array, copy=True, nan=1, posinf=1, neginf=1)
    row_znormed_dist_array[row_znormed_dist_array > 10] = 0
    affinity_array = np.exp(row_znormed_dist_array)
    affinity_array = normalize(affinity_array, norm='l1', axis=1)  # row stoch

    row_list = []
    n_neighbors = neighbor_array.shape[1]
    n_cells = neighbor_array.shape[0]

    row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))

    col_list = neighbor_array.flatten().tolist()

    list_affinity = affinity_array.flatten().tolist()

    csr_knn = csr_matrix((list_affinity, (row_list, col_list)),
                         shape=(n_cells, len(milestone_indices)))  # n_samples*n_milestones
    print(f"{datetime.now()}\tStart computing with diffusion power:{diffusion_op}")
    '''
    r2w_input = pd.read_csv(
        '/home/shobi/Trajectory/Datasets/EB_Phate/RW2/pc20_knn100kseq50krev50RW2_sparse_matrix029_P1_Q10.csv')
    r2w_input = r2w_input.drop(['Unnamed: 0'], axis=1).values
    input = r2w_input[:, 0:30]
    input = input[milestone_indices, :]
    print('USING RW2 COMPS')
    '''

    if embedding_type == 'mds': milestone_mds = sgd_mds(via_graph=milestone_knn, X_pca=X_pca[milestone_indices, :], diff_op=diffusion_op, ndims=2,
                            random_seed=random_seed, double_diffusion=double_diffusion)  # returns an ndarray
    elif embedding_type =='umap': milestone_mds = via_umap(X_input=X_pca[milestone_indices, :], graph=milestone_knn)

    print(f"{datetime.now()}\tEnd computing mds with diffusion power:{diffusion_op}")
    #TESTING
    #plt.scatter(milestone_mds[:, 0], milestone_mds[:, 1], s=1)
    #plt.title('sampled')
    #plt.show()

    milestone_mds = csr_matrix(milestone_mds)

    full_mds = csr_knn * milestone_mds  # is a matrix
    full_mds = np.asarray(full_mds.todense())

    # TESTING
    #plt.scatter(full_mds[:, 0].tolist(), full_mds[:, 1].tolist(), s=1, alpha=0.3, c='green')
    #plt.title('full')
    #plt.show()
    full_mds = np.reshape(full_mds, (n_cells, 2))

    if len(saveto) > 0:
        U_df = pd.DataFrame(full_mds)
        U_df.to_csv(saveto)

    return full_mds

def via_umap(via_object = None, X_input: ndarray = None, graph:csr_matrix=None, n_components:int=2, alpha: float = 1.0, negative_sample_rate: int = 5,
                  gamma: float = 1.0, spread:float=1.0, min_dist:float=0.1, init_pos:Union[str, ndarray]='spectral', random_state:int=0,
                    n_epochs:int=100, distance_metric: str = 'euclidean', layout:Optional[list]=None, cluster_membership:Optional[list]=None, saveto='')-> ndarray:
    '''

    Run dimensionality reduction using the VIA modified HNSW graph

    :param via_object: if via_object is provided then X_input and graph are ignored
    :param X_input: ndarray nsamples x features (PCs)
    :param graph: csr_matrix of knngraph. This usually is via's pruned, sequentially augmented sc-knn graph accessed as an attribute of via v0.csr_full_graph
    :param n_components:
    :param alpha:
    :param negative_sample_rate:
    :param gamma: Weight to apply to negative samples.
    :param spread: The effective scale of embedded points. In combination with min_dist this determines how clustered/clumped the embedded points are.
    :param min_dist: The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points
    :param init_pos: either a string (default) 'spectral', 'via' (uses via graph to initialize). Or a n_cellx2 dimensional ndarray with initial coordinates
    :param random_state:
    :param n_epochs: The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. If 0 is specified a value will be selected based on the size of the input dataset (200 for large datasets, 500 for small).
    :param distance_metric:
    :param layout: ndarray This is required if the init_pos is set to 'via'. layout should then = via0.graph_node_pos (which is a list of lists, of length n_clusters)
    :param cluster_membership: v0.labels (cluster level labels of length n_samples corresponding to the layout)
    :return: ndarray of shape (nsamples,n_components)
    '''
    if via_object is None:
        if (X_input is None) or (graph is None):
            print(f"{datetime.now()}\tERROR: please provide both X_input and graph")
    if via_object is not None:
        X_input = via_object.data
        print('X-input', X_input.shape)
        graph = via_object.csr_full_graph
    #X_input = via0.data
    n_cells = X_input.shape[0]
    #graph = graph+graph.T
    #graph = via0.csr_full_graph
    print(f"{datetime.now()}\tComputing umap on sc-Viagraph")

    from umap.umap_ import find_ab_params, simplicial_set_embedding
    #graph is a csr matrix
    #weight all edges as 1 in order to prevent umap from pruning weaker edges away
    layout_array = np.zeros(shape=(n_cells,2))

    if (init_pos=='via') and (via_object is None):
        #list of lists [[x,y], [x1,y1], []]
        if (layout is None) or (cluster_membership is None): print('please provide via object or values for arguments: layout and cluster_membership')
        else:
            for i in range(n_cells):
                layout_array[i,0]=layout[cluster_membership[i]][0]
                layout_array[i, 1] = layout[cluster_membership[i]][1]
            init_pos = layout_array
            print(f'{datetime.now()}\tusing via cluster graph to initialize embedding')
    elif (init_pos == 'via') and (via_object is not None):
        layout = via_object.graph_node_pos
        cluster_membership = via_object.labels
        for i in range(n_cells):
            layout_array[i, 0] = layout[cluster_membership[i]][0]
            layout_array[i, 1] = layout[cluster_membership[i]][1]
        init_pos = layout_array
        print(f'{datetime.now()}\tusing via cluster graph to initialize embedding')
    a, b = find_ab_params(spread, min_dist)
    #print('a,b, spread, dist', a, b, spread, min_dist)
    t0 = time.time()
    #m = graph.data.max()
    graph.data = np.clip(graph.data, np.percentile(graph.data, 1),np.percentile(graph.data, 99))
    #graph.data = 1 + graph.data/m
    #graph.data.fill(1)
    #print('average graph.data', round(np.mean(graph.data),4), round(np.max(graph.data),2))
    #graph.data = graph.data + np.mean(graph.data)

    #transpose =graph.transpose()

    #prod_matrix = graph.multiply(transpose)
    #graph = graph + transpose - prod_matrix
    X_umap, aux_data = simplicial_set_embedding(data=X_input, graph=graph, n_components=n_components, initial_alpha=alpha,
                                      a=a, b=b, n_epochs=n_epochs, metric_kwds={}, gamma=gamma, metric=distance_metric,
                                      negative_sample_rate=negative_sample_rate, init=init_pos,
                                      random_state=np.random.RandomState(random_state),
                                      verbose=1, output_dens=False, densmap_kwds={}, densmap=False)
    if len(saveto)>0:
        U_df = pd.DataFrame(X_umap)
        U_df.to_csv(saveto)
    return X_umap


def run_umap_hnsw(via_object=None, X_input: ndarray = None, graph: csr_matrix = None, n_components: int = 2,
                  alpha: float = 1.0, negative_sample_rate: int = 5,
                  gamma: float = 1.0, spread: float = 1.0, min_dist: float = 0.1,
                  init_pos: Union[str, ndarray] = 'spectral', random_state: int = 0,
                  n_epochs: int = 0, distance_metric: str = 'euclidean', layout: Optional[list] = None,
                  cluster_membership: list = [], saveto='') -> ndarray:
    print(f"{datetime.now()}\tWarning: in future call via_umap() to run this function")

    return via_umap(via_object=via_object, X_input=X_input, graph=graph, n_components=n_components, alpha=alpha,
                    negative_sample_rate=negative_sample_rate,
                    gamma=gamma, spread=spread, min_dist=min_dist, init_pos=init_pos, random_state=random_state,
                    n_epochs=n_epochs, distance_metric=distance_metric, layout=layout,
                    cluster_membership=cluster_membership, saveto=saveto)

def draw_sc_lineage_probability(via_object, via_fine=None, figsize=(8,4),
                                embedding:ndarray=None, idx:list=None, 
                                cmap_name='plasma', dpi=80, scatter_size =None,
                                marker_lineages = [], fontsize:int=10):
    '''

    G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it's made on full sample
    knn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding
    that corresponds to the single cells in the full graph

    :param via_object:
    :param via_fine: usually just set to same as via_coarse unless you ran a refined run and want to link it to initial via_coarse's terminal clusters
    :param embedding: n_samples x 2. embedding is either the full or downsampled 2D representation of the full dataset.
    :param idx: if one uses a downsampled embedding of the original data, then idx is the selected indices of the downsampled samples used in the visualization
    :param cmap_name:
    :param dpi:
    :param scatter_size:
    :param marker_lineages: Default is to use all lineage pathways. other provide a list of lineage number (terminal cluster number).
    :return: fig, axs
    '''

    if via_fine is None:
        via_fine = via_object
    if len(marker_lineages) == 0:
        marker_lineages = via_fine.terminal_clusters

    else: marker_lineages = [i for i in marker_lineages if i in via_fine.terminal_clusters]
    print(f'{datetime.now()}\tMarker_lineages: {marker_lineages}')
    if embedding is None:
        if via_object.embedding is None:
            print('ERROR: please provide a single cell embedding or run re-via with do_compute_embedding==True using either embedding_type = via-umap OR via-mds')
            return
        else:
            print(f'automatically setting embedding to {via_object.embedding_type}')
            embedding = via_object.embedding

    if idx is None: idx = np.arange(0, via_object.nsamples)
    #G = via_object.full_graph_shortpath
    n_original_comp, n_original_comp_labels = connected_components(via_object.csr_full_graph, directed=False)
    G = via_object.full_graph_paths(via_object.data, n_original_comp)
    knn_hnsw = _make_knn_embeddedspace(embedding)
    y_root = []
    x_root = []
    root1_list = []
    p1_sc_bp = np.nan_to_num(via_fine.single_cell_bp[idx, :],nan=0.0, posinf=0.0, neginf=0.0)
    #row normalize
    row_sums = p1_sc_bp.sum(axis=1)
    p1_sc_bp = p1_sc_bp / row_sums[:, np.newaxis]
    print(f'{datetime.now()}\tCheck sc pb {p1_sc_bp[0,:]}')


    p1_labels = np.asarray(via_fine.labels)[idx]

    p1_cc = via_fine.connected_comp_labels
    p1_sc_pt_markov = list(np.asarray(via_fine.single_cell_pt_markov)[idx])
    X_data = via_fine.data

    X_ds = X_data[idx, :]
    p_ds = hnswlib.Index(space='l2', dim=X_ds.shape[1])
    p_ds.init_index(max_elements=X_ds.shape[0], ef_construction=200, M=16)
    p_ds.add_items(X_ds)
    p_ds.set_ef(50)
    num_cluster = len(set(via_fine.labels))
    G_orange = ig.Graph(n=num_cluster, edges=via_fine.edgelist_maxout, edge_attrs={'weight':via_fine.edgeweights_maxout})
    for ii, r_i in enumerate(via_fine.root):
        loc_i = np.where(p1_labels == via_fine.root[ii])[0]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]

        labels_root, distances_root = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_root.append(embedding[labels_root, 0][0])
        y_root.append(embedding[labels_root, 1][0])

        labelsroot1, distances1 = via_fine.knn_struct.knn_query(X_ds[labels_root[0][0], :], k=1)
        root1_list.append(labelsroot1[0][0])
        for fst_i in via_fine.terminal_clusters:
            path_orange = G_orange.get_shortest_paths(via_fine.root[ii], to=fst_i)[0]
            #if the roots is in the same component as the terminal cluster, then print the path to output
            if len(path_orange)>0: print(f'{datetime.now()}\tCluster path on clustergraph starting from Root Cluster {via_fine.root[ii]} to Terminal Cluster {fst_i}: {path_orange}')

    # single-cell branch probability evolution probability
    n_terminal_clusters = len(marker_lineages)
    fig_ncols = min(4, n_terminal_clusters)
    fig_nrows, mod = divmod(n_terminal_clusters, fig_ncols)
    if mod ==0:
        if fig_nrows==0: fig_nrows+=1
        else: fig_nrows=fig_nrows
    if mod != 0:        fig_nrows+=1

    fig, axs = plt.subplots(fig_nrows,fig_ncols,dpi=dpi,figsize=figsize)

    ts_array_original = np.asarray(via_fine.terminal_clusters)

    ti = 0# counter for terminal cluster
    for r in range(fig_nrows):
        for c in range (fig_ncols):
            if ti < n_terminal_clusters:
                ts_current = marker_lineages[ti]
                loc_ts_current = np.where(ts_array_original==ts_current)[0][0]
                loc_labels = np.where(np.asarray(via_fine.labels) == ts_current)[0]
                majority_composition = func_mode(list(np.asarray(via_fine.true_label)[loc_labels]))

                if fig_nrows ==1:
                    if fig_ncols ==1:plot_sc_pb(axs, fig, embedding, p1_sc_bp[:, loc_ts_current], ti= str(ts_current)+'-'+str(majority_composition), cmap_name=cmap_name, scatter_size=scatter_size, fontsize=fontsize)
                    else: plot_sc_pb(axs[c], fig, embedding, p1_sc_bp[:, loc_ts_current], ti= str(ts_current)+'-'+str(majority_composition), cmap_name=cmap_name, scatter_size=scatter_size, fontsize=fontsize)

                else:
                    plot_sc_pb(axs[r,c], fig, embedding, p1_sc_bp[:, loc_ts_current], ti= str(ts_current)+'-'+str(majority_composition), cmap_name=cmap_name, scatter_size=scatter_size, fontsize=fontsize)

                loc_i = np.where(p1_labels == ts_current)[0]
                val_pt = [p1_sc_pt_markov[i] for i in loc_i]
                th_pt = np.percentile(val_pt, 50)  # 50
                loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
                x = [embedding[xi, 0] for xi in
                     loc_i]  # location of sc nearest to average location of terminal clus in the EMBEDDED space
                y = [embedding[yi, 1] for yi in loc_i]
                labels, distances = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]),
                                                       k=1)  # knn_hnsw is knn of embedded space
                x_sc = embedding[labels[0], 0]  # terminal sc location in the embedded space
                y_sc = embedding[labels[0], 1]

                labelsq1, distances1 =via_fine.knn_struct.knn_query(X_ds[labels[0][0], :],
                                                               k=1)  # find the nearest neighbor in the PCA-space full graph

                path = G.get_shortest_paths(root1_list[p1_cc[loc_ts_current]], to=labelsq1[0][0])  # weights='weight')
                # G is the knn of all sc points

                path_idx = []  # find the single-cell which is nearest to the average-location of a terminal cluster
                # get the nearest-neighbor in this downsampled PCA-space graph. These will make the new path-way points
                path = path[0]

                # clusters of path
                cluster_path = []
                for cell_ in path:
                    cluster_path.append(via_fine.labels[cell_])

                revised_cluster_path = []
                revised_sc_path = []
                for enum_i, clus in enumerate(cluster_path):
                    num_instances_clus = cluster_path.count(clus)
                    if (clus == cluster_path[0]) | (clus == cluster_path[-1]):
                        revised_cluster_path.append(clus)
                        revised_sc_path.append(path[enum_i])
                    else:
                        if num_instances_clus > 1:  # typically intermediate stages spend a few transitions at the sc level within a cluster
                            if clus not in revised_cluster_path: revised_cluster_path.append(clus)  # cluster
                            revised_sc_path.append(path[enum_i])  # index of single cell
                print(f"{datetime.now()}\tRevised Cluster level path on sc-knnGraph from Root Cluster {via_fine.root[p1_cc[ti-1]]} to Terminal Cluster {ts_current} along path: {revised_cluster_path}")
                ti += 1
            fig.patch.set_visible(False)
            if fig_nrows==1:
                if fig_ncols ==1:
                    axs.axis('off')
                    axs.grid(False)
                else:
                    axs[c].axis('off')
                    axs[c].grid(False)
            else:
                axs[r,c].axis('off')
                axs[r, c].grid(False)

    return fig, axs

def draw_clustergraph(via_object, type_data='gene', gene_exp='', gene_list='', arrow_head=0.1,
                      edgeweight_scale=1.5, cmap=None, label_=True,figsize=(8,4)):
    '''
    :param via_object:
    :param type_data:
    :param gene_exp:
    :param gene_list:
    :param arrow_head:
    :param edgeweight_scale:
    :param cmap:
    :param label_:
    :return: fig, axs
    '''
    '''
    #draws the clustergraph for cluster level gene or pseudotime values
    # type_pt can be 'pt' pseudotime or 'gene' for gene expression
    # ax1 is the pseudotime graph
    '''
    n = len(gene_list)

    fig, axs = plt.subplots(1, n,figsize=figsize)
    pt = via_object.markov_hitting_times
    if cmap is None: cmap = 'coolwarm' if type_data == 'gene' else 'viridis_r'

    node_pos = via_object.graph_node_pos
    edgelist = list(via_object.edgelist_maxout)
    edgeweight = via_object.edgeweights_maxout

    node_pos = np.asarray(node_pos)

    import matplotlib.lines as lines


    n_groups = len(set(via_object.labels))  # node_pos.shape[0]
    n_truegroups = len(set(via_object.true_label))
    group_pop = np.zeros([n_groups, 1])
    via_object.cluster_population_dict = {}
    for group_i in set(via_object.labels):
        loc_i = np.where(via_object.labels == group_i)[0]

        group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
        via_object.cluster_population_dict[group_i] = len(loc_i)

    for i in range(n):
        ax_i = axs[i]
        gene_i = gene_list[i]
        '''
        for e_i, (start, end) in enumerate(edgelist):
            if pt[start] > pt[end]:
                start, end = end, start

            ax_i.add_line(
                lines.Line2D([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]],
                             color='black', lw=edgeweight[e_i] * edgeweight_scale, alpha=0.5))
            z = np.polyfit([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]], 1)
            minx = np.min(np.array([node_pos[start, 0], node_pos[end, 0]]))

            direction = 1 if node_pos[start, 0] < node_pos[end, 0] else -1
            maxx = np.max([node_pos[start, 0], node_pos[end, 0]])
            xp = np.linspace(minx, maxx, 500)
            p = np.poly1d(z)
            smooth = p(xp)
            step = 1

            ax_i.arrow(xp[250], smooth[250], xp[250 + direction * step] - xp[250],
                       smooth[250 + direction * step] - smooth[250],
                       shape='full', lw=0, length_includes_head=True, head_width=arrow_head_w, color='grey')
        '''
        c_edge, l_width = [], []
        for ei, pti in enumerate(pt):
            if ei in via_object.terminal_clusters:
                c_edge.append('red')
                l_width.append(1.5)
            else:
                c_edge.append('gray')
                l_width.append(0.0)
        ax_i = plot_edgebundle_viagraph(ax_i, via_object.hammerbundle_cluster, layout=via_object.graph_node_pos, CSM=via_object.CSM,
                                velocity_weight=via_object.velo_weight, pt=pt, headwidth_bundle=arrow_head, alpha_bundle=0.4, linewidth_bundle=edgeweight_scale)
        group_pop_scale = .5 * group_pop * 1000 / max(group_pop)
        pos = ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale, c=gene_exp[gene_i].values, cmap=cmap,
                           edgecolors=c_edge, alpha=1, zorder=3, linewidth=l_width)
        if label_==True:
            for ii in range(node_pos.shape[0]):
                ax_i.text(node_pos[ii, 0] + 0.1, node_pos[ii, 1] + 0.1, 'C'+str(ii)+' '+str(round(gene_exp[gene_i].values[ii], 1)),
                          color='black', zorder=4, fontsize=6)
        divider = make_axes_locatable(ax_i)
        cax = divider.append_axes('right', size='10%', pad=0.05)

        cbar=fig.colorbar(pos, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)
        ax_i.set_title(gene_i)
        ax_i.grid(False)
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        ax_i.axis('off')
    fig.patch.set_visible(False)
    return fig, axs
def plot_edge_bundle(hammerbundle_dict=None, via_object=None, alpha_bundle_factor=1,linewidth_bundle=2, facecolor:str='white', cmap:str = 'plasma', extra_title_text = '',size_scatter:int=3, alpha_scatter:float = 0.3 ,headwidth_bundle:float=0.1, headwidth_alpha:float=0.8, arrow_frequency:float=0.05, show_arrow:bool=True,sc_labels_sequential:list=None,sc_labels_expression:list=None, initial_bandwidth=0.03, decay=0.7, n_milestones:int=None, scale_scatter_size_pop:bool=False, show_milestones:bool=True, sc_labels:list=None, text_labels:bool=False, lineage_pathway:list = [], dpi:int = 300, fontsize_title:int=6, fontsize_labels:int=6):

    '''

    Edges can be colored by time-series numeric labels, pseudotime, or gene expression. If not specificed then time-series is chosen if available, otherwise falls back to pseudotime. to use gene expression the sc_labels_expression is provided as a list.
    To specify other numeric sequential data provide a list of sc_labels_sequential = [] n_samples in length. via_object.embedding must be an ndarray of shape (nsamples,2)

    :param hammer_bundle_dict: dictionary with keys: hammerbundle object with coordinates of all the edges to draw. If hammer_bundle and layout are None, then this will be computed internally
    :param via_object: type via object, if hammerbundle_dict is None, then you must provide a via_object. Ensure that via_object has embedding attribute
    :param layout: coords of cluster nodes and optionally also contains the numeric value associated with each cluster (such as time-stamp) layout[['x','y','numeric label']] sc/cluster/milestone level
    :param CSM: cosine similarity matrix. cosine similarity between the RNA velocity between neighbors and the change in gene expression between these neighbors. Only used when available
    :param velocity_weight: percentage weightage given to the RNA velocity based transition matrix
    :param pt: cluster-level pseudotime
    :param alpha_bundle: alpha when drawing lines
    :param linewidth_bundle: linewidth of bundled lines
    :param edge_color:
    :param arrow_frequency: min dist between arrows (bundled edges otherwise have overcrowding of arrows)
    :param show_direction: True will draw arrows along the lines to indicate direction
    :param milestone_edges: pandas DataFrame milestoone_edges[['source','target']]
    :param milestone_numeric_values: the milestone average of numeric values such as time (days, hours), location (position), or other numeric value used for coloring edges in a sequential manner
            if this is None then the edges are colored by length to distinguish short and long range edges
    :param arrow_frequency: 0.05. higher means fewer arrows
    :param n_milestones: int  None. if no hammerbundle_dict is provided, but via_object is provided, then the user can specify level of granularity by setting the n_milestones. otherwise it will be automatically selected
    :param scale_scatter_size_pop: bool default False
    :param sc_labels_expression: list single cell numeric values used for coloring edges and nodes of corresponding milestones mean expression levels (len n_single_cell samples)
            edges can be colored by time-series numeric labels, pseudotime, or gene expression. If not specificed then time-series is chosen if available, otherwise falls back to pseudotime. to use gene expression the sc_labels_expression is provided as a list
    :param sc_labels_sequential: list single cell numeric sequential values used for directionality inference as replacement for  pseudotime or v0.time_series_labels (len n_samples single cell)
    :param sc_labels: list None list of single-cell level labels (categorial or discrete set of numerical values) to label the nodes
    :param text_labels: bool False if you want to label the nodes based on sc_labels (or true_label if via_object is provided)
    :return: axis with bundled edges plotted
    '''
    cmap_name = cmap
    if hammerbundle_dict is None:
        if via_object is None: print('if hammerbundle_dict is not provided, then you must provide via_object')
        else:
            hammerbundle_dict = via_object.hammerbundle_milestone_dict
            if hammerbundle_dict is None:
                if n_milestones is None: n_milestones = min(via_object.nsamples, max(250, int(0.1 * via_object.nsamples)))
                if sc_labels_sequential is None:
                    if via_object.time_series_labels is not None:
                        sc_labels_sequential = via_object.time_series_labels
                    else:
                        sc_labels_sequential = via_object.single_cell_pt_markov
                print(f'{datetime.now()}\tComputing Edges')
                hammerbundle_dict = make_edgebundle_milestone(via_object=via_object,
                    embedding=via_object.embedding, sc_graph=via_object.ig_full_graph, n_milestones=n_milestones,
                    sc_labels_numeric=sc_labels_sequential, initial_bandwidth=initial_bandwidth, decay=decay, weighted=True)
                via_object.hammerbundle_dict = hammerbundle_dict
            hammer_bundle = hammerbundle_dict['hammerbundle']
            layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
            milestone_edges = hammerbundle_dict['edges']
            if sc_labels_expression is None:
                milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']
            else: milestone_numeric_values = sc_labels_expression
            milestone_pt = hammerbundle_dict['milestone_embedding']['pt']
            if sc_labels_expression is not None:  # if both sclabelexpression and sequential are provided, then sc_labels_expression takes precedence
                df = pd.DataFrame()
                df['sc_milestone_labels'] = hammerbundle_dict['sc_milestone_labels']

                df['sc_expression'] = sc_labels_expression
                df = df.groupby('sc_milestone_labels').mean()

                milestone_numeric_values = df['sc_expression'].values  # used to color edges


    else:
        hammer_bundle = hammerbundle_dict['hammerbundle']
        layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
        milestone_edges = hammerbundle_dict['edges']
        milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']

        if sc_labels_expression is not None: #if both sclabelexpression and sequential are provided, then sc_labels_expression takes precedence
            df = pd.DataFrame()
            df['sc_milestone_labels']= hammerbundle_dict['sc_milestone_labels']
            df['sc_expression'] = sc_labels_expression
            df = df.groupby('sc_milestone_labels').mean()
            milestone_numeric_values = df['sc_expression'].values #used to color edges
        milestone_pt = hammerbundle_dict['milestone_embedding']['pt']
    if len(lineage_pathway)== 0:
        #fig, ax = plt.subplots(facecolor=facecolor)
        fig_nrows, fig_ncols = 1,1
    else:
        n_terminal_clusters = len(lineage_pathway)
        fig_ncols = min(3, n_terminal_clusters)
        fig_nrows, mod = divmod(n_terminal_clusters, fig_ncols)
        if mod ==0:
            if fig_nrows==0: fig_nrows+=1
            else: fig_nrows=fig_nrows
        if mod != 0: fig_nrows+=1
    fig, ax = plt.subplots(fig_nrows,fig_ncols,dpi=dpi)
    counter_ = 0
    n_real_subplots = max(len(lineage_pathway), 1)

    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if (counter_ < n_real_subplots):
                if len(lineage_pathway) > 0:
                    milestone_numeric_values = hammerbundle_dict['milestone_embedding'][
                        'sc_lineage_probability_' + str(lineage_pathway[counter_])]

                x_ = [l[0] for l in layout ]
                y_ =  [l[1] for l in layout ]
                #min_x, max_x = min(x_), max(x_)
                #min_y, max_y = min(y_), max(y_)
                delta_x =  max(x_)- min(x_)
                delta_y = max(y_)- min(y_)

                layout = np.asarray(layout)
                # get each segment. these are separated by nans.
                hbnp = hammer_bundle.to_numpy()
                splits = (np.isnan(hbnp[:, 0])).nonzero()[0] #location of each nan values
                edgelist_segments = []
                start = 0
                segments = []
                arrow_coords=[]
                seg_len = [] #length of a segment
                for stop in splits:
                    seg = hbnp[start:stop, :]
                    segments.append(seg)
                    seg_len.append(seg.shape[0])
                    start = stop

                min_seg_length = min(seg_len)
                max_seg_length = max(seg_len)
                seg_len=np.asarray(seg_len)
                seg_len = np.clip(seg_len, a_min=np.percentile(seg_len, 10),
                                  a_max=np.percentile(seg_len,90))
                #mean_seg_length = sum(seg_len)/len(seg_len)

                step = 1  # every step'th segment is plotted

                cmap = matplotlib.cm.get_cmap(cmap)
                if milestone_numeric_values is not None:
                    max_numerical_value = max(milestone_numeric_values)
                    min_numerical_value = min(milestone_numeric_values)

                seg_count = 0

                for seg in segments[::step]:
                    do_arrow = True

                    #seg_weight = max(0.3, math.log(1+seg[-1,2])) seg[-1,2] column index 2 has the weight information

                    seg_weight=seg[-1,2]*seg_len[seg_count]/(max_seg_length-min_seg_length)##seg.shape[0] / (max_seg_length - min_seg_length)#seg.shape[0]

                    #cant' quite decide yet if sigmoid is desirable
                    #seg_weight=sigmoid_scalar(seg.shape[0] / (max_seg_length - min_seg_length), scale=5, shift=mean_seg_length / (max_seg_length - min_seg_length))
                    alpha_bundle =  max(seg_weight*alpha_bundle_factor,0.1)# max(0.1, math.log(1 + seg[-1, 2]))
                    if alpha_bundle>1: alpha_bundle=1

                    source_milestone = milestone_edges['source'].values[seg_count]
                    target_milestone = milestone_edges['target'].values[seg_count]

                    direction = milestone_pt[target_milestone] - milestone_pt[source_milestone]
                    if direction <0: direction = -1
                    else: direction = 1
                    source_milestone_numerical_value = milestone_numeric_values[source_milestone]

                    target_milestone_numerical_value = milestone_numeric_values[target_milestone]
                    #print('source milestone', source_milestone_numerical_value)
                    #print('target milestone', target_milestone_numerical_value)
                    min_source_target_numerical_value = min(source_milestone_numerical_value,target_milestone_numerical_value)

                    rgba = cmap((min_source_target_numerical_value - min_numerical_value) / (max_numerical_value - min_numerical_value))

                    #else: rgba = cmap(min(seg_weight,0.95))#cmap(seg.shape[0]/(max_seg_length-min_seg_length))
                    #if seg_weight>0.05: seg_weight=0.1
                    #if seg_count%10000==0: print('seg weight', seg_weight)
                    seg = seg[:,0:2].reshape(-1,2)
                    seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

                    if fig_nrows == 1:
                        if fig_ncols == 1: ax.plot(seg_p[:, 0], seg_p[:, 1],linewidth=linewidth_bundle*seg_weight, alpha=alpha_bundle, color=rgba)#edge_color )
                        else:  ax[c].plot(seg_p[:, 0], seg_p[:, 1],linewidth=linewidth_bundle*seg_weight, alpha=alpha_bundle, color=rgba)#edge_color )
                    else: ax[r,c].plot(seg_p[:, 0], seg_p[:, 1],linewidth=linewidth_bundle*seg_weight, alpha=alpha_bundle, color=rgba)#edge_color )

                    if (show_arrow) & (seg_p.shape[0]>3):
                        mid_point = math.floor(seg_p.shape[0] / 2)

                        if len(arrow_coords)>0: #dont draw arrows in overlapping segments
                            for v1 in arrow_coords:
                                dist_ = dist_points(v1,v2=[seg_p[mid_point, 0], seg_p[mid_point, 1]])

                                if dist_< arrow_frequency*delta_x: do_arrow=False
                                if dist_< arrow_frequency*delta_y: do_arrow=False

                        if (do_arrow==True) & (seg_p.shape[0]>3):
                            if fig_nrows == 1:
                                if fig_ncols == 1:
                                    ax.arrow(seg_p[mid_point, 0], seg_p[mid_point, 1],
                                 seg_p[mid_point + (direction * step), 0] - seg_p[mid_point, 0],
                                 seg_p[mid_point + (direction * step), 1] - seg_p[mid_point, 1],
                                 lw=0, length_includes_head=False, head_width=headwidth_bundle, color=rgba,shape='full', alpha= headwidth_alpha, zorder=5)

                                else: ax[c].arrow(seg_p[mid_point, 0], seg_p[mid_point, 1],
                                 seg_p[mid_point + (direction * step), 0] - seg_p[mid_point, 0],
                                 seg_p[mid_point + (direction * step), 1] - seg_p[mid_point, 1],
                                 lw=0, length_includes_head=False, head_width=headwidth_bundle, color=rgba,shape='full', alpha= headwidth_alpha, zorder=5)

                            else: ax[r,c].arrow(seg_p[mid_point, 0], seg_p[mid_point, 1],
                                 seg_p[mid_point + (direction * step), 0] - seg_p[mid_point, 0],
                                 seg_p[mid_point + (direction * step), 1] - seg_p[mid_point, 1],
                                 lw=0, length_includes_head=False, head_width=headwidth_bundle, color=rgba,shape='full', alpha= headwidth_alpha, zorder=5)
                            arrow_coords.append([seg_p[mid_point, 0], seg_p[mid_point, 1]])


                    seg_count+=1
                if show_milestones == False:
                    size_scatter = 0.01
                    show_milestones=True
                    scale_scatter_size_pop = False
                if show_milestones == True:
                    milestone_numeric_values_normed = []
                    milestone_numeric_values_rgba=[]
                    for ei, i in enumerate(milestone_numeric_values):
                        rgba_ = cmap((i - min_numerical_value) / (max_numerical_value - min_numerical_value))
                        color_numeric = (i - min_numerical_value) / (max_numerical_value - min_numerical_value)
                        milestone_numeric_values_normed.append(color_numeric)
                        milestone_numeric_values_rgba.append(rgba_)
                    if scale_scatter_size_pop==True:

                        n_samples = layout.shape[0]
                        sqrt_nsamples = math.sqrt(n_samples)
                        group_pop_scale = [math.log(6+i /sqrt_nsamples) for i in
                                           hammerbundle_dict['milestone_embedding']['cluster population']]
                        size_scatter_scaled = [size_scatter * i for i in group_pop_scale]
                    else: size_scatter_scaled = size_scatter # constant value

                    if fig_nrows == 1:
                        if fig_ncols == 1:
                            im = ax.scatter(layout[:, 0], layout[:, 1], s=0.01,
                                       c=milestone_numeric_values_normed, cmap=cmap_name,
                                       edgecolors='None') #without alpha parameter which otherwise gets passed onto the colorbar
                            ax.scatter(layout[:,0], layout[:,1], s=size_scatter_scaled, c=milestone_numeric_values_normed, cmap= cmap_name, alpha=alpha_scatter, edgecolors='None')
                        else:
                            im = ax[c].scatter(layout[:, 0], layout[:, 1], s=size_scatter_scaled,
                                               c=milestone_numeric_values_normed, cmap=cmap_name,
                                            edgecolors='None') #without alpha parameter which otherwise gets passed onto the colorbar
                            ax[c].scatter(layout[:,0], layout[:,1], s=size_scatter_scaled, c=milestone_numeric_values_normed,cmap= cmap_name, alpha=alpha_scatter, edgecolors='None')
                    else:

                        im = ax[r, c].scatter(layout[:,0], layout[:,1], c=milestone_numeric_values_normed, s=0.01, cmap=cmap_name,  edgecolors='none', vmin= min_numerical_value, vmax=1)  # prevent auto-normalization of colors
                        ax[r, c].scatter(layout[:, 0], layout[:, 1], s=size_scatter_scaled,
                                              c=milestone_numeric_values_normed, cmap=cmap_name,
                                              alpha=alpha_scatter, edgecolors='None', vmin=min_numerical_value,
                                              vmax=1)

                    if text_labels == True:

                        #if text labels is true but user has not provided any labels at the sc level from which to create milestone categorical labels
                        if sc_labels is None:
                            if via_object is not None: sc_labels = via_object.true_label
                            else: print(f'{datetime.now()}\t ERROR: in order to show labels, please provide list of sc_labels at the single cell level OR via_object')
                        for i in range(layout.shape[0]):
                            sc_milestone_labels = hammerbundle_dict['sc_milestone_labels']
                            loc_milestone = np.where(np.asarray(sc_milestone_labels)==i)[0]

                            mode_label = func_mode(list(np.asarray(sc_labels)[loc_milestone]))
                            if scale_scatter_size_pop==True:
                                if fig_nrows == 1:
                                    if fig_ncols == 1:
                                        ax.scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled[i], c=np.array([milestone_numeric_values_rgba[i]]),
                                       alpha=alpha_scatter, edgecolors='None', label=mode_label)
                                    else:
                                        ax[c].scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled[i], c=np.array([milestone_numeric_values_rgba[i]]),
                                       alpha=alpha_scatter, edgecolors='None', label=mode_label)
                                else:
                                    ax[r,c].scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled[i], c=np.array([milestone_numeric_values_rgba[i]]),
                                       alpha=alpha_scatter, edgecolors='None', label=mode_label)
                            else:
                                if fig_nrows == 1:
                                    if fig_ncols == 1:
                                        ax.scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled, c=np.array([milestone_numeric_values_rgba[i]]),
                                       alpha=alpha_scatter, edgecolors='None', label=mode_label)
                                    else: ax[c].scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled, c=np.array([milestone_numeric_values_rgba[i]]),
                                       alpha=alpha_scatter, edgecolors='None', label=mode_label)
                                else: ax[r, c].scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled, c=np.array([milestone_numeric_values_rgba[i]]),
                                       alpha=alpha_scatter, edgecolors='None', label=mode_label)
                            if fig_nrows == 1:
                                if fig_ncols == 1:
                                    ax.text(layout[i, 0], layout[i, 1], mode_label, style='italic', fontsize=fontsize_labels, color="black")
                                else:ax[c].text(layout[i, 0], layout[i, 1], mode_label, style='italic', fontsize=fontsize_labels, color="black")
                            else: ax[r,c].text(layout[i, 0], layout[i, 1], mode_label, style='italic', fontsize=fontsize_labels, color="black")
                time = datetime.now()
                time = time.strftime("%H:%M")
                if len(lineage_pathway)==0: title_ = extra_title_text + ' n_milestones = ' + str(int(layout.shape[0])) #+ ' time: ' + time
                else: title_ = 'lineage:'+str(lineage_pathway[counter_])

                if fig_nrows==1:
                    if fig_ncols ==1:
                        ax.axis('off')
                        ax.grid(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.set_facecolor(facecolor)
                        ax.set_title(label=title_, color='black',fontsize=fontsize_title)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        cb = fig.colorbar(im, cax=cax, orientation='vertical', label='lineage likelihood')
                        ax_cb = cb.ax
                        text = ax_cb.yaxis.label
                        font = matplotlib.font_manager.FontProperties(
                            size=fontsize_title)  # family='times new roman', style='italic',
                        text.set_font_properties(font)
                        ax_cb.tick_params(labelsize=int(fontsize_title * 0.8))
                        cb.outline.set_visible(False)

                    else:
                        ax[c].axis('off')
                        ax[c].grid(False)
                        ax[c].spines['top'].set_visible(False)
                        ax[c].spines['right'].set_visible(False)
                        ax[c].set_facecolor(facecolor)
                        ax[c].set_title(label=title_, color='black',fontsize=fontsize_title)


                        divider = make_axes_locatable(ax[c])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        cb = fig.colorbar(im, cax=cax, orientation='vertical', label='lineage likelihood')
                        ax_cb = cb.ax
                        text = ax_cb.yaxis.label
                        font = matplotlib.font_manager.FontProperties(
                            size=fontsize_title)  # family='times new roman', style='italic',
                        text.set_font_properties(font)
                        ax_cb.tick_params(labelsize=int(fontsize_title * 0.8))
                        cb.outline.set_visible(False)

                else:

                    ax[r,c].axis('off')
                    ax[r, c].grid(False)
                    ax[r, c].spines['top'].set_visible(False)
                    ax[r, c].spines['right'].set_visible(False)
                    ax[r, c].set_facecolor(facecolor)
                    ax[r,c].set_title(label=title_, color='black', fontsize=fontsize_title)

                    divider = make_axes_locatable(ax[r, c])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cb = fig.colorbar(im, cax=cax, orientation='vertical', label='Lineage likelihood',
                                      cmap='plasma')
                    ax_cb = cb.ax
                    text = ax_cb.yaxis.label
                    font = matplotlib.font_manager.FontProperties(
                        size=fontsize_title)  # family='times new roman', style='italic',
                    text.set_font_properties(font)
                    ax_cb.tick_params(labelsize=int(fontsize_title * 0.8))
                    cb.outline.set_visible(False)

                counter_ +=1
            else:
                if fig_nrows==1:
                    if fig_ncols ==1:
                        ax.axis('off')
                        ax.grid(False)
                    else:
                        ax[c].axis('off')
                        ax[c].grid(False)
                else:
                    ax[r, c].axis('off')
                    ax[r, c].grid(False)
    return fig, ax
def animate_edge_bundle(hammerbundle_dict=None,  via_object=None, linewidth_bundle=2, n_milestones:int=None,facecolor:str='white', cmap:str = 'plasma_r', extra_title_text = '',size_scatter:int=1, alpha_scatter:float = 0.2, saveto='/home/shobi/Trajectory/Datasets/animation_default.gif', time_series_labels:list=None, sc_labels_numeric:list=None ):

    '''
    :param ax: axis to plot on
    :param hammer_bundle: hammerbundle object with coordinates of all the edges to draw
    :param layout: coords of cluster nodes and optionally also contains the numeric value associated with each cluster (such as time-stamp) layout[['x','y','numeric label']] sc/cluster/milestone level
    :param CSM: cosine similarity matrix. cosine similarity between the RNA velocity between neighbors and the change in gene expression between these neighbors. Only used when available
    :param velocity_weight: percentage weightage given to the RNA velocity based transition matrix
    :param pt: cluster-level pseudotime
    :param alpha_bundle: alpha when drawing lines
    :param linewidth_bundle: linewidth of bundled lines
    :param edge_color:
    :param headwidth_bundle: headwidth of arrows used in bundled edges
    :param arrow_frequency: min dist between arrows (bundled edges otherwise have overcrowding of arrows)
    :param show_direction: True will draw arrows along the lines to indicate direction
    :param milestone_edges: pandas DataFrame milestone_edges[['source','target']]
    :return: axis with bundled edges plotted
    '''
    import tqdm

    if hammerbundle_dict is None:
        if via_object is None: print(f'{datetime.now()}\tERROR: Hammerbundle_dict needs to be provided either through via_object or by running make_edgebundle_milestone()')
        else:
            hammerbundle_dict = via_object.hammerbundle_milestone_dict
            if hammerbundle_dict is None:
                if n_milestones is None: n_milestones = min(via_object.nsamples, max(250, int(0.1 * via_object.nsamples)))
                if sc_labels_numeric is None:
                    if via_object.time_series_labels is not None:
                        sc_labels_numeric = via_object.time_series_labels
                    else:
                        sc_labels_numeric = via_object.single_cell_pt_markov

                hammerbundle_dict = make_edgebundle_milestone(via_object=via_object,
                    embedding=via_object.embedding, sc_graph=via_object.ig_full_graph, n_milestones=n_milestones,
                    sc_labels_numeric=sc_labels_numeric, initial_bandwidth=0.02, decay=0.7, weighted=True)
            hammer_bundle = hammerbundle_dict['hammerbundle']
            layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
            milestone_edges = hammerbundle_dict['edges']
            milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']
            milestone_pt = hammerbundle_dict['milestone_embedding']['pt'] #used when plotting arrows

    else:
        hammer_bundle = hammerbundle_dict['hammerbundle']
        layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
        milestone_edges = hammerbundle_dict['edges']
        milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']
        milestone_pt = hammerbundle_dict['milestone_embedding']['pt']  # used when plotting arrows
    fig, ax = plt.subplots(facecolor=facecolor,figsize=(15, 12))
    time_thresh = min(milestone_numeric_values)
    #ax.set_facecolor(facecolor)
    ax.grid(False)
    x_ = [l[0] for l in layout ]
    y_ =  [l[1] for l in layout ]
    #min_x, max_x = min(x_), max(x_)
    #min_y, max_y = min(y_), max(y_)
    delta_x =  max(x_)- min(x_)

    delta_y = max(y_)- min(y_)

    layout = np.asarray(layout)
    # make a knn so we can find which clustergraph nodes the segments start and end at


    # get each segment. these are separated by nans.
    hbnp = hammer_bundle.to_numpy()
    splits = (np.isnan(hbnp[:, 0])).nonzero()[0] #location of each nan values
    edgelist_segments = []
    start = 0
    segments = []
    arrow_coords=[]
    seg_len = [] #length of a segment
    for stop in splits:
        seg = hbnp[start:stop, :]
        segments.append(seg)
        seg_len.append(seg.shape[0])
        start = stop

    min_seg_length = min(seg_len)
    max_seg_length = max(seg_len)
    #mean_seg_length = sum(seg_len)/len(seg_len)
    seg_len = np.asarray(seg_len)
    seg_len = np.clip(seg_len, a_min=np.percentile(seg_len, 10),
                      a_max=np.percentile(seg_len, 90))
    step = 1  # every step'th segment is plotted


    cmap = matplotlib.cm.get_cmap(cmap)
    if milestone_numeric_values is not None:
        max_numerical_value = max(milestone_numeric_values)
        min_numerical_value = min(milestone_numeric_values)

    seg_count = 0

    #print('numeric vals', milestone_numeric_values)
    loc_time_thresh = np.where(np.asarray(milestone_numeric_values) <= time_thresh)[0].tolist()
    #print('loc time thres', loc_time_thresh)
    milestone_edges['source_thresh'] = milestone_edges['source'].isin(loc_time_thresh)#apply(lambda x: any([k in x for k in loc_time_thresh]))

    #print(milestone_edges[0:10])
    idx = milestone_edges.index[milestone_edges['source_thresh']].tolist()
    #print('loc time thres', time_thresh, loc_time_thresh)
    for i in idx:
        seg = segments[i]
        source_milestone = milestone_edges['source'].values[i]
        target_milestone = milestone_edges['target'].values[i]

        #seg_weight = max(0.3, math.log(1+seg[-1,2])) seg[-1,2] column index 2 has the weight information

        seg_weight=seg[-1,2]*seg_len[i]/(max_seg_length-min_seg_length)##seg.shape[0] / (max_seg_length - min_seg_length)

        #cant' quite decide yet if sigmoid is desirable
        #seg_weight=sigmoid_scalar(seg.shape[0] / (max_seg_length - min_seg_length), scale=5, shift=mean_seg_length / (max_seg_length - min_seg_length))
        alpha_bundle =  max(seg_weight,0.1)# max(0.1, math.log(1 + seg[-1, 2]))
        if alpha_bundle>1: alpha_bundle=1

        if milestone_numeric_values is not None:
            source_milestone_numerical_value = milestone_numeric_values[source_milestone]
            target_milestone_numerical_value = milestone_numeric_values[target_milestone]
            #print('source milestone', source_milestone_numerical_value)
            #print('target milestone', target_milestone_numerical_value)
            rgba_milestone_value = min(source_milestone_numerical_value, target_milestone_numerical_value)
            rgba = cmap( (rgba_milestone_value - min_numerical_value) / (max_numerical_value - min_numerical_value))
        else: rgba = cmap(min(seg_weight,0.95))#cmap(seg.shape[0]/(max_seg_length-min_seg_length))
        #if seg_weight>0.05: seg_weight=0.1
        if seg_count%10000==0: print('seg weight', seg_weight)
        seg = seg[:,0:2].reshape(-1,2)
        seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

        ax.plot(seg_p[:, 0], seg_p[:, 1],linewidth=linewidth_bundle*seg_weight, alpha=alpha_bundle, color=rgba)#edge_color )
        seg_count+=1
    milestone_numeric_values_rgba=[]

    if milestone_numeric_values is not None:
        for i in milestone_numeric_values:
            rgba_ = cmap((i-min_numerical_value)/(max_numerical_value-min_numerical_value))
            milestone_numeric_values_rgba.append(rgba_)
        ax.scatter(layout[loc_time_thresh,0], layout[loc_time_thresh,1], s=size_scatter, c=np.asarray(milestone_numeric_values_rgba)[loc_time_thresh], alpha=alpha_scatter)
        #if we dont plot all the points, then the size of axis changes and the location of the graph moves/changes as more points are added
        ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter,
                   c=np.asarray(milestone_numeric_values_rgba), alpha=0)
    else: ax.scatter(layout[:,0], layout[:,1], s=size_scatter, c='red', alpha=alpha_scatter)
    ax.set_facecolor(facecolor)
    ax.axis('off')
    time = datetime.now()
    time = time.strftime("%H:%M")
    title_ = 'n_milestones = '+str(int(layout.shape[0])) +' time: '+time + ' ' + extra_title_text
    ax.set_title(label=title_, color = 'black')
    print(f"{datetime.now()}\tFinished plotting edge bundle")
    if time_series_labels is not None:
        time_series_set_order = list(sorted(list(set(time_series_labels))))
        t_diff_mean = 0.25 * np.mean(
            np.array([int(abs(y - x)) for x, y in zip(time_series_set_order[:-1], time_series_set_order[1:])]))
        cycles = (max_numerical_value - min_numerical_value) / t_diff_mean
        print('number cycles', cycles)
    else:
        if via_object is not None:
            time_series_labels=via_object.time_series_labels
            if time_series_labels is None:
                time_series_labels = [int(i*10) for i in via_object.single_cell_pt_markov]
            time_series_set_order = list(sorted(list(set(time_series_labels))))
            print('times series order set', time_series_set_order)
            t_diff_mean = 0.25 * np.mean(np.array([int(abs(y - x)) for x, y in zip(time_series_set_order[:-1], time_series_set_order[1:])]))/10 #divide by 10 because we multiplied the single_cell_pt_markov by 10
            cycles = (max_numerical_value - min_numerical_value) / (t_diff_mean)
            print('number cycles if no time_series labels given', cycles)
    def update_edgebundle(frame_no):

        print('inside update', frame_no, 'out of',int(cycles)*3)
        if len(time_series_labels)>0:
            #time_series_set_order = list(sorted(list(set(time_series_labels))))
            #t_diff_mean = 0.25*np.mean(np.array([int(abs(y - x)) for x, y in zip(time_series_set_order[:-1], time_series_set_order[1:])]))
            #cycles= (max_numerical_value-min_numerical_value)/t_diff_mean
            #print('number cycles', cycles)
            time_thresh = min_numerical_value +frame_no%(cycles+1)*t_diff_mean
        else:
            n_intervals=10
            time_thresh = min_numerical_value + (frame_no %n_intervals)*(max_numerical_value-min_numerical_value)/n_intervals

        loc_time_thresh = np.where((np.asarray(milestone_numeric_values) <= time_thresh) & (np.asarray(milestone_numeric_values) > time_thresh-t_diff_mean))[0].tolist()

        #print('loc time thres', time_thresh, loc_time_thresh)
        milestone_edges['source_thresh'] = milestone_edges['source'].isin(
            loc_time_thresh)  # apply(lambda x: any([k in x for k in loc_time_thresh]))
        #print('animate milestone edges')

        idx = milestone_edges.index[milestone_edges['source_thresh']].tolist()
        for i in idx:
            seg = segments[i]
            source_milestone = milestone_edges['source'].values[i]

            # seg_weight = max(0.3, math.log(1+seg[-1,2])) seg[-1,2] column index 2 has the weight information

            seg_weight = seg[-1, 2] * seg_len[i] / (
                        max_seg_length - min_seg_length)  ##seg.shape[0] / (max_seg_length - min_seg_length)

            # cant' quite decide yet if sigmoid is desirable
            # seg_weight=sigmoid_scalar(seg.shape[0] / (max_seg_length - min_seg_length), scale=5, shift=mean_seg_length / (max_seg_length - min_seg_length))
            alpha_bundle = max(seg_weight, 0.1)  # max(0.1, math.log(1 + seg[-1, 2]))
            if alpha_bundle > 1: alpha_bundle = 1

            if milestone_numeric_values is not None:
                source_milestone_numerical_value = milestone_numeric_values[source_milestone]

                rgba = cmap((source_milestone_numerical_value - min_numerical_value) / (
                            max_numerical_value - min_numerical_value))
            else:
                rgba = cmap(min(seg_weight, 0.95))  # cmap(seg.shape[0]/(max_seg_length-min_seg_length))
            # if seg_weight>0.05: seg_weight=0.1

            seg = seg[:, 0:2].reshape(-1, 2)
            seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

            ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight, alpha=alpha_bundle,
                    color=rgba)  # edge_color )

        milestone_numeric_values_rgba = []

        if milestone_numeric_values is not None:
            for i in milestone_numeric_values:
                rgba_ = cmap((i - min_numerical_value) / (max_numerical_value - min_numerical_value))
                milestone_numeric_values_rgba.append(rgba_)
            if time_thresh > 1.1*max_numerical_value:  ax.clear()   #ax.scatter(layout[loc_time_thresh, 0], layout[loc_time_thresh, 1], s=size_scatter, c='black', alpha=1)
            else:
                ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter,
                           c=np.asarray(milestone_numeric_values_rgba), alpha=0)
                ax.scatter(layout[loc_time_thresh, 0], layout[loc_time_thresh, 1], s=size_scatter, c=np.asarray(milestone_numeric_values_rgba)[loc_time_thresh], alpha=alpha_scatter)
        else:
            ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter, c='red', alpha=alpha_scatter)
        #pbar.update()

    frame_no = int(cycles)*3
    animation = FuncAnimation(fig, update_edgebundle, frames=frame_no, interval=100, repeat=False)
    #pbar = tqdm.tqdm(total=frame_no)
    #pbar.close()
    print('complete animate')

    animation.save(saveto, writer='imagemagick')#, fps=30)
    print('saved animation')
    plt.show()
    return

def animated_streamplot(via_object, embedding , density_grid=1,
 linewidth=0.5,min_mass = 1, cutoff_perc = None,scatter_size=500, scatter_alpha=0.2,marker_edgewidth=0.1, smooth_transition=1, smooth_grid=0.5, color_scheme = 'annotation', other_labels=[] , b_bias=20, n_neighbors_velocity_grid=None, fontsize=8, alpha_animate=0.7,
                        cmap_scatter = 'rainbow', cmap_stream='Blues', segment_length=1, saveto='/home/shobi/Trajectory/Datasets/animation.gif',use_sequentially_augmented=False, facecolor_='white', random_seed=0):
    '''
    Draw Animated vector plots. the Saved .gif file saved at the saveto address, is the best for viewing the animation as the fig, ax output can be slow

    :param via_object: viaobject
    :param embedding: ndarray (nsamples,2) umap, tsne, via-umap, via-mds
    :param density_grid:
    :param linewidth:
    :param min_mass:
    :param cutoff_perc:
    :param scatter_size:
    :param scatter_alpha:
    :param marker_edgewidth:
    :param smooth_transition:
    :param smooth_grid:
    :param color_scheme: 'annotation', 'cluster', 'other'
    :param add_outline_clusters:
    :param cluster_outline_edgewidth:
    :param gp_color:
    :param bg_color:
    :param title:
    :param b_bias:
    :param n_neighbors_velocity_grid:
    :param fontsize:
    :param alpha_animate:
    :param cmap_scatter:
    :param cmap_stream: string of a cmap for streamlines, default = 'Blues' (for dark blue lines) . Consider 'Blues_r' for white lines OR 'Greys/_r' 'gist_yard/_r'
    :param color_stream: string like 'white'. will override cmap_stream
    :param segment_length:

    :return: fig, ax.
    '''
    import tqdm

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers
    from matplotlib.collections import LineCollection
    from pyVIA.windmap import Streamlines
    import matplotlib.patheffects as PathEffects

    #import cartopy.crs as ccrs
    if embedding is None:
        embedding = via_object.embedding
        if embedding is None: print(f'ERROR: please provide input parameter embedding of ndarray with shape (nsamples, 2)')
    V_emb = via_object._velocity_embedding(embedding, smooth_transition, b=b_bias, use_sequentially_augmented=use_sequentially_augmented)

    V_emb *= 10 #the velocity of the samples has shape (n_samples x 2).*100

    #interpolate the velocity along all grid points based on the velocities of the samples in V_emb
    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=embedding,
        V_emb=V_emb,
        density=density_grid,
        smooth=smooth_grid,
        min_mass=min_mass,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=cutoff_perc, n_neighbors=n_neighbors_velocity_grid)
    print(f'{datetime.now()}\tInside animated. File will be saved to location {saveto}')

    '''
    #inspecting V_grid. most values are close to zero except those along data points of cells  
    print('U', V_grid.shape, V_grid[0].shape)
    print('sum is nan of each column', np.sum(np.isnan(V_grid[0]),axis=0))
    msk = np.isnan(V_grid[0])
    V_grid[0][msk]=0
    print('sum is nan of each column', np.sum(np.isnan(V_grid[0]), axis=0))
    print('max of each U column', np.max(V_grid[0], axis=0))
    print('max V', np.max(V_grid[1], axis=0))
    print('V')
    print( V_grid[1])
    '''
    #lengths = np.sqrt((V_grid ** 2).sum(0))

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    fig.patch.set_visible(False)
    if color_scheme == 'time': ax.scatter(embedding[:, 0], embedding[:, 1], c=via_object.single_cell_pt_markov, alpha=scatter_alpha, zorder=0,
               s=scatter_size, linewidths=marker_edgewidth, cmap=cmap_scatter)
    else:
        if color_scheme == 'annotation': color_labels = via_object.true_label
        if color_scheme == 'cluster': color_labels = via_object.labels
        if color_scheme == 'other': color_labels = other_labels

        n_true = len(set(color_labels))
        lin_col = np.linspace(0, 1, n_true)
        col = 0
        cmap = matplotlib.cm.get_cmap(cmap_scatter) #'twilight' is nice too
        cmap = cmap(np.linspace(0.01, 0.80, n_true)) #.95
        #cmap = cmap(np.linspace(0.5, 0.95, n_true))
        for color, group in zip(lin_col, sorted(set(color_labels))):
            color_ = np.asarray(cmap[col]).reshape(-1, 4)

            color_[0,3]=scatter_alpha

            where = np.where(np.array(color_labels) == group)[0]

            ax.scatter(embedding[where, 0], embedding[where, 1], label=group,
                       c=color_,
                       alpha=scatter_alpha, zorder=0, s=scatter_size, linewidths=marker_edgewidth) #plt.cm.rainbow(color))

            x_mean = embedding[where, 0].mean()
            y_mean = embedding[where, 1].mean()
            ax.text(x_mean, y_mean,'' , fontsize=fontsize, zorder=4,
                    path_effects=[PathEffects.withStroke(linewidth=linewidth, foreground='w')], weight='bold')#str(group)
            col += 1
        ax.set_facecolor(facecolor_)

    lengths = []
    colors = []
    lines = []
    linewidths = []
    count  =0
    #X, Y, U, V = interpolate_static_stream(X_grid[0], X_grid[1], V_grid[0],V_grid[1])
    s = Streamlines(X_grid[0], X_grid[1], V_grid[0], V_grid[1])
    #s = Streamlines(X,Y,U,V)

    for streamline in s.streamlines:
        random_seed+=1
        count +=1
        x, y = streamline
        #interpolate x, y data to handle nans
        x_ = np.array(x)
        nans, func_ = nan_helper(x_)
        x_[nans] = np.interp(func_(nans), func_(~nans), x_[~nans])


        y_ = np.array(y)
        nans, func_ = nan_helper(y_)
        y_[nans] = np.interp(func_(nans), func_(~nans), y_[~nans])


        # test=proj.transform_points(x=np.array(x),y=np.array(y),src_crs=proj)
        #points = np.array([x, y]).T.reshape(-1, 1, 2)
        points = np.array([x_, y_]).T.reshape(-1, 1, 2)
        #print('points')
        #print(points.shape)


        segments = np.concatenate([points[:-1], points[1:]], axis=1) #nx2x2

        n = len(segments)

        D = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=-1))/segment_length
        np.random.seed(random_seed)
        L = D.cumsum().reshape(n, 1) + np.random.uniform(0, 1)

        C = np.zeros((n, 4)) #3
        C[::-1] = (L * 1.5) % 1
        C[:,3]= alpha_animate

        lw = L.flatten().tolist()

        line = LineCollection(segments, color=C, linewidth=1)# 0.1 when changing linewidths in update
        #line = LineCollection(segments, color=C_locationbased, linewidth=1)
        lengths.append(L)

        colors.append(C)
        linewidths.append(lw)
        lines.append(line)

        ax.add_collection(line)
    print('total number of stream lines', count)

    ax.set_xlim(min(X_grid[0]), max(X_grid[0]), ax.set_xticks([]))
    ax.set_ylim(min(X_grid[1]), max(X_grid[1]), ax.set_yticks([]))
    plt.tight_layout()
    #print('colors', colors)
    def update(frame_no):
        cmap = matplotlib.cm.get_cmap(cmap_stream)
        #cmap = cmap(np.linspace(0.1, 0.2, 100)) #darker portion
        cmap = cmap(np.linspace(0.8, 0.9, 100)) #lighter portion
        for i in range(len(lines)):
            lengths[i] -= 0.05
            # esthetic factors here by adding 0.1 and doing some clipping, 0.1 ensures no "hard blacks"
            colors[i][::-1] =  np.clip(0.1+(lengths[i] * 1.5) % 1,0.2,0.9)
            colors[i][:, 3] = alpha_animate
            temp = (lengths[i] * 1) % 2 #*1.5

            #temp = (lengths[i] * 1.5) % 1  # *1.5 original until Sep 7 2022
            linewidths[i] = temp.flatten().tolist()
            #if i==5: print('temp', linewidths[i])
            '''
            if i%5 ==0:
                print('lengths',i, lengths[i])
                colors[i][::-1] = (lengths[i] * 1.5) % 1
                colors[i][:, 0] = 1
            '''

            #CMAP COLORS
            #cmap_colors = [cmap(j) for j in colors[i][:,0]] #when using full cmap_stream
            cmap_colors = [cmap[int(j * 100)] for j in colors[i][:, 0]] #when using truncated cmap_stream
            '''
            cmap_colors = [cmap[int(j*100)] for j in colors[i][:, 0]]
            linewidths[i] = [f[0]*2 for f in cmap_colors]
            if i ==5: print('colors', [f[0]for f in cmap_colors])
            '''
            for row in range(colors[i].shape[0]):
                colors[i][row,:] = cmap_colors[row][0:4]
                #colors[i][row, 3] = (1-colors[i][row][0])*0.6#alpha_animate
                #linewidths[i][row] = 2-((colors[i][row][0])%2) #1-colors[i]... #until 7 sept 2022

                #if color_stream is not None: colors[i][row, :] =  matplotlib.colors.to_rgba_array(color_stream)[0] #monochrome is nice 1 or 0
            #if i == 5: print('lw', linewidths[i])
            colors[i][:, 3] = alpha_animate
            lines[i].set_linewidth(linewidths[i])
            lines[i].set_color(colors[i])
        pbar.update()

    n = 250#27

    animation = FuncAnimation(fig, update, frames=n, interval=40)
    pbar = tqdm.tqdm(total=n)
    pbar.close()
    animation.save(saveto, writer='imagemagick', fps=25)
    #animation.save('/home/shobi/Trajectory/Datasets/Toy3/wind_density_ffmpeg.mp4', writer='ffmpeg', fps=60)

    #fig.patch.set_visible(False)
    #ax.axis('off')
    plt.show()
    return fig, ax


def via_streamplot(via_object, embedding:ndarray=None , density_grid:float=0.5, arrow_size:float=0.7, arrow_color:str = 'k',
arrow_style="-|>",  max_length:int=4, linewidth:float=1,min_mass = 1, cutoff_perc:int = 5,scatter_size:int=500, scatter_alpha:float=0.5,marker_edgewidth:float=0.1, density_stream:int = 2, smooth_transition:int=1, smooth_grid:float=0.5, color_scheme:str = 'annotation', add_outline_clusters:bool=False, cluster_outline_edgewidth = 0.001,gp_color = 'white', bg_color='black' , dpi=300 , title='Streamplot', b_bias=20, n_neighbors_velocity_grid=None, other_labels:list = None,use_sequentially_augmented=False, cmap_str:str='rainbow'):
    '''
    Construct vector streamplot on the embedding to show a fine-grained view of inferred directions in the trajectory

    :param via_object:
    :param embedding:  np.ndarray of shape (n_samples, 2) umap or other 2-d embedding on which to project the directionality of cells
    :param density_grid:
    :param arrow_size:
    :param arrow_color:
    :param arrow_style:
    :param max_length:
    :param linewidth:  width of  lines in streamplot, default = 1
    :param min_mass:
    :param cutoff_perc:
    :param scatter_size: size of scatter points default =500
    :param scatter_alpha: transpsarency of scatter points
    :param marker_edgewidth: width of outline arround each scatter point, default = 0.1
    :param density_stream:
    :param smooth_transition:
    :param smooth_grid:
    :param color_scheme: str, default = 'annotation' corresponds to self.true_labels. Other options are 'time' (uses single-cell pseudotime) and 'cluster' (via cluster graph) and 'other'
    :param add_outline_clusters:
    :param cluster_outline_edgewidth:
    :param gp_color:
    :param bg_color:
    :param dpi:
    :param title:
    :param b_bias: default = 20. higher value makes the forward bias of pseudotime stronger
    :param n_neighbors_velocity_grid:
    :param other_labels: list (will be used for the color scheme)
    :param use_sequentially_augmented:
    :param cmap_str:
    :return: fig, ax
    '''
    """
    
   
  
   Parameters
   ----------
   X_emb:

   scatter_size: int, default = 500

   linewidth:

   marker_edgewidth: 

   streamplot matplotlib.pyplot instance of fine-grained trajectories drawn on top of scatter plot
   """

    import matplotlib.patheffects as PathEffects
    if embedding is None:
        embedding = via_object.embedding
        if embedding is None:
            print(f'{datetime.now()}\tWARNING: please assign ambedding attribute to via_object as v0.embedding = ndarray of [n_cells x 2]')

    V_emb = via_object._velocity_embedding(embedding, smooth_transition,b=b_bias, use_sequentially_augmented=use_sequentially_augmented)

    V_emb *=20 #5


    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=embedding,
        V_emb=V_emb,
        density=density_grid,
        smooth=smooth_grid,
        min_mass=min_mass,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=cutoff_perc, n_neighbors=n_neighbors_velocity_grid )

    # adapted from : https://github.com/theislab/scvelo/blob/1805ab4a72d3f34496f0ef246500a159f619d3a2/scvelo/plotting/velocity_embedding_grid.py#L27
    lengths = np.sqrt((V_grid ** 2).sum(0))

    linewidth = 1 if linewidth is None else linewidth
    #linewidth *= 2 * lengths / np.percentile(lengths[~np.isnan(lengths)],90)
    linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()

    #linewidth=0.5
    fig, ax = plt.subplots(dpi=dpi)
    ax.grid(False)
    ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], color=arrow_color, arrowsize=arrow_size, arrowstyle=arrow_style, zorder = 3, linewidth=linewidth, density = density_stream, maxlength=max_length)

    #num_cluster = len(set(super_cluster_labels))

    if add_outline_clusters:
        # add black outline to outer cells and a white inner rim
        #adapted from scanpy (scVelo utils adapts this from scanpy)
        gp_size = (2 * (scatter_size * cluster_outline_edgewidth *.1) + 0.1*scatter_size) ** 2

        bg_size = (2 * (scatter_size * cluster_outline_edgewidth)+ math.sqrt(gp_size)) ** 2

        ax.scatter(embedding[:, 0],embedding[:, 1], s=bg_size, marker=".", c=bg_color, zorder=-2)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=gp_size, marker=".", c=gp_color, zorder=-1)

    if color_scheme == 'time':
        ax.scatter(embedding[:,0],embedding[:,1], c=via_object.single_cell_pt_markov,alpha=scatter_alpha,  zorder = 0, s=scatter_size, linewidths=marker_edgewidth, cmap = 'viridis_r')
    else:
        if color_scheme == 'annotation':color_labels = via_object.true_label
        if color_scheme == 'cluster': color_labels= via_object.labels
        if other_labels is not None: color_labels = other_labels

        cmap_ = plt.get_cmap(cmap_str)
        #plt.cm.rainbow(color)

        line = np.linspace(0, 1, len(set(color_labels)))
        for color, group in zip(line, sorted(set(color_labels))):
            where = np.where(np.array(color_labels) == group)[0]
            ax.scatter(embedding[where, 0], embedding[where, 1], label=group,
                        c=np.asarray(cmap_(color)).reshape(-1, 4),
                        alpha=scatter_alpha,  zorder = 0, s=scatter_size, linewidths=marker_edgewidth)

            x_mean = embedding[where, 0].mean()
            y_mean = embedding[where, 1].mean()
            ax.text(x_mean, y_mean, str(group), fontsize=5, zorder=4, path_effects = [PathEffects.withStroke(linewidth=1, foreground='w')], weight = 'bold')

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.set_title(title)
    return fig, ax

def get_gene_expression(via0, gene_exp:pd.DataFrame, cmap:str='jet', 
                        dpi:int=150, marker_genes:list = [], linewidth:float = 2.0,
                        n_splines:int=10, spline_order:int=4, fontsize_:int=8, marker_lineages=[]):
    '''
    :param gene_exp: dataframe where columns are features (gene) and rows are single cells
    :param cmap: default: 'jet'
    :param dpi: default:150
    :param marker_genes: Default is to use all genes in gene_exp. other provide a list of marker genes that will be used from gene_exp.
    :param linewidth: default:2
    :param n_slines: default:10
    :param spline_order: default:4
    :param marker_lineages: Default is to use all lineage pathways. other provide a list of lineage number (terminal cluster number).
    :return: fig, axs
    '''

    if len(marker_lineages)==0: marker_lineages=via0.terminal_clusters

    if len(marker_genes) >0: gene_exp=gene_exp[marker_genes]
    sc_pt = via0.single_cell_pt_markov
    sc_bp_original = via0.single_cell_bp
    n_terminal_states = sc_bp_original.shape[1]

    palette = cm.get_cmap(cmap, n_terminal_states)
    cmap_ = palette(range(n_terminal_states))
    n_genes = gene_exp.shape[1]

    fig_nrows, mod = divmod(n_genes, 4)
    if mod == 0: fig_nrows = fig_nrows
    if mod != 0: fig_nrows += 1

    fig_ncols = 4
    fig, axs = plt.subplots(fig_nrows, fig_ncols, dpi=dpi)
    fig.patch.set_visible(False)
    i_gene = 0  # counter for number of genes
    i_terminal = 0 #counter for terminal cluster
    # for i in range(n_terminal_states): #[0]

    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if (i_gene < n_genes):
                for i_terminal in range(n_terminal_states):
                    sc_bp = sc_bp_original.copy()
                    if (via0.terminal_clusters[i_terminal] in marker_lineages and len(np.where(sc_bp[:, i_terminal] > 0.9)[ 0]) > 0): # check if terminal state is in marker_lineage and in case this terminal state i cannot be reached (sc_bp is all 0)
                        gene_i = gene_exp.columns[i_gene]
                        loc_i = np.where(sc_bp[:, i_terminal] > 0.9)[0]
                        val_pt = [sc_pt[pt_i] for pt_i in loc_i]  # TODO,  replace with array to speed up

                        max_val_pt = max(val_pt)

                        loc_i_bp = np.where(sc_bp[:, i_terminal] > 0.000)[0]  # 0.001
                        loc_i_sc = np.where(np.asarray(sc_pt) <= max_val_pt)[0]

                        loc_ = np.intersect1d(loc_i_bp, loc_i_sc)

                        gam_in = np.asarray(sc_pt)[loc_]
                        x = gam_in.reshape(-1, 1)

                        y = np.asarray(gene_exp[gene_i])[loc_].reshape(-1, 1)

                        weights = np.asarray(sc_bp[:, i_terminal])[loc_].reshape(-1, 1)

                        if len(loc_) > 1:
                            geneGAM = pg.LinearGAM(n_splines=n_splines, spline_order=spline_order, lam=10).fit(x, y, weights=weights)
                            xval = np.linspace(min(sc_pt), max_val_pt, 100 * 2)
                            yg = geneGAM.predict(X=xval)

                        else:
                            print(f'{datetime.now()}\tLineage {i_terminal} cannot be reached. Exclude this lineage in trend plotting')

                        if fig_nrows >1:
                            axs[r,c].plot(xval, yg, color=cmap_[i_terminal], linewidth=linewidth, zorder=3, label=f"Lineage:{via0.terminal_clusters[i_terminal]}")
                            axs[r, c].set_title(gene_i,fontsize=fontsize_)
                            # Set tick font size
                            for label in (axs[r,c].get_xticklabels() + axs[r,c].get_yticklabels()):
                                label.set_fontsize(fontsize_-1)
                            if i_gene == n_genes -1:
                                axs[r,c].legend(frameon=False, fontsize=fontsize_)
                                axs[r, c].set_xlabel('Time', fontsize=fontsize_)
                                axs[r, c].set_ylabel('Intensity', fontsize=fontsize_)
                            axs[r,c].spines['top'].set_visible(False)
                            axs[r,c].spines['right'].set_visible(False)
                            axs[r, c].grid(False)
                        else:
                            axs[c].plot(xval, yg, color=cmap_[i_terminal], linewidth=linewidth, zorder=3,   label=f"Lineage:{via0.terminal_clusters[i_terminal]}")
                            axs[c].set_title(gene_i, fontsize=fontsize_)
                            # Set tick font size
                            for label in (axs[c].get_xticklabels() + axs[c].get_yticklabels()):
                                label.set_fontsize(fontsize_-1)
                            if i_gene == n_genes -1:
                                axs[c].legend(frameon=False,fontsize=fontsize_)
                                axs[ c].set_xlabel('Time', fontsize=fontsize_)
                                axs[ c].set_ylabel('Intensity', fontsize=fontsize_)
                            axs[c].spines['top'].set_visible(False)
                            axs[c].spines['right'].set_visible(False)
                            axs[c].grid(False)
                i_gene+=1
            else:
                if fig_nrows > 1:
                    axs[r,c].axis('off')
                    axs[r, c].grid(False)
                else:
                    axs[c].axis('off')
                    axs[c].grid(False)
    return fig, axs

def draw_trajectory_gams(via_object, via_fine=None, embedding: ndarray=None, idx:Optional[list]=None,
                         title_str:str= "Pseudotime", draw_all_curves:bool=True, arrow_width_scale_factor:float=15.0,
                         scatter_size:float=50, scatter_alpha:float=0.5,
                         linewidth:float=1.5, marker_edgewidth:float=1, cmap_pseudotime:str='viridis_r',dpi:int=150,highlight_terminal_states:bool=True, use_maxout_edgelist:bool =False):
    '''

    projects the graph based coarse trajectory onto a umap/tsne embedding

    :param via_object: via object
    :param via_fine: via object suggest to use via_object only unless you found that running via_fine gave better pathways
    :param embedding: 2d array [n_samples x 2] with x and y coordinates of all n_samples. Umap, tsne, pca OR use the via computed embedding via_object.embedding
    :param idx: default: None. Or List. if you had previously computed a umap/tsne (embedding) only on a subset of the total n_samples (subsampled as per idx), then the via objects and results will be indexed according to idx too
    :param title_str: title of figure
    :param draw_all_curves: if the clustergraph has too many edges to project in a visually interpretable way, set this to False to get a simplified view of the graph pathways
    :param arrow_width_scale_factor:
    :param scatter_size:
    :param scatter_alpha:
    :param linewidth:
    :param marker_edgewidth:
    :param cmap_pseudotime:
    :param dpi: int default = 150. Use 300 for paper figures
    :param highlight_terminal_states: whether or not to highlight/distinguish the clusters which are detected as the terminal states by via
    :return: f, ax1, ax2
    '''

    if embedding is None:
        embedding = via_object.embedding
        if embedding is None: print(f'{datetime.now()}\t ERROR please provide an embedding or compute using via_mds() or via_umap()')
    if via_fine is None:
        via_fine = via_object
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if idx is None: idx = np.arange(0, via_object.nsamples)
    cluster_labels = list(np.asarray(via_fine.labels)[idx])
    super_cluster_labels = list(np.asarray(via_object.labels)[idx])
    super_edgelist = via_object.edgelist
    if use_maxout_edgelist==True:
        super_edgelist =via_object.edgelist_maxout
    true_label = list(np.asarray(via_fine.true_label)[idx])
    knn = via_fine.knn
    ncomp = via_fine.ncomp
    if len(via_fine.revised_super_terminal_clusters)>0:
        final_super_terminal = via_fine.revised_super_terminal_clusters
    else: final_super_terminal = via_fine.terminal_clusters

    sub_terminal_clusters = via_fine.terminal_clusters

    
    sc_pt_markov = list(np.asarray(via_fine.single_cell_pt_markov)[idx])
    super_root = via_object.root[0]



    sc_supercluster_nn = sc_loc_ofsuperCluster_PCAspace(via_object, via_fine, np.arange(0, len(cluster_labels)))
    # draw_all_curves. True draws all the curves in the piegraph, False simplifies the number of edges
    # arrow_width_scale_factor: size of the arrow head
    X_dimred = embedding * 1. / np.max(embedding, axis=0)
    x = X_dimred[:, 0]
    y = X_dimred[:, 1]
    max_x = np.percentile(x, 90)
    noise0 = max_x / 1000

    df = pd.DataFrame({'x': x, 'y': y, 'cluster': cluster_labels, 'super_cluster': super_cluster_labels,
                       'projected_sc_pt': sc_pt_markov},
                      columns=['x', 'y', 'cluster', 'super_cluster', 'projected_sc_pt'])
    df_mean = df.groupby('cluster', as_index=False).mean()
    sub_cluster_isin_supercluster = df_mean[['cluster', 'super_cluster']]

    sub_cluster_isin_supercluster = sub_cluster_isin_supercluster.sort_values(by='cluster')
    sub_cluster_isin_supercluster['int_supercluster'] = sub_cluster_isin_supercluster['super_cluster'].round(0).astype(
        int)

    df_super_mean = df.groupby('super_cluster', as_index=False).mean()
    pt = df_super_mean['projected_sc_pt'].values

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[20, 10],dpi=dpi)
    num_true_group = len(set(true_label))
    num_cluster = len(set(super_cluster_labels))
    line = np.linspace(0, 1, num_true_group)
    for color, group in zip(line, sorted(set(true_label))):
        where = np.where(np.array(true_label) == group)[0]
        ax1.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group, c=np.asarray(plt.cm.rainbow(color)).reshape(-1, 4),
                    alpha=scatter_alpha, s=scatter_size, linewidths=marker_edgewidth*.1)  # 10 # 0.5 and 4
    ax1.legend(fontsize=6, frameon = False)
    ax1.set_title('True Labels: ncomps:' + str(ncomp) + '. knn:' + str(knn))

    G_orange = ig.Graph(n=num_cluster, edges=super_edgelist)
    ll_ = []  # this can be activated if you intend to simplify the curves
    for fst_i in final_super_terminal:

        path_orange = G_orange.get_shortest_paths(super_root, to=fst_i)[0]
        len_path_orange = len(path_orange)
        for enum_edge, edge_fst in enumerate(path_orange):
            if enum_edge < (len_path_orange - 1):
                ll_.append((edge_fst, path_orange[enum_edge + 1]))

    edges_to_draw = super_edgelist if draw_all_curves else list(set(ll_))
    for e_i, (start, end) in enumerate(edges_to_draw):
        if pt[start] >= pt[end]:
            start, end = end, start

        x_i_start = df[df['super_cluster'] == start]['x'].values
        y_i_start = df[df['super_cluster'] == start]['y'].values
        x_i_end = df[df['super_cluster'] == end]['x'].values
        y_i_end = df[df['super_cluster'] == end]['y'].values


        super_start_x = X_dimred[sc_supercluster_nn[start], 0]
        super_end_x = X_dimred[sc_supercluster_nn[end], 0]
        super_start_y = X_dimred[sc_supercluster_nn[start], 1]
        super_end_y = X_dimred[sc_supercluster_nn[end], 1]
        direction_arrow = -1 if super_start_x > super_end_x else 1

        minx = min(super_start_x, super_end_x)
        maxx = max(super_start_x, super_end_x)

        miny = min(super_start_y, super_end_y)
        maxy = max(super_start_y, super_end_y)

        x_val = np.concatenate([x_i_start, x_i_end])
        y_val = np.concatenate([y_i_start, y_i_end])

        idx_keep = np.where((x_val <= maxx) & (x_val >= minx))[0]
        idy_keep = np.where((y_val <= maxy) & (y_val >= miny))[0]

        idx_keep = np.intersect1d(idy_keep, idx_keep)

        x_val = x_val[idx_keep]
        y_val = y_val[idx_keep]

        super_mid_x = (super_start_x + super_end_x) / 2
        super_mid_y = (super_start_y + super_end_y) / 2
        from scipy.spatial import distance

        very_straight = False
        straight_level = 3
        noise = noise0
        x_super = np.array(
            [super_start_x, super_end_x, super_start_x, super_end_x, super_start_x, super_end_x, super_start_x,
             super_end_x, super_start_x + noise, super_end_x + noise,
             super_start_x - noise, super_end_x - noise])
        y_super = np.array(
            [super_start_y, super_end_y, super_start_y, super_end_y, super_start_y, super_end_y, super_start_y,
             super_end_y, super_start_y + noise, super_end_y + noise,
             super_start_y - noise, super_end_y - noise])

        if abs(minx - maxx) <= 1:
            very_straight = True
            straight_level = 10
            x_super = np.append(x_super, super_mid_x)
            y_super = np.append(y_super, super_mid_y)

        for i in range(straight_level):  # DO THE SAME FOR A MIDPOINT TOO
            y_super = np.concatenate([y_super, y_super])
            x_super = np.concatenate([x_super, x_super])

        list_selected_clus = list(zip(x_val, y_val))

        if len(list_selected_clus) >= 1 & very_straight:
            dist = distance.cdist([(super_mid_x, super_mid_y)], list_selected_clus, 'euclidean')
            k = min(2, len(list_selected_clus))
            midpoint_loc = dist[0].argsort()[:k]

            midpoint_xy = []
            for i in range(k):
                midpoint_xy.append(list_selected_clus[midpoint_loc[i]])

            noise = noise0 * 2

            if k == 1:
                mid_x = np.array([midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][0] - noise])
                mid_y = np.array([midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][1] - noise])
            if k == 2:
                mid_x = np.array(
                    [midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][0] - noise, midpoint_xy[1][0],
                     midpoint_xy[1][0] + noise, midpoint_xy[1][0] - noise])
                mid_y = np.array(
                    [midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][1] - noise, midpoint_xy[1][1],
                     midpoint_xy[1][1] + noise, midpoint_xy[1][1] - noise])
            for i in range(3):
                mid_x = np.concatenate([mid_x, mid_x])
                mid_y = np.concatenate([mid_y, mid_y])

            x_super = np.concatenate([x_super, mid_x])
            y_super = np.concatenate([y_super, mid_y])
        x_val = np.concatenate([x_val, x_super])
        y_val = np.concatenate([y_val, y_super])

        x_val = x_val.reshape((len(x_val), -1))
        y_val = y_val.reshape((len(y_val), -1))
        xp = np.linspace(minx, maxx, 500)

        gam50 = pg.LinearGAM(n_splines=4, spline_order=3, lam=10).gridsearch(x_val, y_val)
        XX = gam50.generate_X_grid(term=0, n=500)
        preds = gam50.predict(XX)

        idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]
        ax2.plot(XX, preds, linewidth=linewidth, c='#323538')  # 3.5#1.5


        mean_temp = np.mean(xp[idx_keep])
        closest_val = xp[idx_keep][0]
        closest_loc = idx_keep[0]

        for i, xp_val in enumerate(xp[idx_keep]):
            if abs(xp_val - mean_temp) < abs(closest_val - mean_temp):
                closest_val = xp_val
                closest_loc = idx_keep[i]
        step = 1

        head_width = noise * arrow_width_scale_factor  # arrow_width needs to be adjusted sometimes # 40#30  ##0.2 #0.05 for mESC #0.00001 (#for 2MORGAN and others) # 0.5#1
        if direction_arrow == 1:
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc + step] - xp[closest_loc],
                      preds[closest_loc + step] - preds[closest_loc], shape='full', lw=0, length_includes_head=False,
                      head_width=head_width, color='#323538')

        else:
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc - step] - xp[closest_loc],
                      preds[closest_loc - step] - preds[closest_loc], shape='full', lw=0, length_includes_head=False,
                      head_width=head_width, color='#323538')

    c_edge = []
    width_edge = []
    pen_color = []
    super_cluster_label = []
    terminal_count_ = 0
    dot_size = []

    for i in sc_supercluster_nn:
        if i in final_super_terminal:
            print(f'{datetime.now()}\tSuper cluster {i} is a super terminal with sub_terminal cluster',
                  sub_terminal_clusters[terminal_count_])
            c_edge.append('yellow')  # ('yellow')
            if highlight_terminal_states == True:
                width_edge.append(2)
                super_cluster_label.append('TS' + str(sub_terminal_clusters[terminal_count_]))
            else:
                width_edge.append(0)
                super_cluster_label.append('')
            pen_color.append('black')
            # super_cluster_label.append('TS' + str(i))  # +'('+str(i)+')')
             # +'('+str(i)+')')
            dot_size.append(60)  # 60
            terminal_count_ = terminal_count_ + 1
        else:
            width_edge.append(0)
            c_edge.append('black')
            pen_color.append('red')
            super_cluster_label.append(str(' '))  # i or ' '
            dot_size.append(00)  # 20

    ax2.set_title(title_str)

    im2 =ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=sc_pt_markov, cmap=cmap_pseudotime,  s=0.01)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im2, cax=cax, orientation='vertical', label='pseudotime') #to avoid lines drawn on the colorbar we need an image instance without alpha variable
    ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=sc_pt_markov, cmap=cmap_pseudotime, alpha=scatter_alpha,
                s=scatter_size, linewidths=marker_edgewidth*.1)
    count_ = 0
    loci = [sc_supercluster_nn[key] for key in sc_supercluster_nn]
    for i, c, w, pc, dsz, lab in zip(loci, c_edge, width_edge, pen_color, dot_size,
                                     super_cluster_label):  # sc_supercluster_nn
        ax2.scatter(X_dimred[i, 0], X_dimred[i, 1], c='black', s=dsz, edgecolors=c, linewidth=w)
        ax2.annotate(str(lab), xy=(X_dimred[i, 0], X_dimred[i, 1]))
        count_ = count_ + 1

    ax1.grid(False)
    ax2.grid(False)
    f.patch.set_visible(False)
    ax1.axis('off')
    ax2.axis('off')
    return f, ax1, ax2

def _draw_sc_lineage_probability_solo(via_object, via_fine=None, embedding:ndarray=None, 
                                      idx=None, cmap_name='plasma', dpi=150, vmax=99):
    '''

    :param via_object:
    :param via_fine: None (or via_object)
    :param embedding: ndarray n_samples x 2
    :param idx:
    :param cmap_name:
    :param dpi:
    :param vmax: int default =99. vmax of the scatterplot cmap scale where vmax is the vmax'th percentile of the lienage probabilities in that lineage
    :return:
    '''
    # embedding is the full or downsampled 2D representation of the full dataset.
    # idx is the list of indices of the full dataset for which the embedding is available. if the dataset is very large the the user only has the visual embedding for a subsample of the data, then these idx can be carried forward
    # idx is the selected indices of the downsampled samples used in the visualization
    # G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it's made on full sample
    # knn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding
    #   that corresponds to the single cells in the full graph
    if idx is None: idx = np.arange(0, via_object.nsamples)
    if via_fine is None: via_fine = via_object
    G = via_object.full_graph_shortpath
    knn_hnsw = _make_knn_embeddedspace(embedding)
    y_root = []
    x_root = []
    root1_list = []
    p1_sc_bp = via_fine.single_cell_bp[idx, :]
    p1_labels = np.asarray(via_fine.labels)[idx]
    p1_cc = via_fine.connected_comp_labels
    p1_sc_pt_markov = list(np.asarray(via_fine.single_cell_pt_markov)[idx])
    X_data = via_fine.data

    X_ds = X_data[idx, :]
    p_ds = hnswlib.Index(space='l2', dim=X_ds.shape[1])
    p_ds.init_index(max_elements=X_ds.shape[0], ef_construction=200, M=16)
    p_ds.add_items(X_ds)
    p_ds.set_ef(50)
    num_cluster = len(set(via_fine.labels))
    G_orange = ig.Graph(n=num_cluster, edges=via_fine.edgelist_maxout, edge_attrs={'weight':via_fine.edgeweights_maxout})
    for ii, r_i in enumerate(via_fine.root):
        loc_i = np.where(p1_labels == via_fine.root[ii])[0]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]

        labels_root, distances_root = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_root.append(embedding[labels_root, 0][0])
        y_root.append(embedding[labels_root, 1][0])

        labelsroot1, distances1 = via_fine.knn_struct.knn_query(X_ds[labels_root[0][0], :], k=1)
        root1_list.append(labelsroot1[0][0])
        for fst_i in via_fine.terminal_clusters:
            path_orange = G_orange.get_shortest_paths(via_fine.root[ii], to=fst_i)[0]
            #if the roots is in the same component as the terminal cluster, then print the path to output
            if len(path_orange)>0:
                print( f"{datetime.now()}\tCluster path on clustergraph starting from Root Cluster {via_fine.root[ii]} to Terminal Cluster {fst_i} : follows {path_orange} ")


    # single-cell branch probability evolution probability
    for i, ti in enumerate(via_fine.terminal_clusters):
        fig, ax = plt.subplots(dpi=dpi)
        plot_sc_pb(ax, fig, embedding, p1_sc_bp[:, i], ti=ti, cmap_name=cmap_name)

        loc_i = np.where(p1_labels == ti)[0]
        val_pt = [p1_sc_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt, 50)  # 50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
        x = [embedding[xi, 0] for xi in
             loc_i]  # location of sc nearest to average location of terminal clus in the EMBEDDED space
        y = [embedding[yi, 1] for yi in loc_i]
        labels, distances = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]),
                                               k=1)  # knn_hnsw is knn of embedded space
        x_sc = embedding[labels[0], 0]  # terminal sc location in the embedded space
        y_sc = embedding[labels[0], 1]
        start_time = time.time()

        labelsq1, distances1 =via_fine.knn_struct.knn_query(X_ds[labels[0][0], :],
                                                       k=1)  # find the nearest neighbor in the PCA-space full graph

        path = G.get_shortest_paths(root1_list[p1_cc[ti]], to=labelsq1[0][0])  # weights='weight')
        # G is the knn of all sc points

        # get the nearest-neighbor in this downsampled PCA-space graph. These will make the new path-way points
        path = path[0]

        # clusters of path
        cluster_path = []
        for cell_ in path:
            cluster_path.append(via_fine.labels[cell_])

        # print(colored('cluster_path', 'green'), colored('terminal state: ', 'blue'), ti, cluster_path)
        revised_cluster_path = []
        revised_sc_path = []
        for enum_i, clus in enumerate(cluster_path):
            num_instances_clus = cluster_path.count(clus)
            if (clus == cluster_path[0]) | (clus == cluster_path[-1]):
                revised_cluster_path.append(clus)
                revised_sc_path.append(path[enum_i])
            else:
                if num_instances_clus > 1:  # typically intermediate stages spend a few transitions at the sc level within a cluster
                    if clus not in revised_cluster_path: revised_cluster_path.append(clus)  # cluster
                    revised_sc_path.append(path[enum_i])  # index of single cell
        print(f"{datetime.now()}\tCluster level path on sc-knnGraph from Root Cluster {via_fine.root[p1_cc[ti]]} to Terminal Cluster {ti} along path: {revised_cluster_path}")

        fig.patch.set_visible(False)
        ax.axis('off')
    return fig, ax

def plot_edgebundle_viagraph(ax=None, hammer_bundle=None, layout:ndarray=None, CSM:ndarray=None, velocity_weight:float=None, pt:list=None, alpha_bundle=1, linewidth_bundle=2, edge_color='darkblue',headwidth_bundle=0.1, arrow_frequency=0.05, show_direction=True,ax_text:bool=True, title:str='', plot_clusters:bool=False, cmap:str='viridis', via_object=None, fontsize:float=9, dpi:int=300):
    '''
    this plots the edgebundles on the via clustergraph level and also adds the relevant arrow directions based on the TI directionality

    :param ax: axis to plot on
    :param hammer_bundle: hammerbundle object with coordinates of all the edges to draw. self.hammer
    :param layout: coords of cluster nodes
    :param CSM: cosine similarity matrix. cosine similarity between the RNA velocity between neighbors and the change in gene expression between these neighbors. Only used when available
    :param velocity_weight: percentage weightage given to the RNA velocity based transition matrix
    :param pt: cluster-level pseudotime (or other intensity level of features at average-cluster level)
    :param alpha_bundle: alpha when drawing lines
    :param linewidth_bundle: linewidth of bundled lines
    :param edge_color:
    :param headwidth_bundle: headwidth of arrows used in bundled edges
    :param arrow_frequency: min dist between arrows (bundled edges otherwise have overcrowding of arrows)
    :param show_direction: bool default True. will draw arrows along the lines to indicate direction
    :param plot_clusters: bool default False. When this function is called on its own (and not from within draw_piechart_graph() then via_object must be provided
    :param ax_text: bool default True. Show labels of the clusters with the cluster population and PARC cluster label
    :param fontsize: float default 9 Font size of labels
    :return: fig, ax with bundled edges plotted
    '''
    return_fig_ax = False #return only the ax
    if ax == None:
        fig, ax = plt.subplots(dpi=dpi)
        ax.set_facecolor('white')
        fig.patch.set_visible(False)
        return_fig_ax = True
    if (plot_clusters == True) and (via_object is None):
        print('Warning: please provide a via object in order to plot the clusters on the graph')
    if via_object is not None:
        if hammer_bundle is None: hammer_bundle = via_object.hammerbundle_cluster
        if layout is None: layout = via_object.graph_node_pos
        if CSM is None: CSM= via_object.CSM
        if velocity_weight is None: velocity_weight = via_object.velo_weight
        if pt is None: pt = via_object.scaled_hitting_times

    x_ = [l[0] for l in layout ]
    y_ =  [l[1] for l in layout ]
    #min_x, max_x = min(x_), max(x_)
    #min_y, max_y = min(y_), max(y_)
    delta_x =  max(x_)- min(x_)

    delta_y = max(y_)- min(y_)

    layout = np.asarray(layout)
    # make a knn so we can find which clustergraph nodes the segments start and end at

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(layout)
    # get each segment. these are separated by nans.
    hbnp = hammer_bundle.to_numpy()
    splits = (np.isnan(hbnp[:, 0])).nonzero()[0] #location of each nan values
    edgelist_segments = []
    start = 0
    segments = []
    arrow_coords=[]
    for stop in splits:
        seg = hbnp[start:stop, :]
        segments.append(seg)
        start = stop

    n = 1  # every nth segment is plotted
    step = 1
    for seg in segments[::n]:
        do_arrow=True

        #seg_weight = max(0.3, math.log(1+seg[-1,2]))
        seg_weight = max(0.05, math.log(1 + seg[-1, 2]))

        #print('seg weight', seg_weight)
        seg = seg[:,0:2].reshape(-1,2)
        seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

        start=neigh.kneighbors(seg_p[0, :].reshape(1, -1), return_distance=False)[0][0]
        end = neigh.kneighbors(seg_p[-1, :].reshape(1, -1), return_distance=False)[0][0]
        #print('start,end',[start, end])

        if ([start, end] in edgelist_segments)|([end,start] in edgelist_segments):
            do_arrow = False
        edgelist_segments.append([start,end])

        direction_ = infer_direction_piegraph(start_node=start, end_node=end, CSM=CSM, velocity_weight=velocity_weight, pt=pt)

        direction = -1 if direction_ <0 else 1


        ax.plot(seg_p[:, 0], seg_p[:, 1],linewidth=linewidth_bundle*seg_weight, alpha=alpha_bundle, color=edge_color )
        mid_point = math.floor(seg_p.shape[0] / 2)

        if len(arrow_coords)>0: #dont draw arrows in overlapping segments
            for v1 in arrow_coords:
                dist_ = dist_points(v1,v2=[seg_p[mid_point, 0], seg_p[mid_point, 1]])
                #print('dist between points', dist_)
                if dist_< arrow_frequency*delta_x: do_arrow=False
                if dist_< arrow_frequency*delta_y: do_arrow=False

        if (do_arrow==True) & (seg_p.shape[0]>3):
            ax.arrow(seg_p[mid_point, 0], seg_p[mid_point, 1],
                 seg_p[mid_point + (direction * step), 0] - seg_p[mid_point, 0],
                 seg_p[mid_point + (direction * step), 1] - seg_p[mid_point, 1],
                 lw=0, length_includes_head=False, head_width=headwidth_bundle, color=edge_color,shape='full', alpha= 0.6, zorder=5)
            arrow_coords.append([seg_p[mid_point, 0], seg_p[mid_point, 1]])
        if plot_clusters == True:
            group_pop = np.ones([layout.shape[0], 1])

            if via_object is not None:
                for group_i in set(via_object.labels):
                    #n_groups = len(set(via_object.labels))
                    loc_i = np.where(via_object.labels == group_i)[0]
                    group_pop[group_i] = len(loc_i)
            gp_scaling = 1000 / max(group_pop)  # 500 / max(group_pop)
            group_pop_scale = group_pop * gp_scaling * 0.5


            c_edge, l_width = [], []
            if via_object is not None: terminal_clusters_placeholder =  via_object.terminal_clusters
            else: terminal_clusters_placeholder = []
            for ei, pti in enumerate(pt):
                if ei in terminal_clusters_placeholder:
                    c_edge.append('red')
                    l_width.append(1.5)
                else:
                    c_edge.append('gray')
                    l_width.append(0.0)

            ax.scatter(layout[:, 0], layout[:, 1], s=group_pop_scale, c=pt, cmap=cmap,
                               edgecolors=c_edge,
                               alpha=1, zorder=3, linewidth=l_width)
            if ax_text:
                x_max_range = np.amax(layout[:, 0]) / 100
                y_max_range = np.amax(layout[:, 1]) / 100

                for ii in range(layout.shape[0]):
                    ax.text(layout[ii, 0] + max(x_max_range, y_max_range),
                              layout[ii, 1] + min(x_max_range, y_max_range),
                              'C' + str(ii) + 'pop' + str(int(group_pop[ii][0])),
                              color='black', zorder=4, fontsize=fontsize)
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_facecolor('white')
    if return_fig_ax==True:
        return fig, ax
    else: return ax

def _slow_sklearn_mds(via_graph: csr_matrix, X_pca:ndarray, t_diff_op:int=1):
    '''

    :param via_graph: via_graph = v0.csr_full_graph #single cell knn graph representation based on hnsw
    :param t_diff_op:
    :param X_pca ndarray adata_counts.obsm['X_pca'][:, 0:ncomps]
    :return: ndarray
    '''
    from sklearn.preprocessing import normalize

    via_graph.data = np.clip(via_graph.data, np.percentile(via_graph.data, 10), np.percentile(via_graph.data, 90))
    row_stoch = normalize(via_graph, norm='l1', axis=1)

    # note that the edge weights are affinities in via_graph

    from sklearn import manifold
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=0,
        dissimilarity="precomputed",
        n_jobs=2,
    )

    row_stoch = row_stoch ** t_diff_op #level of diffusion

    temp = csr_matrix(X_pca)

    X_mds = row_stoch * temp  # matrix multiplication

    X_mds = squareform(pdist(X_mds.todense()))
    X_mds = mds.fit(X_mds).embedding_
    # X_mds = squareform(pdist(adata_counts.obsm['X_pca'][:, 0:ncomps+20])) #no diffusion makes is less streamlined and compact. more fuzzy
    return X_mds

def draw_piechart_graph(via_object, type_data='pt', gene_exp:list=[], title='', cmap:str=None, ax_text=True, dpi=150,headwidth_arrow = 0.1, alpha_edge=0.4, linewidth_edge=2, edge_color='darkblue',reference=None, show_legend:bool=True, pie_size_scale:float=0.8, fontsize:float=8):
    '''
    plot two subplots with a clustergraph level representation of the viagraph showing true-label composition (lhs) and pseudotime/gene expression (rhs)
    Returns matplotlib figure with two axes that plot the clustergraph using edge bundling
    left axis shows the clustergraph with each node colored by annotated ground truth membership.
    right axis shows the same clustergraph with each node colored by the pseudotime or gene expression

    :param via_object: is class VIA (the same function also exists as a method of the class and an external plotting function
    :param type_data: string  default 'pt' for pseudotime colored nodes. or 'gene'
    :param gene_exp: list of values (column of dataframe) corresponding to feature or gene expression to be used to color nodes at CLUSTER level
    :param title: string
    :param cmap: default None. automatically chooses coolwarm for gene expression or viridis_r for pseudotime
    :param ax_text: Bool default= True. Annotates each node with cluster number and population of membership
    :param dpi: int default = 150
    :param headwidth_bundle: default = 0.1. width of arrowhead used to directed edges
    :param reference: None or list. list of categorical (str) labels for cluster composition of the piecharts (LHS subplot) length = n_samples.
    :param pie_size_scale: float default=0.8 scaling factor of the piechart nodes
    :return: f, ax, ax1
    '''

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    f, ((ax, ax1)) = plt.subplots(1, 2, sharey=True, dpi=dpi)

    node_pos = via_object.graph_node_pos

    node_pos = np.asarray(node_pos)
    if cmap is None: cmap = 'coolwarm' if type_data == 'gene' else 'viridis_r'

    if type_data == 'pt':
        pt = via_object.scaled_hitting_times  # these are the final MCMC refined pt then slightly scaled at cluster level
        title_ax1 = "Pseudotime"

    if type_data == 'gene':
        pt = gene_exp
        title_ax1 = title
    if reference is None: reference_labels=via_object.true_label
    else: reference_labels = reference
    n_groups = len(set(via_object.labels))
    n_truegroups = len(set(reference_labels))
    group_pop = np.zeros([n_groups, 1])
    group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=list(set(reference_labels)))
    via_object.cluster_population_dict = {}
    for group_i in set(via_object.labels):
        loc_i = np.where(via_object.labels == group_i)[0]

        group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
        via_object.cluster_population_dict[group_i] = len(loc_i)
        true_label_in_group_i = list(np.asarray(reference_labels)[loc_i])
        for ii in set(true_label_in_group_i):
            group_frac[ii][group_i] = true_label_in_group_i.count(ii)

    line_true = np.linspace(0, 1, n_truegroups)
    color_true_list = [plt.cm.rainbow(color) for color in line_true]

    sct = ax.scatter(node_pos[:, 0], node_pos[:, 1],
                     c='white', edgecolors='face', s=group_pop, cmap='jet')

    bboxes = getbb(sct, ax)

    ax = plot_edgebundle_viagraph(ax, via_object.hammerbundle_cluster, layout=via_object.graph_node_pos, CSM=via_object.CSM,
                            velocity_weight=via_object.velo_weight, pt=pt, headwidth_bundle=headwidth_arrow,
                            alpha_bundle=alpha_edge,linewidth_bundle=linewidth_edge, edge_color=edge_color)

    trans = ax.transData.transform
    bbox = ax.get_position().get_points()
    ax_x_min = bbox[0, 0]
    ax_x_max = bbox[1, 0]
    ax_y_min = bbox[0, 1]
    ax_y_max = bbox[1, 1]
    ax_len_x = ax_x_max - ax_x_min
    ax_len_y = ax_y_max - ax_y_min
    trans2 = ax.transAxes.inverted().transform
    pie_axs = []
    pie_size_ar = ((group_pop - np.min(group_pop)) / (np.max(group_pop) - np.min(group_pop)) + 0.5) / 10  # 10

    for node_i in range(n_groups):

        cluster_i_loc = np.where(np.asarray(via_object.labels) == node_i)[0]
        majority_true = via_object.func_mode(list(np.asarray(reference_labels)[cluster_i_loc]))
        pie_size = pie_size_ar[node_i][0] *pie_size_scale

        x1, y1 = trans(node_pos[node_i])  # data coordinates
        xa, ya = trans2((x1, y1))  # axis coordinates

        xa = ax_x_min + (xa - pie_size / 2) * ax_len_x
        ya = ax_y_min + (ya - pie_size / 2) * ax_len_y
        # clip, the fruchterman layout sometimes places below figure
        if ya < 0: ya = 0
        if xa < 0: xa = 0
        rect = [xa, ya, pie_size * ax_len_x, pie_size * ax_len_y]
        frac = np.asarray([ff for ff in group_frac.iloc[node_i].values])

        pie_axs.append(plt.axes(rect, frameon=False))
        pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
        pie_axs[node_i].set_xticks([])
        pie_axs[node_i].set_yticks([])
        pie_axs[node_i].set_aspect('equal')
        # pie_axs[node_i].text(0.5, 0.5, graph_node_label[node_i])
        if ax_text==True: pie_axs[node_i].text(0.5, 0.5, majority_true, fontsize = fontsize )

    patches, texts = pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
    labels = list(set(reference_labels))
    if show_legend ==True: plt.legend(patches, labels, loc=(-5, -5), fontsize=6, frameon=False)

    if via_object.time_series==True:
        ti = 'Cluster Composition. K=' + str(via_object.knn) + '. ncomp = ' + str(via_object.ncomp)  +'knnseq_'+str(via_object.knn_sequential)# "+ is_sub
    else:
        ti = 'Cluster Composition. K=' + str(via_object.knn) + '. ncomp = ' + str(via_object.ncomp)
    ax.set_title(ti)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    title_list = [title_ax1]
    for i, ax_i in enumerate([ax1]):
        pt = via_object.markov_hitting_times if type_data == 'pt' else gene_exp

        c_edge, l_width = [], []
        for ei, pti in enumerate(pt):
            if ei in via_object.terminal_clusters:
                c_edge.append('red')
                l_width.append(1.5)
            else:
                c_edge.append('gray')
                l_width.append(0.0)

        gp_scaling = 1000 / max(group_pop)

        group_pop_scale = group_pop * gp_scaling * 0.5
        ax_i=plot_edgebundle_viagraph(ax_i, via_object.hammerbundle_cluster, layout=via_object.graph_node_pos,CSM=via_object.CSM, velocity_weight=via_object.velo_weight, pt=pt,headwidth_bundle=headwidth_arrow, alpha_bundle=alpha_edge, linewidth_bundle=linewidth_edge, edge_color=edge_color)

        im1 = ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale, c=pt, cmap=cmap,
                           edgecolors=c_edge,
                           alpha=1, zorder=3, linewidth=l_width)
        if ax_text:
            x_max_range = np.amax(node_pos[:, 0]) / 100
            y_max_range = np.amax(node_pos[:, 1]) / 100

            for ii in range(node_pos.shape[0]):
                ax_i.text(node_pos[ii, 0] + max(x_max_range, y_max_range),
                          node_pos[ii, 1] + min(x_max_range, y_max_range),
                          'C' + str(ii) + 'pop' + str(int(group_pop[ii][0])),
                          color='black', zorder=4, fontsize = fontsize)
        ax_i.set_title(title_list[i])
        ax_i.grid(False)
        ax_i.set_xticks([])
        ax_i.set_yticks([])

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if type_data == 'pt':
        f.colorbar(im1, cax=cax, orientation='vertical', label='pseudotime')
    else:
        f.colorbar(im1, cax=cax, orientation='vertical', label='Gene expression')
    f.patch.set_visible(False)

    ax1.axis('off')
    ax.axis('off')
    ax.set_facecolor('white')
    ax1.set_facecolor('white')
    return f, ax, ax1
