

from ..external.VIA.core import VIA
from ..external.VIA.datasets_via import scRNA_hematopoiesis

import anndata
import numpy as np
import scanpy as sc
import pandas as pd

import igraph as ig
import pygam as pg
from datetime import datetime
from typing import Union,Tuple

from ..external.VIA.utils_via import *
from ..external.VIA.plotting_via import *
from .._settings import add_reference
from ..utils.registry import register_function
#from ..via.utils_via import sc_loc_ofsuperCluster_PCAspace,compute_velocity_on_grid

@register_function(
    aliases=["造血数据集", "hematopoiesis", "scRNA_hematopoiesis", "造血发育数据", "血细胞发育"],
    category="single",
    description="Load scRNA-seq hematopoiesis dataset for trajectory inference and developmental analysis",
    examples=[
        "# Load hematopoiesis dataset",
        "adata = ov.single.scRNA_hematopoiesis()",
        "# Alternative way to load",
        "adata = ov.single.hematopoiesis()",
        "# Check dataset structure",
        "print(adata.obs.columns)",
        "print(adata.obsm.keys())"
    ],
    related=["single.pyVIA", "single.TrajInfer", "pp.preprocess"]
)
def hematopoiesis()->anndata.AnnData:
    r"""Load scRNA-seq hematopoiesis dataset for trajectory inference.
    
    Returns:
        AnnData: Preprocessed hematopoiesis dataset with embeddings and annotations.
    
    Examples:
        >>> import omicverse as ov
        >>> # Load the dataset
        >>> adata = ov.single.scRNA_hematopoiesis()
        >>> print(adata.shape)
    """
    return scRNA_hematopoiesis()

@register_function(
    aliases=["VIA轨迹推断", "pyVIA", "via_analysis", "VIA算法", "轨迹拓扑分析"],
    category="single",
    description="VIA (Velocity and Topology Inference Algorithm) for single-cell trajectory inference with automated terminal state prediction",
    prerequisites={
        'functions': ['pca', 'neighbors'],
        'optional_functions': ['umap', 'leiden']
    },
    requires={
        'obsm': ['X_pca'],  # Required for adata_key parameter
        'uns': ['neighbors']  # VIA uses neighborhood graph
    },
    produces={
        'obs': ['pt_via']  # Pseudotime values
    },
    auto_fix='auto',
    examples=[
        "# Initialize pyVIA",
        "v0 = ov.single.pyVIA(adata=adata, adata_key='X_pca', adata_ncomps=80,",
        "                     basis='tsne', clusters='label', knn=30, random_seed=4)",
        "# Set root cell if known",
        "v0 = ov.single.pyVIA(adata=adata, adata_key='X_pca', basis='umap',",
        "                     clusters='celltype', root_user=[1234])",
        "# Run VIA analysis",
        "v0.run()",
        "# Get pseudotime",
        "v0.get_pseudotime(adata)",
        "# Plot piechart graph",
        "fig, ax, ax1 = v0.plot_piechart_graph(clusters='label', cmap='Reds')",
        "# Plot gene trends",
        "fig, axs = v0.plot_gene_trend(gene_list=['gene1', 'gene2'], figsize=(8,6))",
        "# Plot trajectory projection",
        "fig, ax1, ax2 = v0.plot_trajectory_gams(basis='umap', clusters='celltype')",
        "# Plot stream plot",
        "fig, ax = v0.plot_stream(basis='umap', clusters='celltype', density_grid=0.8)",
        "# Plot lineage probabilities",
        "fig, axs = v0.plot_lineage_probability(figsize=(8,4))",
        "# Plot clustergraph with genes",
        "fig, axs = v0.plot_clustergraph(gene_list=['CD34', 'GATA1'], figsize=(12,3))"
    ],
    related=["single.TrajInfer", "single.scRNA_hematopoiesis", "pp.neighbors", "utils.embedding"]
)
class pyVIA(object):

    def __init__(self,adata:anndata.AnnData,adata_key:str='X_pca',adata_ncomps:int=80,basis:str='X_umap',
                 clusters:str='',dist_std_local:float=2, jac_std_global=0.15, labels:np.ndarray=None,
                 keep_all_local_dist='auto', too_big_factor:float=0.4, resolution_parameter:float=1.0, partition_type:str="ModularityVP", small_pop:int=10,
                 jac_weighted_edges:bool=True, knn:int=30, n_iter_leiden:int=5, random_seed:int=42,
                 num_threads=-1, distance='l2', time_smallpop=15,
                 super_cluster_labels:bool=False,                 super_node_degree_list:bool=False, super_terminal_cells:bool=False, x_lazy:float=0.95, alpha_teleport:float=0.99,
                 root_user=None, preserve_disconnected:bool=True, dataset:str='', super_terminal_clusters:list=[],
                 is_coarse=True, csr_full_graph:np.ndarray='', csr_array_locally_pruned='', ig_full_graph='',
                 full_neighbor_array='', full_distance_array='',  df_annot=None,
                 preserve_disconnected_after_pruning:bool=False,
                 secondary_annotations:list=None, pseudotime_threshold_TS:int=30, cluster_graph_pruning_std:float=0.15,
                 visual_cluster_graph_pruning:float=0.15, neighboring_terminal_states_threshold=3, num_mcmc_simulations=1300,
                 piegraph_arrow_head_width=0.1,
                 piegraph_edgeweight_scalingfactor=1.5, max_visual_outgoing_edges:int=2, via_coarse=None, velocity_matrix=None,
                 gene_matrix=None, velo_weight=0.5, edgebundle_pruning=None, A_velo = None, CSM = None, edgebundle_pruning_twice=False, pca_loadings = None, time_series=False,
                 time_series_labels:list=None, knn_sequential:int = 10, knn_sequential_reverse:int = 0,t_diff_step:int = 1,single_cell_transition_matrix = None,
                 embedding_type:str='via-mds',do_compute_embedding:bool=False, color_dict:dict=None,user_defined_terminal_cell:list=[], user_defined_terminal_group:list=[],
                 do_gaussian_kernel_edgeweights:bool=False,RW2_mode:bool=False,working_dir_fp:str ='/home/shobi/Trajectory/Datasets/') -> None:
        r"""Initialize a pyVIA object for trajectory inference.

        Arguments:
            adata: An AnnData object containing the single-cell RNA-seq data
            adata_key (str): The key of the AnnData in obsm to perform VIA on (default: 'X_pca')
            adata_ncomps (int): The number of components to use from the AnnData in obsm (default: 80)
            basis (str): The key of the AnnData in obsm to use as the basis for embedding (default: 'X_umap')
            clusters (str): The clusters to use for the VIA analysis (default: '')
            dist_std_local: local level of pruning for PARC graph clustering stage. Range (0.1,3) higher numbers mean more edge retention
            jac_std_global: (optional, default = 0.15, can also set as 'median') global level graph pruning for PARC clustering stage. Number of standard deviations below the network’s mean-jaccard-weighted edges. 0.1-1 provide reasonable pruning.higher value means less pruning (more edges retained). e.g. a value of 0.15 means all edges that are above mean(edgeweight)-0.15*std(edge-weights) are retained. We find both 0.15 and ‘median’ to yield good results/starting point and resulting in pruning away ~ 50-60% edges
            labels: default is None. and PARC clusters are used for the viagraph. alternatively provide a list of clustermemberships that are integer values (not strings) to construct the viagraph using another clustering method or available annotations
            keep_all_local_dist: default value of 'auto' means that for smaller datasets local-pruning is done prior to clustering, but for large datasets local pruning is set to False for speed.
            
            too_big_factor: (optional, default=0.4). Forces clusters > 0.4*n_cells to be re-clustered
            resolution_parameter: (float) the resolution parameter for the Louvain algorithm.
            partition_type: (str, default: "ModularityVP") the partitioning algorithm to use.
            small_pop: (int, default: 10) the number of cells to be considered in a small population.
            jac_weighted_edges: (bool, default: True) whether to use weighted edges in the PARC clustering step.
            knn: (int, optional, default: 30) the number of K-Nearest Neighbors for HNSWlib KNN graph. Larger knn means more graph connectivity. Lower knn means more loosely connected clusters/cells.
            n_iter_leiden: (int) the number of iterations for the Leiden algorithm.
            random_seed: (int) the random seed to pass to the clustering algorithm.
            num_threads: (int) the number of threads to use for the clustering algorithm.
            distance: (str, default: 'l2') the distance metric to use for graph construction and similarity. Options are 'l2', 'ip', and 'cosine'.
            visual_cluster_graph_pruning: (float, default: 0.15) the pruning level for the cluster graph. This only comes into play if the user deliberately chooses not to use the default edge-bundling method of visualizating edges (draw_piechart_graph()) and instead calls draw_piechart_graph_nobundle(). It controls the number of edges plotted for visual effect. This does not impact computation of terminal states, pseudotime, or lineage likelihoods.
            cluster_graph_pruning_std: (float, default: 0.15) the pruning level of the cluster graph. Often set to the same value as the PARC clustering level of jac_std_global. Reasonable range is [0.1, 1]. To retain more connectivity in the clustergraph underlying the trajectory computations, increase the value.
            time_smallpop: (max time to be allowed handling singletons) the maximum time allowed to handle singletons.
            super_cluster_labels:
            super_node_degree_list:
            super_terminal_cells:
            x_lazy: (float, default: 0.95) 1-x = probability of staying in the same node (lazy). Values between 0.9-0.99 are reasonable.
            alpha_teleport: (float, default: 0.99) 1-alpha is probability of jumping. Values between 0.95-0.99 are reasonable unless prior knowledge of teleportation.
            root_user: (list, None) the root user list. Can be a list of strings, a list of int, or None. The default is None. When the root_user is set as None and an RNA velocity matrix is available, a root will be automatically computed. If the root_user is None and no velocity matrix is provided, then an arbitrary root is selected. If the root_user is ['celltype_earlystage'] where the str corresponds to an item in true_label, then a suitable starting point will be selected corresponding to this group.
            preserve_disconnected: bool (default = True) If you believe there may be disconnected trajectories then set this to False
            dataset: str Can be set to 'group' or '' (default). this refers to the type of root label (group level root or single cell index) you are going to provide. if your true_label has a sensible group of cells for a root then you can set dataset to 'group' and make the root parameter ['labelname_root_cell_type'] if your root corresponds to one particular cell then set dataset = '' (default)
            embedding: ndarray (optional, default = None) embedding (e.g. precomputed tsne, umap, phate, via-umap) for plotting data. Size n_cells x 2 If an embedding is provided when running VIA, then a scatterplot colored by pseudotime, highlighting terminal fates
            velo_weight: float (optional, default = 0.5) #float between [0,1]. the weight assigned to directionality and connectivity derived from scRNA-velocity
            neighboring_terminal_states_threshold:int (default = 3). Candidates for terminal states that are neighbors of each other may be removed from the list if they have this number of more of terminal states as neighbors
            knn_sequential:int (default =10) number of knn in the adjacent time-point for time-series data (t_i and t_i+1)
            knn_sequential_reverse: int (default = 0) number of knn enforced from current to previous time point
            t_diff_step: int (default =1) Number of permitted temporal intervals between connected nodes. If time data is labeled as [0,25,50,75,100,..] then t_diff_step=1 corresponds to '25' and only edges within t_diff_steps are retained
            is_coarse:bool (default = True) If running VIA in two iterations where you wish to link the second fine-grained iteration with the initial iteration, then you set to False
            via_coarse: VIA (default = None) If instantiating a second iteration of VIA that needs to be linked to a previous iteration (e.g. via0), then set via_coarse to the previous via0 object
            df_annot: DataFrame (default None) used for the Mouse Organ data
            preserve_disconnected_after_pruning:bool (default = True) If you believe there are disconnected trajectories then set this to False and test your hypothesis
            A_velo: ndarray Cluster Graph Transition matrix based on rna velocity [n_clus x n_clus]
            velocity_matrix: matrix (default None) matrix of size [n_samples x n_genes]. this is the velocity matrix computed by scVelo (or similar package) and stored in adata.layers['velocity']. The genes used for computing velocity should correspond to those useing in gene_matrix Requires gene_matrix to be provided too.
            gene_matrix: matrix (default None) Only used if Velocity_matrix is available. matrix of size [n_samples x n_genes]. We recommend using a subset like HVGs rather than full set of genes. (need to densify input if taking from adata = adata.X.todense())
            time_series: if the data has time-series labels then set to True
            time_series_labels:list (default None) list of integer values of temporal annotations corresponding to e.g. hours (post fert), days, or sequential ordering
            pca_loadings: array (default None) the loadings of the pcs used to project the cells (to projected euclidean location based on velocity). n_cells x n_pcs
            secondary_annotations: None (default None)
            edgebundle_pruning:float (default=None) will by default be set to the same as the cluster_graph_pruning_std and influences the visualized level of pruning of edges. Typical values can be between [0,1] with higher numbers retaining more edges
            edgebundle_pruning_twice:bool default: False. When True, the edgebundling is applied to a further visually pruned (visual_cluster_graph_pruning) and can sometimes simplify the visualization. it does not impact the pseudotime and lineage computations
            piegraph_arrow_head_width: float (default = 0.1) size of arrow heads in via cluster graph
            piegraph_edgeweight_scalingfactor: (defaulf = 1.5) scaling factor for edge thickness in via cluster graph
            max_visual_outgoing_edges: int (default =2) Rarely comes into play. Only used if the user later chooses to plot the via-graph without edgebunding using draw_piechart_graph_nobundle() Only allows max_visual_outgoing_edges to come out of any given node.
            edgebundle_pruning:float (default=None) will by default be set to the same as the cluster_graph_pruning_std and influences the visualized level of pruning of edges. Typical values can be between [0,1] with higher numbers retaining more edges
            edgebundle_pruning_twice:bool default: False. When True, the edgebundling is applied to a further visually pruned (visual_cluster_graph_pruning) and can sometimes simplify the visualization. it does not impact the pseudotime and lineage computations
            pseudotime_threshold_TS: int (default = 30) corresponds to the criteria for a state to be considered a candidate terminal cell fate to be 30% or later of the computed pseudotime range
            num_mcmc_simulations:int (default = 1300) number of random walk simulations conducted
            embedding_type: str (default = 'via-mds', other options are 'via-umap' and 'via-force'
            do_compute_embedding: bool (default = False) If you want an embedding (n_samples x2) to be computed on the basis of the via sc graph then set this to True
            do_gaussian_kernel_edgeweights: bool (default = False) Type of edgeweighting on the graph edges

        
        """


        self.adata = adata
        #self.adata_key = adata_key
        data = adata.obsm[adata_key][:, 0:adata_ncomps]
        embedding=self.adata.obsm[basis]
        true_label=adata.obs[clusters]
        self.clusters=clusters
        self.basis=basis
        
        if root_user is not None:
             dataset='group'

        

        self.model=VIA(data=data,true_label=true_label,
                 dist_std_local=dist_std_local,jac_std_global=jac_std_global,labels=labels,
                 keep_all_local_dist=keep_all_local_dist,too_big_factor=too_big_factor,resolution_parameter=resolution_parameter,partition_type=partition_type,small_pop=small_pop,
                 jac_weighted_edges=jac_weighted_edges,knn=knn,n_iter_leiden=n_iter_leiden,random_seed=random_seed,
                 num_threads=num_threads,distance=distance,time_smallpop=time_smallpop,
                 super_cluster_labels=super_cluster_labels,super_node_degree_list=super_node_degree_list,super_terminal_cells=super_terminal_cells,x_lazy=x_lazy,alpha_teleport=alpha_teleport,
                 root_user=root_user,preserve_disconnected=preserve_disconnected,dataset=dataset,super_terminal_clusters=super_terminal_clusters,
                 is_coarse=is_coarse,csr_full_graph=csr_full_graph,csr_array_locally_pruned=csr_array_locally_pruned,ig_full_graph=ig_full_graph,
                 full_neighbor_array=full_neighbor_array,full_distance_array=full_distance_array,embedding=embedding,df_annot=df_annot,
                 preserve_disconnected_after_pruning=preserve_disconnected_after_pruning,
                 secondary_annotations=secondary_annotations,pseudotime_threshold_TS=pseudotime_threshold_TS,cluster_graph_pruning_std=cluster_graph_pruning_std,
                 visual_cluster_graph_pruning=visual_cluster_graph_pruning,neighboring_terminal_states_threshold=neighboring_terminal_states_threshold,num_mcmc_simulations=num_mcmc_simulations,
                 piegraph_arrow_head_width=piegraph_arrow_head_width,
                 piegraph_edgeweight_scalingfactor=piegraph_edgeweight_scalingfactor,max_visual_outgoing_edges=max_visual_outgoing_edges,via_coarse=via_coarse,velocity_matrix=velocity_matrix,
                 gene_matrix=gene_matrix,velo_weight=velo_weight,edgebundle_pruning=edgebundle_pruning,A_velo=A_velo,CSM=CSM,edgebundle_pruning_twice=edgebundle_pruning_twice,pca_loadings=pca_loadings,time_series=time_series,
                 time_series_labels=time_series_labels,knn_sequential=knn_sequential,knn_sequential_reverse=knn_sequential_reverse,t_diff_step=t_diff_step,single_cell_transition_matrix=single_cell_transition_matrix,
                 embedding_type=embedding_type,do_compute_embedding=do_compute_embedding,color_dict=color_dict,user_defined_terminal_cell=user_defined_terminal_cell,user_defined_terminal_group=user_defined_terminal_group,
                 do_gaussian_kernel_edgeweights=do_gaussian_kernel_edgeweights,RW2_mode=RW2_mode,working_dir_fp=working_dir_fp
                 )
    def run(self):
        r"""Calculate the VIA graph and pseudotime.
        
        This method runs the main VIA algorithm to construct the trajectory graph
        and compute pseudotime for each cell.
        """

        self.model.run_VIA()
        add_reference(self.adata,'VIA','trajectory inference with VIA')

    def get_piechart_dict(self,label:int=0,clusters:str='')->dict:
        r"""Get cluster composition dictionary for pie chart visualization.

        Arguments:
            label (int): Cluster label of pie chart (default: 0)
            clusters (str): The celltype column name of interest (default: '')
        
        Returns:
            dict: Cluster composition dictionary for visualization
        """
        if clusters=='':
            clusters=self.clusters
        self.adata.obs[clusters]=self.adata.obs[clusters].astype('category')
        cluster_i_loc=np.where(np.asarray(self.model.labels) == label)[0]
        res_dict=dict(self.adata.obs.iloc[cluster_i_loc].value_counts(clusters))
        return res_dict
    
    def get_pseudotime(self,adata=None):
        r"""Extract the pseudotime values computed by VIA.

        Arguments:
            adata: An AnnData object to add pseudotime to (default: None)
                  If None, pseudotime will be added to self.adata.obs['pt_via']

        """

        print('...the pseudotime of VIA added to AnnData obs named `pt_via`')
        if adata is None:
            self.adata.obs['pt_via']=self.model.single_cell_pt_markov
        else:
            adata.obs['pt_via']=self.model.single_cell_pt_markov

    def plot_piechart_graph(self,clusters:str='', type_data='pt',
                                gene_exp:list=[], title='', 
                                cmap:str=None, ax_text=True, figsize:tuple=(8,4),
                                dpi=150,headwidth_arrow = 0.1, 
                                alpha_edge=0.4, linewidth_edge=2, 
                                edge_color='darkblue',reference=None, 
                                show_legend:bool=True, pie_size_scale:float=0.8, fontsize:float=8)->Tuple[matplotlib.figure.Figure,
                                                                                                          matplotlib.axes._axes.Axes,
                                                                                                          matplotlib.axes._axes.Axes]:
        r"""Plot two subplots with clustergraph representation showing cluster composition and pseudotime/gene expression.
    
        Arguments:
            clusters (str): Column name of the adata.obs dataframe containing cluster labels (default: '')
            type_data (str): Type of data for coloring nodes - 'pt' for pseudotime or 'gene' (default: 'pt')
            gene_exp (list): List of values for gene expression to color nodes at cluster level (default: [])
            title (str): Title for the plot (default: '')
            cmap (str): Colormap - automatically chooses coolwarm for gene or viridis_r for pseudotime (default: None)
            ax_text (bool): Whether to annotate nodes with cluster number and population (default: True)
            figsize (tuple): Figure size (default: (8,4))
            dpi (int): DPI for the figure (default: 150)
            headwidth_arrow (float): Width of arrowhead for directed edges (default: 0.1)
            alpha_edge (float): Transparency of edges (default: 0.4)
            linewidth_edge (float): Width of edges (default: 2)
            edge_color (str): Color of edges (default: 'darkblue')
            reference: List of categorical labels for cluster composition (default: None)
            show_legend (bool): Whether to show legend (default: True)
            pie_size_scale (float): Scaling factor of the piechart nodes (default: 0.8)
            fontsize (float): Font size of text in piecharts (default: 8)

        Returns:
            tuple: (fig, ax, ax1) where fig is matplotlib figure, ax is left axis with cluster composition, ax1 is right axis with pseudotime/gene expression
        """


        if clusters=='':
            clusters=self.clusters
        self.adata.obs[clusters]=self.adata.obs[clusters].astype('category')
        fig, ax, ax1 = draw_piechart_graph_pyomic(clusters=clusters,adata=self.adata,
                                   via_object=self.model, type_data=type_data,
                                gene_exp=gene_exp, title=title, 
                                cmap=cmap, ax_text=ax_text,figsize=figsize,
                                dpi=dpi,headwidth_arrow = headwidth_arrow,
                                alpha_edge=alpha_edge, linewidth_edge=linewidth_edge,
                                edge_color=edge_color,reference=reference,
                                show_legend=show_legend, pie_size_scale=pie_size_scale, fontsize=fontsize)
        return fig, ax, ax1
    
    def plot_stream(self,clusters:str='',basis:str='',
                   density_grid:float=0.5, arrow_size:float=0.7, arrow_color:str = 'k',
                   arrow_style="-|>",  max_length:int=4, linewidth:float=1,min_mass = 1, cutoff_perc:int = 5,
                   scatter_size:int=500, scatter_alpha:float=0.5,marker_edgewidth:float=0.1,
                   density_stream:int = 2, smooth_transition:int=1, smooth_grid:float=0.5,
                   color_scheme:str = 'annotation', add_outline_clusters:bool=False,
                   cluster_outline_edgewidth = 0.001,gp_color = 'white', bg_color='black' ,
                   dpi=80 , title='Streamplot', b_bias=20, n_neighbors_velocity_grid=None,
                   other_labels:list = None,use_sequentially_augmented:bool=False, cmap_str:str='rainbow')->Tuple[matplotlib.figure.Figure,
                                                                                                          matplotlib.axes._axes.Axes]:
        """Construct vector streamplot on the embedding to show a fine-grained view of inferred directions in the trajectory

        Arguments:
            clusters : column name of the adata.obs dataframe that contains the cluster labels
            basis : str, default = 'X_umap', which to use for the embedding
            density_grid : float, default = 0.5, density of the grid on which to project the directionality of cells
            arrow_size : float, default = 0.7, size of the arrows in the streamplot
            arrow_color : str, default = 'k', color of the arrows in the streamplot
            arrow_style : str, default = "-|>", style of the arrows in the streamplot
            max_length : int, default = 4, maximum length of the arrows in the streamplot
            linewidth : float, default = 1, width of  lines in streamplot
            min_mass : float, default = 1, minimum mass of the arrows in the streamplot
            cutoff_perc : int, default = 5, cutoff percentage of the arrows in the streamplot
            scatter_size : int, default = 500, size of scatter points
            scatter_alpha : float, default = 0.5, transpsarency of scatter points
            marker_edgewidth : float, default = 0.1, width of outline arround each scatter point
            density_stream : int, default = 2, density of the streamplot
            smooth_transition : int, default = 1, smoothness of the transition between the streamplot and the scatter points
            smooth_grid : float, default = 0.5, smoothness of the grid on which to project the directionality of cells
            color_scheme : str, default = 'annotation' corresponds to self.true_labels. Other options are 'time' (uses single-cell pseudotime) or 'clusters' (uses self.clusters)
            add_outline_clusters : bool, default = False, whether to add an outline to the clusters
            cluster_outline_edgewidth : float, default = 0.001, width of the outline around the clusters
            gp_color : str, default = 'white', color of the grid points
            bg_color : str, default = 'black', color of the background
            dpi : int, default = 80, dpi of the figure
            title : str, default = 'Streamplot', title of the figure
            b_bias : int, default = 20, higher value makes the forward bias of pseudotime stronger
            n_neighbors_velocity_grid : int, default = None, number of neighbors to use for the velocity grid
            other_labels : list, default = None, list of other labels to plot in the streamplot
            use_sequentially_augmented : bool, default = False, whether to use the sequentially augmented data
            cmap_str : str, default = 'rainbow', color map to use for the streamplot

        Returns:
            fig : matplotlib figure
            ax : matplotlib axis
        """

        if clusters=='':
            clusters=self.clusters
        if basis=='':
            basis=self.basis
        self.adata.obs[clusters]=self.adata.obs[clusters].astype('category')
        embedding=self.adata.obsm[basis]
        fig,ax = via_streamplot_pyomic(adata=self.adata,clusters=clusters,via_object=self.model, 
                                 embedding=embedding,density_grid=density_grid, arrow_size=arrow_size,
                                 arrow_color=arrow_color,arrow_style=arrow_style,  max_length=max_length,
                                 linewidth=linewidth,min_mass = min_mass, cutoff_perc=cutoff_perc,
                                 scatter_size=scatter_size, scatter_alpha=scatter_alpha,marker_edgewidth=marker_edgewidth,
                                 density_stream=density_stream, smooth_transition=smooth_transition, smooth_grid=smooth_grid,
                                 color_scheme=color_scheme, add_outline_clusters=add_outline_clusters,
                                 cluster_outline_edgewidth = cluster_outline_edgewidth,gp_color = gp_color, bg_color=bg_color,
                                 dpi=dpi , title=title, b_bias=b_bias, n_neighbors_velocity_grid=n_neighbors_velocity_grid,
                                 other_labels=other_labels,use_sequentially_augmented=use_sequentially_augmented, cmap_str=cmap_str)
        return fig,ax

    def plot_trajectory_gams(self,clusters:str='',basis:str='',via_fine=None, idx=None,
                         title_str:str= "Pseudotime", draw_all_curves:bool=True, arrow_width_scale_factor:float=15.0,
                         scatter_size:float=50, scatter_alpha:float=0.5,figsize:tuple=(8,4),
                         linewidth:float=1.5, marker_edgewidth:float=1, cmap_pseudotime:str='viridis_r',dpi:int=80,
                         highlight_terminal_states:bool=True, use_maxout_edgelist:bool =False)->Tuple[matplotlib.figure.Figure,
                                                                                                 matplotlib.axes._axes.Axes,
                                                                                                 matplotlib.axes._axes.Axes]:
        """projects the graph based coarse trajectory onto a umap/tsne embedding

        Arguments:
            clusters : column name of the adata.obs dataframe that contains the cluster labels
            basis : str, default = 'X_umap', which to use for the embedding
            via_fine : via object suggest to use via_object only unless you found that running via_fine gave better pathways
            idx : default: None. Or List. if you had previously computed a umap/tsne (embedding) only on a subset of the total n_samples (subsampled as per idx), then the via objects and results will be indexed according to idx too
            title_str : title of figure
            draw_all_curves : if the clustergraph has too many edges to project in a visually interpretable way, set this to False to get a simplified view of the graph pathways
            arrow_width_scale_factor : the width of the arrows is proportional to the edge weight. This factor scales the width of the arrows
            scatter_size : size of the scatter points
            scatter_alpha : transparency of the scatter points
            linewidth : width of the lines
            marker_edgewidth : width of the outline around each scatter point
            cmap_pseudotime : color map to use for the pseudotime
            dpi : dpi of the figure
            highlight_terminal_states :  whether or not to highlight/distinguish the clusters which are detected as the terminal states by via

        Returns:
            fig : matplotlib figure
            ax1 : matplotlib axis
            ax2 : matplotlib axis

        """


        if clusters=='':
            clusters=self.clusters
        if basis=='':
            basis=self.basis
        self.adata.obs[clusters]=self.adata.obs[clusters].astype('category')
        embedding=self.adata.obsm[basis]
        fig,ax1,ax2 = draw_trajectory_gams_pyomic(adata=self.adata,clusters=clusters,via_object=self.model, 
                                            via_fine=via_fine, embedding=embedding, idx=idx,
                                            title_str=title_str, draw_all_curves=draw_all_curves, arrow_width_scale_factor=arrow_width_scale_factor,
                                            scatter_size=scatter_size, scatter_alpha=scatter_alpha,figsize=figsize,
                                            linewidth=linewidth, marker_edgewidth=marker_edgewidth, cmap_pseudotime=cmap_pseudotime,dpi=dpi,
                                            highlight_terminal_states=highlight_terminal_states, use_maxout_edgelist=use_maxout_edgelist)
        return fig,ax1,ax2
    
    def plot_lineage_probability(self,clusters:str='',basis:str='',via_fine=None, 
                                idx=None, figsize:tuple=(8,4),
                                cmap:str='plasma', dpi:int=80, scatter_size =None,
                                marker_lineages:list = [], fontsize:int=12)->Tuple[matplotlib.figure.Figure,
                                                                                   matplotlib.axes._axes.Axes]:
        """G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it's made on full sample, knn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding that corresponds to the single cells in the full graph
        
        Arguments:
            clusters : column name of the adata.obs dataframe that contains the cluster labels
            basis : str, default = 'X_umap', which to use for the embedding
            via_fine : usually just set to same as via_coarse unless you ran a refined run and want to link it to initial via_coarse's terminal clusters
            idx : if one uses a downsampled embedding of the original data, then idx is the selected indices of the downsampled samples used in the visualization
            figsize : size of the figure
            cmap : color map to use for the lineage probability
            dpi : dpi of the figure
            scatter_size : size of the scatter points
            marker_lineages : Default is to use all lineage pathways. other provide a list of lineage number (terminal cluster number).
            fontsize : fontsize of the title

        Returns:
            fig : matplotlib figure
            axs : matplotlib axis
        """


        if clusters=='':
            clusters=self.clusters
        if basis=='':
            basis=self.basis
        self.adata.obs[clusters]=self.adata.obs[clusters].astype('category')
        embedding=self.adata.obsm[basis]
        fig, axs = draw_sc_lineage_probability(via_object=self.model,via_fine=via_fine, embedding=embedding,figsize=figsize,
                                               idx=idx, cmap_name=cmap, dpi=dpi, scatter_size =scatter_size,
                                            marker_lineages = marker_lineages, fontsize=fontsize)
        fig.tight_layout()
        return fig, axs
    
    def plot_gene_trend(self,gene_list:list=None,figsize:tuple=(8,4),
                        magic_steps:int=3, spline_order:int=5, dpi:int=80,cmap:str='jet', 
                        marker_genes:list = [], linewidth:float = 2.0,
                        n_splines:int=10,  fontsize_:int=12, marker_lineages=[])->Tuple[matplotlib.figure.Figure,
                                                                                        matplotlib.axes._axes.Axes]:
        """plots the gene expression trend along the pseudotime

        Arguments:
            gene_list : list of genes to plot
            figsize : size of the figure
            magic_steps : number of magic steps to use for imputation
            spline_order : order of the spline to use for smoothing
            dpi : dpi of the figure
            cmap : color map to use for the gene expression
            marker_genes : Default is to use all genes in gene_exp. other provide a list of marker genes that will be used from gene_exp.
            linewidth : width of the lines
            n_splines : number of splines to use for smoothing
            fontsize_ : fontsize of the title
            marker_lineages : Default is to use all lineage pathways. other provide a list of lineage number (terminal cluster number).

        Returns:
            fig : matplotlib figure
            axs : matplotlib axis
        
        """

        df_magic = self.model.do_impute(self.adata[:,gene_list].to_df(), magic_steps=magic_steps, gene_list=gene_list)
        fig, axs=get_gene_expression_pyomic(self.model,df_magic,spline_order=spline_order,dpi=dpi,
                                   cmap=cmap, marker_genes=marker_genes, linewidth=linewidth,figsize=figsize,
                                   n_splines=n_splines,  fontsize_=fontsize_, marker_lineages=marker_lineages)
        fig.tight_layout()
        return fig, axs
    
    def plot_clustergraph(self,gene_list:list,arrow_head:float=0.1,figsize:tuple=(8,4),dpi=80,magic_steps=3,
                          edgeweight_scale:float=1.5, cmap=None, label_=True,)->Tuple[matplotlib.figure.Figure,
                                                                                        matplotlib.axes._axes.Axes]:
        """plot the gene in pie chart for each cluster

        Arguments:
            gene_list : list of genes to plot
            arrow_head : size of the arrow head
            figsize : size of the figure
            edgeweight_scale : scale of the edge weight
            cmap : color map to use for the gene expression
            label_ : whether to label the nodes

        Returns:
            fig : matplotlib figure
            axs : matplotlib axis
        """
        df_magic = self.model.do_impute(self.adata[:,gene_list].to_df(), magic_steps=magic_steps, gene_list=gene_list)
        df_magic['parc'] = self.model.labels
        df_magic_cluster = df_magic.groupby('parc', as_index=True).mean()
        fig, axs = draw_clustergraph_pyomic(via_object=self.model, type_data='gene', gene_exp=df_magic_cluster, 
                                    gene_list=gene_list, arrow_head=arrow_head,figsize=figsize,
                                    edgeweight_scale=edgeweight_scale, cmap=cmap, label_=label_,dpi=dpi)
        fig.tight_layout()
        return fig,axs
    
    def plot_gene_trend_heatmap(self,gene_list:list,marker_lineages:list = [], 
                             fontsize:int=8,cmap:str='viridis', normalize:bool=True, ytick_labelrotation:int = 0, 
                             figsize:tuple=(2,4))->Tuple[matplotlib.figure.Figure,
                                                                    list]:
        """Plot the gene trends on heatmap: a heatmap is generated for each lineage (identified by terminal cluster number). Default selects all lineages

        Arguments:
            gene_list : list of genes to plot
            marker_lineages : list default = None and plots all detected all lineages. Optionally provide a list of integers corresponding to the cluster number of terminal cell fates
            fontsize : int default = 8
            cmap : str default = 'viridis'
            normalize : bool = True
            ytick_labelrotation : int default = 0
            figsize : size of the figure

        Returns:
            fig : matplotlib figure
            axs : list of matplotlib axis       
        """
        
        df_magic = self.model.do_impute(self.adata[:,gene_list].to_df(), magic_steps=3, gene_list=gene_list)
        df_magic['parc'] = self.model.labels
        df_magic_cluster = df_magic.groupby('parc', as_index=True).mean()
        fig,axs=plot_gene_trend_heatmaps_pyomic(via_object=self.model, df_gene_exp=df_magic, 
                                                cmap=cmap,fontsize=fontsize,normalize=normalize,
                                                ytick_labelrotation=ytick_labelrotation,figsize=figsize,
                                                marker_lineages=marker_lineages)
        fig.tight_layout()
        return fig,axs















def via_streamplot_pyomic(adata,clusters,via_object, embedding:np.ndarray=None , density_grid:float=0.5, arrow_size:float=0.7, arrow_color:str = 'k',
arrow_style="-|>",  max_length:int=4, linewidth:float=1,min_mass = 1, cutoff_perc:int = 5,
                   scatter_size:int=500, scatter_alpha:float=0.5,marker_edgewidth:float=0.1, 
                   density_stream:int = 2, smooth_transition:int=1, smooth_grid:float=0.5, 
                   color_scheme:str = 'annotation', add_outline_clusters:bool=False, 
                   cluster_outline_edgewidth = 0.001,gp_color = 'white', bg_color='black' , 
                   dpi=80 , title='Streamplot', b_bias=20, n_neighbors_velocity_grid=None, 
                   other_labels:list = None,use_sequentially_augmented=False, cmap_str:str='rainbow'):
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
        ax.scatter(embedding[:,0],embedding[:,1], c=via_object.single_cell_pt_markov,alpha=scatter_alpha,  
                   zorder = 0, s=scatter_size, linewidths=marker_edgewidth, cmap = cmap_str)
    else:
        if color_scheme == 'annotation':color_labels = via_object.true_label
        if color_scheme == 'cluster': color_labels= via_object.labels
        if other_labels is not None: color_labels = other_labels

        cmap_ = plt.get_cmap(cmap_str)
        #plt.cm.rainbow(color)

        line = np.linspace(0, 1, len(set(color_labels)))
        
        color_true_list=adata.uns['{}_colors'.format(clusters)]
        line = range(len(color_labels))
        for color, group in zip(line, list(adata.obs[clusters].cat.categories)):
            where = np.where(np.array(color_labels) == group)[0]
            ax.scatter(embedding[where, 0], embedding[where, 1], label=group,
                        c=color_true_list[color],
                        alpha=scatter_alpha,  zorder = 0, s=scatter_size, linewidths=marker_edgewidth)

            x_mean = embedding[where, 0].mean()
            y_mean = embedding[where, 1].mean()
            ax.text(x_mean, y_mean, str(group), fontsize=5, zorder=4, path_effects = [PathEffects.withStroke(linewidth=1, foreground='w')], weight = 'bold')

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.set_title(title)
    return fig, ax


def draw_trajectory_gams_pyomic(adata,clusters,via_object, via_fine=None, embedding: np.ndarray=None, idx=None,
                         title_str:str= "Pseudotime", draw_all_curves:bool=True, arrow_width_scale_factor:float=15.0,
                         scatter_size:float=50, scatter_alpha:float=0.5,figsize:tuple=(8,4),
                         linewidth:float=1.5, marker_edgewidth:float=1, cmap_pseudotime:str='viridis_r',dpi:int=80,highlight_terminal_states:bool=True, use_maxout_edgelist:bool =False):
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
    #true_label=list(adata.obs[clusters].cat.categories)
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

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize,dpi=dpi)
    num_true_group = len(set(true_label))
    color_true_list=adata.uns['{}_colors'.format(clusters)]
    
    num_cluster = len(set(super_cluster_labels))
    line = np.linspace(0, 1, num_true_group)
    
    line = range(len(true_label))
    for color, group in zip(line, list(adata.obs[clusters].cat.categories)):
        where = np.where(np.array(true_label) == group)[0]
        ax1.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group, c=color_true_list[color],
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
            dot_size.append(20)  # 20

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


def draw_piechart_graph_pyomic(adata,clusters,via_object, type_data='pt',
                                gene_exp:list=[], title='', 
                                cmap:str=None, ax_text=True, figsize=(8,4),
                                dpi=150,headwidth_arrow = 0.1, 
                                alpha_edge=0.4, linewidth_edge=2, 
                                edge_color='darkblue',reference=None, 
                                show_legend:bool=True, pie_size_scale:float=0.8, fontsize:float=8):
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
    f, ((ax, ax1)) = plt.subplots(1, 2, sharey=True,figsize=figsize, dpi=dpi)

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
    group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), 
                              columns=list(adata.obs[clusters].cat.categories))
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
    color_true_list=adata.uns['{}_colors'.format(clusters)]

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
        f.colorbar(im1, cax=cax, orientation='vertical', label='pseudotime',shrink=0.5)
    else:
        f.colorbar(im1, cax=cax, orientation='vertical', label='Gene expression',shrink=0.5)
    f.patch.set_visible(False)

    ax1.axis('off')
    ax.axis('off')
    ax.set_facecolor('white')
    ax1.set_facecolor('white')
    return f, ax, ax1


def get_gene_expression_pyomic(via0, gene_exp:pd.DataFrame, cmap:str='jet', figsize=(8,4),
                        dpi:int=150, marker_genes:list = [], linewidth:float = 2.0,
                        n_splines:int=10, spline_order:int=4, fontsize_:int=10, marker_lineages=[]):
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
    fig, axs = plt.subplots(fig_nrows, fig_ncols, dpi=dpi, figsize=figsize)
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
                            axs[r,c].set_title(gene_i,fontsize=fontsize_)
                            # Set tick font size
                            for label in (axs[r,c].get_xticklabels() + axs[r,c].get_yticklabels()):
                                label.set_fontsize(fontsize_-1)
                            #if i_gene == n_genes -1:
                            #    axs[r,c].legend(frameon=False, fontsize=fontsize_)
                            #    axs[r, c].set_xlabel('Time', fontsize=fontsize_)
                            #    axs[r, c].set_ylabel('Intensity', fontsize=fontsize_)
                            axs[r,c].spines['top'].set_visible(False)
                            axs[r,c].spines['right'].set_visible(False)
                            axs[r,c].legend().set_visible(False)
                            axs[r,c].grid(False)
                        else:
                            axs[c].plot(xval, yg, color=cmap_[i_terminal], linewidth=linewidth, zorder=3,   
                                        label=f"Lineage:{via0.terminal_clusters[i_terminal]}")
                            axs[c].set_title(gene_i, fontsize=fontsize_)
                            # Set tick font size
                            for label in (axs[c].get_xticklabels() + axs[c].get_yticklabels()):
                                label.set_fontsize(fontsize_-1)
                            #if i_gene == n_genes -1:
                            #    axs[c].legend(frameon=False,fontsize=fontsize_)
                            #    axs[c].set_xlabel('Time', fontsize=fontsize_)
                            #    axs[c].set_ylabel('Intensity', fontsize=fontsize_)
                            axs[c].spines['top'].set_visible(False)
                            axs[c].spines['right'].set_visible(False)
                            axs[c].legend().set_visible(False)
                            axs[c].grid(False)
                i_gene+=1
            else:
                if fig_nrows > 1:
                    axs[r,c].axis('off')
                    axs[r, c].grid(False)
                else:
                    axs[c].axis('off')
                    axs[c].grid(False)
    

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=fig_ncols, bbox_to_anchor=(0.5, -0.25))
    # 添加全局横纵标签
    #fig.suptitle('My Figure Title', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.03, 'Time', ha='center', fontsize=fontsize_+2)
    fig.text(0.04, 0.5, 'Intensity', va='center', rotation='vertical', fontsize=fontsize_+2)


    return fig, axs

def draw_clustergraph_pyomic(via_object, type_data='gene', gene_exp='', gene_list='', arrow_head=0.1,
                      edgeweight_scale=1.5, cmap=None, label_=True,figsize=(8,4),dpi=80):
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
    fig_nrows, mod = divmod(n, 4)
    if mod == 0: fig_nrows = fig_nrows
    if mod != 0: fig_nrows += 1

    fig_ncols = 4
    fig, axs = plt.subplots(fig_nrows, fig_ncols, dpi=dpi, figsize=figsize)
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

def plot_gene_trend_heatmaps_pyomic(via_object, df_gene_exp:pd.DataFrame, marker_lineages:list = [], 
                             fontsize:int=8,cmap:str='viridis', normalize:bool=True, ytick_labelrotation:int = 0, 
                             figsize:tuple=(2,4)):
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
    fig = plt.figure(figsize=figsize)
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