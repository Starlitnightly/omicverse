r"""Module providing encapsulation of STT for spatial transition tensor analysis."""
from ..external.STT import tl,pl
from typing import Any
import scanpy as sc
import numpy as np
import pandas as pd
from .._settings import add_reference
from .._registry import register_function

@register_function(
    aliases=["STT空间转换张量", "STT", "spatial_transition_tensor", "空间转换分析", "空间动力学"],
    category="space",
    description="Spatial Transition Tensor analysis for modeling spatial dynamics and cell state transitions",
    prerequisites={
        'functions': []  # Requires RNA velocity preprocessing (spliced/unspliced)
    },
    requires={
        'layers': ['spliced', 'unspliced'],  # RNA velocity data required
        'obsm': []  # Spatial coordinates (user-specified)
    },
    produces={
        'layers': ['velocity'],
        'obs': ['pseudotime']
    },
    auto_fix='escalate',
    examples=[
        "# Basic STT analysis",
        "stt = ov.space.STT(adata, spatial_loc='spatial',",
        "                   region='tissue_region')",
        "# Stage estimation",
        "stt.stage_estimate()",
        "# Train STT model",
        "stt.train(n_states=10)",
        "# Vector field analysis",
        "stt.vector_field()",
        "# Spatial dynamics",
        "stt.cal_pseudotime()",
        "# Custom parameters",
        "stt = ov.space.STT(adata, spatial_loc='xy_loc',",
        "                   region='Region')"
    ],
    related=["space.svg", "space.clusters", "external.scvelo"]
)
class STT(object):
    r"""Spatial Transition Tensor (STT) analysis class.
    
    STT models spatial dynamics and transitions by learning spatial-temporal patterns
    in spatial transcriptomics data using transition tensors. This class provides methods
    for analyzing cell state transitions, developmental trajectories, and spatial dynamics
    in tissue organization.

    Arguments:
        adata: AnnData
            Annotated data matrix containing spatial transcriptomics data.
            Must contain:
            - Spliced counts in adata.layers['spliced']
            - Unspliced counts in adata.layers['unspliced']
            - Spatial coordinates in adata.obsm[spatial_loc]
        spatial_loc: str, optional (default='xy_loc')
            Key in adata.obsm containing spatial coordinates.
        region: str, optional (default='Region')
            Column name in adata.obs containing region annotations.

    Attributes:
        adata: AnnData
            Input annotated data matrix.
        adata_aggr: AnnData or None
            Aggregated data after training.
        spatial_loc: str
            Key for spatial coordinates.
        region: str
            Key for region annotations.

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load data with spliced/unspliced counts
        >>> adata = sc.read_h5ad('spatial_velocity.h5ad')
        >>> # Initialize STT object
        >>> stt = ov.space.STT(
        ...     adata,
        ...     spatial_loc='spatial',
        ...     region='tissue_region'
        ... )
        >>> # Estimate cell stages
        >>> stt.stage_estimate()
        >>> # Train the model
        >>> stt.train(n_states=10)
    """
    def __init__(self,adata,spatial_loc='xy_loc',region='Region'):
        r"""Initialize STT spatial transition analysis object.
        
        Arguments:
            adata: AnnData
                Annotated data matrix with spatial and velocity data.
            spatial_loc: str, optional (default='xy_loc')
                Key for spatial coordinates in adata.obsm.
            region: str, optional (default='Region')
                Column name for region annotations in adata.obs.
        """
        self.adata=adata
        self.adata_aggr=None
        self.spatial_loc=spatial_loc
        self.region=region
        if 'attractor' not in self.adata.obs.keys():
            self.adata.obs['attractor'] = self.adata.obs[region]

    def stage_estimate(self):
        r"""Estimate cell stages using joint clustering of spliced and unspliced data.
        
        This method performs joint analysis of spliced and unspliced RNA counts to identify
        developmental stages through clustering. It uses a combination of PCA, nearest
        neighbor graph construction, and Leiden clustering to identify distinct cell stages.

        Arguments:
            None

        Returns:
            None
            Updates adata.obs['joint_leiden'] with stage assignments.

        Notes:
            - Requires spliced/unspliced counts in adata.layers
            - Uses resolution=0.15 for Leiden clustering
            - Number of stages is determined automatically
            - Results are stored in adata.obs['joint_leiden']
        """
        u = self.adata.layers['unspliced']
        s = self.adata.layers['spliced']
        if 'toarray' in dir(u):
            u = u.toarray()
            s = s.toarray()
        x_all = np.concatenate((u,s),axis = 1)
        adata_aggr = sc.AnnData(X=x_all)
        sc.tl.pca(adata_aggr, svd_solver='arpack')
        sc.pp.neighbors(adata_aggr)
        sc.tl.leiden(adata_aggr,resolution = 0.15)
        self.adata.obs['joint_leiden'] = adata_aggr.obs['leiden'].values
        print(f"...estimate stage: {len(self.adata.obs['joint_leiden'].unique())}")
        add_reference(self.adata,'STT','spatial transition tensor with STT')

    def train(self,
            n_states: int = 9,
            n_iter: int = 15,
            weight_connectivities: float = 0.5,
            n_neighbors: int = 50,
            thresh_ms_gene: float = 0.2,
            spa_weight: float = 0.3,
            **kwargs: Any
        ) -> None:
        r"""Train the STT spatial transition model.
        
        This method trains the Spatial Transition Tensor model to learn spatial-temporal
        patterns and cell state transitions. It combines connectivity constraints with
        spatial information to model developmental trajectories.

        Arguments:
            n_states: int, optional (default=9)
                Number of states for transition modeling.
            n_iter: int, optional (default=15)
                Number of iterations for model training.
            weight_connectivities: float, optional (default=0.5)
                Weight for connectivity constraints in range [0,1].
            n_neighbors: int, optional (default=50)
                Number of neighbors for graph construction.
            thresh_ms_gene: float, optional (default=0.2)
                Threshold for marker gene selection.
            spa_weight: float, optional (default=0.3)
                Weight for spatial constraints in range [0,1].
            **kwargs: 
                Additional arguments for dynamical iteration.

        Returns:
            None
            Updates self.adata_aggr with trained model results.

        Notes:
            - Higher spa_weight emphasizes spatial relationships
            - Higher weight_connectivities emphasizes transcriptional similarity
            - Results are stored in self.adata_aggr
            - Spatial coordinates are preserved in obsm
        """
        self.adata_aggr = tl.dynamical_iteration(self.adata,n_states = n_states,
                                    n_iter = n_iter, weight_connectivities = weight_connectivities,
                                    n_neighbors = n_neighbors,thresh_ms_gene = thresh_ms_gene,
                                    spa_weight =spa_weight,**kwargs)
        self.adata.obsm[f'X_{self.spatial_loc}'] = self.adata.obsm[self.spatial_loc]
        self.adata_aggr.obsm[f'X_{self.spatial_loc}']=self.adata.obsm[self.spatial_loc]
        self.adata_aggr.obsm[f'X_{self.spatial_loc}_aggr']=self.adata.obsm[self.spatial_loc]
        self.adata.obsm[f'X_{self.spatial_loc}_aggr']=self.adata.obsm[self.spatial_loc]
        add_reference(self.adata,'STT','spatial transition tensor with STT')

    def load(self,adata,adata_aggr):
        r"""Load pre-trained STT model data.
        
        This method allows loading previously trained STT model results, enabling
        analysis continuation or result sharing.

        Arguments:
            adata: AnnData
                Original AnnData object with spatial data.
            adata_aggr: AnnData
                Aggregated AnnData object from previous training.

        Returns:
            None
            Updates self.adata and self.adata_aggr with loaded data.

        Notes:
            - Useful for sharing analysis results
            - Preserves all STT-specific annotations
            - Maintains spatial coordinate information
        """
        self.adata=adata
        self.adata_aggr=adata_aggr
        add_reference(self.adata,'STT','spatial transition tensor with STT')

    def compute_pathway(self,pathway_dict,n_components=10):
        r"""Compute spatial pathways for cell transitions.
        
        This method identifies and computes transition pathways between cell states
        in spatial context, revealing developmental trajectories and cell fate decisions.

        Arguments:
            pathway_dict: dict
                Dictionary defining pathway parameters including:
                - start_states: list of starting cell states
                - end_states: list of target cell states
                - intermediate_states: optional list of intermediate states
                - constraints: optional spatial or temporal constraints

        Returns:
            dict
                Computed pathway information including:
                - transition_probabilities: array of state transition probabilities
                - path_coordinates: spatial coordinates of the pathway
                - path_states: sequence of states along the pathway

        Notes:
            - Pathways are computed using transition tensors
            - Results can be visualized using plot_pathway()
            - Considers both spatial and transcriptional information
        """
        return tl.compute_pathway(self.adata,self.adata_aggr,pathway_dict,n_components=n_components)

    def plot_pathway(self,label_fontsize=20,**kwargs):
        r"""Plot spatial transition pathways.
        
        This method visualizes computed transition pathways in spatial context,
        showing how cells transition between states while maintaining spatial organization.

        Arguments:
            label_fontsize: int, optional (default=20)
                Font size for axis labels.
            **kwargs: 
                Additional arguments passed to plotting function:
                - color_by: feature to color points by
                - alpha: transparency of points
                - size: size of points
                - title: plot title
                - save: path to save figure

        Returns:
            matplotlib.figure.Figure
                Figure object containing the pathway visualization.

        Notes:
            - Multiple pathways can be visualized simultaneously
            - Colors indicate transition states
            - Arrows show direction of transitions
            - Spatial coordinates are preserved in visualization
        """
        fig = pl.plot_pathway(self.adata,**kwargs)
        for ax in fig.axes:
            ax.set_xlabel('Embedding 1', fontsize=label_fontsize)
            ax.set_ylabel('Embedding 2', fontsize=label_fontsize)
        return fig

    def plot_tensor_pathway(self,pathway_name,**kwargs):
        r"""Plot tensor-based spatial pathways.
        
        This method creates a specialized visualization of transition pathways using
        tensor decomposition results, highlighting the continuous nature of cell state
        transitions in space.

        Arguments:
            pathway_name: str
                Name of the pathway to visualize.
            **kwargs:
                Additional plotting arguments:
                - show_transitions: bool, whether to show transition arrows
                - color_scheme: colormap for states
                - edge_width: width of transition edges
                - node_size: size of state nodes

        Returns:
            matplotlib.axes.Axes
                Axes object containing the tensor pathway plot.

        Notes:
            - Tensor visualization shows transition probabilities
            - Thickness of edges indicates transition likelihood
            - Colors represent different cell states
            - Spatial arrangement reflects tissue organization
        """
        ax=pl.plot_tensor_pathway(self.adata,self.adata_aggr,
                                  pathway_name = pathway_name,**kwargs)
        return ax

    def plot_tensor(self,list_attractor,**kwargs):
        r"""Plot spatial transition tensors.
        
        This method visualizes the learned transition tensors, showing how cells
        transition between states while considering spatial constraints.

        Arguments:
            list_attractor: list
                List of attractor regions or states to visualize.
            **kwargs:
                Additional plotting arguments:
                - mode: visualization mode ('2D' or '3D')
                - show_labels: whether to show state labels
                - cmap: colormap for tensor values
                - edge_threshold: minimum value for showing transitions

        Returns:
            dict
                Dictionary containing plotting results:
                - fig: matplotlib figure object
                - tensors: plotted tensor values
                - coordinates: spatial coordinates used

        Notes:
            - Tensors show transition probabilities between states
            - Higher values indicate more likely transitions
            - Spatial relationships are encoded in tensor structure
            - Can be used to identify major transition paths
        """
        return pl.plot_tensor(self.adata, self.adata_aggr, 
                          list_attractor = list_attractor,basis = self.spatial_loc,**kwargs)

    def construct_landscape(self,coord_key = 'X_xy_loc',**kwargs):
        r"""Construct spatial landscape for transition analysis.
        
        This method builds a continuous landscape representation of cell state transitions
        in spatial context, useful for understanding developmental potential and barriers.

        Arguments:
            coord_key: str, optional (default='X_xy_loc')
                Key for spatial coordinates in adata.obsm.
            **kwargs:
                Additional landscape construction arguments:
                - n_neighbors: number of neighbors for graph construction
                - smoothing: smoothing factor for landscape
                - resolution: grid resolution for landscape

        Returns:
            None
            Updates adata.obsm['trans_coord'] with landscape coordinates.

        Notes:
            - Landscape represents continuous transition space
            - Valleys indicate stable states
            - Ridges represent transition barriers
            - Coordinates preserve both spatial and state information
        """
        tl.construct_landscape(self.adata, coord_key = coord_key,**kwargs)
        self.adata.obsm['trans_coord'] = self.adata.uns['land_out']['trans_coord']

    def infer_lineage(self,**kwargs):
        r"""Infer cell lineage relationships from spatial transitions.
        
        This method reconstructs developmental lineages by analyzing transition
        patterns and spatial relationships between cell states.

        Arguments:
            **kwargs:
                Additional lineage inference arguments:
                - start_state: starting cell state
                - end_states: target cell states
                - min_branch_len: minimum branch length
                - max_steps: maximum steps in lineage

        Returns:
            dict
                Lineage inference results containing:
                - branches: list of inferred lineage branches
                - states: ordered states in each branch
                - confidence: confidence scores for branches

        Notes:
            - Combines spatial and transcriptional information
            - Identifies branching points in development
            - Considers transition probabilities between states
            - Results can be visualized with plot_landscape()
        """
        return pl.infer_lineage(self.adata,**kwargs)

    def plot_landscape(self,**kwargs):
        r"""Plot spatial landscape visualization.
        
        This method creates a visual representation of the constructed transition
        landscape, showing the continuous spectrum of cell states in spatial context.

        Arguments:
            **kwargs:
                Additional plotting arguments:
                - color_by: feature to color points by
                - show_trajectory: whether to show transition trajectories
                - contour: whether to show landscape contours
                - grid_size: resolution of landscape grid

        Returns:
            matplotlib.figure.Figure
                Figure containing the landscape visualization.

        Notes:
            - Colors indicate cell states or features
            - Contours show transition barriers
            - Arrows indicate preferred transition directions
            - Spatial relationships are preserved in layout
        """
        return pl.plot_landscape(self.adata,**kwargs)

    def plot_sankey(self,vector1, vector2):
        r"""Plot Sankey diagram for cell transitions.
        
        This method creates a Sankey diagram showing the flow of cells between
        different states or conditions, useful for understanding transition dynamics.

        Arguments:
            vector1: array-like
                Source state assignments for cells.
            vector2: array-like
                Target state assignments for cells.

        Returns:
            matplotlib.figure.Figure
                Figure containing the Sankey diagram.

        Notes:
            - Width of flows indicates transition frequency
            - Colors can represent different states
            - Useful for visualizing state changes
            - Shows conservation of cell numbers
        """
        return pl.plot_sankey(vector1, vector2)

    def plot_top_genes(self,**kwargs):
        r"""Plot top genes driving spatial transitions.
        
        This method identifies and visualizes genes that are most important in
        determining cell state transitions and spatial patterns.

        Arguments:
            **kwargs:
                Additional plotting arguments:
                - n_genes: number of top genes to show
                - groupby: how to group genes
                - show_labels: whether to show gene names
                - scale: whether to scale expression values

        Returns:
            matplotlib.figure.Figure
                Figure showing top genes visualization.

        Notes:
            - Highlights genes driving state transitions
            - Shows expression patterns across states
            - Useful for identifying key regulators
            - Can reveal transition mechanisms
        """
        return pl.plot_top_genes(self.adata,**kwargs)
    # End-of-file (EOF)
