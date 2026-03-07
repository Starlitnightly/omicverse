from ..external.SEACells import SEACells, summarize_by_SEACell,summarize_by_soft_SEACell
from ..external.SEACells import compute_celltype_purity,separation,compactness
from .._settings import add_reference
from .._registry import register_function
import pandas as pd


@register_function(
    aliases=["元细胞", "MetaCell", "metacell", "元细胞构建", "SEA细胞"],
    category="single",
    description="Construct metacells from single-cell data using SEACells algorithm for dimensionality reduction and noise reduction",
    examples=[
        "# Initialize MetaCell",
        "meta_obj = ov.single.MetaCell(adata, use_rep='X_pca', n_metacells=150)",
        "# Initialize archetypes",
        "meta_obj.initialize_archetypes()",
        "# Train the model",
        "meta_obj.train(min_iter=10, max_iter=50)",
        "# Generate metacells",
        "metacell_adata = meta_obj.predicted(method='soft', celltype_label='clusters')",
        "# Save and load model",
        "meta_obj.save('seacells/model.pkl')",
        "meta_obj.load('seacells/model.pkl')",
        "# Additional training steps",
        "meta_obj.step(n_steps=5)",
        "# Quality metrics",
        "purity = meta_obj.compute_celltype_purity(celltype_label='clusters')",
        "sep = meta_obj.separation(use_rep='X_pca')",
        "comp = meta_obj.compactness(use_rep='X_pca')"
    ],
    related=["single.get_obs_value", "single.plot_metacells", "pp.scale"]
)
class MetaCell(object):
    """SEACells-based metacell construction workflow.

    Parameters
    ----------
    adata : AnnData
        Input single-cell AnnData.
    use_rep : str
        Embedding key in ``adata.obsm`` used for kernel construction.
    n_metacells : int or None, optional
        Number of metacells to learn.
    use_gpu : bool, default=False
        Whether to enable GPU acceleration.
    """

    def __init__(self,adata,use_rep,
                 n_metacells=None,
                 use_gpu: bool = False,
                verbose: bool = True,
                n_waypoint_eigs: int = 10,
                n_neighbors: int = 15,
                convergence_epsilon: float = 1e-3,
                l2_penalty: float = 0,
                max_franke_wolfe_iters: int = 50,
                use_sparse: bool = False,) -> None:
        r"""Initialize a SEACells-based metacell model.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell data matrix and metadata.
        use_rep : str
            Embedding key in ``adata.obsm`` used to build the similarity kernel.
        n_metacells : int or None, default=None
            Number of metacells to learn. If ``None``, defaults to
            ``adata.n_obs // 75``.
        use_gpu : bool, default=False
            Whether to use GPU acceleration in SEACells.
        verbose : bool, default=True
            Whether to print model progress information.
        n_waypoint_eigs : int, default=10
            Number of eigenvectors used during waypoint initialization.
        n_neighbors : int, default=15
            Number of neighbors for graph/kernel construction.
        convergence_epsilon : float, default=1e-3
            Convergence threshold for the Franke-Wolfe optimization.
        l2_penalty : float, default=0
            L2 regularization strength.
        max_franke_wolfe_iters : int, default=50
            Maximum Franke-Wolfe iterations per optimization cycle.
        use_sparse : bool, default=False
            Whether to use sparse operations in the backend.
        """
        
        if n_metacells is None:
            n_metacells=adata.shape[0]//75


        self.model=SEACells(adata, 
                  build_kernel_on=use_rep, 
                  n_SEACells=n_metacells, 
                  use_gpu=use_gpu,
                  verbose=verbose,
                  n_waypoint_eigs=n_waypoint_eigs,
                    n_neighbors=n_neighbors,
                    convergence_epsilon=convergence_epsilon,
                    l2_penalty=l2_penalty,
                    max_franke_wolfe_iters=max_franke_wolfe_iters,
                    use_sparse=use_sparse)
        self.adata=adata



    def initialize_archetypes(self,**kwargs):
        r"""Construct kernel matrix and initialize archetypes.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            ``SEACells.initialize_archetypes``.

        Returns
        -------
        None
            Updates model state in place.
        """
        self.model.construct_kernel_matrix()
        self.M = self.model.kernel_matrix
        self.metacells_ad=None
        self.model.initialize_archetypes(**kwargs)
    
    def train(self,min_iter=10, max_iter=50,**kwargs):
        r"""Train the SEACells model.

        Parameters
        ----------
        min_iter : int, default=10
            Minimum number of optimization iterations.
        max_iter : int, default=50
            Maximum number of optimization iterations.
        **kwargs
            Additional keyword arguments passed to ``SEACells.fit``.

        Returns
        -------
        None
            Writes SEACell assignments to ``adata.obs['SEACell']``.
        """
        self.model.fit(min_iter=min_iter, max_iter=max_iter,**kwargs)
        self.model.seacells_dict=dict(zip(self.adata.obs.index.tolist(),
                                          self.adata.obs['SEACell'].tolist()))
        add_reference(self.adata,'SEACells','metacell clustering with SEACells')

    def predicted(self,method='soft',celltype_label='celltype',
                  summarize_layer='raw',minimum_weight=0.05):
        r"""Summarize single cells into metacell expression profiles.

        Parameters
        ----------
        method : str, default='soft'
            Aggregation strategy: ``'soft'`` uses weighted memberships;
            ``'hard'`` uses discrete SEACell assignments.
        celltype_label : str, default='celltype'
            Obs column used for cell-type metadata propagation.
        summarize_layer : str, default='raw'
            Expression layer used for metacell summarization.
        minimum_weight : float, default=0.05
            Minimum membership weight used in soft aggregation.

        Returns
        -------
        anndata.AnnData
            Metacell-level AnnData object.
        """
        if method=='soft':
            ad=summarize_by_soft_SEACell(self.adata, self.model.A_, 
                                        celltype_label=celltype_label,
                                        summarize_layer=summarize_layer, 
                                        minimum_weight=minimum_weight
                                    )
        elif method=='hard':
            ad=summarize_by_SEACell(self.adata, 
                                    SEACells_label='SEACell', 
                                    summarize_layer=summarize_layer,
                                    celltype_label=celltype_label
                                )
        self.metacells_ad=ad

        
        return ad
    
    def save(self,model_path='seacells/model.pkl'):
        r"""Save trained SEACells model to disk.

        Parameters
        ----------
        model_path : str, default='seacells/model.pkl'
            Output path for serialized model.

        Returns
        -------
        None
        """
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        return None

    def load(self,model_path='seacells/model.pkl'):
        r"""Load a serialized SEACells model from disk.

        Parameters
        ----------
        model_path : str, default='seacells/model.pkl'
            Path to model pickle file.

        Returns
        -------
        None
            Restores model state and reloads ``adata.obs['SEACell']``.
        """
        import pickle
        with open(model_path, "rb") as f:
            self.model=pickle.load(f)
            self.M = self.model.kernel_matrix
            self.metacells_ad=None
            self.adata.obs['SEACell']=[self.model.seacells_dict[i] for i in self.adata.obs.index.tolist()]

    def step(self,n_steps=5):
        r"""Run additional incremental optimization steps.

        Parameters
        ----------
        n_steps : int, default=5
            Number of extra ``SEACells.step()`` calls.

        Returns
        -------
        None
        """
        # You can force the model to run additional iterations step-wise using the .step() function
        print(f'Ran for {len(self.model.RSS_iters)} iterations')
        for _ in range(n_steps):
            self.model.step()
        print(f'Ran for {len(self.model.RSS_iters)} iterations')

    def compute_celltype_purity(self,celltype_label='celltype',):
        if self.metacells_ad is None:
            raise ValueError('Please run .predicted() first')
        else:
            return compute_celltype_purity(self.adata,
                                           celltype_label)
    
    def separation(self,use_rep='X_pca',nth_nbr=1,**kwargs):
        if self.metacells_ad is None:
            raise ValueError('Please run .predicted() first')
        else:
            return separation(self.adata,
                                           use_rep,nth_nbr=nth_nbr,**kwargs)
        
    def compactness(self,use_rep='X_pca',**kwargs):
        if self.metacells_ad is None:
            raise ValueError('Please run .predicted() first')
        else:
            return compactness(self.adata,
                                           use_rep,**kwargs)


@register_function(
    aliases=["绘制元细胞", "plot_metacells", "metacell_plot", "元细胞绘图", "可视化元细胞"],
    category="single",
    description="Plot metacells on existing axis with customizable visualization parameters",
    examples=[
        "# Basic metacell plotting",
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots(figsize=(6, 6))",
        "ov.single.plot_metacells(ax, adata, use_rep='X_umap')",
        "# Custom colors and sizes",
        "ov.single.plot_metacells(ax, adata, color='red', size=20, alpha=0.8)",
        "# With edge styling",
        "ov.single.plot_metacells(ax, adata, edgecolors='black', linewidths=1.0)"
    ],
    related=["single.MetaCell", "utils.embedding", "pl.embedding"]
)
def plot_metacells(ax,metacells_ad,use_rep='X_umap',color='#1f77b4',
                   size=15,
                   edgecolors='b',linewidths=0.6,alpha=1,**kwargs,):
    r"""Plot metacell centroids on a given embedding axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis used for plotting.
    metacells_ad : anndata.AnnData
        Metacell-level AnnData containing ``SEACell`` assignments.
    use_rep : str, default='X_umap'
        Embedding key in ``metacells_ad.obsm`` for coordinates.
    color : str, default='#1f77b4'
        Marker face color.
    size : int, default=15
        Marker size.
    edgecolors : str, default='b'
        Marker edge color.
    linewidths : float, default=0.6
        Marker edge width.
    alpha : float, default=1
        Marker transparency.
    **kwargs
        Additional keyword arguments passed to ``Axes.scatter``.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with metacell points overlaid.
    """
    umap = pd.DataFrame(metacells_ad.obsm[use_rep]).set_index(metacells_ad.obs_names).join(metacells_ad.obs["SEACell"])
    umap["SEACell"] = umap["SEACell"].astype("category")
    mcs = umap.groupby("SEACell").mean().reset_index()

    ax.scatter(mcs[0],mcs[1],s=size,c=color,
           edgecolors=edgecolors,linewidths=linewidths,
          alpha=alpha,**kwargs)
    return ax

@register_function(
    aliases=["获取观测值", "get_obs_value", "transfer_obs", "观测值转移", "元细胞注释转移"],
    category="single",
    description="Transfer observation values from single-cell to metacell data with various aggregation methods",
    examples=[
        "# Transfer cell type annotations",
        "ov.single.get_obs_value(metacell_adata, original_adata, 'celltype', type='str')",
        "# Transfer numeric values with mean aggregation",
        "ov.single.get_obs_value(metacell_adata, original_adata, 'score', type='mean')",
        "# Transfer with maximum aggregation",
        "ov.single.get_obs_value(metacell_adata, original_adata, 'batch', type='max')"
    ],
    related=["single.MetaCell", "single.plot_metacells", "utils.transfer_obs"]
)
def get_obs_value(ad,adata,groupby,type='int'):
    r"""Transfer per-cell annotations/statistics to metacells.

    Parameters
    ----------
    ad : anndata.AnnData
        Metacell AnnData object receiving aggregated values.
    adata : anndata.AnnData
        Original single-cell AnnData object with ``SEACell`` assignments.
    groupby : str
        Obs column in ``adata`` to aggregate into ``ad.obs``.
    type : str, default='int'
        Aggregation mode. ``'str'`` uses majority vote for categorical labels;
        other values are passed to ``groupby.agg`` (for example ``'mean'``,
        ``'max'``, ``'min'``).

    Returns
    -------
    None
        Writes aggregated values into ``ad.obs[groupby]``.
    """
    if type=='str':
        grouped_data = adata.obs.groupby('SEACell')[groupby]
        result_index1=[]
        for i in grouped_data.idxmax().index:
            result_index1.append(grouped_data.get_group(i).value_counts().index[0])
        result_index=pd.Series(result_index1,grouped_data.idxmax().index)
        ad.obs[groupby]=result_index.loc[[f'SEACell-{i}' for i in ad.obs.index]].values.tolist()
    else:
        #type can be set in `max`, `mean`, `min` et al.
        ad.obs[groupby]=adata.obs.groupby('SEACell').agg({groupby: type}).loc[[f'SEACell-{i}' for i in ad.obs.index.tolist()]][groupby].tolist()
    
    print(f'... {groupby} have been added to ad.obs[{groupby}]')
