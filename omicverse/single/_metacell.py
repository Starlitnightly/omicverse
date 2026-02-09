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
        r"""Initialize MetaCell object for constructing metacells using SEACells.

        Arguments:
            adata: AnnData object containing single-cell data
            use_rep (str): Key in adata.obsm for representation used to compute kernel (e.g., 'X_pca')
            n_metacells (int): Number of metacells to compute (default: None, auto-computed as n_cells//75)
            use_gpu (bool): Whether to use GPU for computation (default: False)
            verbose (bool): Whether to show progress information (default: True)
            n_waypoint_eigs (int): Number of eigenvectors for waypoint initialization (default: 10)
            n_neighbors (int): Number of nearest neighbors for graph construction (default: 15)
            convergence_epsilon (float): Convergence threshold for Franke-Wolfe algorithm (default: 1e-3)
            l2_penalty (float): L2 penalty for Franke-Wolfe algorithm (default: 0)
            max_franke_wolfe_iters (int): Maximum iterations for Franke-Wolfe algorithm (default: 50)
            use_sparse (bool): Whether to use sparse matrix operations (default: False)
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
        r"""Initialize metacell archetypes for SEACells algorithm.
        
        This method constructs the kernel matrix and initializes archetype positions.
        
        Arguments:
            **kwargs: Additional arguments passed to initialize_archetypes method
        """
        self.model.construct_kernel_matrix()
        self.M = self.model.kernel_matrix
        self.metacells_ad=None
        self.model.initialize_archetypes(**kwargs)
    
    def train(self,min_iter=10, max_iter=50,**kwargs):
        r"""Train the SEACells model to learn metacell assignments.
        
        Arguments:
            min_iter (int): Minimum number of training iterations (default: 10)
            max_iter (int): Maximum number of training iterations (default: 50)
            **kwargs: Additional arguments passed to the fit method
        """
        self.model.fit(min_iter=min_iter, max_iter=max_iter,**kwargs)
        self.model.seacells_dict=dict(zip(self.adata.obs.index.tolist(),
                                          self.adata.obs['SEACell'].tolist()))
        add_reference(self.adata,'SEACells','metacell clustering with SEACells')

    def predicted(self,method='soft',celltype_label='celltype',
                  summarize_layer='raw',minimum_weight=0.05):
        r"""Generate metacell summary from trained SEACells model.
        
        Arguments:
            method (str): Summarization method - 'soft' or 'hard' (default: 'soft')
            celltype_label (str): Key in adata.obs containing cell type information (default: 'celltype')
            summarize_layer (str): Layer to summarize for gene expression (default: 'raw')
            minimum_weight (float): Minimum weight threshold for soft assignment (default: 0.05)
            
        Returns:
            AnnData: Metacell AnnData object with summarized gene expression
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
        r"""Save the trained SEACells model to disk.
        
        Arguments:
            model_path (str): Path to save the model file (default: 'seacells/model.pkl')
        """
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        return None

    def load(self,model_path='seacells/model.pkl'):
        r"""Load a pre-trained SEACells model from disk.
        
        Arguments:
            model_path (str): Path to the model file (default: 'seacells/model.pkl')
        """
        import pickle
        with open(model_path, "rb") as f:
            self.model=pickle.load(f)
            self.M = self.model.kernel_matrix
            self.metacells_ad=None
            self.adata.obs['SEACell']=[self.model.seacells_dict[i] for i in self.adata.obs.index.tolist()]

    def step(self,n_steps=5):
        r"""Run additional training iterations step-wise.
        
        Arguments:
            n_steps (int): Number of additional training steps to run (default: 5)
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
    r"""Plot metacells on an existing axis.
    
    Arguments:
        ax: Matplotlib axis object for plotting
        metacells_ad: AnnData object containing metacell data
        use_rep (str): Representation to use for coordinates (default: 'X_umap')
        color (str): Color for metacell points (default: '#1f77b4')
        size (int): Size of metacell points (default: 15)
        edgecolors (str): Edge color for points (default: 'b')
        linewidths (float): Line width for point edges (default: 0.6)
        alpha (float): Transparency of points (default: 1)
        **kwargs: Additional arguments passed to scatter plot
        
    Returns:
        matplotlib.axes.Axes: The modified axis object
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
    r"""Transfer observation values from single-cell to metacell data.
    
    Arguments:
        ad: Metacell AnnData object
        adata: Original single-cell AnnData object  
        groupby (str): Column name in adata.obs to transfer
        type (str): Aggregation method - 'str', 'int', 'max', 'mean', 'min' (default: 'int')
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