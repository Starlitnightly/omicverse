import scanpy as sc
import pandas as pd
import anndata
import numpy as np
import time
import numpy as np
from scipy.sparse import issparse, csr_matrix
import igraph
import torch
from tqdm import tqdm
import sys
from ..utils._neighboors import update_rep, eff_n_jobs,neighbors,W_from_rep
from .._settings import add_reference
from .._registry import register_function

from ._cosg import cosg
from ..external.palantir.plot import plot_palantir_results,plot_branch_selection,plot_gene_trends
from ..external.palantir.utils import run_diffusion_maps,determine_multiscale_space,run_magic_imputation
from ..external.palantir.core import run_palantir
from ..external.palantir.presults import select_branch_cells,compute_gene_trends


@register_function(
    aliases=["轨迹推断", "TrajInfer", "trajectory_inference", "轨迹分析", "发育轨迹"],
    category="single",
    description="Comprehensive trajectory inference for single-cell data using multiple algorithms including Palantir, diffusion maps, and Slingshot",
    prerequisites={
        'functions': ['pca', 'neighbors'],
        'optional_functions': ['leiden', 'umap']
    },
    requires={
        'obsm': ['X_pca'],
        'uns': ['neighbors']
    },
    produces={
        'obs': ['palantir_pseudotime'],
        'obsm': ['X_palantir', 'branch_probs'],
        'uns': ['palantir_imp', 'gene_trends']
    },
    auto_fix='auto',
    examples=[
        "# Initialize TrajInfer",
        "traj = ov.single.TrajInfer(adata, basis='X_umap', groupby='clusters',",
        "                           use_rep='X_pca', n_comps=50)",
        "# Set origin and terminal cells",
        "traj.set_origin_cells('stem_cells')",
        "traj.set_terminal_cells(['differentiated_A', 'differentiated_B'])",
        "# Diffusion map trajectory inference",
        "traj.inference(method='diffusion_map')",
        "# Slingshot trajectory inference",
        "traj.inference(method='slingshot', num_epochs=1)",
        "# Palantir trajectory inference",
        "traj.inference(method='palantir', num_waypoints=500)",
        "# Visualize Palantir results",
        "traj.palantir_plot_pseudotime(embedding_basis='X_umap', cmap='RdBu_r')",
        "# Calculate branch probabilities",
        "traj.palantir_cal_branch(eps=0)",
        "# Compute gene expression trends",
        "gene_trends = traj.palantir_cal_gene_trends(layers='MAGIC_imputed_data')",
        "# Plot gene trends along trajectories",
        "traj.palantir_plot_gene_trends(['gene1', 'gene2', 'gene3'])"
    ],
    related=["utils.cal_paga", "utils.plot_paga", "pp.neighbors", "external.palantir"]
)
class TrajInfer(object):
    r"""Trajectory inference class for single-cell data analysis.
    
    This class provides methods for inferring developmental trajectories using
    various algorithms including Palantir, diffusion maps, and Slingshot.
    """
    
    def __init__(self,adata:anndata.AnnData,
                 basis:str='X_umap',use_rep:str='X_pca',n_comps:int=50,
                 n_neighbors:int=15,
                groupby:str='clusters',):
        r"""Initialize trajectory inference object.
        
        Arguments:
            adata: AnnData object containing single-cell data
            basis (str): Embedding key in adata.obsm for visualization (default: 'X_umap')
            use_rep (str): Representation key in adata.obsm for trajectory computation (default: 'X_pca')
            n_comps (int): Number of components to use from representation (default: 50)
            n_neighbors (int): Number of neighbors for graph construction (default: 15)
            groupby (str): Key in adata.obs containing cluster annotations (default: 'clusters')
        """
        self.adata=adata
        self.use_rep=use_rep
        self.n_comps=n_comps
        self.basis=basis
        self.groupby=groupby
        self.n_neighbors=n_neighbors
        
        self.origin=None
        self.terminal=None
        
    def set_terminal_cells(self,terminal:list):
        r"""Set terminal cell types for trajectory inference.
        
        Arguments:
            terminal (list): List of terminal cell type names
        """
        self.terminal=terminal
        
    def set_origin_cells(self,origin:str):
        r"""Set origin cell type for trajectory inference.
        
        Arguments:
            origin (str): Name of the origin cell type
        """
        self.origin=origin
        
    def inference(self,method:str='palantir',**kwargs):
        r"""Perform trajectory inference using specified method.
        
        Arguments:
            method (str): Trajectory inference method - 'palantir', 'diffusion_map', or 'slingshot' (default: 'palantir')
            **kwargs: Additional arguments passed to the chosen method
            
        Returns:
            Depends on method: Palantir results object, None, or sets internal attributes
        """
        
        if method=='palantir':

            dm_res = run_diffusion_maps(self.adata,
                                                       pca_key=self.use_rep, 
                                                       n_components=self.n_comps)
            ms_data = determine_multiscale_space(self.adata)
            imputed_X = run_magic_imputation(self.adata)

            sc.tl.rank_genes_groups(self.adata, groupby=self.groupby, 
                        method='t-test',use_rep=self.use_rep,)
            cosg(self.adata, key_added=f'{self.groupby}_cosg', groupby=self.groupby)
            
            ## terminal cells calculation
            if self.terminal is None:
                terminal_states = None  # let Palantir compute terminal cells
            else:
                terminal_index=[]
                for t in self.terminal:
                    gene=sc.get.rank_genes_groups_df(self.adata, group=t, key=f'{self.groupby}_cosg')['names'][0]
                    terminal_index.append(self.adata[self.adata.obs[self.groupby]==t].to_df()[gene].sort_values().index[-1])
                
                terminal_states = pd.Series(
                    self.terminal,
                    index=terminal_index,
                )
            
            ## origin cells calculation
            origin_cell=self.origin
            gene=sc.get.rank_genes_groups_df(self.adata, group=origin_cell, key=f'{self.groupby}_cosg')['names'][0]
            origin_cell_index=self.adata[self.adata.obs[self.groupby]==origin_cell].to_df()[gene].sort_values().index[-1]
            
            start_cell = origin_cell_index
            pr_res = run_palantir(
                self.adata, early_cell=start_cell, terminal_states=terminal_states,
                **kwargs
            )
            
            self.adata.obs['palantir_pseudotime']=pr_res.pseudotime
            add_reference(self.adata,'Palantir','trajectory inference with Palantir')
            return pr_res
        elif method=='diffusion_map':
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, n_pcs=self.n_comps,
               use_rep=self.use_rep)
            sc.tl.diffmap(self.adata)
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, use_rep='X_diffmap')
            sc.tl.draw_graph(self.adata)
            self.adata.uns['iroot'] = np.flatnonzero(self.adata.obs[self.groupby]  == self.origin)[0]
            sc.tl.dpt(self.adata)
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, n_pcs=self.n_comps,
               use_rep=self.use_rep)
            add_reference(self.adata,'diffusion_map','trajectory inference with diffusion map')
        elif method=='slingshot':
            #sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, n_pcs=self.n_comps,
            #   use_rep=self.use_rep)
            #sc.tl.umap(self.adata)
            from ._pyslingshot import Slingshot
            slingshot = Slingshot(self.adata, 
                      celltype_key=self.groupby, 
                      obsm_key=self.basis, 
                      start_node=self.origin,
                      end_nodes=self.terminal,
                      debug_level='verbose')
            slingshot.fit(**kwargs)
            pseudotime = slingshot.unified_pseudotime
            self.adata.obs['slingshot_pseudotime']=pseudotime
            self.slingshot=slingshot
            add_reference(self.adata,'slingshot','trajectory inference with slingshot')
        elif method=='sctour':
            import sctour as sct
            tnode = sct.train.Trainer(
                self.adata, loss_mode='nb', 
                **kwargs
            )
            tnode.train()
            self.adata.obs['sctour_pseudotime'] = tnode.get_time()
            mix_zs, zs, pred_zs = tnode.get_latentsp(alpha_z=0.5, alpha_predz=0.5)
            self.adata.obsm['X_TNODE'] = mix_zs
            self.adata.obsm['X_VF'] = tnode.get_vector_field(
                self.adata.obs['sctour_pseudotime'].values, 
                self.adata.obsm['X_TNODE']
            )
            add_reference(self.adata,'sctour','trajectory inference with sctour')
            self.tnode=tnode
        else:
            print('Please input the correct method name, such as `palantir` or `diffusion_map`')
            return
        
    def palantir_plot_pseudotime(self,**kwargs):
        r"""Plot Palantir pseudotime results.
        
        Arguments:
            **kwargs: Additional arguments passed to plot_palantir_results
        """
        plot_palantir_results(self.adata,**kwargs)
        
    def palantir_cal_branch(self,**kwargs):
        r"""Calculate and plot branch selection for Palantir results.
        
        Arguments:
            **kwargs: Additional arguments passed to select_branch_cells
        """
        masks = select_branch_cells(self.adata, **kwargs)
        plot_branch_selection(self.adata)

    def palantir_cal_gene_trends(self,layers:str="MAGIC_imputed_data"):
        r"""Calculate gene expression trends along Palantir trajectories.
        
        Arguments:
            layers (str): Key for expression data layer (default: "MAGIC_imputed_data")
            
        Returns:
            Gene trends object containing trajectory-associated expression patterns
        """
        gene_trends = compute_gene_trends(
            self.adata,
            expression_key=layers,
        )
        return gene_trends
        
    def palantir_plot_gene_trends(self,genes):
        r"""Plot gene expression trends along Palantir trajectories.
        
        Arguments:
            genes (list): List of gene names to plot
            
        Returns:
            Matplotlib figure object containing gene trend plots
        """
        #genes = ['Cdca3','Rasl10a','Mog','Aqp4']
        return plot_gene_trends(self.adata, genes)
    
import networkx as nx
from scipy.sparse import csr_matrix

def construct_graph_(
    W: csr_matrix, directed: bool = False, adjust_weights: bool = True
) -> nx.Graph:
    """
    将稀疏邻接矩阵 W 转换为 networkx 图，并保留权重属性。
    """
    assert issparse(W), "W must be a scipy.sparse matrix"

    # 根据 adjust_weights 选项调整权重
    s, t = W.nonzero()
    w = W.data
    if not directed:
        mask = s < t
        s, t, w = s[mask], t[mask], w[mask]

    if adjust_weights:
        # 标准化并四舍五入到小数点后两位
        w = ((w / np.median(w)) * 100.0 + 0.5).astype(int) / 100.0
        mask = w > 0
        s, t, w = s[mask], t[mask], w[mask]

    # 构建 networkx 图
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(W.shape[0]))
    edges = zip(s.tolist(), t.tolist(), w.tolist())
    G.add_weighted_edges_from(edges)
    return G

def construct_graph(
    W, directed: bool = False, adjust_weights: bool = True
) -> "igraph":

    assert issparse(W)

    s, t = W.nonzero()
    w = W.data

    if not directed:
        idx = s < t
        s = s[idx]
        t = t[idx]
        w = w[idx]

    if adjust_weights:
        w = ((w / np.median(w)) * 100.0 + 0.5).astype(
            int
        ) / 100.0  # round to 2 decimal points
        idx = w > 0.0
        if idx.sum() < w.size:
            s = s[idx]
            t = t[idx]
            w = w[idx]

    G = igraph.Graph(directed=directed)
    G.add_vertices(W.shape[0])
    G.add_edges(zip(s, t))
    G.es["weight"] = w

    return G

def calc_force_directed_layout(
    W,
    file_name,
    n_jobs,
    target_change_per_node,
    target_steps,
    is3d,
    memory,
    random_state,
    init=None,
    scaling_ratio=2.0, 
    gravity=1.0, 
    edge_weight_influence=1.0,
    use_gpu=False,  # New parameter to enable GPU acceleration
    optimized_threading=True,  # Enable optimized threading algorithm
    compile_cython=True,  # 是否尝试编译Cython模块以加速
):
    """
    Use our custom ForceAtlas2 implementation to calculate the layout
    """
    G = construct_graph(W)
    try:
        from ..external.forcedirect2.forceatlas2 import ForceAtlas2
        
        # 如果请求编译并且fa2util模块需要编译
        if compile_cython:
            try:
                import os
                import sys
                import importlib.util
                
                # 检查是否已经编译
                force_dir = os.path.dirname(os.path.abspath(importlib.util.find_spec("omicverse.external.forcedirect2.forceatlas2").origin))
                compiled_module_exists = any(f.startswith('fa2util.') and (f.endswith('.so') or f.endswith('.pyd')) 
                                           for f in os.listdir(force_dir))
                
                # 如果没有编译过，尝试编译
                if not compiled_module_exists:
                    print("检测到未编译的fa2util模块。尝试编译以获得10-100倍速度提升...")
                    compile_script = os.path.join(force_dir, "compile.py")
                    
                    # 如果compile.py存在，执行它
                    if os.path.exists(compile_script):
                        import subprocess
                        subprocess.run([sys.executable, compile_script], cwd=force_dir)
                    else:
                        print(f"未找到编译脚本: {compile_script}")
                        print("请手动编译fa2util模块以获得更好性能")
                        print("cd " + force_dir)
                        print("python setup.py build_ext --inplace")
            except Exception as e:
                print(f"尝试编译模块时出错: {str(e)}")
                print("将使用未编译的Python版本 (较慢)")
        
        # Initialize ForceAtlas2 with our parameters
        forceatlas2 = ForceAtlas2(
            # Behavior
            outboundAttractionDistribution=True,
            edgeWeightInfluence=edge_weight_influence,
            
            # Performance
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=True if not use_gpu else False,  # Only use multithreading if not using GPU
            numThreads=n_jobs if n_jobs > 0 else None,  # Use specified number of threads
            useGPU=use_gpu,  # Enable GPU acceleration if requested
            optimizedThreading=optimized_threading,  # Enable optimized threading
            
            # Tuning
            scalingRatio=scaling_ratio,
            strongGravityMode=False,
            gravity=gravity,
            
            # Log
            verbose=True
        )
        
        # Convert igraph to adjacency matrix for forceatlas2
        import numpy as np
        from scipy.sparse import csr_matrix
        
        edges = G.get_edgelist()
        weights = G.es["weight"]
        n_vertices = G.vcount()
        
        rows = [edge[0] for edge in edges]
        cols = [edge[1] for edge in edges]
        
        # Create sparse adjacency matrix
        adj_matrix = csr_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))
        
        # Make symmetric if graph is undirected
        if not G.is_directed():
            adj_matrix = adj_matrix + adj_matrix.T
        
        # Run ForceAtlas2
        positions = forceatlas2.forceatlas2(adj_matrix, iterations=target_steps)
        
        # If 3D layout is requested, add a third coordinate with zeros
        if is3d:
            return np.column_stack([positions, np.zeros(len(positions))])
        
        return np.array(positions)
        
    except ImportError:
        # Fallback to external forceatlas2 if our implementation is not available
        import sys
        print("Could not find custom ForceAtlas2 implementation!")
        sys.exit(-1)


def fle(
    data,
    file_name: str = None,
    n_jobs: int = -1,
    rep: str = "diffmap",
    rep_ncomps: int = None,
    K: int = 50,
    full_speed: bool = False,
    target_change_per_node: float = 2.0,
    target_steps: int = 5000,
    is3d: bool = False,
    memory: int = 8,
    random_state: int = 0,
    out_basis: str = "fle",
    fa2_scaling_ratio: float = 2.0,
    fa2_gravity: float = 1.0,
    fa2_edge_weight_influence: float = 1.0,
    use_gpu: bool = False,  # New parameter to enable GPU acceleration
    optimized_threading: bool = True,  # Enable optimized threading algorithm
    compile_cython: bool = True,  # 是否尝试编译Cython模块以加速
) -> None:
    """Construct the Force-directed (FLE) graph.

    This implementation uses our custom ForceAtlas2 implementation, which is a multilthreaded version
    of the original ForceAtlas2 algorithm.

    See [Jacomy14]_ for details on FLE.

    Parameters
    ----------
    data: ``pegasusio.MultimodalData``
        Annotated data matrix with rows for cells and columns for genes.

    file_name: ``str``, optional, default: ``None``
        Temporary file to store the coordinates as the input to forceatlas2. If ``None``, use ``tempfile.mkstemp`` to generate file name.

    n_jobs: ``int``, optional, default: ``-1``
        Number of threads to use. If ``-1``, use all physical CPU cores.

    rep: ``str``, optional, default: ``"diffmap"``
        Representation of data used for the calculation. By default, use Diffusion Map coordinates. If ``None``, use the count matrix ``data.X``.

    rep_ncomps: ``int``, optional (default: None)
        Number of components to be used in `rep`. If rep_ncomps == None, use all components; otherwise, use the minimum of rep_ncomps and rep's dimensions.

    K: ``int``, optional, default: ``50``
        Number of nearest neighbors to be considered during the computation.

    full_speed: ``bool``, optional, default: ``False``
        * If ``True``, use multiple threads in constructing ``hnsw`` index. However, the kNN results are not reproducible.
        * Otherwise, use only one thread to make sure results are reproducible.

    target_change_per_node: ``float``, optional, default: ``2.0``
        Target change per node to stop ForceAtlas2.

    target_steps: ``int``, optional, default: ``5000``
        Maximum number of iterations before stopping the ForceAtlas2 algorithm.

    is3d: ``bool``, optional, default: ``False``
        If ``True``, calculate 3D force-directed layout.

    memory: ``int``, optional, default: ``8``
        Memory size in GB for the Java FA2 component. By default, use 8GB memory.

    random_state: ``int``, optional, default: ``0``
        Random seed set for reproducing results.

    out_basis: ``str``, optional, default: ``"fle"``
        Key name for calculated FLE coordinates to store.
    
    fa2_scaling_ratio: ``float``, optional, default: ``2.0``
        Scaling ratio parameter for ForceAtlas2.
        
    fa2_gravity: ``float``, optional, default: ``1.0``
        Gravity parameter for ForceAtlas2.
        
    fa2_edge_weight_influence: ``float``, optional, default: ``1.0``
        Edge weight influence parameter for ForceAtlas2.
        
    use_gpu: ``bool``, optional, default: ``False``
        Whether to use GPU acceleration for force calculations. Requires PyTorch to be installed.
        
    optimized_threading: ``bool``, optional, default: ``True``
        Whether to use optimized threading algorithm for better parallel performance.
        
    compile_cython: ``bool``, optional, default: ``True``
        Whether to try compiling fa2util module with Cython for 10-100x speedup.

    Returns
    -------
    ``None``

    Update ``data.obsm``:
        * ``data.obsm['X_' + out_basis]``: FLE coordinates of the data.

    Examples
    --------
    >>> pg.fle(data)
    """

    if file_name is None:
        import tempfile

        _, file_name = tempfile.mkstemp()

    rep = update_rep(rep)
    n_jobs = eff_n_jobs(n_jobs)

    if ("W_" + rep) not in data.uns:
        neighbors(
            data,
            K=K,
            rep=rep,
            n_comps=rep_ncomps,
            n_jobs=n_jobs,
            random_state=random_state,
            full_speed=full_speed,
        )

    key = f"X_{out_basis}"
    data.obsm[key] = calc_force_directed_layout(
        W_from_rep(data, rep),
        file_name,
        n_jobs,
        target_change_per_node,
        target_steps,
        is3d,
        memory,
        random_state,
        scaling_ratio=fa2_scaling_ratio,
        gravity=fa2_gravity,
        edge_weight_influence=fa2_edge_weight_influence,
        use_gpu=use_gpu,
        optimized_threading=optimized_threading,
        compile_cython=compile_cython,
    )


def diffmap_fle(
    adata,
    use_rep: str = 'X_pca',
    n_pcs: int = 50,
    n_comps: int = 15,
    n_neighbors: int = 30,
    random_state: int = 0,
    out_basis: str = "fle",
    target_steps: int = 5000,
    use_gpu: bool = False,
    optimized_threading: bool = True,
    compile_cython: bool = True,
    fa2_scaling_ratio: float = 2.0,
    fa2_gravity: float = 1.0,
    fa2_edge_weight_influence: float = 1.0,
    n_jobs: int = -1,
) -> None:
    """Calculate diffusion map and then use it for Force-directed Layout Embedding (FLE).
    
    This function first computes diffusion maps using scanpy and then applies the optimized
    ForceAtlas2 algorithm on the diffusion map coordinates to create a force-directed layout.
    
    Parameters
    ----------
    adata: ``anndata.AnnData``
        Annotated data matrix with rows for cells and columns for genes.
        
    use_rep: ``str``, optional, default: ``'X_pca'``
        The dimensional reduction to use for neighbors calculation before diffmap.
        
    n_pcs: ``int``, optional, default: ``50``
        Number of PCs to use for neighbors calculation.
        
    n_comps: ``int``, optional, default: ``15``
        Number of diffusion components to compute.
        
    n_neighbors: ``int``, optional, default: ``30``
        Number of neighbors for building the neighborhood graph.
        
    random_state: ``int``, optional, default: ``0``
        Random seed for reproducibility.
        
    out_basis: ``str``, optional, default: ``"fle"``
        Key name for calculated FLE coordinates to store.
        
    target_steps: ``int``, optional, default: ``5000``
        Maximum number of iterations for the ForceAtlas2 algorithm.
        
    use_gpu: ``bool``, optional, default: ``False``
        Whether to use GPU acceleration for force calculations.
        
    optimized_threading: ``bool``, optional, default: ``True``
        Whether to use optimized threading algorithm for better parallel performance.
        
    compile_cython: ``bool``, optional, default: ``True``
        Whether to try compiling fa2util module with Cython for speedup.
        
    fa2_scaling_ratio: ``float``, optional, default: ``2.0``
        Scaling ratio parameter for ForceAtlas2.
        
    fa2_gravity: ``float``, optional, default: ``1.0``
        Gravity parameter for ForceAtlas2.
        
    fa2_edge_weight_influence: ``float``, optional, default: ``1.0``
        Edge weight influence parameter for ForceAtlas2.
        
    n_jobs: ``int``, optional, default: ``-1``
        Number of threads to use. If ``-1``, use all physical CPU cores.
        
    Returns
    -------
    ``None``
    
    Update ``adata.obsm``:
        * ``adata.obsm['X_diffmap']``: Diffusion map coordinates
        * ``adata.obsm['X_' + out_basis]``: FLE coordinates
        
    Examples
    --------
    >>> diffmap_fle(adata, use_gpu=True)
    """
    # Step 1: Compute diffusion map using scanpy
    import scanpy as sc
    
    print("Computing neighbors for diffusion map...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, 
                   use_rep=use_rep, random_state=random_state)
    
    print(f"Computing diffusion map with {n_comps} components...")
    sc.tl.diffmap(adata, n_comps=n_comps)
    
    # Store the diffusion map connectivity matrix for FLE
    W_diffmap = adata.obsp['connectivities']
    
    # Step 2: Apply ForceAtlas2 on the diffusion map
    print("Calculating force-directed layout from diffusion map...")
    
    import tempfile
    _, file_name = tempfile.mkstemp()
    
    n_jobs = eff_n_jobs(n_jobs)
    
    key = f"X_{out_basis}"
    adata.obsm[key] = calc_force_directed_layout(
        W_diffmap,
        file_name,
        n_jobs,
        2.0,  # target_change_per_node
        target_steps,
        False,  # is3d
        8,  # memory
        random_state,
        scaling_ratio=fa2_scaling_ratio,
        gravity=fa2_gravity,
        edge_weight_influence=fa2_edge_weight_influence,
        use_gpu=use_gpu,
        optimized_threading=optimized_threading,
        compile_cython=compile_cython,
    )
    
    print(f"Force-directed layout calculated and stored in adata.obsm['X_{out_basis}']")
    return None

