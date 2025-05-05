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


from ._cosg import cosg
from ..externel.palantir.plot import plot_palantir_results,plot_branch_selection,plot_gene_trends
from ..externel.palantir.utils import run_diffusion_maps,determine_multiscale_space,run_magic_imputation
from ..externel.palantir.core import run_palantir
from ..externel.palantir.presults import select_branch_cells,compute_gene_trends


class TrajInfer(object):
    
    def __init__(self,adata:anndata.AnnData,
                 basis:str='X_umap',use_rep:str='X_pca',n_comps:int=50,
                 n_neighbors:int=15,
                groupby:str='clusters',):
        self.adata=adata
        self.use_rep=use_rep
        self.n_comps=n_comps
        self.basis=basis
        self.groupby=groupby
        self.n_neighbors=n_neighbors
        
        self.origin=None
        self.terminal=None
        
    def set_terminal_cells(self,terminal:list):
        self.terminal=terminal
        
    def set_origin_cells(self,origin:str):
        self.origin=origin
        
    def inference(self,method:str='palantir',**kwargs):
        
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
        else:
            print('Please input the correct method name, such as `palantir` or `diffusion_map`')
            return
        
    def palantir_plot_pseudotime(self,**kwargs):

        plot_palantir_results(self.adata,**kwargs)
        
    def palantir_cal_branch(self,**kwargs):

        masks = select_branch_cells(self.adata, **kwargs)
        plot_branch_selection(self.adata)

    def palantir_cal_gene_trends(self,layers:str="MAGIC_imputed_data"):

        gene_trends = compute_gene_trends(
            self.adata,
            expression_key=layers,
        )
        return gene_trends
        
    def palantir_plot_gene_trends(self,genes):
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
    #edge_weight_influence=1.0,
):
    """
    TODO: Typing
    """
    G = construct_graph(W)
    try:
        import forceatlas2 as fa2
    except ModuleNotFoundError:
        import sys
        print("Need forceatlas2-python!  Try 'pip install forceatlas2-python'.")
        sys.exit(-1)
    return fa2.forceatlas2(
            file_name,
            graph=G,
            n_jobs=n_jobs,
            target_change_per_node=target_change_per_node,
            target_steps=target_steps,
            is3d=is3d,
            memory=memory,
            random_state=random_state,
            init=init,
            scaling_ratio=scaling_ratio,
            gravity=gravity,
            #edge_weight_influence=edge_weight_influence,
        )


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
    #fa2_edge_weight_influence: float = 1.0,
) -> None:
    """Construct the Force-directed (FLE) graph.

    This implementation uses forceatlas2-python_ package, which is a Python wrapper of ForceAtlas2_.

    See [Jacomy14]_ for details on FLE.

    .. _forceatlas2-python: https://github.com/klarman-cell-observatory/forceatlas2-python
    .. _ForceAtlas2: https://github.com/klarman-cell-observatory/forceatlas2

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
        #edge_weight_influence=fa2_edge_weight_influence,
    )
    #data.register_attr(key, "basis")


def calc_force_directed_layout_(
    W,
    n_jobs: int,
    target_change_per_node: float,
    target_steps: int,
    random_state: int,
    init=None,
    fa2_scaling_ratio: float = 2.0,
    fa2_gravity: float = 1.0,
    fa2_edge_weight_influence: float = 1.0,
    verbose: bool = True,
    use_gpu: bool = True,
    device: torch.device = None,
    block_size: int = 1024,
):
    """
    使用 ForceAtlas2 计算 2D 布局。
    GPU 路径可直接处理稀疏边列表，避免密集矩阵转化导致 OOM。
    返回 shape (N,2) numpy array。
    """
    from fa2_modified import ForceAtlas2 as CPUForceAtlas2
    import scipy.sparse as sp
    # 构建 networkx 图并获取 N
    G = construct_graph(W)
    N = W.shape[0]

    # 检查 GPU 可用
    can_gpu = use_gpu and torch.cuda.is_available()
    if can_gpu and device is None:
        device = torch.device('cuda')

    # 初始位置
    pos_init = None
    if init is not None:
        arr = np.asarray(init)
        pos_init = torch.from_numpy(arr).float().to(device) if can_gpu else arr

    # GPU 分支：直接使用边列表，不转稠密
    if can_gpu:
        # 从 W 获取稀疏边
        if sp.issparse(W):
            coo = W.tocoo()
            src = torch.from_numpy(coo.row).long().to(device)
            dst = torch.from_numpy(coo.col).long().to(device)
            weights = torch.from_numpy(coo.data).float().to(device)
        else:
            W_arr = np.asarray(W)
            nz = np.nonzero(np.triu(W_arr,1))
            src = torch.from_numpy(nz[0]).long().to(device)
            dst = torch.from_numpy(nz[1]).long().to(device)
            weights = torch.from_numpy(W_arr[src.cpu(), dst.cpu()]).float().to(device)
        edges = torch.stack([src, dst], dim=1)
        # 调用 GPU 实现
        fa2 = GPUForceAtlas2(
            scaling_ratio=fa2_scaling_ratio,
            edge_weight_influence=fa2_edge_weight_influence,
            gravity=fa2_gravity,
            jitter_tolerance=target_change_per_node,
            barnes_hut_optimize=False,
            barnes_hut_theta=1.2,
            device=device,
            block_size=block_size,
        )
        pos_t = fa2.forceatlas2(
            edges=edges,
            weights=weights,
            N=N,
            pos=pos_init,
            iterations=target_steps,
        )
        return pos_t.cpu().numpy()

    # CPU 回退
    try:
        fa2 = CPUForceAtlas2(
            outboundAttractionDistribution=False,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=fa2_edge_weight_influence,
            jitterTolerance=target_change_per_node,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            scalingRatio=fa2_scaling_ratio,
            strongGravityMode=False,
            gravity=fa2_gravity,
            multiThreaded=False,
            verbose=verbose,
        )
    except ModuleNotFoundError:
        print("Need the `fa2` package. Try: pip install fa2-modified", file=sys.stderr)
        sys.exit(-1)

    pos_dict = fa2.forceatlas2_networkx_layout(
        G,
        pos=init,
        iterations=target_steps,
    )
    nodes = list(G.nodes())
    return np.vstack([pos_dict[n] for n in nodes])


def fle_(
    data,
    n_jobs: int = -1,
    rep: str = "diffmap",
    rep_ncomps: int = None,
    K: int = 50,
    full_speed: bool = False,
    target_change_per_node: float = 2.0,
    target_steps: int = 5000,
    random_state: int = 0,
    out_basis: str = "fle",
    fa2_scaling_ratio: float = 2.0,
    fa2_gravity: float = 1.0,
    fa2_edge_weight_influence: float = 1.0,
    verbose: bool = True,
    use_gpu: bool = True,
    gpu_device: torch.device = None,
    block_size: int = 1024,
) -> None:
    """
    使用 FA2 布局，优先 GPU 实现（支持稀疏），存入 data.obsm[out_basis]。
    """
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
        n_jobs,
        target_change_per_node,
        target_steps,
        random_state,
        init=None,
        fa2_scaling_ratio=fa2_scaling_ratio,
        fa2_gravity=fa2_gravity,
        fa2_edge_weight_influence=fa2_edge_weight_influence,
        verbose=verbose,
        use_gpu=use_gpu,
        device=gpu_device,
        block_size=block_size,
    )


import torch

torch.backends.cuda.matmul.allow_tf32 = True  # allow faster matmuls


def _batched_repulsion(x, y, mass, coefficient, block_size):
    """
    分块计算节点间斥力，避免一次性 OOM。
    x,y: [N] 张量；mass: [N]；返回 dx, dy ([N],[N])
    """
    N = x.size(0)
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(y)
    pos = torch.stack([x, y], dim=1)  # [N,2]
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        pi = pos[i:i_end].unsqueeze(1)   # [B,1,2]
        # broadcast to [B,N,2]
        diff = pi - pos.unsqueeze(0)     # [B,N,2]
        dist2 = (diff * diff).sum(dim=2).clamp(min=1e-9)  # [B,N]
        m_i = mass[i:i_end].unsqueeze(1) # [B,1]
        m_j = mass.unsqueeze(0)         # [1,N]
        factor = coefficient * m_i * m_j / dist2  # [B,N]
        force = diff * factor.unsqueeze(2)       # [B,N,2]
        net = force.sum(dim=1)                   # [B,2]
        dx[i:i_end] += net[:,0]
        dy[i:i_end] += net[:,1]
    return dx, dy


class GPUForceAtlas2:
    def __init__(
        self,
        scaling_ratio: float = 2.0,
        edge_weight_influence: float = 1.0,
        gravity: float = 1.0,
        jitter_tolerance: float = 1.0,
        barnes_hut_optimize: bool = False,
        barnes_hut_theta: float = 1.2,
        device: torch.device = torch.device('cuda'),
        block_size: int = 1024,
    ):
        self.scaling_ratio = scaling_ratio
        self.edge_weight_influence = edge_weight_influence
        self.gravity_const = gravity
        self.jitter_tolerance = jitter_tolerance
        self.barnes_hut_optimize = barnes_hut_optimize  # not used currently
        self.barnes_hut_theta = barnes_hut_theta        # not used currently
        self.device = device
        self.block_size = block_size

    @staticmethod
    def _attraction(x, y, src, dst, weights, coefficient):
        # src,dst: [E], weights: [E]
        pos = torch.stack([x, y], dim=1)  # [N,2]
        disp = pos[src] - pos[dst]        # [E,2]
        wfact = (coefficient * weights).unsqueeze(1)  # [E,1]
        f = disp * wfact                            # [E,2]
        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dx.index_add_(0, src,  f[:,0])
        dy.index_add_(0, src,  f[:,1])
        dx.index_add_(0, dst, -f[:,0])
        dy.index_add_(0, dst, -f[:,1])
        return dx, dy

    def _gravity(self, x, y, mass):
        dist = torch.sqrt(x*x + y*y).clamp(min=1e-9)
        factor = mass * self.gravity_const / dist
        return -x * factor, -y * factor

    def _adjust_and_move(self, x, y, dx, dy, old_dx, old_dy, mass, speed, speed_eff):
        N = x.size(0)
        # 计算 swinging 和 traction
        swinging = torch.sqrt((old_dx - dx)**2 + (old_dy - dy)**2) * mass
        traction = 0.5 * mass * torch.sqrt((old_dx + dx)**2 + (old_dy + dy)**2)
        total_swing = swinging.sum(); total_tract = traction.sum()
        # 更新 jitter tolerance
        est_jt = 0.05 * torch.sqrt(torch.tensor(N, dtype=torch.float, device=x.device))
        jt = self.jitter_tolerance * torch.clamp(
            est_jt * total_tract / (N*N), min=est_jt.sqrt(), max=torch.tensor(10.0, device=x.device)
        )
        if total_tract > 0 and (total_swing / total_tract) > 2.0:
            speed_eff = torch.clamp(speed_eff * 0.5, min=0.05)
            jt = torch.maximum(jt, torch.tensor(self.jitter_tolerance, device=x.device))
        # 更新 speed
        target = (jt * speed_eff * total_tract / total_swing) if total_swing>0 else torch.tensor(float('inf'), device=x.device)
        speed = speed + torch.minimum(target - speed, 0.5 * speed)
        # 应用移动
        factor = speed / (1.0 + torch.sqrt(speed * swinging))
        x = x + dx * factor; y = y + dy * factor
        return x, y, speed, speed_eff

    def forceatlas2(
        self,
        edges: torch.LongTensor,
        weights: torch.FloatTensor,
        N: int,
        pos: torch.Tensor = None,
        iterations: int = 100,
    ) -> torch.FloatTensor:
        """
        基于稀疏边列表的 GPU 实现。
        edges: [E,2], weights: [E], pos: [N,2] 或 None
        返回: [N,2]
        """
        # 初始化节点
        device = self.device
        if pos is None:
            x = torch.rand(N, device=device)
            y = torch.rand(N, device=device)
        else:
            x = pos[:,0].to(device); y = pos[:,1].to(device)
        mass = torch.zeros(N, device=device)
        # mass = 1 + degree
        deg = torch.zeros(N, device=device)
        deg.index_add_(0, edges[:,0], torch.ones_like(weights))
        deg.index_add_(0, edges[:,1], torch.ones_like(weights))
        mass = 1.0 + deg
        dx = torch.zeros(N, device=device); dy = torch.zeros(N, device=device)
        old_dx = torch.zeros_like(dx); old_dy = torch.zeros_like(dy)
        speed = torch.tensor(1.0, device=device)
        speed_eff = torch.tensor(1.0, device=device)
        src, dst = edges[:,0], edges[:,1]

        for _ in tqdm(range(iterations)):
            old_dx.copy_(dx); old_dy.copy_(dy)
            dx.zero_(); dy.zero_()
            # repulsion
            drx, dry = _batched_repulsion(x, y, mass, self.scaling_ratio, self.block_size)
            dx += drx; dy += dry
            # gravity
            ggx, ggy = self._gravity(x, y, mass)
            dx += ggx; dy += ggy
            # attraction
            dax, day = self._attraction(x, y, src, dst, weights, self.edge_weight_influence)
            dx += dax; dy += day
            # move
            x, y, speed, speed_eff = self._adjust_and_move(
                x, y, dx, dy, old_dx, old_dy, mass, speed, speed_eff
            )
        return torch.stack([x, y], dim=1)

