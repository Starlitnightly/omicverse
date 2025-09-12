# ---- 依赖：torch_sparse/torch_scatter 同你之前的版本 ----
import torch
import numpy as np
import pandas as pd
from natsort import natsorted
from scanpy import _utils, logging as logg

try:
    from torch_sparse import SparseTensor
except ImportError as e:
    raise ImportError(
        "Requires torch_sparse. Install PyG deps first, e.g.:\n"
        "pip install torch==2.*\n"
        "pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-$(python -c 'import torch;print(torch.__version__)').html"
    ) from e

try:
    from torch_scatter import scatter_add, scatter_max
    HAS_SCATTER = True
except Exception:
    HAS_SCATTER = False

# ---------- 工具函数（如你已有可复用，避免重复定义） ----------
def _pick_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _ensure_coo(adjacency):
    from scipy.sparse import coo_matrix
    return adjacency if isinstance(adjacency, coo_matrix) else adjacency.tocoo()

def _relabel_contiguous(comm: torch.Tensor) -> tuple[torch.Tensor, int]:
    uniq, inv = torch.unique(comm, sorted=True, return_inverse=True)
    return inv, int(uniq.numel())

def _scatter_add_native(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    return out.scatter_add_(0, index, src)

def _rowwise_argmax(rows: torch.Tensor, values: torch.Tensor, cols: torch.Tensor, n_rows: int):
    if HAS_SCATTER:
        max_val, arg_idx = scatter_max(values, rows, dim=0, dim_size=n_rows)  # (n_rows,)
        best_cols = torch.where(arg_idx >= 0, cols[arg_idx.clamp_min(0)],
                                torch.full((n_rows,), -1, device=cols.device, dtype=cols.dtype))
        return max_val, best_cols
    # fallback
    max_val = torch.full((n_rows,), float("-inf"), device=values.device, dtype=values.dtype)
    max_val.scatter_reduce_(0, rows, values, reduce="amax", include_self=False)
    eq = (values == max_val[rows])
    safe_cols = torch.where(eq, cols, torch.full_like(cols, -1))
    best_cols = torch.full((n_rows,), -1, device=cols.device, dtype=cols.dtype)
    best_cols.scatter_reduce_(0, rows, safe_cols, reduce="amax", include_self=False)
    return max_val, best_cols

def _build_sparse_ic(src_nodes, dst_nodes, weights, communities, n_nodes, n_comms) -> SparseTensor:
    col_c = communities[dst_nodes]
    extra_row = torch.arange(n_nodes, device=src_nodes.device, dtype=src_nodes.dtype)
    extra_col = communities
    extra_val = torch.zeros(n_nodes, device=weights.device, dtype=weights.dtype)
    row = torch.cat([src_nodes, extra_row], dim=0)
    col = torch.cat([col_c,     extra_col], dim=0)
    val = torch.cat([weights,   extra_val], dim=0)
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=(n_nodes, n_comms)).coalesce()

# ---------- 本地移动（local move）阶段（稀疏 + γ 在期望项） ----------
def _local_move_gpu_sparse(A: SparseTensor, resolution: float, n_iterations: int,
                           random_state: int = 0, log_prefix: str = "") -> torch.Tensor:
    torch.manual_seed(random_state)
    n = A.sparse_size(0)
    deg = A.sum(dim=1).to_dense()            # (n,)
    two_m = A.sum().item()
    if two_m <= 0:
        raise ValueError("Graph has no edges (2m == 0).")

    comm = torch.arange(n, device=deg.device, dtype=torch.long)
    sigma_tot = deg.clone()

    Ai, Aj, Aw = A.coo()
    for it in range(n_iterations):
        comm, n_comms = _relabel_contiguous(comm)

        M = _build_sparse_ic(Ai, Aj, Aw, comm, n_nodes=n, n_comms=n_comms)
        rowptr, colc, k_ic = M.csr()  # CSR
        rows = torch.arange(n, device=deg.device, dtype=torch.long)
        rows = torch.repeat_interleave(rows, rowptr[1:] - rowptr[:-1])

        sigma_c = sigma_tot[colc]
        comm_rows = comm[rows]
        ki_rows = deg[rows]
        sigma_eff = sigma_c - (colc == comm_rows).to(sigma_c.dtype) * ki_rows

        gains = k_ic - (resolution * ki_rows * sigma_eff) / (two_m)

        best_gain, best_c = _rowwise_argmax(rows, gains, colc, n_rows=n)

        stay_mask = (colc == comm_rows)
        neg_inf = torch.full_like(gains, float("-inf"))
        stay_entries = torch.where(stay_mask, gains, neg_inf)
        if HAS_SCATTER:
            stay_gain, _ = scatter_max(stay_entries, rows, dim=0, dim_size=n)
        else:
            stay_gain = torch.full((n,), float("-inf"), device=deg.device, dtype=gains.dtype)
            stay_gain.scatter_reduce_(0, rows, stay_entries, reduce="amax", include_self=False)

        improve = best_gain > (stay_gain + 1e-12)
        new_comm = torch.where(improve, best_c, comm)

        moved = int((new_comm != comm).sum().item())
        #if log_prefix:
        #    logg.info(f"{log_prefix} local-move iter {it+1}/{n_iterations}: moved {moved} nodes")
        comm = new_comm
        if moved == 0:
            break

        # 重新按 comm 聚合 Σ_tot（也可做增量更新，这里为清晰直接重算）
        comm, n_comms = _relabel_contiguous(comm)
        if HAS_SCATTER:
            sigma_tot = scatter_add(deg, comm, dim=0, dim_size=n_comms)
        else:
            sigma_tot = _scatter_add_native(deg, comm, dim_size=n_comms)

    # 最终保证连续标签
    comm, _ = _relabel_contiguous(comm)
    return comm

# ---------- 收缩：把社区收缩为超节点，构造新的稀疏图 ----------
def _contract_graph_sparse(A: SparseTensor, comm: torch.Tensor) -> SparseTensor:
    """
    输入：
      A    : (n x n) 稀疏对称图（可含自环）
      comm : (n,) 节点->社区 的映射（0..C-1）
    输出：
      A2   : (C x C) 社区图（保留自环；权重为社际/社内边权之和）
    """
    n = A.sparse_size(0)
    C = int(comm.max().item()) + 1
    Ai, Aj, Aw = A.coo()
    Ci = comm[Ai]
    Cj = comm[Aj]
    A2 = SparseTensor(row=Ci, col=Cj, value=Aw, sparse_sizes=(C, C)).coalesce()
    return A2

# ---------- 多层管线：本地移动 -> 收缩 -> 重复，直到稳定 ----------
def leiden_gpu_sparse_multilevel(
    adata,
    resolution: float = 1.0,
    *,
    random_state: int = 0,
    key_added: str = "leiden_pyg",
    adjacency=None,
    use_weights: bool = True,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
    device: str | None = None,
    local_iterations: int = 10,
    max_levels: int = 10,
) -> "AnnData | None":
    """
    稀疏+GPU 的多层 Louvain/Leiden-style（不 densify）：
      每层做 _local_move（γ 在期望项），然后按社区收缩成超节点图，再继续。
    注：此实现未包含 Leiden 的“refinement”子步骤；如需严格 Leiden，可在每层 local move 后加入连通性细化。
    """
    start = logg.info("running GPU-sparse multilevel (local-move + contraction)")
    ad = adata.copy() if copy else adata
    torch.manual_seed(random_state)
    rs = np.random.RandomState(random_state)

    dev = _pick_device(device)
    logg.info(f"Using device: {dev}")

    # 取邻接并构造对称稀疏图
    if adjacency is None:
        adjacency = _utils._choose_graph(ad, obsp, neighbors_key)
    adjacency = _ensure_coo(adjacency)

    n0 = adjacency.shape[0]
    row = torch.as_tensor(adjacency.row, device=dev, dtype=torch.long)
    col = torch.as_tensor(adjacency.col, device=dev, dtype=torch.long)
    if use_weights and adjacency.data is not None:
        val = torch.as_tensor(adjacency.data, device=dev, dtype=torch.float32)
    else:
        val = torch.ones_like(row, dtype=torch.float32)

    # 对称化 + 合并
    Ai = torch.cat([row, col], dim=0)
    Aj = torch.cat([col, row], dim=0)
    Aw = torch.cat([val, val], dim=0)
    A = SparseTensor(row=Ai, col=Aj, value=Aw, sparse_sizes=(n0, n0)).coalesce()

    # 层级循环
    labels = None  # 原始节点到“当前层社区”的复合映射
    n_nodes = n0
    for level in range(max_levels):
        logg.info(f"level {level}: nodes={n_nodes}")
        comm_l = _local_move_gpu_sparse(
            A, resolution=resolution, n_iterations=local_iterations,
            random_state=random_state + level, log_prefix=f"  L{level}"
        )
        n_comms = int(comm_l.max().item()) + 1

        # 组合到原始节点标签：labels = comm_l[labels]（第一层时 labels = comm_l）
        if labels is None:
            labels = comm_l
        else:
            labels = comm_l[labels]

        # 若无法继续收缩（每点自成一社），提前结束
        if n_comms == n_nodes:
            logg.info(f"no further contraction (level {level}), stopping.")
            break

        # 收缩社区为超节点图，进入下一层
        A = _contract_graph_sparse(A, comm_l)
        n_nodes = n_comms

    # 连续化 & 回写
    labels, _ = _relabel_contiguous(labels)
    lab_np = labels.detach().to("cpu").numpy()
    cats = natsorted(map(str, np.unique(lab_np)))
    ad.obs[key_added] = pd.Categorical(values=lab_np.astype("U"), categories=cats)
    ad.uns[key_added] = {"params": dict(
        resolution=resolution,
        random_state=random_state,
        local_iterations=local_iterations,
        max_levels=max_levels,
        device=str(dev),
        impl="gpu_sparse_multilevel_no_refine",
    )}

    logg.info(
        "    finished",
        time=start,
        deep=f"found {len(cats)} clusters; added {key_added!r} (obs, categorical)",
    )
    return ad if copy else None
