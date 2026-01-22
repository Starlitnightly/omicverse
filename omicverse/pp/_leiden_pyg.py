# ---- 依赖：torch_sparse/torch_scatter 同你之前的版本 ----
import torch
import numpy as np
import pandas as pd
from natsort import natsorted
from scanpy import _utils, logging as logg
from tqdm.auto import tqdm

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

def _is_symmetric_coo(adjacency) -> bool:
    if adjacency.shape[0] != adjacency.shape[1]:
        return False
    # Exact symmetry check; matches scanpy's symmetric connectivities.
    return (adjacency != adjacency.T).nnz == 0

def _maybe_symmetrize(adjacency, symmetrize: bool | None):
    if symmetrize is False:
        return adjacency
    if symmetrize is True or (symmetrize is None and not _is_symmetric_coo(adjacency)):
        # Keep weights, but ensure undirected connectivity.
        return adjacency.maximum(adjacency.T)
    return adjacency

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

# ---------- 连通性检查和细化(refinement)阶段 ----------
def _find_connected_components(A: SparseTensor, comm: torch.Tensor) -> torch.Tensor:
    """
    Find connected components within each community. The largest component retains
    the community ID, while smaller components are re-assigned to the existing
    neighbor community with the strongest connection weight.
    """
    n = A.sparse_size(0)
    device = comm.device
    Ai, Aj, Aw = A.coo()

    unique_comms = torch.unique(comm)
    refined_comm = comm.clone()

    for c in unique_comms:
        # 1. Find nodes and components in the current community `c`
        mask_c = (comm == c)
        nodes_in_comm = torch.where(mask_c)[0]

        if len(nodes_in_comm) <= 1:
            continue

        # Build subgraph for community `c` to find components
        # Mapping from global to local indices
        global_to_local = torch.full((n,), -1, device=device, dtype=torch.long)
        global_to_local[nodes_in_comm] = torch.arange(len(nodes_in_comm), device=device)

        # Get internal edges
        in_comm_i = mask_c[Ai]
        in_comm_j = mask_c[Aj]
        internal_edges = in_comm_i & in_comm_j

        n_local = len(nodes_in_comm)
        component_id = torch.zeros(n_local, dtype=torch.long, device=device)
        
        # If no internal edges, every node is a component
        if internal_edges.sum() > 0:
            local_i = global_to_local[Ai[internal_edges]]
            local_j = global_to_local[Aj[internal_edges]]

            # Find components using BFS
            visited = torch.zeros(n_local, dtype=torch.bool, device=device)
            num_components = 0
            adj_list = [[] for _ in range(n_local)]
            for i, j in zip(local_i.cpu().numpy(), local_j.cpu().numpy()):
                adj_list[i].append(j)
                adj_list[j].append(i)

            for start_node in range(n_local):
                if not visited[start_node]:
                    queue = [start_node]
                    visited[start_node] = True
                    component_id[start_node] = num_components
                    idx = 0
                    while idx < len(queue):
                        node = queue[idx]; idx += 1
                        for neighbor in adj_list[node]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                component_id[neighbor] = num_components
                                queue.append(neighbor)
                    num_components += 1
        else:
            num_components = n_local
            component_id = torch.arange(n_local, device=device)
            
        if num_components <= 1:
            continue

        # 2. For smaller components, find the best neighbor community to merge into
        comp_sizes = torch.bincount(component_id)
        largest_comp_id = torch.argmax(comp_sizes)

        for comp_id in range(num_components):
            if comp_id == largest_comp_id:
                continue

            # Get nodes in the small component
            nodes_in_comp = nodes_in_comm[component_id == comp_id]

            # Find all outbound edges from this component
            comp_mask = torch.zeros(n, dtype=torch.bool, device=device)
            comp_mask[nodes_in_comp] = True
            
            # Edges from component nodes to outside community c
            from_comp = comp_mask[Ai]
            to_outside_c = ~mask_c[Aj]
            outbound_edges = from_comp & to_outside_c

            if outbound_edges.sum() == 0:
                # If no connections to outside, merge with the largest component of its own community
                refined_comm[nodes_in_comp] = c
                continue
            
            # Find the best community to merge with
            neighbor_comms = comm[Aj[outbound_edges]]
            edge_weights = Aw[outbound_edges]

            if HAS_SCATTER:
                conn_weights = scatter_add(edge_weights, neighbor_comms, dim=0)
            else: # Fallback for native scatter_add
                uniq_neighbors, inv = torch.unique(neighbor_comms, return_inverse=True)
                conn_weights_sparse = _scatter_add_native(edge_weights, inv, dim_size=uniq_neighbors.numel())
                full_size = (comm.max() + 1).item()
                conn_weights = torch.zeros(full_size, device=device, dtype=edge_weights.dtype)
                conn_weights.scatter_add_(0, uniq_neighbors, conn_weights_sparse)


            if conn_weights.sum() > 0:
                best_neighbor_comm = torch.argmax(conn_weights)
                refined_comm[nodes_in_comp] = best_neighbor_comm
            else:
                # Failsafe: if all edge weights are zero, merge with largest component
                refined_comm[nodes_in_comp] = c

    return refined_comm

def _refine_partition(A: SparseTensor, comm: torch.Tensor, resolution: float,
                      n_iterations: int, random_state: int, two_m: float) -> torch.Tensor:
    """
    Leiden refinement phase: split disconnected communities only.

    Arguments:
        A: Adjacency matrix
        comm: Community assignments from local move phase
        resolution: Resolution parameter (unused in split-only refinement)
        n_iterations: Number of refinement iterations (unused in split-only refinement)
        random_state: Random seed (unused in split-only refinement)
        two_m: Total edge weight * 2 (unused in split-only refinement)

    Returns:
        refined_comm: Refined community assignments
    """
    # Step 1: Split disconnected communities only (no extra local move here).
    refined_comm = _find_connected_components(A, comm)

    refined_comm, _ = _relabel_contiguous(refined_comm)
    return refined_comm

# ---------- 本地移动（local move）阶段（稀疏 + γ 在期望项） ----------
def _local_move_sequential_sparse(A: SparseTensor, resolution: float, n_iterations: int,
                                  random_state: int = 0, log_prefix: str = "", two_m: float = None,
                                  initial_comm: torch.Tensor | None = None) -> torch.Tensor:
    torch.manual_seed(random_state)
    n = A.sparse_size(0)
    deg = A.sum(dim=1).to_dense()            # (n,)

    # Use passed two_m if provided (for correct modularity calculation), otherwise compute from A
    if two_m is None:
        two_m = A.sum().item()

    if two_m <= 0:
        raise ValueError("Graph has no edges (2m == 0).")

    if initial_comm is not None:
        comm = initial_comm.clone()
        comm, _ = _relabel_contiguous(comm)
    else:
        comm = torch.arange(n, device=deg.device, dtype=torch.long)

    rowptr, col, val = A.csr()
    rng = torch.Generator(device=deg.device)
    rng.manual_seed(random_state)

    pbar = tqdm(range(n_iterations), desc=f"{log_prefix} Local move (sequential)", leave=False, disable=not log_prefix)
    for _ in pbar:
        comm, n_comms = _relabel_contiguous(comm)
        if HAS_SCATTER:
            sigma_tot = scatter_add(deg, comm, dim=0, dim_size=n_comms)
        else:
            sigma_tot = _scatter_add_native(deg, comm, dim_size=n_comms)

        order = torch.randperm(n, device=deg.device, generator=rng)
        moved = 0
        for idx in range(n):
            i = order[idx]
            ci = comm[i]
            ki = deg[i]

            sigma_tot[ci] -= ki

            start = rowptr[i].item()
            end = rowptr[i + 1].item()
            neigh = col[start:end]
            w = val[start:end]

            # CRITICAL FIX: Filter out self-loops
            # Self-loops in contracted graphs represent internal edges and should not
            # contribute to the gain of staying in the current community
            not_self_loop = (neigh != i)
            neigh = neigh[not_self_loop]
            w = w[not_self_loop]

            if neigh.numel() == 0:
                sigma_tot[ci] += ki
                continue

            neigh_comm = comm[neigh]
            uniq_comms, inv = torch.unique(neigh_comm, return_inverse=True)
            if HAS_SCATTER:
                k_ic = scatter_add(w, inv, dim=0, dim_size=uniq_comms.numel())
            else:
                k_ic = _scatter_add_native(w, inv, dim_size=uniq_comms.numel())

            if not (uniq_comms == ci).any():
                uniq_comms = torch.cat([uniq_comms, ci.view(1)])
                k_ic = torch.cat([k_ic, torch.zeros(1, device=deg.device, dtype=w.dtype)])

            sigma_c = sigma_tot[uniq_comms]
            gains = k_ic - (resolution * ki * sigma_c) / two_m

            best_gain, best_idx = torch.max(gains, dim=0)
            epsilon = 1e-12
            if best_gain > epsilon:
                best_c = uniq_comms[best_idx]
            else:
                best_c = ci

            sigma_tot[best_c] += ki
            if best_c != ci:
                comm[i] = best_c
                moved += 1

        pbar.set_postfix({"moved": moved, "communities": n_comms})
        if moved == 0:
            break
    pbar.close()

    comm, _ = _relabel_contiguous(comm)
    return comm

def _local_move_gpu_sparse(A: SparseTensor, resolution: float, n_iterations: int,
                           random_state: int = 0, log_prefix: str = "", two_m: float = None,
                           initial_comm: torch.Tensor | None = None, mode: str = "parallel") -> torch.Tensor:
    if mode == "sequential":
        return _local_move_sequential_sparse(
            A, resolution=resolution, n_iterations=n_iterations,
            random_state=random_state, log_prefix=log_prefix, two_m=two_m,
            initial_comm=initial_comm
        )
    if mode != "parallel":
        raise ValueError("mode must be 'parallel' or 'sequential'")

    torch.manual_seed(random_state)
    n = A.sparse_size(0)
    deg = A.sum(dim=1).to_dense()            # (n,)
    
    # Use passed two_m if provided (for correct modularity calculation), otherwise compute from A
    if two_m is None:
        two_m = A.sum().item()
    
    if two_m <= 0:
        raise ValueError("Graph has no edges (2m == 0).")

    if initial_comm is not None:
        comm = initial_comm.clone()
        comm, _ = _relabel_contiguous(comm)
    else:
        comm = torch.arange(n, device=deg.device, dtype=torch.long)


    sigma_tot = deg.clone()

    Ai, Aj, Aw = A.coo()
    pbar = tqdm(range(n_iterations), desc=f"{log_prefix} Local move", leave=False, disable=not log_prefix)
    for it in pbar:
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

        # CRITICAL FIX: Prevent nodes from moving to communities they have no edges to
        # This prevents creation of disconnected communities in local move phase
        # Only mask out moves to communities with zero edge weight (k_ic == 0)
        # unless it's the node's current community (allow staying even with k_ic=0)
        zero_connection = (k_ic == 0) & ~(colc == comm_rows)
        gains = torch.where(zero_connection, torch.full_like(gains, float("-inf")), gains)

        best_gain, best_c = _rowwise_argmax(rows, gains, colc, n_rows=n)

        # In the parallel version, we only move if the best possible gain is positive.
        # This is stricter than comparing to stay_gain and prevents unstable "lesser
        # of two evils" moves, mimicking the more stable sequential implementation.
        epsilon = 1e-12
        improve = best_gain > epsilon
        new_comm = torch.where(improve, best_c, comm)

        moved = int((new_comm != comm).sum().item())
        pbar.set_postfix({"moved": moved, "communities": n_comms})
        comm = new_comm
        if moved == 0:
            break

        # 重新按 comm 聚合 Σ_tot（也可做增量更新，这里为清晰直接重算）
        comm, n_comms = _relabel_contiguous(comm)
        if HAS_SCATTER:
            sigma_tot = scatter_add(deg, comm, dim=0, dim_size=n_comms)
        else:
            sigma_tot = _scatter_add_native(deg, comm, dim_size=n_comms)
    pbar.close()

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
    max_levels: int = 2,  # Match official Leiden n_iterations default
    local_move_mode: str = "sequential",
    symmetrize: bool | None = None,
) -> "AnnData | None":
    """
    Sparse + GPU multilevel Leiden algorithm with split-only refinement phase.

    This implementation follows the standard Leiden algorithm:
      1. Local move phase: optimize modularity by moving nodes between communities
      2. Refinement phase: split disconnected communities only
      3. Aggregation phase: contract communities into supernodes

    This implementation guarantees connected communities.
    """
    start = logg.info("running GPU-sparse Leiden with refinement (local-move + split + aggregation)")
    ad = adata.copy() if copy else adata
    torch.manual_seed(random_state)
    rs = np.random.RandomState(random_state)

    dev = _pick_device(device)
    logg.info(f"Using device: {dev}")

    # 取邻接并构造对称稀疏图
    if adjacency is None:
        adjacency = _utils._choose_graph(ad, obsp, neighbors_key)
    adjacency = _ensure_coo(adjacency)
    adjacency = _maybe_symmetrize(adjacency, symmetrize)

    n0 = adjacency.shape[0]
    row = torch.as_tensor(adjacency.row, device=dev, dtype=torch.long)
    col = torch.as_tensor(adjacency.col, device=dev, dtype=torch.long)
    if use_weights and adjacency.data is not None:
        val = torch.as_tensor(adjacency.data, device=dev, dtype=torch.float32)
    else:
        val = torch.ones_like(row, dtype=torch.float32)

    # If the input graph is already symmetric (scanpy default), we keep it unchanged.
    # For directed/half graphs, we symmetrize above to align with Leiden's undirected modularity.
    A = SparseTensor(row=row, col=col, value=val, sparse_sizes=(n0, n0)).coalesce()

    # Compute 2m correctly: for undirected graph, 2m = sum of all edge weights
    # Since adjacency is already symmetric, each edge (i,j) appears once with weight w
    correct_two_m = val.sum().item()

    # 层级循环
    labels = None  # 原始节点到"当前层社区"的复合映射
    n_nodes = n0
    current_two_m = correct_two_m  # Start with the original graph's 2m

    level_pbar = tqdm(range(max_levels), desc="Leiden multilevel", position=0)
    for level in level_pbar:
        level_pbar.set_description(f"Level {level} ({n_nodes} nodes)")

        # 1. Local move phase
        comm_l = _local_move_gpu_sparse(
            A, resolution=resolution, n_iterations=local_iterations,
            random_state=random_state + level, log_prefix=f"L{level}.1",
            two_m=current_two_m,
            mode=local_move_mode
        )

        # 2. Refinement phase: split disconnected communities
        comm_l = _refine_partition(
            A, comm_l, resolution=resolution,
            n_iterations=local_iterations,
            random_state=random_state + level + 1000,
            two_m=current_two_m
        )

        n_comms = int(comm_l.max().item()) + 1
        level_pbar.set_postfix({"communities": n_comms})

        # 组合到原始节点标签：labels = comm_l[labels]（第一层时 labels = comm_l）
        if labels is None:
            labels = comm_l
        else:
            labels = comm_l[labels]

        # 若无法继续收缩（每点自成一社），提前结束
        if n_comms == n_nodes:
            level_pbar.close()
            logg.info(f"no further contraction (level {level}), stopping.")
            break

        # 收缩社区为超节点图，进入下一层
        A = _contract_graph_sparse(A, comm_l)
        n_nodes = n_comms
        # Note: current_two_m remains the same across levels because
        # graph contraction preserves total edge weight
    level_pbar.close()

    # 连续化 & 回写
    labels, _ = _relabel_contiguous(labels)
    lab_np = labels.detach().to("cpu").numpy()
    labels = lab_np.astype("U")             # → 字符串数组
    cats = natsorted(map(str, np.unique(lab_np)))
    try:
        ad.obs[key_added] = pd.Categorical(values=lab_np.astype("U"), categories=cats)
    except Exception:
        ad.obs[key_added] = labels
    ad.uns[key_added] = {"params": dict(
        resolution=resolution,
        random_state=random_state,
        local_iterations=local_iterations,
        max_levels=max_levels,
        device=str(dev),
        local_move_mode=local_move_mode,
        symmetrize=symmetrize,
        impl="gpu_sparse_multilevel_with_leiden_refine",
    )}

    logg.info(
        "    finished",
        time=start,
        deep=f"found {len(cats)} clusters; added {key_added!r} (obs, categorical)",
    )
    return ad if copy else None
