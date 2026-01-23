# ---- Corrected Batched Parallel GPU Leiden ----
# Fixed the sigma double-subtraction bug

import torch
import numpy as np
import pandas as pd
from natsort import natsorted
from scanpy import _utils, logging as logg
from tqdm.auto import tqdm

try:
    from torch_sparse import SparseTensor, matmul
except ImportError as e:
    raise ImportError("Requires torch_sparse.") from e

try:
    from torch_scatter import scatter_add, scatter_max
    HAS_SCATTER = True
except Exception:
    HAS_SCATTER = False


def _pick_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_coo(adjacency):
    from scipy.sparse import coo_matrix
    return adjacency if isinstance(adjacency, coo_matrix) else adjacency.tocoo()


def _maybe_symmetrize(adjacency, symmetrize):
    if symmetrize is False:
        return adjacency
    if adjacency.shape[0] != adjacency.shape[1]:
        return adjacency
    is_sym = (adjacency != adjacency.T).nnz == 0
    if symmetrize is True or (symmetrize is None and not is_sym):
        return adjacency.maximum(adjacency.T)
    return adjacency


def _relabel_contiguous(comm):
    uniq, inv = torch.unique(comm, sorted=True, return_inverse=True)
    return inv, int(uniq.numel())


# ==================== Connected Components ====================

def _connected_components_gpu(n, edge_i, edge_j, device):
    if n == 0:
        return torch.tensor([], device=device, dtype=torch.long)
    if edge_i.numel() == 0:
        return torch.arange(n, device=device, dtype=torch.long)
    
    labels = torch.arange(n, device=device, dtype=torch.long)
    all_i = torch.cat([edge_i, edge_j])
    all_j = torch.cat([edge_j, edge_i])
    
    for _ in range(n):
        old = labels.clone()
        neigh_labels = labels[all_j]
        min_neigh = torch.full((n,), n, device=device, dtype=torch.long)
        min_neigh.scatter_reduce_(0, all_i, neigh_labels, reduce="amin")
        labels = torch.minimum(labels, min_neigh)
        labels = labels[labels]
        if torch.equal(labels, old):
            break
    return labels


def _refine_communities(A, comm):
    """Split disconnected communities."""
    n = A.sparse_size(0)
    device = comm.device
    Ai, Aj, Aw = A.coo()
    
    same = comm[Ai] == comm[Aj]
    comp = _connected_components_gpu(n, Ai[same], Aj[same], device)
    
    max_c = comp.max().item() + 1 if comp.numel() > 0 else 1
    composite = comm * max_c + comp
    _, comp_inv = torch.unique(composite, return_inverse=True)
    n_comp = comp_inv.max().item() + 1
    
    sizes = torch.zeros(n_comp, device=device, dtype=torch.long)
    sizes.scatter_add_(0, comp_inv, torch.ones(n, device=device, dtype=torch.long))
    
    comp_comm = torch.zeros(n_comp, device=device, dtype=torch.long)
    comp_comm[comp_inv] = comm
    
    n_comm = comm.max().item() + 1
    max_size = torch.zeros(n_comm, device=device, dtype=torch.long)
    max_size.scatter_reduce_(0, comp_comm, sizes, reduce="amax")
    
    is_small = sizes < max_size[comp_comm]
    node_small = is_small[comp_inv]
    
    if not node_small.any():
        return comm
    
    refined = comm.clone()
    
    out_mask = node_small[Ai] & (comm[Ai] != comm[Aj])
    if not out_mask.any():
        return refined
    
    out_comp = comp_inv[Ai[out_mask]]
    out_neigh = comm[Aj[out_mask]]
    out_w = Aw[out_mask]
    
    pair = out_comp * n_comm + out_neigh
    upair, pinv = torch.unique(pair, return_inverse=True)
    pw = torch.zeros(upair.numel(), device=device, dtype=out_w.dtype)
    pw.scatter_add_(0, pinv, out_w)
    
    pc = upair // n_comm
    pn = upair % n_comm
    
    small_ids = torch.where(is_small)[0]
    n_small = small_ids.numel()
    
    c2i = torch.full((n_comp,), -1, device=device, dtype=torch.long)
    c2i[small_ids] = torch.arange(n_small, device=device)
    
    pi = c2i[pc]
    valid = pi >= 0
    if not valid.any():
        return refined
    
    best_w = torch.full((n_small,), -1.0, device=device, dtype=pw.dtype)
    best_n = torch.full((n_small,), -1, device=device, dtype=torch.long)
    
    best_w.scatter_reduce_(0, pi[valid], pw[valid], reduce="amax")
    is_best = pw[valid] == best_w[pi[valid]]
    best_n.scatter_(0, pi[valid][is_best], pn[valid][is_best])
    
    new_comm = torch.full((n_comp,), -1, device=device, dtype=torch.long)
    has_best = best_n >= 0
    new_comm[small_ids[has_best]] = best_n[has_best]
    new_comm[small_ids[~has_best]] = comp_comm[small_ids[~has_best]]
    
    node_new = new_comm[comp_inv]
    refined[node_new >= 0] = node_new[node_new >= 0]
    
    return refined


# ==================== CORRECTED Batched Local Move ====================

def _compute_best_moves_for_batch(
    batch_nodes,    # Tensor of node indices to process
    A_ns,           # Adjacency without self-loops
    comm,           # Current community assignments
    deg,            # Node degrees
    sigma,          # ORIGINAL sigma (not modified)
    resolution,
    two_m,
    n_comm,
    device,
):
    """
    Compute best community for each node in batch.
    
    IMPORTANT: sigma should be the ORIGINAL sigma (all nodes included).
    The adjustment for "staying in current community" is done internally.
    
    Gain formula:
    - Moving to community c: gain = k_ic - γ * ki * Σc / 2m
    - Staying in community ci: gain = k_i,ci - γ * ki * (Σci - ki) / 2m
    
    The difference is that when staying, we need Σci - ki (exclude self).
    """
    n = comm.numel()
    batch_size = batch_nodes.numel()
    
    if batch_size == 0:
        return torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device)
    
    Ai, Aj, Aw = A_ns.coo()
    
    # Filter edges FROM batch nodes
    batch_set = torch.zeros(n, dtype=torch.bool, device=device)
    batch_set[batch_nodes] = True
    
    from_batch = batch_set[Ai]
    batch_Ai = Ai[from_batch]
    batch_Aj = Aj[from_batch]
    batch_Aw = Aw[from_batch]
    
    if batch_Ai.numel() == 0:
        # No edges - return current communities with zero gain
        return comm[batch_nodes].clone(), torch.zeros(batch_size, device=device)
    
    # Map global -> local
    global_to_local = torch.full((n,), -1, device=device, dtype=torch.long)
    global_to_local[batch_nodes] = torch.arange(batch_size, device=device)
    
    local_i = global_to_local[batch_Ai]
    target_comm = comm[batch_Aj]
    
    # Aggregate k_ic: edge weight from local node to each community
    pair_key = local_i * n_comm + target_comm
    unique_pairs, pair_inv = torch.unique(pair_key, return_inverse=True)
    
    k_ic = torch.zeros(unique_pairs.numel(), device=device, dtype=batch_Aw.dtype)
    k_ic.scatter_add_(0, pair_inv, batch_Aw)
    
    pair_local = unique_pairs // n_comm
    pair_comm = unique_pairs % n_comm
    
    # Get node properties
    ki = deg[batch_nodes[pair_local]]
    own_comm = comm[batch_nodes[pair_local]]
    sig_c = sigma[pair_comm]
    
    # KEY FIX: Correct sigma_effective calculation
    # - For other communities: sig_eff = sigma[c] (unchanged)
    # - For own community: sig_eff = sigma[ci] - ki (exclude self)
    is_own = pair_comm == own_comm
    sig_eff = sig_c - is_own.float() * ki  # Only subtract ki when c == own_comm
    
    gains = k_ic - (resolution * ki * sig_eff) / two_m
    
    # Ensure each node has its current community as option
    has_own = torch.zeros(batch_size, dtype=torch.bool, device=device)
    has_own[pair_local[is_own]] = True
    
    miss_local = torch.where(~has_own)[0]
    if miss_local.numel() > 0:
        miss_global = batch_nodes[miss_local]
        mc = comm[miss_global]
        mk = deg[miss_global]
        ms = sigma[mc] - mk  # sigma_eff for own community
        # k_ic = 0 for missing entries (no edges to own community)
        mg = 0 - (resolution * mk * ms) / two_m
        
        pair_local = torch.cat([pair_local, miss_local])
        pair_comm = torch.cat([pair_comm, mc])
        gains = torch.cat([gains, mg])
    
    # Find best community per node
    if HAS_SCATTER:
        best_gain, argmax = scatter_max(gains, pair_local, dim=0, dim_size=batch_size)
        best_comm = torch.where(argmax >= 0, pair_comm[argmax.clamp(0)], comm[batch_nodes])
    else:
        best_gain = torch.full((batch_size,), float('-inf'), device=device, dtype=gains.dtype)
        best_gain.scatter_reduce_(0, pair_local, gains, reduce="amax")
        is_best = gains == best_gain[pair_local]
        best_comm = comm[batch_nodes].clone()
        best_comm.scatter_(0, pair_local[is_best], pair_comm[is_best])
    
    return best_comm, best_gain


def _local_move_batched(A, resolution, n_iterations, random_state=0,
                        log_prefix="", two_m=None, initial_comm=None,
                        n_batches=10):
    """
    Batched parallel local move with CORRECT sigma handling.
    
    Key fix: Don't modify sigma globally. Each batch computes gains using
    the original sigma, with proper adjustment for "staying" case.
    After each batch, sigma is updated to reflect the actual moves.
    """
    torch.manual_seed(random_state)
    n = A.sparse_size(0)
    device = A.device()
    
    Ai, Aj, Aw = A.coo()
    
    # Remove self-loops
    ns = Ai != Aj
    A_ns = SparseTensor(row=Ai[ns], col=Aj[ns], value=Aw[ns], sparse_sizes=(n, n)).coalesce()
    
    deg = A.sum(dim=1).to_dense()
    
    if two_m is None:
        two_m = deg.sum().item()
    
    if initial_comm is not None:
        comm = initial_comm.clone()
    else:
        comm = torch.arange(n, device=device, dtype=torch.long)
    
    comm, n_comm = _relabel_contiguous(comm)
    
    rng = torch.Generator(device=device)
    rng.manual_seed(random_state)
    
    batch_size = max(1, (n + n_batches - 1) // n_batches)
    
    pbar = tqdm(range(n_iterations), desc=f"{log_prefix} Batched({n_batches})",
                leave=False, disable=not log_prefix)
    
    for _ in pbar:
        # Compute ORIGINAL sigma (all nodes included)
        sigma = torch.zeros(n_comm, device=device, dtype=deg.dtype)
        sigma.scatter_add_(0, comm, deg)
        
        # Random permutation
        order = torch.randperm(n, device=device, generator=rng)
        
        total_moved = 0
        
        # Process in batches
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_nodes = order[batch_start:batch_end]
            
            # Compute best moves using ORIGINAL sigma
            # (the function handles sigma_eff adjustment internally)
            best_comm, best_gain = _compute_best_moves_for_batch(
                batch_nodes, A_ns, comm, deg, sigma, resolution, two_m, n_comm, device
            )
            
            # Only move if gain > 0
            batch_old_comm = comm[batch_nodes]
            move_mask = best_gain > 1e-12
            new_comm = torch.where(move_mask, best_comm, batch_old_comm)
            
            # Find nodes that actually moved
            moved_mask = new_comm != batch_old_comm
            if moved_mask.any():
                moved_nodes = batch_nodes[moved_mask]
                old_c = batch_old_comm[moved_mask]
                new_c = new_comm[moved_mask]
                moved_ki = deg[moved_nodes]
                
                # Update sigma: remove from old, add to new
                sigma.scatter_add_(0, old_c, -moved_ki)
                sigma.scatter_add_(0, new_c, moved_ki)
                
                # Update comm
                comm[moved_nodes] = new_c
                
                total_moved += moved_mask.sum().item()
        
        # Relabel for next iteration
        comm, n_comm = _relabel_contiguous(comm)
        
        pbar.set_postfix({"moved": total_moved, "comms": n_comm})
        
        if total_moved == 0:
            break
    
    pbar.close()
    return comm


# ==================== Sequential (reference) ====================

def _local_move_sequential(A, resolution, n_iterations, random_state=0,
                           log_prefix="", two_m=None, initial_comm=None):
    """Original sequential - correct reference implementation."""
    torch.manual_seed(random_state)
    n = A.sparse_size(0)
    device = A.device()
    
    deg = A.sum(dim=1).to_dense()
    
    if two_m is None:
        two_m = deg.sum().item()
    
    if initial_comm is not None:
        comm = initial_comm.clone()
        comm, _ = _relabel_contiguous(comm)
    else:
        comm = torch.arange(n, device=device, dtype=torch.long)
    
    rowptr, col, val = A.csr()
    rowptr_list = rowptr.tolist()
    
    rng = torch.Generator(device=device)
    rng.manual_seed(random_state)
    
    pbar = tqdm(range(n_iterations), desc=f"{log_prefix} Seq", leave=False, disable=not log_prefix)
    
    for _ in pbar:
        comm, n_comm = _relabel_contiguous(comm)
        
        sigma = torch.zeros(n_comm, device=device, dtype=deg.dtype)
        sigma.scatter_add_(0, comm, deg)
        
        order = torch.randperm(n, device=device, generator=rng).tolist()
        moved = 0
        
        for i in order:
            ci = comm[i]
            ki = deg[i]
            
            # Temporarily remove node i from sigma
            sigma[ci] -= ki
            
            s, e = rowptr_list[i], rowptr_list[i+1]
            neigh = col[s:e]
            w = val[s:e]
            
            # Remove self-loops
            mask = neigh != i
            neigh = neigh[mask]
            w = w[mask]
            
            if neigh.numel() == 0:
                sigma[ci] += ki
                continue
            
            nc = comm[neigh]
            uc, inv = torch.unique(nc, return_inverse=True)
            
            k_ic = torch.zeros(uc.numel(), device=device, dtype=w.dtype)
            k_ic.scatter_add_(0, inv, w)
            
            # Ensure current community is option
            if not (uc == ci).any():
                uc = torch.cat([uc, ci.unsqueeze(0)])
                k_ic = torch.cat([k_ic, torch.zeros(1, device=device, dtype=w.dtype)])
            
            sig_c = sigma[uc]
            gains = k_ic - (resolution * ki * sig_c) / two_m
            
            best_idx = gains.argmax()
            best_gain = gains[best_idx]
            
            best_c = uc[best_idx] if best_gain > 1e-12 else ci
            
            # Add node back to (possibly new) community
            sigma[best_c] += ki
            
            if best_c != ci:
                comm[i] = best_c
                moved += 1
        
        pbar.set_postfix({"moved": moved, "comms": n_comm})
        if moved == 0:
            break
    
    pbar.close()
    comm, _ = _relabel_contiguous(comm)
    return comm


# ==================== Graph Contraction ====================

def _contract_graph(A, comm):
    Ai, Aj, Aw = A.coo()
    Ci, Cj = comm[Ai], comm[Aj]
    nc = comm.max().item() + 1
    return SparseTensor(row=Ci, col=Cj, value=Aw, sparse_sizes=(nc, nc)).coalesce()


# ==================== Main ====================

def leiden_gpu_sparse_multilevel(
    adata,
    resolution=1.0,
    *,
    random_state=0,
    key_added="leiden_pyg",
    adjacency=None,
    use_weights=True,
    neighbors_key=None,
    obsp=None,
    copy=False,
    device=None,
    local_iterations=10,
    max_levels=2,
    local_move_mode="batched",
    n_batches=None,
    symmetrize=None,
):
    """
    GPU Leiden clustering with corrected batched parallel optimization.
    """
    if n_batches is None:
        n_batches = max(10, int(np.sqrt(adata.n_obs)))
        print(f"Using batch size `n_batches` calculated from sqrt(n_obs): {n_batches}")
    start = logg.info(f"Running GPU Leiden ({local_move_mode})")
    ad = adata.copy() if copy else adata
    
    torch.manual_seed(random_state)
    dev = _pick_device(device)
    logg.info(f"Device: {dev}")
    
    if adjacency is None:
        adjacency = _utils._choose_graph(ad, obsp, neighbors_key)
    adjacency = _ensure_coo(adjacency)
    adjacency = _maybe_symmetrize(adjacency, symmetrize)
    
    n0 = adjacency.shape[0]
    
    row = torch.as_tensor(adjacency.row, device=dev, dtype=torch.long)
    col = torch.as_tensor(adjacency.col, device=dev, dtype=torch.long)
    val = torch.as_tensor(adjacency.data, device=dev, dtype=torch.float32) if use_weights else torch.ones_like(row, dtype=torch.float32)
    
    A = SparseTensor(row=row, col=col, value=val, sparse_sizes=(n0, n0)).coalesce()
    two_m = val.sum().item()
    
    if local_move_mode == "sequential":
        local_fn = lambda A, res, iters, rs, lp, tm, ic: _local_move_sequential(
            A, res, iters, rs, lp, tm, ic)
    else:
        local_fn = lambda A, res, iters, rs, lp, tm, ic: _local_move_batched(
            A, res, iters, rs, lp, tm, ic, n_batches=n_batches)
    
    labels = None
    n_nodes = n0
    
    pbar = tqdm(range(max_levels), desc="Leiden", position=0)
    
    for level in pbar:
        pbar.set_description(f"L{level} ({n_nodes})")
        
        comm = local_fn(A, resolution, local_iterations,
                       random_state + level, f"L{level}", two_m, None)
        
        comm = _refine_communities(A, comm)
        comm, nc = _relabel_contiguous(comm)
        
        pbar.set_postfix({"comms": nc})
        
        labels = comm if labels is None else comm[labels]
        
        if nc == n_nodes:
            break
        
        A = _contract_graph(A, comm)
        n_nodes = nc
    
    pbar.close()
    
    labels, _ = _relabel_contiguous(labels)
    lab_np = labels.cpu().numpy()
    cats = natsorted(map(str, np.unique(lab_np)))
    
    ad.obs[key_added] = pd.Categorical(lab_np.astype("U"), categories=cats)
    ad.uns[key_added] = {"params": dict(
        resolution=resolution,
        random_state=random_state,
        local_iterations=local_iterations,
        max_levels=max_levels,
        device=str(dev),
        mode=local_move_mode,
        n_batches=n_batches if local_move_mode == "batched" else None,
    )}
    
    logg.info("done", time=start, deep=f"{len(cats)} clusters")
    return ad if copy else None


# ==================== Comparison ====================

def compare_modes(adata, resolution=1.0, random_state=0, n_batches_list=[1, 5, 10, 20, 50, 100], **kwargs):
    """Compare sequential with different batch sizes."""
    import time
    
    results = {}
    
    # Sequential baseline
    ad_seq = adata.copy()
    t0 = time.time()
    leiden_gpu_sparse_multilevel(ad_seq, resolution=resolution, random_state=random_state,
                                 local_move_mode="sequential", key_added="leiden_seq", **kwargs)
    t_seq = time.time() - t0
    seq_labels = ad_seq.obs["leiden_seq"].astype(int).values
    n_seq = len(np.unique(seq_labels))
    
    print(f"Sequential: {n_seq} clusters, {t_seq:.2f}s (baseline)")
    results['sequential'] = {'n_clusters': n_seq, 'time': t_seq, 'ari': 1.0}
    
    try:
        from sklearn.metrics import adjusted_rand_score
        has_sklearn = True
    except ImportError:
        has_sklearn = False
    
    for nb in n_batches_list:
        ad_batch = adata.copy()
        t0 = time.time()
        leiden_gpu_sparse_multilevel(ad_batch, resolution=resolution, random_state=random_state,
                                     local_move_mode="batched", n_batches=nb,
                                     key_added=f"leiden_b{nb}", **kwargs)
        t_batch = time.time() - t0
        batch_labels = ad_batch.obs[f"leiden_b{nb}"].astype(int).values
        n_batch = len(np.unique(batch_labels))
        
        ari = adjusted_rand_score(seq_labels, batch_labels) if has_sklearn else None
        speedup = t_seq / t_batch if t_batch > 0 else float('inf')
        ari_str = f", ARI={ari:.4f}" if ari is not None else ""
        print(f"Batched(n={nb:3d}): {n_batch} clusters, {t_batch:.2f}s, speedup={speedup:.1f}x{ari_str}")
        
        results[f'batched_{nb}'] = {'n_clusters': n_batch, 'time': t_batch, 'ari': ari, 'speedup': speedup}
    
    return results