from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd
from natsort import natsorted

from scanpy import _utils
from scanpy import logging as logg

try:
    import torch
    import torch_geometric as pyg
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
    
    # Use torch native operations instead of torch_scatter
    def scatter_add(src, index, dim=0, dim_size=None):
        """Native PyTorch implementation of scatter_add."""
        if dim_size is None:
            dim_size = index.max().item() + 1
        
        out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        out = out.scatter_add(dim, index, src)
        return out
        
except ImportError:
    HAS_TORCH_GEOMETRIC = False

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from anndata import AnnData
    from .._compat import CSBase
    from .._utils.random import _LegacyRandom


def _check_pyg_installation():
    """Check if PyTorch Geometric is available."""
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "PyTorch Geometric is required for GPU-accelerated Leiden clustering. "
            "Please install with: pip install torch-geometric"
        )


def _modularity_matrix_gpu(edge_index, edge_weight, num_nodes, device):
    """Compute modularity matrix on GPU using PyTorch operations."""
    # Create adjacency matrix on GPU
    adj = torch.sparse_coo_tensor(
        edge_index, edge_weight, (num_nodes, num_nodes), device=device
    ).coalesce()
    
    # Compute degrees
    degrees = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=num_nodes)
    total_weight = edge_weight.sum()
    
    # Modularity matrix B = A - k_i * k_j / 2m
    expected = torch.outer(degrees, degrees) / (2 * total_weight)
    modularity_matrix = adj.to_dense() - expected
    
    return modularity_matrix, degrees, total_weight


def _leiden_local_move_gpu(modularity_matrix, communities, degrees, total_weight, 
                          resolution, device):
    """Perform local moves in Leiden algorithm on GPU."""
    num_nodes = modularity_matrix.shape[0]
    improved = True
    
    while improved:
        improved = False
        node_order = torch.randperm(num_nodes, device=device)
        
        for node in node_order:
            current_comm = communities[node].item()
            best_comm = current_comm
            best_delta = 0.0
            
            # Get neighbors
            neighbors = torch.nonzero(modularity_matrix[node] != 0).flatten()
            
            # Consider neighboring communities
            neighbor_comms = torch.unique(communities[neighbors])
            
            for comm in neighbor_comms:
                if comm == current_comm:
                    continue
                
                # Calculate modularity change
                delta = _calculate_modularity_delta_gpu(
                    node, current_comm, comm, modularity_matrix, 
                    communities, resolution, total_weight
                )
                
                if delta > best_delta:
                    best_delta = delta
                    best_comm = comm.item()
            
            if best_comm != current_comm:
                communities[node] = best_comm
                improved = True
    
    return communities


def _calculate_modularity_delta_gpu(node, old_comm, new_comm, modularity_matrix,
                                   communities, resolution, total_weight):
    """Calculate modularity change when moving node from old_comm to new_comm."""
    # Edges from node to nodes in new community
    new_comm_mask = (communities == new_comm)
    k_i_in_new = modularity_matrix[node][new_comm_mask].sum()
    
    # Edges from node to nodes in old community  
    old_comm_mask = (communities == old_comm) & (torch.arange(len(communities)) != node)
    k_i_in_old = modularity_matrix[node][old_comm_mask].sum()
    
    delta = 2 * resolution * (k_i_in_new - k_i_in_old) / total_weight
    
    return delta


def _refine_partition_gpu(modularity_matrix, communities, degrees, total_weight,
                         resolution, device):
    """Refine partition using subset optimization."""
    unique_comms = torch.unique(communities)
    
    for comm in unique_comms:
        comm_nodes = torch.nonzero(communities == comm).flatten()
        
        if len(comm_nodes) <= 1:
            continue
            
        # Extract subgraph for this community
        subgraph_mod = modularity_matrix[comm_nodes][:, comm_nodes]
        
        # Try to split this community
        sub_communities = _leiden_local_move_gpu(
            subgraph_mod, torch.zeros(len(comm_nodes), dtype=torch.long, device=device),
            degrees[comm_nodes], total_weight, resolution, device
        )
        
        # Update global community assignments
        max_comm = communities.max().item()
        for i, sub_comm in enumerate(sub_communities):
            if sub_comm > 0:
                communities[comm_nodes[i]] = max_comm + sub_comm
    
    return communities


def leiden_pyg(
    adata: AnnData,
    resolution: float = 1,
    *,
    random_state: _LegacyRandom = 0,
    key_added: str = "leiden_pyg",
    adjacency=None,
    use_weights: bool = True,
    n_iterations: int = 10,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
    device: str | None = None,
) -> AnnData | None:
    """GPU-accelerated Leiden clustering using PyTorch Geometric.
    
    This implementation uses PyTorch Geometric for GPU acceleration of the
    Leiden community detection algorithm.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    resolution
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
    random_state
        Change the initialization of the optimization.
    key_added
        `adata.obs` key under which to add the cluster labels.
    adjacency
        Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
    use_weights
        If `True`, edge weights from the graph are used in the computation.
    n_iterations
        Number of iterations to perform. Default is 10.
    neighbors_key
        Use neighbors connectivities as adjacency.
    obsp
        Use .obsp[obsp] as adjacency.
    copy
        Whether to copy `adata` or modify it inplace.
    device
        PyTorch device to use. If None, uses CUDA if available, else CPU.
        
    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object.
    Sets `adata.obs[key_added]` with cluster labels.
    """
    _check_pyg_installation()
    
    start = logg.info("running GPU-accelerated Leiden clustering")
    adata = adata.copy() if copy else adata
    
    # Set random seed
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Get device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    logg.info(f"Using device: {device}")
    
    # Get adjacency matrix
    if adjacency is None:
        adjacency = _utils._choose_graph(adata, obsp, neighbors_key)
    
    # Convert to PyTorch tensors
    from scipy.sparse import coo_matrix
    if not isinstance(adjacency, coo_matrix):
        adjacency = adjacency.tocoo()
    
    edge_index = torch.tensor(
        np.vstack([adjacency.row, adjacency.col]), 
        dtype=torch.long, device=device
    )
    
    if use_weights:
        edge_weight = torch.tensor(adjacency.data, dtype=torch.float, device=device)
    else:
        edge_weight = torch.ones(len(adjacency.data), dtype=torch.float, device=device)
    
    num_nodes = adjacency.shape[0]
    
    # Compute modularity matrix
    modularity_matrix, degrees, total_weight = _modularity_matrix_gpu(
        edge_index, edge_weight, num_nodes, device
    )
    
    # Initialize communities (each node in its own community)
    communities = torch.arange(num_nodes, dtype=torch.long, device=device)
    
    # Main Leiden iterations
    for iteration in range(n_iterations):
        logg.info(f"Leiden iteration {iteration + 1}/{n_iterations}")
        
        old_communities = communities.clone()
        
        # Local moving phase
        communities = _leiden_local_move_gpu(
            modularity_matrix, communities, degrees, total_weight, 
            resolution, device
        )
        
        # Refinement phase
        communities = _refine_partition_gpu(
            modularity_matrix, communities, degrees, total_weight,
            resolution, device
        )
        
        # Check for convergence
        if torch.equal(communities, old_communities):
            logg.info(f"Converged after {iteration + 1} iterations")
            break
    
    # Convert back to CPU and get final communities
    communities_cpu = communities.cpu().numpy()
    
    # Relabel communities to be contiguous starting from 0
    unique_comms = np.unique(communities_cpu)
    comm_mapping = {old: new for new, old in enumerate(unique_comms)}
    final_communities = np.array([comm_mapping[c] for c in communities_cpu])
    
    # Store results
    adata.obs[key_added] = pd.Categorical(
        values=final_communities.astype("U"),
        categories=natsorted(map(str, np.unique(final_communities))),
    )
    
    # Store parameters
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(
        resolution=resolution,
        random_state=random_state,
        n_iterations=n_iterations,
        device=str(device),
    )
    
    logg.info(
        "    finished",
        time=start,
        deep=(
            f"found {len(np.unique(final_communities))} clusters and added\n"
            f"    {key_added!r}, the cluster labels (adata.obs, categorical)"
        ),
    )
    
    return adata if copy else None