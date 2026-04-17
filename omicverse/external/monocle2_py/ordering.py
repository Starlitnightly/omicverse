"""
Cell ordering and pseudotime assignment for monocle2_py.

Implements orderCells() — assigns pseudotime and state to each cell.
"""

import numpy as np
import pandas as pd
import igraph as ig
from scipy.spatial.distance import pdist, squareform


def _project_point_to_line_segment(p, A, B):
    """Project point p onto line segment [A, B]."""
    AB = B - A
    AB_sq = np.dot(AB, AB)
    if AB_sq == 0:
        return A.copy()
    Ap = p - A
    t = np.dot(Ap, AB) / AB_sq
    t = np.clip(t, 0, 1)
    return A + t * AB


def _proj_point_on_line(p, A, B):
    """Project point p onto infinite line through A and B."""
    AB = B - A
    AB_sq = np.dot(AB, AB)
    if AB_sq == 0:
        return A.copy()
    Ap = p - A
    t = np.dot(Ap, AB) / AB_sq
    return A + t * AB


def _project_cells_to_mst(adata):
    """
    Project each cell onto the nearest edge of the MST (like project2MST in R).
    Updates adata with projected coordinates and new MST.
    """
    monocle = adata.uns['monocle']
    Z = monocle['reducedDimS']  # (dim, N)
    Y = monocle['reducedDimK']  # (dim, K)
    mst = monocle['mst']
    closest_vertex = monocle['pr_graph_cell_proj_closest_vertex']

    N = Z.shape[1]
    dim = Z.shape[0]
    K = Y.shape[1]

    # Get MST edge list
    tip_leaves = [v.index for v in mst.vs if mst.degree(v) == 1]

    P = np.zeros((dim, N))

    for i in range(N):
        cv = closest_vertex[i]
        cv_name = mst.vs[cv]['name']
        neighbors = mst.neighbors(cv)

        best_proj = None
        best_dist = np.inf

        Z_i = Z[:, i]

        for neighbor in neighbors:
            A = Y[:, cv]
            B = Y[:, neighbor]

            if cv in tip_leaves:
                proj = _proj_point_on_line(Z_i, A, B)
            else:
                proj = _project_point_to_line_segment(Z_i, A, B)

            d = np.linalg.norm(Z_i - proj)
            if d < best_dist:
                best_dist = d
                best_proj = proj

        if best_proj is None:
            P[:, i] = Y[:, cv]
        else:
            P[:, i] = best_proj

    # Build new pairwise distance matrix on projected points
    dp = squareform(pdist(P.T))
    min_dist = dp[dp > 0].min() if np.any(dp > 0) else 1e-10
    dp = dp + min_dist
    np.fill_diagonal(dp, 0)

    # Build new MST on projected cells
    N_cells = dp.shape[0]
    g = ig.Graph.Full(N_cells)
    g.vs['name'] = list(adata.obs_names)
    weights = []
    for e in g.es:
        i, j = e.source, e.target
        weights.append(dp[i, j])
    g.es['weight'] = weights
    cell_mst = g.spanning_tree(weights=weights)

    monocle['pr_graph_cell_proj_tree'] = cell_mst
    monocle['pr_graph_cell_proj_dist'] = P
    monocle['projected_dp'] = dp

    return adata


def _extract_ddrtree_ordering(adata, root_cell_name, use_cell_mst=False):
    """
    Extract pseudotime ordering from MST via DFS traversal.

    Parameters
    ----------
    adata : AnnData
    root_cell_name : str
        Name of the root vertex in the MST.
    use_cell_mst : bool
        If True, use the cell projection MST instead of the Y-center MST.

    Returns
    -------
    pd.DataFrame with columns: sample_name, cell_state, pseudo_time, parent
    """
    monocle = adata.uns['monocle']

    if use_cell_mst:
        mst = monocle['pr_graph_cell_proj_tree']
        dp = monocle['projected_dp']
    else:
        mst = monocle['mst']
        dp = monocle['cellPairwiseDistances']

    n_vertices = mst.vcount()
    vertex_names = mst.vs['name']

    # Find root vertex index
    root_idx = vertex_names.index(root_cell_name)

    # BFS traversal: returns (order, layers, parents)
    dfs_result = mst.bfs(root_idx)
    order = dfs_result[0]
    fathers = dfs_result[2]  # parent of each vertex in BFS tree

    pseudotimes = np.zeros(n_vertices)
    states = np.ones(n_vertices, dtype=int)
    parents = [None] * n_vertices
    curr_state = 1

    for node_idx in order:
        if node_idx < 0:
            continue
        node_name = vertex_names[node_idx]
        father_idx = fathers[node_idx]

        if father_idx >= 0:
            parent_name = vertex_names[father_idx]
            parent_pseudotime = pseudotimes[father_idx]
            parent_state = states[father_idx]

            # Distance from parent
            if dp.shape[0] > max(node_idx, father_idx):
                d = dp[node_idx, father_idx]
            else:
                d = 0
            curr_node_pseudotime = parent_pseudotime + d

            # Check if parent is a branch point (degree > 2)
            if mst.degree(father_idx) > 2:
                curr_state += 1
        else:
            parent_name = None
            curr_node_pseudotime = 0

        pseudotimes[node_idx] = curr_node_pseudotime
        states[node_idx] = curr_state
        parents[node_idx] = parent_name

    ordering_df = pd.DataFrame({
        'sample_name': vertex_names,
        'cell_state': states,
        'pseudo_time': pseudotimes,
        'parent': parents,
    })
    ordering_df.index = vertex_names
    return ordering_df


def _select_root_cell(adata, root_state=None, reverse=False):
    """Select root cell for pseudotime ordering."""
    monocle = adata.uns['monocle']

    if root_state is not None:
        if 'State' not in adata.obs.columns:
            raise ValueError("State not set. Call order_cells() without root_state first.")

        root_candidates = adata.obs[adata.obs['State'] == root_state]
        if len(root_candidates) == 0:
            raise ValueError(f"No cells for State = {root_state}")

        mst = monocle['mst']
        tip_leaves = [mst.vs[v]['name'] for v in range(mst.vcount()) if mst.degree(v) == 1]

        if 'Pseudotime' in adata.obs.columns:
            root_cell = root_candidates['Pseudotime'].idxmin()
        else:
            root_cell = root_candidates.index[0]

        if monocle['dim_reduce_type'] == 'DDRTree':
            closest = monocle['pr_graph_cell_proj_closest_vertex']
            cell_idx = list(adata.obs_names).index(root_cell)
            graph_point = closest[cell_idx]
            root_cell = mst.vs[graph_point]['name']

        return root_cell
    else:
        mst = monocle['mst']
        # Use diameter endpoints
        diameter_path = mst.get_diameter(directed=False)
        if len(diameter_path) == 0:
            return mst.vs[0]['name']
        if reverse:
            return mst.vs[diameter_path[-1]]['name']
        else:
            return mst.vs[diameter_path[0]]['name']


def order_cells(adata, root_state=None, reverse=None):
    """
    Order cells along the learned trajectory and assign pseudotime.

    Parameters
    ----------
    adata : AnnData
    root_state : int or None
        If specified, use cells in this state as root.
    reverse : bool or None
        If True, reverse the ordering direction.

    Returns
    -------
    adata with Pseudotime and State in .obs
    """
    monocle = adata.uns['monocle']
    dim_reduce_type = monocle.get('dim_reduce_type')

    if dim_reduce_type is None:
        raise ValueError("Dimensionality not yet reduced. Call reduce_dimension() first.")

    if dim_reduce_type == 'DDRTree':
        Z = monocle['reducedDimS']
        Y = monocle['reducedDimK']
        K = Y.shape[1]
        mst = monocle['mst']

        # Initial ordering on Y-center MST
        root_cell = _select_root_cell(adata, root_state, reverse)
        cc_ordering = _extract_ddrtree_ordering(adata, root_cell, use_cell_mst=False)

        # Project cells onto MST edges
        _project_cells_to_mst(adata)

        cell_mst = monocle['pr_graph_cell_proj_tree']
        closest_vertex = monocle['pr_graph_cell_proj_closest_vertex']

        # Find root in cell MST
        old_root_idx = list(mst.vs['name']).index(root_cell)
        cells_at_root = np.where(closest_vertex == old_root_idx)[0]

        tip_leaves_cell = [v.index for v in cell_mst.vs if cell_mst.degree(v) == 1]
        tip_names = [cell_mst.vs[v]['name'] for v in tip_leaves_cell]

        root_cell_new = None
        for cell_idx in cells_at_root:
            cell_name = adata.obs_names[cell_idx]
            if cell_name in tip_names:
                root_cell_new = cell_name
                break

        if root_cell_new is None:
            root_cell_new = _select_root_cell(adata, root_state, reverse)

        monocle['root_cell'] = root_cell_new

        # Extract ordering on cell projection MST
        cc_ordering_new = _extract_ddrtree_ordering(adata, root_cell_new, use_cell_mst=True)

        # Assign pseudotime
        adata.obs['Pseudotime'] = cc_ordering_new.loc[adata.obs_names, 'pseudo_time'].values

        # Assign state based on Y-center MST structure
        if root_state is None:
            state_map = cc_ordering['cell_state'].values
            cell_states = state_map[closest_vertex]
            adata.obs['State'] = pd.Categorical(cell_states)
        else:
            adata.obs['State'] = pd.Categorical(
                cc_ordering_new.loc[adata.obs_names, 'cell_state'].values
            )

        # Find branch points
        branch_points = [mst.vs[v]['name'] for v in range(mst.vcount()) if mst.degree(v) > 2]
        monocle['branch_points'] = branch_points

    elif dim_reduce_type == 'ICA':
        mst = monocle['mst']
        root_cell = _select_root_cell(adata, root_state, reverse)
        monocle['root_cell'] = root_cell

        dp = monocle['cellPairwiseDistances']
        cc_ordering = _extract_ddrtree_ordering(adata, root_cell, use_cell_mst=False)

        adata.obs['Pseudotime'] = cc_ordering.loc[adata.obs_names, 'pseudo_time'].values
        adata.obs['State'] = pd.Categorical(
            cc_ordering.loc[adata.obs_names, 'cell_state'].values
        )

        branch_points = [mst.vs[v]['name'] for v in range(mst.vcount()) if mst.degree(v) > 2]
        monocle['branch_points'] = branch_points

    return adata
