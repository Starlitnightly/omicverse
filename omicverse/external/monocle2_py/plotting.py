"""
Visualization functions for monocle2_py.

Faithfully reproduces Monocle2's ggplot2-based plots using matplotlib.
Matches Monocle2's visual style: white background, top legend, clean theme.
"""

import colorsys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy import sparse


# ============================================================================
# Monocle2 theme
# ============================================================================

def _monocle_theme(ax):
    """Apply Monocle2 theme to matplotlib axes."""
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(width=0.5, length=3)
    ax.grid(False)


def _resolve_gene_indices(adata, genes):
    """Return a list of integer var-indices for `genes`, one per name.

    Accepts either Ensembl IDs (matching ``adata.var_names``) or gene
    short names (matching ``adata.var['gene_short_name']``).  Missing
    genes yield ``None`` entries rather than raising.

    O(G + k) instead of O(G·k) — caches the lookup dictionary once per
    call so long gene lists (heatmaps, pseudotime heatmaps) don't pay
    the per-gene `list(var_names).index()` scan cost.
    """
    name_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    short_to_idx = {}
    if 'gene_short_name' in adata.var.columns:
        for i, s in enumerate(adata.var['gene_short_name'].values):
            # first-occurrence wins, to match the `.index(...)` semantics
            short_to_idx.setdefault(s, i)

    out = []
    for g in genes:
        if g in name_to_idx:
            out.append(name_to_idx[g])
        elif g in short_to_idx:
            out.append(short_to_idx[g])
        else:
            out.append(None)
    return out


def _get_state_colors(states, cmap_name=None):
    """Get color mapping for categorical states, matching ggplot2 hue palette."""
    unique_states = sorted(set(states))
    n = len(unique_states)
    # Reproduce ggplot2's default hue_pal() with h=(15,375), c=100, l=65
    color_map = {}
    for i, s in enumerate(unique_states):
        hue = (15 + (360 / n) * i) % 360
        # HCL to RGB approximation matching ggplot2 defaults.
        # ggplot2 uses HCL(h, c=100, l=65); we approximate with HSL:
        # lightness 65 → 0.58, saturation → 0.65.
        h_norm = hue / 360.0
        s_val = 0.65
        l_val = 0.58
        rgb = colorsys.hls_to_rgb(h_norm, l_val, s_val)
        color_map[s] = rgb
    return color_map


def _rotation_matrix(theta_degrees):
    """Return 2D rotation matrix for given angle in degrees."""
    theta = np.radians(theta_degrees)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


# ============================================================================
# plot_cell_trajectory
# ============================================================================

def plot_cell_trajectory(adata, x=0, y=1, color_by='State',
                         show_tree=True, show_backbone=True,
                         backbone_color='black',
                         markers=None, use_color_gradient=False,
                         show_cell_names=False, show_state_number=False,
                         cell_size=1.5, cell_link_size=0.75,
                         show_branch_points=True, theta=0,
                         figsize=(8, 6), ax=None, cmap=None,
                         save=None, dpi=150):
    """
    Plot cell trajectory in reduced dimension space.

    Parameters
    ----------
    adata : AnnData
    x, y : int
        Component indices to plot.
    color_by : str
        Column in adata.obs to color by.
    show_tree : bool
        Show MST edges.
    show_backbone : bool
    backbone_color : str
    markers : list of str or None
        Gene names to show expression overlay.
    use_color_gradient : bool
    show_cell_names : bool
    show_state_number : bool
    cell_size : float
    cell_link_size : float
    show_branch_points : bool
    theta : float
        Rotation angle in degrees.
    figsize : tuple
    ax : matplotlib.axes.Axes or None
    cmap : str or None
    save : str or None
    dpi : int

    Returns
    -------
    fig, ax
    """
    monocle = adata.uns['monocle']
    dim_type = monocle.get('dim_reduce_type', 'DDRTree')

    # Get cell coordinates (S = cell projections in reduced space)
    Z = monocle['reducedDimS']  # (dim, N)
    cell_coords = Z.T  # (N, dim)

    # Get tree node coordinates
    if dim_type == 'DDRTree':
        Y = monocle['reducedDimK']  # (dim, K)
        tree_coords = Y.T  # (K, dim)
    else:
        tree_coords = cell_coords

    # Apply rotation
    if theta != 0:
        rot = _rotation_matrix(theta)
        cell_coords = cell_coords[:, [x, y]] @ rot.T
        tree_coords = tree_coords[:, [x, y]] @ rot.T
        x_plot, y_plot = 0, 1
    else:
        x_plot, y_plot = x, y

    # Get MST
    mst = monocle['mst']

    if markers is not None and len(markers) > 0:
        n_markers = len(markers)
        if ax is None:
            fig, axes = plt.subplots(1, n_markers, figsize=(figsize[0] * n_markers, figsize[1]))
            if n_markers == 1:
                axes = [axes]
        else:
            axes = [ax]
            fig = ax.figure

        for idx, marker in enumerate(markers):
            ax_cur = axes[idx] if idx < len(axes) else axes[-1]

            # Get expression
            if marker in adata.var_names:
                gene_idx = list(adata.var_names).index(marker)
            elif 'gene_short_name' in adata.var.columns:
                matches = adata.var[adata.var['gene_short_name'] == marker].index
                if len(matches) > 0:
                    gene_idx = list(adata.var_names).index(matches[0])
                else:
                    continue
            else:
                continue

            expr = adata.X[:, gene_idx]
            if sparse.issparse(expr):
                expr = expr.toarray().flatten()
            else:
                expr = np.asarray(expr).flatten()

            # Draw tree
            if show_tree:
                for e in mst.es:
                    i, j = e.source, e.target
                    ax_cur.plot(
                        [tree_coords[i, x_plot], tree_coords[j, x_plot]],
                        [tree_coords[i, y_plot], tree_coords[j, y_plot]],
                        color='black', linewidth=cell_link_size, zorder=1
                    )

            if use_color_gradient:
                sc = ax_cur.scatter(
                    cell_coords[:, x_plot], cell_coords[:, y_plot],
                    c=np.log10(expr + 0.1), s=cell_size ** 2,
                    cmap=cmap or 'viridis', zorder=2
                )
                plt.colorbar(sc, ax=ax_cur, label='log10(expr + 0.1)')
            else:
                sc = ax_cur.scatter(
                    cell_coords[:, x_plot], cell_coords[:, y_plot],
                    c=np.log10(expr + 0.1), s=cell_size ** 2,
                    cmap=cmap or 'viridis', zorder=2
                )

            ax_cur.set_title(marker)
            ax_cur.set_xlabel(f'Component {x + 1}')
            ax_cur.set_ylabel(f'Component {y + 1}')
            _monocle_theme(ax_cur)

    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        # Draw tree edges
        if show_tree:
            for e in mst.es:
                i, j = e.source, e.target
                ax.plot(
                    [tree_coords[i, x_plot], tree_coords[j, x_plot]],
                    [tree_coords[i, y_plot], tree_coords[j, y_plot]],
                    color='black', linewidth=cell_link_size, zorder=1
                )

        # Color by
        if color_by in adata.obs.columns:
            color_vals = adata.obs[color_by].values
            if hasattr(color_vals, 'categories') or not np.issubdtype(
                np.array(color_vals).dtype, np.floating
            ):
                # Categorical
                color_map = _get_state_colors(color_vals)
                colors = [color_map[v] for v in color_vals]
                ax.scatter(cell_coords[:, x_plot], cell_coords[:, y_plot],
                           c=colors, s=cell_size ** 2, zorder=2, edgecolors='none')

                # Legend
                handles = [Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color_map[s], markersize=6,
                                  label=str(s))
                           for s in sorted(color_map.keys())]
                ax.legend(handles=handles, loc='upper center',
                          bbox_to_anchor=(0.5, 1.15), ncol=min(len(handles), 6),
                          frameon=False)
            else:
                # Continuous
                sc = ax.scatter(cell_coords[:, x_plot], cell_coords[:, y_plot],
                                c=color_vals.astype(float), s=cell_size ** 2,
                                cmap=cmap or 'Blues', zorder=2, edgecolors='none')
                plt.colorbar(sc, ax=ax, label=color_by)
        else:
            ax.scatter(cell_coords[:, x_plot], cell_coords[:, y_plot],
                       s=cell_size ** 2, zorder=2, edgecolors='none')

        # Branch points
        if show_branch_points and dim_type == 'DDRTree':
            branch_points = monocle.get('branch_points', [])
            mst_names = mst.vs['name']
            for bp_idx, bp_name in enumerate(branch_points):
                if bp_name in mst_names:
                    v_idx = mst_names.index(bp_name)
                    ax.scatter(tree_coords[v_idx, x_plot], tree_coords[v_idx, y_plot],
                               s=150, c='black', zorder=5, edgecolors='white', linewidths=1.5)
                    ax.text(tree_coords[v_idx, x_plot], tree_coords[v_idx, y_plot],
                            str(bp_idx + 1), ha='center', va='center',
                            color='white', fontsize=8, fontweight='bold', zorder=6)

        ax.set_xlabel(f'Component {x + 1}')
        ax.set_ylabel(f'Component {y + 1}')
        _monocle_theme(ax)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')

    return fig, ax


# ============================================================================
# plot_genes_in_pseudotime
# ============================================================================

def plot_genes_in_pseudotime(adata, genes, min_expr=None,
                              cell_size=0.75, ncol=1, nrow=None,
                              color_by='State', trend_formula="~sm.ns(Pseudotime, df=3)",
                              label_by_short_name=True, relative_expr=True,
                              figsize=None, save=None, dpi=150):
    """
    Plot gene expression vs pseudotime with smoothed curves.

    Parameters
    ----------
    adata : AnnData
    genes : list of str
        Gene names to plot.
    min_expr : float or None
    cell_size : float
    ncol : int
    nrow : int or None
    color_by : str
    trend_formula : str
    label_by_short_name : bool
    relative_expr : bool
    figsize : tuple or None
    save : str or None
    dpi : int

    Returns
    -------
    fig
    """
    from .differential import gen_smooth_curves

    n_genes = len(genes)
    if nrow is None:
        nrow = int(np.ceil(n_genes / ncol))

    if figsize is None:
        figsize = (5 * ncol, 3.5 * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    pseudotime = adata.obs['Pseudotime'].values
    sort_idx = np.argsort(pseudotime)

    # Get size factors
    if 'Size_Factor' in adata.obs.columns and relative_expr:
        sf = adata.obs['Size_Factor'].values
    else:
        sf = np.ones(adata.n_obs)

    # Color mapping
    if color_by in adata.obs.columns:
        color_vals = adata.obs[color_by].values
        color_map = _get_state_colors(color_vals)
        colors = [color_map[v] for v in color_vals]
    else:
        colors = ['steelblue'] * adata.n_obs

    # Generate smooth curves
    new_data = pd.DataFrame({
        'Pseudotime': pseudotime
    })
    smooth_curves = gen_smooth_curves(adata, new_data=new_data,
                                       trend_formula=trend_formula,
                                       relative_expr=relative_expr)

    # Pre-resolve all gene indices once (O(G+k)) instead of the previous
    # per-gene O(G) linear scan. Saves real time when plotting >20 genes.
    gene_indices = _resolve_gene_indices(adata, genes)

    for idx, gene in enumerate(genes):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]

        gene_idx = gene_indices[idx]
        if gene_idx is None:
            ax.set_title(f'{gene} (not found)')
            _monocle_theme(ax)
            continue

        expr = adata.X[:, gene_idx]
        if sparse.issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()

        if relative_expr:
            expr = expr / sf

        if min_expr is not None:
            expr = np.maximum(expr, min_expr)

        # Scatter plot
        ax.scatter(pseudotime, expr, c=colors, s=cell_size ** 2,
                   alpha=0.6, edgecolors='none', zorder=2)

        # Smooth curve
        curve = smooth_curves[gene_idx, :]
        if not np.all(np.isnan(curve)):
            if min_expr is not None:
                curve = np.maximum(curve, min_expr)
            ax.plot(pseudotime[sort_idx], curve[sort_idx],
                    color='black', linewidth=1.5, zorder=3)

        # Use short name if available
        if label_by_short_name and 'gene_short_name' in adata.var.columns:
            label = adata.var.loc[adata.var_names[gene_idx], 'gene_short_name']
            if pd.isna(label):
                label = gene
        else:
            label = gene

        ax.set_title(label)
        ax.set_xlabel('Pseudotime')
        if relative_expr:
            ax.set_ylabel('Expression')
        else:
            ax.set_ylabel('Absolute Expression')

        if expr.min() > 0:
            ax.set_yscale('log')

        _monocle_theme(ax)

    # Remove empty axes
    for idx in range(n_genes, nrow * ncol):
        row = idx // ncol
        col = idx % ncol
        axes[row, col].set_visible(False)

    # Legend
    if color_by in adata.obs.columns:
        color_vals_unique = sorted(set(adata.obs[color_by].values))
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_get_state_colors(adata.obs[color_by].values)[s],
                          markersize=6, label=str(s))
                   for s in color_vals_unique]
        fig.legend(handles=handles, loc='upper center',
                   bbox_to_anchor=(0.5, 1.02), ncol=min(len(handles), 6),
                   frameon=False)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')

    return fig


# ============================================================================
# plot_genes_branched_heatmap
# ============================================================================

def plot_genes_branched_heatmap(adata, branch_point=1, branch_states=None,
                                 branch_labels=None,
                                 cluster_rows=True, num_clusters=6,
                                 hclust_method='ward',
                                 hmcols=None, show_rownames=False,
                                 use_gene_short_name=True,
                                 scale_max=3, scale_min=-3,
                                 norm_method='log',
                                 trend_formula="~sm.ns(Pseudotime, df=3)*Branch",
                                 return_heatmap=False,
                                 cores=1, figsize=None, save=None, dpi=150):
    """
    Plot branched expression heatmap.

    Parameters
    ----------
    adata : AnnData
    branch_point : int
    branch_states : list or None
    branch_labels : list or None
    cluster_rows : bool
    num_clusters : int
    hclust_method : str
    hmcols : colormap or None
    show_rownames : bool
    use_gene_short_name : bool
    scale_max, scale_min : float
    norm_method : str
    trend_formula : str
    return_heatmap : bool
    cores : int
    figsize : tuple or None
    save : str or None
    dpi : int

    Returns
    -------
    fig or dict (if return_heatmap)
    """
    from .differential import gen_smooth_curves
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list

    if branch_labels is None:
        branch_labels = ['Cell fate 1', 'Cell fate 2']

    monocle = adata.uns.get('monocle', {})

    # Determine branch states
    if branch_states is None:
        states = sorted(adata.obs['State'].unique())
        root_state = adata.obs.loc[adata.obs['Pseudotime'].idxmin(), 'State']
        branch_states = [s for s in states if s != root_state][:2]

    # Build branch CDS
    mask1 = adata.obs['State'].isin([branch_states[0]])
    mask2 = adata.obs['State'].isin([branch_states[1]])
    progenitor_mask = ~adata.obs['State'].isin(branch_states)
    combined_mask = mask1 | mask2 | progenitor_mask

    adata_sub = adata[combined_mask].copy()

    branch_col = np.full(adata_sub.n_obs, 'Pre-branch', dtype=object)
    branch_col[adata_sub.obs['State'].isin([branch_states[0]]).values] = branch_labels[0]
    branch_col[adata_sub.obs['State'].isin([branch_states[1]]).values] = branch_labels[1]
    adata_sub.obs['Branch'] = pd.Categorical(branch_col)

    # Scale pseudotime 0-100
    pt = adata_sub.obs['Pseudotime'].values.copy()
    if pt.max() > pt.min():
        pt = 100 * (pt - pt.min()) / (pt.max() - pt.min())
    adata_sub.obs['Pseudotime'] = pt

    # Generate smooth curves for both branches
    n_points = 100
    newdataA = pd.DataFrame({
        'Pseudotime': np.linspace(0, 100, n_points),
        'Branch': pd.Categorical([branch_labels[0]] * n_points,
                                  categories=[branch_labels[0], branch_labels[1], 'Pre-branch'])
    })
    newdataB = pd.DataFrame({
        'Pseudotime': np.linspace(0, 100, n_points),
        'Branch': pd.Categorical([branch_labels[1]] * n_points,
                                  categories=[branch_labels[0], branch_labels[1], 'Pre-branch'])
    })
    new_data = pd.concat([newdataA, newdataB], ignore_index=True)

    smooth_exprs = gen_smooth_curves(adata_sub, new_data=new_data,
                                      trend_formula=trend_formula,
                                      relative_expr=True, cores=cores)

    BranchA_exprs = smooth_exprs[:, :n_points]
    BranchB_exprs = smooth_exprs[:, n_points:]

    # Normalize
    if norm_method == 'log':
        BranchA_exprs = np.log10(BranchA_exprs + 1)
        BranchB_exprs = np.log10(BranchB_exprs + 1)

    # Build heatmap: reverse branch A, then branch B
    heatmap_matrix = np.hstack([BranchA_exprs[:, ::-1], BranchB_exprs])

    # Remove zero-variance rows
    row_std = heatmap_matrix.std(axis=1)
    valid = row_std > 0
    heatmap_matrix = heatmap_matrix[valid]
    valid_genes = adata_sub.var_names[valid] if hasattr(adata_sub.var_names, '__getitem__') else np.arange(valid.sum())

    # Z-score normalize rows
    row_means = heatmap_matrix.mean(axis=1, keepdims=True)
    row_stds = heatmap_matrix.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1
    heatmap_matrix = (heatmap_matrix - row_means) / row_stds

    # Clip
    heatmap_matrix = np.clip(heatmap_matrix, scale_min, scale_max)
    heatmap_matrix[np.isnan(heatmap_matrix)] = 0

    # Hierarchical clustering
    corr_dist = 1 - np.corrcoef(heatmap_matrix)
    corr_dist[np.isnan(corr_dist)] = 1
    np.fill_diagonal(corr_dist, 0)

    from scipy.spatial.distance import squareform
    condensed_dist = squareform(corr_dist, checks=False)
    condensed_dist[condensed_dist < 0] = 0

    Z = linkage(condensed_dist, method=hclust_method)
    cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')
    order = leaves_list(Z).tolist()  # iterative — matches R's pheatmap ordering

    # Reorder
    heatmap_ordered = heatmap_matrix[order]

    # Plot
    if figsize is None:
        figsize = (10, max(6, len(heatmap_ordered) * 0.05))

    branch_colors_map = {
        'Pre-branch': '#979797',
        branch_labels[0]: '#F05662',
        branch_labels[1]: '#7990C8',
    }

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.03, 1], width_ratios=[0.05, 1],
                          hspace=0.02, wspace=0.02)

    # Branch annotation bar
    ax_col = fig.add_subplot(gs[0, 1])
    col_gap = n_points
    branch_bar = np.zeros((1, 2 * n_points, 3))
    for i in range(n_points):
        branch_bar[0, i] = mcolors.to_rgb(branch_colors_map[branch_labels[0]])
    for i in range(n_points):
        branch_bar[0, n_points + i] = mcolors.to_rgb(branch_colors_map[branch_labels[1]])
    ax_col.imshow(branch_bar, aspect='auto')
    ax_col.set_xticks([])
    ax_col.set_yticks([])
    ax_col.axvline(x=n_points - 0.5, color='white', linewidth=2)

    # Row cluster annotation
    ax_row = fig.add_subplot(gs[1, 0])
    cluster_ordered = cluster_labels[order]
    cluster_cmap = plt.get_cmap('Set3', num_clusters)
    row_bar = np.zeros((len(cluster_ordered), 1, 3))
    for i, c in enumerate(cluster_ordered):
        row_bar[i, 0] = cluster_cmap(c - 1)[:3]
    ax_row.imshow(row_bar, aspect='auto')
    ax_row.set_xticks([])
    ax_row.set_yticks([])

    # Main heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    if hmcols is None:
        from matplotlib.colors import LinearSegmentedColormap
        hmcols = LinearSegmentedColormap.from_list(
            'blue2green2red',
            ['#3C5488', '#00A087', '#E64B35'],
            N=256
        )

    ax_heat.imshow(heatmap_ordered, aspect='auto', cmap=hmcols,
                    vmin=scale_min, vmax=scale_max, interpolation='none')
    ax_heat.axvline(x=n_points - 0.5, color='white', linewidth=2)
    ax_heat.set_xticks([])

    if show_rownames:
        if use_gene_short_name and 'gene_short_name' in adata.var.columns:
            labels = adata.var.loc[valid_genes, 'gene_short_name'].values[order]
        else:
            labels = np.array(valid_genes)[order]
        ax_heat.set_yticks(range(len(labels)))
        ax_heat.set_yticklabels(labels, fontsize=6)
    else:
        ax_heat.set_yticks([])

    # Labels
    ax_heat.set_xlabel('')
    ax_col.set_title(f'{branch_labels[0]}  ←  Pre-branch  →  {branch_labels[1]}',
                      fontsize=10)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')

    if return_heatmap:
        return {
            'fig': fig,
            'heatmap_matrix': heatmap_ordered,
            'cluster_labels': cluster_ordered,
            'BranchA_exprs': BranchA_exprs,
            'BranchB_exprs': BranchB_exprs,
            'gene_order': order,
        }

    return fig


# ============================================================================
# plot_genes_branched_pseudotime
# ============================================================================

def plot_genes_branched_pseudotime(adata, genes, branch_point=1,
                                    branch_states=None, branch_labels=None,
                                    min_expr=None, cell_size=0.75,
                                    ncol=1, nrow=None,
                                    color_by='State',
                                    trend_formula="~sm.ns(Pseudotime, df=3)*Branch",
                                    relative_expr=True,
                                    figsize=None, save=None, dpi=150):
    """
    Plot gene expression along branched pseudotime.
    """
    from .differential import gen_smooth_curves

    if branch_labels is None:
        branch_labels = ['Cell fate 1', 'Cell fate 2']

    n_genes = len(genes)
    if nrow is None:
        nrow = int(np.ceil(n_genes / ncol))
    if figsize is None:
        figsize = (5 * ncol, 3.5 * nrow)

    # Build branch subset
    if branch_states is None:
        states = sorted(adata.obs['State'].unique())
        root_state = adata.obs.loc[adata.obs['Pseudotime'].idxmin(), 'State']
        branch_states = [s for s in states if s != root_state][:2]

    mask = adata.obs['State'].isin(branch_states) | \
           ~adata.obs['State'].isin(branch_states)
    adata_sub = adata[mask].copy()

    branch_col = np.full(adata_sub.n_obs, 'Pre-branch', dtype=object)
    branch_col[adata_sub.obs['State'].isin([branch_states[0]]).values] = branch_labels[0]
    branch_col[adata_sub.obs['State'].isin([branch_states[1]]).values] = branch_labels[1]
    adata_sub.obs['Branch'] = pd.Categorical(branch_col)

    pt = adata_sub.obs['Pseudotime'].values.copy()
    if pt.max() > pt.min():
        pt = 100 * (pt - pt.min()) / (pt.max() - pt.min())
    adata_sub.obs['Pseudotime'] = pt

    # Smooth curves
    n_points = 100
    newdataA = pd.DataFrame({
        'Pseudotime': np.linspace(0, 100, n_points),
        'Branch': pd.Categorical([branch_labels[0]] * n_points)
    })
    newdataB = pd.DataFrame({
        'Pseudotime': np.linspace(0, 100, n_points),
        'Branch': pd.Categorical([branch_labels[1]] * n_points)
    })
    new_data = pd.concat([newdataA, newdataB], ignore_index=True)
    smooth = gen_smooth_curves(adata_sub, new_data=new_data,
                                trend_formula=trend_formula, relative_expr=True)

    branch_colors = {'Pre-branch': '#979797', branch_labels[0]: '#F05662',
                     branch_labels[1]: '#7990C8'}

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    for idx, gene in enumerate(genes):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]

        if gene in adata_sub.var_names:
            gene_idx = list(adata_sub.var_names).index(gene)
        elif 'gene_short_name' in adata_sub.var.columns:
            matches = adata_sub.var[adata_sub.var['gene_short_name'] == gene].index
            if len(matches) > 0:
                gene_idx = list(adata_sub.var_names).index(matches[0])
            else:
                ax.set_title(f'{gene} (not found)')
                continue
        else:
            continue

        expr = adata_sub.X[:, gene_idx]
        if sparse.issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()

        # Scatter by branch
        for br in [branch_labels[0], branch_labels[1], 'Pre-branch']:
            mask_br = adata_sub.obs['Branch'] == br
            ax.scatter(pt[mask_br], expr[mask_br],
                       c=branch_colors[br], s=cell_size ** 2,
                       alpha=0.4, edgecolors='none', label=br)

        # Smooth curves
        curveA = smooth[gene_idx, :n_points]
        curveB = smooth[gene_idx, n_points:]
        t_line = np.linspace(0, 100, n_points)

        if not np.all(np.isnan(curveA)):
            ax.plot(t_line, curveA, color=branch_colors[branch_labels[0]], linewidth=2)
        if not np.all(np.isnan(curveB)):
            ax.plot(t_line, curveB, color=branch_colors[branch_labels[1]], linewidth=2)

        label = gene
        if 'gene_short_name' in adata_sub.var.columns:
            sn = adata_sub.var.iloc[gene_idx].get('gene_short_name')
            if pd.notna(sn):
                label = sn

        ax.set_title(label)
        ax.set_xlabel('Pseudotime')
        ax.set_ylabel('Expression')
        _monocle_theme(ax)

    # Remove empty
    for idx in range(n_genes, nrow * ncol):
        axes[idx // ncol, idx % ncol].set_visible(False)

    handles = [Line2D([0], [0], color=branch_colors[bl], linewidth=2, label=bl)
               for bl in branch_labels]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')

    return fig


# ============================================================================
# plot_cell_clusters
# ============================================================================

def plot_cell_clusters(adata, x=0, y=1, color_by='Cluster',
                       markers=None, cell_size=1.5,
                       figsize=(8, 6), ax=None, save=None, dpi=150):
    """Plot cells colored by cluster."""
    monocle = adata.uns.get('monocle', {})

    if 'X_DDRTree' in adata.obsm:
        coords = adata.obsm['X_DDRTree']
    elif 'X_tSNE' in adata.obsm:
        coords = adata.obsm['X_tSNE']
    elif 'reducedDimA' in monocle:
        coords = monocle['reducedDimA'].T
    else:
        coords = adata.obsm.get('X_pca', adata.X[:, :2] if adata.X.shape[1] >= 2 else None)
        if coords is None:
            raise ValueError("No reduced dimensions found")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if color_by in adata.obs.columns:
        color_vals = adata.obs[color_by].values
        color_map = _get_state_colors(color_vals)
        colors = [color_map[v] for v in color_vals]

        ax.scatter(coords[:, x], coords[:, y], c=colors,
                   s=cell_size ** 2, edgecolors='none')

        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color_map[s], markersize=6,
                          label=str(s))
                   for s in sorted(color_map.keys())]
        ax.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 1.15), ncol=min(len(handles), 6),
                  frameon=False)
    else:
        ax.scatter(coords[:, x], coords[:, y], s=cell_size ** 2)

    ax.set_xlabel(f'Component {x + 1}')
    ax.set_ylabel(f'Component {y + 1}')
    _monocle_theme(ax)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig, ax


# ============================================================================
# plot_genes_jitter
# ============================================================================

def plot_genes_jitter(adata, genes, grouping='State', min_expr=None,
                      cell_size=0.75, ncol=1, nrow=None,
                      color_by=None, plot_trend=False,
                      label_by_short_name=True,
                      figsize=None, save=None, dpi=150):
    """Plot gene expression as jitter/strip plot grouped by a variable."""
    n_genes = len(genes)
    if nrow is None:
        nrow = int(np.ceil(n_genes / ncol))
    if figsize is None:
        figsize = (5 * ncol, 3.5 * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    groups = adata.obs[grouping].values
    unique_groups = sorted(set(groups))
    color_map = _get_state_colors(groups)

    for idx, gene in enumerate(genes):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]

        if gene in adata.var_names:
            gene_idx = list(adata.var_names).index(gene)
        elif 'gene_short_name' in adata.var.columns:
            matches = adata.var[adata.var['gene_short_name'] == gene].index
            if len(matches) > 0:
                gene_idx = list(adata.var_names).index(matches[0])
            else:
                continue
        else:
            continue

        expr = adata.X[:, gene_idx]
        if sparse.issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()

        for g_idx, g in enumerate(unique_groups):
            mask = groups == g
            jitter = np.random.uniform(-0.3, 0.3, mask.sum())
            ax.scatter(g_idx + jitter, expr[mask],
                       c=[color_map[g]], s=cell_size ** 2,
                       alpha=0.5, edgecolors='none')

        ax.set_xticks(range(len(unique_groups)))
        ax.set_xticklabels([str(g) for g in unique_groups])
        ax.set_xlabel(grouping)
        ax.set_ylabel('Expression')

        label = gene
        if label_by_short_name and 'gene_short_name' in adata.var.columns:
            sn = adata.var.iloc[gene_idx].get('gene_short_name')
            if pd.notna(sn):
                label = sn
        ax.set_title(label)
        _monocle_theme(ax)

    for idx in range(n_genes, nrow * ncol):
        axes[idx // ncol, idx % ncol].set_visible(False)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig


# ============================================================================
# plot_genes_violin
# ============================================================================

def plot_genes_violin(adata, genes, grouping='State',
                      min_expr=None, ncol=1, nrow=None,
                      color_by=None, plot_as_count=False,
                      label_by_short_name=True,
                      figsize=None, save=None, dpi=150):
    """Plot gene expression as violin plot grouped by a variable."""
    n_genes = len(genes)
    if nrow is None:
        nrow = int(np.ceil(n_genes / ncol))
    if figsize is None:
        figsize = (5 * ncol, 3.5 * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    groups = adata.obs[grouping].values
    unique_groups = sorted(set(groups))
    color_map = _get_state_colors(groups)

    for idx, gene in enumerate(genes):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]

        if gene in adata.var_names:
            gene_idx = list(adata.var_names).index(gene)
        elif 'gene_short_name' in adata.var.columns:
            matches = adata.var[adata.var['gene_short_name'] == gene].index
            if len(matches) > 0:
                gene_idx = list(adata.var_names).index(matches[0])
            else:
                continue
        else:
            continue

        expr = adata.X[:, gene_idx]
        if sparse.issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()

        data_by_group = [expr[groups == g] for g in unique_groups]
        parts = ax.violinplot(data_by_group, positions=range(len(unique_groups)),
                              showmedians=True, showextrema=False)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(color_map[unique_groups[i]])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(unique_groups)))
        ax.set_xticklabels([str(g) for g in unique_groups])
        ax.set_xlabel(grouping)
        ax.set_ylabel('Expression')

        label = gene
        if label_by_short_name and 'gene_short_name' in adata.var.columns:
            sn = adata.var.iloc[gene_idx].get('gene_short_name')
            if pd.notna(sn):
                label = sn
        ax.set_title(label)
        _monocle_theme(ax)

    for idx in range(n_genes, nrow * ncol):
        axes[idx // ncol, idx % ncol].set_visible(False)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig


# ============================================================================
# plot_ordering_genes
# ============================================================================

def plot_ordering_genes(adata, figsize=(6, 4), save=None, dpi=150):
    """Plot dispersion vs mean expression, highlighting ordering genes."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    mu = adata.var.get('mean_expression', None)
    disp = adata.var.get('dispersion_empirical', None)

    if mu is None or disp is None:
        raise ValueError("Run estimate_dispersions() first")

    use_for_ordering = adata.var.get('use_for_ordering',
                                     pd.Series(False, index=adata.var_names))

    valid = (mu > 0) & np.isfinite(disp) & (disp > 0)

    ax.scatter(np.log10(mu[valid & ~use_for_ordering]),
               np.log10(disp[valid & ~use_for_ordering]),
               c='grey', s=2, alpha=0.3, label='Other genes')

    if use_for_ordering.sum() > 0:
        ax.scatter(np.log10(mu[valid & use_for_ordering]),
                   np.log10(disp[valid & use_for_ordering]),
                   c='red', s=4, alpha=0.6, label='Ordering genes')

    # Fitted curve
    if 'dispersion_fit' in adata.var.columns:
        fitted = adata.var['dispersion_fit']
        sort_idx = np.argsort(mu[valid].values)
        ax.plot(np.log10(mu[valid].values[sort_idx]),
                np.log10(fitted[valid].values[sort_idx]),
                color='blue', linewidth=1, label='Fitted')

    ax.set_xlabel('log10(Mean Expression)')
    ax.set_ylabel('log10(Dispersion)')
    ax.legend(frameon=False)
    _monocle_theme(ax)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig


# ============================================================================
# plot_pseudotime_heatmap
# ============================================================================

def plot_pseudotime_heatmap(adata, genes=None, num_clusters=6,
                             hclust_method='ward',
                             hmcols=None, show_rownames=False,
                             use_gene_short_name=True,
                             scale_max=3, scale_min=-3,
                             norm_method='log',
                             trend_formula="~sm.ns(Pseudotime, df=3)",
                             cores=1, figsize=None, save=None, dpi=150):
    """
    Plot pseudotime heatmap of gene expression.
    """
    from .differential import gen_smooth_curves
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list

    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        adata_sub = adata[:, gene_mask].copy()
    else:
        adata_sub = adata.copy()

    # Sort by pseudotime
    sort_idx = np.argsort(adata_sub.obs['Pseudotime'].values)

    n_points = 100
    new_data = pd.DataFrame({
        'Pseudotime': np.linspace(
            adata_sub.obs['Pseudotime'].min(),
            adata_sub.obs['Pseudotime'].max(),
            n_points
        )
    })

    smooth = gen_smooth_curves(adata_sub, new_data=new_data,
                                trend_formula=trend_formula,
                                relative_expr=True, cores=cores)

    if norm_method == 'log':
        smooth = np.log10(smooth + 1)

    # Z-score rows
    row_means = smooth.mean(axis=1, keepdims=True)
    row_stds = smooth.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1
    heatmap = (smooth - row_means) / row_stds
    heatmap = np.clip(heatmap, scale_min, scale_max)
    heatmap[np.isnan(heatmap)] = 0

    # Remove zero-variance
    valid = heatmap.std(axis=1) > 0
    heatmap = heatmap[valid]

    if heatmap.shape[0] == 0:
        # Nothing plottable — all genes had degenerate fits. Return an
        # empty placeholder figure rather than indexing into a 0-row
        # array. Upstream callers can detect this by the empty axes.
        fig, ax = plt.subplots(1, 1, figsize=figsize or (6, 3))
        ax.text(0.5, 0.5,
                "No genes with non-zero variance after smoothing",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        _monocle_theme(ax)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=dpi, bbox_inches='tight')
        return fig

    # Cluster
    if heatmap.shape[0] > 1:
        corr_dist = 1 - np.corrcoef(heatmap)
        corr_dist[np.isnan(corr_dist)] = 1
        np.fill_diagonal(corr_dist, 0)

        from scipy.spatial.distance import squareform
        condensed = squareform(corr_dist, checks=False)
        condensed[condensed < 0] = 0

        Z = linkage(condensed, method=hclust_method)
        order = leaves_list(Z).tolist()  # iterative — no Python recursion
        cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')
    else:
        order = [0]
        cluster_labels = [1]

    heatmap_ordered = heatmap[order]

    if figsize is None:
        figsize = (8, max(4, len(heatmap_ordered) * 0.05))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if hmcols is None:
        from matplotlib.colors import LinearSegmentedColormap
        hmcols = LinearSegmentedColormap.from_list(
            'blue2green2red',
            ['#3C5488', '#00A087', '#E64B35'],
            N=256
        )

    im = ax.imshow(heatmap_ordered, aspect='auto', cmap=hmcols,
                    vmin=scale_min, vmax=scale_max, interpolation='none')

    ax.set_xlabel('Pseudotime →')
    ax.set_xticks([])

    if show_rownames and adata_sub is not None:
        valid_genes = adata_sub.var_names[valid]
        if use_gene_short_name and 'gene_short_name' in adata_sub.var.columns:
            labels = adata_sub.var.loc[valid_genes, 'gene_short_name'].values[order]
        else:
            labels = np.array(valid_genes)[order]
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_yticks([])

    plt.colorbar(im, ax=ax, label='Scaled Expression', shrink=0.5)
    _monocle_theme(ax)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig


# ============================================================================
# plot_complex_cell_trajectory (dendro-style layered trajectory)
# ============================================================================

def plot_complex_cell_trajectory(adata, color_by='State', show_branch_points=True,
                                  cell_size=0.5, cell_link_size=0.3,
                                  root_states=None, figsize=(8, 6),
                                  cmap=None, ax=None, save=None, dpi=150):
    """
    Plot the trajectory in tree-layout form (like Monocle2's
    plot_complex_cell_trajectory). Uses a dendrogram-style layered layout
    of the cell projection MST so branches are visually separated.
    """
    import igraph as ig_

    monocle = adata.uns['monocle']
    if 'pr_graph_cell_proj_tree' not in monocle:
        raise ValueError("Run order_cells() first to build the cell projection MST")

    cell_mst = monocle['pr_graph_cell_proj_tree']
    vertex_names = cell_mst.vs['name']
    N = len(vertex_names)

    pseudotime = adata.obs.loc[vertex_names, 'Pseudotime'].values

    # Root = cell with min pseudotime
    root_idx = int(np.argmin(pseudotime))

    # BFS traversal from root; layer = pseudotime level
    # x coordinate = horizontal spread based on branch
    # y coordinate = pseudotime
    # Use igraph layout_reingold_tilford for tree layout
    tree_coords = cell_mst.layout_reingold_tilford(root=[root_idx])
    coords = np.array(tree_coords.coords)  # (N, 2)

    # Flip y so root is at top
    coords[:, 1] = coords[:, 1].max() - coords[:, 1]

    # Reorder to match adata.obs_names order
    name_to_idx = {n: i for i, n in enumerate(vertex_names)}
    order = [name_to_idx[n] for n in adata.obs_names]
    coords_sorted = coords[order]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Draw edges
    for e in cell_mst.es:
        i, j = e.source, e.target
        ax.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color='black', linewidth=cell_link_size, zorder=1)

    # Draw cells
    if color_by in adata.obs.columns:
        vals = adata.obs[color_by].values
        if (hasattr(vals, 'categories')
            or not np.issubdtype(np.array(vals).dtype, np.floating)):
            cmap_colors = _get_state_colors(vals)
            colors = [cmap_colors[v] for v in vals]
            ax.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                       c=colors, s=cell_size ** 2 * 10, edgecolors='none', zorder=2)
            handles = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=cmap_colors[s], markersize=6,
                              label=str(s))
                       for s in sorted(cmap_colors.keys())]
            ax.legend(handles=handles, loc='upper center',
                      bbox_to_anchor=(0.5, 1.12),
                      ncol=min(len(handles), 6), frameon=False, fontsize=8)
        else:
            sc = ax.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                            c=vals.astype(float), s=cell_size ** 2 * 10,
                            cmap=cmap or 'viridis', zorder=2, edgecolors='none')
            plt.colorbar(sc, ax=ax, label=color_by)
    else:
        ax.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                   s=cell_size ** 2 * 10, edgecolors='none', zorder=2)

    ax.set_xlabel('')
    ax.set_ylabel('Pseudotime')
    ax.set_xticks([])
    _monocle_theme(ax)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig, ax


# ============================================================================
# plot_multiple_branches_pseudotime
# ============================================================================

def plot_multiple_branches_pseudotime(adata, genes, branches, branches_name=None,
                                       color_by='Branch', trend_formula=None,
                                       ncol=2, nrow=None, cell_size=0.5,
                                       figsize=None, save=None, dpi=150):
    """
    Plot gene expression along MULTIPLE branches simultaneously
    (one curve per branch per gene).

    Parameters
    ----------
    adata : AnnData
    genes : list of gene names
    branches : list of State values, one per branch
    branches_name : list of branch labels (same length as branches)
    color_by : str ('Branch' colors by branch label)
    trend_formula : formula string (default: ~sm.ns(Pseudotime, df=3)*Branch)
    """
    from .differential import gen_smooth_curves

    if branches_name is None:
        branches_name = [f'Branch {s}' for s in branches]

    n_branches = len(branches)
    n_genes = len(genes)
    if nrow is None:
        nrow = int(np.ceil(n_genes / ncol))
    if figsize is None:
        figsize = (5 * ncol, 3.5 * nrow)

    # Default colors for branches
    default_colors = ['#B6A5D0', '#76C143', '#3DC6F2', '#FDB863',
                      '#F0027F', '#666666']
    branch_colors = {bn: default_colors[i % len(default_colors)]
                     for i, bn in enumerate(branches_name)}

    # Build per-branch subset with Branch label
    cells_per_branch = {}
    for st, bn in zip(branches, branches_name):
        mask = adata.obs['State'] == st
        cells_per_branch[bn] = adata.obs_names[mask].tolist()

    # Create combined subset
    all_cells = []
    branch_labels = []
    for bn in branches_name:
        all_cells.extend(cells_per_branch[bn])
        branch_labels.extend([bn] * len(cells_per_branch[bn]))

    adata_sub = adata[all_cells, :].copy()
    adata_sub.obs['Branch'] = pd.Categorical(branch_labels)

    # Normalize Pseudotime 0–100 per branch for comparability
    pt = adata_sub.obs['Pseudotime'].values.astype(float)
    if pt.max() > pt.min():
        pt = 100 * (pt - pt.min()) / (pt.max() - pt.min())
    adata_sub.obs['Pseudotime'] = pt

    # Generate smooth curves for each branch
    n_points = 100
    newdatas = []
    for bn in branches_name:
        newdatas.append(pd.DataFrame({
            'Pseudotime': np.linspace(0, 100, n_points),
            'Branch': pd.Categorical([bn] * n_points, categories=branches_name),
        }))
    new_data = pd.concat(newdatas, ignore_index=True)

    tf = trend_formula or "~sm.ns(Pseudotime, df=3)*Branch"
    smooth = gen_smooth_curves(adata_sub, new_data=new_data,
                                trend_formula=tf, relative_expr=True)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    for gi, gene in enumerate(genes):
        r, c = gi // ncol, gi % ncol
        ax = axes[r, c]

        # Find gene index
        if gene in adata_sub.var_names:
            gidx = list(adata_sub.var_names).index(gene)
        elif 'gene_short_name' in adata_sub.var.columns:
            matches = adata_sub.var[adata_sub.var['gene_short_name'] == gene].index
            if len(matches) == 0:
                ax.set_title(f'{gene} (not found)')
                continue
            gidx = list(adata_sub.var_names).index(matches[0])
        else:
            continue

        expr = adata_sub.X[:, gidx]
        if sparse.issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()

        for bi, bn in enumerate(branches_name):
            mask = adata_sub.obs['Branch'] == bn
            ax.scatter(pt[mask], expr[mask], c=branch_colors[bn],
                       s=cell_size ** 2 * 10, alpha=0.4, edgecolors='none',
                       label=bn if gi == 0 else None)

            curve = smooth[gidx, bi * n_points:(bi + 1) * n_points]
            t_line = np.linspace(0, 100, n_points)
            if not np.all(np.isnan(curve)):
                ax.plot(t_line, curve, color=branch_colors[bn], linewidth=2)

        ax.set_title(gene)
        ax.set_xlabel('Pseudotime')
        ax.set_ylabel('Expression')
        _monocle_theme(ax)

    for gi in range(n_genes, nrow * ncol):
        axes[gi // ncol, gi % ncol].set_visible(False)

    handles = [Line2D([0], [0], color=branch_colors[bn], linewidth=2, label=bn)
               for bn in branches_name]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=len(branches_name),
               frameon=False)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig


# ============================================================================
# plot_multiple_branches_heatmap
# ============================================================================

def plot_multiple_branches_heatmap(adata, branches, branches_name=None,
                                    num_clusters=4, hclust_method='ward',
                                    show_rownames=True, hmcols=None,
                                    scale_max=3, scale_min=-3,
                                    norm_method='log',
                                    trend_formula=None,
                                    figsize=None, save=None, dpi=150):
    """
    Plot multi-branch expression heatmap — columns are pseudotime points of
    each branch in sequence, rows are genes clustered by expression pattern.
    """
    from .differential import gen_smooth_curves
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list

    if branches_name is None:
        branches_name = [f'Branch {s}' for s in branches]

    # Build subset with Branch column
    cells_per_branch = {}
    for st, bn in zip(branches, branches_name):
        mask = adata.obs['State'] == st
        cells_per_branch[bn] = adata.obs_names[mask].tolist()

    all_cells = []
    branch_labels = []
    for bn in branches_name:
        all_cells.extend(cells_per_branch[bn])
        branch_labels.extend([bn] * len(cells_per_branch[bn]))

    adata_sub = adata[all_cells, :].copy()
    adata_sub.obs['Branch'] = pd.Categorical(branch_labels)

    pt = adata_sub.obs['Pseudotime'].values.astype(float)
    if pt.max() > pt.min():
        pt = 100 * (pt - pt.min()) / (pt.max() - pt.min())
    adata_sub.obs['Pseudotime'] = pt

    # Generate smooth curves per branch
    n_points = 100
    newdatas = []
    for bn in branches_name:
        newdatas.append(pd.DataFrame({
            'Pseudotime': np.linspace(0, 100, n_points),
            'Branch': pd.Categorical([bn] * n_points, categories=branches_name),
        }))
    new_data = pd.concat(newdatas, ignore_index=True)

    tf = trend_formula or "~sm.ns(Pseudotime, df=3)*Branch"
    smooth = gen_smooth_curves(adata_sub, new_data=new_data,
                                trend_formula=tf, relative_expr=True)

    # Log-transform
    if norm_method == 'log':
        smooth = np.log10(np.maximum(smooth, 0) + 1)

    # Concatenate branches horizontally
    heatmap = smooth  # shape (G, n_points * n_branches)

    row_std = heatmap.std(axis=1)
    valid = row_std > 0
    heatmap = heatmap[valid]
    valid_genes = adata_sub.var_names[valid]

    # Row z-score
    row_means = heatmap.mean(axis=1, keepdims=True)
    row_stds = heatmap.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1
    heatmap = (heatmap - row_means) / row_stds
    heatmap = np.clip(heatmap, scale_min, scale_max)
    heatmap[np.isnan(heatmap)] = 0

    # Cluster rows
    if heatmap.shape[0] > 1:
        corr = np.corrcoef(heatmap)
        corr[np.isnan(corr)] = 0
        dist = 1 - corr
        np.fill_diagonal(dist, 0)
        from scipy.spatial.distance import squareform as sqf
        condensed = sqf(dist, checks=False)
        condensed[condensed < 0] = 0
        Z = linkage(condensed, method=hclust_method)
        order = leaves_list(Z).tolist()  # iterative — no Python recursion
        cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')
    else:
        order = [0]
        cluster_labels = [1]

    heatmap_ord = heatmap[order]
    cluster_ord = cluster_labels[order]

    if figsize is None:
        figsize = (8, max(5, len(heatmap_ord) * 0.04))

    n_branches = len(branches_name)
    default_colors = ['#B6A5D0', '#76C143', '#3DC6F2', '#FDB863', '#F0027F']
    branch_cmap = {bn: default_colors[i % len(default_colors)]
                   for i, bn in enumerate(branches_name)}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.03, 1],
                          width_ratios=[0.04, 1],
                          hspace=0.02, wspace=0.02)

    # Column annotation (branch bar)
    ax_col = fig.add_subplot(gs[0, 1])
    col_bar = np.zeros((1, n_points * n_branches, 3))
    for bi, bn in enumerate(branches_name):
        rgb = mcolors.to_rgb(branch_cmap[bn])
        for p in range(n_points):
            col_bar[0, bi * n_points + p] = rgb
    ax_col.imshow(col_bar, aspect='auto')
    ax_col.set_xticks([])
    ax_col.set_yticks([])
    for bi in range(1, n_branches):
        ax_col.axvline(bi * n_points - 0.5, color='white', linewidth=2)

    # Row cluster annotation
    ax_row = fig.add_subplot(gs[1, 0])
    cluster_colors = plt.get_cmap('Set3', num_clusters)
    row_bar = np.zeros((len(cluster_ord), 1, 3))
    for i, c in enumerate(cluster_ord):
        row_bar[i, 0] = cluster_colors(c - 1)[:3]
    ax_row.imshow(row_bar, aspect='auto')
    ax_row.set_xticks([])
    ax_row.set_yticks([])

    # Main heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    if hmcols is None:
        from matplotlib.colors import LinearSegmentedColormap
        hmcols = LinearSegmentedColormap.from_list(
            'blue2green2red', ['#3C5488', '#00A087', '#E64B35'], N=256)
    ax_heat.imshow(heatmap_ord, aspect='auto', cmap=hmcols,
                    vmin=scale_min, vmax=scale_max, interpolation='none')
    ax_heat.set_xticks([])
    for bi in range(1, n_branches):
        ax_heat.axvline(bi * n_points - 0.5, color='white', linewidth=2)

    if show_rownames:
        if 'gene_short_name' in adata_sub.var.columns:
            labels = adata_sub.var.loc[valid_genes, 'gene_short_name'].values[order]
        else:
            labels = np.array(valid_genes)[order]
        ax_heat.set_yticks(range(len(labels)))
        ax_heat.set_yticklabels(labels, fontsize=5)
    else:
        ax_heat.set_yticks([])

    ax_col.set_title('  '.join(['← ' + bn for bn in branches_name]), fontsize=10)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig


# ============================================================================
# plot_rho_delta (density peak clustering)
# ============================================================================

def plot_rho_delta(adata, rho_threshold=None, delta_threshold=None,
                   figsize=(5, 4), save=None, dpi=150):
    """Plot rho vs delta from density peak clustering."""
    monocle = adata.uns.get('monocle', {})
    if 'rho' not in monocle or 'delta' not in monocle:
        raise ValueError("Run cluster_cells(method='densityPeak') first")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(monocle['rho'], monocle['delta'], s=8, c='steelblue',
               edgecolors='none', alpha=0.7)
    if rho_threshold is not None:
        ax.axvline(rho_threshold, color='red', linestyle='--', linewidth=1)
    if delta_threshold is not None:
        ax.axhline(delta_threshold, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('rho')
    ax.set_ylabel('delta')
    _monocle_theme(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return fig


# ============================================================================
# plot_pc_variance_explained
# ============================================================================

def plot_pc_variance_explained(adata, max_components=50, norm_method='log',
                                figsize=(5, 4), save=None, dpi=150,
                                return_all=False):
    """Plot variance explained by principal components."""
    from sklearn.decomposition import PCA
    from scipy import sparse as sp_

    X = adata.X
    if sp_.issparse(X):
        X = X.toarray()
    X = np.array(X, dtype=float)

    if 'Size_Factor' in adata.obs.columns:
        sf = adata.obs['Size_Factor'].values
        X = X / sf[:, None]

    if norm_method == 'log':
        X = np.log2(X + 1)

    # Row-standardize (genes are columns here since cells are rows)
    X = X - X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1
    X = X / stds

    n_comp = min(max_components, min(X.shape) - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    variance = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(range(1, n_comp + 1), variance, 'o-', markersize=3, color='steelblue')
    ax.set_xlabel('PC')
    ax.set_ylabel('% Variance Explained')
    _monocle_theme(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    if return_all:
        return {'variance': variance, 'pca': pca, 'fig': fig}
    return fig
