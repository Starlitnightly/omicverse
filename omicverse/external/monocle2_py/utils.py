"""
Utility functions for monocle2_py.

Implements calABCs, calILRs, and other helper functions.
"""

import numpy as np
import pandas as pd
from scipy import sparse


def cal_ABCs(adata, branch_point=1, branch_states=None, branch_labels=None,
             trend_formula="~sm.ns(Pseudotime, df=3)*Branch",
             relative_expr=True, stretch=True, cores=1,
             verbose=False, min_expr=0.5, num=5000):
    """
    Calculate Area Between Curves (ABCs) for branch-specific gene expression.

    Parameters
    ----------
    adata : AnnData
    branch_point : int
    branch_states : list or None
    branch_labels : list or None
    trend_formula : str
    relative_expr : bool
    stretch : bool
    cores : int
    verbose : bool
    min_expr : float
    num : int
        Number of interpolation points.

    Returns
    -------
    pd.DataFrame with ABCs column and gene metadata.
    """
    from .differential import gen_smooth_curves

    monocle = adata.uns.get('monocle', {})

    if branch_states is None:
        states = sorted(adata.obs['State'].unique())
        root_state = adata.obs.loc[adata.obs['Pseudotime'].idxmin(), 'State']
        branch_states = [s for s in states if s != root_state][:2]

    if len(branch_states) != 2:
        raise ValueError("ABCs requires exactly 2 branch states")

    if branch_labels is None:
        branch_labels = [f'State_{branch_states[0]}', f'State_{branch_states[1]}']

    # Build branch subset
    mask1 = adata.obs['State'].isin([branch_states[0]])
    mask2 = adata.obs['State'].isin([branch_states[1]])
    progenitor = ~adata.obs['State'].isin(branch_states)
    combined = mask1 | mask2 | progenitor

    adata_sub = adata[combined].copy()

    branch_col = np.full(adata_sub.n_obs, 'Pre-branch', dtype=object)
    branch_col[adata_sub.obs['State'].isin([branch_states[0]]).values] = branch_labels[0]
    branch_col[adata_sub.obs['State'].isin([branch_states[1]]).values] = branch_labels[1]
    adata_sub.obs['Branch'] = pd.Categorical(branch_col)

    # Rescale pseudotime to [0, 100] **only on the local subset** for
    # smooth-curve fitting. We do NOT touch the caller's adata — each
    # BEAM/ABCs/ILRs call rescales against its own subset's range so
    # the resulting curves are always on [0, 100] regardless of which
    # branch_point was chosen. The original `mono.adata.obs['Pseudotime']`
    # is preserved.
    pt = adata_sub.obs['Pseudotime'].values.copy()
    if pt.max() > pt.min():
        pt = 100 * (pt - pt.min()) / (pt.max() - pt.min())
    adata_sub.obs['Pseudotime'] = pt

    # Smooth curves for both branches
    newdataA = pd.DataFrame({
        'Pseudotime': np.linspace(0, 100, num),
        'Branch': pd.Categorical([branch_labels[0]] * num)
    })
    newdataB = pd.DataFrame({
        'Pseudotime': np.linspace(0, 100, num),
        'Branch': pd.Categorical([branch_labels[1]] * num)
    })
    new_data = pd.concat([newdataA, newdataB], ignore_index=True)

    smooth = gen_smooth_curves(adata_sub, new_data=new_data,
                                trend_formula=trend_formula,
                                relative_expr=relative_expr, cores=cores)

    branchA = smooth[:, :num]
    branchB = smooth[:, num:]

    # Compute ABCs by trapezoidal integration
    delta = branchA - branchB
    step = 100.0 / (num - 1)
    avg_delta = (delta[:, :-1] + delta[:, 1:]) / 2
    ABCs = np.round(np.sum(avg_delta * step, axis=1), 3)

    result = pd.DataFrame({'ABCs': ABCs}, index=adata_sub.var_names)

    if 'gene_short_name' in adata_sub.var.columns:
        result['gene_short_name'] = adata_sub.var['gene_short_name'].values

    return result


def cal_ILRs(adata, branch_point=1, branch_states=None, branch_labels=None,
             trend_formula="~sm.ns(Pseudotime, df=3)*Branch",
             relative_expr=True, cores=1, verbose=False,
             stretch=True, num=5000, return_all=False):
    """
    Calculate Intrinsic Log Ratios (ILRs) for branch-specific expression.

    Parameters
    ----------
    adata : AnnData
    branch_point : int
    branch_states : list or None
    branch_labels : list or None
    trend_formula : str
    relative_expr : bool
    cores : int
    verbose : bool
    stretch : bool
    num : int

    Returns
    -------
    pd.DataFrame with ILR values.
    """
    from .differential import gen_smooth_curves

    if branch_states is None:
        states = sorted(adata.obs['State'].unique())
        root_state = adata.obs.loc[adata.obs['Pseudotime'].idxmin(), 'State']
        branch_states = [s for s in states if s != root_state][:2]

    if len(branch_states) != 2:
        raise ValueError("ILRs require exactly 2 branch states")

    if branch_labels is None:
        branch_labels = [f'State_{branch_states[0]}', f'State_{branch_states[1]}']

    mask1 = adata.obs['State'].isin([branch_states[0]])
    mask2 = adata.obs['State'].isin([branch_states[1]])
    progenitor = ~adata.obs['State'].isin(branch_states)
    combined = mask1 | mask2 | progenitor

    adata_sub = adata[combined].copy()

    branch_col = np.full(adata_sub.n_obs, 'Pre-branch', dtype=object)
    branch_col[adata_sub.obs['State'].isin([branch_states[0]]).values] = branch_labels[0]
    branch_col[adata_sub.obs['State'].isin([branch_states[1]]).values] = branch_labels[1]
    adata_sub.obs['Branch'] = pd.Categorical(branch_col)

    pt = adata_sub.obs['Pseudotime'].values.copy()
    if pt.max() > pt.min():
        pt = 100 * (pt - pt.min()) / (pt.max() - pt.min())
    adata_sub.obs['Pseudotime'] = pt

    newdataA = pd.DataFrame({
        'Pseudotime': np.linspace(0, 100, num),
        'Branch': pd.Categorical([branch_labels[0]] * num)
    })
    newdataB = pd.DataFrame({
        'Pseudotime': np.linspace(0, 100, num),
        'Branch': pd.Categorical([branch_labels[1]] * num)
    })
    new_data = pd.concat([newdataA, newdataB], ignore_index=True)

    smooth = gen_smooth_curves(adata_sub, new_data=new_data,
                                trend_formula=trend_formula,
                                relative_expr=relative_expr, cores=cores)

    branchA = smooth[:, :num]
    branchB = smooth[:, num:]

    # ILR: log ratio at each point
    eps = 1e-10
    ilr = np.log2((branchA + eps) / (branchB + eps))

    if return_all:
        # Return full per-point matrices like Monocle2's return_all=TRUE
        return {
            'str_branchA_expression_curve_matrix': pd.DataFrame(
                branchA, index=adata_sub.var_names),
            'str_branchB_expression_curve_matrix': pd.DataFrame(
                branchB, index=adata_sub.var_names),
            'norm_str_logfc_df': pd.DataFrame(
                ilr, index=adata_sub.var_names),
        }

    # Summary statistics (per-gene) — the per-gene ILR mean across pseudotime
    result = pd.DataFrame({
        'ILR_mean': ilr.mean(axis=1),
        'ILR_max': ilr.max(axis=1),
        'ILR_min': ilr.min(axis=1),
        'ILR_abs_mean': np.abs(ilr).mean(axis=1),
    }, index=adata_sub.var_names)

    if 'gene_short_name' in adata_sub.var.columns:
        result['gene_short_name'] = adata_sub.var['gene_short_name'].values

    return result
