import numpy as np
import pandas as pd

def in_silico_fold_change(concentration, fold_change_limit=9):
    ''' Take a series of known concentration, and a fold change limit, and relabel
    the concentrations such that new labels are within the fold change limit.

    Returns the given concentrations, the replaced concentration, the fold change between them,
    as well as the dictionary used to make the relabeling.
    '''
    conc_df = pd.DataFrame(concentration)
    conc_df.columns = ['concentration']
    global_candidates = conc_df.index

    # Create swappings which are consistent with fold change limit
    candidates = {}
    for e in conc_df.index:
        cc = conc_df.loc[e, 'concentration']
        ll = cc / fold_change_limit
        ul = cc * fold_change_limit
        candidates[e] = conc_df.query('{} < concentration < {}'.format(ll, ul)).index
        candidates[e] = candidates[e].drop(e)

    replacement = {}
    for e in conc_df.index:
        if e not in global_candidates:
            continue

        possible_swaps = global_candidates.intersection(candidates[e])
        # break
        if len(possible_swaps) < 1:
            replacement[e] = e
            global_candidates = global_candidates.drop(replacement[e])
        else:
            replacement[e] = np.random.choice(possible_swaps, replace=False)
            replacement[replacement[e]] = e
            global_candidates = global_candidates.drop(replacement[e])
            global_candidates = global_candidates.drop(e)

    # Make randomly swapped annotation
    shuff_concentration = concentration.copy().rename(replacement)
    concentration = concentration.sort_index()
    shuff_concentration = shuff_concentration.sort_index()

    log2_fc = np.log2(shuff_concentration / concentration)

    return concentration, shuff_concentration, log2_fc, replacement


def in_silico_conditions(expression_table, replacement):
    ''' Takes and expression table, and a dictionary for remapping gene names
    in the expression table.

    This randomly partitions the table in to two conditions, A and B.
    The B condition will have genes renamed based on the replacement dict.

    Returns the new table, and annotation about which is which.
    '''
    # Split samples in two
    n_samples = expression_table.columns.shape[0]
    shuffled_samples = np.random.choice(expression_table.columns, n_samples, replace=False)
    A_samples = shuffled_samples[:n_samples // 2]
    B_samples = shuffled_samples[n_samples // 2:]

    # Swap input abundance annotation for half of the samples
    A_table = expression_table[A_samples]
    B_table = expression_table[B_samples]
    B_table = B_table.rename(index=replacement)

    # Put everything back to a single table
    shuff_table = A_table.join(B_table)

    # Create annotation for the shuffled samples
    sample_info = pd.DataFrame({'n_genes': (shuff_table > 1.).sum(0)})
    sample_info['condition'] = \
    ['A' if s in A_samples else 'B' for s in sample_info.index]

    return shuff_table, sample_info
