import numpy as np
import scanpy as sc


def select_features(adata, groupName, n_features=10000, use_raw=True, verbose=False, sc_kwargs={}):
    r"""
    #TODO Write docstring
    """
    # Subsets adata to features that best distinguish a group given in adata.obs[groupName]
    if "rank_genes_groups" in adata.uns.keys():
        if verbose:
            print("Using existing ranked genes...")  # print options should be optional

    else:
        sc.tl.rank_genes_groups(
            adata, groupName, use_raw=use_raw, n_genes=int(np.round(len(adata.var) / 10)), **sc_kwargs
        )  # explain why

    ranked_features = np.unique([item for sublist in adata.uns["rank_genes_groups"]["names"] for item in sublist])

    if n_features > len(ranked_features):
        if verbose:
            print(f"Maximum number of features: {len(ranked_features)}")  # print options should be optional
        selected_features = ranked_features
        return adata[:, selected_features].var_names

    else:
        i = 1
        selected_features = []

        while len(np.unique(selected_features)) < n_features:
            selected_features = [item for sublist in adata.uns["rank_genes_groups"]["names"][:][:i] for item in sublist]
            i += 1
        return adata[:, np.unique(selected_features)[:n_features]].var_names
