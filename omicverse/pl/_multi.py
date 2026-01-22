from typing import Union, List, Optional, Iterable, Sequence, Dict
import warnings

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from anndata import AnnData
from ._single import embedding

def embedding_multi(
    data: AnnData,
    basis: str,
    color: Optional[Union[str, Sequence[str]]] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    **kwargs,
):
    r"""
    Create embedding scatter plots for multi-modal data (MuData) or single-cell data.
    
    Produces scatter plots on specified embeddings, supporting cross-modality feature visualization.
    For modality-specific embeddings, use format 'modality:embedding' (e.g., 'rna:X_pca').
    
    Args:
        data: AnnData or MuData object containing embedding and feature data
        basis: Name of embedding in obsm (e.g., 'X_umap') or modality-specific ('rna:X_pca')
        color: Gene names or obs columns to color points by (None)
        use_raw: Whether to use .raw attribute for features (None, auto-determined)
        layer: Specific data layer to use for coloring (None)
        **kwargs: Additional arguments passed to embedding plotting function
        
    Returns:
        Plot axes or figure depending on underlying plotting function
    """
    if isinstance(data, AnnData):
        return embedding(
            data, basis=basis, color=color, use_raw=use_raw, layer=layer, **kwargs
        )

    # `data` is MuData
    if basis not in data.obsm and "X_" + basis in data.obsm:
        basis = "X_" + basis

    if basis in data.obsm:
        adata = data
        basis_mod = basis
    else:
        # basis is not a joint embedding
        try:
            mod, basis_mod = basis.split(":")
        except ValueError:
            raise ValueError(f"Basis {basis} is not present in the MuData object (.obsm)")

        if mod not in data.mod:
            raise ValueError(
                f"Modality {mod} is not present in the MuData object with modalities {', '.join(data.mod)}"
            )

        adata = data.mod[mod]
        if basis_mod not in adata.obsm:
            if "X_" + basis_mod in adata.obsm:
                basis_mod = "X_" + basis_mod
            elif len(adata.obsm) > 0:
                raise ValueError(
                    f"Basis {basis_mod} is not present in the modality {mod} with embeddings {', '.join(adata.obsm)}"
                )
            else:
                raise ValueError(
                    f"Basis {basis_mod} is not present in the modality {mod} with no embeddings"
                )

    obs = data.obs.loc[adata.obs.index.values]

    if color is None:
        ad = AnnData(obs=obs, obsm=adata.obsm, obsp=adata.obsp)
        return sc.pl.embedding(ad, basis=basis_mod, **kwargs)

    # Some `color` has been provided
    if isinstance(color, str):
        keys = color = [color]
    elif isinstance(color, Iterable):
        keys = color
    else:
        raise TypeError("Expected color to be a string or an iterable.")

    # Fetch respective features
    if not all([key in obs for key in keys]):
        # {'rna': [True, False], 'prot': [False, True]}
        keys_in_mod = {m: [key in data.mod[m].var_names for key in keys] for m in data.mod}

        # .raw slots might have exclusive var_names
        if use_raw is None or use_raw:
            for i, k in enumerate(keys):
                for m in data.mod:
                    if keys_in_mod[m][i] == False and data.mod[m].raw is not None:
                        keys_in_mod[m][i] = k in data.mod[m].raw.var_names

        # e.g. color="rna:CD8A" - especially relevant for mdata.axis == -1
        mod_key_modifier: dict[str, str] = dict()
        for i, k in enumerate(keys):
            mod_key_modifier[k] = k
            for m in data.mod:
                if not keys_in_mod[m][i]:
                    k_clean = k
                    if k.startswith(f"{m}:"):
                        k_clean = k.split(":", 1)[1]

                    keys_in_mod[m][i] = k_clean in data.mod[m].var_names
                    if keys_in_mod[m][i]:
                        mod_key_modifier[k] = k_clean
                    if use_raw is None or use_raw:
                        if keys_in_mod[m][i] == False and data.mod[m].raw is not None:
                            keys_in_mod[m][i] = k_clean in data.mod[m].raw.var_names

        for m in data.mod:
            if np.sum(keys_in_mod[m]) > 0:
                mod_keys = np.array(keys)[keys_in_mod[m]]
                mod_keys = np.array([mod_key_modifier[k] for k in mod_keys])

                if use_raw is None or use_raw:
                    if data.mod[m].raw is not None:
                        keysidx = data.mod[m].raw.var.index.get_indexer_for(mod_keys)
                        fmod_adata = AnnData(
                            X=data.mod[m].raw.X[:, keysidx],
                            var=pd.DataFrame(index=mod_keys),
                            obs=data.mod[m].obs,
                        )
                    else:
                        if use_raw:
                            warnings.warn(
                                f"Attibute .raw is None for the modality {m}, using .X instead"
                            )
                        fmod_adata = data.mod[m][:, mod_keys]
                else:
                    fmod_adata = data.mod[m][:, mod_keys]

                if layer is not None:
                    if isinstance(layer, Dict):
                        m_layer = layer.get(m, None)
                        if m_layer is not None:
                            x = data.mod[m][:, mod_keys].layers[m_layer]
                            fmod_adata.X = x.todense() if issparse(x) else x
                            if use_raw:
                                warnings.warn(f"Layer='{layer}' superseded use_raw={use_raw}")
                    elif layer in data.mod[m].layers:
                        x = data.mod[m][:, mod_keys].layers[layer]
                        fmod_adata.X = x.todense() if issparse(x) else x
                        if use_raw:
                            warnings.warn(f"Layer='{layer}' superseded use_raw={use_raw}")
                    else:
                        warnings.warn(
                            f"Layer {layer} is not present for the modality {m}, using count matrix instead"
                        )
                x = fmod_adata.X.toarray() if issparse(fmod_adata.X) else fmod_adata.X
                obs = obs.join(
                    pd.DataFrame(x, columns=mod_keys, index=fmod_adata.obs_names),
                    how="left",
                )

        color = [mod_key_modifier[k] for k in keys]

    ad = AnnData(obs=obs, obsm=adata.obsm, obsp=adata.obsp, uns=adata.uns)
    retval = embedding(ad, basis=basis_mod, color=color, **kwargs)
    for key, col in zip(keys, color):
        try:
            adata.uns[f"{key}_colors"] = ad.uns[f"{col}_colors"]
        except KeyError:
            pass
    return retval