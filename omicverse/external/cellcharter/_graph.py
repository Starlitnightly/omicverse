from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

from ._utils import spatial_connectivity_key, spatial_distance_key


def remove_long_links(
    adata: AnnData,
    distance_percentile: float = 99.0,
    connectivity_key: str | None = None,
    distances_key: str | None = None,
    copy: bool = False,
) -> tuple[csr_matrix, csr_matrix] | None:
    """Trim graph edges whose distances exceed a percentile threshold."""
    connectivity_key = spatial_connectivity_key(connectivity_key)
    distances_key = spatial_distance_key(distances_key)

    if connectivity_key not in adata.obsp:
        raise KeyError(f"`{connectivity_key}` was not found in `adata.obsp`.")
    if distances_key not in adata.obsp:
        raise KeyError(f"`{distances_key}` was not found in `adata.obsp`.")

    conns = adata.obsp[connectivity_key].copy() if copy else adata.obsp[connectivity_key]
    dists = adata.obsp[distances_key].copy() if copy else adata.obsp[distances_key]

    positive = np.asarray(dists[dists != 0]).squeeze()
    if positive.size == 0:
        if copy:
            return conns, dists
        return None

    threshold = np.percentile(positive, distance_percentile)
    conns[dists > threshold] = 0
    dists[dists > threshold] = 0

    conns.eliminate_zeros()
    dists.eliminate_zeros()

    if copy:
        return conns, dists

    neighbors_key = "spatial_neighbors"
    if neighbors_key in adata.uns and "params" in adata.uns[neighbors_key]:
        adata.uns[neighbors_key]["params"]["radius"] = threshold
    return None
