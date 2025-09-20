import sys

import numpy as np

from ._spatial_knn import spatial_neighbours, sum_neighbours
from .filtering import filter_genes


def select_slide(adata, s, batch_key="sample"):
    r"""This function selects the data for one slide from the spatial anndata object.

    :param adata: Anndata object with multiple spatial experiments
    :param s: name of selected experiment
    :param batch_key: column in adata.obs listing experiment name for each location
    """

    slide = adata[adata.obs[batch_key].isin([s]), :].copy()
    s_keys = list(slide.uns["spatial"].keys())
    s_spatial = np.array(s_keys)[[s in k for k in s_keys]][0]

    slide.uns["spatial"] = {s_spatial: slide.uns["spatial"][s_spatial]}

    return slide


def list_imported_modules():
    for module in sys.modules:
        try:
            print(module, sys.modules[module].__version__)
        except Exception:
            try:
                if type(sys.modules[module].version) is str:
                    print(module, sys.modules[module].version)
                else:
                    print(module, sys.modules[module].version())
            except Exception:
                try:
                    print(module, sys.modules[module].VERSION)
                except Exception:
                    pass


__all__ = [
    "select_slide",
    "filter_genes",
    "spatial_neighbours",
    "sum_neighbours",
    "list_imported_modules",
]
