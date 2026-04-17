from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from typing import TypeAlias
except ImportError:  # pragma: no cover - Python < 3.10
    TypeAlias = Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from anndata import AnnData
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm
from pandas.api.types import is_numeric_dtype


ColorLike: TypeAlias = Any
VBound: TypeAlias = Any
_FontWeight: TypeAlias = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize: TypeAlias = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]

doc_adata_color_etc = ""
doc_edges_arrows = ""
doc_scatter_embedding = ""
doc_scatter_spatial = ""
doc_show_save_ax = ""

additional_colors: Mapping[str, str] = {}

# Adapted from Scanpy public palettes.py
zeileis_28 = [
    "#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b",
    "#4a6fe3", "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a",
    "#11c638", "#8dd593", "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708",
    "#0fcfc0", "#9cded6", "#d5eae7", "#f3e1eb", "#f6c4e1", "#f79cd4",
    "#7f7f7f", "#c7c7c7", "#1CE6FF", "#336600",
]

default_102 = [
    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6",
    "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693", "#6A3A4C", "#1B4400", "#4FC601",
    "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA",
    "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F", "#372101",
    "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062", "#0CBD66",
    "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459",
    "#456648", "#0086ED", "#886F4C", "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F",
    "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7",
    "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625",
    "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98",
    "#A4E804", "#324E72",
]


class Empty:
    pass


_empty = Empty()


class _LoggerProxy:
    def __init__(self) -> None:
        self._logger = logging.getLogger("omicverse.pl")

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)


logg = _LoggerProxy()


@dataclass
class _FallbackSettings:
    autoshow: bool = True
    _frameon: bool = True
    _vector_friendly: bool = True
    verbosity: int = 0


settings = _FallbackSettings()


def _doc_params(**kwargs):
    def decorator(func):
        if func.__doc__:
            try:
                func.__doc__ = func.__doc__.format(**kwargs)
            except Exception:
                pass
        return func

    return decorator


def sanitize_anndata(adata: AnnData) -> None:
    if not isinstance(adata, AnnData) and not getattr(adata, '_is_oom', False):
        raise TypeError("Expected an AnnData object.")


def check_projection(projection: str) -> None:
    if projection not in {"2d", "3d"}:
        raise ValueError("`projection` must be '2d' or '3d'.")


def check_colornorm(vmin=None, vmax=None, vcenter=None, norm=None):
    if norm is not None:
        return norm
    if vcenter is not None:
        return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    if vmin is not None or vmax is not None:
        return Normalize(vmin=vmin, vmax=vmax)
    return None


def circles(x, y, *, s, ax: Axes, scale_factor=None, **kwargs):
    size = np.asarray(s)
    if scale_factor is not None:
        size = size * float(scale_factor) ** 2
    marker = kwargs.pop("marker", "o")
    return ax.scatter(x, y, s=size, marker=marker, **kwargs)


def savefig_or_show(basename: str, *, show=None, save=None) -> None:
    should_show = settings.autoshow if show is None else show
    if save not in (None, False):
        if isinstance(save, str):
            path = Path(save)
        else:
            path = Path(f"{basename}.png")
        plt.savefig(path, bbox_inches="tight")
    if should_show:
        plt.show()


def plot_edges(ax: Axes, adata: AnnData, basis: str, edges_width: float, edges_color, neighbors_key=None) -> None:
    try:
        coords = adata.obsm[f"X_{basis}" if f"X_{basis}" in adata.obsm else basis]
    except Exception:
        return

    connectivities_key = "connectivities"
    if neighbors_key is not None:
        connectivities_key = adata.uns.get(neighbors_key, {}).get("connectivities_key", connectivities_key)

    graph = adata.obsp.get(connectivities_key)
    if graph is None:
        return
    coo = graph.tocoo()
    mask = coo.row < coo.col
    rows = coo.row[mask]
    cols = coo.col[mask]
    if len(rows) == 0:
        return
    segments = np.stack([coords[rows, :2], coords[cols, :2]], axis=1)
    ax.add_collection(LineCollection(segments, colors=edges_color, linewidths=edges_width, alpha=0.3))


def plot_arrows(ax: Axes, adata: AnnData, basis: str, arrows_kwds=None) -> None:
    logg.warning("Arrow overlays are not implemented in the local Scanpy compatibility layer; skipping.")


def default_palette(count: int) -> list[str]:
    return default_102 if count > 28 else zeileis_28


def _resolve_feature_index(
    var: pd.DataFrame,
    key: str,
    *,
    gene_symbols: str | None,
) -> int:
    if key in var.index:
        return int(var.index.get_loc(key))
    if gene_symbols is not None and gene_symbols in var.columns:
        matches = np.flatnonzero(var[gene_symbols].to_numpy() == key)
        if len(matches) > 0:
            return int(matches[0])
    raise KeyError(f"Key {key!r} not found in adata.obs, adata.var_names, or adata.var[{gene_symbols!r}].")


def obs_df(
    adata: AnnData,
    keys,
    *,
    layer: str | None = None,
    gene_symbols: str | None = None,
    use_raw: bool = False,
) -> pd.DataFrame:
    if isinstance(keys, str):
        keys = [keys]
    if use_raw and layer is not None:
        raise ValueError("Cannot specify use_raw=True and a layer at the same time.")

    source = adata.raw.to_adata() if use_raw and adata.raw is not None else adata
    matrix = source.X if layer is None else adata.layers[layer]
    frame = pd.DataFrame(index=adata.obs_names)

    for key in keys:
        if key in adata.obs.columns:
            frame[key] = adata.obs[key]
            continue

        idx = _resolve_feature_index(source.var, key, gene_symbols=gene_symbols)
        values = matrix[:, idx]
        if issparse(values):
            values = values.toarray().ravel()
        else:
            values = np.asarray(values).reshape(-1)
        frame[key] = values

    return frame


def _prepare_dataframe(
    adata: AnnData,
    var_names,
    *,
    groupby: str,
    use_raw: bool = False,
    log: bool = False,
    num_categories: int = 7,
    layer=None,
    gene_symbols=None,
):
    keys = [groupby] + list(var_names)
    obs_tidy = obs_df(adata, keys=keys, layer=layer, use_raw=use_raw, gene_symbols=gene_symbols)
    group = obs_tidy[groupby]
    if is_numeric_dtype(group):
        categorical = pd.cut(group, num_categories)
    else:
        categorical = group.astype("category")
    obs_tidy = obs_tidy.drop(columns=[groupby]).set_index(categorical)
    if log:
        obs_tidy = np.log1p(obs_tidy)
    categories = getattr(obs_tidy.index, "categories", pd.Index(obs_tidy.index.unique()))
    return categories, obs_tidy


def ranking(adata: AnnData, attr: str, key: str, *, n_points: int = 30, labels: str = "PC", log: bool = False):
    values = np.asarray(getattr(adata, attr)[key])[:n_points]
    x = np.arange(1, len(values) + 1)
    plt.scatter(x, values)
    plt.plot(x, values)
    if log:
        plt.yscale("log")
    plt.xlabel("ranking")
    plt.ylabel(key)
    plt.xticks(x, [f"{labels}{i}" for i in x], rotation=90)


def _calc_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.stats import gaussian_kde

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    min_z = float(np.min(z))
    max_z = float(np.max(z))
    if max_z == min_z:
        return np.zeros_like(z, dtype=float)
    return (z - min_z) / (max_z - min_z)


def embedding_density(
    adata: AnnData,
    basis: str = "umap",
    *,
    groupby: str | None = None,
    key_added: str | None = None,
    components: str | Sequence[str] | None = None,
) -> None:
    sanitize_anndata(adata)
    basis = basis.lower()
    if basis == "fa":
        basis = "draw_graph_fa"

    obsm_key = f"X_{basis}"
    if obsm_key not in adata.obsm:
        raise ValueError(
            f"Cannot find the embedded representation `adata.obsm[{obsm_key!r}]`. Compute the embedding first."
        )

    if components is None:
        components = "1,2"
    if isinstance(components, str):
        components = components.split(",")
    components = np.asarray(components).astype(int) - 1
    if len(components) != 2:
        raise ValueError("Please specify exactly 2 components, or `None`.")
    if basis == "diffmap":
        components += 1

    if groupby is not None:
        if groupby not in adata.obs:
            raise ValueError(f"Could not find {groupby!r} `.obs` column.")
        if adata.obs[groupby].dtype.name != "category":
            raise ValueError(f"{groupby!r} column does not contain categorical data")

    density_covariate = key_added or (f"{basis}_density_{groupby}" if groupby is not None else f"{basis}_density")
    coords = adata.obsm[obsm_key]

    if groupby is not None:
        density_values = np.zeros(adata.n_obs, dtype=float)
        for category in adata.obs[groupby].cat.categories:
            mask = adata.obs[groupby] == category
            embed_x = coords[mask, components[0]]
            embed_y = coords[mask, components[1]]
            density_values[mask] = _calc_density(embed_x, embed_y)
        adata.obs[density_covariate] = density_values
    else:
        adata.obs[density_covariate] = _calc_density(coords[:, components[0]], coords[:, components[1]])

    if basis != "diffmap":
        components += 1
    adata.uns[f"{density_covariate}_params"] = {
        "covariate": groupby,
        "components": components.tolist(),
    }


def plot_violin(
    adata: AnnData,
    *,
    keys,
    groupby: str | None = None,
    ax: Axes | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    gene_symbols: str | None = None,
    **kwargs,
) -> Axes:
    import seaborn as sns

    key_list = [keys] if isinstance(keys, str) else list(keys)
    data_keys = key_list if groupby is None else [groupby] + key_list
    data = obs_df(
        adata,
        keys=data_keys,
        layer=layer,
        use_raw=use_raw,
        gene_symbols=gene_symbols,
    )

    if ax is None:
        _, ax = plt.subplots()

    if groupby is None:
        if len(key_list) == 1:
            sns.violinplot(y=data[key_list[0]], ax=ax, **kwargs)
        else:
            melted = data[key_list].melt(var_name="feature", value_name="value")
            sns.violinplot(data=melted, x="feature", y="value", ax=ax, **kwargs)
    elif len(key_list) == 1:
        sns.violinplot(data=data, x=groupby, y=key_list[0], ax=ax, **kwargs)
    else:
        melted = data.melt(id_vars=[groupby], value_vars=key_list, var_name="feature", value_name="value")
        sns.violinplot(data=melted, x=groupby, y="value", hue="feature", ax=ax, **kwargs)
    return ax
