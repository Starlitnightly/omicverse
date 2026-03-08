"""Spatial segmentation plotting utilities for OmicVerse.

This module provides polygon-based plotting for spatial segmentation outputs.
"""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, List, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle


def _require_geopandas():
    try:
        import geopandas as gpd
        from shapely import wkt
    except ImportError as exc:
        raise ImportError(
            "`spatialseg` requires `geopandas` and `shapely`. "
            "Install with: pip install geopandas shapely"
        ) from exc
    return gpd, wkt


def _get_background_image(spatial_info: Mapping, img_key: Optional[str]):
    if img_key is None:
        images = spatial_info.get("images", {})
        img_key = "hires" if "hires" in images else None

    images = spatial_info.get("images", {})
    if img_key is None or img_key not in images:
        return None, None

    img = images[img_key]
    scale_key = f"tissue_{img_key}_scalef"
    scale_factor = spatial_info.get("scalefactors", {}).get(scale_key, 1.0)

    height, width = img.shape[:2]
    if scale_factor < 1.0:
        extent = [0, width / scale_factor, height / scale_factor, 0]
    else:
        extent = [0, width, height, 0]
    return img, extent


def _resolve_library_id(adata, library_id: Union[str, Sequence[str], None]) -> str:
    if "spatial" not in adata.uns:
        raise ValueError("`adata.uns['spatial']` is required but missing.")

    available = list(adata.uns["spatial"].keys())
    if isinstance(library_id, Sequence) and not isinstance(library_id, str):
        if len(library_id) == 0:
            raise ValueError("`library_id` sequence is empty.")
        if len(library_id) > 1:
            warnings.warn(
                f"`spatialseg` currently supports one panel library at a time. "
                f"Using the first library_id: '{library_id[0]}'."
            )
        library_id = library_id[0]

    if library_id is None:
        if not available:
            raise ValueError("No library_id found in `adata.uns['spatial']`.")
        library_id = available[0]
        if len(available) > 1:
            warnings.warn(
                f"Multiple library_ids found: {available}. Using '{library_id}'. "
                "Specify `library_id` explicitly to select a different one."
            )

    if library_id not in adata.uns["spatial"]:
        raise ValueError(
            f"`library_id` '{library_id}' not found in `adata.uns['spatial']`. "
            f"Available: {available}"
        )
    return library_id


def _normalize_color_keys(color: Optional[Union[str, List[str]]]) -> List[Optional[str]]:
    if color is None:
        return [None]
    if isinstance(color, str):
        return [color]
    return list(color)


def _validate_color_keys(adata, color_keys: Sequence[Optional[str]]) -> None:
    for key in color_keys:
        if key is None:
            continue
        if key in adata.obs.columns or key in adata.var_names:
            continue
        if hasattr(adata, "layers") and key in adata.layers:
            raise ValueError(
                f"`color` key '{key}' is in `adata.layers`, which is not supported yet."
            )
        raise ValueError(
            f"`color` key '{key}' not found in `adata.obs.columns` or `adata.var_names`."
        )


def _prepare_axes(
    color_keys: List[Optional[str]],
    ax: Optional[plt.Axes],
    figsize: Optional[tuple],
):
    if ax is not None:
        if len(color_keys) > 1:
            warnings.warn(
                "Multiple colors were provided with a single `ax`. "
                "Only the first color will be plotted."
            )
            color_keys = [color_keys[0]]
        return ax.figure, [ax], color_keys

    if figsize is None:
        figsize = (5 * len(color_keys), 5) if len(color_keys) > 1 else (10, 10)

    fig, axes = plt.subplots(1, len(color_keys), figsize=figsize, sharex=True, sharey=True)
    if len(color_keys) == 1:
        axes = [axes]
    return fig, list(axes), color_keys


def _resolve_filter_column(adata, color_key: Optional[str], groupby: Optional[str]) -> str:
    if groupby is not None:
        if groupby not in adata.obs.columns:
            raise ValueError(f"`groupby` column '{groupby}' not found in `adata.obs.columns`.")
        return groupby
    if color_key is not None and color_key in adata.obs.columns:
        return color_key
    raise ValueError(
        "`groups` requires either a categorical `color` in `adata.obs` or an explicit `groupby`."
    )


def _build_cell_mask(adata, color_key: Optional[str], groups: Optional[List[str]], groupby: Optional[str]):
    if groups is None:
        return pd.Series(True, index=adata.obs_names)
    filter_col = _resolve_filter_column(adata, color_key, groupby)
    return adata.obs[filter_col].isin(groups)


def _is_valid_geometry(geom) -> bool:
    if geom is None:
        return False
    if not hasattr(geom, "bounds"):
        return False
    try:
        b = geom.bounds
        if len(b) != 4 or not np.all(np.isfinite(b)):
            return False
        if b[2] <= b[0] or b[3] <= b[1]:
            return False
    except Exception:
        return False
    if hasattr(geom, "is_valid") and not geom.is_valid:
        return False
    return True


def _extract_geometries(gpd, wkt, adata, spatial_info, cell_ids: Sequence[str]):
    if "geometries" in spatial_info:
        source = spatial_info["geometries"]
        if "geometry" in getattr(source, "columns", []):
            source = source["geometry"]
        source = source[source.index.isin(cell_ids)]
        valid_index = [idx for idx in source.index if _is_valid_geometry(source.loc[idx])]
        return gpd.GeoSeries(source.loc[valid_index], index=valid_index)

    if "geometry" in adata.obs.columns:
        warnings.warn(
            "Using `adata.obs['geometry']` (WKT strings). "
            "For better performance, store geometries in `adata.uns['spatial'][library_id]['geometries']`."
        )
        geometries = []
        valid_index = []
        for cell_id in cell_ids:
            if cell_id not in adata.obs.index:
                continue
            value = adata.obs.loc[cell_id, "geometry"]
            if pd.isna(value):
                continue
            try:
                geom = wkt.loads(value)
            except Exception:
                continue
            if _is_valid_geometry(geom):
                geometries.append(geom)
                valid_index.append(cell_id)
        return gpd.GeoSeries(geometries, index=valid_index)

    raise ValueError(
        "Cell geometries not found. Expected either "
        "`adata.uns['spatial'][library_id]['geometries']` or `adata.obs['geometry']`."
    )


def _compute_bounds(geometries, adata, basis: str, mask: pd.Series):
    if len(geometries) > 0:
        b = np.array([geometries.loc[idx].bounds for idx in geometries.index])
        return float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())

    if basis in adata.obsm and mask.any():
        coords = adata.obsm[basis][mask.values]
        if len(coords) > 0:
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])

    return 0.0, 0.0, 1.0, 1.0


def _extract_color_series(adata, color_key: str, valid_cells: Sequence[str]) -> pd.Series:
    if color_key in adata.obs.columns:
        return adata.obs.loc[valid_cells, color_key]

    gene_idx = adata.var_names.get_loc(color_key)
    cell_idx = adata.obs_names.get_indexer(valid_cells)
    if hasattr(adata.X, "toarray"):
        values = adata.X[cell_idx, gene_idx].toarray().flatten()
    else:
        values = np.asarray(adata.X[cell_idx, gene_idx]).flatten()
    return pd.Series(values, index=valid_cells, name=color_key)


def _category_color_map(
    series: pd.Series,
    palette: Optional[Union[dict, list, np.ndarray]],
):
    if pd.api.types.is_categorical_dtype(series):
        categories = series.cat.categories.tolist()
    else:
        categories = sorted(series.dropna().unique().tolist())

    if palette is None:
        cmap = plt.get_cmap("tab20" if len(categories) <= 20 else "tab20b")
        return categories, {cat: cmap(i / max(len(categories), 1)) for i, cat in enumerate(categories)}

    if isinstance(palette, Mapping):
        missing = [cat for cat in categories if cat not in palette]
        if missing:
            warnings.warn(
                f"Palette mapping is missing {len(missing)} categories; using gray for them."
            )
        return categories, {cat: palette.get(cat, "gray") for cat in categories}

    palette_arr = np.asarray(palette)
    if len(palette_arr) == 0:
        raise ValueError("`palette` cannot be empty.")
    if len(palette_arr) < len(categories):
        warnings.warn(
            f"Palette has {len(palette_arr)} colors for {len(categories)} categories; colors will cycle."
        )
    return categories, {cat: palette_arr[i % len(palette_arr)] for i, cat in enumerate(categories)}


def _apply_axis_style(
    ax: plt.Axes,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    xlabel: Optional[str],
    ylabel: Optional[str],
    show_ticks: bool,
    force_show_ticks: bool,
    crop_coord: Optional[tuple[float, float, float, float]] = None,
    frameon: Optional[bool] = None,
):
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if crop_coord is None:
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.05 if x_range > 0 else 1.0
        y_pad = y_range * 0.05 if y_range > 0 else 1.0
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_max + y_pad, y_min - y_pad)
    else:
        ax.set_xlim(crop_coord[0], crop_coord[1])
        ax.set_ylim(crop_coord[3], crop_coord[2])

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if force_show_ticks:
        ax.tick_params(axis="both", which="major", labelsize=10)
    elif not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if frameon is False:
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])


def _resolve_crop_coord(
    crop_coord: tuple[int, int, int, int] | Sequence[tuple[int, int, int, int]] | None,
    panel_index: int,
) -> Optional[tuple[float, float, float, float]]:
    if crop_coord is None:
        return None

    if (
        isinstance(crop_coord, Sequence)
        and len(crop_coord) == 4
        and all(isinstance(v, (int, float)) for v in crop_coord)
    ):
        return float(crop_coord[0]), float(crop_coord[1]), float(crop_coord[2]), float(crop_coord[3])

    if isinstance(crop_coord, Sequence) and len(crop_coord) > 0:
        picked = crop_coord[min(panel_index, len(crop_coord) - 1)]
        if len(picked) != 4:
            raise ValueError("Each entry in `crop_coord` must be a 4-tuple: (left, right, top, bottom).")
        return float(picked[0]), float(picked[1]), float(picked[2]), float(picked[3])

    raise ValueError("`crop_coord` must be a 4-tuple or a sequence of 4-tuples.")


def spatialseg(
    adata: AnnData,
    *,
    color: Optional[Union[str, List[str]]] = None,
    groups: Optional[List[str]] = None,
    groupby: Optional[str] = None,
    basis: str = "spatial",
    img: np.ndarray | None = None,
    img_key: str | None = "hires",
    library_id: str | Sequence[str] | None = None,
    crop_coord: tuple[int, int, int, int] | Sequence[tuple[int, int, int, int]] | None = None,
    alpha_img: float = 0.5,
    bw: bool | None = False,
    size: float = 1.0,
    scale_factor: float | Sequence[float] | None = None,
    spot_size: float | None = None,
    na_color: str | tuple[float, ...] | None = None,
    cmap: str = "viridis",
    palette: Optional[Union[dict, list, np.ndarray]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    frameon: bool | None = None,
    linewidth: float = 0.5,
    edgecolor: str = "black",
    alpha: float = 0.8,
    xlabel: Optional[str] = "spatial 1",
    ylabel: Optional[str] = "spatial 2",
    show_ticks: bool = False,
    show: bool | None = None,
    return_fig: bool | None = None,
    save: str | Path | None = None,
    figsize: Optional[tuple] = None,
    edges_width: Optional[float] = None,
    edges_color: Optional[str] = None,
    **kwargs: Any,
):
    """Plot segmentation polygons in spatial coordinates using `ov.pl.spatial`-style arguments."""
    gpd, wkt = _require_geopandas()

    if edges_width is not None:
        linewidth = edges_width
    if edges_color is not None:
        edgecolor = edges_color
    if spot_size is not None:
        size = size * spot_size
    linewidth = linewidth * size

    if isinstance(scale_factor, Sequence) and not isinstance(scale_factor, str):
        scale_factor_val = float(scale_factor[0]) if len(scale_factor) > 0 else 1.0
    else:
        scale_factor_val = 1.0 if scale_factor is None else float(scale_factor)

    lib_id = _resolve_library_id(adata, library_id)
    spatial_info = adata.uns["spatial"][lib_id]

    color_keys = _normalize_color_keys(color)
    _validate_color_keys(adata, color_keys)
    fig, axes, color_keys = _prepare_axes(color_keys, ax, figsize)

    out_axes = []
    for i, color_key in enumerate(color_keys):
        if i >= len(axes):
            break
        cur_ax = axes[i]

        mask = _build_cell_mask(adata, color_key, groups, groupby)
        cells = adata.obs_names[mask].tolist()
        if len(cells) == 0:
            warnings.warn("No cells to plot after filtering.")
            out_axes.append(cur_ax)
            continue

        geometries = _extract_geometries(gpd, wkt, adata, spatial_info, cells)
        if len(geometries) == 0:
            warnings.warn(
                "No valid geometries found for selected cells. "
                "Try re-syncing geometry metadata after subsetting."
            )
            out_axes.append(cur_ax)
            continue

        x_min, y_min, x_max, y_max = _compute_bounds(geometries, adata, basis, mask)

        panel_crop = _resolve_crop_coord(crop_coord, i)
        panel_img = img
        panel_extent = None
        if panel_img is None:
            panel_img, panel_extent = _get_background_image(spatial_info, img_key)
        else:
            h, w = panel_img.shape[:2]
            if scale_factor_val < 1.0:
                panel_extent = [0, w / scale_factor_val, h / scale_factor_val, 0]
            else:
                panel_extent = [0, w, h, 0]

        if panel_img is not None and panel_extent is not None:
            cur_ax.imshow(
                panel_img,
                extent=panel_extent,
                origin="upper",
                alpha=1.0 if color_key is None else alpha_img,
                cmap="gray" if bw else None,
            )

        if color_key is None:
            _apply_axis_style(
                cur_ax,
                x_min,
                y_min,
                x_max,
                y_max,
                xlabel=xlabel,
                ylabel=ylabel,
                show_ticks=show_ticks,
                force_show_ticks=True,
                crop_coord=panel_crop,
                frameon=frameon,
            )
            out_axes.append(cur_ax)
            continue

        valid_cells = geometries.index.tolist()
        color_series = _extract_color_series(adata, color_key, valid_cells)
        plot_gdf = gpd.GeoDataFrame({color_key: color_series}, geometry=geometries, index=valid_cells)

        plot_kwargs = {
            "ax": cur_ax,
            "edgecolor": edgecolor,
            "linewidth": linewidth,
            "alpha": alpha,
            **kwargs,
        }

        is_categorical = not pd.api.types.is_numeric_dtype(plot_gdf[color_key])
        if is_categorical:
            categories, color_map = _category_color_map(plot_gdf[color_key], palette)
            face_colors = plot_gdf[color_key].map(color_map).fillna("gray" if na_color is None else na_color)
            plot_gdf.plot(color=face_colors, **plot_kwargs)

            if legend:
                handles = [Patch(facecolor=color_map[c], label=str(c)) for c in categories]
                if handles:
                    cur_ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
        else:
            plot_kwargs["column"] = color_key
            plot_kwargs["cmap"] = cmap if palette is None else ListedColormap(np.asarray(palette))
            plot_kwargs["legend"] = legend
            if na_color is not None:
                plot_kwargs["missing_kwds"] = {"color": na_color}
            if vmin is not None:
                plot_kwargs["vmin"] = vmin
            if vmax is not None:
                plot_kwargs["vmax"] = vmax
            plot_gdf.plot(**plot_kwargs)

        _apply_axis_style(
            cur_ax,
            x_min,
            y_min,
            x_max,
            y_max,
            xlabel=xlabel,
            ylabel=ylabel,
            show_ticks=show_ticks,
            force_show_ticks=False,
            crop_coord=panel_crop,
            frameon=frameon,
        )
        cur_ax.set_title(color_key)
        out_axes.append(cur_ax)

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    if return_fig:
        return fig

    if show:
        if ax is None:
            fig.tight_layout(rect=[0, 0, 0.95, 1])
        else:
            fig.tight_layout()
        plt.show()
        return None

    if len(out_axes) == 1:
        return out_axes[0]
    return out_axes


def highlight_spatial_region(
    ax: plt.Axes,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    edges_color: str = "red",
    edges_width: float = 1.0,
):
    """Draw a rectangular region of interest on an existing spatial axis."""
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    x_min, x_max = xlim
    y_min, y_max = ylim

    rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=edges_width,
        edgecolor=edges_color,
        facecolor="none",
    )
    ax.add_patch(rect)
    return rect


__all__ = [
    "spatialseg",
    "highlight_spatial_region",
]
