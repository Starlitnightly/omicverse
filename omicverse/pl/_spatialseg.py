"""Spatial segmentation plotting utilities for OmicVerse.

This module provides polygon-based visualization for spatial transcriptomics
cell-segmentation results.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects


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


def _process_background_image(spatial_info, img_key, data_coords_range=None):
    """Process background image with scanpy-like scaling behavior."""
    if img_key is None:
        img_key = "hires" if "hires" in spatial_info.get("images", {}) else None

    if not img_key or "images" not in spatial_info or img_key not in spatial_info["images"]:
        return None, None

    img = spatial_info["images"][img_key]
    scalefactors = spatial_info.get("scalefactors", {})
    scale_key = f"tissue_{img_key}_scalef"
    scale_factor = scalefactors.get(scale_key, 1.0)

    img_height, img_width = img.shape[:2]
    if scale_factor < 1.0:
        img_extent = [0, img_width / scale_factor, img_height / scale_factor, 0]
    else:
        img_extent = [0, img_width, img_height, 0]

    return img, img_extent


def spatialseg(
    adata,
    color: Optional[Union[str, List[str]]] = None,
    groups: Optional[List[str]] = None,
    groupby: Optional[str] = None,
    library_id: Optional[str] = None,
    size: float = 1.0,
    figsize: Optional[tuple] = None,
    cmap: str = "viridis",
    palette: Optional[Union[dict, list, np.ndarray]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    img_key: Optional[str] = None,
    basis: str = "spatial",
    edges_width: float = 0.5,
    edges_color: str = "black",
    alpha: float = 0.8,
    alpha_img: float = 0.5,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    legend_loc: str = "right margin",
    legend_fontsize: Optional[Union[int, float]] = 12,
    legend_fontweight: Union[str, int] = "bold",
    legend_fontoutline: Optional[float] = None,
    na_color: str = "lightgray",
    na_in_legend: bool = True,
    colorbar_loc: Optional[str] = "right",
    xlabel: Optional[str] = "spatial 1",
    ylabel: Optional[str] = "spatial 2",
    show_ticks: bool = False,
    **kwargs,
):
    """Plot spatial transcriptomics data with cell polygons instead of points.

    This implementation follows the TrackCell plotting logic, including robust
    geometry validation and retry behavior for invalid bounds.
    """
    _ = size  # kept for API compatibility with point-based plotting interfaces

    # Backward compatibility for historical usage without exposing a new API entry.
    if "color_dict" in kwargs:
        legacy_color_dict = kwargs.pop("color_dict")
        if legacy_color_dict is not None:
            if palette is not None:
                warnings.warn("Both `palette` and legacy `color_dict` were provided. Using `color_dict`.")
            palette = legacy_color_dict

    gpd, wkt = _require_geopandas()

    if "geometry" not in adata.obs.columns:
        raise ValueError(
            "Cell geometries not found. "
            "Expected `adata.obs['geometry']` containing WKT polygon strings, "
            "as produced by `ov.io.spatial.read_visium_hd_seg`."
        )

    if "spatial" not in adata.uns or not isinstance(adata.uns["spatial"], dict) or len(adata.uns["spatial"]) == 0:
        raise ValueError(
            "`spatialseg` requires non-empty `adata.uns['spatial']` and is strongly coupled to a selected library."
        )
    spatial_uns = adata.uns["spatial"]
    available_library_ids = list(spatial_uns.keys())

    if library_id is None:
        if len(available_library_ids) > 1:
            raise ValueError(
                f"Multiple library_ids found: {available_library_ids}. "
                "Please specify `library_id` explicitly."
            )
        library_id = available_library_ids[0]
    elif library_id not in spatial_uns:
        raise ValueError(
            f"`library_id` '{library_id}' not found in `adata.uns['spatial']`. "
            f"Available library_ids: {available_library_ids}"
        )
    spatial_info = spatial_uns[library_id]

    library_filter_mask = pd.Series(True, index=adata.obs_names)
    if library_id is not None:
        library_col = None
        for candidate in ("library_id", "fov", "FOV", "sample", "library", "library_key"):
            if candidate in adata.obs.columns:
                values = adata.obs[candidate].astype(str)
                if (values == str(library_id)).any():
                    library_col = candidate
                    library_filter_mask = values == str(library_id)
                    break
        if library_col is None and len(available_library_ids) > 1:
            raise ValueError(
                f"`library_id='{library_id}'` was provided but no obs column was found to subset cells "
                "(`library_id`/`fov`/`FOV`/`sample`/`library`). "
                "Please add a library indicator column in `adata.obs`."
            )

    if color is None:
        colors_to_plot = [None]
    elif isinstance(color, str):
        colors_to_plot = [color]
    else:
        colors_to_plot = color

    for color_key in colors_to_plot:
        if color_key is None:
            continue
        if color_key in adata.obs.columns or color_key in adata.var_names:
            continue
        if hasattr(adata, "layers") and color_key in adata.layers:
            raise ValueError(
                f"`color` key '{color_key}' found in `adata.layers`, but layers are not supported here."
            )
        raise ValueError(
            f"`color` key '{color_key}' not found in `adata.obs.columns` or `adata.var_names`."
        )

    if ax is None:
        if figsize is None:
            figsize = (5 * len(colors_to_plot), 5) if len(colors_to_plot) > 1 else (10, 10)

        if len(colors_to_plot) > 1:
            fig, axes = plt.subplots(1, len(colors_to_plot), figsize=figsize, sharex=True, sharey=True)
            if len(colors_to_plot) == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]
        if len(colors_to_plot) > 1:
            warnings.warn("Multiple colors specified but single ax provided. Only first color will be plotted.")
            colors_to_plot = [colors_to_plot[0]]

    axes_list = []

    for idx, color_key in enumerate(colors_to_plot):
        if idx >= len(axes):
            continue
        current_ax = axes[idx]

        if groups is not None:
            filter_column = None
            if groupby is not None:
                if groupby not in adata.obs.columns:
                    raise ValueError(f"`groupby` column '{groupby}' not found in `adata.obs.columns`.")
                filter_column = groupby
            elif color_key is not None and color_key in adata.obs.columns:
                filter_column = color_key
            else:
                raise ValueError(
                    "`groups` requires either:\n"
                    "  - `color` to be a categorical column in `adata.obs`, or\n"
                    "  - `groupby` to specify the column name for filtering."
                )
            mask = adata.obs[filter_column].isin(groups)
        else:
            mask = pd.Series(True, index=adata.obs_names)
        mask = mask & library_filter_mask

        cells_to_plot = adata.obs_names[mask]
        if len(cells_to_plot) == 0:
            warnings.warn("No cells to plot after filtering.")
            axes_list.append(current_ax)
            continue

        coords_list = []
        for cell_id in cells_to_plot:
            wkt_str = adata.obs.loc[cell_id, "geometry"] if cell_id in adata.obs.index else None
            if wkt_str and pd.notna(wkt_str):
                try:
                    geom = wkt.loads(wkt_str)
                    if hasattr(geom, "bounds"):
                        coords_list.append(geom.bounds)
                except Exception:
                    continue

        if coords_list:
            all_bounds = np.array(coords_list)
            x_min = all_bounds[:, 0].min()
            y_min = all_bounds[:, 1].min()
            x_max = all_bounds[:, 2].max()
            y_max = all_bounds[:, 3].max()
        else:
            if basis in adata.obsm and len(adata.obsm[basis]) > 0:
                spatial_coords = adata.obsm[basis][mask]
                if len(spatial_coords) > 0:
                    x_min, y_min = spatial_coords.min(axis=0)
                    x_max, y_max = spatial_coords.max(axis=0)
                else:
                    x_min = y_min = 0
                    x_max = y_max = 1
            else:
                x_min = y_min = 0
                x_max = y_max = 1

        data_coords_range = (x_min, y_min, x_max, y_max)

        img, img_extent = _process_background_image(spatial_info, img_key, data_coords_range)
        if img is not None and img_extent is not None:
            img_alpha = 1.0 if color_key is None else alpha_img
            current_ax.imshow(img, extent=img_extent, origin="upper", alpha=img_alpha)

        if color_key is None:
            if img_extent is not None:
                current_ax.set_xlim(img_extent[0], img_extent[1])
                current_ax.set_ylim(img_extent[2], img_extent[3])
            else:
                current_ax.set_xlim(x_min, x_max)
                current_ax.set_ylim(y_max, y_min)

            current_ax.set_aspect("equal")
            current_ax.invert_yaxis()

            if xlabel is not None:
                current_ax.set_xlabel(xlabel)
            if ylabel is not None:
                current_ax.set_ylabel(ylabel)

            current_ax.tick_params(axis="both", which="major", labelsize=10)
            axes_list.append(current_ax)
            continue

        geom_list = []
        valid_cells = []
        for cell_id in cells_to_plot:
            wkt_str = adata.obs.loc[cell_id, "geometry"] if cell_id in adata.obs.index else None
            if not wkt_str or pd.isna(wkt_str):
                continue
            try:
                geom = wkt.loads(wkt_str)
                if geom is None or not hasattr(geom, "bounds"):
                    continue
                if hasattr(geom, "is_valid") and not geom.is_valid:
                    continue
                bounds = geom.bounds
                if not all(np.isfinite(bounds)) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    continue
                geom_list.append(geom)
                valid_cells.append(cell_id)
            except Exception:
                continue

        if len(valid_cells) == 0:
            warnings.warn("No valid geometries found after filtering.")
            axes_list.append(current_ax)
            continue

        temp_geometries = gpd.GeoSeries(geom_list, index=valid_cells)

        if len(temp_geometries) == 0:
            warnings.warn("No valid geometries found for plotting.")
            axes_list.append(current_ax)
            continue

        try:
            test_gdf = gpd.GeoDataFrame(geometry=temp_geometries)
            bounds = test_gdf.total_bounds
            if not all(np.isfinite(bounds)) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                warnings.warn(
                    f"Invalid geometry bounds detected (bounds: {bounds}). "
                    "This may cause plotting errors."
                )
        except Exception as e:
            warnings.warn(f"Could not validate geometry bounds: {e}.")

        if color_key in adata.obs.columns:
            color_data = adata.obs.loc[valid_cells, color_key]
            temp_gdf = gpd.GeoDataFrame({color_key: color_data}, geometry=temp_geometries, index=valid_cells)
            plot_column = color_key
        else:
            gene_idx = adata.var_names.get_loc(color_key)
            if hasattr(adata.X, "toarray"):
                expression_values = adata.X[
                    adata.obs_names.get_indexer(valid_cells), gene_idx
                ].toarray().flatten()
            else:
                expression_values = adata.X[adata.obs_names.get_indexer(valid_cells), gene_idx]

            color_data = pd.Series(expression_values, index=valid_cells, name=color_key)
            temp_gdf = gpd.GeoDataFrame({color_key: color_data}, geometry=temp_geometries, index=valid_cells)
            plot_column = color_key

        max_retries = 2
        retry_count = 0
        while retry_count <= max_retries:
            try:
                bounds = temp_gdf.total_bounds
                if not all(np.isfinite(bounds)) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    if retry_count == 0:
                        warnings.warn(
                            f"Invalid geometry bounds detected (bounds: {bounds}). "
                            "Filtering out problematic geometries."
                        )

                    valid_mask = pd.Series(True, index=temp_gdf.index)
                    for idx2 in temp_gdf.index:
                        try:
                            geom = temp_gdf.loc[idx2, "geometry"]
                            if geom is None or pd.isna(geom):
                                valid_mask.loc[idx2] = False
                                continue
                            geom_bounds = geom.bounds
                            if (
                                not all(np.isfinite(geom_bounds))
                                or geom_bounds[2] <= geom_bounds[0]
                                or geom_bounds[3] <= geom_bounds[1]
                            ):
                                valid_mask.loc[idx2] = False
                        except Exception:
                            valid_mask.loc[idx2] = False

                    temp_gdf = temp_gdf[valid_mask]
                    if len(temp_gdf) == 0:
                        warnings.warn("No valid geometries remaining after filtering. Skipping plot.")
                        axes_list.append(current_ax)
                        break

                    valid_cells = temp_gdf.index.tolist()
                    bounds = temp_gdf.total_bounds
                    if not all(np.isfinite(bounds)) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                        retry_count += 1
                        continue
                    break
                break
            except Exception as e:
                if retry_count == 0:
                    warnings.warn(f"Error validating geometry bounds: {e}. Will use aspect='equal' fallback.")
                retry_count += 1
                continue

        use_equal_aspect = False
        if retry_count > max_retries:
            try:
                bounds = temp_gdf.total_bounds
                if not all(np.isfinite(bounds)) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    use_equal_aspect = True
            except Exception:
                use_equal_aspect = True

        is_categorical = temp_gdf[plot_column].dtype == bool or (
            not pd.api.types.is_numeric_dtype(temp_gdf[plot_column])
        )

        plot_kwargs = {
            "ax": current_ax,
            "edgecolor": edges_color,
            "linewidth": edges_width,
            "alpha": alpha,
            **kwargs,
        }

        if is_categorical:
            from ..utils._scatterplot import _add_categorical_legend, _color_vector, _get_palette

            color_source_vector = temp_gdf[plot_column]
            if (
                color_source_vector.dtype != bool
                and not isinstance(color_source_vector.dtype, pd.CategoricalDtype)
            ):
                color_source_vector = pd.Series(
                    pd.Categorical(color_source_vector),
                    index=temp_gdf.index,
                    name=plot_column,
                )
            if isinstance(color_source_vector.dtype, pd.CategoricalDtype):
                color_source_vector = color_source_vector.cat.remove_unused_categories()

            color_vector, _ = _color_vector(
                adata,
                plot_column,
                color_source_vector,
                palette=palette,
                na_color=na_color,
            )
            if isinstance(color_vector, pd.Categorical):
                face_colors = np.asarray(color_vector.astype(str), dtype=object)
            else:
                face_colors = np.asarray(color_vector, dtype=object)

            plot_kwargs["color"] = face_colors
            plot_kwargs["legend"] = False
            if use_equal_aspect:
                plot_kwargs["aspect"] = "equal"
            try:
                temp_gdf.plot(**plot_kwargs)
            except ValueError as e:
                if "aspect must be finite and positive" in str(e):
                    plot_kwargs["aspect"] = "equal"
                    temp_gdf.plot(**plot_kwargs)
                else:
                    raise

            if legend:
                if legend_fontoutline is not None:
                    path_effect = [
                        patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")
                    ]
                else:
                    path_effect = None
                try:
                    centroids = np.column_stack(
                        (
                            temp_gdf.geometry.centroid.x.to_numpy(),
                            temp_gdf.geometry.centroid.y.to_numpy(),
                        )
                    )
                except Exception:
                    bounds_df = temp_gdf.bounds
                    centroids = np.column_stack(
                        (
                            (bounds_df["minx"].to_numpy() + bounds_df["maxx"].to_numpy()) / 2.0,
                            (bounds_df["miny"].to_numpy() + bounds_df["maxy"].to_numpy()) / 2.0,
                        )
                    )
                _add_categorical_legend(
                    current_ax,
                    color_source_vector,
                    palette=_get_palette(adata, plot_column, palette=palette),
                    scatter_array=centroids,
                    legend_loc=legend_loc,
                    legend_fontweight=legend_fontweight,
                    legend_fontsize=legend_fontsize,
                    legend_fontoutline=path_effect,
                    na_color=na_color,
                    na_in_legend=na_in_legend,
                    multi_panel=len(colors_to_plot) > 1,
                )
        else:
            plot_kwargs["column"] = plot_column
            plot_kwargs["cmap"] = cmap
            plot_kwargs["legend"] = False
            if vmin is not None:
                plot_kwargs["vmin"] = vmin
            if vmax is not None:
                plot_kwargs["vmax"] = vmax
            if use_equal_aspect:
                plot_kwargs["aspect"] = "equal"
            try:
                temp_gdf.plot(**plot_kwargs)
            except ValueError as e:
                if "aspect must be finite and positive" in str(e):
                    plot_kwargs["aspect"] = "equal"
                    temp_gdf.plot(**plot_kwargs)
                else:
                    raise
            if legend and colorbar_loc is not None and len(current_ax.collections) > 0:
                try:
                    cb = plt.colorbar(
                        current_ax.collections[-1],
                        ax=current_ax,
                        pad=0.01,
                        fraction=0.08,
                        aspect=30,
                        location=colorbar_loc,
                    )
                except TypeError:
                    cb = plt.colorbar(
                        current_ax.collections[-1],
                        ax=current_ax,
                        pad=0.01,
                        fraction=0.08,
                        aspect=30,
                    )
                if legend_fontsize is not None:
                    cb.ax.tick_params(labelsize=legend_fontsize)

        current_ax.set_aspect("equal")
        current_ax.invert_yaxis()

        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * 0.05 if x_range > 0 else 1
        y_padding = y_range * 0.05 if y_range > 0 else 1

        current_ax.set_xlim(x_min - x_padding, x_max + x_padding)
        current_ax.set_ylim(y_max + y_padding, y_min - y_padding)

        if xlabel is not None:
            current_ax.set_xlabel(xlabel)
        if ylabel is not None:
            current_ax.set_ylabel(ylabel)

        if color_key is None:
            current_ax.tick_params(axis="both", which="major", labelsize=10)
        elif not show_ticks:
            current_ax.set_xticks([])
            current_ax.set_yticks([])

        if color_key:
            current_ax.set_title(color_key)

        axes_list.append(current_ax)

    if show:
        if ax is None:
            fig.tight_layout(rect=[0, 0, 0.95, 1])
        else:
            fig.tight_layout()
        plt.show()

    if len(axes_list) == 1:
        return axes_list[0]
    return axes_list


def highlight_spatial_region(
    ax: plt.Axes,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    edges_color: str = "red",
    edges_width: float = 1.0,
):
    """Mark a rectangular region on a spatial plot."""
    from matplotlib.patches import Rectangle

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


__all__ = ["spatialseg", "highlight_spatial_region"]
