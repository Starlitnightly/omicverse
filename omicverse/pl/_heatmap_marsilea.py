from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm, to_hex
from matplotlib.text import Text
from pandas.api.types import CategoricalDtype, is_numeric_dtype
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist

from ._scanpy_compat import _prepare_dataframe, default_palette, obs_df
from ._palette import palette_28, palette_56
from .._registry import register_function
from .._settings import Colors, EMOJI


def _import_marsilea():
    try:
        import marsilea as ma
        import marsilea.plotter as mp
    except ImportError as exc:
        raise ImportError(
            "marsilea package is required for ov.pl heatmap functions. "
            "Please install it with `pip install marsilea`."
        ) from exc
    return ma, mp


def _resolve_palette(n_colors):
    if n_colors <= len(palette_28):
        return palette_28[:n_colors]
    if n_colors <= len(palette_56):
        return palette_56[:n_colors]
    return default_palette(n_colors)[:n_colors]


def _resolve_group_colors(adata, groupby, categories):
    categories = [str(category) for category in categories]
    if isinstance(groupby, str):
        color_key = f"{groupby}_colors"
        if groupby in adata.obs and color_key in adata.uns:
            obs_values = adata.obs[groupby]
            if isinstance(obs_values.dtype, pd.CategoricalDtype):
                ordered = [str(value) for value in obs_values.cat.categories]
                if len(adata.uns[color_key]) >= len(ordered):
                    palette = dict(zip(ordered, adata.uns[color_key][: len(ordered)]))
                    if all(category in palette for category in categories):
                        return {category: palette[category] for category in categories}
        if color_key in adata.uns and len(adata.uns[color_key]) >= len(categories):
            return dict(zip(categories, adata.uns[color_key][: len(categories)]))
    return dict(zip(categories, _resolve_palette(len(categories))))


def _build_gene_groups(var_names):
    if var_names is None:
        return None, None
    if isinstance(var_names, str):
        return [var_names], None
    if isinstance(var_names, Mapping):
        ordered = []
        groups = {}
        for group, genes in var_names.items():
            genes = [genes] if isinstance(genes, str) else list(genes)
            groups[group] = genes
            ordered.extend(genes)
        return ordered, groups
    return list(var_names), None


def _normalize_group_order(values, valid_labels, name):
    if values is None:
        return None
    values = [values] if isinstance(values, str) else list(values)
    values = [str(value) for value in values]
    missing = [value for value in values if value not in valid_labels]
    if missing:
        raise ValueError(
            f"{name} contains labels not present in the heatmap: {missing}."
        )
    return values


def _resolve_lineage_order(values, requested=None):
    observed = [str(value) for value in pd.Series(values).dropna().astype(str)]
    observed_set = set(observed)
    if requested is not None:
        requested = [requested] if isinstance(requested, str) else list(requested)
        requested = [str(value) for value in requested]
        missing = [value for value in requested if value not in observed_set]
        if missing:
            raise ValueError(
                f"lineages contains labels not present in adata.obs: {missing}."
            )
        return requested

    series = pd.Series(values)
    if isinstance(series.dtype, pd.CategoricalDtype):
        ordered = [
            str(value) for value in series.cat.categories if str(value) in observed_set
        ]
        if ordered:
            return ordered
    return list(dict.fromkeys(observed))


def _resolve_reverse_lineages(reverse_ht, lineage_names):
    if reverse_ht is None:
        return set()

    items = (
        [reverse_ht]
        if np.isscalar(reverse_ht) and not isinstance(reverse_ht, str)
        else reverse_ht
    )
    if isinstance(reverse_ht, str):
        items = [reverse_ht]
    else:
        items = list(items)

    resolved = set()
    for item in items:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            position = int(item)
            if position == 0:
                if not lineage_names:
                    continue
                resolved.add(lineage_names[0])
                continue
            if 1 <= position <= len(lineage_names):
                resolved.add(lineage_names[position - 1])
                continue
            raise ValueError(
                f"reverse_ht index {position} is out of range for {len(lineage_names)} lineages."
            )

        lineage_name = str(item)
        if lineage_name not in lineage_names:
            raise ValueError(
                f"reverse_ht contains lineage '{lineage_name}' not present in the heatmap."
            )
        resolved.add(lineage_name)
    return resolved


def _draw_custom_legends(
    plotter,
    fig,
    *,
    side="right",
    pad=0.05,
    stack_by="col",
    stack_size=2,
    align_legends="left",
    align_stacks="baseline",
    legend_spacing=4,
    stack_spacing=5,
    box_padding=0.5,
):
    """Draw legends tightly alongside the heatmap body after render.

    Instead of using marsilea's built-in ``add_legends()`` (which positions
    legends in a separate layout cell and often leaves excess whitespace),
    this function collects all legend artists *after* the figure has been
    rendered and adds a new axes right next to the content.
    """
    try:
        from itertools import batched
    except ImportError:

        def batched(iterable, n):
            it = iter(iterable)
            while True:
                chunk = []
                try:
                    for _ in range(n):
                        chunk.append(next(it))
                except StopIteration:
                    if chunk:
                        yield chunk
                    return
                yield chunk

    from legendkit.layout import vstack as _vstack, hstack as _hstack

    legends = plotter.get_legends()
    user_legends = getattr(plotter, "_user_legends", {})
    if user_legends:
        for k, v in user_legends.items():
            legends[k] = [v()]

    all_legs = []
    for _name, legs in legends.items():
        if legs is not None:
            if not isinstance(legs, (list, tuple)):
                legs = [legs]
            all_legs.extend(legs)
    if not all_legs:
        return

    for leg in all_legs:
        try:
            leg.remove()
        except Exception:
            pass
        for attr in ("_parent_figure", "figure"):
            if hasattr(leg, attr):
                try:
                    setattr(leg, attr, None)
                except Exception:
                    pass

    content_x1 = 0.0
    content_y0, content_y1 = 1.0, 0.0
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        has_visual = bool(ax.images or ax.collections or ax.lines or ax.patches)
        if has_visual:
            bbox = ax.get_position()
            content_x1 = max(content_x1, bbox.x1)
            content_y0 = min(content_y0, bbox.y0)
            content_y1 = max(content_y1, bbox.y1)

    if content_x1 <= 0 or content_y1 <= content_y0:
        return

    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    temp_ax = fig.add_axes([0, 0, 1, 1])
    temp_ax.set_axis_off()

    inner, outer = _vstack, _hstack
    if stack_by == "row":
        inner, outer = outer, inner

    bboxes = []
    for legs_chunk in batched(all_legs, stack_size):
        box = inner(list(legs_chunk), align=align_legends, spacing=legend_spacing)
        bboxes.append(box)
    legend_box = outer(
        bboxes,
        ax=temp_ax,
        align=align_stacks,
        loc="center left",
        spacing=stack_spacing,
        padding=box_padding,
    )

    extent = legend_box.get_window_extent(renderer)
    fig_w, fig_h = fig.get_size_inches()
    dpi = fig.get_dpi()
    legend_width_frac = (extent.xmax - extent.xmin) / (fig_w * dpi)
    temp_ax.remove()

    pad_frac = pad / fig_w
    legend_x = content_x1 + pad_frac
    legend_width = min(legend_width_frac + 0.02, 0.35)
    legend_ax = fig.add_axes(
        [legend_x, content_y0, legend_width, content_y1 - content_y0]
    )
    legend_ax.set_axis_off()

    legends2 = plotter.get_legends()
    if user_legends:
        for k, v in user_legends.items():
            legends2[k] = [v()]
    all_legs2 = []
    for _name, legs in legends2.items():
        if legs is not None:
            if not isinstance(legs, (list, tuple)):
                legs = [legs]
            all_legs2.extend(legs)
    for leg in all_legs2:
        try:
            leg.remove()
        except Exception:
            pass
        for attr in ("_parent_figure", "figure"):
            if hasattr(leg, attr):
                try:
                    setattr(leg, attr, None)
                except Exception:
                    pass

    bboxes2 = []
    for legs_chunk in batched(all_legs2, stack_size):
        box = inner(list(legs_chunk), align=align_legends, spacing=legend_spacing)
        bboxes2.append(box)
    outer(
        bboxes2,
        ax=legend_ax,
        align=align_stacks,
        loc="center left",
        spacing=stack_spacing,
        padding=box_padding,
    )

    for text in legend_ax.findobj(match=Text):
        try:
            text.set_fontweight("normal")
        except Exception:
            pass


def _render_plot(plotter, save_path=None, show=False, legend_kws=None):
    existing_fignums = set(plt.get_fignums())
    fig = getattr(plotter, "figure", None)
    render_figure = fig if getattr(fig, "number", None) in existing_fignums else None
    plotter.render(figure=render_figure)
    fig = getattr(plotter, "figure", None)
    if fig is None:
        raise RuntimeError(
            "Marsilea plot rendering did not produce a matplotlib figure."
        )
    for fignum in list(set(plt.get_fignums()) - existing_fignums):
        if fignum != fig.number:
            plt.close(fignum)
    if legend_kws is not None:
        _draw_custom_legends(plotter, fig, **legend_kws)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _available_var_names(adata, use_raw):
    return adata.raw.var_names if use_raw else adata.var_names


def _dynamic_hvg_candidates(adata, use_raw):
    var = adata.raw.var if use_raw else adata.var
    var_names = adata.raw.var_names if use_raw else adata.var_names

    for key in ("highly_variable_features", "highly_variable"):
        if key not in var.columns:
            continue
        flag = pd.Series(var[key]).fillna(False).astype(bool).to_numpy()
        selected = list(var_names[flag])
        if selected:
            return selected
    return None


def _sanitize_var_names(adata, var_names, use_raw):
    available = _available_var_names(adata, use_raw)
    filtered = [gene for gene in var_names if gene in available]
    filtered = list(dict.fromkeys(filtered))
    if not filtered:
        raise ValueError(
            "No requested genes were found in the selected AnnData object."
        )
    return filtered


def _select_top_variable_features(adata, n_features):
    if adata.n_vars == 0:
        return []
    matrix = adata.X
    if hasattr(matrix, "power"):
        variances = np.asarray(
            matrix.power(2).mean(axis=0) - np.square(matrix.mean(axis=0))
        ).ravel()
    else:
        variances = np.var(np.asarray(matrix), axis=0)
    order = np.argsort(np.nan_to_num(variances, nan=-np.inf))[::-1]
    top_idx = order[: min(int(n_features), adata.n_vars)]
    return list(adata.var_names[top_idx])


def _extract_expression_frame(
    adata,
    var_names,
    *,
    use_raw=False,
    layer=None,
    gene_symbols=None,
):
    return obs_df(
        adata,
        keys=list(var_names),
        layer=layer,
        use_raw=use_raw,
        gene_symbols=gene_symbols,
    )


def _scale_frame(frame, standard_scale):
    if standard_scale == "obs":
        frame = frame.sub(frame.min(axis=1), axis=0)
        frame = frame.div(frame.max(axis=1).replace(0, np.nan), axis=0).fillna(0)
    elif standard_scale == "var":
        frame = frame.sub(frame.min(axis=0), axis=1)
        frame = frame.div(frame.max(axis=0).replace(0, np.nan), axis=1).fillna(0)
    elif standard_scale == "group":
        frame = frame.sub(frame.min(axis=1), axis=0)
        frame = frame.div(frame.max(axis=1).replace(0, np.nan), axis=0).fillna(0)
    elif standard_scale is not None:
        raise ValueError("standard_scale must be one of {'obs', 'var', 'group', None}.")
    return frame


def _scale_dynamic_frame(frame, standard_scale):
    """Scale dynamic heatmap matrices with features on rows and cells/bins on columns."""
    if standard_scale in {"var", "zscore"}:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1, ddof=0).replace(0, np.nan)
        frame = frame.sub(mean, axis=0)
        frame = frame.div(std, axis=0).fillna(0)
    elif standard_scale in {"group", "minmax"}:
        frame = frame.sub(frame.min(axis=1), axis=0)
        frame = frame.div(frame.max(axis=1).replace(0, np.nan), axis=0).fillna(0)
    elif standard_scale == "obs":
        frame = frame.sub(frame.min(axis=0), axis=1)
        frame = frame.div(frame.max(axis=0).replace(0, np.nan), axis=1).fillna(0)
    elif standard_scale not in {None, "raw"}:
        raise ValueError(
            "standard_scale must be one of {'var', 'zscore', 'group', 'minmax', 'obs', 'raw', None}."
        )
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    return frame


def _dynamic_display_limits(matrix, standard_scale):
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, None

    scale_mode = None if standard_scale is None else str(standard_scale).lower()
    if scale_mode in {"var", "zscore"}:
        quantiles = np.quantile(finite, [0.01, 0.99])
        bound = np.ceil(np.min(np.abs(quantiles)) * 2.0) / 2.0
        if not np.isfinite(bound) or bound <= 0:
            bound = float(np.nanmax(np.abs(finite)))
        return -bound, bound, 0

    q_low, q_high = np.quantile(finite, [0.01, 0.99])
    if not np.isfinite(q_low) or not np.isfinite(q_high):
        return None, None, None
    if q_low == q_high:
        eps = 1e-6 if q_low == 0 else abs(q_low) * 0.05
        q_low -= eps
        q_high += eps
    return float(q_low), float(q_high), None


def _suppress_plotter_legend(plotter):
    if plotter is None:
        return None
    plotter.get_legends = lambda: None
    return plotter


def _safe_anno_labels(mp, labels, **kwargs):
    """Create an AnnoLabels plotter with a fallback to Labels if the axis is
    too small during rendering (marsilea raises ValueError in that case)."""
    plotter = mp.AnnoLabels(labels, **kwargs)
    if not hasattr(plotter, "render_ax"):
        return plotter
    original_render_ax = plotter.render_ax

    def _safe_render_ax(spec):
        try:
            original_render_ax(spec)
        except (ValueError, Exception):
            try:
                spec.ax.set_axis_off()
            except Exception:
                pass

    plotter.render_ax = _safe_render_ax
    return plotter


def _add_dynamic_gene_group_strip(board, mp, gene_groups, gene_colors, *, show_legend):
    if not gene_groups:
        return
    legend_kws = None
    if show_legend:
        legend_kws = {
            "title": "Feature groups",
            "title_fontproperties": {"weight": "normal"},
        }
    board.add_left(
        mp.Colors(
            gene_groups,
            palette=gene_colors,
            legend_kws=legend_kws,
        ),
        size=0.15,
        pad=0.05,
        legend=show_legend,
    )


def _add_dynamic_row_labels(board, mp, label_names, *, direct_labels=False):
    if direct_labels:
        board.add_left(
            mp.Labels(label_names, align="right", fontsize=10),
            pad=0.03,
        )
        return
    if any(label_names) and len(label_names) >= 2:
        board.add_left(
            _safe_anno_labels(
                mp,
                _make_masked_labels(label_names),
                text_pad=0.48,
                text_gap=0.08,
                pointer_size=0.28,
                linewidth=0.5,
                connectionstyle="bar,fraction=-0.24",
                fontsize=8,
            ),
            pad=0.03,
        )
    elif any(label_names):
        board.add_left(
            mp.Labels(label_names, fontsize=8),
            pad=0.03,
        )


def _hide_plotter_axis(plotter):
    if plotter is None or not hasattr(plotter, "render"):
        return plotter

    original_render = plotter.render

    def _render_without_axis(axes):
        original_render(axes)
        if isinstance(axes, (list, tuple, np.ndarray)):
            for ax in np.ravel(axes):
                ax.set_axis_off()
        else:
            axes.set_axis_off()

    plotter.render = _render_without_axis
    return plotter


def _attach_post_render_hook(plotter, callback):
    if plotter is None or not hasattr(plotter, "render"):
        return plotter

    original_render = plotter.render

    def _render_with_callback(*args, **kwargs):
        result = original_render(*args, **kwargs)
        callback()
        return result

    plotter.render = _render_with_callback
    return plotter


def _style_dynamic_heatmap_legends(fig):
    if fig is None:
        return

    content_x0 = 1.0
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        has_visual = bool(ax.images or ax.collections or ax.lines or ax.patches)
        if has_visual:
            content_x0 = min(content_x0, ax.get_position().x0)

    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            title = legend.get_title()
            if title is not None:
                title.set_fontweight("normal")

        for getter, setter in (
            (ax.get_title, ax.set_title),
            (ax.get_xlabel, ax.set_xlabel),
            (ax.get_ylabel, ax.set_ylabel),
        ):
            value = getter()
            if value == "Feature groups":
                bbox = ax.get_position()
                if bbox.x1 <= content_x0 + 0.02:
                    setter("")

        for text in ax.texts:
            if text.get_text() == "Feature groups":
                x, _ = text.get_position()
                try:
                    text.set_fontweight("normal")
                except Exception:
                    pass
                if x <= 0:
                    text.set_visible(False)


def _find_main_heatmap_axis(fig, n_rows, n_cols):
    if fig is None:
        return None

    candidates = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        has_visual = bool(ax.collections or ax.images or ax.lines or ax.patches)
        if not has_visual:
            continue

        bbox = ax.get_position()
        area = float(bbox.width * bbox.height)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xspan = abs(float(x1) - float(x0))
        yspan = abs(float(y1) - float(y0))
        dim_penalty = abs(xspan - float(n_cols)) + abs(yspan - float(n_rows))
        candidates.append((dim_penalty, -area, ax))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _annotate_heatmap_values(
    fig,
    values,
    *,
    value_fmt=".2f",
    value_cutoff=0.0,
    use_abs_cutoff=False,
    fontsize=8,
    text_color="auto",
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
):
    if fig is None:
        return

    if isinstance(values, pd.DataFrame):
        matrix = values.to_numpy(dtype=float)
    else:
        matrix = np.asarray(values, dtype=float)

    if matrix.ndim != 2 or matrix.size == 0:
        return

    n_rows, n_cols = matrix.shape
    ax = _find_main_heatmap_axis(fig, n_rows, n_cols)
    if ax is None:
        return

    for text in list(ax.texts):
        if getattr(text, "gid", None) == "ov_heatmap_value":
            text.remove()

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return

    resolved_vmin = float(np.nanmin(finite_values)) if vmin is None else float(vmin)
    resolved_vmax = float(np.nanmax(finite_values)) if vmax is None else float(vmax)
    if resolved_vmin == resolved_vmax:
        eps = 1e-6 if resolved_vmin == 0 else abs(resolved_vmin) * 0.05
        resolved_vmin -= eps
        resolved_vmax += eps

    cmap_obj = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    if resolved_vmin < 0 < resolved_vmax:
        norm = TwoSlopeNorm(vmin=resolved_vmin, vcenter=0.0, vmax=resolved_vmax)
    else:
        norm = Normalize(vmin=resolved_vmin, vmax=resolved_vmax)

    cutoff = 0.0 if value_cutoff is None else float(value_cutoff)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            compare_value = abs(value) if use_abs_cutoff else value
            if compare_value < cutoff:
                continue
            label = format(float(value), value_fmt)
            if text_color == "auto":
                rgba = cmap_obj(norm(float(value)))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                current_text_color = "#F8F8F8" if luminance < 0.5 else "#1F1F1F"
            else:
                current_text_color = text_color
            text = ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=current_text_color,
            )
            text.set_gid("ov_heatmap_value")


def _style_heatmap_axes(fig, border_color="#2B2B2B", border_width=0.8):
    if fig is None:
        return

    for ax in fig.axes:
        has_plotted_content = bool(
            ax.collections or ax.images or ax.lines or ax.patches
        )
        if not has_plotted_content:
            continue

        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(border_width)
            spine.set_edgecolor(border_color)


def _make_masked_labels(labels):
    labels = np.asarray(labels, dtype=object)
    return np.ma.masked_where(labels == "", labels)


def _prepare_group_matrix(
    adata,
    var_names,
    groupby,
    *,
    layer=None,
    use_raw=False,
    gene_symbols=None,
    standard_scale=None,
):
    obs_tidy = obs_df(
        adata,
        keys=[groupby] + list(var_names),
        layer=layer,
        use_raw=use_raw,
        gene_symbols=gene_symbols,
    )

    if is_numeric_dtype(obs_tidy[groupby]):
        categories = pd.cut(obs_tidy[groupby], 7).astype("category")
    else:
        categories = obs_tidy[groupby].astype("category")

    expr = obs_tidy[list(var_names)].copy()
    expr.index = categories
    grouped = expr.groupby(level=0, observed=True).mean()
    grouped = _scale_frame(grouped, standard_scale)
    counts = categories.value_counts().reindex(grouped.index).fillna(0).astype(int)
    return grouped, counts


def _prepare_cell_level_matrix(
    adata,
    var_names,
    *,
    use_raw=False,
    layer=None,
    gene_symbols=None,
    cell_orderby=None,
    groupby=None,
    max_cells=None,
):
    expr = _extract_expression_frame(
        adata,
        var_names,
        use_raw=use_raw,
        layer=layer,
        gene_symbols=gene_symbols,
    )
    expr.index = adata.obs_names

    obs = adata.obs.copy()
    obs = obs.loc[expr.index]

    if groupby is not None and groupby not in obs.columns:
        raise ValueError(f"groupby '{groupby}' is not present in adata.obs.")
    if cell_orderby is not None and cell_orderby not in obs.columns:
        raise ValueError(f"cell_orderby '{cell_orderby}' is not present in adata.obs.")

    working = expr.copy()
    if groupby is not None:
        working[groupby] = obs[groupby].astype(str).values
    if cell_orderby is not None:
        working[cell_orderby] = obs[cell_orderby].values

    if groupby is not None and max_cells is not None:
        sampled = []
        for _, chunk in working.groupby(groupby, sort=False, observed=True):
            sampled.append(chunk.iloc[: int(max_cells)])
        working = pd.concat(sampled, axis=0)

    sort_columns = []
    if groupby is not None:
        sort_columns.append(groupby)
    if cell_orderby is not None:
        sort_columns.append(cell_orderby)
    if sort_columns:
        working = working.sort_values(sort_columns, kind="stable")

    metadata = pd.DataFrame(index=working.index)
    if groupby is not None:
        metadata[groupby] = working[groupby].astype(str)
    if cell_orderby is not None:
        metadata[cell_orderby] = working[cell_orderby]

    return working[list(var_names)], metadata


def _prepare_dynamic_matrix(
    adata,
    var_names,
    pseudotime_key,
    *,
    lineage_key=None,
    lineages=None,
    use_raw=False,
    layer=None,
    gene_symbols=None,
    n_bins=100,
    aggregate="mean",
    standard_scale="var",
    reverse_ht=None,
    use_cell_columns=True,
    annotation_keys=None,
):
    if pseudotime_key not in adata.obs:
        raise ValueError(f"pseudotime '{pseudotime_key}' is not present in adata.obs.")

    obs = adata.obs.copy()
    mask = obs[pseudotime_key].notna()
    if lineage_key is not None:
        if lineage_key not in obs:
            raise ValueError(
                f"lineage_key '{lineage_key}' is not present in adata.obs."
            )
        if lineages is not None:
            lineages = [lineages] if isinstance(lineages, str) else list(lineages)
            mask &= obs[lineage_key].isin(lineages)

    selected_cells = obs.index[mask]
    if len(selected_cells) == 0:
        raise ValueError("No cells remain after filtering dynamic heatmap inputs.")

    expr = _extract_expression_frame(
        adata[selected_cells].copy(),
        var_names,
        use_raw=use_raw,
        layer=layer,
        gene_symbols=gene_symbols,
    )
    expr.index = selected_cells

    meta = obs.loc[selected_cells, [pseudotime_key]].copy()
    if lineage_key is not None:
        meta[lineage_key] = obs.loc[selected_cells, lineage_key]
    if annotation_keys is not None:
        annotation_expr = None
        expression_annotation_keys = []
        available = set(_available_var_names(adata, use_raw))
        for key in annotation_keys:
            if key in obs.columns:
                meta[key] = obs.loc[selected_cells, key]
                continue
            if key in available:
                expression_annotation_keys.append(key)
                continue
            raise ValueError(
                f"cell_annotation '{key}' is not present in adata.obs or the selected expression matrix."
            )
        if expression_annotation_keys:
            annotation_expr = _extract_expression_frame(
                adata[selected_cells].copy(),
                expression_annotation_keys,
                use_raw=use_raw,
                layer=layer,
                gene_symbols=gene_symbols,
            )
            annotation_expr.index = selected_cells
            for key in expression_annotation_keys:
                meta[key] = annotation_expr[key]

    if use_cell_columns:
        if lineage_key is None:
            meta = meta.sort_values([pseudotime_key], kind="stable")
            expr = expr.loc[meta.index]
            matrix = expr.T
            metadata = pd.DataFrame(
                {
                    "cell": list(meta.index),
                    "pseudotime": pd.to_numeric(
                        meta[pseudotime_key], errors="coerce"
                    ).to_numpy(),
                },
                index=meta.index.astype(str),
            )
            matrix.columns = metadata.index
        else:
            lineage_names = _resolve_lineage_order(meta[lineage_key], lineages)
            reverse_lineages = _resolve_reverse_lineages(reverse_ht, lineage_names)
            chunks = []
            metadata_rows = []
            for lineage_name in lineage_names:
                lineage_meta = meta.loc[
                    meta[lineage_key].astype(str) == lineage_name
                ].copy()
                if lineage_meta.empty:
                    continue
                lineage_meta = lineage_meta.sort_values(
                    pseudotime_key,
                    ascending=lineage_name not in reverse_lineages,
                    kind="stable",
                )
                lineage_expr = expr.loc[lineage_meta.index]
                column_names = [f"{lineage_name}:{cell}" for cell in lineage_meta.index]
                lineage_expr.index = column_names
                chunks.append(lineage_expr.T)
                metadata_rows.extend(
                    dict(
                        {
                            "column": column_name,
                            "lineage": lineage_name,
                            "cell": cell_name,
                            "pseudotime": float(pt_value),
                        },
                        **{
                            key: lineage_meta.iloc[idx][key]
                            for key in (annotation_keys or [])
                        },
                    )
                    for idx, (column_name, cell_name, pt_value) in enumerate(
                        zip(
                            column_names,
                            lineage_meta.index,
                            lineage_meta[pseudotime_key].to_numpy(),
                        )
                    )
                )
            matrix = pd.concat(chunks, axis=1)
            metadata = pd.DataFrame(metadata_rows).set_index("column")
    elif lineage_key is None:
        meta = meta.sort_values([pseudotime_key], kind="stable")
        expr = expr.loc[meta.index]
        bins = pd.cut(meta[pseudotime_key], bins=n_bins, duplicates="drop")
        grouped = expr.groupby(bins, observed=True).agg(aggregate)
        column_labels = [f"bin_{i + 1}" for i in range(grouped.shape[0])]
        matrix = grouped.T
        matrix.columns = column_labels
        metadata = pd.DataFrame(
            {"bin": column_labels, "pseudotime": np.linspace(0, 1, len(column_labels))},
            index=column_labels,
        )
        if annotation_keys is not None:
            for key in annotation_keys:
                grouped_meta = meta.groupby(bins, observed=True)[key]
                if is_numeric_dtype(meta[key]):
                    metadata[key] = grouped_meta.mean().to_numpy()
                else:
                    metadata[key] = grouped_meta.agg(
                        lambda x: x.astype(str).mode().iloc[0] if len(x) else ""
                    ).to_numpy()
    else:
        lineage_names = _resolve_lineage_order(meta[lineage_key], lineages)
        reverse_lineages = _resolve_reverse_lineages(reverse_ht, lineage_names)
        chunks = []
        metadata_rows = []
        for lineage_name in lineage_names:
            lineage_meta = meta.loc[
                meta[lineage_key].astype(str) == lineage_name
            ].copy()
            if lineage_meta.empty:
                continue
            lineage_meta = lineage_meta.sort_values(
                pseudotime_key,
                ascending=lineage_name not in reverse_lineages,
                kind="stable",
            )
            lineage_expr = expr.loc[lineage_meta.index]
            bins = pd.cut(lineage_meta[pseudotime_key], bins=n_bins, duplicates="drop")
            grouped = lineage_expr.groupby(bins, observed=True).agg(aggregate)
            if lineage_name in reverse_lineages:
                grouped = grouped.iloc[::-1]
            labels = [f"{lineage_name}_bin_{i + 1}" for i in range(grouped.shape[0])]
            grouped.index = labels
            chunks.append(grouped.T)
            pseudotime_values = np.linspace(0, 1, len(labels))
            if lineage_name in reverse_lineages:
                pseudotime_values = pseudotime_values[::-1]
            annotation_values = {}
            if annotation_keys is not None:
                grouped_bins = lineage_meta.groupby(bins, observed=True)
                for key in annotation_keys:
                    if is_numeric_dtype(lineage_meta[key]):
                        annotation_values[key] = grouped_bins[key].mean().to_numpy()
                    else:
                        annotation_values[key] = (
                            grouped_bins[key]
                            .agg(
                                lambda x: x.astype(str).mode().iloc[0] if len(x) else ""
                            )
                            .to_numpy()
                        )
            metadata_rows.extend(
                {
                    "column": label,
                    "lineage": lineage_name,
                    "bin": idx + 1,
                    "pseudotime": pseudotime_values[idx],
                    **{
                        key: annotation_values[key][idx]
                        for key in (annotation_keys or [])
                    },
                }
                for idx, label in enumerate(labels)
            )
        matrix = pd.concat(chunks, axis=1)
        metadata = pd.DataFrame(metadata_rows).set_index("column")

    matrix = _scale_dynamic_frame(matrix, standard_scale)
    return matrix, metadata


def _compute_dynamic_feature_metadata(
    matrix, metadata, order_by="peak", cluster_features_by=None
):
    lineage_names = (
        list(dict.fromkeys(metadata["lineage"].astype(str)))
        if "lineage" in metadata.columns
        else ["global"]
    )
    feature_metadata = pd.DataFrame(index=matrix.index)

    if "lineage" not in metadata.columns:
        values = matrix.to_numpy()
        peak_idx = np.nanargmax(values, axis=1)
        valley_idx = np.nanargmin(values, axis=1)
        time_values = (
            metadata["pseudotime"].to_numpy(dtype=float)
            if "pseudotime" in metadata.columns
            else np.arange(values.shape[1], dtype=float)
        )
        feature_metadata["peak_time"] = time_values[peak_idx]
        feature_metadata["valley_time"] = time_values[valley_idx]
        feature_metadata["peak_lineage"] = "global"
        feature_metadata["cluster_time"] = (
            feature_metadata["peak_time"]
            if order_by == "peak"
            else feature_metadata["valley_time"]
        )
        return feature_metadata

    time_key = "peak_time" if order_by == "peak" else "valley_time"
    stat_func = np.nanargmax if order_by == "peak" else np.nanargmin

    for lineage in lineage_names:
        lineage_columns = metadata.index[metadata["lineage"].astype(str) == lineage]
        lineage_values = matrix.loc[:, lineage_columns].to_numpy()
        lineage_time = metadata.loc[lineage_columns, "pseudotime"].to_numpy(dtype=float)
        lineage_idx = stat_func(lineage_values, axis=1)
        feature_metadata[f"{lineage}_{time_key}"] = lineage_time[lineage_idx]

    lineage_time_cols = [f"{lineage}_{time_key}" for lineage in lineage_names]
    lineage_time = feature_metadata[lineage_time_cols].copy()

    if cluster_features_by is not None:
        cluster_lineage = str(cluster_features_by)
        if cluster_lineage not in lineage_names:
            raise ValueError(
                f"cluster_features_by '{cluster_features_by}' is not present in dynamic heatmap lineages."
            )
        feature_metadata["cluster_time"] = feature_metadata[
            f"{cluster_lineage}_{time_key}"
        ]
        feature_metadata["peak_lineage"] = cluster_lineage
    else:
        feature_metadata["cluster_time"] = lineage_time.min(axis=1)
        feature_metadata["peak_lineage"] = lineage_time.idxmin(axis=1).str.replace(
            f"_{time_key}", "", regex=False
        )

    return feature_metadata


def _compute_lineage_presence(matrix, metadata, lineage_names, presence_threshold=0.5):
    """Return a dict {lineage_name: bool_array} marking whether each gene is
    'active' in each lineage.  A gene is considered active in lineage L when
    its max expression across L's bins is at least ``presence_threshold``
    times its global max expression.  This mimics scop's ``is.na`` logic:
    in scop genes absent from a lineage have NA; here we approximate absence
    by low relative expression.
    """
    presence = {}
    global_max = matrix.abs().max(axis=1).replace(0, np.nan)
    for lineage_name in lineage_names:
        cols = metadata.index[metadata["lineage"].astype(str) == lineage_name]
        if len(cols) == 0:
            presence[lineage_name] = np.zeros(matrix.shape[0], dtype=bool)
            continue
        lin_max = matrix.loc[:, cols].abs().max(axis=1)
        presence[lineage_name] = (lin_max / global_max).fillna(
            0
        ).to_numpy() >= presence_threshold
    return presence


def _split_dynamic_features(
    matrix, feature_metadata, n_split=None, split_method="kmeans-peaktime"
):
    if n_split is None or n_split <= 1 or matrix.shape[0] <= 1:
        return None

    ordered_index = feature_metadata.sort_values(
        ["cluster_time", "peak_lineage"],
        kind="stable",
    ).index
    working = matrix.loc[ordered_index]
    split_method = str(split_method)

    if split_method not in {"kmeans", "kmeans-peaktime", "hclust", "hclust-peaktime"}:
        raise ValueError(
            "split_method must be one of {'kmeans', 'kmeans-peaktime', 'hclust', 'hclust-peaktime'}."
        )

    if split_method.startswith("kmeans"):
        if split_method.endswith("peaktime"):
            fit_data = feature_metadata.loc[ordered_index, ["cluster_time"]].to_numpy(
                dtype=float
            )
        else:
            fit_data = working.to_numpy()
        _, labels = kmeans2(
            fit_data,
            k=min(n_split, len(ordered_index)),
            minit="points",
            iter=20,
        )
    else:
        if split_method.endswith("peaktime"):
            fit_data = feature_metadata.loc[ordered_index, ["cluster_time"]].to_numpy(
                dtype=float
            )
        else:
            fit_data = working.to_numpy()
        if fit_data.shape[0] == 1:
            labels = np.array([0])
        else:
            tree = linkage(pdist(fit_data), method="average")
            from scipy.cluster.hierarchy import fcluster

            labels = (
                fcluster(tree, t=min(n_split, len(ordered_index)), criterion="maxclust")
                - 1
            )

    split_labels = pd.Series(
        [f"cluster_{int(label) + 1}" for label in labels], index=ordered_index
    )
    ordering = (
        feature_metadata.loc[ordered_index]
        .assign(split=split_labels)
        .sort_values(["split", "cluster_time", "peak_lineage"], kind="stable")
    )
    return split_labels.loc[ordering.index]


def _cluster_dynamic_rows(matrix, feature_metadata):
    if matrix.shape[0] <= 2:
        return matrix.index
    ordering = feature_metadata.sort_values(
        ["cluster_time", "peak_lineage"], kind="stable"
    ).index
    ordered_matrix = matrix.loc[ordering]
    distances = pdist(ordered_matrix.to_numpy())
    if len(distances) == 0:
        return ordering
    tree = linkage(distances, method="average")
    leaves = leaves_list(tree)
    return ordered_matrix.index[leaves]


def _continuous_colors(values, cmap_name="cividis"):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return ["#808080"] * len(values)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if vmax == vmin:
        normed = np.zeros(len(values))
    else:
        normed = (values - vmin) / (vmax - vmin)
    cmap = cm.get_cmap(cmap_name)
    return [to_hex(cmap(float(v))) for v in normed]


def _smooth_matrix_columns(matrix, window=15):
    if window is None or int(window) <= 1 or matrix.shape[1] <= 2:
        return matrix
    window = max(1, int(window))
    return matrix.T.rolling(window=window, center=True, min_periods=1).mean().T


def _smooth_dynamic_matrix(matrix, metadata, window=15):
    if window is None or int(window) <= 1:
        return matrix
    if "lineage" not in metadata.columns:
        return _smooth_matrix_columns(matrix, window=window)

    smoothed_blocks = []
    ordered_columns = []
    for _, meta_block in metadata.groupby("lineage", sort=False, observed=True):
        cols = list(meta_block.index)
        if not cols:
            continue
        smoothed_blocks.append(
            _smooth_matrix_columns(
                matrix.loc[:, cols], window=min(int(window), len(cols))
            )
        )
        ordered_columns.extend(cols)
    if not smoothed_blocks:
        return matrix
    smoothed = pd.concat(smoothed_blocks, axis=1)
    return smoothed.loc[:, ordered_columns]


def _fit_dynamic_matrix(matrix, metadata, window=31):
    if window is None or int(window) <= 1:
        return matrix
    window = max(3, int(window))
    if "lineage" not in metadata.columns:
        return _smooth_matrix_columns(matrix, window=window)

    fitted_blocks = []
    ordered_columns = []
    for lineage_name, meta_block in metadata.groupby(
        "lineage", sort=False, observed=True
    ):
        cols = list(meta_block.index)
        if not cols:
            continue
        block = matrix.loc[:, cols]
        fitted_block = _smooth_matrix_columns(
            block, window=min(window, max(3, len(cols) // 5 or 3))
        )
        fitted_blocks.append(fitted_block)
        ordered_columns.extend(cols)
    if not fitted_blocks:
        return matrix
    fitted = pd.concat(fitted_blocks, axis=1)
    return fitted.loc[:, ordered_columns]


def _rank_dynamic_splits(feature_metadata, split_labels):
    split_frame = (
        feature_metadata.loc[split_labels.index, ["cluster_time"]]
        .assign(split=split_labels.astype(str).values)
        .groupby("split", sort=False, observed=True)["cluster_time"]
        .mean()
        .sort_values(kind="stable")
    )
    return split_frame.index.tolist()


def _select_spaced_row_labels(row_names, n_labels, groups=None):
    if n_labels is None or int(n_labels) <= 0 or len(row_names) == 0:
        return [""] * len(row_names)

    total = len(row_names)
    n_labels = min(int(n_labels), total)
    labels = [""] * total

    if groups is None:
        positions = np.linspace(0, total - 1, num=n_labels, dtype=int)
        for idx in sorted(set(positions.tolist())):
            labels[idx] = row_names[idx]
        return labels

    groups = pd.Series(list(groups), index=np.arange(total))
    per_group_target = max(1, int(np.ceil(n_labels / max(groups.nunique(), 1))))
    chosen = []
    for _, indices in groups.groupby(groups, sort=False, observed=True).groups.items():
        indices = sorted(indices)
        local_n = min(per_group_target, len(indices))
        local_positions = np.linspace(0, len(indices) - 1, num=local_n, dtype=int)
        chosen.extend(indices[pos] for pos in sorted(set(local_positions.tolist())))
    if len(chosen) > n_labels:
        chosen = sorted(chosen)
        keep = np.linspace(0, len(chosen) - 1, num=n_labels, dtype=int)
        chosen = [chosen[i] for i in sorted(set(keep.tolist()))]
    chosen = sorted(chosen)
    min_gap = max(1, int(np.ceil(total / max(n_labels * 2, 1))))
    min_gap_floor = max(1, int(np.ceil(min_gap * 0.6)))
    while True:
        filtered = []
        for idx in chosen:
            if filtered and idx - filtered[-1] < min_gap:
                continue
            filtered.append(idx)
        if len(filtered) >= n_labels or min_gap <= min_gap_floor:
            chosen = filtered
            break
        min_gap -= 1
    for idx in chosen:
        labels[idx] = row_names[idx]
    return labels


def _labels_from_targets(row_names, targets, min_gap=1):
    labels = [""] * len(row_names)
    if not targets:
        return labels

    targets = set(targets)
    chosen = []
    for idx, name in enumerate(row_names):
        if name not in targets:
            continue
        if chosen and idx - chosen[-1] < max(1, int(min_gap)):
            continue
        chosen.append(idx)
    for idx in chosen:
        labels[idx] = row_names[idx]
    return labels


def _build_dynamic_annotation_palette(metadata, key, cmap_name):
    if is_numeric_dtype(metadata[key]):
        return cmap_name, True
    categories = list(metadata[key].astype(str))
    palette = dict(
        zip(
            list(dict.fromkeys(categories)),
            _resolve_palette(len(dict.fromkeys(categories))),
        )
    )
    return palette, False


def _make_dynamic_annotation_track(
    mp,
    values,
    *,
    label,
    palette=None,
    cmap_name="cividis",
    show_label=True,
):
    label_text = label if show_label else None
    label_loc = "left" if show_label else None
    label_props = {"fontsize": 10} if show_label else None
    series = pd.Series(values)

    if is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)[None, :]
        finite = numeric[np.isfinite(numeric)]
        vmin = vmax = None
        if finite.size > 0:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if vmin == vmax:
                delta = 1e-6 if vmin == 0 else abs(vmin) * 0.05
                vmin -= delta
                vmax += delta
        return mp.ColorMesh(
            numeric,
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            linecolor="none",
            label=label_text,
            label_loc=label_loc,
            label_props=label_props,
        )

    categories = list(series.astype(str))
    return mp.Colors(
        categories,
        palette=palette,
        label=label_text,
        label_loc=label_loc,
        label_props=label_props,
    )


def _resolve_dynamic_track_series(
    adata, lineage_meta, item, use_raw=False, layer=None, gene_symbols=None
):
    if item in lineage_meta.columns:
        return pd.Series(
            lineage_meta[item].to_numpy(), index=lineage_meta.index, name=item
        )
    if "cell" not in lineage_meta.columns:
        return None

    cells = lineage_meta["cell"].astype(str)
    if item in adata.obs.columns:
        return pd.Series(
            adata.obs.loc[cells, item].to_numpy(), index=lineage_meta.index, name=item
        )

    available = _available_var_names(adata, use_raw)
    if item in available:
        expr = _extract_expression_frame(
            adata[cells].copy(),
            [item],
            use_raw=use_raw,
            layer=layer,
            gene_symbols=gene_symbols,
        )
        expr.index = cells.to_numpy()
        return pd.Series(expr[item].to_numpy(), index=lineage_meta.index, name=item)
    return None


def _make_separate_track_plotter(
    mp,
    adata,
    lineage_meta,
    spec,
    *,
    use_raw=False,
    layer=None,
    gene_symbols=None,
    smooth_window=21,
    continuous_cmap="Reds",
    show_label=True,
    plot_type="auto",
):
    """Build a separate annotation track plotter.

    Parameters
    ----------
    plot_type : str
        One of ``"auto"``, ``"area"`` (stacked bar), ``"heatmap"``
        (ColorMesh), ``"line"`` (line plot overlay), ``"bar"`` (bar chart).
        ``"auto"`` picks area for categorical data and heatmap for numeric.
    """
    if isinstance(spec, str):
        items = [spec]
        track_label = f"{spec} (separate)"
    else:
        items = list(spec)
        track_label = f"{', '.join(str(x) for x in items)} (separate)"

    series_list = []
    for item in items:
        series = _resolve_dynamic_track_series(
            adata,
            lineage_meta,
            str(item),
            use_raw=use_raw,
            layer=layer,
            gene_symbols=gene_symbols,
        )
        if series is not None:
            series_list.append(series)
    if not series_list:
        return None, 0

    n_items = len(series_list)
    label = track_label if show_label else None
    label_props = {"fontsize": 9} if show_label else None

    is_categorical = len(series_list) == 1 and not is_numeric_dtype(series_list[0])
    if plot_type == "auto":
        plot_type = "area" if is_categorical else "heatmap"

    if plot_type == "area" and is_categorical:
        categories = series_list[0].astype(str)
        onehot = pd.get_dummies(categories)
        prop = onehot.T
        prop = (
            prop.T.rolling(
                window=min(max(3, int(smooth_window)), len(categories)),
                center=True,
                min_periods=1,
            )
            .mean()
            .T
        )
        prop = prop.div(prop.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
        palette_lookup = None
        if len(items) == 1 and items[0] in adata.obs.columns:
            palette_lookup = _resolve_group_colors(adata, items[0], list(prop.index))
        colors = (
            [palette_lookup[item] for item in prop.index]
            if palette_lookup is not None
            else _resolve_palette(prop.shape[0])
        )
        return _hide_plotter_axis(
            mp.StackBar(
                prop,
                items=list(prop.index),
                colors=colors,
                width=1.0,
                linewidth=0,
                edgecolor="none",
                antialiased=False,
                label=label,
                label_loc="left",
                label_props=label_props,
            )
        ), n_items

    if is_categorical:
        categories = series_list[0].astype(str)
        onehot = pd.get_dummies(categories)
        matrix = onehot.T.to_numpy().astype(float)
    else:
        matrix = np.vstack(
            [
                pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
                for s in series_list
            ]
        )
    if matrix.shape[1] > 2:
        matrix = (
            pd.DataFrame(matrix)
            .T.rolling(
                window=min(max(3, int(smooth_window)), matrix.shape[1]),
                center=True,
                min_periods=1,
            )
            .mean()
            .T.to_numpy()
        )

    if plot_type == "heatmap":
        return mp.ColorMesh(
            matrix,
            cmap=continuous_cmap,
            linewidth=0,
            label=label,
            label_loc="left",
            label_props=label_props,
        ), n_items
    elif plot_type == "line":
        names = (
            [s.name for s in series_list]
            if not is_categorical
            else list(pd.get_dummies(series_list[0].astype(str)).columns)
        )
        palette_lookup = None
        if is_categorical and len(items) == 1 and items[0] in adata.obs.columns:
            palette_lookup = _resolve_group_colors(adata, items[0], names)
        colors = (
            [palette_lookup[n] for n in names]
            if palette_lookup
            else _resolve_palette(matrix.shape[0])
        )
        try:
            return mp.Arc(
                matrix,
                colors=colors,
                items=names,
                label=label,
                label_loc="left",
                label_props=label_props,
            ), n_items
        except (TypeError, AttributeError):
            return mp.ColorMesh(
                matrix,
                cmap=continuous_cmap,
                linewidth=0,
                label=label,
                label_loc="left",
                label_props=label_props,
            ), n_items
    elif plot_type == "bar":
        if matrix.shape[0] == 1:
            values = matrix[0]
            return mp.Numbers(
                values,
                color="#4C78A8",
                label=label,
                label_props=label_props,
                show_value=False,
            ), n_items
        return mp.ColorMesh(
            matrix,
            cmap=continuous_cmap,
            linewidth=0,
            label=label,
            label_loc="left",
            label_props=label_props,
        ), n_items
    else:
        return mp.ColorMesh(
            matrix,
            cmap=continuous_cmap,
            linewidth=0,
            label=label,
            label_loc="left",
            label_props=label_props,
        ), n_items


def _score_dynamic_features(matrix, metadata, smooth_window=21):
    scores = pd.Series(0.0, index=matrix.index)
    if "lineage" in metadata.columns:
        groups = metadata.groupby("lineage", sort=False, observed=True)
    else:
        groups = [("global", metadata)]

    for _, meta_block in groups:
        cols = list(meta_block.index)
        if len(cols) < 3:
            continue
        time = pd.Series(meta_block["pseudotime"].to_numpy(dtype=float), index=cols)
        expr = _smooth_matrix_columns(matrix.loc[:, cols], window=smooth_window)
        block_score = []
        for gene in expr.index:
            values = pd.Series(expr.loc[gene].to_numpy(dtype=float), index=cols)
            corr = values.corr(time, method="spearman")
            corr = 0.0 if pd.isna(corr) else abs(float(corr))
            amplitude = float(values.max() - values.min())
            block_score.append(corr * amplitude)
        scores = np.maximum(scores, pd.Series(block_score, index=expr.index))
    return pd.Series(scores, index=matrix.index).sort_values(ascending=False)


def _filter_dynamic_candidates(dynamic_score):
    score = pd.Series(dynamic_score).replace([np.inf, -np.inf], np.nan).dropna()
    score = score[score > 0]
    if score.empty:
        return list(
            pd.Series(dynamic_score).replace([np.inf, -np.inf], np.nan).dropna().index
        )
    return list(score.index)


@register_function(
    aliases=["group_heatmap", "grouped_heatmap", "分组热图"],
    category="pl",
    description="Create a grouped expression heatmap using Marsilea.",
    examples=[
        "ov.pl.group_heatmap(adata, groupby='cell_type', var_names=['CD3D', 'CD79A'])",
        "ov.pl.group_heatmap(adata, groupby='leiden', var_names={'T': ['CD3D'], 'B': ['MS4A1']})",
    ],
    related=["pl.feature_heatmap", "pl.dynamic_heatmap", "pl.dotplot"],
)
def group_heatmap(
    adata: AnnData,
    var_names,
    groupby: str,
    *,
    figsize: tuple = (6, 10),
    layer: str = None,
    use_raw: bool = False,
    gene_symbols: str = None,
    standard_scale: str = None,
    cmap: str = "RdBu_r",
    row_cluster: bool = False,
    col_cluster: bool = False,
    row_split=None,
    col_split=None,
    left_color_bars: dict = None,
    left_color_labels: dict = None,
    right_color_bars: dict = None,
    right_color_labels: dict = None,
    col_color_bars: dict = None,
    legend: bool = True,
    legend_style: str = "tight",
    border: bool = False,
    label: str = "Mean expression",
    show_values: bool = False,
    value_fmt: str = ".2f",
    value_cutoff: float = 0.0,
    save: bool | str = False,
    save_pathway: str = "",
    show: bool = False,
):
    """Plot grouped mean expression as a Marsilea heatmap.

    Parameters
    ----------
    adata
        Annotated data matrix containing the expression values.
    var_names
        Features to plot. Accepts a list of genes, a single gene name, or a
        mapping of group label to gene list.
    groupby
        Key in ``adata.obs`` used to aggregate cells into groups.
    figsize
        Figure size ``(width, height)`` in inches.
    layer, use_raw, gene_symbols
        Expression source selection passed through to ``obs_df``.
    standard_scale
        Optional scaling mode. Supported values are ``'obs'``, ``'var'``, or
        ``None``.
    cmap
        Colormap used for the heatmap body.
    row_cluster, col_cluster
        Whether to add hierarchical dendrograms for rows or columns.
    row_split, col_split
        Optional explicit ordering of row or column groups.
    left_color_bars, left_color_labels, right_color_bars, right_color_labels
        Optional annotation palettes or labels for row-side annotations.
    col_color_bars
        Optional palette for grouped column annotations.
    legend, legend_style
        Legend visibility and layout strategy.
    border
        Whether to draw a visible border around heatmap axes after rendering.
    label
        Label used for the heatmap value legend.
    show_values
        Whether to print heatmap values inside cells.
    value_fmt
        Format string for printed values (e.g. ``'.2f'``).
    value_cutoff
        Only display text labels for values greater than or equal to this threshold.
    save, save_pathway, show
        Output controls. ``save`` may be ``True``/``False`` or a concrete path.

    Returns
    -------
    marsilea.Heatmap or matplotlib.figure.Figure
        Returns the Marsilea plotter by default, or the rendered figure when
        rendering is triggered for saving/showing/tight legends.
    """
    if not groupby:
        raise ValueError("Please provide `groupby` for group_heatmap.")
    if layer is not None:
        use_raw = False
    if use_raw and adata.raw is None:
        raise ValueError("use_raw=True was requested, but adata.raw is not available.")

    selected_var_names, grouped_var_names = _build_gene_groups(var_names)
    if selected_var_names is None:
        raise ValueError("Please provide `var_names` for group_heatmap.")
    selected_var_names = _sanitize_var_names(adata, selected_var_names, use_raw)

    ma, mp = _import_marsilea()
    grouped_expr, group_counts = _prepare_group_matrix(
        adata,
        selected_var_names,
        groupby,
        layer=layer,
        use_raw=use_raw,
        gene_symbols=gene_symbols,
        standard_scale=standard_scale,
    )

    heatmap = ma.Heatmap(
        grouped_expr,
        width=float(figsize[0]),
        height=float(figsize[1]),
        cmap=cmap,
        label=label,
    )

    row_labels = list(grouped_expr.index)
    col_labels = list(grouped_expr.columns)
    group_colors = _resolve_group_colors(adata, groupby, row_labels)

    left_colors = left_color_bars or group_colors
    left_labels = left_color_labels or row_labels
    right_labels = right_color_labels or None

    if isinstance(left_labels, Mapping):
        left_labels = [
            left_labels.get(label_name, label_name) for label_name in row_labels
        ]
    if isinstance(right_labels, Mapping):
        right_labels = [
            right_labels.get(label_name, label_name) for label_name in row_labels
        ]

    heatmap.add_left(
        mp.Colors(row_labels, palette=left_colors), size=0.2, pad=0.05, legend=False
    )
    heatmap.add_left(mp.Labels(left_labels, align="right", fontsize=12), pad=0.05)
    heatmap.add_right(
        mp.Numbers(
            group_counts.values,
            color="#EEB76B",
            label="Count",
            show_value=False,
            label_props={"size": 11},
            props={"size": 11},
        ),
        size=0.5,
        pad=0.05,
    )

    if right_color_bars is not None:
        heatmap.add_right(
            mp.Colors(row_labels, palette=right_color_bars),
            size=0.15,
            pad=0.05,
            legend=False,
        )
    if right_labels is not None:
        heatmap.add_right(mp.Labels(right_labels, fontsize=10), pad=0.05)

    if grouped_var_names is not None:
        gene_groups = []
        for group, genes in grouped_var_names.items():
            gene_groups.extend(
                [group] * len([gene for gene in genes if gene in col_labels])
            )
        ordered_groups = _normalize_group_order(
            col_split,
            list(dict.fromkeys(gene_groups)),
            "col_split",
        ) or list(dict.fromkeys(gene_groups))
        gene_group_colors = col_color_bars or dict(
            zip(ordered_groups, _resolve_palette(len(ordered_groups)))
        )
        heatmap.add_top(
            mp.Colors(gene_groups, palette=gene_group_colors), size=0.2, pad=0.05
        )
        heatmap.group_cols(gene_groups, order=ordered_groups)
    elif col_color_bars is not None:
        heatmap.add_top(
            mp.Colors(col_labels, palette=col_color_bars), size=0.2, pad=0.05
        )

    heatmap.add_bottom(mp.Labels(col_labels, rotation=90, fontsize=10), pad=0.05)

    if row_split is not None:
        row_order = _normalize_group_order(row_split, row_labels, "row_split")
        heatmap.group_rows(row_labels, order=row_order)
    if row_cluster:
        heatmap.add_dendrogram("right", pad=0.05, colors="#33A6B8")
    if col_cluster:
        heatmap.add_dendrogram("top", pad=0.05, colors="#B481BB")
    _legend_kws = None
    if legend:
        if legend_style == "tight":
            _legend_kws = dict(pad=0.05)
        else:
            heatmap.add_legends()
    if border:
        heatmap = _attach_post_render_hook(
            heatmap,
            lambda: _style_heatmap_axes(getattr(heatmap, "figure", None)),
        )
    if show_values:
        heatmap = _attach_post_render_hook(
            heatmap,
            lambda: _annotate_heatmap_values(
                getattr(heatmap, "figure", None),
                grouped_expr,
                value_fmt=value_fmt,
                value_cutoff=value_cutoff,
                use_abs_cutoff=False,
                cmap=cmap,
            ),
        )

    save_path = save if isinstance(save, str) else (save_pathway if save else None)
    fig = None
    if save_path or show or _legend_kws is not None:
        fig = _render_plot(
            heatmap, save_path=save_path, show=show, legend_kws=_legend_kws
        )

    return heatmap if fig is None else fig


@register_function(
    aliases=["feature_heatmap", "单细胞热图", "特征热图"],
    category="pl",
    description="Create a cell-level feature heatmap using Marsilea.",
    examples=[
        "ov.pl.feature_heatmap(adata, var_names=['CD3D', 'NKG7'], groupby='cell_type')",
        "ov.pl.feature_heatmap(adata, var_names=marker_genes, groupby='leiden', cell_orderby='dpt_pseudotime')",
    ],
    related=["pl.group_heatmap", "pl.dynamic_heatmap", "pl.marker_heatmap"],
)
def feature_heatmap(
    adata: AnnData,
    var_names,
    *,
    groupby: str = None,
    cell_orderby: str = None,
    max_cells: int = 100,
    figsize: tuple = (8, 4),
    layer: str = None,
    use_raw: bool = False,
    gene_symbols: str = None,
    standard_scale: str = "var",
    cmap: str = "RdBu_r",
    legend: bool = True,
    legend_style: str = "tight",
    border: bool = False,
    show_row_names: bool = True,
    show_column_names: bool = False,
    save: bool | str = False,
    save_pathway: str = "",
    show: bool = False,
):
    """Plot cell-level feature expression ordered by groups or metadata.

    Parameters
    ----------
    adata
        Annotated data matrix containing the expression values.
    var_names
        Features to plot. Accepts a list of genes, a single gene name, or a
        mapping of group label to gene list.
    groupby
        Optional categorical key in ``adata.obs`` used to color and group
        columns by cell identity.
    cell_orderby
        Optional continuous or categorical key in ``adata.obs`` used to order
        cells before plotting.
    max_cells
        Maximum number of cells retained per group when ``groupby`` is set.
    figsize
        Figure size ``(width, height)`` in inches.
    layer, use_raw, gene_symbols
        Expression source selection passed through to ``obs_df``.
    standard_scale
        Optional scaling mode. Supported values are ``'obs'``, ``'var'``,
        ``'group'``, or ``None``.
    cmap
        Colormap used for the heatmap body.
    legend, legend_style
        Legend visibility and layout strategy.
    border
        Whether to draw a visible border around heatmap axes after rendering.
    show_row_names, show_column_names
        Whether to display feature labels on rows or cell labels on columns.
    save, save_pathway, show
        Output controls. ``save`` may be ``True``/``False`` or a concrete path.

    Returns
    -------
    marsilea.Heatmap or matplotlib.figure.Figure
        Returns the Marsilea plotter by default, or the rendered figure when
        rendering is triggered for saving/showing/tight legends.
    """
    if layer is not None:
        use_raw = False
    if use_raw and adata.raw is None:
        raise ValueError("use_raw=True was requested, but adata.raw is not available.")

    selected_var_names, grouped_var_names = _build_gene_groups(var_names)
    if selected_var_names is None:
        raise ValueError("Please provide `var_names` for feature_heatmap.")
    selected_var_names = _sanitize_var_names(adata, selected_var_names, use_raw)

    ma, mp = _import_marsilea()
    expr, metadata = _prepare_cell_level_matrix(
        adata,
        selected_var_names,
        use_raw=use_raw,
        layer=layer,
        gene_symbols=gene_symbols,
        cell_orderby=cell_orderby,
        groupby=groupby,
        max_cells=max_cells,
    )
    expr = _scale_frame(expr, standard_scale)

    heatmap = ma.Heatmap(
        expr.T,
        width=max(float(figsize[0]) * 0.7, 3.0),
        height=max(float(figsize[1]) * 0.7, 2.5),
        cmap=cmap,
        label="Expression",
    )

    if groupby is not None:
        categories = list(metadata[groupby].astype(str))
        colors = _resolve_group_colors(adata, groupby, list(dict.fromkeys(categories)))
        heatmap.add_top(mp.Colors(categories, palette=colors), size=0.2, pad=0.05)
        heatmap.group_cols(categories, order=list(dict.fromkeys(categories)))

    if show_column_names:
        heatmap.add_bottom(
            mp.Labels(list(expr.index), rotation=90, fontsize=8), pad=0.05
        )
    if show_row_names:
        heatmap.add_left(
            mp.Labels(list(expr.columns), align="right", fontsize=10), pad=0.05
        )

    if grouped_var_names is not None:
        gene_groups = []
        for group, genes in grouped_var_names.items():
            gene_groups.extend(
                [group] * len([gene for gene in genes if gene in expr.columns])
            )
        if gene_groups:
            colors = dict(
                zip(
                    list(dict.fromkeys(gene_groups)),
                    _resolve_palette(len(dict.fromkeys(gene_groups))),
                )
            )
            heatmap.add_left(
                mp.Colors(gene_groups, palette=colors),
                size=0.15,
                pad=0.05,
                legend=False,
            )

    _legend_kws = None
    if legend:
        if legend_style == "tight":
            _legend_kws = dict(pad=0.05)
        else:
            heatmap.add_legends()
    if border:
        heatmap = _attach_post_render_hook(
            heatmap,
            lambda: _style_heatmap_axes(getattr(heatmap, "figure", None)),
        )

    save_path = save if isinstance(save, str) else (save_pathway if save else None)
    fig = None
    if save_path or show or _legend_kws is not None:
        fig = _render_plot(
            heatmap, save_path=save_path, show=show, legend_kws=_legend_kws
        )
    return heatmap if fig is None else fig


@register_function(
    aliases=["dynamic_heatmap", "伪时序热图", "动态热图"],
    category="pl",
    description="Python-native dynamic heatmap for pseudotime-ordered expression trends, with optional lineage-aware layouts and Marsilea annotations.",
    examples=[
        "ov.pl.dynamic_heatmap(adata, var_names=genes, pseudotime='dpt_pseudotime')",
        "ov.pl.dynamic_heatmap(adata, var_names=gene_modules, pseudotime='pt_via', cell_annotation='clusters')",
        "ov.pl.dynamic_heatmap(adata, var_names=genes, pseudotime='palantir_pseudotime', lineage_key='lineage', lineages=['B-cell'])",
        "ov.pl.dynamic_heatmap(adata, var_names=genes, pseudotime='pt_via', show_row_names=True, figsize=(4, 6))",
    ],
    related=[
        "pl.group_heatmap",
        "pl.feature_heatmap",
        "pl.cell_cor_heatmap",
        "pl.marker_heatmap",
    ],
)
def dynamic_heatmap(
    adata: AnnData,
    pseudotime: str,
    var_names=None,
    *,
    lineage_key: str = None,
    lineages=None,
    max_lineages: int = 2,
    top_features: int = None,
    cell_bins: int = 100,
    use_cell_columns: bool = True,
    use_fitted: bool = True,
    aggregate: str = "mean",
    smooth_window: int = 15,
    score_smooth_window: int = 21,
    fitted_window: int = 31,
    figsize: tuple = (8, 6),
    layer: str = None,
    use_raw: bool = False,
    gene_symbols: str = None,
    standard_scale: str = "var",
    cmap: str = "viridis",
    pseudotime_cmap: str = "cividis",
    order_by: str = "peak",
    reverse_ht=None,
    n_split: int = None,
    split_method: str = "kmeans-peaktime",
    cluster_features_by: str = None,
    row_cluster: bool = False,
    col_cluster: bool = False,
    show_row_names: bool = False,
    show_column_names: bool = False,
    pseudotime_label=None,
    cell_annotation=None,
    separate_annotation=None,
    separate_annotation_type: str = "auto",
    separate_smooth_window: int = 21,
    feature_labels=None,
    top_label_features: int = 10,
    legend: bool = True,
    legend_style: str = "tight",
    border: bool = False,
    save: bool | str = False,
    save_pathway: str = "",
    show: bool = False,
    verbose: bool = True,
):
    """Plot dynamic feature trends along pseudotime, optionally by lineage.

    Parameters
    ----------
    adata
        Annotated data matrix containing the expression values.
    var_names
        Features to plot. Accepts a list of genes, a single gene name, a
        mapping of group label to gene list, or ``None`` to start from all
        available features. When ``None``, the function prefers
        ``adata.var['highly_variable_features']`` or
        ``adata.var['highly_variable']`` as the candidate pool before
        falling back to all features.
    pseudotime
        Key in ``adata.obs`` containing the continuous ordering variable.
    lineage_key
        Optional key in ``adata.obs`` describing lineage membership.
    lineages
        Optional lineage subset to include. When provided, the displayed
        lineage order follows this sequence.
    max_lineages
        Maximum number of lineages retained after ordering.
    top_features
        Keep only the highest-scoring dynamic features after ranking. Dynamic
        scores are computed from smoothed pseudotime correlation multiplied by
        feature amplitude, so this parameter acts as a post-ranking cap rather
        than the initial feature selection pool. The printed feature count
        before fitting refers to the candidate pool before this cap is applied.
    cell_bins
        Number of bins used when aggregating cells along pseudotime.
    use_cell_columns
        Whether to keep columns at cell/bin resolution instead of grouped
        lineage blocks.
    use_fitted
        Whether to smooth/fitted the matrix before plotting.
    aggregate
        Aggregation function used when binning cells, for example ``'mean'``.
    smooth_window, score_smooth_window, fitted_window
        Window sizes used for display smoothing, dynamic feature scoring, and
        fitted trend estimation respectively.
    figsize
        Figure size ``(width, height)`` in inches.
    layer, use_raw, gene_symbols
        Expression source selection passed through to ``obs_df``.
    standard_scale
        Scaling mode. Supported values include ``'var'``, ``'zscore'``,
        ``'group'``, ``'minmax'``, ``'obs'``, ``'raw'``, or ``None``.
    cmap, pseudotime_cmap
        Colormaps for the heatmap body and pseudotime annotations.
    order_by
        Feature ordering strategy, e.g. ``'peak'`` or ``'valley'``.
    reverse_ht
        Optional lineage names or positions whose pseudotime direction should
        be reversed before plotting.
    n_split, split_method
        Controls for splitting dynamic features into blocks.
    cluster_features_by
        Optional metadata-driven feature clustering mode.
    row_cluster, col_cluster
        Whether to add hierarchical dendrograms for rows or columns.
    show_row_names, show_column_names
        Whether to show row or column labels directly on the plot. When
        ``show_row_names=True``, gene names are placed directly beside the
        heatmap without connector lines.
    pseudotime_label
        Optional custom label for the pseudotime annotation.
    cell_annotation
        Optional ``adata.obs`` key or keys to annotate cells/bins above the
        heatmap.
    separate_annotation, separate_annotation_type, separate_smooth_window
        Controls for auxiliary smoothed annotation tracks.
    feature_labels
        Explicit feature labels to force-display. When provided, these labels
        take priority over ``top_label_features``.
    top_label_features
        Target number of automatically selected feature labels to display when
        ``feature_labels`` is not provided and ``show_row_names=False``. The
        function tries to honor this count while still enforcing spacing
        between labels, so the final number may be slightly smaller in dense
        layouts.
    legend, legend_style
        Legend visibility and layout strategy.
    border
        Whether to draw a visible border around heatmap axes after rendering.
    save, save_pathway, show
        Output controls. ``save`` may be ``True``/``False`` or a concrete path.
    verbose
        Whether to print a short preparation/rendering summary.

    Returns
    -------
    marsilea.StackBoard or matplotlib.figure.Figure
        Returns the assembled Marsilea board by default. When rendering is
        triggered for saving, showing, or tight-legend export, returns the
        rendered matplotlib figure instead.
    """
    if layer is not None:
        use_raw = False
    if use_raw and adata.raw is None:
        raise ValueError("use_raw=True was requested, but adata.raw is not available.")

    if var_names is None:
        selected_var_names = _dynamic_hvg_candidates(adata, use_raw)
        if not selected_var_names:
            if use_raw:
                selected_var_names = list(adata.raw.var_names)
            else:
                selected_var_names = list(adata.var_names)
        grouped_var_names = None
    else:
        selected_var_names, grouped_var_names = _build_gene_groups(var_names)
        if selected_var_names is None:
            raise ValueError("Please provide `var_names` for dynamic_heatmap.")
        selected_var_names = _sanitize_var_names(adata, selected_var_names, use_raw)
    if verbose:
        print(
            f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Dynamic heatmap:{Colors.ENDC}"
        )
        print(
            f"   {Colors.CYAN}Candidate features: {Colors.BOLD}{len(selected_var_names)}{Colors.ENDC}"
        )
        print(f"   {Colors.CYAN}Pseudotime: {Colors.BOLD}{pseudotime}{Colors.ENDC}")
        if lineage_key is not None:
            print(
                f"   {Colors.CYAN}Lineage key: {Colors.BOLD}{lineage_key}{Colors.ENDC}"
            )
        if cell_annotation is not None:
            print(
                f"   {Colors.CYAN}Cell annotation: {Colors.BOLD}{cell_annotation}{Colors.ENDC}"
            )
        print(
            f"   {Colors.CYAN}use_fitted={Colors.BOLD}{use_fitted}{Colors.ENDC}{Colors.CYAN} | cell_bins={Colors.BOLD}{cell_bins}{Colors.ENDC}{Colors.CYAN} | cmap={Colors.BOLD}{cmap}{Colors.ENDC}"
        )

    ma, mp = _import_marsilea()
    annotation_keys = None
    if cell_annotation is not None:
        annotation_keys = (
            [cell_annotation]
            if isinstance(cell_annotation, str)
            else list(cell_annotation)
        )
    separate_specs = None
    if separate_annotation is not None:
        separate_specs = (
            list(separate_annotation)
            if isinstance(separate_annotation, (list, tuple))
            else [separate_annotation]
        )

    matrix, metadata = _prepare_dynamic_matrix(
        adata,
        selected_var_names,
        pseudotime,
        lineage_key=lineage_key,
        lineages=lineages,
        use_raw=use_raw,
        layer=layer,
        gene_symbols=gene_symbols,
        n_bins=cell_bins,
        aggregate=aggregate,
        standard_scale=standard_scale,
        reverse_ht=reverse_ht,
        use_cell_columns=use_cell_columns,
        annotation_keys=annotation_keys,
    )
    if (
        lineage_key is not None
        and "lineage" in metadata.columns
        and max_lineages is not None
    ):
        lineage_order = list(dict.fromkeys(metadata["lineage"].astype(str)))
        keep_lineages = lineage_order[: max(int(max_lineages), 1)]
        keep_columns = metadata.index[
            metadata["lineage"].astype(str).isin(keep_lineages)
        ]
        metadata = metadata.loc[keep_columns]
        matrix = matrix.loc[:, keep_columns]
    if verbose and lineage_key is not None and "lineage" in metadata.columns:
        lineage_summary = list(dict.fromkeys(metadata["lineage"].astype(str)))
        print(
            f"   {Colors.CYAN}Lineages: {Colors.BOLD}{', '.join(lineage_summary)}{Colors.ENDC}"
        )
    if use_fitted:
        matrix = _fit_dynamic_matrix(matrix, metadata, window=fitted_window)
    matrix = _smooth_dynamic_matrix(matrix, metadata, window=smooth_window)
    heatmap_vmin, heatmap_vmax, heatmap_center = _dynamic_display_limits(
        matrix, standard_scale
    )
    order_by = str(order_by).lower()
    if order_by not in {"peak", "peaktime", "valley", "valleytime"}:
        raise ValueError(
            "order_by must be one of {'peak', 'peaktime', 'valley', 'valleytime'}."
        )
    order_mode = "peak" if order_by in {"peak", "peaktime"} else "valley"

    feature_metadata = _compute_dynamic_feature_metadata(
        matrix,
        metadata,
        order_by=order_mode,
        cluster_features_by=cluster_features_by,
    )
    dynamic_score = None
    if top_features is not None or var_names is None:
        dynamic_score = _score_dynamic_features(
            matrix,
            metadata,
            smooth_window=score_smooth_window,
        )
        if var_names is None:
            dynamic_keep = _filter_dynamic_candidates(dynamic_score)
            if dynamic_keep:
                matrix = matrix.loc[dynamic_keep]
                feature_metadata = feature_metadata.loc[dynamic_keep]
                dynamic_score = dynamic_score.loc[dynamic_keep]
        if (
            top_features is not None
            and top_features > 0
            and matrix.shape[0] > top_features
        ):
            keep_features = dynamic_score.index[: int(top_features)]
            matrix = matrix.loc[keep_features]
            feature_metadata = feature_metadata.loc[keep_features]
    ordered_rows = feature_metadata.sort_values(
        ["cluster_time", "peak_lineage"],
        kind="stable",
    ).index
    matrix = matrix.loc[ordered_rows]
    feature_metadata = feature_metadata.loc[ordered_rows]

    row_split = _split_dynamic_features(
        matrix,
        feature_metadata,
        n_split=n_split,
        split_method=split_method,
    )
    if row_split is not None:
        matrix = matrix.loc[row_split.index]
        feature_metadata = feature_metadata.loc[row_split.index]

    if row_cluster:
        clustered_rows = _cluster_dynamic_rows(matrix, feature_metadata)
        matrix = matrix.loc[clustered_rows]
        feature_metadata = feature_metadata.loc[clustered_rows]
        if row_split is not None:
            row_split = row_split.loc[clustered_rows]

    label_names = [""] * matrix.shape[0]
    if feature_labels is not None:
        label_names = _labels_from_targets(
            list(matrix.index),
            feature_labels,
            min_gap=max(2, matrix.shape[0] // 45),
        )
    elif show_row_names:
        label_names = list(matrix.index)
    elif top_label_features and top_label_features > 0:
        label_names = _select_spaced_row_labels(
            list(matrix.index),
            int(top_label_features),
            groups=row_split if row_split is not None else None,
        )
    gene_groups = None
    gene_colors = None
    if grouped_var_names is not None:
        gene_groups = []
        for group, genes in grouped_var_names.items():
            gene_groups.extend(
                [group] * len([gene for gene in genes if gene in matrix.index])
            )
        if gene_groups:
            group_order = list(dict.fromkeys(gene_groups))
            gene_colors = None
            if annotation_keys is not None:
                for key in annotation_keys:
                    if key in adata.obs.columns:
                        obs_values = adata.obs[key]
                        categories = (
                            [str(value) for value in obs_values.cat.categories]
                            if isinstance(obs_values.dtype, pd.CategoricalDtype)
                            else list(dict.fromkeys(obs_values.astype(str)))
                        )
                        if set(group_order).issubset(set(categories)):
                            gene_colors = _resolve_group_colors(adata, key, group_order)
                            break
            if gene_colors is None:
                gene_colors = dict(zip(group_order, _resolve_palette(len(group_order))))

    split_order = None
    split_colors = None
    if row_split is not None:
        split_order = _rank_dynamic_splits(feature_metadata, row_split)
        split_colors = dict(zip(split_order, _resolve_palette(len(split_order))))

    peak_lineage_order = None
    peak_lineage_colors = None
    if "peak_lineage" in feature_metadata.columns:
        peak_lineages = list(feature_metadata["peak_lineage"].astype(str))
        peak_lineage_order = list(dict.fromkeys(peak_lineages))
        peak_lineage_colors = dict(
            zip(peak_lineage_order, _resolve_palette(len(peak_lineage_order)))
        )

    has_lineage_panels = lineage_key is not None and "lineage" in metadata.columns
    plotter = None
    panel_boards = None
    if has_lineage_panels:
        annotation_palettes = {}
        if annotation_keys is not None:
            for key in annotation_keys:
                ann_palette, is_continuous = _build_dynamic_annotation_palette(
                    metadata, key, pseudotime_cmap
                )
                if not is_continuous and key in adata.obs.columns:
                    categories = list(dict.fromkeys(metadata[key].astype(str)))
                    ann_palette = _resolve_group_colors(adata, key, categories)
                annotation_palettes[key] = (ann_palette, is_continuous)

        lineage_names = list(dict.fromkeys(metadata["lineage"].astype(str)))
        lineage_presence = _compute_lineage_presence(matrix, metadata, lineage_names)
        boards = []
        for idx, lineage_name in enumerate(lineage_names):
            lineage_cols = metadata.index[
                metadata["lineage"].astype(str) == lineage_name
            ]
            lineage_matrix = matrix.loc[:, lineage_cols]
            lineage_meta = metadata.loc[lineage_cols]
            width_ratio = lineage_matrix.shape[1] / max(matrix.shape[1], 1)
            board = ma.Heatmap(
                lineage_matrix,
                width=max(float(figsize[0]) * max(width_ratio, 0.28), 2.2),
                height=max(float(figsize[1]) * 0.78, 3.2),
                cmap=cmap,
                vmin=heatmap_vmin,
                vmax=heatmap_vmax,
                center=heatmap_center,
                label="Dynamic expression (z-score)"
                if idx == 0 and str(standard_scale).lower() in {"var", "zscore"}
                else ("Dynamic expression" if idx == 0 else None),
                legend=idx == 0,
            )

            if separate_specs is not None:
                for spec in separate_specs:
                    track_plotter, n_track_items = _make_separate_track_plotter(
                        mp,
                        adata,
                        lineage_meta,
                        spec,
                        use_raw=use_raw,
                        layer=layer,
                        gene_symbols=gene_symbols,
                        smooth_window=separate_smooth_window,
                        show_label=idx == 0,
                        plot_type=separate_annotation_type,
                    )
                    if track_plotter is not None:
                        if (
                            annotation_keys is not None
                            and isinstance(spec, str)
                            and spec in annotation_keys
                        ):
                            track_plotter = _suppress_plotter_legend(track_plotter)
                        sep_size = 0.18 * max(n_track_items, 1)
                        board.add_top(
                            track_plotter, size=sep_size, pad=0.02, legend=idx == 0
                        )
            if annotation_keys is not None:
                for key in annotation_keys:
                    ann_palette, is_continuous = annotation_palettes[key]
                    ann_values = lineage_meta[key]
                    board.add_top(
                        _make_dynamic_annotation_track(
                            mp,
                            ann_values,
                            label=key,
                            palette=ann_palette if not is_continuous else None,
                            cmap_name=ann_palette if is_continuous else pseudotime_cmap,
                            show_label=idx == 0,
                        ),
                        size=0.14,
                        pad=0.02,
                        legend=idx == 0,
                    )
            board.add_top(
                _make_dynamic_annotation_track(
                    mp,
                    lineage_meta["pseudotime"],
                    label="Pseudotime",
                    cmap_name=pseudotime_cmap,
                    show_label=idx == 0,
                ),
                size=0.16,
                pad=0.03,
                legend=idx == 0,
            )
            if show_column_names:
                board.add_bottom(
                    mp.Labels(
                        list(lineage_matrix.columns),
                        rotation=90,
                        fontsize=4 if use_cell_columns else 8,
                    ),
                    pad=0.05,
                )
            if idx == 0:
                if gene_groups:
                    _add_dynamic_gene_group_strip(
                        board,
                        mp,
                        gene_groups,
                        gene_colors,
                        show_legend=True,
                    )
                if row_split is not None:
                    board.add_left(
                        mp.Colors(list(row_split.astype(str)), palette=split_colors),
                        size=0.15,
                        pad=0.05,
                        legend=False,
                    )
                if row_cluster:
                    board.add_dendrogram("left", pad=0.05, colors="#33A6B8")
                _add_dynamic_row_labels(
                    board,
                    mp,
                    label_names,
                    direct_labels=show_row_names,
                )
            if row_split is not None:
                board.group_rows(list(row_split.astype(str)), order=split_order)
            if idx == len(lineage_names) - 1 and len(lineage_names) > 1:
                for lineage_idx, lineage_label in enumerate(lineage_names):
                    present_arr = lineage_presence[lineage_label]
                    presence_vals = np.where(present_arr, "present", "absent")
                    board.add_right(
                        mp.Colors(
                            presence_vals,
                            palette={"present": "#181830", "absent": "none"},
                            label=lineage_label,
                            label_loc="top",
                            label_props={"fontsize": 9, "rotation": 90},
                        ),
                        size=0.15,
                        pad=0.03 if lineage_idx == 0 else 0.0,
                        legend=False,
                    )
            if col_cluster:
                board.add_dendrogram("top", pad=0.05, colors="#B481BB")
            board.add_title(top=lineage_name, pad=0.03, fontsize=12)
            boards.append(board)
        panel_boards = boards
        plotter = ma.StackBoard(
            boards, direction="horizontal", spacing=0.06, keep_legends=False
        )
        _legend_kws = None
        if legend:
            if legend_style == "tight":
                _legend_kws = dict(
                    pad=0.05,
                    stack_size=2,
                    legend_spacing=4,
                    stack_spacing=5,
                    box_padding=0.5,
                )
            else:
                plotter.add_legends(
                    side="right",
                    pad=-0.2,
                    stack_by="col",
                    stack_size=2,
                    legend_spacing=4,
                    stack_spacing=5,
                    box_padding=0.5,
                )
    else:
        heatmap = ma.Heatmap(
            matrix,
            width=max(float(figsize[0]) * 0.7, 3.0),
            height=max(float(figsize[1]) * 0.7, 2.5),
            cmap=cmap,
            vmin=heatmap_vmin,
            vmax=heatmap_vmax,
            center=heatmap_center,
            label="Dynamic expression (z-score)"
            if str(standard_scale).lower() in {"var", "zscore"}
            else "Dynamic expression",
        )

        if separate_specs is not None:
            for spec in separate_specs:
                track_plotter, n_track_items = _make_separate_track_plotter(
                    mp,
                    adata,
                    metadata,
                    spec,
                    use_raw=use_raw,
                    layer=layer,
                    gene_symbols=gene_symbols,
                    smooth_window=separate_smooth_window,
                    show_label=True,
                    plot_type=separate_annotation_type,
                )
                if track_plotter is not None:
                    if (
                        annotation_keys is not None
                        and isinstance(spec, str)
                        and spec in annotation_keys
                    ):
                        track_plotter = _suppress_plotter_legend(track_plotter)
                    sep_size = 0.18 * max(n_track_items, 1)
                    heatmap.add_top(track_plotter, size=sep_size, pad=0.02)
        if annotation_keys is not None:
            for key in annotation_keys:
                ann_palette, is_continuous = _build_dynamic_annotation_palette(
                    metadata, key, pseudotime_cmap
                )
                if not is_continuous and key in adata.obs.columns:
                    categories = list(dict.fromkeys(metadata[key].astype(str)))
                    ann_palette = _resolve_group_colors(adata, key, categories)
                heatmap.add_top(
                    _make_dynamic_annotation_track(
                        mp,
                        metadata[key],
                        label=key,
                        palette=ann_palette if not is_continuous else None,
                        cmap_name=ann_palette if is_continuous else pseudotime_cmap,
                        show_label=True,
                    ),
                    size=0.14,
                    pad=0.02,
                )
        heatmap.add_top(
            _make_dynamic_annotation_track(
                mp,
                metadata["pseudotime"],
                label="Pseudotime",
                cmap_name=pseudotime_cmap,
                show_label=True,
            ),
            size=0.16,
            pad=0.03,
        )
        if gene_groups:
            _add_dynamic_gene_group_strip(
                heatmap,
                mp,
                gene_groups,
                gene_colors,
                show_legend=True,
            )
        if row_split is not None:
            heatmap.add_left(
                mp.Colors(list(row_split.astype(str)), palette=split_colors),
                size=0.15,
                pad=0.05,
                legend=False,
            )
            heatmap.group_rows(list(row_split.astype(str)), order=split_order)
        _add_dynamic_row_labels(
            heatmap,
            mp,
            label_names,
            direct_labels=show_row_names,
        )
        if show_column_names:
            heatmap.add_bottom(
                mp.Labels(
                    list(matrix.columns),
                    rotation=90,
                    fontsize=4 if use_cell_columns else 8,
                ),
                pad=0.05,
            )
        if peak_lineage_colors is not None:
            for lineage_idx, lineage_label in enumerate(peak_lineage_order):
                lineage_match = np.where(
                    feature_metadata["peak_lineage"].astype(str).to_numpy()
                    == lineage_label,
                    lineage_label,
                    "other",
                )
                heatmap.add_right(
                    mp.Colors(
                        lineage_match,
                        palette={
                            lineage_label: peak_lineage_colors[lineage_label],
                            "other": "#FFFFFF",
                        },
                    ),
                    size=0.08,
                    pad=0.03 if lineage_idx == 0 else 0.0,
                    legend=False,
                )

        if row_cluster:
            heatmap.add_dendrogram("left", pad=0.05, colors="#33A6B8")
        if col_cluster:
            heatmap.add_dendrogram("top", pad=0.05, colors="#B481BB")
        _legend_kws = None
        if legend:
            if legend_style == "tight":
                _legend_kws = dict(
                    pad=0.05,
                    stack_size=2,
                    legend_spacing=4,
                    stack_spacing=5,
                    box_padding=0.5,
                )
            else:
                heatmap.add_legends(
                    side="right",
                    pad=-0.15,
                    stack_by="col",
                    stack_size=2,
                    legend_spacing=4,
                    stack_spacing=5,
                    box_padding=0.5,
                )
        plotter = heatmap

    if pseudotime_label is not None:
        label_values = (
            [pseudotime_label]
            if np.isscalar(pseudotime_label)
            else list(pseudotime_label)
        )

        def _add_pseudotime_guides():
            if getattr(plotter, "_dynamic_pseudotime_guides_added", False):
                return
            if not hasattr(plotter, "get_main_ax") and not hasattr(plotter, "boards"):
                return

            if has_lineage_panels and panel_boards is not None:
                lineage_order = list(dict.fromkeys(metadata["lineage"].astype(str)))
                for board, lineage_name in zip(panel_boards, lineage_order):
                    if not hasattr(board, "get_main_ax"):
                        continue
                    ax = board.get_main_ax()
                    if ax is None:
                        continue
                    lineage_meta = metadata.loc[
                        metadata["lineage"].astype(str) == lineage_name
                    ]
                    panel_time = lineage_meta["pseudotime"].to_numpy(dtype=float)
                    if panel_time.size == 0:
                        continue
                    for value in label_values:
                        target = float(value)
                        xpos = int(np.argmin(np.abs(panel_time - target))) + 0.5
                        ax.axvline(
                            x=xpos,
                            color="#111111",
                            linestyle="--",
                            linewidth=0.8,
                            alpha=0.65,
                        )
            else:
                ax = plotter.get_main_ax()
                if ax is None:
                    return
                panel_time = metadata["pseudotime"].to_numpy(dtype=float)
                if panel_time.size == 0:
                    return
                for value in label_values:
                    target = float(value)
                    xpos = int(np.argmin(np.abs(panel_time - target))) + 0.5
                    ax.axvline(
                        x=xpos,
                        color="#111111",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.65,
                    )
            plotter._dynamic_pseudotime_guides_added = True

        plotter = _attach_post_render_hook(plotter, _add_pseudotime_guides)

    if legend or gene_groups:
        plotter = _attach_post_render_hook(
            plotter,
            lambda: _style_dynamic_heatmap_legends(getattr(plotter, "figure", None)),
        )

    if border:
        plotter = _attach_post_render_hook(
            plotter,
            lambda: _style_heatmap_axes(getattr(plotter, "figure", None)),
        )

    save_path = save if isinstance(save, str) else (save_pathway if save else None)
    fig = None
    if save_path or show or _legend_kws is not None:
        fig = _render_plot(
            plotter, save_path=save_path, show=show, legend_kws=_legend_kws
        )
    if verbose:
        print(
            f"\n{Colors.GREEN}{EMOJI['done']} Dynamic heatmap completed!{Colors.ENDC}"
        )
        print(
            f"   {Colors.GREEN}✓ Matrix shape: {Colors.BOLD}{matrix.shape[0]}{Colors.ENDC}{Colors.GREEN} features × {Colors.BOLD}{matrix.shape[1]}{Colors.ENDC}{Colors.GREEN} columns{Colors.ENDC}"
        )

    return plotter if fig is None else fig


@register_function(
    aliases=["cell_cor_heatmap", "细胞相关性热图", "cell_similarity"],
    category="pl",
    description="Python-native group similarity heatmap for within-dataset or cross-dataset annotation concordance analysis.",
    examples=[
        "ov.pl.cell_cor_heatmap(adata, group_by='cell_type', method='pearson')",
        "ov.pl.cell_cor_heatmap(adata, group_by='major_celltype', ref_adata=adata_ref, ref_group_by='label_transfer')",
        "ov.pl.cell_cor_heatmap(adata, group_by='leiden', show_values=True, value_cutoff=0.3)",
        "ov.pl.cell_cor_heatmap(adata, group_by='cell_type', method='cosine', figsize=(5, 4))",
    ],
    related=[
        "pl.group_heatmap",
        "pl.feature_heatmap",
        "pl.dynamic_heatmap",
        "pl.dotplot",
    ],
)
def cell_cor_heatmap(
    adata: AnnData,
    group_by: str,
    *,
    ref_adata: AnnData = None,
    ref_group_by: str = None,
    features=None,
    n_features: int = 2000,
    method: str = "pearson",
    layer: str = None,
    use_raw: bool = False,
    standard_scale: str = "var",
    cmap: str = "RdBu_r",
    figsize: tuple = (6, 6),
    show_values: bool = True,
    value_fmt: str = ".2f",
    value_cutoff: float = 0.0,
    row_cluster: bool = True,
    col_cluster: bool = True,
    vmin: float = None,
    vmax: float = None,
    legend: bool = True,
    legend_style: str = "tight",
    border: bool = False,
    save: bool | str = False,
    save_pathway: str = "",
    show: bool = False,
):
    """Compute pairwise correlation/similarity between cell groups and plot as heatmap.

    Computes group-level mean expression and plots the resulting similarity
    matrix.

    Parameters
    ----------
    adata
        Annotated data matrix.
    group_by
        Key in ``adata.obs`` for grouping cells (e.g. ``'cell_type'``).
    ref_adata
        Optional second AnnData for cross-dataset comparison.  If *None*,
        the query ``adata`` is used as both query and reference.
    ref_group_by
        Grouping key in ``ref_adata``.  Defaults to ``group_by``.
    features
        Specific features to use.  If *None*, highly-variable genes are used.
    n_features
        Number of top variable features to select when ``features`` is *None*.
    method
        Similarity metric: ``'pearson'``, ``'spearman'``, or ``'cosine'``.
    layer, use_raw
        Which expression slot to read.
    standard_scale
        ``'var'`` (per-gene) or ``'obs'`` (per-cell) z-scoring.
    cmap
        Colour map for the heatmap.
    figsize
        Figure size ``(width, height)`` in inches.
    show_values
        Whether to print correlation values inside cells.
    value_fmt
        Format string for printed values (e.g. ``'.2f'``).
    value_cutoff
        Only display text labels for cells with absolute similarity greater than
        or equal to this threshold.
    row_cluster, col_cluster
        Whether to hierarchically cluster rows / columns.
    vmin, vmax
        Colour limits.
    legend
        Show colour bar.
    save, save_pathway, show
        Output controls. ``save`` may be ``True``/``False`` or a concrete path.

    Returns
    -------
    marsilea.Heatmap or matplotlib.figure.Figure
        Returns the Marsilea plotter by default, or the rendered figure when
        rendering is triggered for saving/showing/tight legends.
    """
    ma, mp = _import_marsilea()

    if layer is not None:
        use_raw = False
    if use_raw and adata.raw is not None:
        expr_adata = adata.raw.to_adata()
    else:
        expr_adata = adata

    if features is None:
        if "highly_variable" in expr_adata.var.columns:
            features = list(expr_adata.var_names[expr_adata.var["highly_variable"]])
        if not features or features is None:
            features = _select_top_variable_features(expr_adata, n_features)
    features = [f for f in features if f in expr_adata.var_names][:n_features]

    def _aggregate(ad, key, feats, lyr):
        df = obs_df(ad, keys=[key] + feats, layer=lyr)
        return df.groupby(key).mean()

    query_agg = _aggregate(expr_adata, group_by, features, layer)
    if ref_adata is not None:
        ref_key = ref_group_by or group_by
        if use_raw and ref_adata.raw is not None:
            ref_expr = ref_adata.raw.to_adata()
        else:
            ref_expr = ref_adata
        shared = sorted(set(features) & set(ref_expr.var_names))
        ref_agg = _aggregate(ref_expr, ref_key, shared, layer)
        query_agg = query_agg.loc[:, shared]
    else:
        ref_agg = query_agg

    if standard_scale == "var":
        query_agg = (query_agg - query_agg.mean()) / query_agg.std().replace(0, 1)
        ref_agg = (ref_agg - ref_agg.mean()) / ref_agg.std().replace(0, 1)
    elif standard_scale == "obs":
        query_agg = query_agg.sub(query_agg.mean(axis=1), axis=0).div(
            query_agg.std(axis=1).replace(0, 1), axis=0
        )
        ref_agg = ref_agg.sub(ref_agg.mean(axis=1), axis=0).div(
            ref_agg.std(axis=1).replace(0, 1), axis=0
        )

    if method == "pearson":
        sim = query_agg.T.corrwith(ref_agg.T, method="pearson")
        sim_matrix = pd.DataFrame(
            np.corrcoef(query_agg.to_numpy(), ref_agg.to_numpy())[
                : len(query_agg), len(query_agg) :
            ],
            index=query_agg.index,
            columns=ref_agg.index,
        )
    elif method == "spearman":
        from scipy.stats import spearmanr

        combined = np.vstack([query_agg.to_numpy(), ref_agg.to_numpy()])
        corr, _ = spearmanr(combined, axis=1)
        nq = len(query_agg)
        sim_matrix = pd.DataFrame(
            corr[:nq, nq:],
            index=query_agg.index,
            columns=ref_agg.index,
        )
    elif method == "cosine":
        from scipy.spatial.distance import cdist

        dist = cdist(query_agg.to_numpy(), ref_agg.to_numpy(), metric="cosine")
        sim_matrix = pd.DataFrame(
            1 - dist,
            index=query_agg.index,
            columns=ref_agg.index,
        )
    else:
        raise ValueError(
            f"method must be 'pearson', 'spearman', or 'cosine', got '{method}'."
        )

    if row_cluster and sim_matrix.shape[0] > 2:
        from scipy.cluster.hierarchy import leaves_list as _ll, linkage as _lk

        tree = _lk(pdist(sim_matrix.to_numpy()), method="average")
        row_order = sim_matrix.index[_ll(tree)]
        sim_matrix = sim_matrix.loc[row_order]
    if col_cluster and sim_matrix.shape[1] > 2:
        from scipy.cluster.hierarchy import leaves_list as _ll, linkage as _lk

        tree = _lk(pdist(sim_matrix.to_numpy().T), method="average")
        col_order = sim_matrix.columns[_ll(tree)]
        sim_matrix = sim_matrix[col_order]

    heatmap = ma.Heatmap(
        sim_matrix,
        width=float(figsize[0]),
        height=float(figsize[1]),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        label=f"{method.capitalize()} correlation",
    )

    query_colors = _resolve_group_colors(adata, group_by, list(sim_matrix.index))
    heatmap.add_left(
        mp.Colors(list(sim_matrix.index), palette=query_colors),
        size=0.15,
        pad=0.05,
        legend=False,
    )
    heatmap.add_left(
        mp.Labels(list(sim_matrix.index), align="right", fontsize=10),
        pad=0.05,
    )
    ref_key_name = ref_group_by or group_by
    if ref_adata is not None:
        ref_colors = _resolve_group_colors(
            ref_adata, ref_key_name, list(sim_matrix.columns)
        )
    else:
        ref_colors = query_colors
    heatmap.add_top(
        mp.Colors(list(sim_matrix.columns), palette=ref_colors),
        size=0.15,
        pad=0.05,
        legend=False,
    )
    heatmap.add_top(
        mp.Labels(list(sim_matrix.columns), fontsize=10, rotation=90),
        pad=0.05,
    )

    if row_cluster and sim_matrix.shape[0] > 2:
        heatmap.add_dendrogram("left", pad=0.05, colors="#33A6B8")
    if col_cluster and sim_matrix.shape[1] > 2:
        heatmap.add_dendrogram("top", pad=0.05, colors="#B481BB")

    _legend_kws = None
    if legend:
        if legend_style == "tight":
            _legend_kws = dict(pad=0.05)
        else:
            heatmap.add_legends()
    if border:
        heatmap = _attach_post_render_hook(
            heatmap,
            lambda: _style_heatmap_axes(getattr(heatmap, "figure", None)),
        )
    if show_values:
        heatmap = _attach_post_render_hook(
            heatmap,
            lambda: _annotate_heatmap_values(
                getattr(heatmap, "figure", None),
                sim_matrix,
                value_fmt=value_fmt,
                value_cutoff=value_cutoff,
                use_abs_cutoff=True,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            ),
        )

    save_path = save if isinstance(save, str) else (save_pathway if save else None)
    fig = None
    if save_path or show or _legend_kws is not None:
        fig = _render_plot(
            heatmap, save_path=save_path, show=show, legend_kws=_legend_kws
        )
    return heatmap if fig is None else fig


def global_imports(modulename, shortname=None, asfunction=False):
    if shortname is None:
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:
        globals()[shortname] = __import__(modulename)
