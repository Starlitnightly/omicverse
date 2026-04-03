from __future__ import annotations

import re
import textwrap
import warnings
from typing import Mapping, Sequence, Literal

import anndata
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path
from pandas.api.types import is_numeric_dtype
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from .._registry import register_function
from ._heatmap import _attach_post_render_hook, _import_marsilea, _render_plot, _style_heatmap_axes
from ._palette import palette_28, palette_56, palette_112


def _to_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _pick_first_available(frame: pd.DataFrame, candidates: Sequence[str], fallback):
    fallback_series = pd.Series(fallback, index=frame.index, dtype=object)
    resolved = pd.Series(np.nan, index=frame.index, dtype=object)
    for key in candidates:
        if key not in frame.columns:
            continue
        values = frame[key]
        valid = values.notna() & values.astype(str).str.strip().ne("")
        candidate_values = values.where(valid)
        resolved = resolved.where(resolved.notna(), candidate_values)
    resolved = resolved.where(resolved.notna(), fallback_series)
    return resolved.astype(str)


def _ensure_comm_adata(adata: anndata.AnnData) -> None:
    missing_obs = [key for key in ("sender", "receiver") if key not in adata.obs.columns]
    missing_layers = [key for key in ("means", "pvalues") if key not in adata.layers]

    if missing_obs:
        raise ValueError(
            "Communication AnnData is missing required obs columns: "
            f"{missing_obs}. Expected at least 'sender' and 'receiver'."
        )
    if missing_layers:
        raise ValueError(
            "Communication AnnData is missing required layers: "
            f"{missing_layers}. Expected both 'means' and 'pvalues'."
        )


def _normalize_use_arg(values) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _is_provided(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict, pd.Index, np.ndarray)):
        return len(value) > 0
    return True


def _raise_for_unsupported_arguments(plot_type: str, **arguments) -> None:
    unsupported = [name for name, value in arguments.items() if _is_provided(value)]
    if not unsupported:
        return
    formatted = ", ".join(f"`{name}`" for name in unsupported)
    raise ValueError(
        f"`plot_type='{plot_type}'` does not support the following arguments: {formatted}."
    )


def _ensure_allowed_metrics(
    plot_type: str,
    *,
    color_by: Literal["score", "pvalue"],
    allowed_color_by: Sequence[str],
    value: Literal["sum", "mean", "max", "count"],
    allowed_value: Sequence[str],
) -> None:
    if color_by not in allowed_color_by:
        allowed = ", ".join(f"`{item}`" for item in allowed_color_by)
        raise ValueError(f"`plot_type='{plot_type}'` only supports `color_by` values: {allowed}.")
    if value not in allowed_value:
        allowed = ", ".join(f"`{item}`" for item in allowed_value)
        raise ValueError(f"`plot_type='{plot_type}'` only supports `value` values: {allowed}.")


def _require_use_argument(plot_type: str, arg_name: str, values) -> list[str]:
    normalized = _normalize_use_arg(values)
    if normalized is None:
        raise ValueError(f"`{arg_name}` is required when `plot_type='{plot_type}'`.")
    return normalized


def _aggregate_series(grouped, value: Literal["sum", "mean", "max", "count"], *, field: str = "score") -> pd.Series:
    if value == "sum":
        return grouped[field].sum()
    if value == "mean":
        return grouped[field].mean()
    if value == "max":
        return grouped[field].max()
    return grouped["significant"].sum().astype(float)


def _communication_long_table(
    adata: anndata.AnnData,
    *,
    sender_use=None,
    receiver_use=None,
    signaling=None,
    interaction_use=None,
    pair_lr_use=None,
    pvalue_threshold: float = 0.05,
) -> pd.DataFrame:
    _ensure_comm_adata(adata)

    sender_use = _normalize_use_arg(sender_use)
    receiver_use = _normalize_use_arg(receiver_use)
    signaling = _normalize_use_arg(signaling)
    interaction_use = _normalize_use_arg(interaction_use)
    pair_lr_use = _normalize_use_arg(pair_lr_use)

    means = pd.DataFrame(_to_dense(adata.layers["means"]), index=adata.obs_names, columns=adata.var_names)
    pvalues = pd.DataFrame(_to_dense(adata.layers["pvalues"]), index=adata.obs_names, columns=adata.var_names)

    obs_meta = adata.obs.loc[:, ["sender", "receiver"]].copy()
    obs_meta["sender"] = obs_meta["sender"].astype(str)
    obs_meta["receiver"] = obs_meta["receiver"].astype(str)
    obs_meta["pair"] = obs_meta["sender"] + " -> " + obs_meta["receiver"]

    var_meta = adata.var.copy()
    var_meta["interaction"] = _pick_first_available(
        var_meta,
        ["interaction_name_2", "interaction_name", "interacting_pair", "gene_name"],
        var_meta.index.astype(str),
    )
    var_meta["pair_lr"] = _pick_first_available(
        var_meta,
        ["interacting_pair", "interaction_name_2", "interaction_name", "gene_name"],
        var_meta.index.astype(str),
    )
    var_meta["classification"] = _pick_first_available(
        var_meta,
        ["classification", "pathway_name", "signaling"],
        np.repeat("Unclassified", adata.n_vars),
    )
    var_meta["ligand"] = _pick_first_available(
        var_meta,
        ["gene_a", "partner_a", "ligand"],
        np.repeat("", adata.n_vars),
    )
    var_meta["receptor"] = _pick_first_available(
        var_meta,
        ["gene_b", "partner_b", "receptor"],
        np.repeat("", adata.n_vars),
    )

    long_df = means.stack(future_stack=True).rename("score").to_frame()
    long_df["pvalue"] = pvalues.stack(future_stack=True)
    long_df = long_df.reset_index().rename(columns={"level_0": "pair_id", "level_1": "feature_id"})
    long_df = long_df.merge(obs_meta, left_on="pair_id", right_index=True, how="left")
    long_df = long_df.merge(
        var_meta.loc[:, ["interaction", "pair_lr", "classification", "ligand", "receptor"]],
        left_on="feature_id",
        right_index=True,
        how="left",
    )
    long_df["interaction_display"] = long_df["interaction"].map(_display_interaction_label)
    long_df["ligand_display"] = long_df["ligand"].map(_display_gene_label)
    long_df["receptor_display"] = long_df["receptor"].map(_display_gene_label)
    long_df["significant"] = long_df["pvalue"].astype(float) < float(pvalue_threshold)
    long_df["neglog10_pvalue"] = -np.log10(np.clip(long_df["pvalue"].astype(float), 1e-300, 1.0))

    if sender_use is not None:
        long_df = long_df.loc[long_df["sender"].isin(sender_use)]
    if receiver_use is not None:
        long_df = long_df.loc[long_df["receiver"].isin(receiver_use)]
    if signaling is not None:
        long_df = long_df.loc[long_df["classification"].isin(signaling)]
    if interaction_use is not None:
        long_df = long_df.loc[long_df["interaction"].isin(interaction_use)]
    if pair_lr_use is not None:
        long_df = long_df.loc[long_df["pair_lr"].isin(pair_lr_use)]

    return long_df.reset_index(drop=True)


def _choose_palette(categories: Sequence[str], palette=None) -> dict[str, str]:
    categories = [str(category) for category in categories]
    if palette is None:
        if len(categories) <= len(palette_28):
            colors = palette_28
        elif len(categories) <= len(palette_56):
            colors = palette_56
        else:
            colors = palette_112
        return {category: colors[idx % len(colors)] for idx, category in enumerate(categories)}

    if isinstance(palette, Mapping):
        return {category: palette.get(category, "#808080") for category in categories}

    if isinstance(palette, str):
        if mcolors.is_color_like(palette):
            colors = [mcolors.to_hex(palette)]
        else:
            try:
                colors = [mcolors.to_hex(color) for color in sns.color_palette(palette, n_colors=len(categories))]
            except ValueError:
                try:
                    cmap = plt.get_cmap(palette)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported palette specification: {palette!r}. "
                        "Provide a matplotlib/seaborn palette name, a color string, or a mapping."
                    ) from exc
                positions = np.linspace(0.0, 1.0, max(len(categories), 2))[: len(categories)]
                colors = [mcolors.to_hex(cmap(position)) for position in positions]
    else:
        colors = [mcolors.to_hex(color) if mcolors.is_color_like(color) else color for color in list(palette)]

    if not colors:
        raise ValueError("Palette must provide at least one color.")
    return {category: colors[idx % len(colors)] for idx, category in enumerate(categories)}


def _cluster_axis_order(matrix: pd.DataFrame, axis: Literal["rows", "columns"]) -> list[str]:
    if axis == "rows":
        working = matrix
    else:
        working = matrix.T

    if working.shape[0] <= 2:
        return list(working.index)
    if np.allclose(working.to_numpy(), working.to_numpy()[0]):
        return list(working.index)

    condensed = pdist(working.to_numpy())
    if not np.isfinite(condensed).all() or np.allclose(condensed, 0):
        return list(working.index)

    order = leaves_list(linkage(condensed, method="average"))
    return list(working.index[order])


def _apply_cluster_order(matrix: pd.DataFrame, *, cluster_rows: bool, cluster_columns: bool) -> pd.DataFrame:
    if cluster_rows:
        matrix = matrix.loc[_cluster_axis_order(matrix, "rows")]
    if cluster_columns:
        matrix = matrix.loc[:, _cluster_axis_order(matrix, "columns")]
    return matrix


def _aggregate_summary(
    long_df: pd.DataFrame,
    *,
    color_by: Literal["score", "pvalue"],
    value: Literal["sum", "mean", "max", "count"],
) -> tuple[pd.Series, str]:
    if color_by == "score":
        summary = _aggregate_series(long_df.groupby("interaction", observed=True), value)
        label = "Communication score"
    else:
        summary = long_df.groupby("interaction", observed=True)["neglog10_pvalue"].max()
        label = "-log10(p-value)"
    return summary.sort_values(ascending=False), label


def _aggregate_metric_frame(
    long_df: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    color_by: Literal["score", "pvalue"],
    value: Literal["sum", "mean", "max", "count"],
) -> tuple[pd.DataFrame, str]:
    grouped = long_df.groupby(list(group_cols), observed=True)
    if color_by == "score":
        metric = _aggregate_series(grouped, value).rename("value").reset_index()
        label = "Communication score" if value != "count" else "Significant interactions"
    else:
        metric = grouped["neglog10_pvalue"].max().rename("value").reset_index()
        label = "-log10(p-value)"
    return metric, label


def _truncate_matrix_columns(matrix: pd.DataFrame, top_n: int | None, *, absolute: bool = False) -> pd.DataFrame:
    if top_n is None or top_n <= 0 or matrix.shape[1] <= top_n:
        return matrix
    summary = matrix.abs().sum(axis=0) if absolute else matrix.sum(axis=0)
    keep = summary.sort_values(ascending=False).head(int(top_n)).index
    return matrix.loc[:, keep]


def _normalize_side_annotations(values, *, default: Sequence[str] = ()) -> list[str]:
    if values is None:
        tokens = list(default)
    elif isinstance(values, str):
        tokens = [values]
    else:
        tokens = [str(value) for value in values]

    normalized = []
    for token in tokens:
        item = str(token).strip().lower()
        if not item:
            continue
        if item not in {"bar", "cell"}:
            raise ValueError(
                f"Unsupported annotation type {token!r}. "
                "Currently supported values are 'bar' and 'cell'."
            )
        if item not in normalized:
            normalized.append(item)
    return normalized


def _matrix_side_totals(
    matrix: pd.DataFrame,
    size_matrix: pd.DataFrame | None,
    *,
    axis: Literal["rows", "columns"],
    metric: Literal["count", "sum", "mean", "max"],
) -> pd.Series:
    primary = matrix if axis == "rows" else matrix.T
    support = None if size_matrix is None else (size_matrix if axis == "rows" else size_matrix.T)

    if metric == "count":
        if support is not None:
            return support.sum(axis=1).astype(float)
        return primary.gt(0).sum(axis=1).astype(float)
    if metric == "sum":
        return primary.sum(axis=1).astype(float)
    if metric == "mean":
        return primary.mean(axis=1).astype(float)
    return primary.max(axis=1).astype(float)


def _shared_palette_legend(row_palette: Mapping[str, str] | None, col_palette: Mapping[str, str] | None):
    if not row_palette or not col_palette:
        return None
    row_keys = list(row_palette.keys())
    col_keys = list(col_palette.keys())
    if row_keys != col_keys:
        return None
    return {key: col_palette[key] for key in col_keys}


def _attach_category_legend(fig, palette: Mapping[str, str] | None, *, title: str = "") -> None:
    if not palette or len(palette) > 20:
        return
    handles = [Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", label=label) for label, color in palette.items()]
    legend = fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.90, 0.34),
        frameon=False,
        title=title,
        borderaxespad=0.0,
        handlelength=0.9,
        handletextpad=0.5,
    )
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontsize(9)


def _use_shared_category_legend(
    row_palette: Mapping[str, str] | None,
    col_palette: Mapping[str, str] | None,
) -> bool:
    shared = _shared_palette_legend(row_palette, col_palette)
    return bool(shared) and len(shared) <= 20


def _display_axis_labels(
    labels: Sequence[str],
    *,
    wrap_width: int,
    max_labels: int | None = None,
) -> list[str]:
    labels = [str(label) for label in labels]
    if max_labels is None or len(labels) <= max_labels:
        return [_wrap_plot_label(label, width=wrap_width) for label in labels]

    _, keep_indices = _smart_tick_positions(len(labels), max_labels=max_labels)
    keep = set(keep_indices)
    rendered = []
    for idx, label in enumerate(labels):
        rendered.append(_wrap_plot_label(label, width=wrap_width) if idx in keep else "")
    return rendered


def _role_matrix(
    long_df: pd.DataFrame,
    *,
    pattern: Literal["outgoing", "incoming", "all"],
    color_by: Literal["score", "pvalue"],
    value: Literal["sum", "mean", "max", "count"],
    top_n: int | None = None,
) -> tuple[pd.DataFrame, str]:
    outgoing_df, label = _aggregate_metric_frame(
        long_df,
        ["sender", "classification"],
        color_by=color_by,
        value=value,
    )
    incoming_df, incoming_label = _aggregate_metric_frame(
        long_df,
        ["receiver", "classification"],
        color_by=color_by,
        value=value,
    )
    outgoing = outgoing_df.pivot(index="sender", columns="classification", values="value").fillna(0.0)
    incoming = incoming_df.pivot(index="receiver", columns="classification", values="value").fillna(0.0)

    if pattern == "outgoing":
        matrix = outgoing
        role_label = f"Outgoing {label.lower()}"
    elif pattern == "incoming":
        matrix = incoming
        role_label = f"Incoming {incoming_label.lower()}"
    else:
        matrix = outgoing.add(incoming, fill_value=0.0)
        role_label = f"Combined {label.lower()}"

    matrix = _truncate_matrix_columns(matrix, top_n)
    if matrix.empty:
        raise ValueError("No communication roles remain after filtering.")
    return matrix, role_label[:1].upper() + role_label[1:]


def _interaction_stage_label(long_df: pd.DataFrame) -> pd.Series:
    pair_lr = long_df["pair_lr"].astype(str).str.strip()
    interaction = long_df["interaction"].astype(str).str.strip()
    return pair_lr.where(pair_lr.ne(""), interaction)


def _rank_flow_levels(labels: Sequence[str], weights: Sequence[float]) -> list[str]:
    frame = pd.DataFrame({"label": [str(label) for label in labels], "weight": np.asarray(weights, dtype=float)})
    frame = frame.loc[frame["label"].str.strip().ne("")]
    if frame.empty:
        return []
    ranked = frame.groupby("label", observed=True)["weight"].sum().sort_values(ascending=False)
    return ranked.index.astype(str).tolist()


def _flow_nodes_for_column(
    labels: list[str],
    *,
    x: float,
    column: str,
    plot_df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    if not labels:
        return pd.DataFrame(columns=["node_id", "label", "column", "x", "y", "weight"])
    totals = (
        plot_df.groupby(label_col, observed=True)["weight"].sum().reindex(labels).fillna(0.0)
    )
    positions = _weighted_positions(labels, totals)
    return pd.DataFrame(
        {
            "node_id": [f"{column}::{label}" for label in labels],
            "label": labels,
            "column": column,
            "x": float(x),
            "y": [positions[label][1] for label in labels],
            "weight": totals.to_numpy(dtype=float),
        }
    )


def _draw_flow_stage_node(
    ax,
    *,
    x_coord: float,
    y_coord: float,
    label: str,
    weight: float,
    node_max_weight: float,
    wrap_width: int = 16,
) -> None:
    wrapped_label = _wrap_plot_label(label, width=wrap_width)
    n_lines = max(len(wrapped_label.splitlines()), 1)
    base_scale = float(weight) / max(float(node_max_weight), 1e-9)
    width = 0.10 + 0.025 * base_scale
    height = 0.07 + 0.038 * (n_lines - 1) + 0.01 * base_scale
    height = min(height, 0.18)

    ax.add_patch(
        Rectangle(
            (x_coord - width / 2.0, y_coord - height / 2.0),
            width,
            height,
            facecolor="white",
            edgecolor="#6E6E6E",
            linewidth=1.6,
            zorder=3,
        )
    )
    ax.text(
        x_coord,
        y_coord,
        wrapped_label,
        ha="center",
        va="center",
        fontsize=9,
        zorder=4,
        linespacing=1.05,
    )


def _build_flow_plot_frames(
    long_df: pd.DataFrame,
    *,
    display_by: Literal["aggregation", "interaction"],
    value: Literal["sum", "mean", "max", "count"],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[float, str]]]:
    if display_by == "interaction":
        plot_df = (
            _aggregate_series(
                long_df.assign(interaction_label=_interaction_stage_label(long_df)).groupby(
                    ["sender", "interaction_label", "receiver"], observed=True
                ),
                value,
            )
            .rename("weight")
            .reset_index()
        )
        interaction_scores = (
            plot_df.groupby("interaction_label", observed=True)["weight"].sum().sort_values(ascending=False)
        )
        keep = interaction_scores.head(int(top_n)).index.astype(str).tolist()
        plot_df = plot_df.loc[plot_df["interaction_label"].astype(str).isin(keep)].copy()
        plot_df = plot_df.loc[plot_df["weight"] > 0].copy()
        if plot_df.empty:
            raise ValueError("No interaction-level communication edges remain after filtering.")

        # Keep the interaction-flow panels readable by pruning each ligand-receptor
        # stage to its strongest sender/receiver branches.
        branch_limit = max(3, min(5, int(np.ceil(np.sqrt(max(int(top_n), 1))))))
        sender_keep = (
            plot_df.groupby(["interaction_label", "sender"], observed=True)["weight"]
            .sum()
            .rename("weight")
            .reset_index()
            .sort_values(["interaction_label", "weight"], ascending=[True, False])
            .groupby("interaction_label", observed=True)
            .head(branch_limit)
        )
        receiver_keep = (
            plot_df.groupby(["interaction_label", "receiver"], observed=True)["weight"]
            .sum()
            .rename("weight")
            .reset_index()
            .sort_values(["interaction_label", "weight"], ascending=[True, False])
            .groupby("interaction_label", observed=True)
            .head(branch_limit)
        )
        sender_pairs = set(zip(sender_keep["interaction_label"].astype(str), sender_keep["sender"].astype(str)))
        receiver_pairs = set(zip(receiver_keep["interaction_label"].astype(str), receiver_keep["receiver"].astype(str)))
        plot_df = plot_df.loc[
            [
                (str(interaction), str(sender)) in sender_pairs and (str(interaction), str(receiver)) in receiver_pairs
                for interaction, sender, receiver in zip(
                    plot_df["interaction_label"], plot_df["sender"], plot_df["receiver"]
                )
            ]
        ].copy()
        if plot_df.empty:
            raise ValueError("No interaction-level communication edges remain after branch pruning.")

        sender_levels = _rank_flow_levels(plot_df["sender"], plot_df["weight"])
        interaction_levels = [label for label in interaction_scores.index.astype(str).tolist() if label in set(keep)]
        receiver_levels = _rank_flow_levels(plot_df["receiver"], plot_df["weight"])

        sender_nodes = _flow_nodes_for_column(sender_levels, x=0.0, column="sender", plot_df=plot_df, label_col="sender")
        interaction_nodes = _flow_nodes_for_column(
            interaction_levels,
            x=0.5,
            column="interaction",
            plot_df=plot_df,
            label_col="interaction_label",
        )
        receiver_nodes = _flow_nodes_for_column(
            receiver_levels,
            x=1.0,
            column="receiver",
            plot_df=plot_df,
            label_col="receiver",
        )
        node_df = pd.concat([sender_nodes, interaction_nodes, receiver_nodes], ignore_index=True)

        left_edges = (
            plot_df.groupby(["sender", "interaction_label"], observed=True)["weight"].sum().rename("weight").reset_index()
        )
        left_edges["from_id"] = "sender::" + left_edges["sender"].astype(str)
        left_edges["to_id"] = "interaction::" + left_edges["interaction_label"].astype(str)
        left_edges["edge_group"] = left_edges["sender"].astype(str)
        left_edges["edge_label"] = left_edges["interaction_label"].astype(str)

        right_edges = (
            plot_df.groupby(["interaction_label", "receiver"], observed=True)["weight"].sum().rename("weight").reset_index()
        )
        sender_lookup = (
            plot_df.groupby(["interaction_label", "receiver"], observed=True)["weight"]
            .sum()
            .reset_index()
            .merge(
                plot_df.groupby(["interaction_label", "receiver", "sender"], observed=True)["weight"]
                .sum()
                .rename("sender_weight")
                .reset_index(),
                on=["interaction_label", "receiver"],
                how="left",
            )
            .sort_values(["interaction_label", "receiver", "sender_weight"], ascending=[True, True, False])
            .drop_duplicates(["interaction_label", "receiver"])
            .set_index(["interaction_label", "receiver"])["sender"]
        )
        right_edges["from_id"] = "interaction::" + right_edges["interaction_label"].astype(str)
        right_edges["to_id"] = "receiver::" + right_edges["receiver"].astype(str)
        right_edges["edge_group"] = [
            str(sender_lookup.get((row["interaction_label"], row["receiver"]), row["receiver"]))
            for _, row in right_edges.iterrows()
        ]
        right_edges["edge_label"] = right_edges["interaction_label"].astype(str)

        edge_df = pd.concat(
            [
                left_edges.loc[:, ["from_id", "to_id", "weight", "edge_group", "edge_label"]],
                right_edges.loc[:, ["from_id", "to_id", "weight", "edge_group", "edge_label"]],
            ],
            ignore_index=True,
        )
        column_titles = [(0.0, "Sender"), (0.5, "Ligand-Receptor"), (1.0, "Receiver")]
    else:
        plot_df = _aggregate_series(long_df.groupby(["sender", "receiver"], observed=True), value).rename("weight").reset_index()
        plot_df = plot_df.loc[plot_df["weight"] > 0].sort_values("weight", ascending=False).head(int(top_n))
        if plot_df.empty:
            raise ValueError("No communication edges remain after filtering.")

        sender_levels = _rank_flow_levels(plot_df["sender"], plot_df["weight"])
        receiver_levels = _rank_flow_levels(plot_df["receiver"], plot_df["weight"])
        sender_nodes = _flow_nodes_for_column(sender_levels, x=0.0, column="sender", plot_df=plot_df, label_col="sender")
        receiver_nodes = _flow_nodes_for_column(
            receiver_levels,
            x=1.0,
            column="receiver",
            plot_df=plot_df,
            label_col="receiver",
        )
        node_df = pd.concat([sender_nodes, receiver_nodes], ignore_index=True)
        edge_df = pd.DataFrame(
            {
                "from_id": "sender::" + plot_df["sender"].astype(str),
                "to_id": "receiver::" + plot_df["receiver"].astype(str),
                "weight": plot_df["weight"].astype(float),
                "edge_group": plot_df["sender"].astype(str),
                "edge_label": plot_df["sender"].astype(str) + " -> " + plot_df["receiver"].astype(str),
            }
        )
        column_titles = [(0.0, "Sender"), (1.0, "Receiver")]

    if node_df.empty or edge_df.empty:
        raise ValueError("No communication flow nodes remain after filtering.")

    node_lookup = node_df.set_index("node_id")[["x", "y"]]
    edge_df = edge_df.merge(node_lookup, left_on="from_id", right_index=True, how="left")
    edge_df = edge_df.merge(node_lookup, left_on="to_id", right_index=True, how="left", suffixes=("_from", "_to"))
    edge_df = edge_df.dropna(subset=["x_from", "y_from", "x_to", "y_to"]).reset_index(drop=True)
    if edge_df.empty:
        raise ValueError("No communication flow edges remain after filtering.")
    return node_df.reset_index(drop=True), edge_df, column_titles


def _plot_heatmap_matrix(
    matrix: pd.DataFrame,
    *,
    color_label: str,
    title: str,
    cmap: str,
    figsize: tuple[float, float],
    border: bool,
    add_text: bool,
    center: float | None = None,
    row_palette: Mapping[str, str] | None = None,
    col_palette: Mapping[str, str] | None = None,
    row_totals: pd.Series | None = None,
    col_totals: pd.Series | None = None,
    row_total_label: str | None = None,
    col_total_label: str | None = None,
    top_annos: Sequence[str] = (),
    bottom_annos: Sequence[str] = (),
    left_annos: Sequence[str] = (),
    right_annos: Sequence[str] = (),
    row_wrap_width: int = 20,
    col_wrap_width: int = 18,
    max_col_labels: int = 18,
    clip_quantile: float | None = None,
):
    ma, mp = _import_marsilea()
    display_matrix = matrix.copy()
    if clip_quantile is not None and 0.0 < clip_quantile < 1.0:
        values = display_matrix.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            if center is None:
                upper = float(np.quantile(finite, clip_quantile))
                lower = float(np.quantile(finite, 1.0 - clip_quantile)) if np.min(finite) < 0 else None
                if np.isfinite(upper):
                    display_matrix = display_matrix.clip(upper=upper)
                if lower is not None and np.isfinite(lower):
                    display_matrix = display_matrix.clip(lower=lower)
            else:
                bound = float(np.quantile(np.abs(finite), clip_quantile))
                if np.isfinite(bound) and bound > 0:
                    display_matrix = display_matrix.clip(lower=-bound, upper=bound)
    row_names = matrix.index.astype(str).tolist()
    col_names = matrix.columns.astype(str).tolist()
    display_matrix.index = _display_axis_labels(row_names, wrap_width=row_wrap_width)
    display_matrix.columns = _display_axis_labels(
        col_names,
        wrap_width=col_wrap_width,
        max_labels=max_col_labels,
    )
    plot_width = max(float(figsize[0]) * 0.68, 3.0)
    plot_height = max(float(figsize[1]) * 0.68, 2.5)
    heatmap = ma.Heatmap(
        display_matrix,
        width=plot_width,
        height=plot_height,
        cmap=cmap,
        center=center,
        label=color_label,
        linewidth=0.5,
    )
    if row_totals is not None:
        row_totals = pd.Series(row_totals, index=matrix.index).reindex(matrix.index, fill_value=0.0)
    if col_totals is not None:
        col_totals = pd.Series(col_totals, index=matrix.columns).reindex(matrix.columns, fill_value=0.0)

    if "bar" in left_annos and row_totals is not None:
        heatmap.add_left(
            mp.Numbers(
                row_totals.to_numpy(dtype=float),
                color="#F05454",
                label=row_total_label or "Total",
                show_value=False,
                label_props={"size": 11},
                props={"size": 11},
            ),
            size=0.45,
            pad=0.04,
        )
    if "cell" in left_annos and row_palette is not None:
        heatmap.add_left(mp.Colors(row_names, palette=row_palette), size=0.2, pad=0.04, legend=False)
    if "cell" in right_annos and row_palette is not None:
        heatmap.add_right(mp.Colors(row_names, palette=row_palette), size=0.12, pad=0.03, legend=False)
    if "bar" in right_annos and row_totals is not None:
        heatmap.add_right(
            mp.Numbers(
                row_totals.to_numpy(dtype=float),
                color="#D3B27B",
                label=None,
                show_value=False,
                props={"size": 9},
            ),
            size=0.22,
            pad=0.03,
        )
    if "bar" in top_annos and col_totals is not None:
        heatmap.add_top(
            mp.Numbers(
                col_totals.to_numpy(dtype=float),
                color="#4A90E2",
                label=col_total_label or "Total",
                show_value=False,
                label_props={"size": 11},
                props={"size": 11},
            ),
            size=0.42,
            pad=0.04,
        )
    if "cell" in top_annos and col_palette is not None:
        heatmap.add_top(mp.Colors(col_names, palette=col_palette), size=0.18, pad=0.03)
    if "cell" in bottom_annos and col_palette is not None:
        heatmap.add_bottom(mp.Colors(col_names, palette=col_palette), size=0.12, pad=0.03, legend=False)
    if "bar" in bottom_annos and col_totals is not None:
        heatmap.add_bottom(
            mp.Numbers(
                col_totals.to_numpy(dtype=float),
                color="#BFC7D5",
                label=None,
                show_value=False,
                props={"size": 9},
            ),
            size=0.26,
            pad=0.03,
        )
    heatmap.add_title(title)
    if hasattr(heatmap, "add_legends"):
        heatmap.add_legends()

    if border:
        heatmap = _attach_post_render_hook(
            heatmap,
            lambda: _style_heatmap_axes(getattr(heatmap, "figure", None)),
        )

    fig = _render_plot(heatmap)

    heatmap_axes = [ax for ax in fig.axes if ax.collections or ax.images]
    ax = (
        max(
            heatmap_axes,
            key=lambda axis: axis.get_position().width * axis.get_position().height,
        )
        if heatmap_axes
        else fig.axes[0]
    )
    _attach_semantic_ticks(ax, row_names, col_names)

    if add_text and matrix.size <= 49:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    f"{matrix.iat[row_idx, col_idx]:.2g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#2B2B2B",
                )
    return fig, ax


def _largest_content_axis(fig):
    content_axes = [ax for ax in fig.axes if ax.collections or ax.images or ax.lines or ax.patches]
    if not content_axes:
        return fig.axes[0]
    return max(content_axes, key=lambda axis: axis.get_position().width * axis.get_position().height)


def _render_plotter_figure(plotter, *, title: str | None = None, add_custom_legends: bool = False):
    legend_kws = dict(pad=0.05) if add_custom_legends else None
    fig = _render_plot(plotter, legend_kws=legend_kws)
    ax = _largest_content_axis(fig)
    if title:
        ax.set_title(title, fontsize=12)
    return fig, ax


def _attach_semantic_ticks(ax, row_names: Sequence[str], col_names: Sequence[str]) -> None:
    if row_names:
        ax.set_yticks(np.arange(len(row_names)) + 0.5, labels=[str(name) for name in row_names])
    if col_names:
        ax.set_xticks(np.arange(len(col_names)) + 0.5, labels=[str(name) for name in col_names], rotation=90)
    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=True,
        labeltop=False,
        length=0,
        pad=0,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=True,
        labelright=False,
        length=0,
        pad=0,
    )
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_alpha(0.0)


def _resolve_lr_pairs(
    adata: anndata.AnnData,
    *,
    pair_lr_use=None,
    interaction_use=None,
) -> list[str] | None:
    pair_values = _normalize_use_arg(pair_lr_use)
    if pair_values is not None:
        return pair_values

    interaction_values = _normalize_use_arg(interaction_use)
    if interaction_values is None:
        return None

    var_meta = adata.var.copy()
    var_meta["interaction"] = _pick_first_available(
        var_meta,
        ["interaction_name_2", "interaction_name", "interacting_pair", "gene_name"],
        adata.var_names.astype(str),
    )
    var_meta["pair_lr"] = _pick_first_available(
        var_meta,
        ["interacting_pair", "interaction_name_2", "interaction_name", "gene_name"],
        adata.var_names.astype(str),
    )
    pairs = (
        var_meta.loc[var_meta["interaction"].isin(interaction_values), "pair_lr"]
        .dropna()
        .astype(str)
        .tolist()
    )
    pairs = list(dict.fromkeys(pairs))
    return pairs or interaction_values


def _build_cellchatviz(adata: anndata.AnnData, *, palette=None):
    from ._cpdbviz import CellChatViz

    nodes = sorted(set(adata.obs["sender"].astype(str)) | set(adata.obs["receiver"].astype(str)))
    palette_map = _choose_palette(nodes, palette=palette)
    return CellChatViz(adata, palette=palette_map)


def _pathway_summary_table(
    adata: anndata.AnnData,
    *,
    palette=None,
    signaling=None,
    pathway_method: str = "mean",
    min_lr_pairs: int = 1,
    min_expression: float = 0.1,
    strength_threshold: float = 0.5,
    pvalue_threshold: float = 0.05,
    min_significant_pairs: int = 1,
) -> pd.DataFrame:
    viz = _build_cellchatviz(adata, palette=palette)
    pathway_comm = viz.compute_pathway_communication(
        method=pathway_method,
        min_lr_pairs=min_lr_pairs,
        min_expression=min_expression,
    )
    _, summary = viz.get_significant_pathways_v2(
        pathway_comm,
        strength_threshold=strength_threshold,
        pvalue_threshold=pvalue_threshold,
        min_significant_pairs=min_significant_pairs,
    )
    if summary is None or len(summary) == 0:
        raise ValueError("No signaling pathways remain after applying the pathway summary thresholds.")
    summary = summary.copy()
    summary["pathway"] = summary["pathway"].astype(str)

    signaling_values = _normalize_use_arg(signaling)
    if signaling_values is not None:
        summary = summary.loc[summary["pathway"].isin(signaling_values)].copy()
    if summary.empty:
        raise ValueError("No signaling pathways remain after filtering.")
    return summary


def _pair_color_metadata(labels: Sequence[str], *, palette=None) -> Mapping[str, str]:
    values = [str(label) for label in labels]
    if not values:
        return {}
    senders = []
    receivers = []
    for value in values:
        if " -> " in value:
            sender, receiver = value.split(" -> ", 1)
        else:
            sender, receiver = value, value
        senders.append(sender)
        receivers.append(receiver)
    sender_palette = _choose_palette(list(dict.fromkeys(senders)), palette=palette)
    receiver_palette = _choose_palette(list(dict.fromkeys(receivers)), palette=palette)
    combined = {}
    for label, sender, receiver in zip(values, senders, receivers):
        sender_rgb = np.array(mcolors.to_rgb(sender_palette[sender]))
        receiver_rgb = np.array(mcolors.to_rgb(receiver_palette[receiver]))
        combined[label] = mcolors.to_hex((sender_rgb + receiver_rgb) / 2.0)
    return combined


def _diff_role_matrix(
    adata: anndata.AnnData,
    comparison_adata: anndata.AnnData,
    *,
    sender_use=None,
    receiver_use=None,
    signaling=None,
    interaction_use=None,
    pair_lr_use=None,
    pvalue_threshold: float,
    pattern: Literal["outgoing", "incoming", "all"],
    color_by: Literal["score", "pvalue"],
    value: Literal["sum", "mean", "max", "count"],
    top_n: int | None = None,
) -> tuple[pd.DataFrame, str]:
    reference_long_df = _communication_long_table(
        adata,
        sender_use=sender_use,
        receiver_use=receiver_use,
        signaling=signaling,
        interaction_use=interaction_use,
        pair_lr_use=pair_lr_use,
        pvalue_threshold=pvalue_threshold,
    )
    comparison_long_df = _communication_long_table(
        comparison_adata,
        sender_use=sender_use,
        receiver_use=receiver_use,
        signaling=signaling,
        interaction_use=interaction_use,
        pair_lr_use=pair_lr_use,
        pvalue_threshold=pvalue_threshold,
    )
    reference_matrix, label = _role_matrix(
        reference_long_df,
        pattern=pattern,
        color_by=color_by,
        value=value,
        top_n=None,
    )
    comparison_matrix, _ = _role_matrix(
        comparison_long_df,
        pattern=pattern,
        color_by=color_by,
        value=value,
        top_n=None,
    )
    delta = comparison_matrix.add(-reference_matrix, fill_value=0.0)
    delta = _truncate_matrix_columns(delta, top_n, absolute=True)
    if delta.empty:
        raise ValueError("No differential communication roles remain after filtering.")
    return delta, f"Delta {label.lower()}"


def _communication_matrix(
    long_df: pd.DataFrame,
    *,
    display_by: Literal["aggregation", "interaction"],
    color_by: Literal["score", "pvalue"],
    value: Literal["sum", "mean", "max", "count"],
    top_n: int,
    top_n_pairs: int | None = None,
    facet_by: Literal["sender", "receiver"] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    data = long_df.copy()
    if data.empty:
        raise ValueError("No communication records remain after filtering.")

    if display_by == "aggregation":
        rows = "receiver"
        cols = "sender"
        if color_by == "score":
            if value == "sum":
                matrix = data.groupby([rows, cols], observed=True)["score"].sum().unstack(fill_value=0.0)
            elif value == "mean":
                matrix = data.groupby([rows, cols], observed=True)["score"].mean().unstack(fill_value=0.0)
            elif value == "max":
                matrix = data.groupby([rows, cols], observed=True)["score"].max().unstack(fill_value=0.0)
            else:
                matrix = (
                    data.groupby([rows, cols], observed=True)["significant"]
                    .sum()
                    .unstack(fill_value=0.0)
                    .astype(float)
                )
            label = "Communication score"
        else:
            matrix = data.groupby([rows, cols], observed=True)["neglog10_pvalue"].max().unstack(fill_value=0.0)
            label = "-log10(p-value)"

        size_matrix = (
            data.groupby([rows, cols], observed=True)["significant"]
            .sum()
            .unstack(fill_value=0.0)
            .astype(float)
        )
        return matrix, size_matrix, label

    rows = "interaction"
    if facet_by == "sender":
        ordered_pairs = (
            data.loc[:, ["sender", "receiver"]]
            .drop_duplicates()
            .sort_values(["sender", "receiver"], kind="stable")
        )
        ordered_columns = (ordered_pairs["sender"] + " -> " + ordered_pairs["receiver"]).tolist()
        data = data.assign(plot_pair=data["sender"] + " -> " + data["receiver"])
        cols = "plot_pair"
    elif facet_by == "receiver":
        ordered_pairs = (
            data.loc[:, ["receiver", "sender"]]
            .drop_duplicates()
            .sort_values(["receiver", "sender"], kind="stable")
        )
        ordered_columns = (ordered_pairs["sender"] + " -> " + ordered_pairs["receiver"]).tolist()
        data = data.assign(plot_pair=data["sender"] + " -> " + data["receiver"])
        cols = "plot_pair"
    else:
        ordered_columns = None
        cols = "pair"

    score_col = "score" if color_by == "score" else "neglog10_pvalue"
    matrix = data.pivot_table(index=rows, columns=cols, values=score_col, aggfunc="mean", fill_value=0.0)
    size_matrix = data.pivot_table(index=rows, columns=cols, values="significant", aggfunc="sum", fill_value=0.0)
    if ordered_columns is not None:
        ordered_columns = [column for column in ordered_columns if column in matrix.columns]
        matrix = matrix.reindex(columns=ordered_columns, fill_value=0.0)
        size_matrix = size_matrix.reindex(columns=ordered_columns, fill_value=0.0)

    if top_n is not None and top_n > 0 and matrix.shape[0] > top_n:
        ranked = data.groupby("interaction", observed=True)[score_col].sum().sort_values(ascending=False)
        keep = ranked.head(int(top_n)).index
        matrix = matrix.loc[matrix.index.intersection(keep)]
        size_matrix = size_matrix.loc[size_matrix.index.intersection(keep)]

    if top_n_pairs is not None and top_n_pairs > 0 and matrix.shape[1] > top_n_pairs:
        ranked_pairs = data.groupby(cols, observed=True)[score_col].sum().sort_values(ascending=False)
        keep_pairs = ranked_pairs.head(int(top_n_pairs)).index
        if ordered_columns is not None:
            keep_pairs = [column for column in ordered_columns if column in set(keep_pairs)]
        matrix = matrix.loc[:, matrix.columns.intersection(keep_pairs)]
        size_matrix = size_matrix.loc[:, size_matrix.columns.intersection(keep_pairs)]
        if ordered_columns is not None:
            matrix = matrix.reindex(columns=keep_pairs, fill_value=0.0)
            size_matrix = size_matrix.reindex(columns=keep_pairs, fill_value=0.0)

    label = "Communication score" if color_by == "score" else "-log10(p-value)"
    return matrix, size_matrix, label


def _maybe_save_show(fig, *, show: bool, save):
    if save:
        save_path = save if isinstance(save, str) else "communication_plot.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _wrap_plot_label(label: str, *, width: int = 18) -> str:
    text = str(label).replace("_", " ")
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def _clean_ccc_identifier(label: str, *, drop_receptor_suffix: bool = False) -> str:
    text = str(label).strip()
    if not text or text.lower() == "nan":
        return ""
    text = re.sub(r"^(complex:|simple:)", "", text)
    text = re.sub(r"_complex$", "", text)
    if drop_receptor_suffix:
        text = re.sub(r"_receptor_inhibitor$", "", text)
        text = re.sub(r"_receptor$", "", text)
        text = re.sub(r"_ligand$", "", text)
    return text


def _display_interaction_label(label: str) -> str:
    return _clean_ccc_identifier(label, drop_receptor_suffix=False)


def _display_gene_label(label: str) -> str:
    return _clean_ccc_identifier(label, drop_receptor_suffix=True)


def _outline_texts(texts: Sequence) -> None:
    for text in texts:
        text.set_path_effects([patheffects.withStroke(linewidth=2.6, foreground="white")])


def _nudge_texts_from_axis_center(ax, texts: Sequence, *, x_scale: float = 0.018, y_scale: float = 0.024) -> None:
    text_items = [text for text in texts if str(text.get_text()).strip()]
    if not text_items:
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = max(float(abs(x_max - x_min)), 1e-6)
    y_span = max(float(abs(y_max - y_min)), 1e-6)
    center_x = float((x_min + x_max) / 2.0)
    center_y = float((y_min + y_max) / 2.0)

    for idx, text in enumerate(text_items):
        x_pos, y_pos = text.get_position()
        dx = float(x_pos - center_x)
        dy = float(y_pos - center_y)
        if np.isclose(dx, 0.0):
            dx = 1.0 if idx % 2 == 0 else -1.0
        if np.isclose(dy, 0.0):
            dy = 1.0 if idx % 3 != 0 else -1.0
        norm = max(np.hypot(dx, dy), 1e-6)
        text.set_position((x_pos + (dx / norm) * x_span * x_scale, y_pos + (dy / norm) * y_span * y_scale))


def _constrain_texts_to_axis(ax, texts: Sequence, *, x_pad: float = 0.03, y_pad: float = 0.04) -> None:
    text_items = [text for text in texts if str(text.get_text()).strip()]
    if not text_items:
        return

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_min, x_max = (float(min(x0, x1)), float(max(x0, x1)))
    y_min, y_max = (float(min(y0, y1)), float(max(y0, y1)))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_low = x_min + x_span * x_pad
    x_high = x_max - x_span * x_pad
    y_low = y_min + y_span * y_pad
    y_high = y_max - y_span * y_pad
    x_align_low = x_min + x_span * (x_pad + 0.04)
    x_align_high = x_max - x_span * (x_pad + 0.04)
    y_align_low = y_min + y_span * (y_pad + 0.05)
    y_align_high = y_max - y_span * (y_pad + 0.05)

    for text in text_items:
        x_pos, y_pos = text.get_position()
        x_new = min(max(float(x_pos), x_low), x_high)
        y_new = min(max(float(y_pos), y_low), y_high)
        text.set_position((x_new, y_new))
        text.set_clip_on(True)

        if x_new >= x_align_high:
            text.set_ha("right")
        elif x_new <= x_align_low:
            text.set_ha("left")
        else:
            text.set_ha("center")

        if y_new >= y_align_high:
            text.set_va("top")
        elif y_new <= y_align_low:
            text.set_va("bottom")
        else:
            text.set_va("center")


def _repel_text_labels(ax, texts: Sequence, *, arrow: bool = False) -> None:
    text_items = [text for text in texts if str(text.get_text()).strip()]
    if not text_items:
        return

    _nudge_texts_from_axis_center(ax, text_items)

    try:
        from adjustText import adjust_text
    except ImportError:
        pass
    else:
        adjust_text(
            text_items,
            ax=ax,
            expand_points=(1.2, 1.28),
            expand_text=(1.18, 1.35),
            force_points=0.5,
            force_text=0.5,
            ensure_inside_axes=False,
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "color": "#8A8A8A",
                    "alpha": 0.55,
                    "lw": 0.6,
                }
                if arrow
                else None
            ),
        )

    _constrain_texts_to_axis(ax, text_items)
    _outline_texts(text_items)


def _style_scatter_axis(ax, *, max_marker_size: float | None = None, arrow: bool = False) -> None:
    if max_marker_size is not None:
        for collection in ax.collections:
            if not hasattr(collection, "get_sizes"):
                continue
            sizes = np.asarray(collection.get_sizes(), dtype=float)
            if sizes.size == 0:
                continue
            if float(sizes.max()) > float(max_marker_size):
                scaled = np.sqrt(np.clip(sizes, a_min=0.0, a_max=None))
                if float(scaled.max()) > 0.0:
                    scaled = 36.0 + (scaled / float(scaled.max())) * (float(max_marker_size) - 36.0)
                collection.set_sizes(np.clip(scaled, 24.0, float(max_marker_size)))
    _repel_text_labels(ax, ax.texts, arrow=arrow)
    ax.margins(x=0.08, y=0.10)


def _draw_role_scatter_style(
    ax,
    *,
    x_values,
    y_values,
    labels: Sequence[str],
    colors,
    sizes,
    xlabel: str,
    ylabel: str,
    title: str,
    add_zero_guides: bool = False,
) -> None:
    ax.scatter(
        x_values,
        y_values,
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )

    try:
        from adjustText import adjust_text

        texts = []
        for x_pos, y_pos, label in zip(x_values, y_values, labels):
            text = ax.text(
                x_pos,
                y_pos,
                str(label),
                fontsize=10,
                alpha=0.8,
                ha="center",
                va="center",
            )
            texts.append(text)

        adjust_text(
            texts,
            ax=ax,
            expand_points=(1.2, 1.2),
            expand_text=(1.2, 1.2),
            force_points=0.3,
            force_text=0.3,
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6, lw=0.8),
        )
    except ImportError:
        warnings.warn("adjustText library not found. Using default ax.annotate instead.")
        for x_pos, y_pos, label in zip(x_values, y_values, labels):
            ax.annotate(
                str(label),
                (x_pos, y_pos),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                alpha=0.8,
            )

    if add_zero_guides:
        ax.axhline(0.0, color="#B8B8B8", linewidth=1.0, linestyle="--", zorder=0)
        ax.axvline(0.0, color="#B8B8B8", linewidth=1.0, linestyle="--", zorder=0)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)


def _apply_category_tick_labels(ax, labels: Sequence[str], *, wrap_width: int = 18, rotation: float = 0.0) -> None:
    wrapped = [_wrap_plot_label(label, width=wrap_width) for label in labels]
    ax.set_xticks(ax.get_xticks(), labels=wrapped, rotation=rotation)
    for tick in ax.get_xticklabels():
        tick.set_ha("center" if rotation == 0 else "right")


def _apply_smart_category_tick_labels(ax, labels: Sequence[str], *, wrap_width: int = 18) -> None:
    labels = list(labels)
    if len(labels) <= 10:
        _apply_category_tick_labels(ax, labels, wrap_width=wrap_width, rotation=0.0)
        return

    rotation = 55.0
    tick_positions, tick_indices = _smart_tick_positions(len(labels), max_labels=10)
    tick_labels = [_wrap_plot_label(labels[idx], width=wrap_width) for idx in tick_indices]
    ax.set_xticks(tick_positions, labels=tick_labels, rotation=rotation)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")


def _smart_tick_positions(n_items: int, *, max_labels: int = 16) -> tuple[np.ndarray, list[int]]:
    if n_items <= 0:
        return np.array([], dtype=float), []
    if n_items <= max_labels:
        indices = list(range(n_items))
    else:
        step = max(int(np.ceil(n_items / max_labels)), 1)
        indices = list(range(0, n_items, step))
        if indices[-1] != n_items - 1:
            indices.append(n_items - 1)
    return np.array(indices, dtype=float), indices


def _dot_matrix_plot(
    matrix: pd.DataFrame,
    size_matrix: pd.DataFrame,
    *,
    color_label: str,
    title: str | None,
    cmap: str,
    figsize: tuple[float, float],
    border: bool,
    add_text: bool,
    row_palette: Mapping[str, str] | None = None,
    col_palette: Mapping[str, str] | None = None,
    row_totals: pd.Series | None = None,
    col_totals: pd.Series | None = None,
    row_total_label: str | None = None,
    col_total_label: str | None = None,
    top_annos: Sequence[str] = (),
    bottom_annos: Sequence[str] = (),
    left_annos: Sequence[str] = (),
    right_annos: Sequence[str] = (),
):
    ma, mp = _import_marsilea()
    display_color = matrix.copy()
    display_size = size_matrix.reindex(index=matrix.index, columns=matrix.columns, fill_value=0.0).copy()
    row_names = matrix.index.astype(str).tolist()
    col_names = matrix.columns.astype(str).tolist()
    display_color.index = _display_axis_labels(row_names, wrap_width=20)
    display_color.columns = _display_axis_labels(
        col_names,
        wrap_width=14,
        max_labels=14 if len(col_names) > 28 else 18,
    )
    display_size.index = display_color.index
    display_size.columns = display_color.columns

    plotter = ma.SizedHeatmap(
        size=display_size,
        color=display_color,
        cmap=cmap,
        width=max(float(figsize[0]) * 0.68, 3.0),
        height=max(float(figsize[1]) * 0.68, 2.5),
        legend=True,
        sizes=(40, 360) if border else (36, 340),
        linewidth=0.5 if border else 0.0,
        edgecolor="#2B2B2B" if border else "none",
        color_legend_kws=dict(title=color_label),
        size_legend_kws=dict(
            title="Significant interactions",
            show_at=[float(np.nanmin(display_size.to_numpy())), float(np.nanmax(display_size.to_numpy()))]
            if np.nanmax(display_size.to_numpy()) > 0
            else [0.0],
        ),
    )

    if row_totals is not None:
        row_totals = pd.Series(row_totals, index=matrix.index).reindex(matrix.index, fill_value=0.0)
    if col_totals is not None:
        col_totals = pd.Series(col_totals, index=matrix.columns).reindex(matrix.columns, fill_value=0.0)

    if "bar" in left_annos and row_totals is not None:
        plotter.add_left(
            mp.Numbers(
                row_totals.to_numpy(dtype=float),
                color="#F05454",
                label=row_total_label or "Total",
                show_value=False,
                label_props={"size": 11},
                props={"size": 11},
            )
        )
    if "cell" in left_annos and row_palette is not None:
        plotter.add_left(mp.Colors(row_names, palette=row_palette), size=0.2, legend=False)
    if "bar" in top_annos and col_totals is not None:
        plotter.add_top(
            mp.Numbers(
                col_totals.to_numpy(dtype=float),
                color="#4A90E2",
                label=col_total_label or "Total",
                show_value=False,
                label_props={"size": 11},
                props={"size": 11},
            )
        )
    if "cell" in top_annos and col_palette is not None:
        plotter.add_top(mp.Colors(col_names, palette=col_palette), size=0.2)
    if "cell" in right_annos and row_palette is not None:
        plotter.add_right(mp.Colors(row_names, palette=row_palette), size=0.12, legend=False)
    if "bar" in right_annos and row_totals is not None:
        plotter.add_right(
            mp.Numbers(
                row_totals.to_numpy(dtype=float),
                color="#D3B27B",
                show_value=False,
                label=None,
            )
        )
    if "cell" in bottom_annos and col_palette is not None:
        plotter.add_bottom(mp.Colors(col_names, palette=col_palette), size=0.12, legend=False)
    if "bar" in bottom_annos and col_totals is not None:
        plotter.add_bottom(
            mp.Numbers(
                col_totals.to_numpy(dtype=float),
                color="#BFC7D5",
                show_value=False,
                label=None,
            )
        )

    plotter.add_title(title or "Communication dot matrix")
    if hasattr(plotter, "add_legends"):
        plotter.add_legends()
    fig = _render_plot(plotter)
    ax = _largest_content_axis(fig)
    _attach_semantic_ticks(ax, row_names, col_names)

    if add_text:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    f"{matrix.iat[row_idx, col_idx]:.2g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    return fig, ax


def _draw_diff_circle_network(
    edge_df: pd.DataFrame,
    *,
    palette=None,
    figsize: tuple[float, float],
    title: str | None,
    up_color: str = "#C23B22",
    down_color: str = "#2F6DB3",
):
    nodes = sorted(set(edge_df["sender"]) | set(edge_df["receiver"]))
    graph = nx.DiGraph()
    for _, row in edge_df.iterrows():
        graph.add_edge(row["sender"], row["receiver"], weight=float(row["delta"]))

    pos = nx.circular_layout(graph)
    node_weights = {
        node: sum(abs(data["weight"]) for _, _, data in graph.in_edges(node, data=True))
        + sum(abs(data["weight"]) for _, _, data in graph.out_edges(node, data=True))
        for node in graph.nodes
    }
    max_node_weight = max(node_weights.values(), default=1.0)
    node_palette = _choose_palette(nodes, palette=palette)
    node_sizes = [350 + 1850 * (node_weights[node] / max_node_weight) for node in graph.nodes]
    edge_deltas = np.array([graph.edges[edge]["weight"] for edge in graph.edges], dtype=float)
    max_abs_delta = float(np.max(np.abs(edge_deltas))) if edge_deltas.size else 1.0
    edge_widths = 0.45 + 3.0 * (np.abs(edge_deltas) / max_abs_delta) if edge_deltas.size else []
    edge_colors = [up_color if delta >= 0 else down_color for delta in edge_deltas]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=[node_palette[node] for node in graph.nodes],
        linewidths=0.9,
        edgecolors="#3A3A3A",
        ax=ax,
        alpha=0.92,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        alpha=0.82,
        ax=ax,
        connectionstyle="arc3,rad=0.15",
    )
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax)
    ax.set_title(title or "Differential communication network")
    ax.set_axis_off()
    return fig, ax


@register_function(
    aliases=["ccc_heatmap", "communication_heatmap", "细胞通讯热图"],
    category="pl",
    description="Python-native cell-cell communication heatmap and dot-matrix plots for CellPhoneDB-style communication AnnData.",
    examples=[
        "ov.pl.ccc_heatmap(comm_adata, plot_type='heatmap')",
        "ov.pl.ccc_heatmap(comm_adata, plot_type='dot', display_by='interaction', top_n=20)",
        "ov.pl.ccc_heatmap(comm_adata, plot_type='role_heatmap', pattern='incoming')",
        "ov.pl.ccc_heatmap(comm_adata, signaling='MK', sender_use='Ductal')",
    ],
    related=["pl.ccc_network_plot", "pl.ccc_stat_plot", "pl.CellChatViz"],
)
def ccc_heatmap(
    adata: anndata.AnnData,
    *,
    plot_type: Literal[
        "heatmap",
        "focused_heatmap",
        "dot",
        "bubble",
        "bubble_lr",
        "pathway_bubble",
        "role_heatmap",
        "role_network",
        "role_network_marsilea",
        "diff_heatmap",
    ] = "heatmap",
    comparison_adata: anndata.AnnData | None = None,
    display_by: Literal["aggregation", "interaction"] = "aggregation",
    sender_use=None,
    receiver_use=None,
    signaling=None,
    interaction_use=None,
    pair_lr_use=None,
    pvalue_threshold: float = 0.05,
    pattern: Literal["outgoing", "incoming", "all"] = "all",
    color_by: Literal["score", "pvalue"] = "score",
    value: Literal["sum", "mean", "max", "count"] = "sum",
    top_n: int = 20,
    top_n_pairs: int | None = None,
    top_anno: str | Sequence[str] | None = "bar",
    bottom_anno: str | Sequence[str] | None = "cell",
    left_anno: str | Sequence[str] | None = "bar",
    right_anno: str | Sequence[str] | None = "cell",
    bar_value: Literal["count", "sum", "mean", "max"] = "sum",
    facet_by: Literal["sender", "receiver"] | None = None,
    pathway_method: str = "mean",
    min_lr_pairs: int = 1,
    min_expression: float = 0.1,
    group_pathways: bool = True,
    transpose: bool = False,
    add_violin: bool = False,
    remove_isolate: bool = False,
    min_interaction_threshold: float = 0.0,
    cluster_rows: bool = False,
    cluster_columns: bool = False,
    add_text: bool | None = None,
    border: bool = False,
    cmap: str = "Reds",
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
    show: bool = False,
    save: str | bool = False,
):
    """Plot communication matrices as heatmaps or dot/bubble matrices."""
    default_top_bar = top_anno == "bar"
    default_left_bar = left_anno == "bar"
    default_bottom_cell = bottom_anno == "cell"
    default_right_cell = right_anno == "cell"
    top_annos = _normalize_side_annotations(top_anno, default=("bar",))
    bottom_annos = _normalize_side_annotations(bottom_anno, default=("cell",))
    left_annos = _normalize_side_annotations(left_anno, default=("bar",))
    right_annos = _normalize_side_annotations(right_anno, default=("cell",))
    if plot_type in {"heatmap", "dot", "bubble"} and default_top_bar:
        top_annos = ["bar", "cell"]
    if plot_type in {"heatmap", "dot", "bubble"} and display_by == "aggregation" and default_left_bar:
        left_annos = ["bar", "cell"]
    if plot_type in {"heatmap", "dot", "bubble", "pathway_bubble"} and default_bottom_cell:
        bottom_annos = []
    if plot_type in {"heatmap", "dot", "bubble", "pathway_bubble"} and default_right_cell:
        right_annos = []
    if display_by == "interaction" and top_n_pairs is None:
        top_n_pairs = top_n

    if plot_type == "heatmap" and display_by == "aggregation":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            facet_by=facet_by,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
        )
        _ensure_allowed_metrics(
            plot_type,
            color_by=color_by,
            allowed_color_by=("score",),
            value=value,
            allowed_value=("sum", "mean"),
        )
        viz = _build_cellchatviz(adata)
        plotter = viz.netVisual_heatmap_marsilea(
            signaling=_normalize_use_arg(signaling),
            pvalue_threshold=pvalue_threshold,
            color_heatmap=cmap,
            add_dendrogram=cluster_rows or cluster_columns,
            add_row_sum=True,
            add_col_sum=True,
            title=title or "Communication heatmap",
        )
        fig, ax = _render_plotter_figure(plotter, title=None)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "focused_heatmap":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            facet_by=facet_by,
        )
        _ensure_allowed_metrics(
            plot_type,
            color_by=color_by,
            allowed_color_by=("score",),
            value=value,
            allowed_value=("sum", "mean"),
        )
        viz = _build_cellchatviz(adata)
        plotter = viz.netVisual_heatmap_marsilea_focused(
            signaling=_normalize_use_arg(signaling),
            pvalue_threshold=pvalue_threshold,
            min_interaction_threshold=min_interaction_threshold,
            color_heatmap=cmap,
            add_dendrogram=cluster_rows or cluster_columns,
            add_row_sum=True,
            add_col_sum=True,
            figsize=figsize,
            title=title or "Focused communication heatmap",
        )
        fig, ax = _render_plotter_figure(plotter, title=None)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "role_heatmap":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            facet_by=facet_by,
        )
        _ensure_allowed_metrics(
            plot_type,
            color_by=color_by,
            allowed_color_by=("score",),
            value=value,
            allowed_value=("sum", "mean"),
        )
        viz = _build_cellchatviz(adata)
        role_pattern = "overall" if pattern == "all" else pattern
        signaling_values = _normalize_use_arg(signaling)
        if signaling_values is None and top_n is not None and top_n > 0:
            pathway_summary = _pathway_summary_table(
                adata,
                signaling=None,
                pathway_method=pathway_method,
                min_lr_pairs=min_lr_pairs,
                min_expression=min_expression,
                pvalue_threshold=pvalue_threshold,
            )
            signaling_values = pathway_summary["pathway"].astype(str).head(int(top_n)).tolist()
        plotter, _, _ = viz.netAnalysis_signalingRole_heatmap(
            pattern=role_pattern,
            signaling=signaling_values,
            row_scale=False,
            figsize=figsize,
            cmap=cmap,
            show_totals=True,
            title=title or f"Communication role heatmap ({role_pattern})",
        )
        fig, ax = _render_plotter_figure(plotter, title=None)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "role_network":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            facet_by=facet_by,
        )
        _ensure_allowed_metrics(
            plot_type,
            color_by=color_by,
            allowed_color_by=("score",),
            value=value,
            allowed_value=("sum", "mean"),
        )
        viz = _build_cellchatviz(adata)
        fig = viz.netAnalysis_signalingRole_network(
            signaling=_normalize_use_arg(signaling),
            measures=None,
            color_heatmap=cmap,
            width=figsize[0],
            height=figsize[1],
            title=title or "Signaling role network heatmap",
            save=None,
            show_values=True,
        )
        ax = _largest_content_axis(fig)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "role_network_marsilea":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            facet_by=facet_by,
        )
        _ensure_allowed_metrics(
            plot_type,
            color_by=color_by,
            allowed_color_by=("score",),
            value=value,
            allowed_value=("sum", "mean"),
        )
        viz = _build_cellchatviz(adata)
        plotter = viz.netAnalysis_signalingRole_network_marsilea(
            signaling=_normalize_use_arg(signaling),
            measures=None,
            color_heatmap=cmap,
            width=figsize[0],
            height=figsize[1],
            title=title or "Signaling role network heatmap",
            add_dendrogram=True,
            add_cell_colors=True,
            add_importance_bars=True,
            show_values=True,
            save=None,
        )
        fig, ax = _render_plotter_figure(plotter, title=None)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "pathway_bubble":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            facet_by=facet_by,
        )
        _ensure_allowed_metrics(
            plot_type,
            color_by=color_by,
            allowed_color_by=("score",),
            value=value,
            allowed_value=("sum", "mean", "count"),
        )
        signaling_values = _require_use_argument(plot_type, "signaling", signaling)
        viz = _build_cellchatviz(adata)
        plotter = viz.netVisual_bubble_marsilea(
            sources_use=sender_use,
            targets_use=receiver_use,
            signaling=signaling_values,
            pvalue_threshold=pvalue_threshold,
            mean_threshold=min_expression,
            top_interactions=top_n,
            show_pvalue=True,
            show_mean=True,
            show_count=(value == "count"),
            add_violin=add_violin,
            add_dendrogram=cluster_rows or cluster_columns,
            group_pathways=group_pathways,
            figsize=figsize,
            title=title or "Pathway bubble summary",
            remove_isolate=remove_isolate,
            cmap=cmap,
            transpose=transpose,
            show_sender_colors=True,
            show_receiver_colors=False,
        )
        if plotter is None:
            raise ValueError("No communication records remain after filtering.")
        fig, ax = _render_plotter_figure(plotter, title=None)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "bubble_lr":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            signaling=signaling,
            facet_by=facet_by,
        )
        _ensure_allowed_metrics(
            plot_type,
            color_by=color_by,
            allowed_color_by=("score",),
            value=value,
            allowed_value=("sum", "mean", "count"),
        )
        lr_pairs = _resolve_lr_pairs(
            adata,
            pair_lr_use=pair_lr_use,
            interaction_use=interaction_use,
        )
        if lr_pairs is None:
            raise ValueError("`pair_lr_use` or `interaction_use` is required when `plot_type='bubble_lr'`.")
        viz = _build_cellchatviz(adata)
        plotter = viz.netVisual_bubble_lr(
            sources_use=sender_use,
            targets_use=receiver_use,
            lr_pairs=lr_pairs[0] if len(lr_pairs) == 1 else lr_pairs,
            pvalue_threshold=max(float(pvalue_threshold), 0.0),
            mean_threshold=min_expression,
            show_all_pairs=True,
            show_pvalue=True,
            show_mean=True,
            show_count=(value == "count"),
            add_violin=add_violin,
            add_dendrogram=cluster_rows or cluster_columns,
            figsize=figsize,
            title=title or "Ligand-receptor bubble plot",
            remove_isolate=remove_isolate,
            cmap=cmap,
            transpose=transpose,
            show_sender_colors=True,
            show_receiver_colors=False,
        )
        if plotter is None:
            raise ValueError("No communication records remain after filtering.")
        fig, ax = _render_plotter_figure(plotter, title=None)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "diff_heatmap":
        if comparison_adata is None:
            raise ValueError("`comparison_adata` is required when `plot_type='diff_heatmap'`.")
        matrix, color_label = _diff_role_matrix(
            adata,
            comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            signaling=signaling,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            pvalue_threshold=pvalue_threshold,
            pattern=pattern,
            color_by=color_by,
            value=value,
            top_n=top_n,
        )
        matrix = _apply_cluster_order(matrix, cluster_rows=cluster_rows, cluster_columns=cluster_columns)
        if add_text is None:
            finite_values = matrix.to_numpy(dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]
            unique_count = int(np.unique(np.round(finite_values, 8)).size) if finite_values.size else 0
            add_text = matrix.size <= 36 or (matrix.size <= 100 and unique_count <= 4)
        diff_metric = "count" if bar_value == "count" else "sum"
        row_annotation = _matrix_side_totals(matrix.abs(), None, axis="rows", metric=diff_metric)
        col_annotation = _matrix_side_totals(matrix.abs(), None, axis="columns", metric=diff_metric)
        diff_cmap = "RdBu_r" if cmap == "Reds" else cmap
        fig, ax = _plot_heatmap_matrix(
            matrix,
            color_label=color_label,
            title=title or f"Differential communication heatmap ({color_label})",
            cmap=diff_cmap,
            figsize=figsize,
            border=border,
            add_text=bool(add_text),
            center=0.0,
            row_palette=_choose_palette(matrix.index.astype(str).tolist()),
            col_palette=_choose_palette(matrix.columns.astype(str).tolist()),
            row_totals=row_annotation,
            col_totals=col_annotation,
            row_total_label="Delta total",
            col_total_label="Delta total",
            top_annos=top_annos,
            bottom_annos=bottom_annos,
            left_annos=left_annos,
            right_annos=right_annos,
            max_col_labels=14,
            clip_quantile=0.95,
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    long_df = _communication_long_table(
        adata,
        sender_use=sender_use,
        receiver_use=receiver_use,
        signaling=signaling,
        interaction_use=interaction_use,
        pair_lr_use=pair_lr_use,
        pvalue_threshold=pvalue_threshold,
    )
    matrix, size_matrix, color_label = _communication_matrix(
        long_df,
        display_by=display_by,
        color_by=color_by,
        value=value,
        top_n=top_n,
        top_n_pairs=top_n_pairs,
        facet_by=facet_by,
    )
    display_color_label = "Interaction Strength" if display_by == "aggregation" and color_by == "score" else color_label
    matrix = _apply_cluster_order(matrix, cluster_rows=cluster_rows, cluster_columns=cluster_columns)
    size_matrix = size_matrix.reindex(index=matrix.index, columns=matrix.columns, fill_value=0.0)

    if add_text is None:
        add_text = plot_type == "heatmap" and display_by == "aggregation" and matrix.size <= 49

    if plot_type in {"dot", "bubble"}:
        fig, ax = _dot_matrix_plot(
            matrix,
            size_matrix,
            color_label=display_color_label,
            title=title,
            cmap=cmap,
            figsize=figsize,
            border=border,
            add_text=bool(add_text),
            row_palette=_choose_palette(matrix.index.astype(str).tolist()) if display_by == "aggregation" else None,
            col_palette=(
                _choose_palette(matrix.columns.astype(str).tolist())
                if display_by == "aggregation"
                else _pair_color_metadata(matrix.columns.astype(str).tolist())
            ),
            row_totals=_matrix_side_totals(matrix, size_matrix, axis="rows", metric=bar_value),
            col_totals=_matrix_side_totals(matrix, size_matrix, axis="columns", metric=bar_value),
            row_total_label="Incoming" if display_by == "aggregation" else "Interaction total",
            col_total_label="Outgoing" if display_by == "aggregation" else "Pair total",
            top_annos=top_annos,
            bottom_annos=bottom_annos,
            left_annos=left_annos if display_by == "aggregation" else tuple(item for item in left_annos if item != "cell"),
            right_annos=right_annos if display_by == "aggregation" else tuple(item for item in right_annos if item != "cell"),
        )
        default_title = "Communication bubble plot" if plot_type == "bubble" else "Communication dot matrix"
        ax.set_title(title or f"{default_title} ({display_color_label})")
        return _maybe_save_show(fig, show=show, save=save), ax

    fig, ax = _plot_heatmap_matrix(
        matrix,
        color_label=display_color_label,
        title=title or f"Communication heatmap ({display_color_label})",
        cmap=cmap,
        figsize=figsize,
        border=border,
        add_text=bool(add_text),
        row_palette=_choose_palette(matrix.index.astype(str).tolist()),
        col_palette=(
            _choose_palette(matrix.columns.astype(str).tolist())
            if display_by == "aggregation"
            else _pair_color_metadata(matrix.columns.astype(str).tolist())
        ),
        row_totals=_matrix_side_totals(matrix, size_matrix, axis="rows", metric=bar_value),
        col_totals=_matrix_side_totals(matrix, size_matrix, axis="columns", metric=bar_value),
        row_total_label="Incoming" if display_by == "aggregation" else "Interaction total",
        col_total_label="Outgoing" if display_by == "aggregation" else "Pair total",
        top_annos=top_annos,
        bottom_annos=bottom_annos,
        left_annos=left_annos,
        right_annos=right_annos,
        col_wrap_width=14 if display_by == "interaction" else 18,
        max_col_labels=9 if display_by == "interaction" else 16,
        clip_quantile=0.9 if display_by == "interaction" and color_by == "score" else (0.95 if color_by == "score" else None),
    )
    return _maybe_save_show(fig, show=show, save=save), ax


def _aggregated_pair_frame(
    adata: anndata.AnnData,
    *,
    sender_use=None,
    receiver_use=None,
    signaling=None,
    interaction_use=None,
    pair_lr_use=None,
    pvalue_threshold: float = 0.05,
    value: Literal["sum", "mean", "max", "count"] = "sum",
) -> pd.DataFrame:
    long_df = _communication_long_table(
        adata,
        sender_use=sender_use,
        receiver_use=receiver_use,
        signaling=signaling,
        interaction_use=interaction_use,
        pair_lr_use=pair_lr_use,
        pvalue_threshold=pvalue_threshold,
    )
    grouped = long_df.groupby(["sender", "receiver"], observed=True)
    agg = _aggregate_series(grouped, value)
    edge_df = agg.rename("weight").reset_index()
    edge_df = edge_df.loc[edge_df["weight"] > 0].sort_values("weight", ascending=False)
    return edge_df


def _draw_circle_network(
    edge_df: pd.DataFrame,
    *,
    palette,
    figsize: tuple[float, float],
    title: str | None,
):
    nodes = sorted(set(edge_df["sender"]) | set(edge_df["receiver"]))
    colors = _choose_palette(nodes, palette=palette)

    graph = nx.DiGraph()
    for _, row in edge_df.iterrows():
        graph.add_edge(row["sender"], row["receiver"], weight=float(row["weight"]))

    pos = nx.circular_layout(graph)
    node_weights = {
        node: sum(data["weight"] for _, _, data in graph.in_edges(node, data=True))
        + sum(data["weight"] for _, _, data in graph.out_edges(node, data=True))
        for node in graph.nodes
    }
    max_node_weight = max(node_weights.values(), default=1.0)
    node_sizes = [300 + 1800 * (node_weights[node] / max_node_weight) for node in graph.nodes]
    edge_weights = np.array([graph.edges[edge]["weight"] for edge in graph.edges], dtype=float)
    max_edge_weight = float(edge_weights.max()) if edge_weights.size else 1.0
    edge_widths = 0.6 + 5.0 * (edge_weights / max_edge_weight) if edge_weights.size else []

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=[colors[node] for node in graph.nodes],
        linewidths=1.0,
        edgecolors="white",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color=[colors[source] for source, _ in graph.edges],
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        alpha=0.8,
        ax=ax,
        connectionstyle="arc3,rad=0.15",
    )
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax)
    ax.set_title(title or "Communication network")
    ax.set_axis_off()
    return fig, ax


def _draw_arrow_network(
    edge_df: pd.DataFrame,
    *,
    node_df: pd.DataFrame,
    column_titles: Sequence[tuple[float, str]],
    palette,
    figsize: tuple[float, float],
    title: str | None,
):
    cell_labels = node_df.loc[node_df["column"].isin(["sender", "receiver"]), "label"].astype(str).tolist()
    colors = _choose_palette(list(dict.fromkeys(cell_labels)), palette=palette)
    fig, ax = plt.subplots(figsize=figsize)
    max_weight = float(edge_df["weight"].max()) if not edge_df.empty else 1.0

    for _, row in edge_df.iterrows():
        linewidth = 0.8 + 4.2 * (float(row["weight"]) / max_weight)
        patch = FancyArrowPatch(
            (float(row["x_from"]), float(row["y_from"])),
            (float(row["x_to"]), float(row["y_to"])),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=linewidth,
            color=colors.get(str(row["edge_group"]), "#6C757D"),
            alpha=0.75,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(patch)

    node_max_weight = float(node_df["weight"].max()) if not node_df.empty else 1.0
    for _, row in node_df.iterrows():
        label = str(row["label"])
        x_coord = float(row["x"])
        y_coord = float(row["y"])
        node_size = 220 + 720 * (float(row["weight"]) / max(node_max_weight, 1e-9))
        if row["column"] in {"sender", "receiver"}:
            ax.scatter(
                x_coord,
                y_coord,
                s=node_size,
                c=colors.get(label, "#BDBDBD"),
                edgecolors="white",
                linewidths=1.0,
                zorder=3,
            )
            text_x = x_coord - 0.04 if row["column"] == "sender" else x_coord + 0.04
            ax.text(
                text_x,
                y_coord,
                _wrap_plot_label(label, width=16),
                ha="right" if row["column"] == "sender" else "left",
                va="center",
                fontsize=10,
            )
        else:
            _draw_flow_stage_node(
                ax,
                x_coord=x_coord,
                y_coord=y_coord,
                label=label,
                weight=float(row["weight"]),
                node_max_weight=node_max_weight,
                wrap_width=16,
            )

    for x_coord, label in column_titles:
        ax.text(x_coord, 1.06, label, ha="center", va="bottom", fontsize=11, transform=ax.transData)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.08, 1.12)
    ax.set_title(title or "Communication flow")
    ax.set_axis_off()
    return fig, ax


def _draw_sigmoid_network(
    edge_df: pd.DataFrame,
    *,
    node_df: pd.DataFrame,
    column_titles: Sequence[tuple[float, str]],
    palette,
    figsize: tuple[float, float],
    title: str | None,
):
    cell_labels = node_df.loc[node_df["column"].isin(["sender", "receiver"]), "label"].astype(str).tolist()
    colors = _choose_palette(list(dict.fromkeys(cell_labels)), palette=palette)
    fig, ax = plt.subplots(figsize=figsize)
    max_weight = float(edge_df["weight"].max()) if not edge_df.empty else 1.0

    for _, row in edge_df.iterrows():
        linewidth = 0.8 + 4.2 * (float(row["weight"]) / max_weight)
        xs = np.linspace(float(row["x_from"]), float(row["x_to"]), 80)
        sigmoid = 1.0 / (1.0 + np.exp(-np.linspace(-6.0, 6.0, 80)))
        sigmoid = (sigmoid - sigmoid.min()) / max(sigmoid.max() - sigmoid.min(), 1e-9)
        ys = float(row["y_from"]) + (float(row["y_to"]) - float(row["y_from"])) * sigmoid
        edge_color = colors.get(str(row["edge_group"]), "#6C757D")
        ax.plot(
            xs,
            ys,
            color=edge_color,
            linewidth=linewidth,
            alpha=0.75,
            solid_capstyle="round",
        )
        ax.add_patch(
            FancyArrowPatch(
                (xs[-2], ys[-2]),
                (xs[-1], ys[-1]),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=0.0,
                color=edge_color,
                alpha=0.75,
            )
        )

    node_max_weight = float(node_df["weight"].max()) if not node_df.empty else 1.0
    for _, row in node_df.iterrows():
        label = str(row["label"])
        x_coord = float(row["x"])
        y_coord = float(row["y"])
        node_size = 220 + 720 * (float(row["weight"]) / max(node_max_weight, 1e-9))
        if row["column"] in {"sender", "receiver"}:
            ax.scatter(
                x_coord,
                y_coord,
                s=node_size,
                c=colors.get(label, "#BDBDBD"),
                edgecolors="white",
                linewidths=1.0,
                zorder=3,
            )
            text_x = x_coord - 0.04 if row["column"] == "sender" else x_coord + 0.04
            ax.text(
                text_x,
                y_coord,
                _wrap_plot_label(label, width=16),
                ha="right" if row["column"] == "sender" else "left",
                va="center",
                fontsize=10,
            )
        else:
            _draw_flow_stage_node(
                ax,
                x_coord=x_coord,
                y_coord=y_coord,
                label=label,
                weight=float(row["weight"]),
                node_max_weight=node_max_weight,
                wrap_width=16,
            )

    for x_coord, label in column_titles:
        ax.text(x_coord, 1.06, label, ha="center", va="bottom", fontsize=11, transform=ax.transData)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.08, 1.12)
    ax.set_title(title or "Communication sigmoid flow")
    ax.set_axis_off()
    return fig, ax


def _choose_focus_ligand(long_df: pd.DataFrame) -> str:
    ligand_scores = long_df.loc[long_df["ligand"].astype(str).ne(""), :].groupby("ligand", observed=True)["score"].sum()
    if ligand_scores.empty:
        raise ValueError("No ligand annotations remain after filtering.")
    return str(ligand_scores.sort_values(ascending=False).index[0])


def _draw_bipartite_network(
    long_df: pd.DataFrame,
    *,
    ligand: str | None,
    receptor=None,
    top_n: int | None,
    palette,
    figsize: tuple[float, float],
    title: str | None,
):
    if long_df.empty:
        raise ValueError("No communication records remain after filtering.")

    ligand_name = str(ligand) if ligand is not None else _choose_focus_ligand(long_df)
    data = long_df.loc[long_df["ligand"].astype(str) == ligand_name].copy()
    if receptor is not None:
        receptor_use = _normalize_use_arg(receptor)
        data = data.loc[data["receptor"].astype(str).isin(receptor_use)]
    if data.empty:
        raise ValueError(f"No communication records remain for ligand '{ligand_name}'.")

    if top_n is not None and top_n > 0:
        sender_scores = data.groupby("sender", observed=True)["score"].sum().sort_values(ascending=False)
        receiver_scores = data.groupby("receiver", observed=True)["score"].sum().sort_values(ascending=False)
        keep_senders = sender_scores.head(int(top_n)).index.astype(str).tolist()
        keep_receivers = receiver_scores.head(int(top_n)).index.astype(str).tolist()
        data = data.loc[data["sender"].astype(str).isin(keep_senders) & data["receiver"].astype(str).isin(keep_receivers)].copy()
    if data.empty:
        raise ValueError(f"No communication records remain after top_n filtering for ligand '{ligand_name}'.")

    senders = data.groupby("sender", observed=True)["score"].sum().sort_values(ascending=False).index.astype(str).tolist()
    receptors = data.groupby("receptor", observed=True)["score"].sum().sort_values(ascending=False).index.astype(str).tolist()
    receivers = data.groupby("receiver", observed=True)["score"].sum().sort_values(ascending=False).index.astype(str).tolist()
    cell_types = list(dict.fromkeys(senders + receivers))
    cell_colors = _choose_palette(cell_types, palette=palette)

    sender_totals = data.groupby("sender", observed=True)["score"].sum().reindex(senders).fillna(0.0)
    receptor_totals = data.groupby("receptor", observed=True)["score"].sum().reindex(receptors).fillna(0.0)
    receiver_totals = data.groupby("receiver", observed=True)["score"].sum().reindex(receivers).fillna(0.0)

    sender_pos = {label: (0.0, pos[1]) for label, pos in _weighted_positions(senders, sender_totals).items()}
    sender_center = np.median([value[1] for value in sender_pos.values()]) if sender_pos else 0.5
    receiver_pos = {label: (3.0, pos[1]) for label, pos in _weighted_positions(receivers, receiver_totals).items()}
    receiver_center = np.median([value[1] for value in receiver_pos.values()]) if receiver_pos else sender_center
    bridge_center = float((sender_center + receiver_center) / 2.0)
    ligand_pos = {ligand_name: (1.0, bridge_center)}
    receptor_base_pos = {label: (2.0, pos[1]) for label, pos in _weighted_positions(receptors, receptor_totals).items()}
    if receptor_base_pos:
        receptor_center = np.median([value[1] for value in receptor_base_pos.values()])
        receptor_shift = bridge_center - float(receptor_center)
        receptor_pos = {label: (2.0, float(pos[1] + receptor_shift)) for label, pos in receptor_base_pos.items()}
    else:
        receptor_pos = {}

    sender_to_ligand = _aggregate_series(data.groupby(["sender", "ligand"], observed=True), "sum").rename("weight").reset_index()
    ligand_to_receptor = _aggregate_series(data.groupby(["ligand", "receptor"], observed=True), "sum").rename("weight").reset_index()
    receptor_to_receiver = _aggregate_series(data.groupby(["receptor", "receiver"], observed=True), "sum").rename("weight").reset_index()
    edge_max = max(
        float(sender_to_ligand["weight"].max()) if not sender_to_ligand.empty else 0.0,
        float(ligand_to_receptor["weight"].max()) if not ligand_to_receptor.empty else 0.0,
        float(receptor_to_receiver["weight"].max()) if not receptor_to_receiver.empty else 0.0,
        1.0,
    )

    fig, ax = plt.subplots(figsize=figsize)
    for _, row in sender_to_ligand.iterrows():
        linewidth = 0.8 + 4.8 * (float(row["weight"]) / edge_max)
        ax.add_patch(
            FancyArrowPatch(
                sender_pos[str(row["sender"])],
                ligand_pos[str(row["ligand"])],
                arrowstyle="-",
                linewidth=linewidth,
                color=cell_colors.get(str(row["sender"]), "#6C757D"),
                alpha=0.62,
                connectionstyle="arc3,rad=0.0",
            )
        )
    for _, row in ligand_to_receptor.iterrows():
        linewidth = 0.5 + 2.0 * (float(row["weight"]) / edge_max)
        ax.add_patch(
            FancyArrowPatch(
                ligand_pos[str(row["ligand"])],
                receptor_pos[str(row["receptor"])],
                arrowstyle="-",
                linewidth=linewidth,
                linestyle="--",
                color="#777777",
                alpha=0.75,
                connectionstyle="arc3,rad=0.0",
            )
        )
    sender_lookup = (
        data.groupby(["receptor", "receiver", "sender"], observed=True)["score"]
        .sum()
        .rename("sender_weight")
        .reset_index()
        .sort_values(["receptor", "receiver", "sender_weight"], ascending=[True, True, False])
        .drop_duplicates(["receptor", "receiver"])
        .set_index(["receptor", "receiver"])["sender"]
    )
    for _, row in receptor_to_receiver.iterrows():
        linewidth = 0.8 + 4.8 * (float(row["weight"]) / edge_max)
        sender_name = str(sender_lookup.get((row["receptor"], row["receiver"]), row["receiver"]))
        ax.add_patch(
            FancyArrowPatch(
                receptor_pos[str(row["receptor"])],
                receiver_pos[str(row["receiver"])],
                arrowstyle="-",
                linewidth=linewidth,
                color=cell_colors.get(sender_name, "#6C757D"),
                alpha=0.62,
                connectionstyle="arc3,rad=0.0",
            )
        )

    bridge_label_offset = 0.065
    for sender, (x_coord, y_coord) in sender_pos.items():
        ax.scatter(x_coord, y_coord, s=280, c=cell_colors[sender], edgecolors="white", linewidths=1.0, zorder=3)
        ax.text(x_coord - 0.08, y_coord, sender, ha="right", va="center", fontsize=10)
    for ligand_label, (x_coord, y_coord) in ligand_pos.items():
        ax.scatter(x_coord, y_coord, s=240, c="white", edgecolors="#5C5C5C", linewidths=1.0, marker="s", zorder=3)
        ax.text(
            x_coord,
            y_coord + bridge_label_offset,
            _wrap_plot_label(ligand_label.replace("_", "\n"), width=16),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for receptor_label, (x_coord, y_coord) in receptor_pos.items():
        ax.scatter(x_coord, y_coord, s=220, c="white", edgecolors="#5C5C5C", linewidths=1.0, marker="s", zorder=3)
        ax.text(
            x_coord,
            y_coord + bridge_label_offset,
            _wrap_plot_label(receptor_label.replace("_", "\n"), width=14),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    for receiver, (x_coord, y_coord) in receiver_pos.items():
        ax.scatter(x_coord, y_coord, s=280, c=cell_colors[receiver], edgecolors="white", linewidths=1.0, zorder=3)
        ax.text(x_coord + 0.08, y_coord, receiver, ha="left", va="center", fontsize=10)

    for x_coord, label in ((0.0, "Sender"), (1.0, "Ligand"), (2.0, "Receptor"), (3.0, "Receiver")):
        ax.text(x_coord, 1.06, label, ha="center", va="bottom", fontsize=11)

    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(-0.1, 1.18)
    ax.set_title(title or "Communication bipartite network")
    ax.set_axis_off()
    return fig, ax


def _coerce_node_position_frame(node_positions) -> pd.DataFrame:
    if node_positions is None:
        raise ValueError("`node_positions` is required for `plot_type='embedding_network'`.")

    if isinstance(node_positions, pd.DataFrame):
        frame = node_positions.copy()
        if {"x", "y"}.issubset(frame.columns):
            coords = frame.loc[:, ["x", "y"]]
        elif frame.shape[1] >= 2:
            coords = frame.iloc[:, :2].copy()
            coords.columns = ["x", "y"]
        else:
            raise ValueError("`node_positions` DataFrame must contain `x` and `y` columns.")
    elif isinstance(node_positions, Mapping):
        frame = pd.DataFrame.from_dict(node_positions, orient="index")
        if frame.shape[1] < 2:
            raise ValueError("Each `node_positions` entry must provide at least two coordinates.")
        coords = frame.iloc[:, :2].copy()
        coords.columns = ["x", "y"]
    else:
        raise ValueError(
            "`node_positions` must be a mapping of cell types to (x, y) coordinates "
            "or a DataFrame indexed by cell type."
        )

    coords.index = coords.index.astype(str)
    coords = coords.astype(float)
    return coords


def _resolve_node_positions(
    adata: anndata.AnnData,
    nodes: Sequence[str],
    node_positions=None,
) -> pd.DataFrame:
    positions = node_positions
    if positions is None:
        for key in ("node_positions", "ccc_node_positions", "cell_type_positions"):
            if key in adata.uns:
                positions = adata.uns[key]
                break
    coords = _coerce_node_position_frame(positions)
    missing = [node for node in nodes if node not in coords.index]
    if missing:
        raise ValueError(
            "Missing coordinates for cell types: "
            f"{missing}. Provide `node_positions` or store them in `adata.uns['node_positions']`."
        )
    return coords.loc[list(nodes), ["x", "y"]].copy()


def _embedding_label_positions(coords: pd.DataFrame) -> dict[str, tuple[float, float, str, str]]:
    center_x = float(coords["x"].mean())
    center_y = float(coords["y"].mean())
    span_x = float(coords["x"].max() - coords["x"].min())
    span_y = float(coords["y"].max() - coords["y"].min())
    offset = max(span_x, span_y, 1.0) * 0.06
    labels = {}

    for node, row in coords.iterrows():
        dx = float(row["x"] - center_x)
        dy = float(row["y"] - center_y)
        if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
            dx, dy = 0.0, 1.0
        norm = max(np.hypot(dx, dy), 1e-6)
        ux = dx / norm
        uy = dy / norm
        ha = "left" if ux >= 0 else "right"
        va = "bottom" if uy >= 0 else "top"
        labels[str(node)] = (float(row["x"] + ux * offset), float(row["y"] + uy * offset), ha, va)
    return labels


def _coerce_embedding_frame(embedding_points) -> pd.DataFrame | None:
    if embedding_points is None:
        return None
    if isinstance(embedding_points, pd.DataFrame):
        frame = embedding_points.copy()
    else:
        frame = pd.DataFrame(embedding_points).copy()
    required = {"x", "y", "cell_type"}
    if not required.issubset(frame.columns):
        raise ValueError("`embedding_points` must provide `x`, `y`, and `cell_type` columns.")
    frame = frame.loc[:, ["x", "y", "cell_type"]].copy()
    frame["x"] = frame["x"].astype(float)
    frame["y"] = frame["y"].astype(float)
    frame["cell_type"] = frame["cell_type"].astype(str)
    return frame


def _resolve_embedding_points(adata: anndata.AnnData, embedding_points=None) -> pd.DataFrame | None:
    points = embedding_points
    if points is None:
        for key in ("embedding_points", "ccc_embedding_points", "cell_embedding"):
            if key in adata.uns:
                points = adata.uns[key]
                break
    return _coerce_embedding_frame(points)


def _draw_embedding_network(
    adata: anndata.AnnData,
    edge_df: pd.DataFrame,
    *,
    node_positions=None,
    embedding_points=None,
    palette,
    figsize: tuple[float, float],
    title: str | None,
):
    nodes = sorted(set(edge_df["sender"]) | set(edge_df["receiver"]))
    coords = _resolve_node_positions(adata, nodes, node_positions=node_positions)
    colors = _choose_palette(nodes, palette=palette)
    background = _resolve_embedding_points(adata, embedding_points=embedding_points)

    graph = nx.DiGraph()
    for _, row in edge_df.iterrows():
        graph.add_edge(str(row["sender"]), str(row["receiver"]), weight=float(row["weight"]))

    node_weights = {
        node: sum(data["weight"] for _, _, data in graph.in_edges(node, data=True))
        + sum(data["weight"] for _, _, data in graph.out_edges(node, data=True))
        for node in graph.nodes
    }
    max_node_weight = max(node_weights.values(), default=1.0)
    edge_weights = np.array([graph.edges[edge]["weight"] for edge in graph.edges], dtype=float)
    max_edge_weight = float(edge_weights.max()) if edge_weights.size else 1.0
    label_positions = _embedding_label_positions(coords)

    span_x = float(coords["x"].max() - coords["x"].min())
    span_y = float(coords["y"].max() - coords["y"].min())
    pad = max(span_x, span_y, 1.0) * 0.18
    loop_scale = max(span_x, span_y, 1.0) * 0.05

    fig, ax = plt.subplots(figsize=figsize)
    if background is not None and not background.empty:
        background_types = list(dict.fromkeys(background["cell_type"].tolist()))
        background_colors = _choose_palette(background_types, palette=palette)
        for cell_type, group in background.groupby("cell_type", sort=False, observed=True):
            ax.scatter(
                group["x"],
                group["y"],
                s=10,
                c=background_colors[str(cell_type)],
                alpha=0.9,
                linewidths=0.0,
                zorder=0,
                label=f"{cell_type}({len(group)})",
            )

    for source, target, data in graph.edges(data=True):
        start = coords.loc[source]
        end = coords.loc[target]
        linewidth = 0.8 + 4.8 * (float(data["weight"]) / max_edge_weight)
        if source == target:
            loop_center_x = float(start["x"])
            loop_center_y = float(start["y"]) + loop_scale * 0.2
            patch = FancyArrowPatch(
                (loop_center_x - loop_scale * 0.45, loop_center_y),
                (loop_center_x + loop_scale * 0.45, loop_center_y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=linewidth,
                color=colors[source],
                alpha=0.72,
                connectionstyle="arc3,rad=1.5",
            )
        else:
            patch = FancyArrowPatch(
                (float(start["x"]), float(start["y"])),
                (float(end["x"]), float(end["y"])),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=linewidth,
                color=colors[source],
                alpha=0.72,
                connectionstyle="arc3,rad=0.08",
            )
        ax.add_patch(patch)

    for node in nodes:
        row = coords.loc[node]
        node_size = 180 + 1100 * (node_weights[node] / max_node_weight)
        ax.scatter(
            float(row["x"]),
            float(row["y"]),
            s=node_size,
            c=colors[node],
            edgecolors="#F7F7F7",
            linewidths=2.0,
            zorder=3,
        )
        label_x, label_y, ha, va = label_positions[node]
        ax.text(
            label_x,
            label_y,
            node,
            ha=ha,
            va=va,
            fontsize=11,
            color="#1F1F1F",
            path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")],
        )

    ax.set_xlim(float(coords["x"].min()) - pad, float(coords["x"].max()) + pad)
    ax.set_ylim(float(coords["y"].min()) - pad, float(coords["y"].max()) + pad)
    ax.set_aspect("equal")
    ax.set_title(title or "Communication embedding network")
    if background is None or background.empty:
        ax.set_axis_off()
    else:
        axis_labels = adata.uns.get("embedding_axes", ("UMAP_1", "UMAP_2"))
        if len(axis_labels) >= 2:
            ax.set_xlabel(str(axis_labels[0]))
            ax.set_ylabel(str(axis_labels[1]))
        ax.grid(False)
        n_cells = int(len(background))
        ax.text(0.0, 1.02, f"nCells:{n_cells}", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(handles, labels, title="CellType", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            for text in legend.get_texts():
                text.set_fontsize(9)
    return fig, ax


def _weighted_positions(labels: list[str], totals: pd.Series) -> dict[str, tuple[float, float]]:
    if not labels:
        return {}
    heights = totals.reindex(labels).fillna(0.0).to_numpy(dtype=float)
    if np.allclose(heights.sum(), 0.0):
        heights = np.ones(len(labels), dtype=float)
    heights = heights / heights.sum()
    gap = 0.02
    usable = 1.0 - gap * (len(labels) - 1)
    centers = []
    cursor = 1.0
    for height in heights:
        block = height * usable
        center = cursor - block / 2.0
        centers.append(center)
        cursor -= block + gap
    return {label: (0.0, centers[idx]) for idx, label in enumerate(labels)}


def _draw_sankey_flows(
    ax,
    flow_df: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    left_x: float,
    right_x: float,
    colors: Mapping[str, str],
    width_scale: float,
):
    for _, row in flow_df.iterrows():
        y0 = row[f"{left_col}_y"]
        y1 = row[f"{right_col}_y"]
        verts = [
            (left_x, y0),
            (left_x + (right_x - left_x) * 0.35, y0),
            (left_x + (right_x - left_x) * 0.65, y1),
            (right_x, y1),
        ]
        path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=colors[str(row[left_col])],
            linewidth=1.0 + width_scale * float(row["weight"]),
            alpha=0.45,
            capstyle="round",
        )
        ax.add_patch(patch)


def _draw_sankey_plot(
    long_df: pd.DataFrame,
    *,
    display_by: Literal["aggregation", "interaction"],
    value: Literal["sum", "mean", "max", "count"],
    top_n: int,
    palette,
    figsize: tuple[float, float],
    title: str | None,
):
    if display_by == "aggregation":
        flow_df = _aggregate_series(long_df.groupby(["sender", "receiver"], observed=True), value).rename("weight").reset_index()
        flow_df = flow_df.sort_values("weight", ascending=False).head(int(top_n))
        if flow_df.empty:
            raise ValueError("No communication flows remain for sankey plotting.")
        senders = list(dict.fromkeys(flow_df["sender"]))
        receivers = list(dict.fromkeys(flow_df["receiver"]))
        sender_totals = flow_df.groupby("sender", observed=True)["weight"].sum().sort_values(ascending=False)
        receiver_totals = flow_df.groupby("receiver", observed=True)["weight"].sum().sort_values(ascending=False)
        sender_pos = _weighted_positions(senders, sender_totals)
        receiver_pos = _weighted_positions(receivers, receiver_totals)
        flow_df["sender_y"] = flow_df["sender"].map(lambda item: sender_pos[str(item)][1])
        flow_df["receiver_y"] = flow_df["receiver"].map(lambda item: receiver_pos[str(item)][1])

        fig, ax = plt.subplots(figsize=figsize)
        node_colors = _choose_palette(sorted(set(senders) | set(receivers)), palette=palette)
        width_scale = 10.0 / max(float(flow_df["weight"].max()), 1.0)
        _draw_sankey_flows(
            ax,
            flow_df,
            left_col="sender",
            right_col="receiver",
            left_x=0.15,
            right_x=0.85,
            colors=node_colors,
            width_scale=width_scale,
        )
        for sender, (x_coord, y_coord) in sender_pos.items():
            ax.add_patch(Rectangle((0.08, y_coord - 0.03), 0.04, 0.06, facecolor=node_colors[sender], edgecolor="white"))
            ax.text(0.06, y_coord, sender, ha="right", va="center", fontsize=10)
        for receiver, (_, y_coord) in receiver_pos.items():
            ax.add_patch(Rectangle((0.88, y_coord - 0.03), 0.04, 0.06, facecolor=node_colors[receiver], edgecolor="white"))
            ax.text(0.94, y_coord, receiver, ha="left", va="center", fontsize=10)
        ax.text(0.10, 1.02, "Sender", ha="center", va="bottom", fontsize=11)
        ax.text(0.90, 1.02, "Receiver", ha="center", va="bottom", fontsize=11)
        ax.set_title(title or "Communication sankey")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        return fig, ax

    flow_df = _aggregate_series(
        long_df.groupby(["sender", "interaction", "receiver"], observed=True),
        value,
    ).rename("weight").reset_index()
    interaction_scores = flow_df.groupby("interaction", observed=True)["weight"].sum().sort_values(ascending=False)
    keep = interaction_scores.head(int(top_n)).index
    flow_df = flow_df.loc[flow_df["interaction"].isin(keep)].copy()
    if flow_df.empty:
        raise ValueError("No interaction-level communication flows remain for sankey plotting.")

    senders = list(dict.fromkeys(flow_df["sender"]))
    interactions = list(interaction_scores.loc[keep].index.astype(str))
    receivers = list(dict.fromkeys(flow_df["receiver"]))
    sender_totals = flow_df.groupby("sender", observed=True)["weight"].sum().sort_values(ascending=False)
    interaction_totals = flow_df.groupby("interaction", observed=True)["weight"].sum().sort_values(ascending=False)
    receiver_totals = flow_df.groupby("receiver", observed=True)["weight"].sum().sort_values(ascending=False)
    sender_pos = _weighted_positions(senders, sender_totals)
    interaction_pos = _weighted_positions(interactions, interaction_totals)
    receiver_pos = _weighted_positions(receivers, receiver_totals)

    flow_left = flow_df.groupby(["sender", "interaction"], observed=True)["weight"].sum().rename("weight").reset_index()
    flow_left["sender_y"] = flow_left["sender"].map(lambda item: sender_pos[str(item)][1])
    flow_left["interaction_y"] = flow_left["interaction"].map(lambda item: interaction_pos[str(item)][1])
    flow_right = flow_df.groupby(["interaction", "receiver"], observed=True)["weight"].sum().rename("weight").reset_index()
    flow_right["interaction_y"] = flow_right["interaction"].map(lambda item: interaction_pos[str(item)][1])
    flow_right["receiver_y"] = flow_right["receiver"].map(lambda item: receiver_pos[str(item)][1])

    fig, ax = plt.subplots(figsize=figsize)
    cell_colors = _choose_palette(sorted(set(senders) | set(receivers)), palette=palette)
    interaction_colors = _choose_palette(interactions, palette="Set2")
    width_scale = 9.0 / max(float(max(flow_left["weight"].max(), flow_right["weight"].max())), 1.0)
    _draw_sankey_flows(
        ax,
        flow_left,
        left_col="sender",
        right_col="interaction",
        left_x=0.12,
        right_x=0.5,
        colors=cell_colors,
        width_scale=width_scale,
    )
    _draw_sankey_flows(
        ax,
        flow_right,
        left_col="interaction",
        right_col="receiver",
        left_x=0.5,
        right_x=0.88,
        colors=interaction_colors,
        width_scale=width_scale,
    )
    for sender, (_, y_coord) in sender_pos.items():
        ax.add_patch(Rectangle((0.06, y_coord - 0.025), 0.04, 0.05, facecolor=cell_colors[sender], edgecolor="white"))
        ax.text(0.04, y_coord, sender, ha="right", va="center", fontsize=9)
    for interaction, (_, y_coord) in interaction_pos.items():
        ax.add_patch(
            Rectangle((0.48, y_coord - 0.025), 0.04, 0.05, facecolor=interaction_colors[interaction], edgecolor="white")
        )
        ax.text(
            0.535,
            y_coord,
            _wrap_plot_label(str(interaction), width=18),
            ha="left",
            va="center",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
        )
    for receiver, (_, y_coord) in receiver_pos.items():
        ax.add_patch(Rectangle((0.9, y_coord - 0.025), 0.04, 0.05, facecolor=cell_colors[receiver], edgecolor="white"))
        ax.text(0.96, y_coord, receiver, ha="left", va="center", fontsize=9)
    ax.text(0.08, 1.02, "Sender", ha="center", va="bottom", fontsize=11)
    ax.text(0.50, 1.02, "Ligand-Receptor", ha="center", va="bottom", fontsize=11)
    ax.text(0.92, 1.02, "Receiver", ha="center", va="bottom", fontsize=11)
    ax.set_title(title or "Interaction sankey")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax


@register_function(
    aliases=["ccc_network_plot", "communication_network_plot", "细胞通讯网络图"],
    category="pl",
    description="Python-native communication network plots for CellPhoneDB-style communication AnnData.",
    examples=[
        "ov.pl.ccc_network_plot(comm_adata, plot_type='circle')",
        "ov.pl.ccc_network_plot(comm_adata, plot_type='arrow', display_by='interaction', signaling='MK')",
        "ov.pl.ccc_network_plot(comm_adata, plot_type='bipartite', ligand='TGFB1')",
        "ov.pl.ccc_network_plot(comm_adata, plot_type='embedding_network', node_positions=coords)",
    ],
    related=["pl.ccc_heatmap", "pl.ccc_stat_plot", "pl.cpdb_network"],
)
def ccc_network_plot(
    adata: anndata.AnnData,
    *,
    plot_type: Literal[
        "circle",
        "circle_focused",
        "pathway",
        "individual_lr",
        "individual",
        "individual_outgoing",
        "individual_incoming",
        "arrow",
        "sigmoid",
        "bipartite",
        "embedding_network",
        "diff_network",
        "chord",
        "lr_chord",
        "gene_chord",
        "diffusion",
    ] = "circle",
    comparison_adata: anndata.AnnData | None = None,
    display_by: Literal["aggregation", "interaction"] = "aggregation",
    sender_use=None,
    receiver_use=None,
    signaling=None,
    interaction_use=None,
    pair_lr_use=None,
    pvalue_threshold: float = 0.05,
    value: Literal["sum", "mean", "max", "count"] = "sum",
    top_n: int = 20,
    ligand: str | None = None,
    receptor=None,
    layout: Literal["circle", "hierarchy"] = "circle",
    normalize_to_sender: bool = True,
    rotate_names: bool = True,
    min_interaction_threshold: float = 0.0,
    node_positions=None,
    embedding_points=None,
    palette=None,
    figsize: tuple[float, float] = (7, 7),
    title: str | None = None,
    show: bool = False,
    save: str | bool = False,
):
    """Plot aggregated communication as circle, flow, bipartite, or differential networks."""
    if plot_type == "circle_focused":
        signaling_values = _normalize_use_arg(signaling)
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
        )
        if signaling_values is None:
            _raise_for_unsupported_arguments(
                plot_type,
                sender_use=sender_use,
                receiver_use=receiver_use,
            )
        viz = _build_cellchatviz(adata, palette=palette)
        if signaling_values is not None:
            fig, ax = viz.netVisual_aggregate(
                signaling=signaling_values,
                layout="circle",
                vertex_receiver=_normalize_use_arg(receiver_use),
                vertex_sender=_normalize_use_arg(sender_use),
                pvalue_threshold=pvalue_threshold,
                figsize=figsize,
                vertex_size_max=10,
                focused_view=True,
                use_sender_colors=True,
            )
        else:
            _, weight_matrix = viz.compute_aggregated_network(pvalue_threshold)
            fig, ax = viz.netVisual_circle_focused(
                weight_matrix,
                title=title or "Focused cell-cell communication network",
                figsize=figsize,
                min_interaction_threshold=min_interaction_threshold,
                use_sender_colors=True,
            )
        if title:
            ax.set_title(title)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "individual_outgoing":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            signaling=signaling,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            ligand=ligand,
            receptor=receptor,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        fig = viz.netVisual_individual_circle(
            pvalue_threshold=pvalue_threshold,
            vertex_size_max=10,
            edge_width_max=12,
            show_labels=True,
            figsize=figsize,
            ncols=4,
            use_sender_colors=True,
        )
        ax = _largest_content_axis(fig)
        if title:
            fig.suptitle(title)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "individual_incoming":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            signaling=signaling,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            ligand=ligand,
            receptor=receptor,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        fig = viz.netVisual_individual_circle_incoming(
            pvalue_threshold=pvalue_threshold,
            vertex_size_max=10,
            edge_width_max=12,
            show_labels=True,
            figsize=figsize,
            ncols=4,
            use_sender_colors=True,
        )
        ax = _largest_content_axis(fig)
        if title:
            fig.suptitle(title)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "individual":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            ligand=ligand,
            receptor=receptor,
        )
        signaling_values = _require_use_argument(plot_type, "signaling", signaling)
        viz = _build_cellchatviz(adata, palette=palette)
        lr_pairs = _resolve_lr_pairs(
            adata,
            pair_lr_use=pair_lr_use,
            interaction_use=interaction_use,
        )
        pair_lr = None if lr_pairs is None else (lr_pairs[0] if len(lr_pairs) == 1 else lr_pairs[0])
        fig, ax = viz.netVisual_individual(
            signaling=signaling_values[0] if len(signaling_values) == 1 else signaling_values,
            pairLR_use=pair_lr,
            sources_use=_normalize_use_arg(sender_use),
            targets_use=_normalize_use_arg(receiver_use),
            layout=layout,
            vertex_receiver=_normalize_use_arg(receiver_use) if layout == "hierarchy" else None,
            pvalue_threshold=pvalue_threshold,
            figsize=figsize,
            title=title or "Individual ligand-receptor communication",
            save=None,
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "chord":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            ligand=ligand,
            receptor=receptor,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        fig, ax = viz.netVisual_chord_cell(
            signaling=_normalize_use_arg(signaling),
            sources=_normalize_use_arg(sender_use),
            targets=_normalize_use_arg(receiver_use),
            pvalue_threshold=pvalue_threshold,
            count_min=1,
            rotate_names=rotate_names,
            figsize=figsize,
            title_name=title or "Communication chord diagram",
            normalize_to_sender=normalize_to_sender,
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "gene_chord":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            ligand=ligand,
            receptor=receptor,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        fig, ax = viz.netVisual_chord_gene(
            sources_use=sender_use,
            targets_use=receiver_use,
            signaling=_normalize_use_arg(signaling),
            pvalue_threshold=pvalue_threshold,
            mean_threshold=0.1,
            rotate_names=rotate_names,
            figsize=figsize,
            title_name=title or "Gene-level chord diagram",
            save=None,
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "lr_chord":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            signaling=signaling,
            ligand=ligand,
            receptor=receptor,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        lr_pairs = _resolve_lr_pairs(
            adata,
            pair_lr_use=pair_lr_use,
            interaction_use=interaction_use,
        )
        if lr_pairs is None:
            raise ValueError("`pair_lr_use` or `interaction_use` is required when `plot_type='lr_chord'`.")
        fig, ax = viz.netVisual_chord_LR(
            ligand_receptor_pairs=lr_pairs[0] if len(lr_pairs) == 1 else lr_pairs,
            sources=_normalize_use_arg(sender_use),
            targets=_normalize_use_arg(receiver_use),
            pvalue_threshold=pvalue_threshold,
            count_min=1,
            rotate_names=rotate_names,
            figsize=figsize,
            title_name=title or "Ligand-receptor chord diagram",
            normalize_to_sender=normalize_to_sender,
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "diff_network":
        if comparison_adata is None:
            raise ValueError("`comparison_adata` is required when `plot_type='diff_network'`.")
        reference_df = _aggregated_pair_frame(
            adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            signaling=signaling,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            pvalue_threshold=pvalue_threshold,
            value=value,
        )
        comparison_df = _aggregated_pair_frame(
            comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            signaling=signaling,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            pvalue_threshold=pvalue_threshold,
            value=value,
        )
        edge_df = reference_df.merge(
            comparison_df,
            on=["sender", "receiver"],
            how="outer",
            suffixes=("_reference", "_comparison"),
        ).fillna(0.0)
        edge_df["delta"] = edge_df["weight_comparison"] - edge_df["weight_reference"]
        edge_df = edge_df.loc[edge_df["delta"] != 0].copy()
        edge_df["abs_delta"] = edge_df["delta"].abs()
        edge_df = edge_df.sort_values("abs_delta", ascending=False)
        if top_n is not None and top_n > 0:
            edge_df = edge_df.head(int(top_n))
        if edge_df.empty:
            raise ValueError("No differential communication edges remain after filtering.")
        fig, ax = _draw_diff_circle_network(edge_df, palette=palette, figsize=figsize, title=title)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "diffusion":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            signaling=signaling,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            ligand=ligand,
            receptor=receptor,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        fig, ax = viz.netVisual_diffusion(
            similarity_type="functional",
            figsize=figsize,
            title=title or "Communication diffusion network",
            save=None,
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "bipartite":
        long_df = _communication_long_table(
            adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            signaling=signaling,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            pvalue_threshold=pvalue_threshold,
        )
        fig, ax = _draw_bipartite_network(
            long_df,
            ligand=ligand,
            receptor=receptor,
            top_n=top_n,
            palette=palette,
            figsize=figsize,
            title=title,
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    pathway_signaling = None
    if plot_type == "pathway":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            ligand=ligand,
            receptor=receptor,
        )
        pathway_signaling = _require_use_argument(plot_type, "signaling", signaling)
        if layout != "hierarchy":
            _raise_for_unsupported_arguments(
                plot_type,
                sender_use=sender_use,
                receiver_use=receiver_use,
            )

    edge_df = _aggregated_pair_frame(
        adata,
        sender_use=sender_use,
        receiver_use=receiver_use,
        signaling=signaling,
        interaction_use=interaction_use,
        pair_lr_use=pair_lr_use,
        pvalue_threshold=pvalue_threshold,
        value=value,
    )
    if top_n is not None and top_n > 0:
        edge_df = edge_df.head(int(top_n))
    if edge_df.empty:
        raise ValueError("No communication edges remain after filtering.")

    if plot_type == "embedding_network":
        fig, ax = _draw_embedding_network(
            adata,
            edge_df,
            node_positions=node_positions,
            embedding_points=embedding_points,
            palette=palette,
            figsize=figsize,
            title=title,
        )
    elif plot_type == "arrow":
        flow_nodes, flow_edges, column_titles = _build_flow_plot_frames(
            _communication_long_table(
                adata,
                sender_use=sender_use,
                receiver_use=receiver_use,
                signaling=signaling,
                interaction_use=interaction_use,
                pair_lr_use=pair_lr_use,
                pvalue_threshold=pvalue_threshold,
            ),
            display_by=display_by,
            value=value,
            top_n=top_n,
        )
        fig, ax = _draw_arrow_network(
            flow_edges,
            node_df=flow_nodes,
            column_titles=column_titles,
            palette=palette,
            figsize=figsize,
            title=title,
        )
    elif plot_type == "sigmoid":
        flow_nodes, flow_edges, column_titles = _build_flow_plot_frames(
            _communication_long_table(
                adata,
                sender_use=sender_use,
                receiver_use=receiver_use,
                signaling=signaling,
                interaction_use=interaction_use,
                pair_lr_use=pair_lr_use,
                pvalue_threshold=pvalue_threshold,
            ),
            display_by=display_by,
            value=value,
            top_n=top_n,
        )
        fig, ax = _draw_sigmoid_network(
            flow_edges,
            node_df=flow_nodes,
            column_titles=column_titles,
            palette=palette,
            figsize=figsize,
            title=title,
        )
    elif plot_type == "pathway":
        viz = _build_cellchatviz(adata, palette=palette)
        fig, ax = viz.netVisual_aggregate(
            signaling=pathway_signaling,
            layout=layout,
            vertex_receiver=_normalize_use_arg(receiver_use) if layout == "hierarchy" else None,
            vertex_sender=_normalize_use_arg(sender_use) if layout == "hierarchy" else None,
            pvalue_threshold=pvalue_threshold,
            figsize=figsize,
            vertex_size_max=10,
        )
        if title:
            ax.set_title(title)
    elif plot_type == "individual_lr":
        fig, ax = _draw_circle_network(
            edge_df,
            palette=palette,
            figsize=figsize,
            title=title or "Ligand-receptor communication network",
        )
    else:
        fig, ax = _draw_circle_network(edge_df, palette=palette, figsize=figsize, title=title)
    return _maybe_save_show(fig, show=show, save=save), ax


def _resolve_group_field(long_df: pd.DataFrame, group_by: Literal["interaction", "classification", "pair", "sender", "receiver"]):
    if group_by == "pair":
        return long_df["pair"]
    if group_by == "interaction" and "interaction_display" in long_df.columns:
        return long_df["interaction_display"]
    return long_df[group_by]


def _interaction_contribution(
    long_df: pd.DataFrame,
    *,
    value: Literal["sum", "mean", "max", "count"],
    top_n: int,
) -> pd.Series:
    summary = _aggregate_series(long_df.groupby("interaction", observed=True), value)
    return summary.sort_values(ascending=False).head(int(top_n))


def _group_metric_summary(
    long_df: pd.DataFrame,
    *,
    group_key: str,
    value: Literal["sum", "mean", "max", "count"],
) -> pd.Series:
    return _aggregate_series(long_df.groupby(group_key, observed=True), value).sort_values(ascending=False)


def _metric_axis_label(
    *,
    measure: Literal["weight", "count"],
    value: Literal["sum", "mean", "max", "count"],
) -> str:
    if measure == "count" or value == "count":
        return "Significant interactions"
    if value == "mean":
        return "Mean communication score"
    if value == "max":
        return "Max communication score"
    return "Communication score"


def _overall_metric(
    long_df: pd.DataFrame,
    *,
    measure: Literal["weight", "count"],
    value: Literal["sum", "mean", "max", "count"],
) -> float:
    if measure == "count":
        return float(long_df["significant"].sum())
    if value == "mean":
        return float(long_df["score"].mean())
    if value == "max":
        return float(long_df["score"].max())
    return float(long_df["score"].sum())


def _group_metric_by_measure(
    long_df: pd.DataFrame,
    *,
    group_key: str,
    measure: Literal["weight", "count"],
    value: Literal["sum", "mean", "max", "count"],
) -> pd.Series:
    if measure == "count":
        return long_df.groupby(group_key, observed=True)["significant"].sum().astype(float).sort_values(ascending=False)
    return _group_metric_summary(long_df, group_key=group_key, value=value)


def _celltype_role_summary(
    long_df: pd.DataFrame,
    *,
    pattern: Literal["outgoing", "incoming", "all"],
    measure: Literal["weight", "count"],
    value: Literal["sum", "mean", "max", "count"],
) -> pd.Series:
    outgoing = _group_metric_by_measure(long_df, group_key="sender", measure=measure, value=value)
    incoming = _group_metric_by_measure(long_df, group_key="receiver", measure=measure, value=value)
    if pattern == "outgoing":
        return outgoing
    if pattern == "incoming":
        return incoming
    return outgoing.add(incoming, fill_value=0.0).sort_values(ascending=False)


def _pathway_metric_summary(
    long_df: pd.DataFrame,
    *,
    measure: Literal["weight", "count"],
    value: Literal["sum", "mean", "max", "count"],
) -> pd.Series:
    return _group_metric_by_measure(long_df, group_key="classification", measure=measure, value=value)


def _gene_metric_frame(
    long_df: pd.DataFrame,
    *,
    measure: Literal["weight", "count"],
    value: Literal["sum", "mean", "max", "count"],
    top_n: int,
) -> pd.DataFrame:
    ligand_col = "ligand_display" if "ligand_display" in long_df.columns else "ligand"
    receptor_col = "receptor_display" if "receptor_display" in long_df.columns else "receptor"
    ligand_df = long_df.loc[long_df[ligand_col].astype(str).ne(""), [ligand_col, "score", "significant"]].copy()
    receptor_df = long_df.loc[long_df[receptor_col].astype(str).ne(""), [receptor_col, "score", "significant"]].copy()
    ligand_df = ligand_df.rename(columns={ligand_col: "gene"})
    receptor_df = receptor_df.rename(columns={receptor_col: "gene"})

    ligand_summary = _group_metric_by_measure(ligand_df, group_key="gene", measure=measure, value=value).head(int(top_n))
    receptor_summary = _group_metric_by_measure(
        receptor_df, group_key="gene", measure=measure, value=value
    ).head(int(top_n))

    frames = []
    if not ligand_summary.empty:
        frames.append(
            pd.DataFrame(
                {
                    "gene": ligand_summary.index.astype(str),
                    "metric": ligand_summary.values,
                    "role": "Ligand",
                }
            )
        )
    if not receptor_summary.empty:
        frames.append(
            pd.DataFrame(
                {
                    "gene": receptor_summary.index.astype(str),
                    "metric": receptor_summary.values,
                    "role": "Receptor",
                }
            )
        )
    if not frames:
        raise ValueError("No ligand or receptor annotations remain after filtering.")

    gene_frame = pd.concat(frames, ignore_index=True)
    gene_frame["gene_label"] = gene_frame["gene"].map(lambda item: _wrap_plot_label(item, width=16))
    return gene_frame


def _interaction_distribution_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    interaction_col = "interaction_display" if "interaction_display" in long_df.columns else "interaction"
    frame = long_df.loc[
        long_df[interaction_col].astype(str).str.strip().ne(""),
        ["sender", "receiver", "pair", interaction_col, "score"],
    ].copy()
    frame = frame.rename(columns={interaction_col: "interaction_label"})
    return (
        frame.groupby(["sender", "receiver", "pair", "interaction_label"], observed=True)["score"]
        .sum()
        .rename("score")
        .reset_index()
    )


def _draw_distribution_axis(
    ax,
    data: pd.DataFrame,
    *,
    plot_type: Literal["box", "violin"],
    group_var: str,
    palette,
    title: str | None = None,
) -> None:
    ranked_groups = (
        data.groupby(group_var, observed=True)["score"].sum().sort_values(ascending=False).index.astype(str).tolist()
    )
    plot_df = data.copy()
    plot_df[group_var] = pd.Categorical(plot_df[group_var].astype(str), categories=ranked_groups, ordered=True)
    group_colors = _choose_palette(ranked_groups, palette=palette)

    if plot_type == "violin":
        sns.violinplot(
            data=plot_df,
            x=group_var,
            y="score",
            hue=group_var,
            palette=group_colors,
            dodge=False,
            inner="box",
            cut=0,
            ax=ax,
        )
    else:
        sns.boxplot(
            data=plot_df,
            x=group_var,
            y="score",
            hue=group_var,
            palette=group_colors,
            dodge=False,
            ax=ax,
        )

    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel(group_var.capitalize())
    ax.set_ylabel("Communication score")
    _apply_category_tick_labels(ax, ranked_groups, wrap_width=14, rotation=45.0)
    if title is not None:
        ax.set_title(str(title))


def _draw_distribution_facets(
    plot_data: pd.DataFrame,
    *,
    plot_type: Literal["box", "violin"],
    facet_by: Literal["sender", "receiver", "pair"],
    palette,
    figsize: tuple[float, float],
    title: str | None,
):
    group_var = "receiver" if facet_by == "sender" else ("sender" if facet_by == "receiver" else "pair")
    if facet_by == "pair":
        fig, ax = plt.subplots(figsize=figsize)
        _draw_distribution_axis(ax, plot_data, plot_type=plot_type, group_var=group_var, palette=palette, title=title)
        return fig, ax

    facet_levels = (
        plot_data.groupby(facet_by, observed=True)["score"].sum().sort_values(ascending=False).index.astype(str).tolist()
    )
    n_panels = len(facet_levels)
    ncols = min(4, max(1, int(np.ceil(np.sqrt(n_panels)))))
    nrows = int(np.ceil(n_panels / ncols))
    fig_width = max(figsize[0], 3.8 * ncols)
    fig_height = max(figsize[1], 3.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False, sharey=True)
    axes_flat = axes.ravel()

    for axis, facet_value in zip(axes_flat, facet_levels):
        subset = plot_data.loc[plot_data[facet_by].astype(str) == str(facet_value)].copy()
        _draw_distribution_axis(
            axis,
            subset,
            plot_type=plot_type,
            group_var=group_var,
            palette=palette,
            title=str(facet_value),
        )
    for axis in axes_flat[n_panels:]:
        axis.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, _largest_content_axis(fig)


def _pathway_role_delta_frame(
    reference_long_df: pd.DataFrame,
    comparison_long_df: pd.DataFrame,
    *,
    idents_use,
    measure: Literal["weight", "count"],
    value: Literal["sum", "mean", "max", "count"],
) -> pd.DataFrame:
    ident_labels = _normalize_use_arg(idents_use)
    if ident_labels is None:
        raise ValueError("`idents_use` is required when `plot_type='role_change'`.")

    frames = []
    for ident in ident_labels:
        reference_out = _group_metric_by_measure(
            reference_long_df.loc[reference_long_df["sender"].astype(str) == ident],
            group_key="classification",
            measure=measure,
            value=value,
        )
        reference_in = _group_metric_by_measure(
            reference_long_df.loc[reference_long_df["receiver"].astype(str) == ident],
            group_key="classification",
            measure=measure,
            value=value,
        )
        comparison_out = _group_metric_by_measure(
            comparison_long_df.loc[comparison_long_df["sender"].astype(str) == ident],
            group_key="classification",
            measure=measure,
            value=value,
        )
        comparison_in = _group_metric_by_measure(
            comparison_long_df.loc[comparison_long_df["receiver"].astype(str) == ident],
            group_key="classification",
            measure=measure,
            value=value,
        )
        merged = pd.DataFrame(
            {
                "outgoing_reference": reference_out,
                "incoming_reference": reference_in,
                "outgoing_comparison": comparison_out,
                "incoming_comparison": comparison_in,
            }
        ).fillna(0.0)
        if merged.empty:
            continue
        merged["celltype"] = ident
        merged["signaling"] = merged.index.astype(str)
        merged["delta_outgoing"] = merged["outgoing_comparison"] - merged["outgoing_reference"]
        merged["delta_incoming"] = merged["incoming_comparison"] - merged["incoming_reference"]
        merged["magnitude"] = np.hypot(merged["delta_outgoing"], merged["delta_incoming"])
        frames.append(merged.reset_index(drop=True))

    if not frames:
        return pd.DataFrame(columns=["celltype", "signaling", "delta_outgoing", "delta_incoming", "magnitude"])
    role_df = pd.concat(frames, ignore_index=True)
    return role_df.loc[role_df["signaling"].astype(str).str.strip().ne("")].reset_index(drop=True)


def _infer_expression_groupby(source_adata: anndata.AnnData, comm_adata: anndata.AnnData) -> str:
    node_labels = set(comm_adata.obs["sender"].astype(str)) | set(comm_adata.obs["receiver"].astype(str))
    preferred = ["cell_labels", "cell_type", "CellType", "celltypes", "ident", "cluster", "clusters"]
    columns = preferred + [column for column in source_adata.obs.columns if column not in preferred]
    for column in columns:
        if column not in source_adata.obs.columns:
            continue
        values = set(source_adata.obs[column].astype(str).dropna().tolist())
        if node_labels.issubset(values):
            return column
    raise ValueError(
        "Could not infer a grouping column in `source_adata.obs` that contains all sender/receiver cell types. "
        "Provide `source_groupby` explicitly."
    )


def _draw_gene_expression_plot(
    source_adata: anndata.AnnData,
    comm_adata: anndata.AnnData,
    gene_frame: pd.DataFrame,
    *,
    source_groupby: str | None,
    cmap: str,
    figsize: tuple[float, float],
    title: str | None,
):
    if source_groupby is None:
        source_groupby = _infer_expression_groupby(source_adata, comm_adata)
    if source_groupby not in source_adata.obs.columns:
        raise ValueError(f"`source_groupby='{source_groupby}'` was not found in `source_adata.obs`.")

    genes = gene_frame["gene"].astype(str).drop_duplicates().tolist()
    if not genes:
        raise ValueError("No genes remain for expression plotting.")
    available = [gene for gene in genes if gene in source_adata.var_names]
    if not available:
        raise ValueError(
            "None of the selected ligand/receptor genes were found in `source_adata.var_names`. "
            "Pass a compatible expression AnnData to `source_adata`."
        )

    expr = pd.DataFrame(
        _to_dense(source_adata[:, available].X),
        index=source_adata.obs_names,
        columns=available,
    )
    group_series = source_adata.obs[source_groupby].astype(str)
    order = [
        label for label in dict.fromkeys(comm_adata.obs["sender"].astype(str).tolist() + comm_adata.obs["receiver"].astype(str).tolist())
        if label in set(group_series.tolist())
    ]
    if not order:
        order = group_series.drop_duplicates().tolist()

    mean_expr = expr.join(group_series.rename("group")).groupby("group", observed=True)[available].mean()
    mean_expr = mean_expr.reindex(order).dropna(how="all")
    if mean_expr.empty:
        raise ValueError("No grouped expression values remain for gene plotting.")

    gene_roles = gene_frame.drop_duplicates("gene").set_index("gene")["role"].to_dict()
    ordered_genes = [gene for gene in genes if gene in mean_expr.columns]
    mean_expr = mean_expr.loc[:, ordered_genes]
    plot_matrix = mean_expr.T
    plot_matrix.index = [
        f"{gene} ({'L' if gene_roles.get(gene) == 'Ligand' else 'R'})"
        for gene in plot_matrix.index.astype(str)
    ]

    fig_height = max(figsize[1], 0.38 * len(plot_matrix.index) + 1.8)
    fig_width = max(figsize[0], 0.52 * len(plot_matrix.columns) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        plot_matrix,
        cmap=cmap,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Mean expression"},
        ax=ax,
    )
    ax.set_xlabel(source_groupby)
    ax.set_ylabel("")
    ax.set_title(title or "Pathway gene expression")
    ax.set_yticklabels([_wrap_plot_label(label, width=20) for label in plot_matrix.index.astype(str)], rotation=0)
    ax.set_xticklabels([_wrap_plot_label(label, width=14) for label in plot_matrix.columns.astype(str)], rotation=45, ha="right")
    return fig, ax


@register_function(
    aliases=["ccc_stat_plot", "communication_stat_plot", "细胞通讯统计图"],
    category="pl",
    description="Python-native summary and distribution plots for CellPhoneDB-style communication AnnData.",
    examples=[
        "ov.pl.ccc_stat_plot(comm_adata, plot_type='bar', group_by='interaction')",
        "ov.pl.ccc_stat_plot(comm_adata, plot_type='sankey', display_by='interaction')",
        "ov.pl.ccc_stat_plot(comm_adata, plot_type='scatter')",
    ],
    related=["pl.ccc_heatmap", "pl.ccc_network_plot", "pl.CellChatViz"],
)
def ccc_stat_plot(
    adata: anndata.AnnData,
    *,
    plot_type: Literal[
        "bar",
        "sankey",
        "box",
        "violin",
        "scatter",
        "role_scatter",
        "role_network",
        "role_network_marsilea",
        "pathway_summary",
        "lr_contribution",
        "comparison",
        "gene",
        "ranknet",
        "role_change",
    ] = "bar",
    comparison_adata: anndata.AnnData | None = None,
    source_adata: anndata.AnnData | None = None,
    source_groupby: str | None = None,
    display_by: Literal["aggregation", "interaction"] = "aggregation",
    sender_use=None,
    receiver_use=None,
    signaling=None,
    interaction_use=None,
    pair_lr_use=None,
    pvalue_threshold: float = 0.05,
    group_by: Literal["interaction", "classification", "pair", "sender", "receiver"] = "interaction",
    compare_by: Literal["overall", "celltype"] = "overall",
    measure: Literal["weight", "count"] = "weight",
    pattern: Literal["outgoing", "incoming", "all"] = "all",
    idents_use=None,
    facet_by: Literal["sender", "receiver"] | None = None,
    value: Literal["sum", "mean", "max", "count"] = "sum",
    top_n: int = 20,
    pathway_method: str = "mean",
    min_lr_pairs: int = 1,
    min_expression: float = 0.1,
    strength_threshold: float = 0.5,
    min_significant_pairs: int = 1,
    palette=None,
    measures: Sequence[str] | None = None,
    cmap: str = "RdYlBu_r",
    figsize: tuple[float, float] = (8, 5),
    title: str | None = None,
    show: bool = False,
    save: str | bool = False,
):
    """Plot communication summaries and score distributions."""
    if plot_type == "role_network":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            group_by=group_by if group_by != "interaction" else None,
            compare_by=compare_by if compare_by != "overall" else None,
            measure=measure if measure != "weight" else None,
            pattern=pattern if pattern != "all" else None,
            idents_use=idents_use,
            facet_by=facet_by,
            value=value if value != "sum" else None,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        fig = viz.netAnalysis_signalingRole_network(
            signaling=_normalize_use_arg(signaling),
            measures=list(measures) if measures is not None else None,
            color_heatmap=cmap,
            width=figsize[0],
            height=figsize[1],
            title=title or "Signaling role analysis",
            save=None,
            show_values=True,
        )
        ax = _largest_content_axis(fig)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "role_network_marsilea":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            group_by=group_by if group_by != "interaction" else None,
            compare_by=compare_by if compare_by != "overall" else None,
            measure=measure if measure != "weight" else None,
            pattern=pattern if pattern != "all" else None,
            idents_use=idents_use,
            facet_by=facet_by,
            value=value if value != "sum" else None,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        plotter = viz.netAnalysis_signalingRole_network_marsilea(
            signaling=_normalize_use_arg(signaling),
            measures=list(measures) if measures is not None else None,
            color_heatmap=cmap,
            width=figsize[0],
            height=figsize[1],
            title=title or "Signaling role analysis",
            add_dendrogram=True,
            add_cell_colors=True,
            add_importance_bars=True,
            show_values=True,
            save=None,
        )
        fig, ax = _render_plotter_figure(plotter, title=None)
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "role_scatter":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            group_by=group_by if group_by != "interaction" else None,
            compare_by=compare_by if compare_by != "overall" else None,
            measure=measure if measure != "weight" else None,
            pattern=pattern if pattern != "all" else None,
            idents_use=idents_use,
            facet_by=facet_by,
            value=value if value != "sum" else None,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        fig, ax = viz.netAnalysis_signalingRole_scatter(
            signaling=_normalize_use_arg(signaling),
            x_measure="outdegree",
            y_measure="indegree",
            figsize=figsize,
            title=title or "Signaling role scatter",
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "pathway_summary":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            group_by=group_by if group_by != "interaction" else None,
            compare_by=compare_by if compare_by != "overall" else None,
            measure=measure if measure != "weight" else None,
            pattern=pattern if pattern != "all" else None,
            idents_use=idents_use,
            facet_by=facet_by,
            value=value if value != "sum" else None,
        )
        summary = _pathway_summary_table(
            adata,
            palette=palette,
            signaling=signaling,
            pathway_method=pathway_method,
            min_lr_pairs=min_lr_pairs,
            min_expression=min_expression,
            strength_threshold=strength_threshold,
            pvalue_threshold=pvalue_threshold,
            min_significant_pairs=min_significant_pairs,
        )
        summary = summary.sort_values(["is_significant", "total_strength"], ascending=[False, False]).head(int(top_n))
        min_height = max(figsize[1], 0.42 * len(summary.index) + 1.6)
        if min_height > figsize[1]:
            figsize = (figsize[0], min_height)
        fig, ax = plt.subplots(figsize=figsize)
        pathway_names = summary["pathway"].astype(str).tolist()
        colors = _choose_palette(pathway_names, palette=palette)
        bar_colors = [colors[name] if is_sig else "#D9D9D9" for name, is_sig in zip(pathway_names, summary["is_significant"])]
        ax.barh(pathway_names, summary["total_strength"], color=bar_colors)
        ax.invert_yaxis()
        for y_pos, (_, row) in enumerate(summary.iterrows()):
            ax.text(
                float(row["total_strength"]),
                y_pos,
                f"  {int(row['n_significant_pairs'])}/{int(row['n_active_cell_pairs'])} sig",
                va="center",
                ha="left",
                fontsize=8,
                color="#4A4A4A",
            )
        ax.set_xlabel("Total pathway communication strength")
        ax.set_ylabel("")
        ax.set_title(title or "Significant pathway communication summary")
        return _maybe_save_show(fig, show=show, save=save), ax

    long_df = _communication_long_table(
        adata,
        sender_use=sender_use,
        receiver_use=receiver_use,
        signaling=signaling,
        interaction_use=interaction_use,
        pair_lr_use=pair_lr_use,
        pvalue_threshold=pvalue_threshold,
    )
    if long_df.empty:
        raise ValueError("No communication records remain after filtering.")

    if plot_type == "sankey":
        fig, ax = _draw_sankey_plot(
            long_df,
            display_by=display_by,
            value=value,
            top_n=top_n,
            palette=palette,
            figsize=figsize,
            title=title,
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "scatter":
        _raise_for_unsupported_arguments(
            plot_type,
            comparison_adata=comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            group_by=group_by if group_by != "interaction" else None,
            compare_by=compare_by if compare_by != "overall" else None,
            measure=measure if measure != "weight" else None,
            pattern=pattern if pattern != "all" else None,
            idents_use=idents_use,
            facet_by=facet_by,
            value=value if value != "sum" else None,
        )
        viz = _build_cellchatviz(adata, palette=palette)
        fig, ax = viz.netAnalysis_signalingRole_scatter(
            signaling=_normalize_use_arg(signaling),
            x_measure="outdegree",
            y_measure="indegree",
            figsize=figsize,
            title=title or "Outgoing vs incoming communication",
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "lr_contribution":
        if signaling is not None:
            _raise_for_unsupported_arguments(
                plot_type,
                comparison_adata=comparison_adata,
                sender_use=sender_use,
                receiver_use=receiver_use,
                interaction_use=interaction_use,
                pair_lr_use=pair_lr_use,
                group_by=group_by if group_by != "interaction" else None,
                compare_by=compare_by if compare_by != "overall" else None,
                measure=measure if measure != "weight" else None,
                pattern=pattern if pattern != "all" else None,
                idents_use=idents_use,
                facet_by=facet_by,
            )
            viz = _build_cellchatviz(adata, palette=palette)
            contribution_result = viz.netAnalysis_contribution(
                signaling=_normalize_use_arg(signaling),
                pvalue_threshold=pvalue_threshold,
                top_pairs=top_n,
                figsize=figsize,
                save=None,
            )
            if contribution_result is None:
                raise ValueError("No significant contributions remain after filtering.")
            if len(contribution_result) == 3 and hasattr(contribution_result[1], "savefig"):
                _, fig, axes = contribution_result
                if isinstance(axes, tuple):
                    ax = axes[0]
                else:
                    ax = _largest_content_axis(fig)
            else:
                fig, _ = contribution_result
                ax = _largest_content_axis(fig)
            if title:
                ax.set_title(title)
            for contribution_ax in fig.axes:
                if "Activity vs Significance" in contribution_ax.get_title():
                    _style_scatter_axis(contribution_ax, max_marker_size=260.0, arrow=False)
            return _maybe_save_show(fig, show=show, save=save), ax

        summary = _interaction_contribution(long_df, value=value, top_n=top_n)
        colors = _choose_palette(summary.index.astype(str).tolist(), palette=palette)
        ax.barh(summary.index.astype(str), summary.values, color=[colors[idx] for idx in summary.index.astype(str)])
        ax.invert_yaxis()
        pathway_label = ""
        if signaling is not None:
            pathway_label = f" in {', '.join(_normalize_use_arg(signaling))}"
        x_label = "Communication score" if value != "count" else "Significant interactions"
        ax.set_xlabel(x_label)
        ax.set_ylabel("")
        ax.set_title(title or f"Ligand-receptor contribution{pathway_label}")
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type == "gene":
        gene_frame = _gene_metric_frame(long_df, measure=measure, value=value, top_n=top_n)
        if source_adata is not None:
            fig, ax = _draw_gene_expression_plot(
                source_adata,
                adata,
                gene_frame,
                source_groupby=source_groupby,
                cmap=cmap,
                figsize=figsize,
                title=title,
            )
            return _maybe_save_show(fig, show=show, save=save), ax
        min_height = max(figsize[1], 0.42 * len(gene_frame) + 1.6)
        if min_height > figsize[1]:
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(figsize[0], min_height))
        sns.barplot(
            data=gene_frame,
            x="metric",
            y="gene_label",
            hue="role",
            orient="h",
            dodge=False,
            palette={"Ligand": "#2F6DB3", "Receptor": "#D96B27"},
            ax=ax,
        )
        pathway_label = ""
        if signaling is not None:
            pathway_label = f" in {', '.join(_normalize_use_arg(signaling))}"
        ax.set_xlabel(_metric_axis_label(measure=measure, value=value))
        ax.set_ylabel("")
        ax.set_title(title or f"Pathway gene contribution{pathway_label}")
        ax.tick_params(axis="y", labelsize=9)
        ax.legend(frameon=False, title="")
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type in {"comparison", "ranknet", "role_change"}:
        if comparison_adata is None:
            raise ValueError(f"`comparison_adata` is required when `plot_type='{plot_type}'`.")
        comparison_long_df = _communication_long_table(
            comparison_adata,
            sender_use=sender_use,
            receiver_use=receiver_use,
            signaling=signaling,
            interaction_use=interaction_use,
            pair_lr_use=pair_lr_use,
            pvalue_threshold=pvalue_threshold,
        )
        if comparison_long_df.empty:
            raise ValueError("No communication records remain in `comparison_adata` after filtering.")

        reference_color = "#6BA3D6"
        comparison_color = "#D96B27"

        if plot_type == "comparison":
            if compare_by == "overall":
                summary = pd.Series(
                    {
                        "Reference": _overall_metric(long_df, measure=measure, value=value),
                        "Comparison": _overall_metric(comparison_long_df, measure=measure, value=value),
                    }
                )
                colors = [reference_color, comparison_color]
                ax.bar(summary.index.tolist(), summary.values, color=colors, width=0.6)
                ax.set_ylabel(_metric_axis_label(measure=measure, value=value))
                ax.set_xlabel("")
                ax.set_title(title or "Communication comparison")
                return _maybe_save_show(fig, show=show, save=save), ax

            reference_series = _celltype_role_summary(long_df, pattern=pattern, measure=measure, value=value)
            comparison_series = _celltype_role_summary(
                comparison_long_df, pattern=pattern, measure=measure, value=value
            )
            merged = pd.DataFrame(
                {
                    "Reference": reference_series,
                    "Comparison": comparison_series,
                }
            ).fillna(0.0)
            merged["rank"] = merged.max(axis=1)
            merged = merged.sort_values("rank", ascending=False).head(int(top_n))
            merged = merged.drop(columns="rank")
            min_height = max(figsize[1], 0.42 * len(merged.index) + 1.4)
            if min_height > figsize[1]:
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(figsize[0], min_height))
            y_positions = np.arange(len(merged.index), dtype=float)
            height = 0.36
            ax.barh(y_positions - height / 2, merged["Reference"], height=height, color=reference_color, label="Reference")
            ax.barh(
                y_positions + height / 2,
                merged["Comparison"],
                height=height,
                color=comparison_color,
                label="Comparison",
            )
            ax.set_yticks(y_positions, labels=[_wrap_plot_label(label, width=20) for label in merged.index.astype(str)])
            ax.invert_yaxis()
            ax.set_xlabel(_metric_axis_label(measure=measure, value=value))
            ax.set_ylabel("")
            ax.set_title(title or f"Cell-type communication comparison ({pattern})")
            ax.legend(frameon=False, title="")
            return _maybe_save_show(fig, show=show, save=save), ax

        if plot_type == "ranknet":
            reference_series = _pathway_metric_summary(long_df, measure=measure, value=value)
            comparison_series = _pathway_metric_summary(comparison_long_df, measure=measure, value=value)
            merged = pd.DataFrame(
                {
                    "Reference": reference_series,
                    "Comparison": comparison_series,
                }
            ).fillna(0.0)
            merged["abs_delta"] = (merged["Comparison"] - merged["Reference"]).abs()
            merged["rank"] = merged[["Reference", "Comparison"]].max(axis=1)
            merged = merged.sort_values(["abs_delta", "rank"], ascending=False).head(int(top_n))
            merged = merged.sort_values("abs_delta", ascending=True)
            min_height = max(figsize[1], 0.42 * len(merged.index) + 1.4)
            if min_height > figsize[1]:
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(figsize[0], min_height))
            y_positions = np.arange(len(merged.index), dtype=float)
            for idx, (_, row) in enumerate(merged.iterrows()):
                ax.plot([row["Reference"], row["Comparison"]], [idx, idx], color="#BDBDBD", linewidth=2.2, zorder=1)
            ax.scatter(merged["Reference"], y_positions, s=90, color=reference_color, label="Reference", zorder=2)
            ax.scatter(merged["Comparison"], y_positions, s=90, color=comparison_color, label="Comparison", zorder=2)
            ax.set_yticks(y_positions, labels=[_wrap_plot_label(label, width=22) for label in merged.index.astype(str)])
            ax.tick_params(axis="y", labelsize=9)
            ax.set_xlabel(_metric_axis_label(measure=measure, value=value))
            ax.set_ylabel("")
            ax.set_title(title or "Pathway rank comparison")
            ax.legend(frameon=False, title="", loc="lower right")
            return _maybe_save_show(fig, show=show, save=save), ax

        role_delta = _pathway_role_delta_frame(
            long_df,
            comparison_long_df,
            idents_use=idents_use,
            measure=measure,
            value=value,
        )
        if role_delta.empty:
            raise ValueError("No signaling pathways remain for role-change plotting.")

        ident_labels = role_delta["celltype"].astype(str).drop_duplicates().tolist()
        role_delta = (
            role_delta.sort_values(["celltype", "magnitude"], ascending=[True, False])
            .groupby("celltype", observed=True, sort=False)
            .head(int(top_n))
            .reset_index(drop=True)
        )
        colors = _choose_palette(ident_labels, palette=palette)

        if len(ident_labels) == 1:
            celltype = ident_labels[0]
            role_subset = role_delta.loc[role_delta["celltype"].astype(str) == celltype].copy()
            sizes = np.repeat(120.0, len(role_subset.index))
            if float(role_subset["magnitude"].max()) > 0.0:
                sizes = 70.0 + 220.0 * (role_subset["magnitude"] / float(role_subset["magnitude"].max()))
            _draw_role_scatter_style(
                ax,
                x_values=role_subset["delta_outgoing"].to_numpy(dtype=float),
                y_values=role_subset["delta_incoming"].to_numpy(dtype=float),
                labels=role_subset["signaling"].astype(str).tolist(),
                colors=[colors[celltype]] * len(role_subset.index),
                sizes=sizes,
                xlabel="Delta outgoing communication",
                ylabel="Delta incoming communication",
                title=title or "Communication role change",
                add_zero_guides=True,
            )
            return _maybe_save_show(fig, show=show, save=save), ax

        plt.close(fig)
        ncols = min(3, len(ident_labels))
        nrows = int(np.ceil(len(ident_labels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(figsize[0], 4.5 * ncols), max(figsize[1], 4.0 * nrows)), squeeze=False)
        axes_flat = axes.ravel()
        for axis, celltype in zip(axes_flat, ident_labels):
            role_subset = role_delta.loc[role_delta["celltype"].astype(str) == celltype].copy()
            sizes = np.repeat(120.0, len(role_subset.index))
            if float(role_subset["magnitude"].max()) > 0.0:
                sizes = 70.0 + 220.0 * (role_subset["magnitude"] / float(role_subset["magnitude"].max()))
            _draw_role_scatter_style(
                axis,
                x_values=role_subset["delta_outgoing"].to_numpy(dtype=float),
                y_values=role_subset["delta_incoming"].to_numpy(dtype=float),
                labels=role_subset["signaling"].astype(str).tolist(),
                colors=[colors[celltype]] * len(role_subset.index),
                sizes=sizes,
                xlabel="Delta outgoing communication",
                ylabel="Delta incoming communication",
                title=str(celltype),
                add_zero_guides=True,
            )
        for axis in axes_flat[len(ident_labels):]:
            axis.axis("off")
        if title:
            fig.suptitle(title, fontsize=14, y=1.01)
        fig.tight_layout()
        return _maybe_save_show(fig, show=show, save=save), _largest_content_axis(fig)

    groups = _resolve_group_field(long_df, group_by)
    long_df = long_df.assign(plot_group=groups.astype(str))

    if plot_type == "bar":
        summary = _group_metric_summary(long_df, group_key="plot_group", value=value).head(int(top_n))
        colors = _choose_palette(summary.index.astype(str).tolist(), palette=palette)
        ax.barh(summary.index.astype(str), summary.values, color=[colors[idx] for idx in summary.index.astype(str)])
        ax.invert_yaxis()
        ax.set_yticks(
            np.arange(len(summary.index), dtype=float),
            labels=[_wrap_plot_label(label, width=22) for label in summary.index.astype(str)],
        )
        ax.set_xlabel("Communication score" if value != "count" else "Significant interactions")
        ax.set_ylabel("")
        ax.set_title(title or f"Top communication groups by {group_by}")
        return _maybe_save_show(fig, show=show, save=save), ax

    if plot_type in {"violin", "box"} and facet_by in {"sender", "receiver", "pair"} and group_by == "interaction":
        plt.close(fig)
        distribution_df = _interaction_distribution_frame(long_df)
        ranked_interactions = (
            distribution_df.groupby("interaction_label", observed=True)["score"]
            .sum()
            .sort_values(ascending=False)
            .head(int(top_n))
            .index.astype(str)
            .tolist()
        )
        plot_data = distribution_df.loc[distribution_df["interaction_label"].isin(ranked_interactions)].copy()
        if plot_data.empty:
            raise ValueError("No interaction-level communication records remain for distribution plotting.")
        fig, ax = _draw_distribution_facets(
            plot_data,
            plot_type=plot_type,
            facet_by=facet_by,
            palette=palette,
            figsize=figsize,
            title=title or f"Communication score distribution by {group_by}",
        )
        return _maybe_save_show(fig, show=show, save=save), ax

    plot_data = long_df.copy()
    ranked_groups = _group_metric_summary(plot_data, group_key="plot_group", value=value).head(int(top_n)).index.astype(str).tolist()
    plot_data = plot_data.loc[plot_data["plot_group"].isin(ranked_groups)]
    plot_data["plot_group"] = pd.Categorical(plot_data["plot_group"], categories=ranked_groups, ordered=True)

    hue_colors = None
    if facet_by is not None and facet_by in plot_data.columns:
        hue_labels = plot_data[facet_by].astype(str).dropna().unique().tolist()
        hue_colors = _choose_palette(hue_labels, palette=palette)

    if plot_type == "violin":
        sns.violinplot(
            data=plot_data,
            x="plot_group",
            y="score",
            hue=facet_by if hue_colors is not None else None,
            palette=hue_colors,
            ax=ax,
            inner="box",
            cut=0,
        )
    else:
        sns.boxplot(
            data=plot_data,
            x="plot_group",
            y="score",
            hue=facet_by if hue_colors is not None else None,
            palette=hue_colors,
            ax=ax,
        )

    ax.set_xlabel(group_by.capitalize())
    ax.set_ylabel("Communication score")
    _apply_smart_category_tick_labels(ax, ranked_groups, wrap_width=20)
    ax.set_title(title or f"Communication score distribution by {group_by}")
    if hue_colors is not None and ax.get_legend() is not None:
        ax.legend(frameon=False, title=facet_by.capitalize())
    return _maybe_save_show(fig, show=show, save=save), ax
