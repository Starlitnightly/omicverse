from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_omicverse_plot_packages():
    saved = {
        name: sys.modules.get(name)
        for name in [
            "omicverse",
            "omicverse.pl",
            "omicverse.single",
            "omicverse._registry",
            "omicverse._settings",
        ]
    }
    for name in saved:
        sys.modules.pop(name, None)

    ov_pkg = types.ModuleType("omicverse")
    ov_pkg.__path__ = [str(PACKAGE_ROOT)]
    ov_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse", loader=None, is_package=True)
    sys.modules["omicverse"] = ov_pkg

    pl_pkg = types.ModuleType("omicverse.pl")
    pl_pkg.__path__ = [str(PACKAGE_ROOT / "pl")]
    pl_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse.pl", loader=None, is_package=True)
    sys.modules["omicverse.pl"] = pl_pkg
    ov_pkg.pl = pl_pkg

    single_pkg = types.ModuleType("omicverse.single")
    single_pkg.__path__ = [str(PACKAGE_ROOT / "single")]
    single_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse.single", loader=None, is_package=True)
    sys.modules["omicverse.single"] = single_pkg
    ov_pkg.single = single_pkg

    registry_mod = types.ModuleType("omicverse._registry")

    def register_function(**_kwargs):
        def decorator(func):
            return func

        return decorator

    registry_mod.register_function = register_function
    sys.modules["omicverse._registry"] = registry_mod
    ov_pkg._registry = registry_mod

    settings_mod = types.ModuleType("omicverse._settings")

    class _Colors:
        HEADER = ""
        BOLD = ""
        CYAN = ""
        GREEN = ""
        ENDC = ""

    settings_mod.Colors = _Colors
    settings_mod.EMOJI = {"start": "", "done": ""}
    sys.modules["omicverse._settings"] = settings_mod
    ov_pkg._settings = settings_mod

    return saved


@pytest.fixture
def dynamic_modules():
    saved = _bootstrap_omicverse_plot_packages()
    try:
        dynamic_features_mod = importlib.import_module("omicverse.single._dynamic_features")
        dynamic_trends_mod = importlib.import_module("omicverse.pl._dynamic_trends")
        yield dynamic_features_mod, dynamic_trends_mod
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_dynamic_trends_compare_features_and_groups_share_one_panel(dynamic_modules):
    dynamic_features_mod, dynamic_trends_mod = dynamic_modules
    result = dynamic_features_mod.DynamicFeaturesResult(
        stats=pd.DataFrame(
            [
                {"dataset": "LineageA", "gene": "g1", "success": True},
                {"dataset": "LineageB", "gene": "g1", "success": True},
                {"dataset": "LineageA", "gene": "g2", "success": True},
                {"dataset": "LineageB", "gene": "g2", "success": True},
            ]
        ),
        fitted=pd.DataFrame(
            [
                {"dataset": dataset, "gene": gene, "pseudotime": pt, "fitted": value, "lower": value - 0.1, "upper": value + 0.1}
                for dataset, offset in [("LineageA", 0.0), ("LineageB", 1.0)]
                for gene, scale in [("g1", 1.0), ("g2", 2.0)]
                for pt, value in [(0.0, offset + scale), (1.0, offset + scale + 0.5)]
            ]
        ),
        raw=None,
        models={},
        config={},
    )

    fig, ax = dynamic_trends_mod.dynamic_trends(
        result,
        genes=["g1", "g2"],
        compare_features=True,
        compare_groups=True,
        return_fig=True,
    )

    labels = ax.get_legend_handles_labels()[1]
    assert len(ax.lines) == 4
    assert set(labels) == {
        "LineageA | g1",
        "LineageA | g2",
        "LineageB | g1",
        "LineageB | g2",
    }
    plt.close(fig)


def test_dynamic_trends_compare_features_panels_by_group(dynamic_modules):
    dynamic_features_mod, dynamic_trends_mod = dynamic_modules
    result = dynamic_features_mod.DynamicFeaturesResult(
        stats=pd.DataFrame(
            [
                {"dataset": "LineageA", "gene": "g1", "success": True},
                {"dataset": "LineageA", "gene": "g2", "success": True},
                {"dataset": "LineageB", "gene": "g1", "success": True},
                {"dataset": "LineageB", "gene": "g2", "success": True},
            ]
        ),
        fitted=pd.DataFrame(
            [
                {"dataset": dataset, "gene": gene, "pseudotime": pt, "fitted": value, "lower": value - 0.1, "upper": value + 0.1}
                for dataset, offset in [("LineageA", 0.0), ("LineageB", 1.0)]
                for gene, scale in [("g1", 1.0), ("g2", 2.0)]
                for pt, value in [(0.0, offset + scale), (1.0, offset + scale + 0.5)]
            ]
        ),
        raw=None,
        models={},
        config={},
    )

    fig, axes = dynamic_trends_mod.dynamic_trends(
        result,
        genes=["g1", "g2"],
        compare_features=True,
        return_fig=True,
    )

    assert len(axes) == 2
    assert [ax.get_title() for ax in axes] == ["LineageA", "LineageB"]
    assert all(len(ax.lines) == 2 for ax in axes)
    plt.close(fig)


def test_dynamic_trends_single_group_combined_request_uses_gene_labels_and_blank_title(dynamic_modules):
    dynamic_features_mod, dynamic_trends_mod = dynamic_modules
    result = dynamic_features_mod.DynamicFeaturesResult(
        stats=pd.DataFrame(
            [
                {"dataset": "adata", "gene": "g1", "success": True},
                {"dataset": "adata", "gene": "g2", "success": True},
            ]
        ),
        fitted=pd.DataFrame(
            [
                {"dataset": "adata", "gene": gene, "pseudotime": pt, "fitted": value, "lower": value - 0.1, "upper": value + 0.1}
                for gene, scale in [("g1", 1.0), ("g2", 2.0)]
                for pt, value in [(0.0, scale), (1.0, scale + 0.5)]
            ]
        ),
        raw=None,
        models={},
        config={},
    )

    fig, ax = dynamic_trends_mod.dynamic_trends(
        result,
        genes=["g1", "g2"],
        compare_features=True,
        compare_groups=True,
        return_fig=True,
    )

    labels = ax.get_legend_handles_labels()[1]
    assert labels == ["g1", "g2"]
    assert ax.get_title() == ""
    assert any(line.get_visible() for line in ax.xaxis.get_gridlines())
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_bbox_to_anchor() is not None
    plt.close(fig)


def test_dynamic_trends_returns_none_by_default_and_accepts_nrows_ncols(dynamic_modules):
    dynamic_features_mod, dynamic_trends_mod = dynamic_modules
    result = dynamic_features_mod.DynamicFeaturesResult(
        stats=pd.DataFrame(
            [
                {"dataset": "LineageA", "gene": "g1", "success": True},
                {"dataset": "LineageA", "gene": "g2", "success": True},
                {"dataset": "LineageB", "gene": "g1", "success": True},
                {"dataset": "LineageB", "gene": "g2", "success": True},
            ]
        ),
        fitted=pd.DataFrame(
            [
                {"dataset": dataset, "gene": gene, "pseudotime": pt, "fitted": value, "lower": value - 0.1, "upper": value + 0.1}
                for dataset, offset in [("LineageA", 0.0), ("LineageB", 1.0)]
                for gene, scale in [("g1", 1.0), ("g2", 2.0)]
                for pt, value in [(0.0, offset + scale), (1.0, offset + scale + 0.5)]
            ]
        ),
        raw=None,
        models={},
        config={},
    )

    returned = dynamic_trends_mod.dynamic_trends(
        result,
        genes=["g1", "g2"],
        compare_features=True,
        nrows=1,
        ncols=2,
    )
    assert returned is None
    plt.close("all")


def test_dynamic_trends_add_point_requires_store_raw(dynamic_modules):
    dynamic_features_mod, dynamic_trends_mod = dynamic_modules
    result = dynamic_features_mod.DynamicFeaturesResult(
        stats=pd.DataFrame([{"dataset": "adata", "gene": "g1", "success": True}]),
        fitted=pd.DataFrame(
            [
                {"dataset": "adata", "gene": "g1", "pseudotime": 0.0, "fitted": 1.0, "lower": 0.9, "upper": 1.1},
                {"dataset": "adata", "gene": "g1", "pseudotime": 1.0, "fitted": 1.5, "lower": 1.4, "upper": 1.6},
            ]
        ),
        raw=None,
        models={},
        config={},
    )

    with pytest.raises(ValueError, match="store_raw=True"):
        dynamic_trends_mod.dynamic_trends(result, genes=["g1"], add_point=True)
