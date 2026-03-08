import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILS_INIT = PROJECT_ROOT / "omicverse" / "utils" / "__init__.py"


def test_utils_exports_agent_entrypoints(monkeypatch):
    """Ensure package-level Agent symbols resolve to the smart_agent module."""

    # Provide a lightweight omicverse package so relative imports succeed
    omicverse_pkg = types.ModuleType("omicverse")
    omicverse_pkg.__path__ = [str(PROJECT_ROOT / "omicverse")]
    monkeypatch.setitem(sys.modules, "omicverse", omicverse_pkg)

    # Pre-seed required submodules with lightweight stubs
    stub_names = [
        "_data",
        "_anndata_rust_patch",
        "_plot",
        "_mde",
        "_syn",
        "_scatterplot",
        "_knn",
        "_heatmap",
        "_roe",
        "_odds_ratio",
        "_shannon_diversity",
        "_resolution",
        "_paga",
        "_cluster",
        "_venn",
        "_lsi",
        "_neighboors",
    ]
    attribute_map = {
        "_data": [
            "read", "read_csv", "read_10x_mtx", "read_h5ad", "read_10x_h5",
            "convert_to_pandas", "wrap_dataframe",
            "download_CaDRReS_model", "download_GDSC_data",
            "download_pathway_database", "download_geneid_annotation_pair",
            "gtf_to_pair_tsv", "download_tosica_gmt", "geneset_prepare",
            "get_gene_annotation", "correlation_pseudotime", "store_layers",
            "retrieve_layers", "easter_egg", "save", "load", "convert_adata_for_rust",
            "anndata_sparse", "np_mean", "np_std", "load_signatures_from_file",
            "predefined_signatures"
        ],
        "_anndata_rust_patch": ["patch_rust_adata"],
        "_plot": [
            "plot_set", "plotset", "ov_plot_set", "pyomic_palette", "palette",
            "blue_palette", "orange_palette", "red_palette", "green_palette",
            "plot_text_set", "ticks_range", "plot_boxplot", "plot_network",
            "plot_cellproportion", "plot_embedding_celltype", "geneset_wordcloud",
            "plot_pca_variance_ratio", "gen_mpl_labels"
        ],
        "_mde": ["mde"],
        "_syn": ["logger", "pancreas", "synthetic_iid", "url_datadir"],
        "_scatterplot": ["diffmap", "draw_graph", "embedding", "pca", "spatial", "tsne", "umap"],
        "_knn": ["weighted_knn_trainer", "weighted_knn_transfer"],
        "_heatmap": [
            "additional_colors", "adjust_palette", "clip", "default_color",
            "default_palette", "get_colors", "interpret_colorkey", "is_categorical",
            "is_list", "is_list_of_str", "is_list_or_array", "is_view", "make_dense",
            "plot_heatmap", "set_colors_for_categorical_obs", "strings_to_categoricals",
            "to_list"
        ],
        "_roe": ["roe", "roe_plot_heatmap", "transform_roe_values"],
        "_odds_ratio": ["odds_ratio", "plot_odds_ratio_heatmap"],
        "_shannon_diversity": [
            "shannon_diversity",
            "compare_shannon_diversity",
            "plot_shannon_diversity",
        ],
        "_resolution": [
            "optimal_resolution",
            "plot_resolution_optimization",
            "resolution_stability_analysis",
        ],
        "_paga": ["cal_paga", "plot_paga", "PAGA_tree"],
        "_cluster": ["cluster", "LDA_topic", "filtered", "refine_label"],
        "_venn": ["venny4py"],
        "_lsi": ["Array", "lsi", "tfidf"],
        "_neighboors": ["neighbors", "calc_kBET", "calc_kSIM"],
    }
    for name in stub_names:
        module = types.ModuleType(f"omicverse.utils.{name}")
        module.__all__ = []  # wildcard imports succeed
        for attr in attribute_map.get(name, []):
            setattr(module, attr, object())
        monkeypatch.setitem(sys.modules, f"omicverse.utils.{name}", module)

    smart_agent = types.ModuleType("omicverse.utils.smart_agent")

    class DummyAgent:  # pragma: no cover - simple sentinel class
        pass

    def fake_list_supported_models():
        return ["dummy-model"]

    smart_agent.Agent = DummyAgent
    smart_agent.OmicVerseAgent = DummyAgent
    smart_agent.list_supported_models = fake_list_supported_models
    monkeypatch.setitem(sys.modules, "omicverse.utils.smart_agent", smart_agent)

    spec = importlib.util.spec_from_file_location("omicverse.utils", UTILS_INIT)
    utils_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "omicverse.utils", utils_module)
    assert spec.loader is not None
    spec.loader.exec_module(utils_module)

    assert utils_module.Agent is DummyAgent
    assert utils_module.OmicVerseAgent is DummyAgent
    assert utils_module.list_supported_models is fake_list_supported_models
    # The module should keep a reference to the imported smart_agent module
    assert utils_module.smart_agent is smart_agent


def test_smart_agent_import_does_not_probe_parent_utils_via___getattr__(monkeypatch):
    """Import compatibility wiring must not trigger omicverse.__getattr__('utils')."""

    package_root = PROJECT_ROOT / "omicverse"
    smart_agent_path = package_root / "utils" / "smart_agent.py"

    for name in [
        "omicverse",
        "omicverse.utils",
        "omicverse.utils.smart_agent",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    getattr_calls = []

    omicverse_pkg = types.ModuleType("omicverse")
    omicverse_pkg.__path__ = [str(package_root)]
    omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
        "omicverse", loader=None, is_package=True
    )

    def _pkg_getattr(name):
        getattr_calls.append(name)
        raise AttributeError(name)

    omicverse_pkg.__getattr__ = _pkg_getattr
    monkeypatch.setitem(sys.modules, "omicverse", omicverse_pkg)

    utils_pkg = types.ModuleType("omicverse.utils")
    utils_pkg.__path__ = [str(package_root / "utils")]
    utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
        "omicverse.utils", loader=None, is_package=True
    )
    monkeypatch.setitem(sys.modules, "omicverse.utils", utils_pkg)

    spec = importlib.util.spec_from_file_location(
        "omicverse.utils.smart_agent",
        smart_agent_path,
    )
    smart_agent_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "omicverse.utils.smart_agent", smart_agent_module)
    assert spec.loader is not None
    spec.loader.exec_module(smart_agent_module)

    assert "utils" not in getattr_calls
    assert omicverse_pkg.utils is utils_pkg
    assert utils_pkg.smart_agent is smart_agent_module
