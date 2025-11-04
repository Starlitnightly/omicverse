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
        "_roe": ["roe", "roe_plot_heatmap"],
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
        "_paga": ["cal_paga", "plot_paga"],
        "_cluster": ["cluster", "LDA_topic", "filtered", "refine_label"],
        "_venn": ["venny4py"],
        "_neighboors": ["neighbors"],
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
