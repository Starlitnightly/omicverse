"""Tests for the extracted ovagent.registry_scanner module.

Uses direct file import to avoid triggering the full omicverse package
import chain, matching the pattern in test_smart_agent.py.
"""

import ast
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Snapshot ALL omicverse.* modules so we can restore after stub-phase imports.
_ORIGINAL_MODULES = {
    name: mod
    for name, mod in list(sys.modules.items())
    if name == "omicverse" or name.startswith("omicverse.")
}

# Ensure ovagent subpackage can resolve relative imports by
# registering minimal parent stubs if not already present.
_stubs_installed: dict = {}
for _mod_name, _mod_path in [
    ("omicverse", PACKAGE_ROOT),
    ("omicverse.utils", PACKAGE_ROOT / "utils"),
    ("omicverse.utils.ovagent", PACKAGE_ROOT / "utils" / "ovagent"),
]:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        _stub.__path__ = [str(_mod_path)]
        _stub.__spec__ = importlib.machinery.ModuleSpec(
            _mod_name, loader=None, is_package=True,
        )
        sys.modules[_mod_name] = _stub
        _stubs_installed[_mod_name] = _stub

# Wire parent attributes
if "omicverse" in sys.modules and "omicverse.utils" in sys.modules:
    sys.modules["omicverse"].utils = sys.modules["omicverse.utils"]
if "omicverse.utils" in sys.modules and "omicverse.utils.ovagent" in sys.modules:
    sys.modules["omicverse.utils"].ovagent = sys.modules["omicverse.utils.ovagent"]

# Load the module under test via spec
_scanner_path = PACKAGE_ROOT / "utils" / "ovagent" / "registry_scanner.py"
_spec = importlib.util.spec_from_file_location(
    "omicverse.utils.ovagent.registry_scanner", _scanner_path,
)
_scanner_mod = importlib.util.module_from_spec(_spec)
sys.modules["omicverse.utils.ovagent.registry_scanner"] = _scanner_mod
_spec.loader.exec_module(_scanner_mod)

RegistryScanner = _scanner_mod.RegistryScanner
build_compact_registry_summary = _scanner_mod.build_compact_registry_summary
_find_register_function_decorator = _scanner_mod._find_register_function_decorator
_derive_static_signature = _scanner_mod._derive_static_signature
_branch_subject_name = _scanner_mod._branch_subject_name
_branch_string_values = _scanner_mod._branch_string_values
_literal_eval_or_default = _scanner_mod._literal_eval_or_default

# Restore sys.modules to pre-stub state so later test files see a clean namespace.
for _name in list(sys.modules):
    if _name == "omicverse" or _name.startswith("omicverse."):
        sys.modules.pop(_name, None)
for _name, _mod in _ORIGINAL_MODULES.items():
    if _mod is not None:
        sys.modules[_name] = _mod


class TestRegistryScannerInit:
    def test_create_scanner(self):
        scanner = RegistryScanner()
        assert scanner._static_registry_entries_cache is None

    def test_load_caches_entries(self):
        scanner = RegistryScanner()
        entries = scanner.load_static_entries()
        assert isinstance(entries, list)
        assert len(entries) > 0
        # Second call returns same cached list
        assert scanner.load_static_entries() is entries

    def test_ensure_runtime_registry_prefers_full_hydration(self, monkeypatch):
        calls = []

        class _FakeRegistry:
            _registry = None

        def _fake_full_hydrate():
            calls.append("full")
            _FakeRegistry._registry = {"ov.pp.qc": object()}

        def _fake_manifest_hydrate():
            calls.append("manifest")

        monkeypatch.setattr(_scanner_mod, "_global_registry", _FakeRegistry)
        monkeypatch.setattr(_scanner_mod, "_hydrate_registry_for_export", _fake_full_hydrate)

        import types
        manifest_mod = types.ModuleType("omicverse.mcp.manifest")
        manifest_mod.ensure_registry_populated = _fake_manifest_hydrate
        monkeypatch.setitem(sys.modules, "omicverse.mcp.manifest", manifest_mod)

        RegistryScanner.ensure_runtime_registry()

        assert calls == ["full"]

    def test_ensure_runtime_registry_falls_back_to_manifest(self, monkeypatch):
        calls = []

        class _FakeRegistry:
            _registry = None

        def _fake_full_hydrate():
            calls.append("full")
            raise RuntimeError("boom")

        def _fake_manifest_hydrate():
            calls.append("manifest")
            _FakeRegistry._registry = {"ov.pp.qc": object()}

        monkeypatch.setattr(_scanner_mod, "_global_registry", _FakeRegistry)
        monkeypatch.setattr(_scanner_mod, "_hydrate_registry_for_export", _fake_full_hydrate)

        import types
        manifest_mod = types.ModuleType("omicverse.mcp.manifest")
        manifest_mod.ensure_registry_populated = _fake_manifest_hydrate
        monkeypatch.setitem(sys.modules, "omicverse.mcp.manifest", manifest_mod)

        RegistryScanner.ensure_runtime_registry()

        assert calls == ["full", "manifest"]


class TestStaticScan:
    @pytest.fixture(scope="module")
    def scanner(self):
        return RegistryScanner()

    @pytest.fixture(scope="module")
    def entries(self, scanner):
        return scanner.load_static_entries()

    def test_entries_have_required_keys(self, entries):
        required = {"name", "full_name", "source"}
        for entry in entries[:20]:
            assert required.issubset(entry.keys()), f"Missing keys in {entry.get('full_name')}"

    def test_celltypist_branch_indexed(self, entries):
        assert any(
            entry.get("source") == "static_ast_branch"
            and "celltypist" == str(entry.get("branch_value", "")).lower()
            for entry in entries
        )

    def test_dynamo_branch_indexed(self, entries):
        assert any(
            entry.get("source") == "static_ast_branch"
            and "dynamo" == str(entry.get("branch_value", "")).lower()
            for entry in entries
        )

    def test_method_entries_indexed(self, entries):
        assert any(
            entry.get("source") == "static_ast_method"
            for entry in entries
        )

    def test_full_names_start_with_ov(self, entries):
        for entry in entries:
            assert entry["full_name"].startswith("ov.") or "[" in entry["full_name"]


class TestNormalization:
    def test_omicverse_prefix_normalized(self):
        entry = {"full_name": "omicverse.pp.qc", "short_name": "qc"}
        result = RegistryScanner.normalize_entry(entry)
        assert result["full_name"] == "ov.pp.qc"

    def test_settings_domain_mapped(self):
        entry = {"full_name": "omicverse._settings.set_gpu", "short_name": "set_gpu"}
        result = RegistryScanner.normalize_entry(entry)
        assert result["full_name"] == "ov.core.set_gpu"

    def test_already_ov_prefix_unchanged(self):
        entry = {"full_name": "ov.pp.pca", "source": "static_ast"}
        result = RegistryScanner.normalize_entry(entry)
        assert result["full_name"] == "ov.pp.pca"

    def test_registry_full_name_preserved(self):
        entry = {"full_name": "omicverse.pp.qc", "short_name": "qc"}
        result = RegistryScanner.normalize_entry(entry)
        assert result["registry_full_name"] == "omicverse.pp.qc"

    def test_runtime_method_name_preserves_parent_context(self):
        entry = {
            "full_name": "omicverse.single._anno.Annotation.annotate",
            "short_name": "annotate",
            "parent_full_name": "omicverse.single._anno.Annotation",
        }
        result = RegistryScanner.normalize_entry(entry)
        assert result["full_name"] == "ov.single.Annotation.annotate"
        assert result["parent_full_name"] == "ov.single.Annotation"

    def test_runtime_branch_name_preserves_parent_context(self):
        entry = {
            "full_name": "omicverse.pp._preprocess.pca[mode=cpu]",
            "short_name": "cpu",
            "parent_full_name": "omicverse.pp._preprocess.pca",
            "branch_parameter": "mode",
            "branch_value": "cpu",
        }
        result = RegistryScanner.normalize_entry(entry)
        assert result["full_name"] == "ov.pp.pca[mode=cpu]"
        assert result["parent_full_name"] == "ov.pp.pca"


class TestScoring:
    def test_exact_full_name_match_scores_high(self):
        entry = {"full_name": "ov.pp.qc", "short_name": "qc", "aliases": [], "description": "quality control"}
        score = RegistryScanner.score_entry("ov.pp.qc", entry)
        assert score >= 10.0

    def test_alias_match_scores_well(self):
        entry = {"full_name": "ov.pp.qc", "short_name": "qc", "aliases": ["quality control"], "description": ""}
        score = RegistryScanner.score_entry("quality control", entry)
        assert score >= 8.0

    def test_irrelevant_query_scores_low(self):
        # ov.pp.* entries get a +0.5 domain bonus, so truly irrelevant queries
        # score at most the domain bonus (no keyword overlap)
        entry = {"full_name": "ov.pp.qc", "short_name": "qc", "aliases": [], "description": "quality control"}
        score = RegistryScanner.score_entry("zzzznothing", entry)
        assert score <= 0.5  # only the domain bonus, no keyword matches

    def test_dataset_penalty_applied(self):
        entry = {"full_name": "ov.datasets.pbmc3k", "short_name": "pbmc3k", "aliases": [], "description": "pbmc"}
        score = RegistryScanner.score_entry("pbmc3k", entry)
        entry_pp = {"full_name": "ov.pp.pbmc3k", "short_name": "pbmc3k", "aliases": [], "description": "pbmc"}
        score_pp = RegistryScanner.score_entry("pbmc3k", entry_pp)
        assert score < score_pp


class TestCollectStatic:
    def test_collect_returns_entries(self):
        scanner = RegistryScanner()
        results = scanner.collect_static_entries("qc", max_entries=5)
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_empty_query_returns_empty(self):
        scanner = RegistryScanner()
        results = scanner.collect_static_entries("", max_entries=5)
        assert results == []


class TestCollectRelevant:
    def test_exact_api_match_beats_weak_prerequisite_mentions(self, monkeypatch):
        scanner = RegistryScanner()

        runtime_entries = [
            {
                "full_name": "omicverse.pp._preprocess.pca",
                "short_name": "pca",
                "aliases": ["pca", "principal component analysis"],
                "description": "Run PCA on scaled data.",
                "source": "runtime",
            },
            {
                "full_name": "omicverse.utils._mde.mde",
                "short_name": "mde",
                "aliases": ["mde"],
                "description": "Minimum Distortion Embedding for visualization.",
                "prerequisites": {"functions": ["pca"]},
                "requires": {"obsm": ["X_pca"]},
                "source": "runtime",
            },
            {
                "full_name": "omicverse.pp._preprocess.pca[mode=cpu]",
                "short_name": "cpu",
                "aliases": ["pca cpu"],
                "description": "CPU branch for PCA.",
                "parent_full_name": "omicverse.pp._preprocess.pca",
                "branch_parameter": "mode",
                "branch_value": "cpu",
                "source": "runtime_derived_branch",
            },
        ]

        monkeypatch.setattr(
            RegistryScanner,
            "_iter_runtime_entries",
            staticmethod(lambda: runtime_entries),
        )
        monkeypatch.setattr(scanner, "load_static_entries", lambda: [])

        results = scanner.collect_relevant_entries("pca", max_entries=3)

        assert [entry["full_name"] for entry in results] == [
            "ov.pp.pca",
            "ov.pp.pca[mode=cpu]",
            "ov.utils.mde",
        ]


class TestCompactSummary:
    def test_build_compact_registry_summary_formats_category_lines(self, monkeypatch):
        scanner = RegistryScanner()

        monkeypatch.setattr(
            scanner,
            "load_static_entries",
            lambda: [
                {"full_name": "ov.pp.qc", "category": "preprocessing"},
                {"full_name": "ov.pp.pca", "category": "preprocessing"},
                {"full_name": "ov.pl.umap", "category": "pl"},
            ],
        )

        summary = build_compact_registry_summary(scanner)

        assert "- **pl** (1 functions): ov.pl.umap" in summary
        assert "- **preprocessing** (2 functions): ov.pp.qc, ov.pp.pca" in summary


class TestHelpers:
    def test_literal_eval_or_default_string(self):
        node = ast.Constant(value="hello")
        assert _literal_eval_or_default(node, "default") == "hello"

    def test_literal_eval_or_default_none(self):
        assert _literal_eval_or_default(None, "default") == "default"

    def test_derive_static_signature_class(self):
        node = ast.ClassDef(name="MyClass", bases=[], keywords=[], body=[], decorator_list=[])
        assert _derive_static_signature(node) == "MyClass(...)"

    def test_branch_subject_name(self):
        assert _branch_subject_name(ast.Name(id="method")) == "method"
        assert _branch_subject_name(ast.Attribute(attr="mode", value=ast.Name(id="self"))) == "mode"
        assert _branch_subject_name(ast.Constant(value=42)) == ""

    def test_branch_string_values(self):
        assert _branch_string_values(ast.Constant(value="hello")) == ["hello"]
        assert _branch_string_values(ast.Constant(value=42)) == []
        assert _branch_string_values(ast.List(elts=[
            ast.Constant(value="a"), ast.Constant(value="b"),
        ])) == ["a", "b"]
