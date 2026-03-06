"""Tests for ``omicverse.fm.registry`` plugin system."""

import importlib
import importlib.metadata
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from omicverse.fm.registry import (
    GeneIDScheme,
    HardwareRequirements,
    ModelRegistry,
    ModelSpec,
    Modality,
    OutputKeys,
    SkillReadyStatus,
    TaskType,
    get_registry,
)
from omicverse.fm.adapters.base import BaseAdapter


# ===========================================================================
# Helpers
# ===========================================================================


def _make_spec(name: str = "test_model", **kwargs) -> ModelSpec:
    """Create a minimal ModelSpec for testing."""
    defaults = dict(
        name=name,
        version="1.0.0",
        tasks=[TaskType.EMBED],
        modalities=[Modality.RNA],
        species=["human"],
        output_keys=OutputKeys(embedding_key=f"X_{name}"),
        hardware=HardwareRequirements(),
    )
    defaults.update(kwargs)
    return ModelSpec(**defaults)


class DummyAdapter(BaseAdapter):
    """Minimal adapter for plugin tests."""

    def run(self, task, adata_path, output_path, **kwargs):
        return {"status": "success"}

    def _load_model(self, device):
        pass

    def _preprocess(self, adata, task):
        return adata

    def _postprocess(self, adata, embeddings, task):
        return ["X_dummy"]


def _fresh_registry() -> ModelRegistry:
    """Create a fresh registry (not the global singleton)."""
    return ModelRegistry()


# ===========================================================================
# Builtin adapter resolution
# ===========================================================================


class TestBuiltinAdapterResolution:
    def test_builtin_adapter_class_resolves(self):
        """Built-in adapter classes should be lazily loadable."""
        registry = get_registry()
        for name in ("scgpt", "geneformer", "uce"):
            cls = registry.get_adapter_class(name)
            assert cls is not None, f"{name} adapter class should resolve"

    def test_builtin_adapter_class_cached(self):
        registry = get_registry()
        cls1 = registry.get_adapter_class("scgpt")
        cls2 = registry.get_adapter_class("scgpt")
        assert cls1 is cls2

    def test_builtin_unknown_model_returns_none(self):
        registry = get_registry()
        cls = registry.get_adapter_class("nonexistent_model_xyz")
        assert cls is None

    def test_all_builtin_names_have_specs(self):
        registry = _fresh_registry()
        for name in registry._builtin_adapter_imports:
            spec = registry.get(name)
            assert spec is not None, f"Builtin adapter '{name}' has no matching spec"


# ===========================================================================
# Plugin registration
# ===========================================================================


class TestPluginRegistration:
    def test_register_plugin_model(self):
        registry = _fresh_registry()
        spec = _make_spec("my_plugin_model")
        registry.register(spec, DummyAdapter, source="test-plugin")

        assert registry.get("my_plugin_model") is not None
        assert registry.get_adapter_class("my_plugin_model") is DummyAdapter

    def test_builtin_protected_from_override(self):
        registry = _fresh_registry()
        # Try to override a built-in model
        evil_spec = _make_spec("scgpt", version="999")
        registry.register(evil_spec, DummyAdapter, source="evil-plugin")

        # Built-in should still be there with original version
        spec = registry.get("scgpt")
        assert spec.version != "999"

    def test_plugin_overrides_plugin(self):
        registry = _fresh_registry()
        spec_v1 = _make_spec("plugin_model", version="1.0")
        spec_v2 = _make_spec("plugin_model", version="2.0")

        registry.register(spec_v1, DummyAdapter, source="plugin-a")
        registry.register(spec_v2, DummyAdapter, source="plugin-b")

        spec = registry.get("plugin_model")
        assert spec.version == "2.0"


# ===========================================================================
# Entry point discovery
# ===========================================================================


class TestEntryPointDiscovery:
    def _make_entry_points_mock(self, eps_list):
        """Create a mock for importlib.metadata.entry_points that returns eps_list for 'omicverse.fm'."""
        # The registry code handles both dict-style and select-style
        result = MagicMock()
        result.select.return_value = eps_list
        # Also handle dict-style access
        result.get.return_value = eps_list
        result.__getitem__ = lambda self, key: eps_list if key == "omicverse.fm" else []
        result.__contains__ = lambda self, key: key == "omicverse.fm"
        return result

    def test_entry_point_single_model(self):
        spec = _make_spec("ep_model")

        mock_ep = MagicMock()
        mock_ep.name = "ep_model"
        mock_ep.load.return_value = lambda: (spec, DummyAdapter)

        with patch.object(
            importlib.metadata, "entry_points",
            return_value=self._make_entry_points_mock([mock_ep]),
        ):
            registry = ModelRegistry()

        assert registry.get("ep_model") is not None

    def test_entry_point_multiple_models(self):
        spec_a = _make_spec("ep_multi_a")
        spec_b = _make_spec("ep_multi_b")

        mock_ep = MagicMock()
        mock_ep.name = "ep_multi"
        mock_ep.load.return_value = lambda: [(spec_a, DummyAdapter), (spec_b, DummyAdapter)]

        with patch.object(
            importlib.metadata, "entry_points",
            return_value=self._make_entry_points_mock([mock_ep]),
        ):
            registry = ModelRegistry()

        assert registry.get("ep_multi_a") is not None
        assert registry.get("ep_multi_b") is not None

    def test_entry_point_bad_return_type_skipped(self):
        mock_ep = MagicMock()
        mock_ep.name = "bad_ep"
        mock_ep.load.return_value = lambda: "not a tuple"

        with patch.object(
            importlib.metadata, "entry_points",
            return_value=self._make_entry_points_mock([mock_ep]),
        ):
            registry = ModelRegistry()

        assert registry.get("bad_ep") is None

    def test_entry_point_load_exception_skipped(self):
        mock_ep = MagicMock()
        mock_ep.name = "crash_ep"
        mock_ep.load.side_effect = ImportError("boom")

        with patch.object(
            importlib.metadata, "entry_points",
            return_value=self._make_entry_points_mock([mock_ep]),
        ):
            registry = ModelRegistry()

        # Should not crash; model just not registered
        assert registry.get("crash_ep") is None


# ===========================================================================
# Local plugin discovery
# ===========================================================================


class TestLocalPluginDiscovery:
    def test_local_plugin_loaded(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / ".omicverse" / "plugins" / "fm"
        plugin_dir.mkdir(parents=True)

        plugin_code = textwrap.dedent("""\
            from omicverse.fm.registry import (
                ModelSpec, TaskType, Modality, OutputKeys, HardwareRequirements,
            )
            from omicverse.fm.adapters.base import BaseAdapter

            class LocalTestAdapter(BaseAdapter):
                def run(self, task, adata_path, output_path, **kw):
                    return {"status": "success"}
                def _load_model(self, device): pass
                def _preprocess(self, adata, task): return adata
                def _postprocess(self, adata, emb, task): return ["X_local"]

            def register():
                spec = ModelSpec(
                    name="local_test_model", version="1.0.0",
                    tasks=[TaskType.EMBED], modalities=[Modality.RNA],
                    species=["human"],
                    output_keys=OutputKeys(embedding_key="X_local"),
                    hardware=HardwareRequirements(),
                )
                return (spec, LocalTestAdapter)
        """)
        (plugin_dir / "my_local_model.py").write_text(plugin_code)

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        registry = ModelRegistry()

        assert registry.get("local_test_model") is not None

    def test_local_plugin_no_register_skipped(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / ".omicverse" / "plugins" / "fm"
        plugin_dir.mkdir(parents=True)

        (plugin_dir / "no_register.py").write_text("x = 42\n")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        registry = ModelRegistry()
        # Should not crash
        assert registry.get("no_register") is None

    def test_local_plugin_underscore_skipped(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / ".omicverse" / "plugins" / "fm"
        plugin_dir.mkdir(parents=True)

        (plugin_dir / "_private.py").write_text("def register(): return None\n")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        registry = ModelRegistry()
        # Private files should be skipped
        assert True  # No crash

    def test_local_plugin_syntax_error_skipped(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / ".omicverse" / "plugins" / "fm"
        plugin_dir.mkdir(parents=True)

        (plugin_dir / "broken.py").write_text("def register(:\n")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        registry = ModelRegistry()
        # Should not crash
        assert True

    def test_no_plugin_dir_is_fine(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        registry = ModelRegistry()
        # No ~/.omicverse/plugins/fm/ dir should be fine
        assert len(registry.list_models()) >= 22


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_non_model_spec_rejected(self):
        registry = _fresh_registry()
        not_a_spec = {"name": "fake", "version": "1.0"}

        # Should not crash, but should not register
        registry._validate_and_register(not_a_spec, DummyAdapter, "test")
        assert registry.get("fake") is None

    def test_non_base_adapter_rejected(self):
        registry = _fresh_registry()
        spec = _make_spec("valid_spec")

        class NotAnAdapter:
            pass

        registry._validate_and_register(spec, NotAnAdapter, "test")
        assert registry.get("valid_spec") is None

    def test_valid_plugin_accepted(self):
        registry = _fresh_registry()
        spec = _make_spec("valid_plugin")
        registry._validate_and_register(spec, DummyAdapter, "test")
        assert registry.get("valid_plugin") is not None
