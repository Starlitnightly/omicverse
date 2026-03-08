"""Tests for manifest generation."""

import pytest
from omicverse.mcp.manifest import (
    build_registry_manifest,
    iter_registry_entries,
    dedupe_registry_entries,
    classify_execution_class,
    build_source_ref,
    filter_enabled_entries,
)


class TestIterRegistryEntries:
    def test_extracts_unique_entries(self, mock_registry):
        entries = iter_registry_entries(mock_registry)
        full_names = [e["full_name"] for e in entries]
        assert len(full_names) == len(set(full_names)), "Duplicates in iter output"

    def test_count_matches_mock(self, mock_registry):
        entries = iter_registry_entries(mock_registry)
        # conftest has 15 mock entries (12 function + 3 class)
        assert len(entries) == 15


class TestDedupeRegistryEntries:
    def test_no_dupes_passthrough(self, mock_registry):
        entries = iter_registry_entries(mock_registry)
        deduped = dedupe_registry_entries(entries)
        assert len(deduped) == len(entries)

    def test_handles_duplicate_full_name(self):
        e1 = {"full_name": "omicverse.utils._data.convert_to_pandas", "module": "omicverse.utils._data"}
        e2 = {"full_name": "omicverse.utils._data.convert_to_pandas", "module": "omicverse.utils._data2"}
        deduped = dedupe_registry_entries([e1, e2])
        assert len(deduped) == 1


class TestClassifyExecutionClass:
    def test_stateless_function(self):
        def func(x=1): return x
        entry = {"function": func}
        assert classify_execution_class(entry) == "stateless"

    def test_adata_function(self):
        def func(adata, n=10): return adata
        entry = {"function": func}
        assert classify_execution_class(entry) == "adata"

    def test_class_detection(self):
        class MyClass:
            pass
        entry = {"function": MyClass}
        assert classify_execution_class(entry) == "class"

    def test_adata_from_signature_string(self):
        entry = {"function": None, "signature": "(adata, **kwargs)"}
        assert classify_execution_class(entry) == "adata"


class TestBuildRegistryManifest:
    def test_generates_entries(self, mock_registry):
        manifest = build_registry_manifest(registry=mock_registry)
        assert len(manifest) > 0

    def test_sorted_by_tool_name(self, mock_registry):
        manifest = build_registry_manifest(registry=mock_registry)
        names = [e["tool_name"] for e in manifest]
        assert names == sorted(names)

    def test_all_entries_have_required_fields(self, mock_registry):
        manifest = build_registry_manifest(registry=mock_registry)
        required = [
            "tool_name", "full_name", "kind", "execution_class",
            "adapter_type", "category", "description",
            "parameter_schema", "state_contract", "dependency_contract",
            "return_contract", "availability", "risk_level",
            "rollout_phase", "status",
        ]
        for entry in manifest:
            for field in required:
                assert field in entry, f"Missing {field} in {entry['tool_name']}"

    def test_tool_names_are_canonical(self, mock_registry):
        manifest = build_registry_manifest(registry=mock_registry)
        for entry in manifest:
            assert entry["tool_name"].startswith("ov."), f"Bad name: {entry['tool_name']}"

    def test_no_duplicate_tool_names(self, mock_registry):
        manifest = build_registry_manifest(registry=mock_registry)
        names = [e["tool_name"] for e in manifest]
        assert len(names) == len(set(names))

    def test_execution_classes_correct(self, mock_registry):
        manifest = build_registry_manifest(registry=mock_registry)
        by_name = {e["tool_name"]: e for e in manifest}

        # read is stateless (override makes it special but base class is stateless)
        assert by_name["ov.utils.read"]["execution_class"] == "stateless"
        # pca is adata
        assert by_name["ov.pp.pca"]["execution_class"] == "adata"


class TestFilterEnabledEntries:
    def test_p0_filter(self, mock_registry):
        manifest = build_registry_manifest(registry=mock_registry)
        p0 = filter_enabled_entries(manifest, "P0")
        names = {e["tool_name"] for e in p0}
        assert "ov.pp.pca" in names
        assert "ov.utils.read" in names

    def test_p0_p05_combined(self, mock_registry):
        manifest = build_registry_manifest(registry=mock_registry)
        both = filter_enabled_entries(manifest, "P0+P0.5")
        names = {e["tool_name"] for e in both}
        assert "ov.pp.pca" in names
        assert "ov.single.find_markers" in names

    def test_phase_filter_excludes_deferred(self, mock_registry):
        manifest = build_registry_manifest(registry=mock_registry)
        p0 = filter_enabled_entries(manifest, "P0")
        for e in p0:
            assert e["rollout_phase"] == "P0"


class TestBuildSourceRef:
    def test_returns_string(self):
        def dummy(): pass
        ref = build_source_ref(dummy)
        assert isinstance(ref, str)

    def test_contains_line_number(self):
        def dummy(): pass
        ref = build_source_ref(dummy)
        if ref:
            assert ":" in ref
