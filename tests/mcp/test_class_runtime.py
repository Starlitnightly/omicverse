"""Runtime hardening tests for class-backed tools (Phase 2).

Tests availability probing, manifest enrichment, and graceful unavailability.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from omicverse.mcp.class_specs import (
    get_spec, all_specs, build_class_tool_schema,
    ClassWrapperSpec, ActionSpec, get_spec_by_tool_name,
)
from omicverse.mcp.availability import (
    build_availability, check_class_availability,
)


# ---------------------------------------------------------------------------
# Runtime requirements field
# ---------------------------------------------------------------------------


class TestRuntimeRequirements:
    """Verify runtime_requirements on the REAL specs (not mock overrides)."""

    def setup_method(self):
        # Re-register real specs to undo any mock overwrites
        from omicverse.mcp.class_specs import _register_all_specs
        _register_all_specs()

    def test_pydeg_has_runtime_requirements(self):
        spec = get_spec("omicverse.bulk._Deseq2.pyDEG")
        assert "modules" in spec.runtime_requirements
        assert "omicverse.bulk._Deseq2" in spec.runtime_requirements["modules"]

    def test_metacell_requires_seacells(self):
        spec = get_spec("omicverse.single._metacell.MetaCell")
        assert "SEACells" in spec.runtime_requirements.get("packages", [])

    def test_dct_requires_pertpy(self):
        spec = get_spec("omicverse.single._deg_ct.DCT")
        assert "pertpy" in spec.runtime_requirements.get("packages", [])

    def test_lda_requires_mira(self):
        spec = get_spec("omicverse.utils._cluster.LDA_topic")
        assert "mira" in spec.runtime_requirements.get("packages", [])


# ---------------------------------------------------------------------------
# check_class_availability
# ---------------------------------------------------------------------------


class TestCheckClassAvailability:
    def test_deferred_spec_unavailable(self):
        spec = get_spec("omicverse.single._deg_ct.DCT")
        ok, reason = check_class_availability(spec)
        assert ok is False
        assert "Deferred" in reason

    def test_available_spec_with_no_packages(self):
        spec = get_spec("omicverse.bulk._Deseq2.pyDEG")
        ok, reason = check_class_availability(spec)
        assert ok is True
        assert reason == ""

    def test_missing_package_detected(self):
        """Simulate a spec with a non-existent package."""
        spec = ClassWrapperSpec(
            full_name="test.fake.FakeClass",
            tool_name="ov.test.fake",
            available=True,
            runtime_requirements={"packages": ["nonexistent_package_xyz_99"]},
        )
        ok, reason = check_class_availability(spec)
        assert ok is False
        assert "nonexistent_package_xyz_99" in reason

    @patch("importlib.util.find_spec", return_value=None)
    def test_find_spec_returns_none(self, mock_find):
        spec = ClassWrapperSpec(
            full_name="test.mock.MockClass",
            tool_name="ov.test.mock",
            available=True,
            runtime_requirements={"packages": ["some_package"]},
        )
        ok, reason = check_class_availability(spec)
        assert ok is False
        assert "some_package" in reason


# ---------------------------------------------------------------------------
# build_availability with class_spec
# ---------------------------------------------------------------------------


class TestBuildAvailabilityWithClassSpec:
    def test_unavailable_class_spec_propagates(self):
        spec = get_spec("omicverse.single._deg_ct.DCT")
        avail = build_availability(
            {"full_name": spec.full_name, "category": "single"},
            class_spec=spec,
        )
        assert avail["available"] is False
        assert "Deferred" in avail["reason"]

    def test_available_class_spec_ok(self):
        spec = get_spec("omicverse.bulk._Deseq2.pyDEG")
        avail = build_availability(
            {"full_name": spec.full_name, "category": "bulk"},
            class_spec=spec,
        )
        assert avail["available"] is True

    def test_no_class_spec_defaults_true(self):
        avail = build_availability(
            {"full_name": "omicverse.pp._preprocess.pca", "category": "preprocessing"}
        )
        assert avail["available"] is True


# ---------------------------------------------------------------------------
# Manifest enrichment for class tools
# ---------------------------------------------------------------------------


class TestManifestClassEnrichment:
    def test_class_tool_has_action_enum_in_schema(self, mock_registry):
        from tests.mcp.conftest import register_mock_class_specs
        register_mock_class_specs()
        from omicverse.mcp.manifest import build_registry_manifest
        manifest = build_registry_manifest(registry=mock_registry, phase="P0+P0.5+P2")
        pydeg_entries = [e for e in manifest if e["tool_name"] == "ov.bulk.pydeg"]
        assert len(pydeg_entries) == 1
        schema = pydeg_entries[0]["parameter_schema"]
        assert "action" in schema["properties"]
        assert "enum" in schema["properties"]["action"]

    def test_class_tool_has_class_actions_field(self, mock_registry):
        from tests.mcp.conftest import register_mock_class_specs
        register_mock_class_specs()
        from omicverse.mcp.manifest import build_registry_manifest
        manifest = build_registry_manifest(registry=mock_registry, phase="P0+P0.5+P2")
        pydeg = next(e for e in manifest if e["tool_name"] == "ov.bulk.pydeg")
        assert "class_actions" in pydeg
        action_names = [a["name"] for a in pydeg["class_actions"]]
        assert "create" in action_names
        assert "run" in action_names
        assert "destroy" in action_names

    def test_class_tool_schema_has_instance_id(self, mock_registry):
        from tests.mcp.conftest import register_mock_class_specs
        register_mock_class_specs()
        from omicverse.mcp.manifest import build_registry_manifest
        manifest = build_registry_manifest(registry=mock_registry, phase="P0+P0.5+P2")
        pydeg = next(e for e in manifest if e["tool_name"] == "ov.bulk.pydeg")
        assert "instance_id" in pydeg["parameter_schema"]["properties"]


# ---------------------------------------------------------------------------
# Meta tools availability
# ---------------------------------------------------------------------------


class TestMetaToolsAvailability:
    def test_list_tools_includes_availability(self, server_with_p2):
        result = server_with_p2.call_tool("ov.list_tools", {})
        data = result["outputs"][0]["data"]
        assert len(data) > 0
        for item in data:
            assert "availability" in item

    def test_search_tools_includes_availability(self, server_with_p2):
        result = server_with_p2.call_tool("ov.search_tools", {"query": "deg"})
        data = result["outputs"][0]["data"]
        assert len(data) > 0
        for item in data:
            assert "availability" in item

    def test_describe_class_tool_shows_actions(self, server_with_p2):
        result = server_with_p2.call_tool("ov.describe_tool", {"tool_name": "ov.bulk.pydeg"})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert "class_actions" in data
        action_names = [a["name"] for a in data["class_actions"]]
        assert "create" in action_names
        assert "run" in action_names

    def test_describe_class_tool_shows_availability(self, server_with_p2):
        result = server_with_p2.call_tool("ov.describe_tool", {"tool_name": "ov.bulk.pydeg"})
        data = result["outputs"][0]["data"]
        assert "availability" in data
        assert "available" in data["availability"]
        assert "reason" in data["availability"]


# ---------------------------------------------------------------------------
# get_spec_by_tool_name
# ---------------------------------------------------------------------------


class TestGetSpecByToolName:
    def test_finds_existing(self):
        spec = get_spec_by_tool_name("ov.bulk.pydeg")
        assert spec is not None
        assert spec.full_name == "omicverse.bulk._Deseq2.pyDEG"

    def test_returns_none_for_unknown(self):
        spec = get_spec_by_tool_name("ov.nonexistent.tool")
        assert spec is None
