"""Tests for real registry hydration across all phases.

These tests import real OmicVerse modules and require anndata + scanpy.
They verify that the manifest builder produces correct entries when backed
by real function registrations (not mock).
"""

import pytest

from omicverse.mcp.manifest import build_registry_manifest
from tests.mcp._env import skip_no_core, skip_no_scientific


@skip_no_core
@pytest.mark.core
@pytest.mark.real_runtime
class TestRealRegistryP0:
    """Verify P0 tools hydrate with real imports."""

    def setup_method(self):
        import omicverse.mcp.manifest as m
        m._HYDRATED = False

    def test_p0_manifest_has_minimum_tools(self):
        manifest = build_registry_manifest(phase="P0")
        assert len(manifest) >= 9

    def test_p0_has_core_pipeline_tools(self):
        manifest = build_registry_manifest(phase="P0")
        names = {e["tool_name"] for e in manifest}
        expected = {"ov.utils.read", "ov.pp.qc", "ov.pp.scale", "ov.pp.pca",
                    "ov.pp.neighbors", "ov.pp.umap", "ov.pp.leiden"}
        assert expected.issubset(names), f"Missing: {expected - names}"

    def test_p0_tools_have_valid_schemas(self):
        manifest = build_registry_manifest(phase="P0")
        for entry in manifest:
            assert "tool_name" in entry
            assert "parameter_schema" in entry
            schema = entry["parameter_schema"]
            assert "type" in schema
            assert schema["type"] == "object"

    def test_p0_tools_have_availability_info(self):
        manifest = build_registry_manifest(phase="P0")
        for entry in manifest:
            assert "availability" in entry
            avail = entry["availability"]
            assert "available" in avail


@skip_no_core
@pytest.mark.core
@pytest.mark.real_runtime
class TestRealRegistryP05:
    """P0+P0.5 includes markers and viz."""

    def setup_method(self):
        import omicverse.mcp.manifest as m
        m._HYDRATED = False

    def test_p05_includes_find_markers(self):
        manifest = build_registry_manifest(phase="P0+P0.5")
        names = {e["tool_name"] for e in manifest}
        assert "ov.single.find_markers" in names

    def test_p05_tool_count_exceeds_p0(self):
        p0 = build_registry_manifest(phase="P0")
        p05 = build_registry_manifest(phase="P0+P0.5")
        assert len(p05) > len(p0)

    def test_p05_entries_have_rollout_phase(self):
        manifest = build_registry_manifest(phase="P0+P0.5")
        for entry in manifest:
            assert entry["rollout_phase"] in ("P0", "P0.5")


@skip_no_core
@pytest.mark.core
@pytest.mark.real_runtime
class TestRealRegistryP2:
    """P2 class tools register even if deps missing."""

    def setup_method(self):
        import omicverse.mcp.manifest as m
        m._HYDRATED = False

    def test_p2_manifest_includes_class_tools(self):
        manifest = build_registry_manifest(phase="P0+P0.5+P2")
        names = {e["tool_name"] for e in manifest}
        # pySCSA should always be present (core deps only);
        # pyDEG lives in _Deseq2 which may need torch_geometric to import
        assert "ov.single.pyscsa" in names

    def test_unavailable_specs_flagged(self):
        """DCT (pertpy) and LDA_topic (mira) are marked available=False in spec."""
        from omicverse.mcp.class_specs import get_spec
        from omicverse.mcp.availability import check_class_availability

        dct = get_spec("omicverse.single._deg_ct.DCT")
        if dct is not None:
            available, reason = check_class_availability(dct)
            assert available is False

        lda = get_spec("omicverse.utils._cluster.LDA_topic")
        if lda is not None:
            available, reason = check_class_availability(lda)
            assert available is False


@skip_no_scientific
@pytest.mark.scientific
@pytest.mark.real_runtime
class TestRealRegistryScientific:
    """Verify scientific-stack tools are importable and register correctly."""

    def setup_method(self):
        import omicverse.mcp.manifest as m
        m._HYDRATED = False

    def test_scvelo_importable(self):
        import scvelo
        assert hasattr(scvelo, "__version__")

    def test_squidpy_importable(self):
        import squidpy
        assert hasattr(squidpy, "__version__")

    def test_scientific_tools_in_full_manifest(self):
        """P0+P0.5 manifest builds without error when scientific stack present."""
        manifest = build_registry_manifest(phase="P0+P0.5")
        assert len(manifest) >= 15
