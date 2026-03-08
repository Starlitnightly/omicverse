"""Tests for P2 class-tool availability detection with real imports.

Verifies that ``check_class_availability()`` correctly probes runtime
requirements for each P2 class tool spec.
"""

import pytest

from omicverse.mcp.class_specs import get_spec, all_specs
from omicverse.mcp.availability import check_class_availability
from tests.mcp._env import (
    skip_no_core, skip_no_seacells,
    has_seacells, has_pertpy, has_mira,
)


@skip_no_core
@pytest.mark.core
@pytest.mark.real_runtime
class TestP2AvailabilityProbing:
    """Verify availability detection for P2 class tools."""

    def test_pydeg_available_with_core(self):
        spec = get_spec("omicverse.bulk._Deseq2.pyDEG")
        assert spec is not None
        available, reason = check_class_availability(spec)
        assert available is True, f"pyDEG should be available: {reason}"

    def test_pyscsa_available_with_core(self):
        spec = get_spec("omicverse.single._anno.pySCSA")
        assert spec is not None
        available, reason = check_class_availability(spec)
        assert available is True, f"pySCSA should be available: {reason}"

    def test_metacell_availability_matches_seacells(self):
        spec = get_spec("omicverse.single._metacell.MetaCell")
        assert spec is not None
        available, reason = check_class_availability(spec)
        if available:
            assert reason == ""
        else:
            assert "SEACells" in reason

    def test_dct_marked_unavailable(self):
        spec = get_spec("omicverse.single._deg_ct.DCT")
        assert spec is not None
        available, reason = check_class_availability(spec)
        # available=False in spec itself (deferred)
        assert available is False

    def test_lda_topic_marked_unavailable(self):
        spec = get_spec("omicverse.utils._cluster.LDA_topic")
        assert spec is not None
        available, reason = check_class_availability(spec)
        assert available is False

    def test_all_specs_have_runtime_requirements(self):
        """Every registered spec should have the runtime_requirements field."""
        specs = all_specs()
        assert len(specs) >= 3  # at least pyDEG, pySCSA, MetaCell
        for name, spec in specs.items():
            assert hasattr(spec, "runtime_requirements"), f"{name} missing runtime_requirements"


@skip_no_seacells
@pytest.mark.extended
@pytest.mark.real_runtime
class TestMetaCellWithSEACells:
    """Only runs if SEACells is installed."""

    def test_metacell_available(self):
        spec = get_spec("omicverse.single._metacell.MetaCell")
        assert spec is not None
        available, reason = check_class_availability(spec)
        assert available is True

    def test_metacell_spec_has_actions(self):
        spec = get_spec("omicverse.single._metacell.MetaCell")
        assert spec is not None
        assert len(spec.actions) >= 1


@skip_no_core
@pytest.mark.core
@pytest.mark.real_runtime
class TestAvailabilityInManifest:
    """Verify availability info propagates into manifest entries."""

    def setup_method(self):
        import omicverse.mcp.manifest as m
        m._HYDRATED = False

    def test_manifest_entries_have_availability(self):
        from omicverse.mcp.manifest import build_registry_manifest
        manifest = build_registry_manifest(phase="P0+P0.5")
        for entry in manifest:
            assert "availability" in entry, f"{entry['tool_name']} missing availability"
            avail = entry["availability"]
            assert "available" in avail
            assert "requires_gpu" in avail

    def test_p0_tools_all_available(self):
        from omicverse.mcp.manifest import build_registry_manifest
        manifest = build_registry_manifest(phase="P0")
        for entry in manifest:
            avail = entry["availability"]
            assert avail["available"] is True, (
                f"{entry['tool_name']} unexpectedly unavailable: {avail['reason']}"
            )
