"""Regression tests for import-order stability (task-060).

Verifies that the agent submodule test subset passes deterministically
regardless of whether bootstrap/scanner imports run first, and that
dotted monkeypatch paths resolve after stub-phase module manipulation.

Root cause fixed: test_ovagent_bootstrap.py executed smart_agent.py which
loaded agent_backend* into sys.modules, then cleanup only popped three
named entries.  When test_ovagent_registry_scanner.py installed a NEW
stub for omicverse.utils, the leftover sys.modules entries prevented
Python from re-setting submodule attributes on the new stub, causing
monkeypatch.setattr("omicverse.utils.agent_backend_streaming.<sym>", ...)
to fail with AttributeError.
"""

import sys
import types

import pytest


# ============================================================================
# Test 1: omicverse.utils exposes agent submodules as direct attributes
# ============================================================================


class TestUtilsNamespaceExports:
    """After a normal import, omicverse.utils must expose agent submodules."""

    def test_agent_backend_is_attribute(self):
        import omicverse.utils
        assert hasattr(omicverse.utils, "agent_backend")

    def test_agent_backend_streaming_is_attribute(self):
        import omicverse.utils
        assert hasattr(omicverse.utils, "agent_backend_streaming")

    def test_ovagent_is_attribute(self):
        import omicverse.utils
        assert hasattr(omicverse.utils, "ovagent")

    def test_smart_agent_is_attribute(self):
        import omicverse.utils
        assert hasattr(omicverse.utils, "smart_agent")

    def test_verifier_is_attribute(self):
        import omicverse.utils
        assert hasattr(omicverse.utils, "verifier")


# ============================================================================
# Test 2: dotted monkeypatch paths resolve for streaming module
# ============================================================================


class TestMonkeypatchPathResolution:
    """monkeypatch.setattr with dotted string must resolve after imports."""

    def test_streaming_function_patchable(self, monkeypatch):
        """The dotted path used by test_agent_backend_streaming must resolve."""
        sentinel = object()
        monkeypatch.setattr(
            "omicverse.utils.agent_backend_streaming._stream_openai_http_fallback",
            sentinel,
        )
        from omicverse.utils import agent_backend_streaming

        assert agent_backend_streaming._stream_openai_http_fallback is sentinel

    def test_streaming_responses_patchable(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(
            "omicverse.utils.agent_backend_streaming._stream_openai_responses",
            sentinel,
        )
        from omicverse.utils import agent_backend_streaming

        assert agent_backend_streaming._stream_openai_responses is sentinel


# ============================================================================
# Test 3: __getattr__ fallback resolves submodules from sys.modules
# ============================================================================


class TestGetAttrFallback:
    """The __getattr__ on omicverse.utils must resolve submodules that are
    in sys.modules but not set as attributes (e.g. after stub replacement)."""

    def test_getattr_resolves_known_submodule(self):
        import omicverse.utils  # noqa: F401 — trigger import for side effect
        utils_mod = sys.modules["omicverse.utils"]

        # Only run this test if the real module (with __getattr__) is loaded
        if not hasattr(type(utils_mod), "__getattr__") and not hasattr(utils_mod, "__getattr__"):
            # Module-level __getattr__ is a function, not a type method
            mod_dict = getattr(utils_mod, "__dict__", {})
            if "__getattr__" not in mod_dict:
                pytest.skip("omicverse.utils is a stub without __getattr__")

        # Ensure agent_backend_streaming is in sys.modules
        assert "omicverse.utils.agent_backend_streaming" in sys.modules

        # Delete the attribute to simulate a stub that never set it
        real_mod = sys.modules["omicverse.utils.agent_backend_streaming"]
        if hasattr(utils_mod, "agent_backend_streaming"):
            delattr(utils_mod, "agent_backend_streaming")

        # __getattr__ should resolve it from sys.modules
        resolved = getattr(utils_mod, "agent_backend_streaming")
        assert resolved is real_mod

        # Restore to avoid affecting other tests
        utils_mod.agent_backend_streaming = real_mod


# ============================================================================
# Test 4: public OmicVerseAgent / ov.Agent entrypoints are not regressed
# ============================================================================


class TestPublicEntrypoints:
    """ov.Agent and OmicVerseAgent must remain accessible."""

    def test_ov_agent_factory_returns_omicverse_agent(self):
        from omicverse.utils.smart_agent import Agent, OmicVerseAgent

        # Agent is a factory function, not the class itself
        assert callable(Agent)
        assert OmicVerseAgent is not None

    @pytest.mark.skipif(
        "omicverse.utils" in sys.modules
        and not hasattr(sys.modules["omicverse.utils"], "__all__"),
        reason="omicverse.utils is a test stub",
    )
    def test_utils_exposes_agent(self):
        import omicverse.utils

        assert hasattr(omicverse.utils, "Agent")
        assert hasattr(omicverse.utils, "OmicVerseAgent")

    @pytest.mark.skipif(
        "omicverse.utils" in sys.modules
        and not hasattr(sys.modules["omicverse.utils"], "__all__"),
        reason="omicverse.utils is a test stub",
    )
    def test_agent_in_all(self):
        import omicverse.utils

        assert "Agent" in omicverse.utils.__all__
        assert "OmicVerseAgent" in omicverse.utils.__all__

    @pytest.mark.skipif(
        "omicverse.utils" in sys.modules
        and not hasattr(sys.modules["omicverse.utils"], "__all__"),
        reason="omicverse.utils is a test stub",
    )
    def test_ovagent_in_all(self):
        import omicverse.utils

        assert "ovagent" in omicverse.utils.__all__
        assert "agent_backend_streaming" in omicverse.utils.__all__


# ============================================================================
# Test 5: bootstrap test cleanup does not leave orphan modules
# ============================================================================


class TestBootstrapCleanup:
    """Simulate the bootstrap test's module-manipulation pattern and verify
    no orphan omicverse.* entries remain in sys.modules after cleanup."""

    def test_no_orphan_modules_after_stub_cycle(self):
        from pathlib import Path
        import importlib.machinery
        import importlib.util

        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

        # Snapshot current state
        pre_modules = {
            name: mod
            for name, mod in list(sys.modules.items())
            if name == "omicverse" or name.startswith("omicverse.")
        }

        # Install stubs (simulate bootstrap)
        originals = {
            name: mod
            for name, mod in list(sys.modules.items())
            if name == "omicverse" or name.startswith("omicverse.")
        }
        for name in list(originals):
            sys.modules.pop(name, None)

        ov_stub = types.ModuleType("omicverse")
        ov_stub.__path__ = [str(PACKAGE_ROOT)]
        ov_stub.__spec__ = importlib.machinery.ModuleSpec(
            "omicverse", loader=None, is_package=True
        )
        sys.modules["omicverse"] = ov_stub

        utils_stub = types.ModuleType("omicverse.utils")
        utils_stub.__path__ = [str(PACKAGE_ROOT / "utils")]
        utils_stub.__spec__ = importlib.machinery.ModuleSpec(
            "omicverse.utils", loader=None, is_package=True
        )
        sys.modules["omicverse.utils"] = utils_stub
        ov_stub.utils = utils_stub

        # Import something that pulls in agent_backend chain
        sa_spec = importlib.util.spec_from_file_location(
            "omicverse.utils.smart_agent",
            PACKAGE_ROOT / "utils" / "smart_agent.py",
        )
        sa_mod = importlib.util.module_from_spec(sa_spec)
        sys.modules["omicverse.utils.smart_agent"] = sa_mod
        sa_spec.loader.exec_module(sa_mod)

        # Cleanup: remove ALL omicverse.* then restore originals
        for name in list(sys.modules):
            if name == "omicverse" or name.startswith("omicverse."):
                sys.modules.pop(name, None)
        for name, mod in originals.items():
            if mod is not None:
                sys.modules[name] = mod

        # Verify: should have exactly the original modules back
        post_modules = {
            name: mod
            for name, mod in list(sys.modules.items())
            if name == "omicverse" or name.startswith("omicverse.")
        }
        assert set(post_modules.keys()) == set(pre_modules.keys()), (
            f"Orphan modules: {set(post_modules.keys()) - set(pre_modules.keys())}"
        )
