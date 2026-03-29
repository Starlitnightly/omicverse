"""Regression tests: MODEL_CONTEXT_WINDOWS stays synchronized with model_config.py.

Validates that:
- Every non-local model ID registered in PROVIDER_REGISTRY has a corresponding
  entry in MODEL_CONTEXT_WINDOWS (the static context-budget registry).
- The unknown-model fallback still returns _DEFAULT_CONTEXT_WINDOW.
- A debug log is emitted when get_context_window() falls back for an unknown model.

These tests run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""

import importlib
import importlib.machinery
import logging
import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Context budget registry sync tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs (matches test_context_budget.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SAVED = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils"]
}
for name in ["omicverse", "omicverse.utils"]:
    sys.modules.pop(name, None)

_ov_pkg = types.ModuleType("omicverse")
_ov_pkg.__path__ = [str(PACKAGE_ROOT)]
_ov_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = _ov_pkg

_utils_pkg = types.ModuleType("omicverse.utils")
_utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
_utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = _utils_pkg
_ov_pkg.utils = _utils_pkg

from omicverse.utils.ovagent.context_budget import (
    MODEL_CONTEXT_WINDOWS,
    _DEFAULT_CONTEXT_WINDOW,
    get_context_window,
)
from omicverse.utils.model_config import PROVIDER_REGISTRY, WireAPI

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _non_local_registered_model_ids():
    """Collect all model IDs from PROVIDER_REGISTRY, excluding LOCAL providers
    and providers with empty model dicts (ollama, openai_compatible)."""
    ids = set()
    for info in PROVIDER_REGISTRY.values():
        if info.wire_api == WireAPI.LOCAL:
            continue
        ids.update(info.models.keys())
    return ids


# ===================================================================
# 1. Registry synchronization
# ===================================================================


class TestRegistrySync:
    """MODEL_CONTEXT_WINDOWS covers every non-local registered model ID."""

    def test_all_registered_models_have_context_windows(self):
        """Every non-local model in PROVIDER_REGISTRY must appear in
        MODEL_CONTEXT_WINDOWS so that get_context_window() never silently
        falls back to the 32k default for a known model."""
        registered = _non_local_registered_model_ids()
        missing = registered - set(MODEL_CONTEXT_WINDOWS.keys())
        assert not missing, (
            f"Models in PROVIDER_REGISTRY but missing from MODEL_CONTEXT_WINDOWS: "
            f"{sorted(missing)}"
        )

    def test_no_stale_entries_outside_registry(self):
        """Every key in MODEL_CONTEXT_WINDOWS should correspond to a model
        actually registered in PROVIDER_REGISTRY (catches stale entries
        left behind after model removals)."""
        registered = _non_local_registered_model_ids()
        budget_keys = set(MODEL_CONTEXT_WINDOWS.keys())
        stale = budget_keys - registered
        assert not stale, (
            f"MODEL_CONTEXT_WINDOWS has entries not in PROVIDER_REGISTRY: "
            f"{sorted(stale)}"
        )

    def test_context_windows_are_positive_integers(self):
        """All context window values must be positive integers."""
        for model, window in MODEL_CONTEXT_WINDOWS.items():
            assert isinstance(window, int), (
                f"{model}: expected int, got {type(window).__name__}"
            )
            assert window > 0, f"{model}: window must be positive, got {window}"


# ===================================================================
# 2. Unknown-model fallback
# ===================================================================


class TestUnknownModelFallback:
    """get_context_window() returns _DEFAULT_CONTEXT_WINDOW for unknown models."""

    def test_unknown_model_returns_default(self):
        result = get_context_window("nonexistent-model-xyz-999")
        assert result == _DEFAULT_CONTEXT_WINDOW

    def test_default_is_32k(self):
        assert _DEFAULT_CONTEXT_WINDOW == 32_000

    def test_known_model_does_not_return_default(self):
        """A known model should return its registered window, not the default."""
        result = get_context_window("gpt-4o")
        assert result != _DEFAULT_CONTEXT_WINDOW
        assert result == 128_000


# ===================================================================
# 3. Debug logging on fallback
# ===================================================================


class TestFallbackDebugLogging:
    """get_context_window() emits a debug log when falling back."""

    def test_debug_log_emitted_for_unknown_model(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="omicverse.utils.ovagent.context_budget"):
            result = get_context_window("totally-unknown-model")
        assert result == _DEFAULT_CONTEXT_WINDOW
        assert any("totally-unknown-model" in r.message for r in caplog.records), (
            "Expected a debug log mentioning the unknown model name"
        )
        assert any("falling back" in r.message.lower() for r in caplog.records), (
            "Expected log to mention fallback behavior"
        )

    def test_no_debug_log_for_known_model(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="omicverse.utils.ovagent.context_budget"):
            get_context_window("gpt-4o")
        fallback_logs = [
            r for r in caplog.records if "falling back" in r.message.lower()
        ]
        assert not fallback_logs, "No fallback log should be emitted for a known model"
