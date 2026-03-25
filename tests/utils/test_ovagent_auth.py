"""Tests for ovagent.auth — credential resolution and API key management."""

import os
import sys
import types
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load the modules in isolation (same pattern as test_smart_agent.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ORIGINAL_MODULES = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]
}
for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]:
    sys.modules.pop(name, None)

omicverse_pkg = types.ModuleType("omicverse")
omicverse_pkg.__path__ = [str(PACKAGE_ROOT)]
omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = omicverse_pkg

utils_pkg = types.ModuleType("omicverse.utils")
utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = utils_pkg
omicverse_pkg.utils = utils_pkg

# Force-import smart_agent so transitive deps are resolved
smart_agent_spec = importlib.util.spec_from_file_location(
    "omicverse.utils.smart_agent", PACKAGE_ROOT / "utils" / "smart_agent.py"
)
smart_agent_module = importlib.util.module_from_spec(smart_agent_spec)
sys.modules["omicverse.utils.smart_agent"] = smart_agent_module
assert smart_agent_spec.loader is not None
smart_agent_spec.loader.exec_module(smart_agent_module)

from omicverse.utils.ovagent.auth import (
    ResolvedBackend,
    resolve_model_and_provider,
    collect_api_key_env,
    temporary_api_keys,
    display_backend_info,
)

for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


# ===================================================================
# resolve_model_and_provider
# ===================================================================

class TestResolveModelAndProvider:

    def test_proxy_mode_keeps_model_as_is(self, capsys):
        """When endpoint is provided, model name is not normalized."""
        result = resolve_model_and_provider(
            model="my-custom-model",
            api_key="test-key",
            endpoint="https://proxy.example.com/v1",
        )
        assert isinstance(result, ResolvedBackend)
        assert result.model == "my-custom-model"
        assert result.endpoint == "https://proxy.example.com/v1"
        captured = capsys.readouterr()
        assert "Proxy mode" in captured.out

    def test_returns_resolved_backend_dataclass(self):
        """resolve_model_and_provider returns a ResolvedBackend with all fields."""
        result = resolve_model_and_provider(
            model="gpt-5.2",
            api_key="test-key",
            endpoint="https://proxy.example.com/v1",
        )
        assert hasattr(result, "model")
        assert hasattr(result, "endpoint")
        assert hasattr(result, "provider")

    def test_invalid_model_raises_valueerror(self):
        """Non-proxy mode with invalid model/key raises ValueError."""
        with pytest.raises(ValueError):
            resolve_model_and_provider(
                model="nonexistent-model-xyz-9999",
                api_key=None,
                endpoint=None,
            )


# ===================================================================
# collect_api_key_env
# ===================================================================

class TestCollectApiKeyEnv:

    def test_no_api_key_returns_empty(self):
        result = collect_api_key_env("gpt-5.2", None, None)
        assert result == {}

    def test_empty_string_api_key_returns_empty(self):
        result = collect_api_key_env("gpt-5.2", None, "")
        assert result == {}

    def test_openai_model_includes_openai_key(self):
        result = collect_api_key_env(
            "gpt-5.2",
            "https://api.openai.com/v1",
            "sk-test123",
        )
        assert "OPENAI_API_KEY" in result
        assert result["OPENAI_API_KEY"] == "sk-test123"

    def test_provider_key_lookup_uses_resolved_provider(self):
        result = collect_api_key_env(
            "anthropic/claude-opus-4-6-20260201",
            None,
            "anthropic-test-key",
        )
        assert result["ANTHROPIC_API_KEY"] == "anthropic-test-key"


# ===================================================================
# temporary_api_keys
# ===================================================================

class TestTemporaryApiKeys:

    def test_empty_mapping_is_noop(self):
        """Empty mapping yields without touching env."""
        before = dict(os.environ)
        with temporary_api_keys({}):
            assert dict(os.environ) == before

    def test_injects_and_cleans_up(self, monkeypatch):
        """Keys are set during context and removed after."""
        monkeypatch.delenv("__OV_TEST_KEY__", raising=False)
        mapping = {"__OV_TEST_KEY__": "secret-value"}

        with temporary_api_keys(mapping):
            assert os.environ["__OV_TEST_KEY__"] == "secret-value"

        assert "__OV_TEST_KEY__" not in os.environ

    def test_restores_previous_values(self, monkeypatch):
        """Pre-existing env var is restored after context exits."""
        monkeypatch.setenv("__OV_TEST_KEY__", "original")
        mapping = {"__OV_TEST_KEY__": "temporary"}

        with temporary_api_keys(mapping):
            assert os.environ["__OV_TEST_KEY__"] == "temporary"

        assert os.environ["__OV_TEST_KEY__"] == "original"
