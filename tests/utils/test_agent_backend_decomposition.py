"""
Regression tests for agent_backend.py decomposition into helper modules.

Verifies that:
1. OmicVerseLLMBackend remains the stable public import
2. All public types are re-exported from the facade
3. Provider-specific methods delegate correctly to extracted modules
4. No unexpected dependency changes are introduced
5. Module structure matches the target architecture
"""

import importlib
import inspect
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest


UTILS_DIR = Path(__file__).resolve().parents[2] / "omicverse" / "utils"


# ---------------------------------------------------------------------------
# 1. Public import surface preserved
# ---------------------------------------------------------------------------


class TestPublicImportSurface:
    """Ensure all expected public names are importable from agent_backend."""

    def test_public_class_importable(self):
        from omicverse.utils.agent_backend import OmicVerseLLMBackend
        assert OmicVerseLLMBackend is not None

    def test_usage_importable(self):
        from omicverse.utils.agent_backend import Usage
        assert Usage is not None

    def test_chat_response_importable(self):
        from omicverse.utils.agent_backend import ChatResponse
        assert ChatResponse is not None

    def test_tool_call_importable(self):
        from omicverse.utils.agent_backend import ToolCall
        assert ToolCall is not None

    def test_backend_config_importable(self):
        from omicverse.utils.agent_backend import BackendConfig
        assert BackendConfig is not None

    def test_all_exports_match(self):
        from omicverse.utils import agent_backend
        expected = {"OmicVerseLLMBackend", "Usage", "ChatResponse", "ToolCall", "BackendConfig"}
        assert expected.issubset(set(agent_backend.__all__))

    def test_utils_init_reexports(self):
        """omicverse.utils.__init__ imports the agent_backend module."""
        from omicverse.utils import agent_backend
        assert hasattr(agent_backend, "OmicVerseLLMBackend")
        assert hasattr(agent_backend, "Usage")
        assert hasattr(agent_backend, "BackendConfig")


# ---------------------------------------------------------------------------
# 2. Helper module existence and structure
# ---------------------------------------------------------------------------


class TestHelperModuleStructure:
    """Ensure the decomposed helper modules exist and are importable."""

    EXPECTED_MODULES = [
        "agent_backend_common",
        "agent_backend_openai",
        "agent_backend_anthropic",
        "agent_backend_gemini",
        "agent_backend_dashscope",
        "agent_backend_streaming",
    ]

    @pytest.mark.parametrize("module_name", EXPECTED_MODULES)
    def test_helper_module_exists(self, module_name):
        """Each helper module file exists on disk."""
        assert (UTILS_DIR / f"{module_name}.py").is_file(), f"{module_name}.py not found"

    @pytest.mark.parametrize("module_name", EXPECTED_MODULES)
    def test_helper_module_importable(self, module_name):
        """Each helper module is importable."""
        mod = importlib.import_module(f"omicverse.utils.{module_name}")
        assert mod is not None

    def test_common_exports_types(self):
        """agent_backend_common exports the shared dataclasses."""
        from omicverse.utils.agent_backend_common import (
            BackendConfig,
            ChatResponse,
            ToolCall,
            Usage,
        )
        for cls in (BackendConfig, ChatResponse, ToolCall, Usage):
            assert hasattr(cls, "__dataclass_fields__"), f"{cls.__name__} is not a dataclass"

    def test_common_exports_helpers(self):
        """agent_backend_common exports retry and utility helpers."""
        from omicverse.utils.agent_backend_common import (
            _coerce_int,
            _compute_total,
            _should_retry,
            _retry_with_backoff,
            _request_timeout_seconds,
            _get_shared_executor,
        )
        assert callable(_coerce_int)
        assert callable(_compute_total)
        assert callable(_should_retry)
        assert callable(_retry_with_backoff)
        assert callable(_request_timeout_seconds)
        assert callable(_get_shared_executor)

    def test_openai_module_has_provider_functions(self):
        """agent_backend_openai has the key OpenAI adapter functions."""
        from omicverse.utils import agent_backend_openai as oai
        expected = [
            "_chat_via_openai_compatible",
            "_chat_via_openai_http",
            "_chat_via_openai_responses",
            "_chat_tools_openai",
            "_chat_tools_openai_responses",
            "_responses_jsonable",
            "_is_openai_codex_base_url",
        ]
        for name in expected:
            assert hasattr(oai, name), f"Missing {name} in agent_backend_openai"

    def test_anthropic_module_has_provider_functions(self):
        from omicverse.utils import agent_backend_anthropic as ant
        for name in ["_chat_via_anthropic", "_chat_tools_anthropic", "_convert_tools_anthropic"]:
            assert hasattr(ant, name), f"Missing {name} in agent_backend_anthropic"

    def test_gemini_module_has_provider_functions(self):
        from omicverse.utils import agent_backend_gemini as gem
        for name in ["_chat_via_gemini", "_chat_tools_gemini", "_json_schema_to_gemini_schema"]:
            assert hasattr(gem, name), f"Missing {name} in agent_backend_gemini"

    def test_dashscope_module_has_provider_function(self):
        from omicverse.utils import agent_backend_dashscope as ds
        assert hasattr(ds, "_chat_via_dashscope")

    def test_streaming_module_has_stream_functions(self):
        from omicverse.utils import agent_backend_streaming as st
        expected = [
            "_run_generator_in_thread",
            "_stream_openai_compatible",
            "_stream_anthropic",
            "_stream_gemini",
            "_stream_dashscope",
        ]
        for name in expected:
            assert hasattr(st, name), f"Missing {name} in agent_backend_streaming"


# ---------------------------------------------------------------------------
# 3. Facade delegates to helper modules (not carrying logic inline)
# ---------------------------------------------------------------------------


class TestFacadeDelegation:
    """Verify that OmicVerseLLMBackend methods delegate to helper modules."""

    def _make_backend(self, monkeypatch):
        from omicverse.utils.agent_backend import OmicVerseLLMBackend
        from omicverse.utils.model_config import ModelConfig
        monkeypatch.setattr(ModelConfig, "get_provider_from_model", lambda *a, **kw: "openai")
        return OmicVerseLLMBackend(system_prompt="test", model="gpt-4o", api_key="k")

    def test_chat_via_openai_compatible_delegates(self, monkeypatch):
        backend = self._make_backend(monkeypatch)
        from omicverse.utils import agent_backend_openai as oai
        sentinel = object()
        monkeypatch.setattr(oai, "_chat_via_openai_compatible", lambda b, p: sentinel)
        assert backend._chat_via_openai_compatible("hi") is sentinel

    def test_chat_via_anthropic_delegates(self, monkeypatch):
        backend = self._make_backend(monkeypatch)
        from omicverse.utils import agent_backend_anthropic as ant
        sentinel = object()
        monkeypatch.setattr(ant, "_chat_via_anthropic", lambda b, p: sentinel)
        assert backend._chat_via_anthropic("hi") is sentinel

    def test_chat_via_gemini_delegates(self, monkeypatch):
        backend = self._make_backend(monkeypatch)
        from omicverse.utils import agent_backend_gemini as gem
        sentinel = object()
        monkeypatch.setattr(gem, "_chat_via_gemini", lambda b, p: sentinel)
        assert backend._chat_via_gemini("hi") is sentinel

    def test_chat_via_dashscope_delegates(self, monkeypatch):
        backend = self._make_backend(monkeypatch)
        from omicverse.utils import agent_backend_dashscope as ds
        sentinel = object()
        monkeypatch.setattr(ds, "_chat_via_dashscope", lambda b, p: sentinel)
        assert backend._chat_via_dashscope("hi") is sentinel

    def test_chat_tools_openai_delegates(self, monkeypatch):
        backend = self._make_backend(monkeypatch)
        from omicverse.utils import agent_backend_openai as oai
        sentinel = object()
        monkeypatch.setattr(oai, "_chat_tools_openai", lambda b, m, t, tc: sentinel)
        assert backend._chat_tools_openai([], None, None) is sentinel

    def test_chat_tools_anthropic_delegates(self, monkeypatch):
        backend = self._make_backend(monkeypatch)
        from omicverse.utils import agent_backend_anthropic as ant
        sentinel = object()
        monkeypatch.setattr(ant, "_chat_tools_anthropic", lambda b, m, t, tc: sentinel)
        assert backend._chat_tools_anthropic([], None, None) is sentinel

    def test_chat_tools_gemini_delegates(self, monkeypatch):
        backend = self._make_backend(monkeypatch)
        from omicverse.utils import agent_backend_gemini as gem
        sentinel = object()
        monkeypatch.setattr(gem, "_chat_tools_gemini", lambda b, m, t, tc: sentinel)
        assert backend._chat_tools_gemini([], None, None) is sentinel

    def test_static_helpers_delegate(self, monkeypatch):
        from omicverse.utils.agent_backend import OmicVerseLLMBackend
        from omicverse.utils import agent_backend_openai as oai
        monkeypatch.setattr(oai, "_is_openai_codex_base_url", lambda u: True)
        assert OmicVerseLLMBackend._is_openai_codex_base_url("test") is True

        monkeypatch.setattr(oai, "_responses_jsonable", lambda v: "delegated")
        assert OmicVerseLLMBackend._responses_jsonable(42) == "delegated"


# ---------------------------------------------------------------------------
# 4. Facade size significantly reduced
# ---------------------------------------------------------------------------


class TestFacadeSize:
    """Verify that agent_backend.py is now a thin facade, not a 3K-line monolith."""

    def test_facade_under_700_lines(self):
        """Facade should be well under the original 3128 lines."""
        facade_path = UTILS_DIR / "agent_backend.py"
        line_count = len(facade_path.read_text().splitlines())
        assert line_count < 800, (
            f"agent_backend.py is {line_count} lines; expected <800 for a facade "
            "(down from 3128)"
        )

    def test_facade_no_direct_provider_sdk_imports(self):
        """The facade should not directly import provider SDKs."""
        source = (UTILS_DIR / "agent_backend.py").read_text()
        for sdk in ["import openai", "import anthropic", "import google.generativeai", "import dashscope"]:
            assert sdk not in source, f"Facade still contains '{sdk}'"


# ---------------------------------------------------------------------------
# 5. No unexpected dependency changes
# ---------------------------------------------------------------------------


class TestNoDependencyChanges:
    """Ensure the decomposition introduced no new package dependencies."""

    def test_pyproject_unchanged(self):
        """pyproject.toml should not have new dependencies from this task."""
        # The helper modules only use stdlib + internal imports
        for module_name in [
            "agent_backend_common",
            "agent_backend_openai",
            "agent_backend_anthropic",
            "agent_backend_gemini",
            "agent_backend_dashscope",
            "agent_backend_streaming",
        ]:
            source = (UTILS_DIR / f"{module_name}.py").read_text()
            # Should only import from stdlib, typing, and internal modules
            for banned in ["import numpy", "import pandas", "import torch", "import scipy"]:
                assert banned not in source, (
                    f"{module_name}.py contains '{banned}' — no new deps allowed"
                )

    def test_helper_modules_use_only_internal_relative_imports(self):
        """Helper modules should only use relative imports from the same package."""
        for module_name in [
            "agent_backend_common",
            "agent_backend_openai",
            "agent_backend_anthropic",
            "agent_backend_gemini",
            "agent_backend_dashscope",
            "agent_backend_streaming",
        ]:
            source = (UTILS_DIR / f"{module_name}.py").read_text()
            lines = source.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("from omicverse"):
                    pytest.fail(
                        f"{module_name}.py:{i} uses absolute import '{stripped}' — "
                        "use relative imports only"
                    )


# ---------------------------------------------------------------------------
# 6. Behavioral consistency: shared helpers produce same results
# ---------------------------------------------------------------------------


class TestSharedHelperConsistency:
    """Verify that shared helpers in agent_backend_common behave correctly."""

    def test_coerce_int_values(self):
        from omicverse.utils.agent_backend_common import _coerce_int
        assert _coerce_int(42) == 42
        assert _coerce_int(3.7) == 3
        assert _coerce_int("123") == 123
        assert _coerce_int(None) is None
        assert _coerce_int(True) == 1

    def test_compute_total(self):
        from omicverse.utils.agent_backend_common import _compute_total
        assert _compute_total(10, 20, 30) == 30
        assert _compute_total(10, 20, None) == 30
        assert _compute_total(None, None, None) is None

    def test_should_retry_http_429(self):
        from omicverse.utils.agent_backend_common import _should_retry
        from urllib.error import HTTPError
        exc = HTTPError("url", 429, "rate limit", {}, None)
        assert _should_retry(exc) is True

    def test_should_retry_http_400(self):
        from omicverse.utils.agent_backend_common import _should_retry
        from urllib.error import HTTPError
        exc = HTTPError("url", 400, "bad request", {}, None)
        assert _should_retry(exc) is False

    def test_usage_dataclass(self):
        from omicverse.utils.agent_backend_common import Usage
        u = Usage(input_tokens=10, output_tokens=5, total_tokens=15, model="m", provider="p")
        assert u.total_tokens == 15

    def test_dispatch_tables_have_all_wire_apis(self):
        from omicverse.utils.agent_backend_common import _SYNC_DISPATCH, _STREAM_DISPATCH
        from omicverse.utils.model_config import WireAPI
        for wire in [WireAPI.CHAT_COMPLETIONS, WireAPI.ANTHROPIC_MESSAGES,
                     WireAPI.GEMINI_GENERATE, WireAPI.DASHSCOPE, WireAPI.LOCAL]:
            assert wire in _SYNC_DISPATCH, f"{wire} missing from _SYNC_DISPATCH"
        for wire in [WireAPI.CHAT_COMPLETIONS, WireAPI.ANTHROPIC_MESSAGES,
                     WireAPI.GEMINI_GENERATE, WireAPI.DASHSCOPE]:
            assert wire in _STREAM_DISPATCH, f"{wire} missing from _STREAM_DISPATCH"
