"""Integration tests: provider switching and dispatch routing.

Validates that ``OmicVerseLLMBackend`` correctly resolves providers,
routes requests through the right dispatch tables, formats tool results
per wire API, and maintains per-instance state isolation -- all without
making real external API calls.

Gate: ``OV_AGENT_RUN_HARNESS_TESTS=1``
"""
from __future__ import annotations

import os

import pytest

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Provider switching integration tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# -- production imports under test ------------------------------------------

from omicverse.utils.model_config import (
    PROVIDER_REGISTRY,
    WireAPI,
    get_provider,
)
from omicverse.utils.agent_backend_common import (
    _SYNC_DISPATCH,
    _STREAM_DISPATCH,
)
from omicverse.utils.agent_backend import (
    _CHAT_DISPATCH,
    OmicVerseLLMBackend,
)

# -- harness imports --------------------------------------------------------

from tests.integration.fakes import Usage
from tests.integration.helpers import build_fake_llm


# ===================================================================
#  1. Provider dispatch resolution
# ===================================================================


class TestProviderDispatchResolution:
    """Every registered provider resolves to a valid dispatch entry."""

    # Subset of providers that ship with the codebase and must always
    # resolve to a dispatch entry in _SYNC_DISPATCH.
    _CORE_PROVIDERS = ("openai", "anthropic", "google", "dashscope", "python")

    @pytest.mark.parametrize("provider_name", _CORE_PROVIDERS)
    def test_core_provider_in_sync_dispatch(self, provider_name: str):
        info = get_provider(provider_name)
        assert info is not None, f"Provider {provider_name!r} not in registry"
        assert info.wire_api in _SYNC_DISPATCH, (
            f"wire_api {info.wire_api} for {provider_name!r} missing from _SYNC_DISPATCH"
        )

    @pytest.mark.parametrize("provider_name", _CORE_PROVIDERS)
    def test_core_provider_in_stream_dispatch(self, provider_name: str):
        info = get_provider(provider_name)
        assert info is not None
        assert info.wire_api in _STREAM_DISPATCH, (
            f"wire_api {info.wire_api} for {provider_name!r} missing from _STREAM_DISPATCH"
        )

    def test_unregistered_provider_not_in_registry(self):
        assert get_provider("nonexistent_provider_xyz") is None

    def test_dispatch_tables_cover_all_wire_api_variants(self):
        """Every WireAPI enum member appears in at least _SYNC_DISPATCH."""
        for wire in WireAPI:
            assert wire in _SYNC_DISPATCH, (
                f"WireAPI.{wire.name} missing from _SYNC_DISPATCH"
            )

    def test_every_registered_provider_has_sync_dispatch(self):
        """All providers in the registry have a matching sync dispatch entry."""
        for name, info in PROVIDER_REGISTRY.items():
            assert info.wire_api in _SYNC_DISPATCH, (
                f"Provider {name!r} uses wire_api {info.wire_api} "
                "which is not in _SYNC_DISPATCH"
            )


# ===================================================================
#  2. Backend config immutability per provider
# ===================================================================


class TestBackendConfigIsolation:
    """Two backend instances carry independent configuration."""

    def test_instances_have_independent_configs(self):
        backend_a = OmicVerseLLMBackend(model="python", system_prompt="prompt-a")
        backend_b = OmicVerseLLMBackend(
            model="gpt-4o", system_prompt="prompt-b", api_key="sk-fake"
        )

        assert backend_a.config.model == "python"
        assert backend_b.config.model == "gpt-4o"
        assert backend_a.config.provider != backend_b.config.provider
        assert backend_a.config.system_prompt == "prompt-a"
        assert backend_b.config.system_prompt == "prompt-b"

    def test_last_usage_starts_none_for_both(self):
        a = OmicVerseLLMBackend(model="python", system_prompt="a")
        b = OmicVerseLLMBackend(model="python", system_prompt="b")

        assert a.last_usage is None
        assert b.last_usage is None

    @pytest.mark.asyncio
    async def test_last_usage_resets_independently(self, monkeypatch):
        a = OmicVerseLLMBackend(model="python", system_prompt="a")
        b = OmicVerseLLMBackend(model="python", system_prompt="b")

        monkeypatch.setattr(a, "_run_python_local", lambda prompt: "result-a")
        monkeypatch.setattr(b, "_run_python_local", lambda prompt: "result-b")

        await a.run("x")
        assert a.last_usage is None  # monkeypatched method does not set usage

        await b.run("y")
        assert b.last_usage is None
        # The calls were independent -- neither touched the other's usage
        assert a.last_usage is None


# ===================================================================
#  3. Wire API routing consistency
# ===================================================================


class TestWireAPIRoutingConsistency:
    """Dispatch tables are consistent across sync / chat / stream."""

    _NON_LOCAL_WIRES = [w for w in WireAPI if w != WireAPI.LOCAL]

    @pytest.mark.parametrize("wire", _NON_LOCAL_WIRES, ids=lambda w: w.name)
    def test_non_local_wire_in_all_three_dispatches(self, wire: WireAPI):
        assert wire in _SYNC_DISPATCH, f"{wire} missing from _SYNC_DISPATCH"
        assert wire in _CHAT_DISPATCH, f"{wire} missing from _CHAT_DISPATCH"
        assert wire in _STREAM_DISPATCH, f"{wire} missing from _STREAM_DISPATCH"

    def test_local_wire_not_in_chat_dispatch(self):
        """LOCAL provider has no multi-turn chat support."""
        assert WireAPI.LOCAL not in _CHAT_DISPATCH

    def test_local_wire_in_sync_dispatch(self):
        assert WireAPI.LOCAL in _SYNC_DISPATCH

    def test_local_wire_in_stream_dispatch(self):
        assert WireAPI.LOCAL in _STREAM_DISPATCH

    def test_sync_and_stream_have_same_keys(self):
        """_SYNC_DISPATCH and _STREAM_DISPATCH cover the same wire APIs."""
        assert set(_SYNC_DISPATCH.keys()) == set(_STREAM_DISPATCH.keys())


# ===================================================================
#  4. Provider-to-method resolution (monkeypatch approach)
# ===================================================================


class TestProviderMethodResolution:
    """Full dispatch path for the LOCAL (python) provider."""

    @pytest.mark.asyncio
    async def test_python_provider_dispatch(self, monkeypatch):
        backend = OmicVerseLLMBackend(model="python", system_prompt="test")

        # Monkeypatch the local executor to return a known sentinel.
        monkeypatch.setattr(
            backend, "_run_python_local", lambda prompt: "mocked_local_result"
        )

        result = await backend.run("print('hello')")
        assert result == "mocked_local_result"

    @pytest.mark.asyncio
    async def test_python_provider_config(self):
        backend = OmicVerseLLMBackend(model="python", system_prompt="test")
        assert backend.config.provider == "python"

        info = get_provider(backend.config.provider)
        assert info is not None
        assert info.wire_api == WireAPI.LOCAL

    def test_unregistered_provider_raises_on_sync_dispatch(self, monkeypatch):
        """_run_sync raises RuntimeError for an unregistered provider."""
        backend = OmicVerseLLMBackend(model="python", system_prompt="test")
        # Force an unregistered provider name
        monkeypatch.setattr(backend.config, "provider", "nonexistent_provider_xyz")

        with pytest.raises(RuntimeError, match="not registered"):
            backend._run_sync("hello")

    def test_sync_dispatch_resolves_method_name(self):
        """The sync dispatch entry for LOCAL points to _run_python_local."""
        assert _SYNC_DISPATCH[WireAPI.LOCAL] == "_run_python_local"

    def test_chat_dispatch_resolves_openai_method(self):
        """The chat dispatch entry for CHAT_COMPLETIONS points to _chat_tools_openai."""
        assert _CHAT_DISPATCH[WireAPI.CHAT_COMPLETIONS] == "_chat_tools_openai"

    def test_chat_dispatch_resolves_anthropic_method(self):
        assert _CHAT_DISPATCH[WireAPI.ANTHROPIC_MESSAGES] == "_chat_tools_anthropic"


# ===================================================================
#  5. Format tool result message per wire API
# ===================================================================


class TestFormatToolResultByWireAPI:
    """format_tool_result_message() returns provider-specific shapes."""

    def test_openai_compatible_format(self):
        backend = OmicVerseLLMBackend(
            model="gpt-4o", system_prompt="test", api_key="sk-fake"
        )
        msg = backend.format_tool_result_message(
            tool_call_id="call_abc123",
            tool_name="inspect_data",
            result='{"rows": 100}',
        )

        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_abc123"
        assert msg["name"] == "inspect_data"
        assert msg["content"] == '{"rows": 100}'

    def test_anthropic_format(self):
        backend = OmicVerseLLMBackend(
            model="anthropic/claude-opus-4-6-20260201",
            system_prompt="test",
            api_key="sk-fake",
        )
        msg = backend.format_tool_result_message(
            tool_call_id="toolu_xyz789",
            tool_name="execute_code",
            result="output: 42",
        )

        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1

        block = msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_xyz789"
        assert block["content"] == "output: 42"

    def test_gemini_uses_openai_compatible_format(self):
        """Gemini wire API falls through to the default (OpenAI-compatible) branch."""
        backend = OmicVerseLLMBackend(
            model="gemini/gemini-2.5-pro",
            system_prompt="test",
            api_key="fake-key",
        )
        msg = backend.format_tool_result_message(
            tool_call_id="call_gem1",
            tool_name="search",
            result="found it",
        )

        # Gemini uses the else branch (same shape as OpenAI-compatible)
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_gem1"
        assert msg["name"] == "search"
        assert msg["content"] == "found it"

    def test_dashscope_uses_openai_compatible_format(self):
        """DashScope wire API falls through to the default branch."""
        backend = OmicVerseLLMBackend(
            model="qwen-max", system_prompt="test", api_key="fake-key"
        )
        msg = backend.format_tool_result_message(
            tool_call_id="call_ds1",
            tool_name="analyze",
            result="done",
        )

        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_ds1"


# ===================================================================
#  6. Multiple backend instances do not share state
# ===================================================================


class TestBackendStateIsolation:
    """Independent backend instances maintain separate state."""

    @pytest.mark.asyncio
    async def test_run_on_one_does_not_affect_other(self, monkeypatch):
        backend_a = OmicVerseLLMBackend(model="python", system_prompt="a")
        backend_b = OmicVerseLLMBackend(model="python", system_prompt="b")

        monkeypatch.setattr(
            backend_a, "_run_python_local", lambda prompt: "result-a"
        )

        await backend_a.run("code")

        # backend_b was never called -- its last_usage must still be None.
        assert backend_b.last_usage is None

    @pytest.mark.asyncio
    async def test_usage_set_on_called_backend_only(self, monkeypatch):
        backend_a = OmicVerseLLMBackend(model="python", system_prompt="a")
        backend_b = OmicVerseLLMBackend(model="python", system_prompt="b")

        # Provide a mock that actually sets last_usage (like the real method)
        def _mock_local(prompt):
            backend_a.last_usage = Usage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model="python",
                provider="python",
            )
            return "done"

        monkeypatch.setattr(backend_a, "_run_python_local", _mock_local)

        await backend_a.run("code")

        assert backend_a.last_usage is not None
        assert backend_b.last_usage is None

    def test_config_objects_are_distinct(self):
        a = OmicVerseLLMBackend(model="python", system_prompt="a")
        b = OmicVerseLLMBackend(model="python", system_prompt="b")

        assert a.config is not b.config
        assert a.config.system_prompt != b.config.system_prompt
