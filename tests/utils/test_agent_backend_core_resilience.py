"""
Regression tests for core backend resilience hardening (task-001).

Covers:
1. Timeout enforcement — run() and chat() honour _request_timeout_seconds()
2. Dispatch table consistency — _STREAM_DISPATCH includes WireAPI.LOCAL
3. Exception narrowing — fallback behaviour preserved with narrower handlers
"""

import asyncio
import time
from unittest.mock import Mock, patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(monkeypatch, model="gpt-4o", provider="openai"):
    """Create a lightweight OmicVerseLLMBackend without network calls."""
    from omicverse.utils.agent_backend import OmicVerseLLMBackend
    from omicverse.utils.model_config import ModelConfig

    monkeypatch.setattr(
        ModelConfig, "get_provider_from_model", lambda *a, **kw: provider
    )
    return OmicVerseLLMBackend(
        system_prompt="test", model=model, api_key="test-key"
    )


# ---------------------------------------------------------------------------
# 1. Timeout enforcement
# ---------------------------------------------------------------------------


class TestTimeoutEnforcement:
    """run() and chat() must apply asyncio.wait_for with _request_timeout_seconds."""

    @pytest.mark.asyncio
    async def test_run_raises_timeout_error(self, monkeypatch):
        """run() raises TimeoutError when the sync worker exceeds the timeout."""
        backend = _make_backend(monkeypatch)

        def _slow_sync(prompt):
            time.sleep(0.5)
            return "too late"

        monkeypatch.setattr(backend, "_run_sync", _slow_sync)
        monkeypatch.setattr(
            "omicverse.utils.agent_backend._request_timeout_seconds",
            lambda: 0.1,
        )

        with pytest.raises(TimeoutError, match="timed out"):
            await backend.run("hello")

    @pytest.mark.asyncio
    async def test_chat_raises_timeout_error(self, monkeypatch):
        """chat() raises TimeoutError when the sync worker exceeds the timeout."""
        backend = _make_backend(monkeypatch)

        def _slow_chat_sync(messages, tools, tool_choice):
            time.sleep(0.5)

        monkeypatch.setattr(backend, "_chat_sync", _slow_chat_sync)
        monkeypatch.setattr(
            "omicverse.utils.agent_backend._request_timeout_seconds",
            lambda: 0.1,
        )

        with pytest.raises(TimeoutError, match="timed out"):
            await backend.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_run_succeeds_within_timeout(self, monkeypatch):
        """run() returns normally when sync work completes within the timeout."""
        backend = _make_backend(monkeypatch)

        monkeypatch.setattr(backend, "_run_sync", lambda p: "ok")
        monkeypatch.setattr(
            "omicverse.utils.agent_backend._request_timeout_seconds",
            lambda: 5.0,
        )

        result = await backend.run("hello")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_chat_succeeds_within_timeout(self, monkeypatch):
        """chat() returns normally when sync work completes within the timeout."""
        from omicverse.utils.agent_backend_common import ChatResponse, ToolCall

        backend = _make_backend(monkeypatch)
        expected = ChatResponse(
            content="done", tool_calls=[], stop_reason="end_turn"
        )
        monkeypatch.setattr(
            backend, "_chat_sync", lambda m, t, tc: expected
        )
        monkeypatch.setattr(
            "omicverse.utils.agent_backend._request_timeout_seconds",
            lambda: 5.0,
        )

        result = await backend.chat([{"role": "user", "content": "hi"}])
        assert result is expected

    @pytest.mark.asyncio
    async def test_timeout_error_includes_model_info(self, monkeypatch):
        """TimeoutError message includes model and provider for diagnostics."""
        backend = _make_backend(monkeypatch, model="test-model", provider="test-prov")

        monkeypatch.setattr(backend, "_run_sync", lambda p: time.sleep(0.5))
        monkeypatch.setattr(
            "omicverse.utils.agent_backend._request_timeout_seconds",
            lambda: 0.1,
        )

        with pytest.raises(TimeoutError) as exc_info:
            await backend.run("hello")

        msg = str(exc_info.value)
        assert "test-model" in msg
        assert "test-prov" in msg


# ---------------------------------------------------------------------------
# 2. Dispatch table consistency
# ---------------------------------------------------------------------------


class TestDispatchConsistency:
    """_STREAM_DISPATCH must explicitly include WireAPI.LOCAL."""

    def test_stream_dispatch_includes_local(self):
        from omicverse.utils.agent_backend_common import _STREAM_DISPATCH
        from omicverse.utils.model_config import WireAPI

        assert WireAPI.LOCAL in _STREAM_DISPATCH, (
            "WireAPI.LOCAL must be present in _STREAM_DISPATCH"
        )

    def test_sync_and_stream_dispatch_cover_same_wire_apis(self):
        from omicverse.utils.agent_backend_common import (
            _SYNC_DISPATCH,
            _STREAM_DISPATCH,
        )

        assert set(_SYNC_DISPATCH.keys()) == set(_STREAM_DISPATCH.keys()), (
            "_SYNC_DISPATCH and _STREAM_DISPATCH must cover the same WireAPI values"
        )

    def test_stream_dispatch_local_value_is_string(self):
        from omicverse.utils.agent_backend_common import _STREAM_DISPATCH
        from omicverse.utils.model_config import WireAPI

        value = _STREAM_DISPATCH[WireAPI.LOCAL]
        assert isinstance(value, str) and value, (
            "LOCAL dispatch entry must be a non-empty method name string"
        )

    def test_all_wireapi_members_in_sync_dispatch(self):
        """Every WireAPI member must have a _SYNC_DISPATCH entry."""
        from omicverse.utils.agent_backend_common import _SYNC_DISPATCH
        from omicverse.utils.model_config import WireAPI

        for wire in WireAPI:
            assert wire in _SYNC_DISPATCH, f"{wire} missing from _SYNC_DISPATCH"

    def test_all_wireapi_members_in_stream_dispatch(self):
        """Every WireAPI member must have a _STREAM_DISPATCH entry."""
        from omicverse.utils.agent_backend_common import _STREAM_DISPATCH
        from omicverse.utils.model_config import WireAPI

        for wire in WireAPI:
            assert wire in _STREAM_DISPATCH, f"{wire} missing from _STREAM_DISPATCH"


# ---------------------------------------------------------------------------
# 3. Non-breaking normalization fallbacks
# ---------------------------------------------------------------------------


class TestNormalizationFallbacks:
    """Exception narrowing must preserve existing fallback behaviour."""

    def test_wire_model_name_fallback_on_bad_normalize(self, monkeypatch):
        """_wire_model_name falls back to raw model when normalize raises."""
        backend = _make_backend(monkeypatch, model="openai/my-model")
        from omicverse.utils.model_config import ModelConfig

        monkeypatch.setattr(
            ModelConfig,
            "normalize_model_id",
            Mock(side_effect=ValueError("boom")),
        )
        result = backend._wire_model_name()
        # Should strip the openai/ prefix and return the raw suffix
        assert result == "my-model"

    def test_wire_model_name_strips_provider_prefix(self, monkeypatch):
        """_wire_model_name strips known provider prefixes after normalization."""
        backend = _make_backend(monkeypatch, model="openai/gpt-4o")
        result = backend._wire_model_name()
        assert not result.startswith("openai/")

    def test_coerce_int_returns_none_on_type_error(self):
        from omicverse.utils.agent_backend_common import _coerce_int

        assert _coerce_int(None) is None
        assert _coerce_int("abc") is None
        assert _coerce_int(object()) is None

    def test_coerce_int_handles_valid_values(self):
        from omicverse.utils.agent_backend_common import _coerce_int

        assert _coerce_int(42) == 42
        assert _coerce_int(3.7) == 3
        assert _coerce_int("123") == 123
        assert _coerce_int(True) == 1

    def test_normalize_model_for_routing_fallback(self, monkeypatch):
        """_normalize_model_for_routing returns input on normalization failure."""
        from omicverse.utils.model_config import ModelConfig

        monkeypatch.setattr(
            ModelConfig,
            "normalize_model_id",
            Mock(side_effect=AttributeError("no such method")),
        )
        from omicverse.utils.smart_agent import _normalize_model_for_routing

        result = _normalize_model_for_routing("some-model")
        assert result == "some-model"

    def test_normalize_model_for_routing_empty_string(self):
        from omicverse.utils.smart_agent import _normalize_model_for_routing

        assert _normalize_model_for_routing("") == ""
        assert _normalize_model_for_routing("  ") == ""


# ---------------------------------------------------------------------------
# 4. _request_timeout_seconds configurability
# ---------------------------------------------------------------------------


class TestRequestTimeoutSeconds:
    """_request_timeout_seconds must be configurable via environment."""

    def test_default_timeout(self, monkeypatch):
        from omicverse.utils.agent_backend_common import _request_timeout_seconds

        monkeypatch.delenv("OV_AGENT_CHAT_TIMEOUT_SECONDS", raising=False)
        assert _request_timeout_seconds() == 120.0

    def test_custom_timeout_from_env(self, monkeypatch):
        from omicverse.utils.agent_backend_common import _request_timeout_seconds

        monkeypatch.setenv("OV_AGENT_CHAT_TIMEOUT_SECONDS", "30")
        assert _request_timeout_seconds() == 30.0

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        from omicverse.utils.agent_backend_common import _request_timeout_seconds

        monkeypatch.setenv("OV_AGENT_CHAT_TIMEOUT_SECONDS", "not-a-number")
        assert _request_timeout_seconds() == 120.0

    def test_zero_env_falls_back_to_default(self, monkeypatch):
        from omicverse.utils.agent_backend_common import _request_timeout_seconds

        monkeypatch.setenv("OV_AGENT_CHAT_TIMEOUT_SECONDS", "0")
        assert _request_timeout_seconds() == 120.0

    def test_negative_env_falls_back_to_default(self, monkeypatch):
        from omicverse.utils.agent_backend_common import _request_timeout_seconds

        monkeypatch.setenv("OV_AGENT_CHAT_TIMEOUT_SECONDS", "-5")
        assert _request_timeout_seconds() == 120.0
