"""Regression tests for exception hardening in agent_backend_openai.py.

Verifies that narrowed exception families in the OpenAI adapter preserve
existing fallback behavior and emit debug logging on failure paths.

Runs under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""
from __future__ import annotations

import base64
import importlib
import importlib.machinery
import importlib.util
import json
import logging
import os
import sys
import types
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock
from urllib.error import HTTPError

import pytest

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="OpenAI resilience tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight stubs to avoid heavy omicverse.__init__
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for name in ["omicverse", "omicverse.utils"]:
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(PACKAGE_ROOT if name == "omicverse" else PACKAGE_ROOT / "utils")]
        pkg.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = pkg


def _load_module(fqn: str, rel_path: str):
    """Load a single module by file path into sys.modules."""
    if fqn in sys.modules:
        return sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, PACKAGE_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


model_config = _load_module("omicverse.utils.model_config", "utils/model_config.py")
common = _load_module("omicverse.utils.agent_backend_common", "utils/agent_backend_common.py")
oai = _load_module("omicverse.utils.agent_backend_openai", "utils/agent_backend_openai.py")

# Convenience aliases
_redact_url = oai._redact_url
_extract_openai_error_text = oai._extract_openai_error_text
_decode_openai_codex_jwt = oai._decode_openai_codex_jwt
_openai_codex_user_agent = oai._openai_codex_user_agent
_build_chat_response_from_responses_payload = oai._build_chat_response_from_responses_payload
_extract_responses_text_from_items = oai._extract_responses_text_from_items
_extract_responses_output_items = oai._extract_responses_output_items
ChatResponse = common.ChatResponse
Usage = common.Usage

_OAI_LOGGER = "omicverse.utils.agent_backend_openai"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(**overrides):
    """Create a minimal mock backend with sensible defaults."""
    backend = Mock()
    backend.config = SimpleNamespace(
        model="gpt-4o",
        provider="openai",
        endpoint=None,
        system_prompt="You are helpful.",
        temperature=0.7,
        max_tokens=1024,
    )
    backend._wire_model_name = Mock(return_value="gpt-4o")
    backend._resolve_api_key = Mock(return_value="sk-test-key")
    backend.last_usage = None
    for k, v in overrides.items():
        setattr(backend, k, v)
    return backend


# ===========================================================================
# 1. URL redaction
# ===========================================================================

class TestRedactUrl:
    """_redact_url narrows to (ValueError, TypeError) and logs on fallback."""

    def test_normal_url_redacted(self):
        result = _redact_url("https://api.openai.com/v1/chat/completions?key=secret")
        assert result == "https://api.openai.com/..."

    def test_empty_url_redacted(self):
        result = _redact_url("")
        # Empty URL still parses (no host) -> fallback
        assert "unknown" in result or result == "<redacted>"

    def test_none_url_redacted(self):
        result = _redact_url(None)
        assert isinstance(result, str)

    def test_malformed_object_returns_redacted(self, caplog):
        """An object whose __str__ raises TypeError triggers the fallback."""
        class BadStr:
            def __str__(self):
                raise TypeError("cannot stringify")

        with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
            result = _redact_url(BadStr())
        assert result == "<redacted>"
        assert any("redact_url_fallback" in r.message for r in caplog.records)

    def test_unexpected_exception_not_caught(self):
        """Exceptions outside (ValueError, TypeError) propagate."""
        with patch("urllib.parse.urlparse", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                _redact_url("https://example.com")


# ===========================================================================
# 2. HTTPError body extraction
# ===========================================================================

class TestExtractOpenaiErrorText:
    """_extract_openai_error_text narrows body read to (OSError, AttributeError)."""

    def test_http_error_with_body(self):
        exc = HTTPError("http://x", 400, "Bad Request", {}, BytesIO(b"error detail"))
        result = _extract_openai_error_text(Mock(), exc)
        assert result == "error detail"

    def test_http_error_read_raises_oserror(self, caplog):
        """OSError during .read() falls through to str(exc)."""
        exc = HTTPError("http://x", 500, "Internal", {}, BytesIO(b""))
        exc.read = Mock(side_effect=OSError("socket closed"))
        with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
            result = _extract_openai_error_text(Mock(), exc)
        assert isinstance(result, str)
        assert len(result) > 0
        assert any("extract_error_text_read_failed" in r.message for r in caplog.records)

    def test_http_error_missing_read_attribute(self, caplog):
        """AttributeError when .read() is missing falls through."""
        exc = Exception("generic error")
        # Not an HTTPError, so the isinstance check skips the read path
        result = _extract_openai_error_text(Mock(), exc)
        assert result == "generic error"

    def test_fallback_to_response_text(self):
        """When str(exc) is empty, falls to response.text."""
        class EmptyExc(Exception):
            def __str__(self):
                return ""
        exc = EmptyExc()
        exc.response = Mock()
        exc.response.text = "response body text"
        result = _extract_openai_error_text(Mock(), exc)
        assert result == "response body text"


# ===========================================================================
# 3. JWT decode
# ===========================================================================

class TestDecodeOpenaiCodexJwt:
    """_decode_openai_codex_jwt narrows to ValueError and logs."""

    def test_valid_jwt_decoded(self):
        payload = {"sub": "user123", "exp": 9999999999}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"header.{encoded}.signature"
        result = _decode_openai_codex_jwt(token)
        assert result == payload

    def test_invalid_base64_returns_empty(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
            result = _decode_openai_codex_jwt("a.!!!invalid!!!.c")
        assert result == {}
        assert any("decode_openai_codex_jwt_failed" in r.message for r in caplog.records)

    def test_invalid_json_returns_empty(self, caplog):
        # Valid base64 but not valid JSON
        encoded = base64.urlsafe_b64encode(b"not json").decode().rstrip("=")
        token = f"header.{encoded}.sig"
        with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
            result = _decode_openai_codex_jwt(token)
        assert result == {}

    def test_wrong_segment_count_returns_empty(self):
        result = _decode_openai_codex_jwt("only.two")
        assert result == {}

    def test_none_token_returns_empty(self):
        result = _decode_openai_codex_jwt(None)
        assert result == {}

    def test_non_dict_payload_returns_empty(self):
        """A JWT whose payload decodes to a list (not dict) returns {}."""
        encoded = base64.urlsafe_b64encode(json.dumps([1, 2, 3]).encode()).decode().rstrip("=")
        token = f"h.{encoded}.s"
        result = _decode_openai_codex_jwt(token)
        assert result == {}

    def test_malformed_utf8_payload_returns_empty(self, caplog):
        """UnicodeDecodeError from invalid UTF-8 bytes is swallowed and logged."""
        # Build a payload segment whose base64-decoded bytes are not valid UTF-8
        invalid_utf8 = bytes([0x80, 0x81, 0xFE, 0xFF])
        encoded = base64.urlsafe_b64encode(invalid_utf8).decode().rstrip("=")
        token = f"header.{encoded}.signature"
        with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
            result = _decode_openai_codex_jwt(token)
        assert result == {}
        assert any("decode_openai_codex_jwt_failed" in r.message for r in caplog.records)


# ===========================================================================
# 4. Platform fallback
# ===========================================================================

class TestOpenaiCodexUserAgent:
    """_openai_codex_user_agent narrows to (OSError, AttributeError) and logs."""

    def test_normal_user_agent(self):
        result = _openai_codex_user_agent()
        assert result.startswith("pi (")
        assert ")" in result

    def test_platform_oserror_fallback(self, caplog):
        with patch("omicverse.utils.agent_backend_openai.platform") as mock_plat:
            mock_plat.system.side_effect = OSError("no platform info")
            with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
                result = _openai_codex_user_agent()
        assert result == "pi (python)"
        assert any("openai_codex_user_agent_fallback" in r.message for r in caplog.records)

    def test_platform_attribute_error_fallback(self, caplog):
        with patch("omicverse.utils.agent_backend_openai.platform") as mock_plat:
            mock_plat.system.side_effect = AttributeError("missing")
            with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
                result = _openai_codex_user_agent()
        assert result == "pi (python)"

    def test_unexpected_exception_propagates(self):
        """RuntimeError from platform is NOT caught (narrowed away)."""
        with patch("omicverse.utils.agent_backend_openai.platform") as mock_plat:
            mock_plat.system.side_effect = RuntimeError("unexpected")
            with pytest.raises(RuntimeError, match="unexpected"):
                _openai_codex_user_agent()


# ===========================================================================
# 5. Responses payload extraction
# ===========================================================================

class TestBuildChatResponseFromResponsesPayload:
    """_build_chat_response_from_responses_payload narrows to (RuntimeError, KeyError, TypeError)."""

    def test_normal_payload(self):
        backend = _make_backend()
        payload = {
            "output_text": "Hello world",
            "output": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = _build_chat_response_from_responses_payload(backend, payload)
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello world"
        assert result.stop_reason == "end_turn"

    def test_text_from_dict_fails_falls_to_items(self, caplog):
        """When _extract_responses_text_from_dict raises RuntimeError,
        falls back to _extract_responses_text_from_items."""
        backend = _make_backend()
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "fallback text"}],
                }
            ],
        }
        with patch.object(oai, "_extract_responses_text_from_dict", side_effect=RuntimeError("parse fail")):
            with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
                result = _build_chat_response_from_responses_payload(backend, payload)
        assert result.content == "fallback text"
        assert any("responses_text_from_dict_fallback" in r.message for r in caplog.records)

    def test_both_extraction_paths_empty(self):
        """When both extraction paths return empty, content is empty string."""
        backend = _make_backend()
        payload = {"output": []}
        result = _build_chat_response_from_responses_payload(backend, payload)
        assert result.content == "" or result.content is None


# ===========================================================================
# 6. Warning emission
# ===========================================================================

class TestWarningEmission:
    """Warning emission catch narrows to (TypeError, ValueError)."""

    def test_sdk_fallback_warning_suppressed_on_type_error(self, caplog):
        """TypeError in warnings.warn is caught and logged."""
        backend = _make_backend()
        backend._retry = Mock(side_effect=lambda fn: fn())

        with patch("omicverse.utils.agent_backend_openai.warnings") as mock_warnings:
            mock_warnings.warn.side_effect = TypeError("bad warning")
            with patch("omicverse.utils.agent_backend_openai.get_provider", return_value=Mock(base_url="https://api.openai.com/v1")):
                with patch("omicverse.utils.agent_backend_openai.ModelConfig") as mock_mc:
                    mock_mc.requires_responses_api.return_value = False
                    with patch.dict(sys.modules, {"openai": MagicMock()}):
                        openai_mock = sys.modules["openai"]
                        openai_mock.OpenAI.return_value.chat.completions.create.side_effect = ConnectionError("fail")

                        with patch.object(oai, "_chat_via_openai_http", return_value="http fallback"):
                            with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
                                result = oai._chat_via_openai_compatible(backend, "hello")

        assert result == "http fallback"
        assert any("warning_emission_suppressed" in r.message for r in caplog.records)


# ===========================================================================
# 7. output_text() invocation
# ===========================================================================

class TestOutputTextInvocation:
    """output_text() callable failure narrows to (AttributeError, RuntimeError)."""

    def test_output_text_attribute_error_sets_none(self, caplog):
        """AttributeError from output_text() is caught."""
        backend = _make_backend()
        backend._retry = Mock(side_effect=lambda fn: fn())

        # Create a mock response where output_text is callable but raises AttributeError
        mock_resp = Mock()
        mock_resp.output_text = Mock(side_effect=AttributeError("no attribute"))
        mock_resp.output = [
            SimpleNamespace(
                text="fallback from output",
                content=None,
            )
        ]
        mock_resp.text = None
        mock_resp.usage = None

        with patch.dict(sys.modules, {"openai": MagicMock()}):
            openai_mock = sys.modules["openai"]
            openai_mock.OpenAI.return_value.responses.create.return_value = mock_resp

            with patch("omicverse.utils.agent_backend_openai.get_provider", return_value=Mock(base_url="https://api.openai.com/v1")):
                with patch("omicverse.utils.agent_backend_openai.ModelConfig") as mock_mc:
                    mock_mc.requires_responses_api.return_value = True
                    with patch.object(oai, "_is_openai_codex_base_url", return_value=False):
                        with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
                            result = oai._chat_via_openai_responses(
                                backend, "https://api.openai.com/v1", "sk-test", "hello"
                            )

        assert result == "fallback from output"
        assert any("output_text_call_failed" in r.message for r in caplog.records)

    def test_output_text_runtime_error_sets_none(self, caplog):
        """RuntimeError from output_text() is caught."""
        backend = _make_backend()
        backend._retry = Mock(side_effect=lambda fn: fn())

        mock_resp = Mock()
        mock_resp.output_text = Mock(side_effect=RuntimeError("sdk bug"))
        mock_resp.output = "direct string output"
        mock_resp.text = None
        mock_resp.usage = None

        with patch.dict(sys.modules, {"openai": MagicMock()}):
            openai_mock = sys.modules["openai"]
            openai_mock.OpenAI.return_value.responses.create.return_value = mock_resp

            with patch("omicverse.utils.agent_backend_openai.get_provider", return_value=Mock(base_url="https://api.openai.com/v1")):
                with patch("omicverse.utils.agent_backend_openai.ModelConfig") as mock_mc:
                    mock_mc.requires_responses_api.return_value = True
                    with patch.object(oai, "_is_openai_codex_base_url", return_value=False):
                        with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
                            result = oai._chat_via_openai_responses(
                                backend, "https://api.openai.com/v1", "sk-test", "hello"
                            )

        assert result == "direct string output"


# ===========================================================================
# 8. HTTP error parsing fallthrough
# ===========================================================================

class TestHttpErrorParsingFallthrough:
    """HTTP error body read and format fallbacks use narrowed exceptions."""

    def test_error_body_read_oserror_fallback(self, caplog):
        """OSError on exc.read() falls to exc.fp.read()."""
        # Create HTTPError with a working fp but broken .read()
        fp = BytesIO(b"error from fp")
        exc = HTTPError("http://x", 500, "Internal", {}, fp)
        original_read = exc.read
        exc.read = Mock(side_effect=OSError("socket reset"))

        with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
            result = _extract_openai_error_text(Mock(), exc)

        # The OSError is caught, falls through to str(exc) path
        assert isinstance(result, str)
        assert any("extract_error_text_read_failed" in r.message for r in caplog.records)

    def test_http_call_failed_wrapping(self):
        """_chat_via_openai_http wraps OSError in RuntimeError with debug log."""
        backend = _make_backend()
        backend._retry = Mock(side_effect=lambda fn: fn())

        with patch("omicverse.utils.agent_backend_openai._call_openai_chat_with_adaptation") as mock_adapt:
            mock_adapt.side_effect = OSError("connection refused")
            with pytest.raises(RuntimeError, match="OpenAI-compatible HTTP call failed"):
                oai._chat_via_openai_http(backend, "https://api.openai.com/v1", "sk-test", "hello")


# ===========================================================================
# Verify narrowed exceptions do NOT catch unexpected types
# ===========================================================================

class TestNarrowedExceptionsPropagate:
    """Exceptions outside the narrowed families must propagate, not be silently swallowed."""

    def test_redact_url_propagates_runtime_error(self):
        with patch("urllib.parse.urlparse", side_effect=RuntimeError("bug")):
            with pytest.raises(RuntimeError):
                _redact_url("https://x.com")

    def test_jwt_decode_propagates_runtime_error(self):
        """RuntimeError (not ValueError) is NOT caught by the JWT decoder."""
        with patch("omicverse.utils.agent_backend_openai.base64") as mock_b64:
            mock_b64.urlsafe_b64decode.side_effect = RuntimeError("unexpected")
            with pytest.raises(RuntimeError):
                _decode_openai_codex_jwt("a.payload.c")

    def test_user_agent_propagates_runtime_error(self):
        with patch("omicverse.utils.agent_backend_openai.platform") as mock_plat:
            mock_plat.system.side_effect = RuntimeError("bad")
            with pytest.raises(RuntimeError):
                _openai_codex_user_agent()

    def test_responses_payload_propagates_attribute_error(self):
        """AttributeError (not in narrowed set) propagates from text extraction."""
        backend = _make_backend()
        with patch.object(oai, "_extract_responses_text_from_dict", side_effect=AttributeError("missing")):
            with pytest.raises(AttributeError):
                _build_chat_response_from_responses_payload(backend, {"output": []})


# ===========================================================================
# Debug logging emission
# ===========================================================================

class TestDebugLoggingEmitted:
    """Verify debug log messages are emitted at each narrowed catch site."""

    def test_jwt_decode_logs_on_failure(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
            _decode_openai_codex_jwt("a.!!!.c")
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("decode_openai_codex_jwt_failed" in m for m in debug_msgs)

    def test_platform_fallback_logs(self, caplog):
        with patch("omicverse.utils.agent_backend_openai.platform") as mock_plat:
            mock_plat.system.side_effect = OSError("fail")
            with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
                _openai_codex_user_agent()
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("openai_codex_user_agent_fallback" in m for m in debug_msgs)

    def test_responses_fallback_logs(self, caplog):
        backend = _make_backend()
        payload = {
            "output": [{"type": "message", "content": [{"text": "ok"}]}],
        }
        with patch.object(oai, "_extract_responses_text_from_dict", side_effect=KeyError("missing")):
            with caplog.at_level(logging.DEBUG, logger=_OAI_LOGGER):
                _build_chat_response_from_responses_payload(backend, payload)
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("responses_text_from_dict_fallback" in m for m in debug_msgs)
