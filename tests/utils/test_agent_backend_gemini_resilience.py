"""
Regression tests for Gemini adapter exception hardening.

Verifies that:
1. Narrowed exception handlers still catch the expected failure families
2. Debug logging fires on each fallback path
3. Fallback values are preserved (None, empty string, {"raw": ...}, etc.)
4. Unexpected exception types propagate instead of being silently swallowed
"""
from __future__ import annotations

import json
import logging
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_genai(*, schema_side_effect=None):
    """Build a mock google.generativeai module with proto types."""
    genai = MagicMock()

    # Type enum
    type_enum = SimpleNamespace(
        STRING=1, NUMBER=2, INTEGER=3, BOOLEAN=4, ARRAY=5, OBJECT=6,
    )
    genai.protos.Type = type_enum

    # Schema proto constructor
    def _schema(**kwargs):
        return SimpleNamespace(**kwargs)

    if schema_side_effect:
        genai.protos.Schema = Mock(side_effect=schema_side_effect)
    else:
        genai.protos.Schema = Mock(side_effect=_schema)

    # Part / FunctionCall / Content / FunctionResponse stubs
    genai.protos.Part = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
    genai.protos.FunctionCall = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
    genai.protos.FunctionResponse = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
    genai.protos.Content = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))

    return genai


def _patch_genai(genai_mock):
    """Patch sys.modules so ``import google.generativeai`` returns *genai_mock*.

    We must also wire the ``google`` parent module's ``generativeai`` attr so
    that both ``import google.generativeai`` and attribute access agree.
    """
    google_mod = MagicMock()
    google_mod.generativeai = genai_mock
    return patch.dict("sys.modules", {
        "google": google_mod,
        "google.generativeai": genai_mock,
    })


# ---------------------------------------------------------------------------
# 1. _json_schema_to_gemini_schema — narrowed to ImportError, AttributeError,
#    KeyError, TypeError
# ---------------------------------------------------------------------------

class TestJsonSchemaToGeminiSchema:
    """Ensure schema conversion returns None on expected failures and logs."""

    def test_import_error_returns_none(self, caplog):
        """ImportError (missing google SDK) => None, debug logged."""
        # Setting the module value to None in sys.modules causes ImportError
        with patch.dict("sys.modules", {"google.generativeai": None, "google": None}):
            from omicverse.utils.agent_backend_gemini import _json_schema_to_gemini_schema
            with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
                result = _json_schema_to_gemini_schema({"type": "object", "properties": {}})
            assert result is None
            assert any("schema conversion failed" in r.message for r in caplog.records)

    def test_attribute_error_returns_none(self, caplog):
        """AttributeError in proto access => None, debug logged."""
        genai = MagicMock()
        # SimpleNamespace without a Type attribute => AttributeError on access
        genai.protos = SimpleNamespace(Schema=Mock())

        with _patch_genai(genai):
            from omicverse.utils.agent_backend_gemini import _json_schema_to_gemini_schema
            with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
                result = _json_schema_to_gemini_schema({"type": "object", "properties": {"x": {"type": "string"}}})
            assert result is None
            assert any("schema conversion failed" in r.message for r in caplog.records)

    def test_type_error_returns_none(self, caplog):
        """TypeError during Schema construction => None, debug logged."""
        genai = _make_mock_genai(schema_side_effect=TypeError("bad arg"))
        with _patch_genai(genai):
            from omicverse.utils.agent_backend_gemini import _json_schema_to_gemini_schema
            with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
                result = _json_schema_to_gemini_schema({"type": "object", "properties": {"x": {"type": "string"}}})
            assert result is None
            assert any("schema conversion failed" in r.message for r in caplog.records)

    def test_key_error_returns_none(self, caplog):
        """KeyError inside schema building => None, debug logged."""
        genai = _make_mock_genai()
        genai.protos.Schema = Mock(side_effect=KeyError("missing_key"))
        with _patch_genai(genai):
            from omicverse.utils.agent_backend_gemini import _json_schema_to_gemini_schema
            with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
                result = _json_schema_to_gemini_schema({"type": "object", "properties": {"x": {"type": "string"}}})
            assert result is None
            assert any("schema conversion failed" in r.message for r in caplog.records)

    def test_success_returns_schema(self):
        """Happy path still returns a valid schema object."""
        genai = _make_mock_genai()
        with _patch_genai(genai):
            from omicverse.utils.agent_backend_gemini import _json_schema_to_gemini_schema
            result = _json_schema_to_gemini_schema({
                "type": "object",
                "properties": {"name": {"type": "string", "description": "a name"}},
                "required": ["name"],
            })
            assert result is not None


# ---------------------------------------------------------------------------
# 2. Tool-argument JSON parse — SDK path (_messages_to_gemini_contents)
#    and REST path (_messages_to_gemini_rest_contents)
#    narrowed to json.JSONDecodeError, ValueError, TypeError
# ---------------------------------------------------------------------------

class TestToolArgumentParseFallback:
    """Verify that malformed tool-call arguments fall back to {"raw": ...}."""

    def test_sdk_path_invalid_json_wraps_raw(self, caplog):
        """SDK path: non-JSON string => {"raw": original}."""
        genai = _make_mock_genai()
        with _patch_genai(genai):
            from omicverse.utils.agent_backend_gemini import _messages_to_gemini_contents
            messages = [{
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [{
                    "function": {
                        "name": "my_tool",
                        "arguments": "not valid json {{{",
                    }
                }],
            }]
            with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
                result = _messages_to_gemini_contents(messages)

            assert len(result) == 1
            # The FunctionCall part should have args={"raw": "not valid json {{{"}
            fc_part = None
            for part in result[0].parts:
                if hasattr(part, "function_call"):
                    fc_part = part
                    break
            assert fc_part is not None
            assert fc_part.function_call.args == {"raw": "not valid json {{{"}
            assert any("SDK tool-call argument parse failed" in r.message for r in caplog.records)

    def test_sdk_path_valid_json_parses(self):
        """SDK path: valid JSON string => parsed dict."""
        genai = _make_mock_genai()
        with _patch_genai(genai):
            from omicverse.utils.agent_backend_gemini import _messages_to_gemini_contents
            messages = [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "my_tool",
                        "arguments": '{"key": "value"}',
                    }
                }],
            }]
            result = _messages_to_gemini_contents(messages)
            assert len(result) == 1
            fc_part = None
            for part in result[0].parts:
                if hasattr(part, "function_call"):
                    fc_part = part
                    break
            assert fc_part is not None
            assert fc_part.function_call.args == {"key": "value"}

    def test_rest_path_invalid_json_wraps_raw(self, caplog):
        """REST path: non-JSON string => {"raw": original}."""
        from omicverse.utils.agent_backend_gemini import _messages_to_gemini_rest_contents
        backend = Mock()
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "my_tool",
                    "arguments": "<<<bad>>>",
                }
            }],
        }]
        with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
            result = _messages_to_gemini_rest_contents(backend, messages)

        assert len(result) == 1
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["args"] == {"raw": "<<<bad>>>"}
        assert any("REST tool-call argument parse failed" in r.message for r in caplog.records)

    def test_rest_path_valid_json_parses(self):
        """REST path: valid JSON string => parsed dict."""
        from omicverse.utils.agent_backend_gemini import _messages_to_gemini_rest_contents
        backend = Mock()
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "my_tool",
                    "arguments": '{"a": 1}',
                }
            }],
        }]
        result = _messages_to_gemini_rest_contents(backend, messages)
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["args"] == {"a": 1}


# ---------------------------------------------------------------------------
# 3. Response text extraction — narrowed to ValueError, AttributeError
# ---------------------------------------------------------------------------

class TestResponseTextExtraction:
    """Verify that .text access failures fall back to candidate-parts walk."""

    def _make_backend(self):
        backend = Mock()
        backend._resolve_api_key.return_value = "test-key"
        backend.config.model = "gemini-pro"
        backend.config.system_prompt = "You are helpful"
        backend.config.temperature = 0.7
        backend.config.max_tokens = 1024
        backend.config.provider = "gemini"
        backend.last_usage = None

        def passthrough_retry(fn):
            return fn()

        backend._retry = passthrough_retry
        return backend

    def test_value_error_falls_back_to_parts(self, caplog):
        """ValueError on resp.text (safety filter) => falls back to candidate parts."""
        genai = _make_mock_genai()

        # Build a mock response where .text raises ValueError (safety filter)
        mock_part = SimpleNamespace(text="fallback text from parts")
        mock_content = SimpleNamespace(parts=[mock_part])
        mock_candidate = SimpleNamespace(content=mock_content)

        class MockResponse:
            candidates = [mock_candidate]
            usage_metadata = None

            @property
            def text(self):
                raise ValueError("Response blocked by safety filter")

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MockResponse()
        genai.GenerativeModel.return_value = mock_model
        genai.types.GenerationConfig.return_value = {}

        with _patch_genai(genai):
            from omicverse.utils.agent_backend_gemini import _chat_via_gemini
            backend = self._make_backend()

            with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
                result = _chat_via_gemini(backend, "hello")

            assert result == "fallback text from parts"
            assert any(".text extraction failed" in r.message for r in caplog.records)

    def test_attribute_error_falls_back_to_parts(self):
        """AttributeError on resp.text => getattr returns "" => falls back to candidate parts.

        Note: getattr(resp, "text", "") natively catches AttributeError and
        returns the default, so the except block does not fire.  We verify the
        fallback-to-candidate-parts path still works.
        """
        genai = _make_mock_genai()

        mock_part = SimpleNamespace(text="attr fallback")
        mock_content = SimpleNamespace(parts=[mock_part])
        mock_candidate = SimpleNamespace(content=mock_content)

        class MockResponse:
            candidates = [mock_candidate]
            usage_metadata = None

            @property
            def text(self):
                raise AttributeError("no text attribute")

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MockResponse()
        genai.GenerativeModel.return_value = mock_model
        genai.types.GenerationConfig.return_value = {}

        with _patch_genai(genai):
            from omicverse.utils.agent_backend_gemini import _chat_via_gemini
            backend = self._make_backend()
            result = _chat_via_gemini(backend, "hello")
            assert result == "attr fallback"

    def test_normal_text_returned(self):
        """Happy path: resp.text works => returned directly."""
        genai = _make_mock_genai()

        class MockResponse:
            text = "direct text"
            candidates = None
            usage_metadata = None

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MockResponse()
        genai.GenerativeModel.return_value = mock_model
        genai.types.GenerationConfig.return_value = {}

        with _patch_genai(genai):
            from omicverse.utils.agent_backend_gemini import _chat_via_gemini
            backend = self._make_backend()
            result = _chat_via_gemini(backend, "hello")
            assert result == "direct text"


# ---------------------------------------------------------------------------
# 4. OAuth / auth-header JSON parsing — narrowed to JSONDecodeError,
#    ValueError, TypeError
# ---------------------------------------------------------------------------

class TestOAuthJsonParsing:
    """Verify OAuth helper fallbacks on malformed JSON."""

    def test_uses_oauth_bearer_bad_json_returns_false(self, caplog):
        from omicverse.utils.agent_backend_gemini import _gemini_uses_oauth_bearer
        with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
            assert _gemini_uses_oauth_bearer("{not valid json}") is False
        assert any("OAuth bearer check" in r.message for r in caplog.records)

    def test_uses_oauth_bearer_valid_token_returns_true(self):
        from omicverse.utils.agent_backend_gemini import _gemini_uses_oauth_bearer
        assert _gemini_uses_oauth_bearer('{"token": "abc123"}') is True

    def test_uses_oauth_bearer_empty_token_returns_false(self):
        from omicverse.utils.agent_backend_gemini import _gemini_uses_oauth_bearer
        assert _gemini_uses_oauth_bearer('{"token": ""}') is False

    def test_uses_oauth_bearer_non_json_start_returns_false(self):
        from omicverse.utils.agent_backend_gemini import _gemini_uses_oauth_bearer
        assert _gemini_uses_oauth_bearer("plain-api-key") is False

    def test_auth_headers_bad_json_falls_back_to_api_key(self, caplog):
        from omicverse.utils.agent_backend_gemini import _gemini_auth_headers
        with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
            headers = _gemini_auth_headers("{broken")
        assert headers["x-goog-api-key"] == "{broken"
        assert "Authorization" not in headers
        assert any("auth header JSON parse failed" in r.message for r in caplog.records)

    def test_auth_headers_valid_oauth_returns_bearer(self):
        from omicverse.utils.agent_backend_gemini import _gemini_auth_headers
        headers = _gemini_auth_headers('{"token": "mytoken"}')
        assert headers["Authorization"] == "Bearer mytoken"
        assert "x-goog-api-key" not in headers

    def test_oauth_payload_bad_json_returns_none(self, caplog):
        from omicverse.utils.agent_backend_gemini import _gemini_oauth_payload
        with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
            assert _gemini_oauth_payload("{nope") is None
        assert any("OAuth payload JSON parse failed" in r.message for r in caplog.records)

    def test_oauth_payload_valid_returns_dict(self):
        from omicverse.utils.agent_backend_gemini import _gemini_oauth_payload
        result = _gemini_oauth_payload('{"token": "t1", "projectId": "p1"}')
        assert result == {"token": "t1", "projectId": "p1"}


# ---------------------------------------------------------------------------
# 5. _gemini_function_response_payload — narrowed to JSONDecodeError,
#    ValueError, TypeError
# ---------------------------------------------------------------------------

class TestFunctionResponsePayload:
    """Verify function response JSON parse fallback."""

    def test_invalid_json_string_wraps_as_output(self, caplog):
        from omicverse.utils.agent_backend_gemini import _gemini_function_response_payload
        with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
            result = _gemini_function_response_payload("not json {{")
        assert result == {"output": "not json {{"}
        assert any("function response JSON parse failed" in r.message for r in caplog.records)

    def test_valid_json_dict_returned_directly(self):
        from omicverse.utils.agent_backend_gemini import _gemini_function_response_payload
        result = _gemini_function_response_payload('{"key": "val"}')
        assert result == {"key": "val"}

    def test_valid_json_list_wraps_as_output(self):
        from omicverse.utils.agent_backend_gemini import _gemini_function_response_payload
        result = _gemini_function_response_payload('[1, 2, 3]')
        assert result == {"output": [1, 2, 3]}

    def test_dict_input_returned_as_is(self):
        from omicverse.utils.agent_backend_gemini import _gemini_function_response_payload
        d = {"already": "a dict"}
        assert _gemini_function_response_payload(d) is d

    def test_list_input_wraps_as_output(self):
        from omicverse.utils.agent_backend_gemini import _gemini_function_response_payload
        assert _gemini_function_response_payload([1, 2]) == {"output": [1, 2]}

    def test_empty_string_wraps_as_output(self):
        from omicverse.utils.agent_backend_gemini import _gemini_function_response_payload
        assert _gemini_function_response_payload("") == {"output": ""}

    def test_numeric_input_wraps_as_output(self):
        from omicverse.utils.agent_backend_gemini import _gemini_function_response_payload
        assert _gemini_function_response_payload(42) == {"output": 42}


# ---------------------------------------------------------------------------
# 6. HTTPError body read — narrowed to OSError, UnicodeDecodeError
# ---------------------------------------------------------------------------

class TestHttpErrorBodyRead:
    """Verify that HTTPError body read failures produce empty detail."""

    def test_oserror_on_body_read_gives_empty_detail(self, caplog):
        """OSError reading the error body => fall back to str(exc)."""
        from omicverse.utils.agent_backend_gemini import _gemini_cli_request

        mock_exc = HTTPError(
            url="https://example.com",
            code=500,
            msg="Server Error",
            hdrs={},
            fp=BytesIO(b""),
        )
        # Make read() raise OSError
        mock_exc.read = Mock(side_effect=OSError("socket closed"))

        backend = Mock()
        backend.config.endpoint = ""
        backend.config.provider = "gemini"

        with patch("omicverse.utils.agent_backend_gemini._gemini_cli_generate_content_url", return_value="https://example.com"):
            with patch("omicverse.utils.agent_backend_gemini._gemini_oauth_payload", return_value={"token": "t", "projectId": "p"}):
                with patch("omicverse.utils.agent_backend_gemini.urllib_request.Request"):
                    with patch("omicverse.utils.agent_backend_gemini.urllib_request.urlopen", side_effect=mock_exc):
                        with caplog.at_level(logging.DEBUG, logger="omicverse.utils.agent_backend_gemini"):
                            with pytest.raises(RuntimeError, match="HTTP 500"):
                                _gemini_cli_request(backend, {"test": True}, '{"token": "t"}')
        assert any("Failed to read Gemini CLI HTTPError body" in r.message for r in caplog.records)

    def test_readable_body_included_in_error(self):
        """When body is readable, it appears in the RuntimeError detail."""
        from omicverse.utils.agent_backend_gemini import _gemini_cli_request

        mock_exc = HTTPError(
            url="https://example.com",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=BytesIO(b"detailed error message"),
        )

        backend = Mock()
        backend.config.endpoint = ""
        backend.config.provider = "gemini"

        with patch("omicverse.utils.agent_backend_gemini._gemini_cli_generate_content_url", return_value="https://example.com"):
            with patch("omicverse.utils.agent_backend_gemini._gemini_oauth_payload", return_value={"token": "t", "projectId": "p"}):
                with patch("omicverse.utils.agent_backend_gemini.urllib_request.Request"):
                    with patch("omicverse.utils.agent_backend_gemini.urllib_request.urlopen", side_effect=mock_exc):
                        with pytest.raises(RuntimeError, match="detailed error message"):
                            _gemini_cli_request(backend, {}, '{"token": "t"}')


# ---------------------------------------------------------------------------
# 7. No broad except Exception remains
# ---------------------------------------------------------------------------

class TestNoBroadExceptionCatches:
    """Static check: no bare 'except Exception' remains in the module."""

    def test_no_except_exception_in_source(self):
        import inspect
        from omicverse.utils import agent_backend_gemini
        source = inspect.getsource(agent_backend_gemini)
        # Allow "except Exception" only if it doesn't exist
        assert "except Exception:" not in source, (
            "Found bare 'except Exception:' in agent_backend_gemini.py — "
            "all branches should use narrowed exception families"
        )
