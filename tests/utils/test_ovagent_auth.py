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
    resolve_credentials,
    _resolve_gemini_cli_oauth,
    _resolve_openai_oauth,
    _resolve_saved_openai_key,
    _normalize_auth_mode,
    _is_openai_oauth_supported_model,
    _is_explicit_openai_oauth_model,
    _normalize_model_for_routing,
    _is_custom_openai_endpoint,
    _looks_like_openai_endpoint,
    _extract_openai_codex_account_id,
    _resolve_saved_provider_api_key,
    OPENAI_CODEX_DEFAULT_MODEL,
    _LEGACY_OPENAI_CODEX_BASE_URL,
)
from unittest.mock import patch, MagicMock

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

    def test_non_openai_model_uses_model_specific_env_key(self):
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

    def test_cleanup_on_exception_during_yield(self, monkeypatch):
        """Keys are cleaned up even when the with-block raises."""
        monkeypatch.delenv("__OV_TEST_KEY__", raising=False)
        mapping = {"__OV_TEST_KEY__": "secret-value"}

        with pytest.raises(RuntimeError):
            with temporary_api_keys(mapping):
                assert os.environ["__OV_TEST_KEY__"] == "secret-value"
                raise RuntimeError("inner error")

        assert "__OV_TEST_KEY__" not in os.environ

    def test_exposure_window_bounded_by_context(self, monkeypatch):
        """Keys are only visible inside the with-block, not after."""
        monkeypatch.delenv("__OV_TEST_KEY__", raising=False)
        mapping = {"__OV_TEST_KEY__": "visible-during-context"}

        visible_inside = None
        with temporary_api_keys(mapping):
            visible_inside = os.environ.get("__OV_TEST_KEY__")
        visible_after = os.environ.get("__OV_TEST_KEY__")

        assert visible_inside == "visible-during-context"
        assert visible_after is None

    def test_docstring_documents_exposure_tradeoff(self):
        """The docstring explicitly documents the environment-exposure tradeoff."""
        doc = temporary_api_keys.__doc__ or ""
        assert "exposure" in doc.lower()
        assert "tradeoff" in doc.lower() or "trade-off" in doc.lower()
        assert "finally" in doc.lower()


# ===================================================================
# Moved helper functions
# ===================================================================

class TestNormalizeAuthMode:

    def test_environment_default(self):
        assert _normalize_auth_mode(None) == "environment"
        assert _normalize_auth_mode("environment") == "environment"

    def test_openai_api_key_alias(self):
        assert _normalize_auth_mode("openai_api_key") == "saved_api_key"

    def test_openai_codex_alias(self):
        assert _normalize_auth_mode("openai_codex") == "openai_oauth"

    def test_google_oauth_passthrough(self):
        assert _normalize_auth_mode("google_oauth") == "google_oauth"
        assert _normalize_auth_mode("gemini_cli_oauth") == "gemini_cli_oauth"


class TestOpenAIOAuthModelHelpers:

    def test_supported_model_recognized(self):
        assert _is_openai_oauth_supported_model("gpt-5.3-codex") is True
        assert _is_openai_oauth_supported_model("gpt-5.2") is True

    def test_unsupported_model_rejected(self):
        assert _is_openai_oauth_supported_model("claude-3-opus") is False

    def test_explicit_oauth_model_codex_only(self):
        assert _is_explicit_openai_oauth_model("gpt-5.3-codex") is True
        assert _is_explicit_openai_oauth_model("gpt-5.2") is False

    def test_case_insensitive(self):
        assert _is_openai_oauth_supported_model("GPT-5.3-CODEX") is True
        assert _is_explicit_openai_oauth_model("GPT-5.3-CODEX") is True


class TestEndpointHelpers:

    def test_custom_endpoint_detected(self):
        assert _is_custom_openai_endpoint("https://my-proxy.com/v1") is True

    def test_standard_endpoints_not_custom(self):
        assert _is_custom_openai_endpoint(_LEGACY_OPENAI_CODEX_BASE_URL) is False
        assert _is_custom_openai_endpoint(None) is False

    def test_looks_like_openai_endpoint(self):
        assert _looks_like_openai_endpoint("https://api.openai.com/v1") is True
        assert _looks_like_openai_endpoint(None) is False
        assert _looks_like_openai_endpoint("https://my-proxy.com/v1") is False


# ===================================================================
# _resolve_gemini_cli_oauth
# ===================================================================

class TestResolveGeminiCliOAuth:

    @patch("omicverse.utils.ovagent.auth.GeminiCliOAuthManager")
    def test_success_returns_credential_tuple(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.build_api_key_payload.return_value = "gemini-token-123"
        mock_mgr_cls.return_value = mock_mgr

        model, api_key, endpoint, auth_mode = _resolve_gemini_cli_oauth(
            "gemini-2.5-pro", None, None
        )
        assert model == "gemini-2.5-pro"
        assert api_key == "gemini-token-123"
        assert auth_mode == "gemini_cli_oauth"
        # Endpoint should be set to the Google Code Assist endpoint
        assert endpoint is not None

    @patch("omicverse.utils.ovagent.auth.GeminiCliOAuthManager")
    def test_empty_payload_raises(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.build_api_key_payload.return_value = None
        mock_mgr_cls.return_value = mock_mgr

        with pytest.raises(ValueError, match="No saved Gemini CLI OAuth"):
            _resolve_gemini_cli_oauth("gemini-2.5-pro", None, None)

    @patch("omicverse.utils.ovagent.auth.GeminiCliOAuthManager")
    def test_oauth_error_raises_valueerror(self, mock_mgr_cls):
        from omicverse.jarvis.gemini_cli_oauth import GeminiCliOAuthError

        mock_mgr = MagicMock()
        mock_mgr.build_api_key_payload.side_effect = GeminiCliOAuthError("fail")
        mock_mgr_cls.return_value = mock_mgr

        with pytest.raises(ValueError, match="Failed to load saved Gemini CLI OAuth"):
            _resolve_gemini_cli_oauth("gemini-2.5-pro", None, None)

    @patch("omicverse.utils.ovagent.auth.GeminiCliOAuthManager")
    def test_preserves_custom_endpoint(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.build_api_key_payload.return_value = "token"
        mock_mgr_cls.return_value = mock_mgr

        _, _, endpoint, _ = _resolve_gemini_cli_oauth(
            "gemini-2.5-pro", "https://custom.google.endpoint/v1", None
        )
        assert endpoint == "https://custom.google.endpoint/v1"


# ===================================================================
# _resolve_openai_oauth
# ===================================================================

class TestResolveOpenAIOAuth:

    @patch("omicverse.utils.ovagent.auth._extract_openai_codex_account_id")
    def test_valid_token_returns_immediately(self, mock_extract):
        mock_extract.return_value = "acct-123"
        model, api_key, endpoint, auth_mode = _resolve_openai_oauth(
            "gpt-5.3-codex",
            "valid-token",
            None,
            "gpt-5.3-codex",
            "explicit",
            None,
        )
        assert api_key == "valid-token"
        assert auth_mode == "openai_oauth"

    @patch("omicverse.utils.ovagent.auth._extract_openai_codex_account_id")
    def test_explicit_key_without_account_id_raises(self, mock_extract):
        mock_extract.return_value = ""
        with pytest.raises(ValueError, match="chatgpt_account_id"):
            _resolve_openai_oauth(
                "gpt-5.3-codex",
                "bad-token",
                None,
                "gpt-5.3-codex",
                "explicit",
                None,
            )

    @patch("omicverse.utils.ovagent.auth.OpenAIOAuthManager")
    @patch("omicverse.utils.ovagent.auth._extract_openai_codex_account_id")
    def test_falls_back_to_oauth_manager(self, mock_extract, mock_mgr_cls):
        # Only one call to _extract happens: after OAuth manager returns a token
        mock_extract.return_value = "acct-456"
        mock_mgr = MagicMock()
        mock_mgr.ensure_access_token_with_codex_fallback.return_value = "oauth-token"
        mock_mgr_cls.return_value = mock_mgr

        model, api_key, endpoint, auth_mode = _resolve_openai_oauth(
            "gpt-5.3-codex",
            None,
            None,
            "gpt-5.3-codex",
            None,
            None,
        )
        assert api_key == "oauth-token"
        assert auth_mode == "openai_oauth"

    @patch("omicverse.utils.ovagent.auth.OpenAIOAuthManager")
    @patch("omicverse.utils.ovagent.auth._extract_openai_codex_account_id")
    def test_no_saved_login_raises(self, mock_extract, mock_mgr_cls):
        mock_extract.return_value = ""
        mock_mgr = MagicMock()
        mock_mgr.ensure_access_token_with_codex_fallback.return_value = None
        mock_mgr_cls.return_value = mock_mgr

        with pytest.raises(ValueError, match="No saved OpenAI Codex login"):
            _resolve_openai_oauth(
                "gpt-5.3-codex", None, None, "gpt-5.3-codex", None, None,
            )

    @patch("omicverse.utils.ovagent.auth.OpenAIOAuthManager")
    @patch("omicverse.utils.ovagent.auth._extract_openai_codex_account_id")
    def test_unsupported_model_gets_default(self, mock_extract, mock_mgr_cls):
        mock_extract.return_value = "acct-789"
        mock_mgr = MagicMock()
        mock_mgr.ensure_access_token_with_codex_fallback.return_value = "token"
        mock_mgr_cls.return_value = mock_mgr

        model, _, _, _ = _resolve_openai_oauth(
            "unknown-model", None, None, "unknown-model", None, None,
        )
        assert model == OPENAI_CODEX_DEFAULT_MODEL


# ===================================================================
# _resolve_saved_openai_key
# ===================================================================

class TestResolveSavedOpenaiKey:

    @patch("omicverse.utils.ovagent.auth.load_auth")
    def test_returns_key_from_auth_file(self, mock_load):
        mock_load.return_value = {"OPENAI_API_KEY": "sk-saved-key"}
        result = _resolve_saved_openai_key(None)
        assert result == "sk-saved-key"

    @patch("omicverse.utils.ovagent.auth.load_auth")
    def test_returns_none_when_missing(self, mock_load):
        mock_load.return_value = {}
        result = _resolve_saved_openai_key(None)
        assert result is None


# ===================================================================
# resolve_credentials (main dispatcher)
# ===================================================================

class TestResolveCredentials:

    @patch("omicverse.utils.ovagent.auth.ModelConfig")
    def test_explicit_key_passthrough(self, mock_mc):
        """Explicit API key + environment mode returns a passthrough tuple."""
        mock_mc.get_provider_from_model.return_value = "openai"
        mock_mc.normalize_model_id.return_value = "gpt-5.2"

        result = resolve_credentials(
            model="gpt-5.2",
            api_key="sk-explicit",
            endpoint=None,
            auth_mode="environment",
            auth_provider=None,
            auth_file=None,
        )
        model, api_key, endpoint, auth_mode = result
        assert model == "gpt-5.2"
        assert api_key == "sk-explicit"
        assert auth_mode == "environment"

    @patch("omicverse.utils.ovagent.auth._resolve_gemini_cli_oauth")
    @patch("omicverse.utils.ovagent.auth.ModelConfig")
    def test_gemini_oauth_dispatches(self, mock_mc, mock_gemini):
        """google_oauth mode with Google provider dispatches to Gemini resolver."""
        mock_mc.get_provider_from_model.return_value = "google"
        mock_mc.normalize_model_id.return_value = "gemini-2.5-pro"
        mock_gemini.return_value = ("gemini-2.5-pro", "token", "endpoint", "gemini_cli_oauth")

        result = resolve_credentials(
            model="gemini-2.5-pro",
            api_key=None,
            endpoint=None,
            auth_mode="google_oauth",
            auth_provider=None,
            auth_file=None,
        )
        assert result == ("gemini-2.5-pro", "token", "endpoint", "gemini_cli_oauth")
        mock_gemini.assert_called_once()

    @patch("omicverse.utils.ovagent.auth._resolve_openai_oauth")
    @patch("omicverse.utils.ovagent.auth.ModelConfig")
    def test_openai_oauth_dispatches(self, mock_mc, mock_oai):
        """openai_oauth mode with OpenAI provider dispatches to OpenAI resolver."""
        mock_mc.get_provider_from_model.return_value = "openai"
        mock_mc.normalize_model_id.return_value = "gpt-5.3-codex"
        mock_oai.return_value = ("gpt-5.3-codex", "oauth-tok", "ep", "openai_oauth")

        result = resolve_credentials(
            model="gpt-5.3-codex",
            api_key=None,
            endpoint=None,
            auth_mode="openai_oauth",
            auth_provider=None,
            auth_file=None,
        )
        assert result == ("gpt-5.3-codex", "oauth-tok", "ep", "openai_oauth")
        mock_oai.assert_called_once()

    @patch("omicverse.utils.ovagent.auth._resolve_saved_openai_key")
    @patch("omicverse.utils.ovagent.auth.ModelConfig")
    def test_saved_api_key_dispatches(self, mock_mc, mock_saved):
        """saved_api_key mode looks up the saved key then returns passthrough."""
        mock_mc.get_provider_from_model.return_value = "openai"
        mock_mc.normalize_model_id.return_value = "gpt-5.2"
        mock_saved.return_value = "sk-from-file"

        model, api_key, endpoint, auth_mode = resolve_credentials(
            model="gpt-5.2",
            api_key=None,
            endpoint=None,
            auth_mode="saved_api_key",
            auth_provider=None,
            auth_file=None,
        )
        assert api_key == "sk-from-file"
        assert auth_mode == "saved_api_key"
        mock_saved.assert_called_once()

    @patch("omicverse.utils.ovagent.auth.ModelConfig")
    def test_returns_four_element_tuple(self, mock_mc):
        """Return value is always a 4-element tuple (the contract)."""
        mock_mc.get_provider_from_model.return_value = "anthropic"
        mock_mc.normalize_model_id.return_value = "claude-opus-4-6"

        result = resolve_credentials(
            model="claude-opus-4-6",
            api_key="key",
            endpoint=None,
            auth_mode=None,
            auth_provider=None,
            auth_file=None,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4


# ===================================================================
# Agent() thin wrapper
# ===================================================================

class TestAgentFactory:

    def test_agent_is_callable(self):
        """Agent is importable and callable."""
        Agent = smart_agent_module.Agent
        assert callable(Agent)

    def test_agent_returns_omicverse_agent_type(self):
        """Agent() must return an OmicVerseAgent instance (when construction
        succeeds). We verify the function signature rather than constructing,
        since full construction requires API credentials."""
        import inspect
        Agent = smart_agent_module.Agent
        sig = inspect.signature(Agent)
        params = list(sig.parameters.keys())
        # After task-033, Agent() mirrors the explicit OmicVerseAgent.__init__ signature
        assert "model" in params
