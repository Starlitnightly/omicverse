import importlib.machinery
import importlib.util
import json
import sys
import types
from base64 import urlsafe_b64encode
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"


def _load_smart_agent_module():
    original_modules = {
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

    spec = importlib.util.spec_from_file_location(
        "omicverse.utils.smart_agent",
        PACKAGE_ROOT / "utils" / "smart_agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["omicverse.utils.smart_agent"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    for name, existing in original_modules.items():
        if existing is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = existing

    return module


smart_agent = _load_smart_agent_module()


def _make_oauth_token(account_id: str = "acct_test") -> str:
    def _encode(payload):
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    header = _encode({"alg": "none", "typ": "JWT"})
    payload = _encode(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": account_id,
            }
        }
    )
    return f"{header}.{payload}.signature"


def test_resolve_agent_llm_credentials_uses_saved_codex_auth(monkeypatch):
    class DummyManager:
        def __init__(self, auth_path=None):
            self.auth_path = auth_path

        def ensure_access_token_with_codex_fallback(self, **kwargs):
            return _make_oauth_token()

    monkeypatch.setattr(smart_agent, "OpenAIOAuthManager", DummyManager)

    model, api_key, endpoint, auth_mode = smart_agent._resolve_agent_llm_credentials(
        model="gpt-5.2",
        api_key=None,
        endpoint=None,
        auth_mode="openai_oauth",
        auth_provider=None,
        auth_file=None,
    )

    assert model == "gpt-5.2"
    assert api_key == _make_oauth_token()
    assert endpoint == smart_agent.OPENAI_CODEX_BASE_URL
    assert auth_mode == "openai_oauth"


def test_resolve_agent_llm_credentials_auto_enables_codex_oauth(monkeypatch):
    class DummyManager:
        def __init__(self, auth_path=None):
            self.auth_path = auth_path

        def ensure_access_token_with_codex_fallback(self, **kwargs):
            return _make_oauth_token("acct_alias")

    monkeypatch.setattr(smart_agent, "OpenAIOAuthManager", DummyManager)

    model, api_key, endpoint, auth_mode = smart_agent._resolve_agent_llm_credentials(
        model="gpt-5.3-codex",
        api_key=None,
        endpoint=None,
        auth_mode="environment",
        auth_provider=None,
        auth_file=None,
    )

    assert model == "gpt-5.3-codex"
    assert api_key == _make_oauth_token("acct_alias")
    assert endpoint == smart_agent.OPENAI_CODEX_BASE_URL
    assert auth_mode == "openai_oauth"


def test_resolve_agent_llm_credentials_auto_enables_codex_oauth_for_alias(monkeypatch):
    class DummyManager:
        def __init__(self, auth_path=None):
            self.auth_path = auth_path

        def ensure_access_token_with_codex_fallback(self, **kwargs):
            return _make_oauth_token("acct_alias_auto")

    monkeypatch.setattr(smart_agent, "OpenAIOAuthManager", DummyManager)

    model, api_key, endpoint, auth_mode = smart_agent._resolve_agent_llm_credentials(
        model="openai/gpt-5.3-codex",
        api_key=None,
        endpoint=None,
        auth_mode="environment",
        auth_provider=None,
        auth_file=None,
    )

    assert model == "gpt-5.3-codex"
    assert api_key == _make_oauth_token("acct_alias_auto")
    assert endpoint == smart_agent.OPENAI_CODEX_BASE_URL
    assert auth_mode == "openai_oauth"


def test_resolve_agent_llm_credentials_keeps_standard_openai_path(monkeypatch):
    class DummyManager:
        def __init__(self, auth_path=None):
            raise AssertionError("Codex OAuth manager should not be used for standard OpenAI auth")

    monkeypatch.setattr(smart_agent, "OpenAIOAuthManager", DummyManager)

    model, api_key, endpoint, auth_mode = smart_agent._resolve_agent_llm_credentials(
        model="gpt-5.2",
        api_key="api-key",
        endpoint=None,
        auth_mode="environment",
        auth_provider=None,
        auth_file=None,
    )

    assert model == "gpt-5.2"
    assert api_key == "api-key"
    assert endpoint is None
    assert auth_mode == "environment"


def test_resolve_agent_llm_credentials_keeps_supported_oauth_chat_models(monkeypatch):
    class DummyManager:
        def __init__(self, auth_path=None):
            self.auth_path = auth_path

        def ensure_access_token_with_codex_fallback(self, **kwargs):
            return _make_oauth_token("acct_chat")

    monkeypatch.setattr(smart_agent, "OpenAIOAuthManager", DummyManager)

    model, api_key, endpoint, auth_mode = smart_agent._resolve_agent_llm_credentials(
        model="gpt-5.4",
        api_key=None,
        endpoint=None,
        auth_mode="openai_oauth",
        auth_provider=None,
        auth_file=None,
    )

    assert model == "gpt-5.4"
    assert api_key == _make_oauth_token("acct_chat")
    assert endpoint == smart_agent.OPENAI_CODEX_BASE_URL
    assert auth_mode == "openai_oauth"


def test_resolve_agent_llm_credentials_normalizes_supported_oauth_alias(monkeypatch):
    class DummyManager:
        def __init__(self, auth_path=None):
            self.auth_path = auth_path

        def ensure_access_token_with_codex_fallback(self, **kwargs):
            return _make_oauth_token("acct_spark")

    monkeypatch.setattr(smart_agent, "OpenAIOAuthManager", DummyManager)

    model, api_key, endpoint, auth_mode = smart_agent._resolve_agent_llm_credentials(
        model="openai/gpt-5.3-codex-spark",
        api_key=None,
        endpoint=None,
        auth_mode="openai_oauth",
        auth_provider=None,
        auth_file=None,
    )

    assert model == "gpt-5.3-codex-spark"
    assert api_key == _make_oauth_token("acct_spark")
    assert endpoint == smart_agent.OPENAI_CODEX_BASE_URL
    assert auth_mode == "openai_oauth"


def test_resolve_agent_llm_credentials_rejects_standard_api_key_for_codex():
    with pytest.raises(ValueError, match="valid OpenAI OAuth access token"):
        smart_agent._resolve_agent_llm_credentials(
            model="gpt-5.3-codex",
            api_key="sk-test",
            endpoint=None,
            auth_mode="openai_oauth",
            auth_provider=None,
            auth_file=None,
        )


def test_resolve_agent_llm_credentials_falls_back_when_saved_api_key_is_not_oauth(monkeypatch):
    class DummyManager:
        def __init__(self, auth_path=None):
            self.auth_path = auth_path

        def ensure_access_token_with_codex_fallback(self, **kwargs):
            return _make_oauth_token("acct_fallback")

    monkeypatch.setattr(
        smart_agent,
        "_resolve_saved_provider_api_key",
        lambda provider_name, auth_path: "sk-test",
    )
    monkeypatch.setattr(smart_agent, "OpenAIOAuthManager", DummyManager)

    model, api_key, endpoint, auth_mode = smart_agent._resolve_agent_llm_credentials(
        model="gpt-5.3-codex",
        api_key=None,
        endpoint=None,
        auth_mode="saved_api_key",
        auth_provider=None,
        auth_file=None,
    )

    assert model == "gpt-5.3-codex"
    assert api_key == _make_oauth_token("acct_fallback")
    assert endpoint == smart_agent.OPENAI_CODEX_BASE_URL
    assert auth_mode == "openai_oauth"


def test_resolve_agent_llm_credentials_errors_without_saved_login(monkeypatch):
    class DummyManager:
        def __init__(self, auth_path=None):
            self.auth_path = auth_path

        def ensure_access_token_with_codex_fallback(self, **kwargs):
            return None

    monkeypatch.setattr(smart_agent, "OpenAIOAuthManager", DummyManager)

    with pytest.raises(ValueError, match="No saved OpenAI Codex login found"):
        smart_agent._resolve_agent_llm_credentials(
            model="gpt-5.3-codex",
            api_key=None,
            endpoint=None,
            auth_mode="openai_oauth",
            auth_provider=None,
            auth_file=None,
        )
