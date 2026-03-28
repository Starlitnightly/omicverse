"""Credential resolution, model normalization, and API key management.

Extracted from ``OmicVerseAgent.__init__`` so that auth and backend
selection logic lives in one focused module.  All functions are
side-effect-free except for ``temporary_api_keys`` which temporarily
mutates ``os.environ``.

The main entry point is :func:`resolve_credentials`, which dispatches to
named resolver helpers for each auth flow:

- :func:`_resolve_gemini_cli_oauth` — Gemini CLI OAuth token flow
- :func:`_resolve_openai_oauth` — OpenAI Codex / ChatGPT OAuth flow
- :func:`_resolve_saved_openai_key` — saved ``api_key`` from auth file
- passthrough — explicit key or environment-based resolution
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from ..model_config import ModelConfig, PROVIDER_API_KEYS
from ...jarvis.config import load_auth
from ...jarvis.gemini_cli_oauth import (
    GOOGLE_CODE_ASSIST_ENDPOINT_PROD,
    GeminiCliOAuthError,
    GeminiCliOAuthManager,
)
from ...jarvis.openai_oauth import OPENAI_CODEX_BASE_URL, OpenAIOAuthManager
from ..agent_backend_openai import (
    _extract_openai_codex_account_id as _backend_extract_codex_account_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (moved from smart_agent.py)
# ---------------------------------------------------------------------------

_LEGACY_OPENAI_CODEX_BASE_URL = "https://api.openai.com/v1"
OPENAI_CODEX_DEFAULT_MODEL = "gpt-5.3-codex"
_OPENAI_OAUTH_SUPPORTED_MODELS = {
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2-codex",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.4",
    "gpt-5.4-mini",
}
_OPENAI_EXPLICIT_OAUTH_MODELS = {
    model_name
    for model_name in _OPENAI_OAUTH_SUPPORTED_MODELS
    if "codex" in model_name
}


# ---------------------------------------------------------------------------
# Small helpers (moved from smart_agent.py)
# ---------------------------------------------------------------------------

def _normalize_auth_mode(auth_mode: Optional[str]) -> str:
    if auth_mode == "openai_api_key":
        return "saved_api_key"
    if auth_mode == "openai_codex":
        return "openai_oauth"
    if auth_mode in {"google_oauth", "gemini_cli_oauth"}:
        return str(auth_mode)
    return str(auth_mode or "environment")


def _is_openai_oauth_supported_model(model: str) -> bool:
    return str(model or "").strip().lower() in {
        name.lower() for name in _OPENAI_OAUTH_SUPPORTED_MODELS
    }


def _is_explicit_openai_oauth_model(model: str) -> bool:
    return str(model or "").strip().lower() in {
        name.lower() for name in _OPENAI_EXPLICIT_OAUTH_MODELS
    }


def _normalize_model_for_routing(model: str) -> str:
    normalized = str(model or "").strip()
    if not normalized:
        return normalized
    try:
        return ModelConfig.normalize_model_id(normalized)
    except (AttributeError, KeyError, ValueError, TypeError) as exc:
        logger.debug("_normalize_model_for_routing fallback for %r: %s", normalized, exc)
        return normalized


def _is_custom_openai_endpoint(endpoint: Optional[str]) -> bool:
    normalized = str(endpoint or "").strip().rstrip("/")
    return bool(normalized) and normalized not in {
        _LEGACY_OPENAI_CODEX_BASE_URL,
        OPENAI_CODEX_BASE_URL,
    }


def _looks_like_openai_endpoint(endpoint: Optional[str]) -> bool:
    normalized = str(endpoint or "").strip().rstrip("/").lower()
    if not normalized:
        return False
    return (
        normalized == _LEGACY_OPENAI_CODEX_BASE_URL
        or normalized == OPENAI_CODEX_BASE_URL
        or "api.openai.com" in normalized
        or "chatgpt.com/backend-api" in normalized
    )


def _extract_openai_codex_account_id(token: Optional[str]) -> str:
    try:
        return _backend_extract_codex_account_id(str(token or ""))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        logger.debug("_extract_openai_codex_account_id fallback: %s", exc)
        return ""


def _resolve_saved_provider_api_key(
    provider_name: str, auth_path: Optional[Path]
) -> Optional[str]:
    auth = load_auth(auth_path)
    providers = dict(auth.get("providers") or {})

    if provider_name == "openai":
        top_level = str(auth.get("OPENAI_API_KEY") or "").strip()
        if top_level:
            return top_level

    provider_auth = dict(providers.get(provider_name) or {})
    api_key = str(provider_auth.get("api_key") or "").strip()
    return api_key or None


@dataclass
class ResolvedBackend:
    """Result of model normalization and provider resolution."""

    model: str
    endpoint: str
    provider: str


def resolve_model_and_provider(
    model: str,
    api_key: Optional[str],
    endpoint: Optional[str],
) -> ResolvedBackend:
    """Normalize *model* ID, validate setup, and resolve endpoint/provider.

    When a custom *endpoint* (proxy) is provided the model name is kept
    as-is since proxies expect the exact model name the user typed.

    Raises ``ValueError`` when validation fails (no proxy, invalid model/key).
    """
    if endpoint:
        logger.info("Proxy mode enabled for model %s via endpoint %s", model, endpoint)
        print(f"   🔌 Proxy mode: model={model}, endpoint={endpoint}")
    else:
        original_model = model
        try:
            model = ModelConfig.normalize_model_id(model)
        except Exception:
            logger.warning(
                "Model ID normalization failed for %s; proceeding with the original ID",
                model,
                exc_info=True,
            )
        if model != original_model:
            logger.info("Model ID normalized from %s to %s", original_model, model)
            print(f"   📝 Model ID normalized: {original_model} → {model}")

        is_valid, validation_msg = ModelConfig.validate_model_setup(model, api_key)
        if not is_valid:
            logger.error("Model setup validation failed for %s: %s", model, validation_msg)
            print(f"❌ {validation_msg}")
            raise ValueError(validation_msg)

    resolved_endpoint = endpoint or ModelConfig.get_endpoint_for_model(model)
    provider = ModelConfig.get_provider_from_model(model, resolved_endpoint)
    return ResolvedBackend(
        model=model,
        endpoint=resolved_endpoint,
        provider=provider,
    )


def collect_api_key_env(
    model: str,
    endpoint: Optional[str],
    api_key: Optional[str],
) -> Dict[str, str]:
    """Build dict of environment variables required for API authentication."""
    if not api_key:
        return {}

    try:
        normalized_model = ModelConfig.normalize_model_id(model)
    except Exception:
        normalized_model = model

    env_mapping: Dict[str, str] = {}
    required_key = PROVIDER_API_KEYS.get(normalized_model)
    if required_key:
        env_mapping[required_key] = api_key

    provider = ModelConfig.get_provider_from_model(normalized_model, endpoint)
    if provider == "openai":
        env_mapping.setdefault("OPENAI_API_KEY", api_key)
    elif provider == "google":
        env_mapping.setdefault("GOOGLE_API_KEY", api_key)

    return env_mapping


@contextmanager
def temporary_api_keys(env_mapping: Dict[str, str]):
    """Temporarily inject API keys into ``os.environ`` and clean up afterwards."""
    if not env_mapping:
        yield
        return

    previous_values: Dict[str, Optional[str]] = {}
    try:
        for key, value in env_mapping.items():
            previous_values[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, previous in previous_values.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def display_backend_info(
    model: str,
    endpoint: str,
    provider: str,
    api_key: Optional[str],
    managed_env: Dict[str, str],
) -> None:
    """Print model / provider / key-status banner during init."""
    model_desc = ModelConfig.get_model_description(model)
    logger.info(
        "Resolved backend: model=%s provider=%s endpoint=%s managed_env_keys=%s",
        model,
        provider,
        endpoint,
        sorted(managed_env),
    )
    print(f"    Model: {model_desc}")
    print(f"    Provider: {provider.title()}")
    print(f"    Endpoint: {endpoint}")

    with temporary_api_keys(managed_env):
        key_available, key_msg = ModelConfig.check_api_key_availability(model)
    if key_available:
        logger.info("API key availability check passed for %s: %s", model, key_msg)
        print(f"   ✅ {key_msg}")
    else:
        logger.warning("API key availability check did not pass for %s: %s", model, key_msg)
        print(f"   ⚠️  {key_msg}")


# ---------------------------------------------------------------------------
# Named credential resolver paths
# ---------------------------------------------------------------------------

# Type alias for the credential tuple contract used by the bootstrap path.
CredentialTuple = Tuple[str, Optional[str], Optional[str], str]


def _resolve_gemini_cli_oauth(
    resolved_model: str,
    resolved_endpoint: Optional[str],
    auth_path: Optional[Path],
) -> CredentialTuple:
    """Resolve credentials via the Gemini CLI OAuth flow.

    Returns ``(model, api_key, endpoint, "gemini_cli_oauth")``.
    Raises ``ValueError`` on failure.
    """
    manager = GeminiCliOAuthManager(auth_path)
    try:
        payload = manager.build_api_key_payload(
            refresh_if_needed=True,
            import_if_missing=True,
        )
    except GeminiCliOAuthError as exc:
        raise ValueError(
            "Failed to load saved Gemini CLI OAuth credentials. "
            "This is an unofficial integration; some users report account restrictions. "
            "Use at your own risk."
        ) from exc
    if not payload:
        raise ValueError(
            "No saved Gemini CLI OAuth login found. Complete Gemini CLI OAuth first. "
            "This is an unofficial integration; some users report account restrictions. "
            "Use at your own risk."
        )
    if (
        not resolved_endpoint
        or _looks_like_openai_endpoint(resolved_endpoint)
        or "generativelanguage.googleapis.com" in str(resolved_endpoint)
    ):
        resolved_endpoint = GOOGLE_CODE_ASSIST_ENDPOINT_PROD
    return resolved_model, payload, resolved_endpoint, "gemini_cli_oauth"


def _resolve_openai_oauth(
    resolved_model: str,
    resolved_api_key: Optional[str],
    resolved_endpoint: Optional[str],
    normalized_model: str,
    api_key_source: Optional[str],
    auth_path: Optional[Path],
) -> CredentialTuple:
    """Resolve credentials via the OpenAI Codex / ChatGPT OAuth flow.

    Returns ``(model, api_key, endpoint, "openai_oauth")``.
    Raises ``ValueError`` on failure.
    """
    preserve_model_id = _is_custom_openai_endpoint(resolved_endpoint)
    if not resolved_endpoint or resolved_endpoint.rstrip("/") == _LEGACY_OPENAI_CODEX_BASE_URL:
        resolved_endpoint = OPENAI_CODEX_BASE_URL
        preserve_model_id = False
    if _is_openai_oauth_supported_model(normalized_model):
        if not preserve_model_id:
            resolved_model = normalized_model
    elif not preserve_model_id:
        resolved_model = OPENAI_CODEX_DEFAULT_MODEL

    if resolved_api_key:
        if _extract_openai_codex_account_id(resolved_api_key):
            return resolved_model, resolved_api_key, resolved_endpoint, "openai_oauth"
        if api_key_source == "explicit":
            raise ValueError(
                "OpenAI Codex models require a ChatGPT OAuth access token with "
                "chatgpt_account_id. Run `omicverse jarvis --codex-login` or "
                "pass a valid OpenAI OAuth access token."
            )
        resolved_api_key = None

    manager = OpenAIOAuthManager(auth_path)
    resolved_api_key = manager.ensure_access_token_with_codex_fallback(
        refresh_if_needed=True,
        import_codex_if_missing=True,
    )
    if resolved_api_key and _extract_openai_codex_account_id(resolved_api_key):
        return resolved_model, resolved_api_key, resolved_endpoint, "openai_oauth"
    if resolved_api_key:
        raise ValueError(
            "Saved OpenAI Codex login is missing chatgpt_account_id. "
            "Run `omicverse jarvis --codex-login` again."
        )

    raise ValueError(
        "No saved OpenAI Codex login found. Run `omicverse jarvis --codex-login` "
        "or pass a valid OpenAI OAuth access token."
    )


def _resolve_saved_openai_key(
    auth_path: Optional[Path],
) -> Optional[str]:
    """Look up a previously saved OpenAI API key from the auth file."""
    return _resolve_saved_provider_api_key("openai", auth_path)


def resolve_credentials(
    *,
    model: str,
    api_key: Optional[str],
    endpoint: Optional[str],
    auth_mode: Optional[str],
    auth_provider: Optional[str],
    auth_file: Optional[Union[str, Path]],
) -> CredentialTuple:
    """Resolve model/auth settings across all supported credential flows.

    This is the main dispatcher that routes to the appropriate resolver
    based on the detected provider and requested auth mode.

    Returns a ``(model, api_key, endpoint, auth_mode)`` tuple that
    matches the contract expected by the ``OmicVerseAgent`` bootstrap.
    """
    normalized_mode = _normalize_auth_mode(auth_mode)
    resolved_model = model
    normalized_model = _normalize_model_for_routing(model)
    resolved_api_key = api_key
    api_key_source = "explicit" if resolved_api_key else None
    resolved_endpoint = endpoint
    resolved_auth_path = Path(auth_file).expanduser() if auth_file else None
    normalized_auth_provider = str(auth_provider or "").strip().lower() or "codex"

    provider = ModelConfig.get_provider_from_model(
        normalized_model or resolved_model, resolved_endpoint
    )

    # --- Gemini CLI OAuth path ---
    wants_gemini_cli_oauth = provider == "google" and (
        normalized_mode in {"google_oauth", "gemini_cli_oauth"}
        or (normalized_mode == "openai_oauth" and normalized_auth_provider == "gemini_cli")
    )
    if wants_gemini_cli_oauth:
        return _resolve_gemini_cli_oauth(resolved_model, resolved_endpoint, resolved_auth_path)

    # --- OpenAI OAuth detection ---
    wants_codex_oauth = provider == "openai" and (
        normalized_mode == "openai_oauth"
        or _is_explicit_openai_oauth_model(normalized_model)
        or str(resolved_endpoint or "").rstrip("/") == OPENAI_CODEX_BASE_URL
    )

    # --- Saved OpenAI API key path ---
    if provider == "openai" and normalized_mode == "saved_api_key" and not resolved_api_key:
        resolved_api_key = _resolve_saved_openai_key(resolved_auth_path)
        api_key_source = "saved_api_key" if resolved_api_key else None

    # --- Passthrough (explicit key / environment) ---
    if not wants_codex_oauth:
        return resolved_model, resolved_api_key, resolved_endpoint, normalized_mode

    # --- OpenAI Codex OAuth path ---
    return _resolve_openai_oauth(
        resolved_model,
        resolved_api_key,
        resolved_endpoint,
        normalized_model,
        api_key_source,
        resolved_auth_path,
    )
