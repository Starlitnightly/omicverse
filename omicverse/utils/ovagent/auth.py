"""Credential resolution, model normalization, and API key management.

Extracted from ``OmicVerseAgent.__init__`` so that auth and backend
selection logic lives in one focused module.  All functions are
side-effect-free except for ``temporary_api_keys`` which temporarily
mutates ``os.environ``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

from ..model_config import ModelConfig, PROVIDER_API_KEYS

logger = logging.getLogger(__name__)


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
