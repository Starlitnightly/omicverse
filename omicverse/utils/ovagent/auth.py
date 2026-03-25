"""Credential resolution, model normalization, and API key management.

Extracted from ``OmicVerseAgent.__init__`` so that auth and backend
selection logic lives in one focused module.  All functions are
side-effect-free except for ``temporary_api_keys`` which temporarily
mutates ``os.environ``.

Status messages are emitted through an optional *event_bus* parameter
instead of raw ``print()`` calls, enabling structured observability.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

from ..model_config import ModelConfig, PROVIDER_API_KEYS

from .event_stream import EventBus, make_event_bus

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
    event_bus: "Optional[EventBus]" = None,
) -> ResolvedBackend:
    """Normalize *model* ID, validate setup, and resolve endpoint/provider.

    When a custom *endpoint* (proxy) is provided the model name is kept
    as-is since proxies expect the exact model name the user typed.

    Raises ``ValueError`` when validation fails (no proxy, invalid model/key).
    """
    eb = event_bus if event_bus is not None else make_event_bus()
    _emit = eb.init
    _emit_warn = eb.init_warning
    _emit_err = eb.init_error

    if endpoint:
        _emit(f"   🔌 Proxy mode: model={model}, endpoint={endpoint}")
    else:
        original_model = model
        try:
            model = ModelConfig.normalize_model_id(model)
        except Exception:
            pass  # Older ModelConfig without normalization: proceed as-is
        if model != original_model:
            _emit(f"   📝 Model ID normalized: {original_model} → {model}")

        is_valid, validation_msg = ModelConfig.validate_model_setup(model, api_key)
        if not is_valid:
            _emit_err(f"❌ {validation_msg}")
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

    env_mapping: Dict[str, str] = {}
    required_key = PROVIDER_API_KEYS.get(model)
    if required_key:
        env_mapping[required_key] = api_key

    provider = ModelConfig.get_provider_from_model(model, endpoint)
    if provider == "openai":
        env_mapping.setdefault("OPENAI_API_KEY", api_key)

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
    event_bus: "Optional[EventBus]" = None,
) -> None:
    """Emit model / provider / key-status banner during init."""
    eb = event_bus if event_bus is not None else make_event_bus()
    _emit = eb.init
    _emit_warn = eb.init_warning

    model_desc = ModelConfig.get_model_description(model)
    _emit(f"    Model: {model_desc}")
    _emit(f"    Provider: {provider.title()}")
    _emit(f"    Endpoint: {endpoint}")

    with temporary_api_keys(managed_env):
        key_available, key_msg = ModelConfig.check_api_key_availability(model)
    if key_available:
        _emit(f"   ✅ {key_msg}")
    else:
        _emit_warn(f"   ⚠️  {key_msg}")
