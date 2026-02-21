"""
Model Configuration and Management for OmicVerse Smart Agent

Provider Registry pattern inspired by OpenAI Codex CLI ModelProviderInfo.
Each provider is registered once with all metadata (base_url, env_key,
wire_api, models).  Legacy dicts (AVAILABLE_MODELS, PROVIDER_API_KEYS, etc.)
are computed from the registry for backward compatibility.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple
import os
import sys


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------

class WireAPI(Enum):
    """LLM communication protocol type."""
    CHAT_COMPLETIONS = "chat"        # OpenAI-compatible /v1/chat/completions
    ANTHROPIC_MESSAGES = "anthropic" # Anthropic Messages API
    GEMINI_GENERATE = "gemini"       # Google Gemini generateContent
    DASHSCOPE = "dashscope"          # Alibaba DashScope
    LOCAL = "local"                  # Local Python execution


@dataclass
class ProviderInfo:
    """Complete configuration for a single LLM provider."""
    name: str                                    # "openai", "anthropic", ...
    display_name: str                            # "OpenAI", "Anthropic", ...
    base_url: str                                # API endpoint
    env_key: str                                 # Primary API key env var name
    wire_api: WireAPI                            # Communication protocol
    alt_env_keys: Tuple[str, ...] = ()           # Alternative env var names
    model_prefixes: Tuple[str, ...] = ()         # Model ID prefixes for routing
    models: Dict[str, str] = field(default_factory=dict)  # model_id -> description


# Global provider registry (single source of truth)
PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}


def register_provider(info: ProviderInfo) -> None:
    """Register a provider.  Can be called at runtime to add custom providers."""
    PROVIDER_REGISTRY[info.name] = info


def get_provider(name: str) -> Optional[ProviderInfo]:
    """Look up a provider by name.  Returns *None* if not found."""
    return PROVIDER_REGISTRY.get(name)


# ---------------------------------------------------------------------------
# Built-in provider registrations
# ---------------------------------------------------------------------------

register_provider(ProviderInfo(
    name="python",
    display_name="Local Python",
    base_url="local-python",
    env_key="",
    wire_api=WireAPI.LOCAL,
    model_prefixes=("python",),
    models={
        "python": "Local Python executor (no LLM)",
    },
))

register_provider(ProviderInfo(
    name="openai",
    display_name="OpenAI",
    base_url="https://api.openai.com/v1",
    env_key="OPENAI_API_KEY",
    wire_api=WireAPI.CHAT_COMPLETIONS,
    model_prefixes=("gpt-", "o1", "o3"),
    models={
        "gpt-5": "OpenAI GPT-5 (Latest)",
        "gpt-5-mini": "OpenAI GPT-5 Mini",
        "gpt-5-nano": "OpenAI GPT-5 Nano",
        "gpt-5-chat-latest": "OpenAI GPT-5 Chat Latest",
        "gpt-4.1": "OpenAI GPT-4.1",
        "gpt-4.1-mini": "OpenAI GPT-4.1 Mini",
        "gpt-4.1-nano": "OpenAI GPT-4.1 Nano",
        "gpt-4o": "OpenAI GPT-4o",
        "gpt-4o-2024-05-13": "OpenAI GPT-4o (2024-05-13)",
        "gpt-4o-mini": "OpenAI GPT-4o Mini",
        "o1": "OpenAI o1 (Reasoning)",
        "o1-pro": "OpenAI o1 Pro (Reasoning)",
        "o3-pro": "OpenAI o3 Pro (Reasoning)",
        "o3": "OpenAI o3 (Reasoning)",
        "o3-mini": "OpenAI o3 Mini (Reasoning)",
        "o1-mini": "OpenAI o1 Mini (Reasoning)",
    },
))

register_provider(ProviderInfo(
    name="anthropic",
    display_name="Anthropic",
    base_url="https://api.anthropic.com",
    env_key="ANTHROPIC_API_KEY",
    wire_api=WireAPI.ANTHROPIC_MESSAGES,
    model_prefixes=("anthropic/",),
    models={
        "anthropic/claude-opus-4-1-20250805": "Claude Opus 4.1 (Latest)",
        "anthropic/claude-opus-4-20250514": "Claude Opus 4",
        "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
        "anthropic/claude-3-7-sonnet-20250219": "Claude Sonnet 3.7",
        "anthropic/claude-3-5-haiku-20241022": "Claude Haiku 3.5",
        "anthropic/claude-3-opus-20240229": "Claude 3 Opus (Legacy)",
        "anthropic/claude-3-sonnet-20240229": "Claude 3 Sonnet (Legacy)",
        "anthropic/claude-3-haiku-20240307": "Claude 3 Haiku (Legacy)",
    },
))

register_provider(ProviderInfo(
    name="google",
    display_name="Google",
    base_url="https://generativelanguage.googleapis.com/v1beta",
    env_key="GOOGLE_API_KEY",
    wire_api=WireAPI.GEMINI_GENERATE,
    model_prefixes=("gemini/",),
    models={
        "gemini/gemini-3-flash-preview": "Gemini 3 Flash Preview",
        "gemini/gemini-2.5-pro": "Gemini 2.5 Pro",
        "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
        "gemini/gemini-2.0-pro": "Gemini 2.0 Pro",
        "gemini/gemini-2.0-flash": "Gemini 2.0 Flash",
        "gemini/gemini-pro": "Gemini Pro",
    },
))

register_provider(ProviderInfo(
    name="deepseek",
    display_name="DeepSeek",
    base_url="https://api.deepseek.com/v1",
    env_key="DEEPSEEK_API_KEY",
    wire_api=WireAPI.CHAT_COMPLETIONS,
    model_prefixes=("deepseek/",),
    models={
        "deepseek/deepseek-chat": "DeepSeek Chat",
        "deepseek/deepseek-reasoner": "DeepSeek Reasoner",
    },
))

register_provider(ProviderInfo(
    name="dashscope",
    display_name="Qwen",
    base_url="https://dashscope.aliyuncs.com/api/v1",
    env_key="DASHSCOPE_API_KEY",
    wire_api=WireAPI.DASHSCOPE,
    model_prefixes=("qwq-", "qwen-", "qvq-", "qwen/"),
    models={
        "qwq-plus": "QwQ Plus (Reasoning)",
        "qwen-max": "Qwen Max (Latest)",
        "qwen-max-latest": "Qwen Max Latest",
        "qwen-plus": "Qwen Plus (Latest)",
        "qwen-turbo": "Qwen Turbo (Latest)",
    },
))

register_provider(ProviderInfo(
    name="moonshot",
    display_name="Moonshot",
    base_url="https://api.moonshot.cn/v1",
    env_key="MOONSHOT_API_KEY",
    wire_api=WireAPI.CHAT_COMPLETIONS,
    model_prefixes=("moonshot/", "kimi-", "moonshot-"),
    models={
        "moonshot/kimi-k2-0711-preview": "Kimi K2 (Preview)",
        "moonshot/kimi-k2-turbo-preview": "Kimi K2 Turbo (Preview)",
        "moonshot/kimi-latest": "Kimi Latest (Auto Context)",
        "moonshot/moonshot-v1-8k": "Moonshot V1 8K",
        "moonshot/moonshot-v1-32k": "Moonshot V1 32K",
        "moonshot/moonshot-v1-128k": "Moonshot V1 128K",
    },
))

register_provider(ProviderInfo(
    name="xai",
    display_name="Grok",
    base_url="https://api.x.ai/v1",
    env_key="XAI_API_KEY",
    wire_api=WireAPI.CHAT_COMPLETIONS,
    model_prefixes=("grok/",),
    models={
        "grok/grok-beta": "Grok Beta",
        "grok/grok-2": "Grok 2",
    },
))

register_provider(ProviderInfo(
    name="zhipu",
    display_name="Zhipu AI",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    env_key="ZAI_API_KEY",
    alt_env_keys=("ZHIPUAI_API_KEY",),
    wire_api=WireAPI.CHAT_COMPLETIONS,
    model_prefixes=("zhipu/",),
    models={
        "zhipu/glm-4.5": "GLM-4.5 (Zhipu AI - Latest)",
        "zhipu/glm-4.5-air": "GLM-4.5 Air (Zhipu AI - Latest)",
        "zhipu/glm-4.5-flash": "GLM-4.5 Flash (Zhipu AI - Latest)",
        "zhipu/glm-4": "GLM-4 (Zhipu AI)",
        "zhipu/glm-4-plus": "GLM-4 Plus (Zhipu AI)",
        "zhipu/glm-4-air": "GLM-4 Air (Zhipu AI)",
        "zhipu/glm-4-flash": "GLM-4 Flash (Zhipu AI - Free)",
    },
))


# ---------------------------------------------------------------------------
# Backward-compatible computed views
# ---------------------------------------------------------------------------

def _build_available_models() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for info in PROVIDER_REGISTRY.values():
        result.update(info.models)
    return result

def _build_provider_api_keys() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for info in PROVIDER_REGISTRY.values():
        if info.env_key:
            for model_id in info.models:
                result[model_id] = info.env_key
    return result

def _build_provider_endpoints() -> Dict[str, str]:
    return {info.name: info.base_url for info in PROVIDER_REGISTRY.values()}

def _build_provider_default_keys() -> Dict[str, str]:
    return {info.name: info.env_key for info in PROVIDER_REGISTRY.values() if info.env_key}


AVAILABLE_MODELS: Dict[str, str] = _build_available_models()
PROVIDER_API_KEYS: Dict[str, str] = _build_provider_api_keys()
PROVIDER_ENDPOINTS: Dict[str, str] = _build_provider_endpoints()
PROVIDER_DEFAULT_KEYS: Dict[str, str] = _build_provider_default_keys()


# ---------------------------------------------------------------------------
# Model ID aliases for backward compatibility
# ---------------------------------------------------------------------------

_RAW_MODEL_ALIASES = {
    # Local execution aliases
    "local-python": "python",
    "python-local": "python",
    "py-local": "python",

    # Claude 4.5 variations
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-20250514",
    "claude-4-5-sonnet": "anthropic/claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929": "anthropic/claude-sonnet-4-20250514",
    # Claude 4 Opus variations
    "claude-4-opus": "anthropic/claude-opus-4-20250514",
    "claude-opus-4": "anthropic/claude-opus-4-20250514",
    # Claude 3.7
    "claude-sonnet-3-7": "anthropic/claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet": "anthropic/claude-3-7-sonnet-20250219",
    # Claude 3.5
    "claude-3-5-haiku": "anthropic/claude-3-5-haiku-20241022",
    "claude-haiku-3-5": "anthropic/claude-3-5-haiku-20241022",
    # Claude 3 legacy
    "claude-3-opus": "anthropic/claude-3-opus-20240229",
    "claude-opus-3": "anthropic/claude-3-opus-20240229",
    "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
    "claude-sonnet-3": "anthropic/claude-3-sonnet-20240229",
    "claude-3-haiku": "anthropic/claude-3-haiku-20240307",
    "claude-haiku-3": "anthropic/claude-3-haiku-20240307",
    # Gemini
    "gemini-3-flash-preview": "gemini/gemini-3-flash-preview",
    "gemini-3-flash": "gemini/gemini-3-flash-preview",
    "gemini-2.5-pro": "gemini/gemini-2.5-pro",
    "gemini-2-5-pro": "gemini/gemini-2.5-pro",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
    "gemini-2-5-flash": "gemini/gemini-2.5-flash",
    "gemini-2.0-pro": "gemini/gemini-2.0-pro",
    "gemini-2-0-pro": "gemini/gemini-2.0-pro",
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",
    "gemini-2-0-flash": "gemini/gemini-2.0-flash",
    "gemini-pro": "gemini/gemini-pro",
    # Deepseek
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-reasoner": "deepseek/deepseek-reasoner",
}

MODEL_ALIASES: Dict[str, str] = {key.lower(): value for key, value in _RAW_MODEL_ALIASES.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _supports_unicode_output() -> bool:
    """Best-effort detection for whether stdout can render emoji."""
    encoding = getattr(sys.stdout, "encoding", None) or os.getenv("PYTHONIOENCODING") or "utf-8"
    try:
        "\U0001f4a1".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _provider_for_model(model: str) -> Optional[ProviderInfo]:
    """Find the ProviderInfo that owns *model* (after normalization)."""
    # Exact match in any provider's model catalog
    for info in PROVIDER_REGISTRY.values():
        if model in info.models:
            return info
    # Prefix-based fallback for custom / unlisted models
    for info in PROVIDER_REGISTRY.values():
        for prefix in info.model_prefixes:
            if model.startswith(prefix):
                return info
    return None


# ---------------------------------------------------------------------------
# ModelConfig (public API – all static methods, backward compatible)
# ---------------------------------------------------------------------------

class ModelConfig:
    """Model configuration and validation for OmicVerse Smart Agent"""

    @staticmethod
    def normalize_model_id(model: str) -> str:
        """Normalize model ID to canonical form using alias mapping."""
        if model in AVAILABLE_MODELS:
            return model
        alias = MODEL_ALIASES.get(model.lower())
        return alias or model

    @staticmethod
    def is_model_supported(model: str) -> bool:
        """Check if a model is supported"""
        normalized = ModelConfig.normalize_model_id(model)
        return normalized in AVAILABLE_MODELS

    @staticmethod
    def get_model_description(model: str) -> str:
        """Get human-readable description for a model"""
        normalized = ModelConfig.normalize_model_id(model)
        return AVAILABLE_MODELS.get(normalized, f"Unknown model: {model}")

    @staticmethod
    def get_provider_from_model(model: str) -> str:
        """Determine provider from model name using the registry."""
        model = ModelConfig.normalize_model_id(model)
        info = _provider_for_model(model)
        if info is not None:
            return info.name
        # Default to openai for unrecognized models (backward compat)
        return "openai"

    @staticmethod
    def check_api_key_availability(model: str) -> Tuple[bool, str]:
        """Check if required API key is available for the model"""
        normalized = ModelConfig.normalize_model_id(model)
        provider_name = ModelConfig.get_provider_from_model(normalized)
        info = get_provider(provider_name)
        if info is None or not info.env_key:
            return True, "No API key required"
        if os.getenv(info.env_key):
            return True, f"{info.display_name} API key available"
        for alt in info.alt_env_keys:
            if os.getenv(alt):
                return True, f"{info.display_name} API key available"
        return False, f"{info.display_name} API key required: set {info.env_key}"

    @staticmethod
    def get_endpoint_for_model(model: str) -> str:
        """Get the API endpoint for a model"""
        provider_name = ModelConfig.get_provider_from_model(model)
        info = get_provider(provider_name)
        if info is not None:
            return info.base_url
        return PROVIDER_REGISTRY["openai"].base_url

    @staticmethod
    def list_supported_models(show_all: bool = False) -> str:
        """List all supported models grouped by provider"""
        result = "\U0001f916 Supported Models:\n\n"

        # Group models by provider using registry order
        for info in PROVIDER_REGISTRY.values():
            if not info.models:
                continue
            result += f"**{info.display_name}**:\n"
            items = list(info.models.items())
            display = items if show_all else items[:3]

            for model_id, description in display:
                key_available, _ = ModelConfig.check_api_key_availability(model_id)
                key_status = " \u2705" if key_available else " \u274c"
                result += f"  \u2022 `{model_id}`: {description}{key_status}\n"

            if not show_all and len(items) > 3:
                result += f"  ... and {len(items) - 3} more models\n"
            result += "\n"

        result += "Legend: \u2705 API key available | \u274c API key missing\n\n"
        usage_hint = "Usage: `agent = ov.Agent(model='model_id', api_key='your_key')`"
        if _supports_unicode_output():
            result += f"\U0001f4a1 {usage_hint}"
        else:
            result += usage_hint

        return result

    @staticmethod
    def requires_responses_api(model: str) -> bool:
        """Check if model requires OpenAI Responses API instead of Chat Completions API."""
        normalized = ModelConfig.normalize_model_id(model)
        if normalized.startswith("gpt-5"):
            return True
        return False

    @staticmethod
    def validate_model_setup(model: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
        """Validate if model can be used with current setup"""
        normalized = ModelConfig.normalize_model_id(model)
        if not ModelConfig.is_model_supported(normalized):
            return False, f"Model '{model}' is not supported. Use ov.list_supported_models() to see available models."

        provider_name = ModelConfig.get_provider_from_model(normalized)
        info = get_provider(provider_name)
        if info is None or not info.env_key:
            return True, f"\u2705 Model {normalized} ready to use"

        if api_key or os.getenv(info.env_key) or any(os.getenv(k) for k in info.alt_env_keys):
            return True, f"\u2705 Model {normalized} ready to use"
        return False, f"\u274c Model {normalized} requires {info.env_key}. Set environment variable or pass api_key parameter."
