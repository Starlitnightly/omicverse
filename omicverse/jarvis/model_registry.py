"""
Best-effort access to OmicVerse model/provider metadata.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

_MISSING = object()
_MODEL_CONFIG_MODULE: Any = _MISSING

_FALLBACK_PROVIDER_ENV_VARS: Dict[str, Tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "zhipu": ("ZAI_API_KEY", "ZHIPUAI_API_KEY"),
    "ollama": ("OPENAI_API_KEY",),
    "openai_compatible": ("OPENAI_API_KEY",),
    "python": (),
}

_FALLBACK_PROVIDER_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3"),
    "anthropic": ("anthropic/", "claude-"),
    "google": ("gemini/", "gemini-"),
    "deepseek": ("deepseek/", "deepseek-"),
    "dashscope": ("qwq-", "qwen-", "qvq-", "qwen/"),
    "moonshot": ("moonshot/", "kimi-", "moonshot-"),
    "xai": ("grok/", "grok-"),
    "zhipu": ("zhipu/", "glm-"),
    "python": ("python",),
}

_FALLBACK_PROVIDER_MODELS: Dict[str, Dict[str, str]] = {
    "openai": {
        "gpt-5.4": "OpenAI GPT-5.4",
        "gpt-5": "OpenAI GPT-5",
        "gpt-5-mini": "OpenAI GPT-5 Mini",
        "gpt-4.1": "OpenAI GPT-4.1",
        "gpt-4o": "OpenAI GPT-4o",
    },
    "anthropic": {
        "claude-sonnet-4-6": "Claude Sonnet 4.6",
        "claude-opus-4-6": "Claude Opus 4.6",
        "claude-haiku-4-5": "Claude Haiku 4.5",
        "claude-opus-4-1": "Claude Opus 4.1",
        "claude-sonnet-3-7": "Claude Sonnet 3.7",
    },
    "google": {
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "gemini-2.5-flash": "Gemini 2.5 Flash",
        "gemini-3-flash-preview": "Gemini 3 Flash Preview",
        "gemini-pro": "Gemini Pro",
    },
    "deepseek": {
        "deepseek-chat": "DeepSeek Chat",
        "deepseek-reasoner": "DeepSeek Reasoner",
    },
    "dashscope": {
        "qwen-max": "Qwen Max",
        "qwen-max-latest": "Qwen Max Latest",
        "qwen-plus": "Qwen Plus",
        "qwen-turbo": "Qwen Turbo",
        "qwq-plus": "QwQ Plus",
    },
    "moonshot": {
        "moonshot/kimi-latest": "Kimi Latest",
        "moonshot/kimi-k2-0711-preview": "Kimi K2 Preview",
        "moonshot/kimi-k2-turbo-preview": "Kimi K2 Turbo Preview",
        "moonshot/moonshot-v1-32k": "Moonshot V1 32K",
        "moonshot/moonshot-v1-128k": "Moonshot V1 128K",
    },
    "xai": {
        "grok/grok-2": "Grok 2",
        "grok/grok-beta": "Grok Beta",
    },
    "zhipu": {
        "zhipu/glm-4.5": "GLM-4.5",
        "zhipu/glm-4.5-air": "GLM-4.5 Air",
        "zhipu/glm-4.5-flash": "GLM-4.5 Flash",
        "zhipu/glm-4-flash": "GLM-4 Flash",
    },
    "python": {
        "python": "Local Python executor",
    },
}

_CHAT_PROVIDER_ORDER: Tuple[str, ...] = (
    "openai",
    "anthropic",
    "dashscope",
    "moonshot",
    "deepseek",
    "zhipu",
    "google",
    "xai",
    "ollama",
    "openai_compatible",
    "python",
)

_FALLBACK_PROVIDER_DISPLAY_NAMES: Dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "deepseek": "DeepSeek",
    "dashscope": "Qwen",
    "moonshot": "Moonshot",
    "xai": "Grok",
    "zhipu": "Zhipu AI",
    "ollama": "Ollama",
    "openai_compatible": "OpenAI-Compatible",
    "python": "Local Python",
}

_CHAT_EXTRA_PROVIDER_MODELS: Dict[str, Dict[str, str]] = {
    "openai": {
        "gpt-5.3-codex": "OpenAI Codex GPT-5.3",
        "gpt-5.2-codex": "OpenAI Codex GPT-5.2",
        "gpt-5.1-codex-mini": "OpenAI Codex GPT-5.1 Mini",
        "gpt-5.1-codex-max": "OpenAI Codex GPT-5.1 Max",
    },
    "ollama": {
        "qwen3:8b": "Ollama Qwen3 8B",
        "qwen2.5:7b": "Ollama Qwen2.5 7B",
        "qwen2.5-coder:7b": "Ollama Qwen2.5 Coder 7B",
        "deepseek-r1:8b": "Ollama DeepSeek R1 8B",
        "llama3.2:3b": "Ollama Llama 3.2 3B",
        "gemma3:4b": "Ollama Gemma 3 4B",
    },
    "openai_compatible": {
        "gpt-4o-mini": "OpenAI-compatible GPT-4o Mini",
        "qwen2.5:7b": "OpenAI-compatible Qwen2.5 7B",
        "deepseek-r1:8b": "OpenAI-compatible DeepSeek R1 8B",
        "llama3.1:8b": "OpenAI-compatible Llama 3.1 8B",
    },
}


def _load_model_config_module() -> Optional[Any]:
    global _MODEL_CONFIG_MODULE
    if _MODEL_CONFIG_MODULE is not _MISSING:
        return _MODEL_CONFIG_MODULE

    try:
        from omicverse.utils import model_config as module  # type: ignore

        _MODEL_CONFIG_MODULE = module
        return module
    except Exception:
        pass

    candidate = Path(__file__).resolve().parents[1] / "utils" / "model_config.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(
            "omicverse.jarvis._model_config_fallback",
            candidate,
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODEL_CONFIG_MODULE = module
            return module

    _MODEL_CONFIG_MODULE = None
    return None


def iter_known_provider_env_vars() -> Iterable[str]:
    module = _load_model_config_module()
    seen = set()

    if module is not None:
        registry = getattr(module, "PROVIDER_REGISTRY", None)
        if isinstance(registry, dict):
            for info in registry.values():
                for env_name in (getattr(info, "env_key", ""), *getattr(info, "alt_env_keys", ())):
                    if env_name and env_name not in seen:
                        seen.add(env_name)
                        yield env_name

    for env_names in _FALLBACK_PROVIDER_ENV_VARS.values():
        for env_name in env_names:
            if env_name and env_name not in seen:
                seen.add(env_name)
                yield env_name


def provider_env_vars(provider_name: str) -> Tuple[str, ...]:
    if provider_name in {"ollama", "openai_compatible"}:
        return _FALLBACK_PROVIDER_ENV_VARS[provider_name]

    module = _load_model_config_module()
    if module is not None:
        get_provider = getattr(module, "get_provider", None)
        if callable(get_provider):
            info = get_provider(provider_name)
            if info is not None:
                env_names = tuple(name for name in (getattr(info, "env_key", ""), *getattr(info, "alt_env_keys", ())) if name)
                if env_names:
                    return env_names

    return _FALLBACK_PROVIDER_ENV_VARS.get(provider_name, ())


def provider_from_model(model: str) -> str:
    normalized = str(model or "").strip()
    if not normalized:
        return "openai"

    module = _load_model_config_module()
    if module is not None:
        model_config = getattr(module, "ModelConfig", None)
        if model_config is not None:
            try:
                return str(model_config.get_provider_from_model(normalized))
            except Exception:
                pass

    lowered = normalized.lower()
    for provider_name, models in _FALLBACK_PROVIDER_MODELS.items():
        if normalized in models or lowered in {name.lower() for name in models}:
            return provider_name
    for provider_name, prefixes in _FALLBACK_PROVIDER_PREFIXES.items():
        if any(lowered.startswith(prefix.lower()) for prefix in prefixes):
            return provider_name
    return "openai"


def provider_models(provider_name: str) -> Dict[str, str]:
    if provider_name in {"ollama", "openai_compatible"}:
        return {}

    module = _load_model_config_module()
    if module is not None:
        get_provider = getattr(module, "get_provider", None)
        if callable(get_provider):
            info = get_provider(provider_name)
            if info is not None:
                models = getattr(info, "models", None)
                if isinstance(models, dict):
                    return dict(models)

    return dict(_FALLBACK_PROVIDER_MODELS.get(provider_name, {}))


def provider_display_name(provider_name: str) -> str:
    module = _load_model_config_module()
    if module is not None:
        get_provider = getattr(module, "get_provider", None)
        if callable(get_provider):
            info = get_provider(provider_name)
            if info is not None:
                display_name = str(getattr(info, "display_name", "") or "").strip()
                if display_name:
                    return display_name
    return _FALLBACK_PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.replace("_", " ").title())


def iter_supported_model_catalog() -> Iterator[Tuple[str, str, Dict[str, str]]]:
    for provider_name in _CHAT_PROVIDER_ORDER:
        models = provider_models(provider_name)
        merged = dict(models)
        for model_id, description in _CHAT_EXTRA_PROVIDER_MODELS.get(provider_name, {}).items():
            merged.setdefault(model_id, description)
        if not merged:
            continue
        yield provider_name, provider_display_name(provider_name), merged


def model_description(model: str) -> str:
    normalized = str(model or "").strip()
    if not normalized:
        return ""

    module = _load_model_config_module()
    if module is not None:
        model_config = getattr(module, "ModelConfig", None)
        if model_config is not None:
            try:
                return str(model_config.get_model_description(normalized))
            except Exception:
                pass

    provider_name = provider_from_model(normalized)
    return provider_models(provider_name).get(normalized, normalized)
