"""
Best-effort access to OmicVerse model/provider metadata.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import requests

_MISSING = object()
_MODEL_CONFIG_MODULE: Any = _MISSING

_FALLBACK_PROVIDER_ENV_VARS: Dict[str, Tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "qianfan": ("QIANFAN_API_KEY",),
    "xiaomi": ("XIAOMI_API_KEY",),
    "synthetic": ("SYNTHETIC_API_KEY",),
    "zhipu": ("ZAI_API_KEY", "ZHIPUAI_API_KEY"),
    "ollama": ("OPENAI_API_KEY",),
    "openai_compatible": ("OPENAI_API_KEY",),
    "python": (),
}

_FALLBACK_PROVIDER_BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "deepseek": "https://api.deepseek.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "minimax": "https://api.minimax.chat/v1",
    "together": "https://api.together.xyz/v1",
    "xai": "https://api.x.ai/v1",
    "qianfan": "https://qianfan.baidubce.com/v2",
    "xiaomi": "https://api.xiaomimimo.com/anthropic",
    "synthetic": "https://api.synthetic.new/anthropic",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "ollama": "http://127.0.0.1:11434/v1",
    "openai_compatible": "https://api.openai.com/v1",
}

_FALLBACK_PROVIDER_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3", "o4"),
    "anthropic": ("anthropic/", "claude-"),
    "google": ("gemini/", "gemini-"),
    "deepseek": ("deepseek/", "deepseek-"),
    "dashscope": ("qwq-", "qwen-", "qvq-", "qwen/"),
    "moonshot": ("moonshot/", "kimi-", "moonshot-"),
    "minimax": ("minimax/",),
    "together": ("together/",),
    "xai": ("xai/", "grok/", "grok-"),
    "qianfan": ("qianfan/",),
    "xiaomi": ("xiaomi/",),
    "synthetic": ("synthetic/",),
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
        "moonshot/kimi-k2.5": "Kimi K2.5",
        "moonshot/kimi-latest": "Kimi Latest",
        "moonshot/kimi-k2-0711-preview": "Kimi K2 Preview",
        "moonshot/kimi-k2-turbo-preview": "Kimi K2 Turbo Preview",
        "moonshot/moonshot-v1-32k": "Moonshot V1 32K",
        "moonshot/moonshot-v1-128k": "Moonshot V1 128K",
    },
    "minimax": {
        "minimax/MiniMax-M2.1": "MiniMax M2.1",
        "minimax/MiniMax-M2.1-lightning": "MiniMax M2.1 Lightning",
        "minimax/MiniMax-VL-01": "MiniMax VL 01",
    },
    "together": {
        "together/moonshotai/Kimi-K2.5": "Kimi K2.5 (Together)",
        "together/zai-org/GLM-4.7": "GLM 4.7 (Together)",
        "together/deepseek-ai/DeepSeek-V3.1": "DeepSeek V3.1 (Together)",
        "together/deepseek-ai/DeepSeek-R1": "DeepSeek R1 (Together)",
    },
    "xai": {
        "xai/grok-2": "Grok 2",
        "xai/grok-beta": "Grok Beta",
        "xai/grok-4": "Grok 4",
    },
    "qianfan": {
        "qianfan/deepseek-v3.2": "DEEPSEEK V3.2",
        "qianfan/ernie-5.0-thinking-preview": "ERNIE-5.0 Thinking Preview",
    },
    "xiaomi": {
        "xiaomi/mimo-v2-flash": "Xiaomi MiMo V2 Flash",
    },
    "synthetic": {
        "synthetic/hf:MiniMaxAI/MiniMax-M2.1": "MiniMax M2.1 (Synthetic)",
        "synthetic/hf:moonshotai/Kimi-K2.5": "Kimi K2.5 (Synthetic)",
        "synthetic/hf:zai-org/GLM-4.5": "GLM-4.5 (Synthetic)",
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
    "minimax",
    "together",
    "deepseek",
    "qianfan",
    "xiaomi",
    "synthetic",
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
    "minimax": "MiniMax",
    "together": "Together AI",
    "xai": "Grok",
    "qianfan": "Qianfan",
    "xiaomi": "Xiaomi MiMo",
    "synthetic": "Synthetic",
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
    "minimax": {
        "minimax/MiniMax-M2.1": "MiniMax M2.1",
    },
    "together": {
        "together/moonshotai/Kimi-K2.5": "Together Kimi K2.5",
    },
    "qianfan": {
        "qianfan/deepseek-v3.2": "Qianfan DeepSeek V3.2",
    },
    "xiaomi": {
        "xiaomi/mimo-v2-flash": "Xiaomi MiMo V2 Flash",
    },
    "synthetic": {
        "synthetic/hf:MiniMaxAI/MiniMax-M2.1": "Synthetic MiniMax M2.1",
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


def provider_base_url(provider_name: str) -> Optional[str]:
    if provider_name == "openai_compatible":
        return _FALLBACK_PROVIDER_BASE_URLS[provider_name]

    module = _load_model_config_module()
    if module is not None:
        get_provider = getattr(module, "get_provider", None)
        if callable(get_provider):
            info = get_provider(provider_name)
            if info is not None:
                base_url = str(getattr(info, "base_url", "") or "").strip()
                if base_url:
                    return base_url

    return _FALLBACK_PROVIDER_BASE_URLS.get(provider_name)


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


def resolve_ollama_api_base(configured_base_url: Optional[str]) -> str:
    base_url = str(configured_base_url or provider_base_url("ollama") or "").strip()
    trimmed = base_url.rstrip("/")
    if trimmed.lower().endswith("/v1"):
        trimmed = trimmed[:-3]
    return trimmed or "http://127.0.0.1:11434"


def _env_api_key(provider_name: str) -> Optional[str]:
    for env_name in provider_env_vars(provider_name):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _normalize_google_model_id(model_id: str) -> str:
    normalized = str(model_id or "").strip()
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]
    return normalized


def _looks_like_chat_model(provider_name: str, model_id: str) -> bool:
    lowered = str(model_id or "").strip().lower()
    if not lowered:
        return False

    negative_markers = (
        "embedding",
        "embed",
        "whisper",
        "tts",
        "transcribe",
        "transcription",
        "moderation",
        "omni-moderation",
        "rerank",
        "image",
        "vision-preview",
        "search",
        "audio",
    )
    if any(marker in lowered for marker in negative_markers):
        return False

    if provider_name == "google":
        return "gemini" in lowered
    if provider_name == "anthropic":
        return lowered.startswith("claude-")
    if provider_name == "openai":
        return lowered.startswith(("gpt-", "o1", "o3"))

    return True


def _request_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    timeout_seconds: float = 5.0,
) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=timeout_seconds,
        )
        if not response.ok:
            return None
        payload = response.json()
    except Exception:
        return None

    return payload if isinstance(payload, dict) else None


def _discover_openai_compatible_models(
    base_url: str,
    *,
    api_key: Optional[str],
    provider_name: str,
    timeout_seconds: float,
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = _request_json(
        f"{base_url.rstrip('/')}/models",
        headers=headers or None,
        timeout_seconds=timeout_seconds,
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {}

    models: Dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id or not _looks_like_chat_model(provider_name, model_id):
            continue
        models[model_id] = model_id
    return models


def _discover_anthropic_models(*, api_key: Optional[str], timeout_seconds: float) -> Dict[str, str]:
    if not api_key:
        return {}
    payload = _request_json(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout_seconds=timeout_seconds,
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {}

    models: Dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id or not _looks_like_chat_model("anthropic", model_id):
            continue
        models[model_id] = str(item.get("display_name") or model_id).strip() or model_id
    return models


def _discover_google_models(*, api_key: Optional[str], timeout_seconds: float) -> Dict[str, str]:
    if not api_key:
        return {}
    payload = _request_json(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key},
        timeout_seconds=timeout_seconds,
    )
    data = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {}

    models: Dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = _normalize_google_model_id(str(item.get("name") or ""))
        if not model_id or not _looks_like_chat_model("google", model_id):
            continue
        models[model_id] = str(item.get("displayName") or model_id).strip() or model_id
    return models


def _discover_ollama_models(base_url: Optional[str], *, timeout_seconds: float) -> Dict[str, str]:
    payload = _request_json(
        f"{resolve_ollama_api_base(base_url)}/api/tags",
        timeout_seconds=timeout_seconds,
    )
    data = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {}

    models: Dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("name") or "").strip()
        if not model_id:
            continue
        models[model_id] = model_id
    return models


def discover_provider_models(
    provider_name: str,
    *,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: float = 5.0,
) -> Dict[str, str]:
    if provider_name == "python":
        return {"python": "Local Python executor"}

    resolved_key = api_key or _env_api_key(provider_name)

    if provider_name == "ollama":
        return _discover_ollama_models(endpoint, timeout_seconds=timeout_seconds)

    if provider_name == "anthropic":
        return _discover_anthropic_models(api_key=resolved_key, timeout_seconds=timeout_seconds)

    if provider_name == "google":
        return _discover_google_models(api_key=resolved_key, timeout_seconds=timeout_seconds)

    base_url = str(endpoint or provider_base_url(provider_name) or "").strip()
    if not base_url:
        return {}
    if provider_name == "openai" and "chatgpt.com/backend-api" in base_url:
        return {}

    return _discover_openai_compatible_models(
        base_url,
        api_key=resolved_key,
        provider_name=provider_name,
        timeout_seconds=timeout_seconds,
    )


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
