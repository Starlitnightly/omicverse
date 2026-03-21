"""
Interactive setup wizard for Jarvis.
"""
from __future__ import annotations

import getpass
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import load_auth, save_auth
from .model_registry import (
    discover_provider_models,
    model_description,
    provider_base_url,
    provider_env_vars,
    provider_from_model,
    provider_models,
)
from .openai_oauth import OPENAI_CODEX_BASE_URL, OpenAIOAuthManager

Language = str

OLLAMA_DEFAULT_ENDPOINT = "http://127.0.0.1:11434/v1"
OPENAI_COMPATIBLE_DEFAULT_ENDPOINT = "https://api.openai.com/v1"
OPENAI_CODEX_DEFAULT_ENDPOINT = OPENAI_CODEX_BASE_URL
OPENAI_CODEX_DEFAULT_MODEL = "gpt-5.3-codex"

_PROVIDER_ORDER: List[str] = [
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
]

_PROVIDER_MODEL_CHOICES: Dict[str, List[str]] = {
    "openai": [
        "gpt-5.4",
        "gpt-5",
        "gpt-5-mini",
        "gpt-4.1",
        "gpt-4o",
    ],
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5",
        "claude-opus-4-1",
        "claude-sonnet-3-7",
    ],
    "dashscope": [
        "qwen-max",
        "qwen-max-latest",
        "qwen-plus",
        "qwen-turbo",
        "qwq-plus",
    ],
    "moonshot": [
        "moonshot/kimi-k2.5",
        "moonshot/kimi-latest",
        "moonshot/kimi-k2-0711-preview",
        "moonshot/kimi-k2-turbo-preview",
        "moonshot/moonshot-v1-32k",
        "moonshot/moonshot-v1-128k",
    ],
    "minimax": [
        "minimax/MiniMax-M2.1",
        "minimax/MiniMax-M2.1-lightning",
        "minimax/MiniMax-VL-01",
    ],
    "together": [
        "together/moonshotai/Kimi-K2.5",
        "together/zai-org/GLM-4.7",
        "together/deepseek-ai/DeepSeek-V3.1",
        "together/deepseek-ai/DeepSeek-R1",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "qianfan": [
        "qianfan/deepseek-v3.2",
        "qianfan/ernie-5.0-thinking-preview",
    ],
    "xiaomi": [
        "xiaomi/mimo-v2-flash",
    ],
    "synthetic": [
        "synthetic/hf:MiniMaxAI/MiniMax-M2.1",
        "synthetic/hf:moonshotai/Kimi-K2.5",
        "synthetic/hf:zai-org/GLM-4.5",
    ],
    "zhipu": [
        "zhipu/glm-4.5",
        "zhipu/glm-4.5-air",
        "zhipu/glm-4.5-flash",
        "zhipu/glm-4-flash",
    ],
    "google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-pro",
    ],
    "xai": [
        "grok/grok-2",
        "grok/grok-beta",
    ],
    "ollama": [
        "qwen3:8b",
        "qwen2.5:7b",
        "qwen2.5-coder:7b",
        "deepseek-r1:8b",
        "llama3.2:3b",
        "gemma3:4b",
    ],
    "openai_compatible": [
        "gpt-4o-mini",
        "qwen2.5:7b",
        "deepseek-r1:8b",
        "llama3.1:8b",
    ],
    "python": ["python"],
}

_OPENAI_CODEX_MODEL_CHOICES: List[str] = [
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
]

_COPY: Dict[Language, Dict[str, str]] = {
    "en": {
        "default_marker": " (default)",
        "choice_prompt": "Select [default {index}]: ",
        "invalid_choice": "Invalid input, please choose again.",
        "menu_hint": "Use Up/Down and Enter.",
        "yes_label": "Yes",
        "no_label": "No",
        "custom_model": "Enter model name manually",
        "wizard_title": "OmicVerse Jarvis Setup",
        "wizard_intro": (
            "This wizard saves your channel, provider, authentication method, and default model. "
            "You can rerun `omicverse jarvis --setup` any time."
        ),
        "language_title": "Setup language",
        "language_english": "English",
        "language_chinese": "简体中文",
        "channel_title": "Choose a messaging channel",
        "telegram_config": "Telegram configuration",
        "discord_config": "Discord configuration",
        "bot_token": "Bot token",
        "allowed_users": "Allowed usernames or IDs (comma-separated, optional)",
        "feishu_config": "Feishu configuration",
        "connection_mode": "Connection mode",
        "websocket_mode": "WebSocket long connection",
        "webhook_mode": "Webhook",
        "verification_token": "Verification token (optional)",
        "encrypt_key": "Encrypt key (optional)",
        "webhook_host": "Webhook host",
        "webhook_port": "Webhook port",
        "webhook_path": "Webhook path",
        "imessage_config": "iMessage configuration",
        "imessage_hint": "Requires imsg on macOS, for example: brew install steipete/tap/imsg",
        "imessage_cli_path": "imsg CLI path",
        "imessage_db_path": "Messages chat.db path",
        "imessage_include_attachments": "Include inbound attachment metadata",
        "qq_config": "QQ configuration",
        "qq_app_id": "QQ Bot AppID",
        "qq_client_secret": "QQ Bot ClientSecret / AppSecret",
        "qq_image_host": "Public image host URL (optional)",
        "qq_image_server_port": "Image hosting port",
        "qq_markdown": "Enable QQ markdown replies",
        "provider_title": "Choose an LLM provider",
        "provider_openai": "OpenAI",
        "provider_anthropic": "Anthropic (Claude)",
        "provider_dashscope": "Qwen / DashScope",
        "provider_moonshot": "Kimi / Moonshot",
        "provider_minimax": "MiniMax",
        "provider_together": "Together AI",
        "provider_deepseek": "DeepSeek",
        "provider_qianfan": "Baidu Qianfan",
        "provider_xiaomi": "Xiaomi MiMo",
        "provider_synthetic": "Synthetic",
        "provider_zhipu": "Zhipu AI (GLM)",
        "provider_google": "Google Gemini",
        "provider_xai": "Grok / xAI",
        "provider_ollama": "Ollama (local)",
        "provider_openai_compatible": "OpenAI-compatible endpoint",
        "provider_python": "Local Python (no LLM)",
        "auth_title": "Choose an authentication method",
        "auth_saved_api_key": "Save an API key in Jarvis",
        "auth_environment": "Use environment variables",
        "auth_no_auth": "No API key",
        "auth_openai_oauth": "OpenAI Codex OAuth (ChatGPT sign-in, Recommended)",
        "oauth_open_browser": "A browser page will open for OpenAI sign-in.",
        "oauth_manual_hint": (
            "If the callback does not complete automatically, finish login and paste the "
            "redirect URL from the browser address bar."
        ),
        "oauth_auth_url": "Authorization URL",
        "oauth_callback_prompt": "Callback URL (or code#state)",
        "provider_model_title": "Choose a model for {provider}",
        "provider_model_input": "Enter model name",
        "api_key_config": "API key configuration",
        "api_key_prompt": "{provider} API key (leave blank to keep the current one)",
        "api_key_required": "{provider} API key cannot be empty.",
        "endpoint_config": "Endpoint configuration",
        "endpoint_prompt": "API base URL",
        "endpoint_required": "API base URL cannot be empty.",
        "ollama_hint": (
            "Jarvis uses the OpenAI-compatible Ollama endpoint. "
            "The usual value is http://127.0.0.1:11434/v1"
        ),
        "openai_compatible_hint": (
            "Use this for local vLLM, LM Studio, One API, OpenRouter, or any other "
            "OpenAI-compatible gateway."
        ),
        "env_auth_format": "Use environment variable(s): {env_vars}",
        "session_dir": "Session directory (blank for ~/.ovjarvis)",
        "max_prompts": "Max prompts per kernel (0 means unlimited)",
    },
    "zh": {
        "default_marker": " (默认)",
        "choice_prompt": "选择 [默认 {index}]: ",
        "invalid_choice": "输入无效，请重新选择。",
        "menu_hint": "使用上下方向键，回车确认。",
        "yes_label": "是",
        "no_label": "否",
        "custom_model": "手动输入模型",
        "wizard_title": "OmicVerse Jarvis 设置向导",
        "wizard_intro": "这一步会保存渠道、模型提供方、认证方式和默认模型，后续可重复运行 `omicverse jarvis --setup` 修改。",
        "language_title": "设置语言",
        "language_english": "English",
        "language_chinese": "简体中文",
        "channel_title": "选择消息渠道",
        "telegram_config": "Telegram 配置",
        "discord_config": "Discord 配置",
        "bot_token": "Bot Token",
        "allowed_users": "允许的用户名或 ID（逗号分隔，可留空）",
        "feishu_config": "Feishu 配置",
        "connection_mode": "连接模式",
        "websocket_mode": "WebSocket 长连接",
        "webhook_mode": "Webhook",
        "verification_token": "Verification Token（可留空）",
        "encrypt_key": "Encrypt Key（可留空）",
        "webhook_host": "Webhook Host",
        "webhook_port": "Webhook Port",
        "webhook_path": "Webhook Path",
        "imessage_config": "iMessage 配置",
        "imessage_hint": "需要在 macOS 上安装并授权 imsg，例如: brew install steipete/tap/imsg",
        "imessage_cli_path": "imsg CLI 路径",
        "imessage_db_path": "Messages chat.db 路径",
        "imessage_include_attachments": "读取入站附件元数据",
        "qq_config": "QQ 配置",
        "qq_app_id": "QQ Bot AppID",
        "qq_client_secret": "QQ Bot ClientSecret / AppSecret",
        "qq_image_host": "公网图片地址前缀（可留空）",
        "qq_image_server_port": "图片服务端口",
        "qq_markdown": "启用 QQ Markdown 回复",
        "provider_title": "选择 LLM 提供方",
        "provider_openai": "OpenAI",
        "provider_anthropic": "Anthropic（Claude）",
        "provider_dashscope": "千问 / DashScope",
        "provider_moonshot": "Kimi / Moonshot",
        "provider_minimax": "MiniMax",
        "provider_together": "Together AI",
        "provider_deepseek": "DeepSeek",
        "provider_qianfan": "百度千帆",
        "provider_xiaomi": "小米 MiMo",
        "provider_synthetic": "Synthetic",
        "provider_zhipu": "智谱 AI（GLM）",
        "provider_google": "Google Gemini",
        "provider_xai": "Grok / xAI",
        "provider_ollama": "Ollama（本地）",
        "provider_openai_compatible": "OpenAI 兼容端点",
        "provider_python": "本地 Python（不使用 LLM）",
        "auth_title": "选择认证方式",
        "auth_saved_api_key": "把 API Key 保存到 Jarvis",
        "auth_environment": "使用环境变量",
        "auth_no_auth": "不使用 API Key",
        "auth_openai_oauth": "OpenAI Codex OAuth（ChatGPT 登录，推荐）",
        "oauth_open_browser": "浏览器中将打开 OpenAI 登录页面。",
        "oauth_manual_hint": "如果没有自动跳回，请完成登录后，把浏览器地址栏中的回调 URL 粘贴回来。",
        "oauth_auth_url": "授权地址",
        "oauth_callback_prompt": "回调 URL（或 code#state）",
        "provider_model_title": "为 {provider} 选择模型",
        "provider_model_input": "输入模型名称",
        "api_key_config": "API Key 配置",
        "api_key_prompt": "{provider} API Key（留空则保留现有）",
        "api_key_required": "{provider} API Key 不能为空。",
        "endpoint_config": "端点配置",
        "endpoint_prompt": "API Base URL",
        "endpoint_required": "API Base URL 不能为空。",
        "ollama_hint": "Jarvis 使用 Ollama 的 OpenAI 兼容端点，通常填写 http://127.0.0.1:11434/v1",
        "openai_compatible_hint": "适用于本地 vLLM、LM Studio、One API、OpenRouter 或其他 OpenAI 兼容网关。",
        "env_auth_format": "使用环境变量：{env_vars}",
        "session_dir": "Session 目录（留空使用 ~/.ovjarvis）",
        "max_prompts": "每个 kernel 最大 prompts（0 表示不限制）",
    },
}


def _copy(language: Language, key: str) -> str:
    return _COPY.get(language, _COPY["en"])[key]


def _prompt_text(prompt: str, default: str = "", secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    label = f"{prompt}{suffix}: "
    value = getpass.getpass(label) if secret else input(label)
    value = value.strip()
    return value if value else default


def _supports_arrow_menu() -> bool:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return True


def _read_menu_key() -> str:
    if os.name == "nt":
        import msvcrt

        while True:
            char = msvcrt.getwch()
            if char in {"\r", "\n"}:
                return "enter"
            if char == "\x03":
                raise KeyboardInterrupt
            if char in {"\x00", "\xe0"}:
                code = msvcrt.getwch()
                if code == "H":
                    return "up"
                if code == "P":
                    return "down"
        # pragma: no cover

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            if char in {"\r", "\n"}:
                return "enter"
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x1b":
                second = sys.stdin.read(1)
                if second != "[":
                    continue
                third = sys.stdin.read(1)
                if third == "A":
                    return "up"
                if third == "B":
                    return "down"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _clear_menu_render(lines_rendered: int) -> None:
    if lines_rendered <= 0:
        return
    sys.stdout.write(f"\x1b[{lines_rendered}A")
    for index in range(lines_rendered):
        sys.stdout.write("\r\x1b[2K")
        if index < lines_rendered - 1:
            sys.stdout.write("\x1b[1B")
    if lines_rendered > 1:
        sys.stdout.write(f"\x1b[{lines_rendered - 1}A")
    sys.stdout.write("\r")
    sys.stdout.flush()


def _render_arrow_menu(
    title: str,
    options: Sequence[Tuple[str, str]],
    selected_index: int,
    *,
    default: str,
    language: Language,
    lines_rendered: int,
) -> int:
    _clear_menu_render(lines_rendered)

    lines = ["", title]
    for index, (value, label) in enumerate(options):
        marker = _copy(language, "default_marker") if value == default else ""
        line = f"  {label}{marker}"
        if index == selected_index:
            lines.append(f"\x1b[7m{line}\x1b[0m")
        else:
            lines.append(line)
    lines.append(f"  {_copy(language, 'menu_hint')}")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
    return len(lines)


def _prompt_bool(prompt: str, default: bool = True, language: Language = "en") -> bool:
    if _supports_arrow_menu():
        value = _prompt_choice(
            prompt,
            [
                ("yes", _copy(language, "yes_label")),
                ("no", _copy(language, "no_label")),
            ],
            default="yes" if default else "no",
            language=language,
        )
        return value == "yes"

    label = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{label}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def _prompt_choice(
    title: str,
    options: Sequence[Tuple[str, str]],
    *,
    default: str,
    language: Language,
) -> str:
    if _supports_arrow_menu():
        selected_index = next(
            (index for index, (value, _label) in enumerate(options) if value == default),
            0,
        )
        lines_rendered = 0
        while True:
            lines_rendered = _render_arrow_menu(
                title,
                options,
                selected_index,
                default=default,
                language=language,
                lines_rendered=lines_rendered,
            )
            key = _read_menu_key()
            if key == "up":
                selected_index = (selected_index - 1) % len(options)
                continue
            if key == "down":
                selected_index = (selected_index + 1) % len(options)
                continue
            if key == "enter":
                return options[selected_index][0]

    print(f"\n{title}")
    default_idx = 1
    for idx, (value, label) in enumerate(options, start=1):
        marker = _copy(language, "default_marker") if value == default else ""
        print(f"  {idx}. {label}{marker}")
        if value == default:
            default_idx = idx
    prompt = _copy(language, "choice_prompt").format(index=default_idx)
    while True:
        raw = input(prompt).strip()
        if not raw:
            return options[default_idx - 1][0]
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(options):
                return options[index - 1][0]
        values = {value for value, _label in options}
        if raw in values:
            return raw
        print(_copy(language, "invalid_choice"))


def _prompt_csv(prompt: str, default_values: Sequence[str]) -> List[str]:
    default = ",".join(str(item) for item in default_values if str(item).strip())
    raw = _prompt_text(prompt, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _select_language(config: Dict[str, Any], forced_language: Optional[str]) -> Language:
    if forced_language in {"en", "zh"}:
        return forced_language
    default = str(config.get("setup_language") or "en")
    return _prompt_choice(
        "Setup language",
        [
            ("en", "English"),
            ("zh", "简体中文"),
        ],
        default=default if default in {"en", "zh"} else "en",
        language="en",
    )


def _provider_label(provider_name: str, language: Language) -> str:
    key = f"provider_{provider_name}"
    if key in _COPY.get(language, {}):
        return _copy(language, key)
    return provider_name


def _provider_env_vars(provider_name: str) -> str:
    env_names = provider_env_vars(provider_name)
    return ", ".join(env_names) if env_names else "None"


def _provider_options(language: Language) -> List[Tuple[str, str]]:
    return [(provider_name, _provider_label(provider_name, language)) for provider_name in _PROVIDER_ORDER]


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:8]}..."


def _saved_provider_api_key(auth_path: Any, provider_name: str) -> str:
    auth = load_auth(auth_path)
    providers = dict(auth.get("providers") or {})
    if provider_name == "openai":
        top_level = str(auth.get("OPENAI_API_KEY") or "").strip()
        if top_level:
            return top_level
    entry = dict(providers.get(provider_name) or {})
    return str(entry.get("api_key") or "").strip()


def _store_provider_api_key(auth_path: Any, provider_name: str, api_key: str) -> None:
    auth = load_auth(auth_path)
    providers = dict(auth.get("providers") or {})
    entry = dict(providers.get(provider_name) or {})
    entry["api_key"] = api_key
    providers[provider_name] = entry
    auth["providers"] = providers
    if provider_name == "openai":
        auth["OPENAI_API_KEY"] = api_key
    save_auth(auth, auth_path)


def _infer_provider(config: Dict[str, Any]) -> str:
    explicit = str(config.get("llm_provider") or "").strip()
    if explicit:
        return explicit

    model = str(config.get("model") or "").strip()
    if not model:
        return "anthropic"

    return provider_from_model(model)


def _default_model_for_provider(provider_name: str) -> str:
    options = _PROVIDER_MODEL_CHOICES.get(provider_name) or []
    if options:
        return options[0]
    registry_models = list(provider_models(provider_name).keys())
    if registry_models:
        return registry_models[0]
    return "gpt-5.4"


def _model_default_for_provider(provider_name: str, current_model: str) -> str:
    if provider_name in {"ollama", "openai_compatible"}:
        return current_model or _default_model_for_provider(provider_name)

    if current_model and provider_from_model(current_model) == provider_name:
        return current_model
    return _default_model_for_provider(provider_name)


def _is_openai_codex_model(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return False
    return normalized in {name.lower() for name in _OPENAI_CODEX_MODEL_CHOICES}


def _openai_model_default(current_model: str, auth_mode: str) -> str:
    if auth_mode == "openai_oauth":
        return current_model if _is_openai_codex_model(current_model) else OPENAI_CODEX_DEFAULT_MODEL
    if current_model and provider_from_model(current_model) == "openai" and not _is_openai_codex_model(current_model):
        return current_model
    return _default_model_for_provider("openai")


def _model_label(model: str) -> str:
    description = model_description(model)
    if not description or description == model or description.startswith("Unknown model:"):
        return model
    return f"{model} - {description}"


def _provider_model_options(
    provider_name: str,
    *,
    current_model: str = "",
    discovered_models: Optional[Sequence[str]] = None,
    preferred_models: Optional[Sequence[str]] = None,
    include_registry: bool = True,
) -> List[str]:
    preferred = list(preferred_models if preferred_models is not None else (_PROVIDER_MODEL_CHOICES.get(provider_name) or []))
    seen = set()
    result: List[str] = []

    registry_models = list(provider_models(provider_name).keys()) if include_registry else []
    dynamic_models = list(discovered_models or [])
    if current_model:
        preferred = [current_model, *preferred]
    for model in dynamic_models + preferred + registry_models:
        if model in seen:
            continue
        seen.add(model)
        result.append(model)
    return result or preferred


def _prompt_model(
    provider_name: str,
    default_model: str,
    language: Language,
    *,
    current_model: str = "",
    discovered_models: Optional[Sequence[str]] = None,
    preferred_models: Optional[Sequence[str]] = None,
    include_registry: bool = True,
) -> str:
    if provider_name == "python":
        return "python"

    models = _provider_model_options(
        provider_name,
        current_model=current_model,
        discovered_models=discovered_models,
        preferred_models=preferred_models,
        include_registry=include_registry,
    )
    options: List[Tuple[str, str]] = [(model, _model_label(model)) for model in models[:16]]
    options.append(("__custom__", _copy(language, "custom_model")))
    option_values = {value for value, _label in options}
    selected = _prompt_choice(
        _copy(language, "provider_model_title").format(provider=_provider_label(provider_name, language)),
        options,
        default=default_model if default_model in option_values else "__custom__",
        language=language,
    )
    if selected == "__custom__":
        return _prompt_text(_copy(language, "provider_model_input"), default_model or _default_model_for_provider(provider_name))
    return selected


def _prompt_channel_config(channel: str, config: Dict[str, Any], language: Language) -> Dict[str, Any]:
    next_config = dict(config)

    if channel == "telegram":
        cur = dict(config.get("telegram") or {})
        print(f"\n{_copy(language, 'telegram_config')}")
        cur["token"] = _prompt_text(_copy(language, "bot_token"), str(cur.get("token") or ""))
        cur["allowed_users"] = _prompt_csv(_copy(language, "allowed_users"), cur.get("allowed_users") or [])
        next_config["telegram"] = cur
        return next_config

    if channel == "discord":
        cur = dict(config.get("discord") or {})
        print(f"\n{_copy(language, 'discord_config')}")
        cur["token"] = _prompt_text(_copy(language, "bot_token"), str(cur.get("token") or ""), secret=True)
        next_config["discord"] = cur
        return next_config

    if channel == "feishu":
        cur = dict(config.get("feishu") or {})
        print(f"\n{_copy(language, 'feishu_config')}")
        cur["app_id"] = _prompt_text("App ID", str(cur.get("app_id") or ""))
        cur["app_secret"] = _prompt_text("App Secret", str(cur.get("app_secret") or ""), secret=True)
        cur["connection_mode"] = _prompt_choice(
            _copy(language, "connection_mode"),
            [("websocket", _copy(language, "websocket_mode")), ("webhook", _copy(language, "webhook_mode"))],
            default=str(cur.get("connection_mode") or "websocket"),
            language=language,
        )
        cur["verification_token"] = _prompt_text(
            _copy(language, "verification_token"),
            str(cur.get("verification_token") or ""),
        )
        cur["encrypt_key"] = _prompt_text(
            _copy(language, "encrypt_key"),
            str(cur.get("encrypt_key") or ""),
            secret=True,
        )
        if cur["connection_mode"] == "webhook":
            cur["host"] = _prompt_text(_copy(language, "webhook_host"), str(cur.get("host") or "0.0.0.0"))
            cur["port"] = int(_prompt_text(_copy(language, "webhook_port"), str(cur.get("port") or 8080)))
            cur["path"] = _prompt_text(_copy(language, "webhook_path"), str(cur.get("path") or "/feishu/events"))
        next_config["feishu"] = cur
        return next_config

    if channel == "qq":
        cur = dict(config.get("qq") or {})
        print(f"\n{_copy(language, 'qq_config')}")
        cur["app_id"] = _prompt_text(_copy(language, "qq_app_id"), str(cur.get("app_id") or ""))
        cur["client_secret"] = _prompt_text(
            _copy(language, "qq_client_secret"),
            str(cur.get("client_secret") or ""),
            secret=True,
        )
        cur["image_host"] = _prompt_text(_copy(language, "qq_image_host"), str(cur.get("image_host") or "")) or None
        cur["image_server_port"] = int(
            _prompt_text(
                _copy(language, "qq_image_server_port"),
                str(cur.get("image_server_port") or 8081),
            )
        )
        cur["markdown"] = _prompt_bool(
            _copy(language, "qq_markdown"),
            bool(cur.get("markdown") or False),
            language=language,
        )
        next_config["qq"] = cur
        return next_config

    cur = dict(config.get("imessage") or {})
    detected_imsg = shutil.which("imsg") or str(cur.get("cli_path") or "imsg")
    print(f"\n{_copy(language, 'imessage_config')}")
    print(f"  {_copy(language, 'imessage_hint')}")
    cur["cli_path"] = _prompt_text(_copy(language, "imessage_cli_path"), detected_imsg)
    cur["db_path"] = _prompt_text(
        _copy(language, "imessage_db_path"),
        str(cur.get("db_path") or "~/Library/Messages/chat.db"),
    )
    cur["include_attachments"] = _prompt_bool(
        _copy(language, "imessage_include_attachments"),
        bool(cur.get("include_attachments") or False),
        language=language,
    )
    next_config["imessage"] = cur
    return next_config


def _prompt_oauth_manual_callback(auth_url: str, language: Language) -> str:
    print(f"\n{_copy(language, 'oauth_open_browser')}")
    print(_copy(language, "oauth_manual_hint"))
    print(f"\n{_copy(language, 'oauth_auth_url')}:\n{auth_url}\n")
    return input(f"{_copy(language, 'oauth_callback_prompt')}: ").strip()


def _prompt_saved_api_key(
    *,
    provider_name: str,
    auth_manager: OpenAIOAuthManager,
    language: Language,
) -> None:
    existing_key = _saved_provider_api_key(auth_manager.auth_path, provider_name)
    masked = _mask_secret(existing_key)
    print(f"\n{_copy(language, 'api_key_config')}")
    api_key = _prompt_text(
        _copy(language, "api_key_prompt").format(provider=_provider_label(provider_name, language)),
        masked,
        secret=True,
    )
    if api_key == masked:
        api_key = existing_key
    if not api_key:
        raise RuntimeError(
            _copy(language, "api_key_required").format(provider=_provider_label(provider_name, language))
        )
    _store_provider_api_key(auth_manager.auth_path, provider_name, api_key)


def _discovery_api_key(
    *,
    provider_name: str,
    auth_mode: str,
    auth_manager: OpenAIOAuthManager,
) -> Optional[str]:
    if auth_mode == "saved_api_key":
        return _saved_provider_api_key(auth_manager.auth_path, provider_name) or None
    if auth_mode == "environment":
        for env_name in provider_env_vars(provider_name):
            value = os.environ.get(env_name)
            if value:
                return value
        return None
    if auth_mode == "openai_oauth" and provider_name == "openai":
        try:
            return auth_manager.ensure_access_token(refresh_if_needed=True)
        except Exception:
            return None
    return None


def _discover_models_for_prompt(
    *,
    provider_name: str,
    auth_mode: str,
    endpoint: Optional[str],
    auth_manager: OpenAIOAuthManager,
) -> List[str]:
    api_key = _discovery_api_key(
        provider_name=provider_name,
        auth_mode=auth_mode,
        auth_manager=auth_manager,
    )
    discovered = discover_provider_models(
        provider_name,
        endpoint=endpoint,
        api_key=api_key,
    )
    return list(discovered.keys())


def _configure_llm(
    config: Dict[str, Any],
    auth_manager: OpenAIOAuthManager,
    language: Language,
) -> Dict[str, Any]:
    next_config = dict(config)
    previous_provider = _infer_provider(next_config)
    provider_name = _prompt_choice(
        _copy(language, "provider_title"),
        _provider_options(language),
        default=previous_provider,
        language=language,
    )
    next_config["llm_provider"] = provider_name

    current_mode = str(next_config.get("auth_mode") or "environment")
    if current_mode == "openai_api_key":
        current_mode = "saved_api_key"
    if current_mode == "openai_codex":
        current_mode = "openai_oauth"

    if provider_name == "python":
        next_config["auth_mode"] = "no_auth"
        next_config["endpoint"] = None
        next_config["model"] = "python"
        return next_config

    if provider_name == "openai":
        auth_mode = _prompt_choice(
            _copy(language, "auth_title"),
            [
                ("openai_oauth", _copy(language, "auth_openai_oauth")),
                ("saved_api_key", _copy(language, "auth_saved_api_key")),
                ("environment", _copy(language, "env_auth_format").format(env_vars="OPENAI_API_KEY")),
            ],
            default=(
                current_mode
                if previous_provider == provider_name and current_mode in {"openai_oauth", "saved_api_key", "environment"}
                else "openai_oauth"
            ),
            language=language,
        )
        next_config["auth_mode"] = auth_mode

        if auth_mode == "openai_oauth":
            auth_manager.login(prompt_for_redirect=lambda auth_url: _prompt_oauth_manual_callback(auth_url, language))
            next_config["endpoint"] = OPENAI_CODEX_DEFAULT_ENDPOINT
            default_model = _openai_model_default(str(next_config.get("model") or ""), auth_mode)
            next_config["model"] = _prompt_model(
                provider_name,
                default_model,
                language,
                current_model=str(next_config.get("model") or ""),
                discovered_models=_discover_models_for_prompt(
                    provider_name=provider_name,
                    auth_mode=auth_mode,
                    endpoint=next_config["endpoint"],
                    auth_manager=auth_manager,
                ),
                preferred_models=_OPENAI_CODEX_MODEL_CHOICES,
                include_registry=False,
            )
            return next_config

        next_config["endpoint"] = None
        if auth_mode == "saved_api_key":
            _prompt_saved_api_key(provider_name=provider_name, auth_manager=auth_manager, language=language)

        default_model = _openai_model_default(str(next_config.get("model") or ""), auth_mode)
        next_config["model"] = _prompt_model(
            provider_name,
            default_model,
            language,
            current_model=str(next_config.get("model") or ""),
            discovered_models=_discover_models_for_prompt(
                provider_name=provider_name,
                auth_mode=auth_mode,
                endpoint=None,
                auth_manager=auth_manager,
            ),
        )
        return next_config

    auth_options: List[Tuple[str, str]] = [
        ("saved_api_key", _copy(language, "auth_saved_api_key")),
        ("environment", _copy(language, "env_auth_format").format(env_vars=_provider_env_vars(provider_name))),
    ]
    default_auth_mode = "environment"
    if provider_name in {"ollama", "openai_compatible"}:
        auth_options.append(("no_auth", _copy(language, "auth_no_auth")))
        default_auth_mode = "no_auth" if provider_name == "ollama" else "saved_api_key"

    auth_mode = _prompt_choice(
        _copy(language, "auth_title"),
        auth_options,
        default=(
            current_mode
            if previous_provider == provider_name and current_mode in {value for value, _label in auth_options}
            else default_auth_mode
        ),
        language=language,
    )
    next_config["auth_mode"] = auth_mode

    if provider_name == "ollama":
        print(f"\n{_copy(language, 'ollama_hint')}")
        print(f"\n{_copy(language, 'endpoint_config')}")
        endpoint = _prompt_text(
            _copy(language, "endpoint_prompt"),
            str(next_config.get("endpoint") or provider_base_url("ollama") or OLLAMA_DEFAULT_ENDPOINT),
        )
        if not endpoint.strip():
            raise RuntimeError(_copy(language, "endpoint_required"))
        next_config["endpoint"] = endpoint.strip()
    elif provider_name == "openai_compatible":
        print(f"\n{_copy(language, 'openai_compatible_hint')}")
        print(f"\n{_copy(language, 'endpoint_config')}")
        endpoint = _prompt_text(
            _copy(language, "endpoint_prompt"),
            str(next_config.get("endpoint") or provider_base_url("openai_compatible") or OPENAI_COMPATIBLE_DEFAULT_ENDPOINT),
        )
        if not endpoint.strip():
            raise RuntimeError(_copy(language, "endpoint_required"))
        next_config["endpoint"] = endpoint.strip()
    else:
        next_config["endpoint"] = None

    if auth_mode == "saved_api_key":
        _prompt_saved_api_key(provider_name=provider_name, auth_manager=auth_manager, language=language)

    default_model = _model_default_for_provider(provider_name, str(next_config.get("model") or ""))
    next_config["model"] = _prompt_model(
        provider_name,
        default_model,
        language,
        current_model=str(next_config.get("model") or ""),
        discovered_models=_discover_models_for_prompt(
            provider_name=provider_name,
            auth_mode=auth_mode,
            endpoint=next_config.get("endpoint"),
            auth_manager=auth_manager,
        ),
    )
    return next_config


def run_setup_wizard(
    config: Dict[str, Any],
    *,
    auth_manager: OpenAIOAuthManager,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    next_config = dict(config)
    selected_language = _select_language(next_config, language)
    next_config["setup_language"] = selected_language

    print(f"\n{_copy(selected_language, 'wizard_title')}")
    print(f"{_copy(selected_language, 'wizard_intro')}\n")

    channel = _prompt_choice(
        _copy(selected_language, "channel_title"),
        [
            ("imessage", "iMessage"),
            ("telegram", "Telegram"),
            ("discord", "Discord"),
            ("feishu", "Feishu"),
            ("qq", "QQ"),
        ],
        default=str(next_config.get("channel") or "imessage"),
        language=selected_language,
    )
    next_config["channel"] = channel
    next_config = _prompt_channel_config(channel, next_config, selected_language)
    next_config = _configure_llm(next_config, auth_manager, selected_language)

    current_session_dir = next_config.get("session_dir")
    next_config["session_dir"] = _prompt_text(
        _copy(selected_language, "session_dir"),
        str(current_session_dir or ""),
    ) or None
    next_config["max_prompts"] = int(
        _prompt_text(_copy(selected_language, "max_prompts"), str(next_config.get("max_prompts") or 0))
    )
    return next_config
