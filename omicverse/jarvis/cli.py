"""
CLI entry point for ``omicverse jarvis``.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config_exists, default_auth_path, default_config_path, load_auth, load_config, save_config
from .model_registry import iter_known_provider_env_vars, provider_env_vars, provider_from_model
from .openai_oauth import OPENAI_CODEX_BASE_URL, OpenAIOAuthManager, OpenAIOAuthError


OLLAMA_DEFAULT_ENDPOINT = "http://127.0.0.1:11434/v1"
OPENAI_CODEX_DEFAULT_MODEL = "gpt-5.3-codex"
_LEGACY_OPENAI_CODEX_BASE_URL = "https://api.openai.com/v1"
_OPENAI_CODEX_MODELS = {
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omicverse jarvis",
        description=(
            "Launch OmicVerse Jarvis — multi-channel assistant for mobile single-cell analysis."
        ),
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run the interactive setup wizard before launch",
    )
    parser.add_argument(
        "--setup-language",
        default=None,
        dest="setup_language",
        choices=["en", "zh"],
        help="Wizard language: en or zh (default: English)",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        dest="config_file",
        help="Jarvis config file path (default: ~/.ovjarvis/config.json)",
    )
    parser.add_argument(
        "--auth-file",
        default=None,
        dest="auth_file",
        help="Jarvis auth state file path (default: ~/.ovjarvis/auth.json)",
    )
    parser.add_argument(
        "--channel",
        default=None,
        choices=["telegram", "discord", "feishu", "imessage", "qq"],
        help="Channel backend to run",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Telegram bot token (or set TELEGRAM_BOT_TOKEN env var)",
    )
    parser.add_argument(
        "--discord-token",
        default=None,
        dest="discord_token",
        help="Discord bot token (or set DISCORD_BOT_TOKEN env var)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name",
    )
    parser.add_argument(
        "--auth-mode",
        default=None,
        dest="auth_mode",
        choices=["environment", "openai_oauth", "openai_codex", "openai_api_key", "saved_api_key", "no_auth"],
        help="Authentication mode for saved Jarvis config",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        dest="api_key",
        help=(
            "LLM API key. Falls back to saved provider auth or the provider-specific "
            "environment variable selected in setup."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        dest="endpoint",
        help="Custom LLM API base URL, for example Ollama or another OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        dest="max_prompts",
        help="Max LLM prompts per kernel session before auto-restart; 0 disables auto-restart",
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        dest="session_dir",
        help="Directory to store per-user workspaces (default: ~/.ovjarvis)",
    )
    parser.add_argument(
        "--allowed-user",
        action="append",
        default=None,
        dest="allowed_users",
        metavar="USER",
        help=(
            "Allowed Telegram username or numeric ID. "
            "Repeat for multiple users. Omit to allow everyone."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    # ── Code-generation / claw mode ──────────────────────────────────────
    claw_group = parser.add_argument_group(
        "code generation (claw mode)",
        "These flags switch to one-shot code generation instead of starting the bot.",
    )
    claw_group.add_argument(
        "-q", "--question",
        nargs="*",
        dest="question",
        default=None,
        metavar="WORD",
        help=(
            "Ask a one-shot natural-language question. "
            "Uses the same model / api-key / endpoint resolved from Jarvis config. "
            "When omitted the bot is started normally."
        ),
    )
    claw_group.add_argument(
        "--output",
        default=None,
        dest="claw_output",
        metavar="FILE",
        help="Write generated code to FILE instead of stdout.",
    )
    claw_group.add_argument(
        "--max-functions",
        type=int,
        default=8,
        dest="claw_max_functions",
        metavar="N",
        help="Max registry functions included in the code-gen prompt (default: 8).",
    )
    claw_group.add_argument(
        "--no-reflection",
        action="store_true",
        dest="claw_no_reflection",
        help="Skip the review pass and return first-pass generated code.",
    )
    claw_group.add_argument(
        "--debug-registry",
        action="store_true",
        dest="claw_debug_registry",
        help="Print matched registry entries to stderr before code generation.",
    )
    # ── Gateway (web launcher) ────────────────────────────────────────────
    gw_group = parser.add_argument_group(
        "gateway (web launcher)",
        "Launch the OmicVerse web UI alongside the channel bot (unified gateway mode).",
    )
    gw_group.add_argument(
        "--with-web",
        action="store_true",
        dest="with_web",
        help="Start the OmicVerse web server in the background when the bot starts.",
    )
    gw_group.add_argument(
        "--gateway-daemon",
        action="store_true",
        dest="gateway_daemon",
        help=argparse.SUPPRESS,
    )
    gw_group.add_argument(
        "--web-port",
        type=int,
        default=0,
        dest="web_port",
        metavar="PORT",
        help="Web server port (default: auto-select from 5050).",
    )
    gw_group.add_argument(
        "--web-host",
        default="127.0.0.1",
        dest="web_host",
        metavar="HOST",
        help="Web server bind host (default: 127.0.0.1).",
    )
    gw_group.add_argument(
        "--no-browser",
        action="store_true",
        dest="no_browser",
        help="Do not automatically open the browser when the web server starts.",
    )
    gw_group.add_argument(
        "--codex-login",
        action="store_true",
        dest="codex_login",
        help=(
            "Force a fresh OpenAI Codex OAuth login before starting. "
            "Use this to switch to a different Codex account."
        ),
    )
    # ── Daemon control (claw daemon) ──────────────────────────────────────
    daemon_group = parser.add_argument_group(
        "daemon control (claw daemon)",
        "Manage the persistent claw daemon that keeps OmicVerse pre-imported.",
    )
    daemon_group.add_argument(
        "--daemon",
        action="store_true",
        dest="claw_daemon",
        help="Start a persistent claw daemon process.",
    )
    daemon_group.add_argument(
        "--use-daemon",
        action="store_true",
        dest="claw_use_daemon",
        help="Send the -q request to a running claw daemon instead of a fresh process.",
    )
    daemon_group.add_argument(
        "--stop-daemon",
        action="store_true",
        dest="claw_stop_daemon",
        help="Ask the running claw daemon to stop.",
    )
    daemon_group.add_argument(
        "--socket",
        default=None,
        dest="claw_socket",
        metavar="PATH",
        help="Unix socket path for claw daemon communication.",
    )
    parser.add_argument(
        "--feishu-app-id",
        default=None,
        dest="feishu_app_id",
        help="Feishu app_id (or FEISHU_APP_ID env var)",
    )
    parser.add_argument(
        "--feishu-app-secret",
        default=None,
        dest="feishu_app_secret",
        help="Feishu app_secret (or FEISHU_APP_SECRET env var)",
    )
    parser.add_argument(
        "--feishu-host",
        default=None,
        dest="feishu_host",
        help="Feishu webhook bind host",
    )
    parser.add_argument(
        "--feishu-port",
        type=int,
        default=None,
        dest="feishu_port",
        help="Feishu webhook bind port",
    )
    parser.add_argument(
        "--feishu-path",
        default=None,
        dest="feishu_path",
        help="Feishu webhook path",
    )
    parser.add_argument(
        "--feishu-connection-mode",
        default=None,
        choices=["webhook", "websocket"],
        dest="feishu_connection_mode",
        help="Feishu connection mode: webhook or websocket",
    )
    parser.add_argument(
        "--feishu-verification-token",
        default=None,
        dest="feishu_verification_token",
        help="Feishu webhook verification token (or FEISHU_VERIFICATION_TOKEN env var)",
    )
    parser.add_argument(
        "--feishu-encrypt-key",
        default=None,
        dest="feishu_encrypt_key",
        help="Feishu webhook encrypt key (or FEISHU_ENCRYPT_KEY env var)",
    )
    parser.add_argument(
        "--imessage-cli-path",
        default=None,
        dest="imessage_cli_path",
        help="Path to the imsg CLI binary or wrapper script",
    )
    parser.add_argument(
        "--imessage-db-path",
        default=None,
        dest="imessage_db_path",
        help="Path to Messages chat.db (default: ~/Library/Messages/chat.db)",
    )
    parser.add_argument(
        "--imessage-include-attachments",
        action="store_true",
        default=None,
        dest="imessage_include_attachments",
        help="Subscribe with iMessage attachment metadata enabled",
    )
    # QQ Bot arguments
    parser.add_argument(
        "--qq-app-id",
        default=None,
        dest="qq_app_id",
        help="QQ Bot AppID (or QQ_APP_ID env var)",
    )
    parser.add_argument(
        "--qq-client-secret",
        default=None,
        dest="qq_client_secret",
        help="QQ Bot ClientSecret / AppSecret (or QQ_CLIENT_SECRET env var)",
    )
    parser.add_argument(
        "--qq-image-host",
        default=None,
        dest="qq_image_host",
        help=(
            "Public base URL for serving analysis figures to QQ "
            "(e.g. http://YOUR_IP:8081). Required to send figures. "
            "Also set via QQ_IMAGE_HOST env var."
        ),
    )
    parser.add_argument(
        "--qq-image-server-port",
        type=int,
        default=None,
        dest="qq_image_server_port",
        help="Local port for QQ image hosting server",
    )
    parser.add_argument(
        "--qq-markdown",
        action="store_true",
        default=None,
        dest="qq_markdown",
        help=(
            "Enable markdown reply format for QQ Bot (msg_type=2). "
            "Requires bot to have markdown message permission on QQ Open Platform."
        ),
    )
    return parser


def _configured_env_api_keys() -> bool:
    for env_name in iter_known_provider_env_vars():
        if os.environ.get(env_name):
            return True
    return False


def _has_bootstrap_credentials(args: argparse.Namespace) -> bool:
    return bool(
        args.token
        or os.environ.get("TELEGRAM_BOT_TOKEN")
        or args.discord_token
        or os.environ.get("DISCORD_BOT_TOKEN")
        or (
            (args.feishu_app_id or os.environ.get("FEISHU_APP_ID"))
            and (args.feishu_app_secret or os.environ.get("FEISHU_APP_SECRET"))
        )
        or (
            (args.qq_app_id or os.environ.get("QQ_APP_ID"))
            and (args.qq_client_secret or os.environ.get("QQ_CLIENT_SECRET"))
        )
        or args.api_key
        or args.endpoint
        or _configured_env_api_keys()
    )


def _should_run_setup(
    args: argparse.Namespace,
    argv: Optional[List[str]],
    config_path: Path,
) -> bool:
    if args.setup:
        return True
    if config_exists(config_path):
        return False
    if not sys.stdin.isatty():
        return False
    if argv:
        return False
    return not _has_bootstrap_credentials(args)


def _run_setup(args: argparse.Namespace, config_path: Path, auth_path: Path) -> Dict[str, Any]:
    from .setup_wizard import run_setup_wizard

    config = load_config(config_path)
    auth_manager = OpenAIOAuthManager(auth_path)
    updated = run_setup_wizard(
        config,
        auth_manager=auth_manager,
        language=args.setup_language,
    )
    save_config(updated, config_path)
    print(f"Jarvis configuration saved to {config_path}")
    return updated


def _resolve_value(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _normalize_auth_mode(auth_mode: Optional[str]) -> str:
    if auth_mode == "openai_api_key":
        return "saved_api_key"
    if auth_mode == "openai_codex":
        return "openai_oauth"
    return str(auth_mode or "environment")


def _resolve_provider(
    *,
    config: Dict[str, Any],
    model: str,
    cli_model: Optional[str] = None,
) -> str:
    if str(cli_model or "").strip():
        return provider_from_model(str(cli_model).strip())
    explicit = str(config.get("llm_provider") or "").strip()
    if explicit:
        return explicit
    return provider_from_model(model)


def _resolve_provider_env_api_key(provider_name: str, model: str) -> Optional[str]:
    lookup_name = provider_name
    if lookup_name in {"ollama", "openai_compatible"}:
        lookup_name = "openai"
    elif lookup_name == "python":
        return None
    elif not provider_env_vars(lookup_name):
        lookup_name = provider_from_model(model)

    for env_name in provider_env_vars(lookup_name):
        if env_name:
            value = os.environ.get(env_name)
            if value:
                return value
    return None


def _resolve_saved_api_key(provider_name: str, auth_path: Path) -> Optional[str]:
    auth = load_auth(auth_path)
    providers = dict(auth.get("providers") or {})

    if provider_name == "openai":
        saved = str(auth.get("OPENAI_API_KEY") or "").strip()
        if saved:
            return saved

    entry = dict(providers.get(provider_name) or {})
    saved = str(entry.get("api_key") or "").strip()
    return saved or None


def _placeholder_api_key(provider_name: str, endpoint: Optional[str]) -> Optional[str]:
    if provider_name == "python":
        return None
    if provider_name == "ollama":
        return "ollama"
    if endpoint:
        return "jarvis-local"
    return None


def _resolve_endpoint(
    *,
    cli_endpoint: Optional[str],
    config_endpoint: Optional[str],
    llm_provider: str,
    auth_mode: Optional[str] = None,
) -> Optional[str]:
    normalized_mode = _normalize_auth_mode(auth_mode)
    endpoint = _resolve_value(cli_endpoint, config_endpoint)
    endpoint = str(endpoint or "").strip() or None
    if endpoint and endpoint.rstrip("/") == _LEGACY_OPENAI_CODEX_BASE_URL:
        endpoint = OPENAI_CODEX_BASE_URL if llm_provider == "openai" and normalized_mode == "openai_oauth" else None
    if endpoint:
        if (
            cli_endpoint is None
            and llm_provider == "openai"
            and normalized_mode != "openai_oauth"
            and endpoint.rstrip("/") == OPENAI_CODEX_BASE_URL
        ):
            return None
        return endpoint
    if llm_provider == "ollama":
        return OLLAMA_DEFAULT_ENDPOINT
    if llm_provider == "openai" and normalized_mode == "openai_oauth":
        return OPENAI_CODEX_BASE_URL
    return None


def _is_openai_codex_model(model: str) -> bool:
    return str(model or "").strip().lower() in {name.lower() for name in _OPENAI_CODEX_MODELS}


def _resolve_model(
    *,
    cli_model: Optional[str],
    config_model: Optional[str],
    llm_provider: str,
    auth_mode: Optional[str],
) -> str:
    model = str(_resolve_value(cli_model, config_model, "claude-sonnet-4-6") or "").strip()
    if llm_provider == "openai" and _normalize_auth_mode(auth_mode) == "openai_oauth":
        return model if _is_openai_codex_model(model) else OPENAI_CODEX_DEFAULT_MODEL
    return model or "claude-sonnet-4-6"


def _resolve_api_key(
    *,
    cli_key: Optional[str],
    model: str,
    llm_provider: str,
    endpoint: Optional[str],
    auth_mode: str,
    auth_path: Path,
) -> Optional[str]:
    if cli_key:
        return cli_key

    normalized_mode = _normalize_auth_mode(auth_mode)
    provider = llm_provider or "openai"

    if normalized_mode == "no_auth":
        return _placeholder_api_key(provider, endpoint)

    if normalized_mode == "environment":
        return _resolve_provider_env_api_key(provider, model)

    if normalized_mode == "openai_oauth":
        try:
            manager = OpenAIOAuthManager(auth_path)
            api_key = manager.ensure_access_token(refresh_if_needed=True)
        except OpenAIOAuthError as exc:
            raise RuntimeError(f"Failed to load saved OpenAI auth: {exc}") from exc
        return api_key

    if normalized_mode == "saved_api_key":
        return _resolve_saved_api_key(provider, auth_path)

    return None


def _web_only_loop(mode_label: str = "web-only") -> int:
    """Block until Ctrl-C while the gateway/web server is running."""
    print(f"Gateway running in {mode_label} mode. Press Ctrl+C to stop.")
    import time as _time
    try:
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        print("\nGateway stopped.")
    return 0


def _web_project_candidates() -> List[Path]:
    base_candidates = [
        Path(__file__).resolve().parents[3] / "omicverse-project",
        Path.home() / "Desktop" / "analysis" / "omicverse-project",
    ]
    project_names = ["omicclaw", "omicverse-web"]
    candidates: List[Path] = []
    for base in base_candidates:
        for project_name in project_names:
            candidates.append(base / project_name)
    return candidates


def _resolve_gateway_web_root() -> Optional[Path]:
    candidates = _web_project_candidates()
    for candidate in candidates:
        if (candidate / "gateway" / "server.py").exists() and (candidate / "services").exists():
            return candidate
    return None


def _import_installed_web_runtime():
    errors: List[str] = []
    for package_name in ("omicclaw", "omicverse_web"):
        try:
            gateway_mod = importlib.import_module(f"{package_name}.gateway.server")
            session_mod = importlib.import_module(f"{package_name}.services.agent_session_service")
            registry_mod = importlib.import_module(f"{package_name}.gateway.registry")
            return (
                gateway_mod.GatewayServer,
                session_mod.SessionManager,
                registry_mod.GatewayChannelRegistry,
            )
        except ImportError as exc:
            errors.append(f"{package_name}: {exc}")
    raise ImportError("; ".join(errors))


def _gateway_daemon_loop(
    *,
    web_host: str,
    web_port: int,
    no_browser: bool,
    config_path: Path,
    session_dir: Optional[str],
    model: str,
    api_key: Optional[str],
    endpoint: Optional[str],
    max_prompts: int,
    verbose: bool,
) -> int:
    """Start the daemon-style gateway: web UI plus auto-started channels."""
    try:
        gateway_web_root = _resolve_gateway_web_root()
        if gateway_web_root is None:
            raise ImportError("Cannot locate omicverse-project/omicclaw or legacy omicverse-web")
        gateway_web_root_str = str(gateway_web_root)
        if gateway_web_root_str not in sys.path:
            sys.path.insert(0, gateway_web_root_str)
        from gateway.server import GatewayServer  # type: ignore
        from services.agent_session_service import SessionManager as WebSM  # type: ignore
        from gateway.inprocess_channel_manager import InProcessChannelManager  # type: ignore
        from .gateway.web_bridge import WebSessionBridge
        from .memory.store import MemoryStore as _MemStore
    except ImportError as exc:
        print(
            "WARNING: gateway daemon mode requires OmicClaw to be installed. "
            "Legacy omicverse-web is still supported.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    web_sm = WebSM(max_sessions=20)
    mem_db = os.path.join(os.path.expanduser(session_dir or "~/.ovjarvis"), "memory.db")
    try:
        mem_store = _MemStore(mem_db)
    except Exception:
        mem_store = None
    gateway_web_bridge = WebSessionBridge(web_sm, memory_store=mem_store)

    from .session import SessionManager

    jarvis_sm = SessionManager(
        session_dir=session_dir,
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        max_prompts=max_prompts,
        verbose=verbose,
        gateway_web_bridge=gateway_web_bridge,
        shared_kernel=True,
    )
    try:
        web_sm.set_adata_sync_handler(lambda _session_id, adata: jarvis_sm.set_shared_adata(adata))
    except Exception:
        pass
    channel_manager = InProcessChannelManager(jarvis_sm)
    gw = GatewayServer()
    _gw_thread, _gw_url = gw.start(
        host=web_host,
        port=web_port,
        session_manager=web_sm,
        channel_manager=channel_manager,
        memory_db_path=mem_db,
        auto_start_channels=True,
        jarvis_config_path=str(config_path),
    )
    deadline = time.time() + 10.0
    while time.time() < deadline:
        web_app = sys.modules.get("app")
        if web_app is None:
            try:
                web_app = importlib.import_module("app")
            except Exception:
                web_app = None
        web_state = getattr(web_app, "state", None) if web_app is not None else None
        web_kernel_executor = getattr(web_state, "kernel_executor", None) if web_state is not None else None
        if web_kernel_executor is not None:
            jarvis_sm.attach_shared_kernel_executor(web_kernel_executor)
            current_adata = getattr(web_state, "current_adata", None)
            if current_adata is not None:
                jarvis_sm.set_shared_adata(current_adata)
            break
        time.sleep(0.1)
    print(f"OmicClaw Gateway ready at {_gw_url}")
    if not no_browser:
        gw.open_browser()
    return _web_only_loop("gateway daemon")


def main(argv: Optional[List[str]] = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(effective_argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config_path = Path(os.path.expanduser(args.config_file)) if args.config_file else default_config_path()
    auth_path = Path(os.path.expanduser(args.auth_file)) if args.auth_file else default_auth_path()

    if not args.setup and _should_run_setup(args, effective_argv, config_path):
        try:
            _run_setup(args, config_path, auth_path)
        except (OpenAIOAuthError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    config = load_config(config_path)
    if args.setup:
        try:
            config = _run_setup(args, config_path, auth_path)
        except (OpenAIOAuthError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if getattr(args, "codex_login", False):
        print("Starting OpenAI Codex account login...")
        try:
            auth_manager = OpenAIOAuthManager(auth_path)
            auth_manager.login(
                prompt_for_redirect=lambda auth_url: (
                    print(f"\nAuthorization URL:\n  {auth_url}\n"),
                    input("Paste the callback URL (or code#state): "),
                )[-1],
            )
            print("Codex login successful.")
            config["auth_mode"] = "openai_oauth"
            config["llm_provider"] = "openai"
            if not config.get("model") or config["model"] not in _OPENAI_CODEX_MODELS:
                config["model"] = OPENAI_CODEX_DEFAULT_MODEL
            if not config.get("endpoint"):
                config["endpoint"] = OPENAI_CODEX_BASE_URL
            save_config(config, config_path)
        except OpenAIOAuthError as exc:
            print(f"ERROR: Codex login failed: {exc}", file=sys.stderr)
            return 1

    channel = _resolve_value(args.channel, config.get("channel"), "telegram")
    auth_mode = _resolve_value(args.auth_mode, config.get("auth_mode"), "environment")
    raw_model = str(_resolve_value(args.model, config.get("model"), "claude-sonnet-4-6") or "")
    llm_provider = _resolve_provider(config=config, model=raw_model, cli_model=args.model)
    model = _resolve_model(
        cli_model=args.model,
        config_model=config.get("model"),
        llm_provider=llm_provider,
        auth_mode=auth_mode,
    )
    endpoint = _resolve_endpoint(
        cli_endpoint=args.endpoint,
        config_endpoint=config.get("endpoint"),
        llm_provider=llm_provider,
        auth_mode=auth_mode,
    )
    session_dir = _resolve_value(args.session_dir, config.get("session_dir"))
    max_prompts = _resolve_value(args.max_prompts, config.get("max_prompts"), 0)

    def _do_resolve_api_key() -> Optional[str]:
        try:
            key = _resolve_api_key(
                cli_key=args.api_key,
                model=model,
                llm_provider=llm_provider,
                endpoint=endpoint,
                auth_mode=auth_mode,
                auth_path=auth_path,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return None
        # Always fall back to auth.json regardless of auth_mode —
        # the setup wizard saves keys there even when auth_mode="environment".
        if key is None:
            key = _resolve_saved_api_key(llm_provider, auth_path)
        return key

    api_key = _do_resolve_api_key()

    if getattr(args, "gateway_daemon", False):
        return _gateway_daemon_loop(
            web_host=getattr(args, "web_host", "127.0.0.1"),
            web_port=getattr(args, "web_port", 0),
            no_browser=getattr(args, "no_browser", False),
            config_path=config_path,
            session_dir=session_dir,
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            max_prompts=max_prompts,
            verbose=args.verbose,
        )

    # ── Auto-setup if model or api_key is not configured ─────────────────
    _key_not_needed = auth_mode in {"no_auth"} or llm_provider in {"python", "ollama"}
    _model_missing = not config.get("model") and args.model is None
    _key_missing = api_key is None and not _key_not_needed

    # Only trigger for meaningful operations (not --stop-daemon which needs no key)
    _needs_credentials = not args.claw_stop_daemon

    if (_model_missing or (_key_missing and _needs_credentials)) and not args.setup:
        if not sys.stdin.isatty():
            if _model_missing:
                print("ERROR: No model configured. Run with --setup to configure.", file=sys.stderr)
            else:
                print("ERROR: API key could not be resolved. Run with --setup to configure.", file=sys.stderr)
            return 1

        reason = "No model configured." if _model_missing else "API key could not be resolved."
        print(f"⚠  {reason} Launching setup wizard...\n")
        try:
            config = _run_setup(args, config_path, auth_path)
        except (OpenAIOAuthError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        # Re-resolve everything with the updated config
        auth_mode = _resolve_value(args.auth_mode, config.get("auth_mode"), "environment")
        raw_model = str(_resolve_value(args.model, config.get("model"), "claude-sonnet-4-6") or "")
        llm_provider = _resolve_provider(config=config, model=raw_model, cli_model=args.model)
        model = _resolve_model(
            cli_model=args.model,
            config_model=config.get("model"),
            llm_provider=llm_provider,
            auth_mode=auth_mode,
        )
        endpoint = _resolve_endpoint(
            cli_endpoint=args.endpoint,
            config_endpoint=config.get("endpoint"),
            llm_provider=llm_provider,
            auth_mode=auth_mode,
        )
        api_key = _do_resolve_api_key()

    # ── Claw mode: daemon control or one-shot question ────────────────────
    _claw_mode = (
        args.question is not None
        or args.claw_daemon
        or args.claw_use_daemon
        or args.claw_stop_daemon
    )

    if _claw_mode:
        from omicverse import claw as _claw_mod

        claw_argv: List[str] = []

        # Daemon control flags (mutually exclusive with -q in practice)
        if args.claw_daemon:
            claw_argv.append("--daemon")
        if args.claw_use_daemon:
            claw_argv.append("--use-daemon")
        if args.claw_stop_daemon:
            claw_argv.append("--stop-daemon")
        if args.claw_socket:
            claw_argv += ["--socket", args.claw_socket]

        # Question (positional in claw)
        if args.question is not None:
            question_text = " ".join(args.question).strip()
            if not question_text and not args.claw_daemon and not args.claw_stop_daemon:
                print("ERROR: -q requires a non-empty question.", file=sys.stderr)
                return 1
            claw_argv += list(args.question)

        # Model / auth (resolved from jarvis config, passed through)
        if model:
            claw_argv += ["--model", model]
        if api_key:
            claw_argv += ["--api-key", api_key]
        if endpoint:
            claw_argv += ["--endpoint", endpoint]

        # Code-gen tuning flags
        if args.claw_output:
            claw_argv += ["--output", args.claw_output]
        if args.claw_max_functions != 8:
            claw_argv += ["--max-functions", str(args.claw_max_functions)]
        if args.claw_no_reflection:
            claw_argv.append("--no-reflection")
        if args.claw_debug_registry:
            claw_argv.append("--debug-registry")

        return _claw_mod.main(claw_argv)

    from .session import SessionManager

    # ── Gateway mode: start web server + build bridge ─────────────────────
    _gateway_web_bridge = None
    _web_sm = None
    if getattr(args, "with_web", False):
        try:
            GatewayServer, WebSM, GatewayChannelRegistry = _import_installed_web_runtime()
            from .gateway.web_bridge import WebSessionBridge

            _web_sm = WebSM(max_sessions=20)
            _web_registry = GatewayChannelRegistry()
            _gw = GatewayServer()
            _mem_db = os.path.join(os.path.expanduser(session_dir or "~/.ovjarvis"), "memory.db")
            # Collect channel names configured at startup for the web status page
            _startup_channels = []
            _ch_arg = getattr(args, "channel", None) or config.get("channel")
            if _ch_arg:
                _startup_channels = [_ch_arg] if isinstance(_ch_arg, str) else list(_ch_arg)
            _gw_thread, _gw_url = _gw.start(
                host=getattr(args, "web_host", "127.0.0.1"),
                port=getattr(args, "web_port", 0),
                session_manager=_web_sm,
                channel_registry=_web_registry,
                memory_db_path=_mem_db,
                channels=_startup_channels or None,
            )
            print(f"OmicClaw Web ready at {_gw_url}")
            if not getattr(args, "no_browser", False):
                _gw.open_browser()
            try:
                from omicverse.jarvis.memory.store import MemoryStore as _MemStore
                _mem_store = _MemStore(_mem_db)
            except Exception:
                _mem_store = None
            _gateway_web_bridge = WebSessionBridge(_web_sm, memory_store=_mem_store)
        except ImportError as _gw_err:
            print(
                f"WARNING: --with-web requires omicclaw to be installed. "
                f"Legacy omicverse-web is still supported. "
                f"({_gw_err})",
                file=sys.stderr,
            )

    sm = SessionManager(
        session_dir=session_dir,
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        max_prompts=max_prompts,
        verbose=args.verbose,
        gateway_web_bridge=_gateway_web_bridge,
    )

    if _web_sm is not None:
        try:
            shared_adata = _web_sm.get_shared_adata()
        except Exception:
            shared_adata = None
        if shared_adata is not None:
            sm.set_shared_adata(shared_adata)
        try:
            _web_sm.set_adata_sync_handler(
                lambda _session_id, adata: sm.set_shared_adata(adata)
            )
        except Exception:
            pass

    # ── Gateway-only mode: --with-web without an explicit --channel ───────
    # If the user ran `omicverse gateway` (or `--with-web`) without specifying
    # a channel, we don't require any bot credentials — just keep the web
    # server running until Ctrl-C.
    _channel_explicitly_set = bool(args.channel) or bool(config.get("channel"))
    if getattr(args, "with_web", False) and not _channel_explicitly_set:
        print("Gateway running — no channel bot configured. Press Ctrl+C to stop.")
        try:
            import time as _time
            while True:
                _time.sleep(1)
        except KeyboardInterrupt:
            print("\nGateway stopped.")
        return 0

    if channel == "telegram":
        telegram_cfg = dict(config.get("telegram") or {})
        token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN") or telegram_cfg.get("token")
        allowed_users = (
            args.allowed_users
            if args.allowed_users is not None
            else list(telegram_cfg.get("allowed_users") or [])
        )
        if not token:
            msg = (
                "ERROR: Telegram bot token is required.\n"
                "  Pass --token, run `omicverse jarvis --setup`, or set TELEGRAM_BOT_TOKEN."
            )
            if getattr(args, "with_web", False):
                print(f"WARNING: {msg}\n  Running in web-only gateway mode.", file=sys.stderr)
                return _web_only_loop()
            print(msg, file=sys.stderr)
            return 1
        from .channels.telegram import AccessControl, run_bot

        ac = AccessControl(allowed=allowed_users or None)
        print(f"OmicVerse Jarvis starting (channel=telegram, model={model}) ...")
        try:
            run_bot(token=token, session_manager=sm, access_control=ac, verbose=args.verbose)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if channel == "discord":
        discord_cfg = dict(config.get("discord") or {})
        discord_token = args.discord_token or os.environ.get("DISCORD_BOT_TOKEN") or discord_cfg.get("token")
        if not discord_token:
            msg = (
                "ERROR: Discord bot token is required.\n"
                "  Pass --discord-token, run `omicverse jarvis --setup`, or set DISCORD_BOT_TOKEN."
            )
            if getattr(args, "with_web", False):
                print(f"WARNING: {msg}\n  Running in web-only gateway mode.", file=sys.stderr)
                return _web_only_loop()
            print(msg, file=sys.stderr)
            return 1
        from .channels.discord import run_discord_bot

        print(f"OmicVerse Jarvis starting (channel=discord, model={model}) ...")
        try:
            run_discord_bot(
                token=discord_token,
                session_manager=sm,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if channel == "imessage":
        imessage_cfg = dict(config.get("imessage") or {})
        cli_path = (
            args.imessage_cli_path
            or os.environ.get("IMESSAGE_CLI_PATH")
            or imessage_cfg.get("cli_path")
            or "imsg"
        )
        db_path = (
            args.imessage_db_path
            or os.environ.get("IMESSAGE_DB_PATH")
            or imessage_cfg.get("db_path")
            or "~/Library/Messages/chat.db"
        )
        include_attachments = (
            args.imessage_include_attachments
            if args.imessage_include_attachments is not None
            else bool(imessage_cfg.get("include_attachments") or False)
        )
        from .channels.imessage import run_imessage_bot

        print(
            f"OmicVerse Jarvis starting (channel=imessage, model={model}, "
            f"cli={cli_path}, attachments={'on' if include_attachments else 'off'}) ..."
        )
        run_imessage_bot(
            session_manager=sm,
            cli_path=cli_path,
            db_path=db_path,
            include_attachments=include_attachments,
        )
        return 0

    if channel == "qq":
        qq_cfg = dict(config.get("qq") or {})
        qq_app_id = args.qq_app_id or os.environ.get("QQ_APP_ID") or qq_cfg.get("app_id")
        qq_client_secret = (
            args.qq_client_secret or os.environ.get("QQ_CLIENT_SECRET") or qq_cfg.get("client_secret")
        )
        if not qq_app_id or not qq_client_secret:
            msg = (
                "ERROR: QQ Bot credentials are required.\n"
                "  Pass --qq-app-id/--qq-client-secret, run `omicverse jarvis --setup`, "
                "or set QQ_APP_ID/QQ_CLIENT_SECRET."
            )
            if getattr(args, "with_web", False):
                print(f"WARNING: {msg}\n  Running in web-only gateway mode.", file=sys.stderr)
                return _web_only_loop()
            print(msg, file=sys.stderr)
            return 1
        qq_image_host = args.qq_image_host or os.environ.get("QQ_IMAGE_HOST") or qq_cfg.get("image_host")
        qq_image_server_port = int(
            _resolve_value(args.qq_image_server_port, qq_cfg.get("image_server_port"), 8081)
        )
        qq_markdown = bool(_resolve_value(args.qq_markdown, qq_cfg.get("markdown"), False))
        from .channels.qq import run_qq_bot

        print(
            f"OmicVerse Jarvis starting (channel=qq, model={model}, "
            f"markdown={'on' if qq_markdown else 'off'}, "
            f"image_host={'set' if qq_image_host else 'not configured'}) ..."
        )
        run_qq_bot(
            app_id=qq_app_id,
            client_secret=qq_client_secret,
            session_manager=sm,
            markdown=qq_markdown,
            image_host=qq_image_host,
            image_server_port=qq_image_server_port,
        )
        return 0

    feishu_cfg = dict(config.get("feishu") or {})
    app_id = args.feishu_app_id or os.environ.get("FEISHU_APP_ID") or feishu_cfg.get("app_id")
    app_secret = (
        args.feishu_app_secret or os.environ.get("FEISHU_APP_SECRET") or feishu_cfg.get("app_secret")
    )
    verification_token = (
        args.feishu_verification_token
        or os.environ.get("FEISHU_VERIFICATION_TOKEN")
        or feishu_cfg.get("verification_token")
    )
    encrypt_key = (
        args.feishu_encrypt_key
        or os.environ.get("FEISHU_ENCRYPT_KEY")
        or feishu_cfg.get("encrypt_key")
    )
    connection_mode = _resolve_value(
        args.feishu_connection_mode,
        feishu_cfg.get("connection_mode"),
        "websocket",
    )
    host = _resolve_value(args.feishu_host, feishu_cfg.get("host"), "0.0.0.0")
    port = int(_resolve_value(args.feishu_port, feishu_cfg.get("port"), 8080))
    path = _resolve_value(args.feishu_path, feishu_cfg.get("path"), "/feishu/events")

    if not app_id or not app_secret:
        msg = (
            "ERROR: Feishu app credentials are required.\n"
            "  Pass --feishu-app-id/--feishu-app-secret, run `omicverse jarvis --setup`, "
            "or set FEISHU_APP_ID/FEISHU_APP_SECRET."
        )
        if getattr(args, "with_web", False):
            print(f"WARNING: {msg}\n  Running in web-only gateway mode.", file=sys.stderr)
            return _web_only_loop()
        print(msg, file=sys.stderr)
        return 1
    from .channels.feishu import run_feishu_bot, run_feishu_ws_bot

    if connection_mode == "websocket":
        print(
            f"OmicVerse Jarvis starting (channel=feishu, model={model}, mode=websocket) ..."
        )
        run_feishu_ws_bot(
            app_id=app_id,
            app_secret=app_secret,
            session_manager=sm,
            verification_token=verification_token,
            encrypt_key=encrypt_key,
        )
    else:
        print(
            f"OmicVerse Jarvis starting (channel=feishu, model={model}, "
            f"mode=webhook, listen={host}:{port}{path}) ..."
        )
        run_feishu_bot(
            app_id=app_id,
            app_secret=app_secret,
            session_manager=sm,
            host=host,
            port=port,
            path=path,
            verification_token=verification_token,
            encrypt_key=encrypt_key,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
