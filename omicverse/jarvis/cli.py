"""
CLI entry point for ``omicverse jarvis``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omicverse jarvis",
        description=(
            "Launch OmicVerse Jarvis — multi-channel assistant for mobile single-cell analysis."
        ),
    )
    parser.add_argument(
        "--channel",
        default="telegram",
        choices=["telegram", "feishu"],
        help="Channel backend to run (default: telegram)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Telegram bot token (or set TELEGRAM_BOT_TOKEN env var)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="LLM model name (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        dest="api_key",
        help=(
            "LLM API key.  Falls back to ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY / GEMINI_API_KEY env vars."
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        dest="max_prompts",
        help="Max LLM prompts per kernel session before auto-restart; 0 disables auto-restart (default: 0)",
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
        default=[],
        dest="allowed_users",
        metavar="USER",
        help=(
            "Allowed Telegram username or numeric ID.  "
            "Repeat for multiple users.  Omit to allow everyone."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
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
        default="0.0.0.0",
        dest="feishu_host",
        help="Feishu webhook bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--feishu-port",
        type=int,
        default=8080,
        dest="feishu_port",
        help="Feishu webhook bind port (default: 8080)",
    )
    parser.add_argument(
        "--feishu-path",
        default="/feishu/events",
        dest="feishu_path",
        help="Feishu webhook path (default: /feishu/events)",
    )
    parser.add_argument(
        "--feishu-connection-mode",
        default="websocket",
        choices=["webhook", "websocket"],
        dest="feishu_connection_mode",
        help="Feishu connection mode: webhook or websocket (default: websocket)",
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
    return parser


def _resolve_api_key(cli_key: Optional[str]) -> Optional[str]:
    if cli_key:
        return cli_key
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    api_key = _resolve_api_key(args.api_key)

    from .session import SessionManager

    sm = SessionManager(
        session_dir=args.session_dir,
        model=args.model,
        api_key=api_key,
        max_prompts=args.max_prompts,
        verbose=args.verbose,
    )

    if args.channel == "telegram":
        token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            print(
                "ERROR: Telegram bot token is required.\n"
                "  Pass --token or set TELEGRAM_BOT_TOKEN env var.",
                file=sys.stderr,
            )
            return 1
        from .channels.telegram import AccessControl, run_bot

        ac = AccessControl(allowed=args.allowed_users or None)
        print(f"OmicVerse Jarvis starting (channel=telegram, model={args.model}) ...")
        run_bot(token=token, session_manager=sm, access_control=ac, verbose=args.verbose)
        return 0

    app_id = args.feishu_app_id or os.environ.get("FEISHU_APP_ID")
    app_secret = args.feishu_app_secret or os.environ.get("FEISHU_APP_SECRET")
    verification_token = args.feishu_verification_token or os.environ.get("FEISHU_VERIFICATION_TOKEN")
    encrypt_key = args.feishu_encrypt_key or os.environ.get("FEISHU_ENCRYPT_KEY")
    if not app_id or not app_secret:
        print(
            "ERROR: Feishu app credentials are required.\n"
            "  Pass --feishu-app-id/--feishu-app-secret or set FEISHU_APP_ID/FEISHU_APP_SECRET.",
            file=sys.stderr,
        )
        return 1
    from .channels.feishu import run_feishu_bot, run_feishu_ws_bot

    if args.feishu_connection_mode == "websocket":
        print(
            f"OmicVerse Jarvis starting (channel=feishu, model={args.model}, "
            "mode=websocket) ..."
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
            f"OmicVerse Jarvis starting (channel=feishu, model={args.model}, "
            f"mode=webhook, listen={args.feishu_host}:{args.feishu_port}{args.feishu_path}) ..."
        )
        run_feishu_bot(
            app_id=app_id,
            app_secret=app_secret,
            session_manager=sm,
            host=args.feishu_host,
            port=args.feishu_port,
            path=args.feishu_path,
            verification_token=verification_token,
            encrypt_key=encrypt_key,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
