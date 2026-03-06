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
            "Launch OmicVerse Jarvis — a Telegram bot for mobile single-cell analysis."
        ),
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

    # Resolve token
    token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print(
            "ERROR: Telegram bot token is required.\n"
            "  Pass --token or set TELEGRAM_BOT_TOKEN env var.",
            file=sys.stderr,
        )
        return 1

    api_key = _resolve_api_key(args.api_key)

    from .session import SessionManager
    from .bot import AccessControl, run_bot

    sm = SessionManager(
        session_dir=args.session_dir,
        model=args.model,
        api_key=api_key,
        max_prompts=args.max_prompts,
        verbose=args.verbose,
    )
    ac = AccessControl(allowed=args.allowed_users or None)

    print(f"OmicVerse Jarvis starting (model={args.model}) ...")
    run_bot(token=token, session_manager=sm, access_control=ac, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
