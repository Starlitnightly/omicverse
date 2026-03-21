"""
OmicVerse main CLI dispatcher.

Entry point: ``omicverse`` (registered via pyproject.toml scripts).

Sub-commands
------------
claw          Start the Jarvis bot and/or generate code with -q.
jarvis        Alias for ``claw`` (legacy name).
web           Launch the OmicVerse web interface.
skill-seeker  OmicVerse Skill Seeker utilities (list/validate/package skills).
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _run_jarvis(argv: List[str]) -> int:
    from omicverse.jarvis.cli import main as jarvis_main
    return jarvis_main(argv)


def _run_claw(argv: List[str]) -> int:
    """Default `omicverse claw` to gateway mode unless the user asked for claw-only actions."""
    claw_passthrough_flags = {
        "-q",
        "--question",
        "--daemon",
        "--use-daemon",
        "--stop-daemon",
    }
    if any(flag in argv for flag in claw_passthrough_flags):
        return _run_jarvis(argv)
    return _run_gateway(argv)


def _run_skill_seeker(argv: List[str]) -> int:
    from omicverse.ov_skill_seeker.cli import main as ss_main
    return ss_main(argv)


def _run_gateway(argv: List[str]) -> int:
    """Start the gateway daemon mode."""
    # Inject the internal gateway-daemon flag unless the user already passed it.
    # Keep --with-web for compatibility with the existing Jarvis launcher.
    if "--gateway-daemon" not in argv:
        argv = ["--gateway-daemon"] + list(argv)
    if "--with-web" not in argv:
        argv = ["--with-web"] + list(argv)
    return _run_jarvis(argv)


def _run_web(argv: List[str]) -> int:
    try:
        from omicverse_web.start_server import main as web_main
    except ImportError as exc:
        print(
            "OmicVerse web interface is unavailable. "
            "Ensure `omicverse_web` is installed and on PYTHONPATH.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1
    return web_main(argv)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omicverse",
        description="OmicVerse command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    subparsers.add_parser(
        "claw",
        help=(
            "Start OmicVerse gateway mode by default. "
            "Use -q or daemon flags for one-shot claw actions."
        ),
        add_help=False,
    ).set_defaults(func=_run_claw)

    subparsers.add_parser(
        "jarvis",
        help="Alias for 'claw' (legacy name).",
        add_help=False,
    ).set_defaults(func=_run_jarvis)

    subparsers.add_parser(
        "web",
        help="Launch the OmicVerse web interface.",
        add_help=False,
    ).set_defaults(func=_run_web)

    subparsers.add_parser(
        "gateway",
        help=(
            "Start the gateway daemon and web UI. "
            "Configured channels are auto-started in the background."
        ),
        add_help=False,
    ).set_defaults(func=_run_gateway)

    subparsers.add_parser(
        "skill-seeker",
        help="List, validate, and package OmicVerse Agent Skills.",
        add_help=False,
    ).set_defaults(func=_run_skill_seeker)

    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(remaining)


if __name__ == "__main__":
    sys.exit(main())
