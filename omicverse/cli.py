"""
OmicVerse main CLI dispatcher.

Entry point: ``omicverse`` (registered via pyproject.toml scripts).

Sub-commands
------------
jarvis        Launch the Telegram bot for mobile bioinformatics.
claw          Generate OmicVerse Python code from a natural-language request.
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


def _run_skill_seeker(argv: List[str]) -> int:
    from omicverse.ov_skill_seeker.cli import main as ss_main
    return ss_main(argv)


def _run_claw(argv: List[str]) -> int:
    from omicverse.claw import main as claw_main
    return claw_main(argv)


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
        "jarvis",
        help="Launch Telegram bot for mobile single-cell analysis.",
        add_help=False,
    ).set_defaults(func=_run_jarvis)

    subparsers.add_parser(
        "claw",
        help="Generate OmicVerse Python code from a natural-language request.",
        add_help=False,
    ).set_defaults(func=_run_claw)

    subparsers.add_parser(
        "web",
        help="Launch the OmicVerse web interface.",
        add_help=False,
    ).set_defaults(func=_run_web)

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
