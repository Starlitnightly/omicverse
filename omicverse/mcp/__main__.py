"""Allow ``python -m omicverse.mcp`` to start the MCP server."""

import sys


def _requested_transport(argv: list[str]) -> str | None:
    """Return the requested transport, or ``None`` when argparse should handle exit-only flags."""
    args = list(argv[1:])
    for index, token in enumerate(args):
        if token in {"-h", "--help", "--version"}:
            return None
        if token == "--transport" and index + 1 < len(args):
            return args[index + 1]
        if token.startswith("--transport="):
            return token.split("=", 1)[1]
    return "stdio"


def _preflight_transport_dependencies() -> None:
    """Fail fast on missing transport deps before importing the heavy server module."""
    transport = _requested_transport(sys.argv)
    if transport is None:
        return

    if transport == "stdio":
        try:
            from mcp.server.stdio import stdio_server  # noqa: F401
        except ImportError as exc:
            print(
                f"Error: could not start the OmicVerse MCP server.\n"
                f"  {exc}\n\n"
                f"Install the MCP extras with:\n"
                f"  pip install 'omicverse[mcp]'\n",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return

    if transport == "streamable-http":
        try:
            import uvicorn  # noqa: F401
            from starlette.applications import Starlette  # noqa: F401
            from mcp.server.fastmcp.server import StreamableHTTPASGIApp  # noqa: F401
        except ImportError as exc:
            print(
                f"Error: could not start the OmicVerse MCP server.\n"
                f"  {exc}\n\n"
                f"Install the streamable-http extras with:\n"
                f"  pip install mcp uvicorn starlette\n",
                file=sys.stderr,
            )
            raise SystemExit(1)


_preflight_transport_dependencies()

try:
    from .server import main
except ImportError as exc:
    print(
        f"Error: could not load the OmicVerse MCP server.\n"
        f"  {exc}\n\n"
        f"Install the MCP extras with:\n"
        f"  pip install 'omicverse[mcp]'\n",
        file=sys.stderr,
    )
    raise SystemExit(1)

try:
    main()
except ImportError as exc:
    print(
        f"Error: could not start the OmicVerse MCP server.\n"
        f"  {exc}\n\n"
        f"Install the MCP extras with:\n"
        f"  pip install 'omicverse[mcp]'\n",
        file=sys.stderr,
    )
    raise SystemExit(1)
