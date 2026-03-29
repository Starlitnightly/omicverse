"""Allow ``python -m omicverse.mcp`` to start the MCP server."""

import sys

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

main()
