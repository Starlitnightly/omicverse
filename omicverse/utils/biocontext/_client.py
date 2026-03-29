"""Lightweight MCP client for BioContext biomedical knowledge tools.

Uses the MCP Streamable HTTP transport to communicate with the remote
BioContext server.  No external dependencies — only ``urllib.request``
from the standard library.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
import urllib.error
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://biocontext-kb.fastmcp.app/mcp"
_TIMEOUT = 30  # seconds per HTTP request


class BioContextClient:
    """Stateless MCP client for BioContext.

    Each :meth:`call_tool` creates a short-lived MCP session
    (initialize → notification → tools/call) so that the client
    remains thread-safe and needs no persistent connection.
    """

    def __init__(self, url: str = _DEFAULT_URL, timeout: int = _TIMEOUT) -> None:
        self.url = url
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _post(
        self,
        payload: dict,
        session_id: Optional[str] = None,
    ) -> tuple[Optional[dict], Optional[str], int]:
        """Send a JSON-RPC request and parse the SSE response."""
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if session_id:
            headers["Mcp-Session-Id"] = session_id

        req = urllib.request.Request(self.url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("mcp-session-id", session_id)
                raw = resp.read().decode("utf-8")
                # Parse Server-Sent Events
                result = None
                for line in raw.split("\n"):
                    if line.startswith("data: "):
                        result = json.loads(line[6:])
                return result, sid, resp.status
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            logger.warning("BioContext HTTP %s: %s", e.code, body)
            return None, session_id, e.code
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning("BioContext network error: %s", e)
            return None, session_id, 0

    def _create_session(self) -> Optional[str]:
        """Initialize an MCP session and send the ``initialized`` notification."""
        result, sid, code = self._post(
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1,
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "omicverse", "version": "1.7.10"},
                },
            }
        )
        if code != 200 or sid is None:
            raise ConnectionError(
                f"Failed to initialize BioContext session (HTTP {code}). "
                "Check your network connection."
            )
        # Send the required ``initialized`` notification
        self._post(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=sid,
        )
        return sid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Call a BioContext MCP tool and return its result.

        Parameters
        ----------
        tool_name : str
            Name of the BioContext tool (e.g. ``"get_uniprot_protein_info"``).
        arguments : dict, optional
            Tool arguments as key-value pairs.

        Returns
        -------
        dict or str
            Parsed tool result.  Returns the text content if the result
            contains a single text block, otherwise the full content list.
        """
        sid = self._create_session()
        result, _, code = self._post(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": 2,
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {},
                },
            },
            session_id=sid,
        )
        if result is None:
            raise ConnectionError(
                f"BioContext tool '{tool_name}' returned no result (HTTP {code})."
            )
        if "error" in result:
            err = result["error"]
            raise RuntimeError(
                f"BioContext tool error: {err.get('message', err)}"
            )

        content = result.get("result", {}).get("content", [])
        # Flatten single text block into a parsed dict/string
        texts = [c["text"] for c in content if c.get("type") == "text"]
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except (json.JSONDecodeError, TypeError):
                return texts[0]
        if texts:
            parsed = []
            for t in texts:
                try:
                    parsed.append(json.loads(t))
                except (json.JSONDecodeError, TypeError):
                    parsed.append(t)
            return parsed
        return content

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the list of available BioContext tools with metadata."""
        sid = self._create_session()
        result, _, code = self._post(
            {"jsonrpc": "2.0", "method": "tools/list", "id": 2, "params": {}},
            session_id=sid,
        )
        if result is None:
            raise ConnectionError(
                f"Failed to list BioContext tools (HTTP {code})."
            )
        return result.get("result", {}).get("tools", [])


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_lock = threading.Lock()
_client: Optional[BioContextClient] = None


def get_client(url: str = _DEFAULT_URL, timeout: int = _TIMEOUT) -> BioContextClient:
    """Return (or create) the module-level :class:`BioContextClient`."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = BioContextClient(url=url, timeout=timeout)
    return _client
