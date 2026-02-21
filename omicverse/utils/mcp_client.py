"""
MCPClientManager - Generic MCP (Model Context Protocol) client for OmicVerse Agent.

Manages connections to one or more MCP servers (HTTP or stdio transport),
discovers their tools, and provides a unified ``call()`` interface that the
Agent sandbox can use as ``mcp_call(tool_name, args)``.

Design choices
--------------
* HTTP transport uses only ``urllib.request`` – zero extra dependencies.
* stdio transport requires the ``mcp`` SDK (``pip install "mcp[cli]"``);
  a clear ``ImportError`` is raised when missing.
* All public methods are **synchronous**.  An internal ``_run_sync`` helper
  bridges async code via a background thread so that Jupyter's nested
  event-loop does not deadlock.

Example
-------
>>> from omicverse.utils.mcp_client import MCPClientManager
>>> mgr = MCPClientManager()
>>> info = mgr.connect("biocontext", url="https://mcp.biocontext.ai/mcp/")
>>> print([t.name for t in info.tools])
>>> result = mgr.call("string_interaction_partners", {"identifiers": "TP53"})
"""

from __future__ import annotations

import json
import logging
import threading
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MCPToolParam:
    """Single parameter of an MCP tool."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False


@dataclass
class MCPTool:
    """Discovered MCP tool with its metadata."""
    name: str
    description: str = ""
    parameters: List[MCPToolParam] = field(default_factory=list)
    server_name: str = ""

    @property
    def signature_text(self) -> str:
        """One-line signature suitable for LLM prompt injection."""
        params = ", ".join(
            f"{p.name}: {p.type}" + ("" if p.required else "?")
            for p in self.parameters
        )
        return f"{self.name}({params})"

    @property
    def prompt_block(self) -> str:
        """Multi-line block for the system prompt."""
        lines = [f"  - **{self.name}**: {self.description}"]
        if self.parameters:
            for p in self.parameters:
                req = " (required)" if p.required else ""
                lines.append(f"    - `{p.name}` ({p.type}{req}): {p.description}")
        return "\n".join(lines)


@dataclass
class MCPServerInfo:
    """Metadata for a connected MCP server."""
    name: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    tools: List[MCPTool] = field(default_factory=list)
    transport: str = "http"  # "http" | "stdio"
    server_info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lightweight HTTP MCP client (JSON-RPC over HTTP)
# ---------------------------------------------------------------------------

class _HTTPMCPClient:
    """Minimal MCP client that speaks JSON-RPC 2.0 over HTTP.

    Compatible with the *Streamable HTTP* transport used by BioContext
    and other hosted MCP servers.
    """

    def __init__(self, url: str, *, timeout: int = 30):
        self._url = url.rstrip("/")
        self._timeout = timeout
        self._request_id = 0
        self._session_id: Optional[str] = None

    # -- low-level ----------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and return the parsed response."""
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        req = Request(self._url, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                # Capture session id from response headers
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid

                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read().decode("utf-8")

                # Handle SSE responses (text/event-stream)
                if "text/event-stream" in content_type:
                    return self._parse_sse(raw)

                return json.loads(raw)
        except HTTPError as exc:
            raise ConnectionError(
                f"MCP HTTP error {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise ConnectionError(
                f"MCP connection failed ({self._url}): {exc.reason}"
            ) from exc

    @staticmethod
    def _parse_sse(raw: str) -> Dict[str, Any]:
        """Extract the last ``data:`` payload from an SSE stream."""
        last_data: Optional[str] = None
        for line in raw.splitlines():
            if line.startswith("data:"):
                last_data = line[len("data:"):].strip()
        if last_data is None:
            raise ValueError("No data field found in SSE response")
        return json.loads(last_data)

    # -- MCP protocol -------------------------------------------------------

    def initialize(self) -> Dict[str, Any]:
        """Send ``initialize`` and ``initialized`` notification."""
        resp = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "omicverse-agent", "version": "0.1.0"},
            },
        })
        result = resp.get("result", resp)
        # Send initialized notification (no id → notification)
        try:
            self._post({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            })
        except Exception:
            pass  # notification failure is non-fatal
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        """Send ``tools/list`` and return the tool descriptors."""
        resp = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        })
        result = resp.get("result", resp)
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Send ``tools/call`` and return the content."""
        resp = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        result = resp.get("result", resp)
        # Extract text content from MCP response
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_val = item.get("text", "")
                        # Try to parse as JSON
                        try:
                            texts.append(json.loads(text_val))
                        except (json.JSONDecodeError, TypeError):
                            texts.append(text_val)
                    else:
                        texts.append(item)
                else:
                    texts.append(item)
            if len(texts) == 1:
                return texts[0]
            return texts
        return content


# ---------------------------------------------------------------------------
# Sync bridge for async code in Jupyter
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run an async coroutine synchronously, safe for Jupyter notebooks."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # No running loop – just use asyncio.run
        return asyncio.run(coro)

    # Running inside an existing loop (Jupyter) – run in a thread
    result = None
    exception = None

    def _thread_target():
        nonlocal result, exception
        try:
            result = asyncio.run(coro)
        except Exception as exc:
            exception = exc

    thread = threading.Thread(target=_thread_target, daemon=True)
    thread.start()
    thread.join(timeout=120)
    if exception is not None:
        raise exception
    return result


# ---------------------------------------------------------------------------
# Tool schema parsing
# ---------------------------------------------------------------------------

def _parse_tool_params(input_schema: Any) -> List[MCPToolParam]:
    """Parse MCP tool parameters from either a dict or SDK object."""
    params: List[MCPToolParam] = []

    if input_schema is None:
        return params

    # Normalise to dict
    if hasattr(input_schema, "model_dump"):
        schema = input_schema.model_dump()
    elif hasattr(input_schema, "__dict__"):
        schema = vars(input_schema)
    elif isinstance(input_schema, dict):
        schema = input_schema
    else:
        return params

    properties = schema.get("properties", {})
    required_set = set(schema.get("required", []))

    for name, prop in properties.items():
        if isinstance(prop, dict):
            params.append(MCPToolParam(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description", ""),
                required=name in required_set,
            ))
    return params


def _parse_tool(raw: Any, server_name: str = "") -> MCPTool:
    """Convert a raw tool descriptor (dict or SDK object) to ``MCPTool``."""
    if isinstance(raw, dict):
        name = raw.get("name", "unknown")
        description = raw.get("description", "")
        input_schema = raw.get("inputSchema", raw.get("input_schema"))
    elif hasattr(raw, "name"):
        name = raw.name
        description = getattr(raw, "description", "") or ""
        input_schema = getattr(raw, "inputSchema", None) or getattr(raw, "input_schema", None)
    else:
        name = str(raw)
        description = ""
        input_schema = None

    return MCPTool(
        name=name,
        description=description,
        parameters=_parse_tool_params(input_schema),
        server_name=server_name,
    )


# ---------------------------------------------------------------------------
# MCPClientManager
# ---------------------------------------------------------------------------

class MCPClientManager:
    """Manage one or more MCP server connections.

    Each server is identified by a unique ``name``.  After calling
    :meth:`connect`, the server's tools are available via :meth:`call`.

    Parameters
    ----------
    default_timeout : int
        Default HTTP timeout in seconds for all connections.
    """

    def __init__(self, *, default_timeout: int = 30):
        self._timeout = default_timeout
        self._servers: Dict[str, MCPServerInfo] = {}
        self._http_clients: Dict[str, _HTTPMCPClient] = {}
        self._stdio_clients: Dict[str, Any] = {}  # SDK client objects
        # Reverse map: tool_name → server_name (for auto-routing)
        self._tool_map: Dict[str, str] = {}

    # -- connection lifecycle ------------------------------------------------

    def connect(
        self,
        name: str,
        *,
        url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> MCPServerInfo:
        """Connect to an MCP server and discover its tools.

        Parameters
        ----------
        name : str
            Unique human-readable name for this server.
        url : str, optional
            HTTP(S) endpoint for Streamable HTTP transport.
        command : str, optional
            Executable for stdio transport (e.g. ``"uvx"``).
        args : list[str], optional
            Arguments for the stdio command.
        timeout : int, optional
            HTTP timeout override for this connection.

        Returns
        -------
        MCPServerInfo
            Server metadata including discovered tools.
        """
        if url:
            return self._connect_http(name, url, timeout or self._timeout)
        elif command:
            return self._connect_stdio(name, command, args or [])
        else:
            raise ValueError("Either 'url' (HTTP) or 'command' (stdio) must be provided")

    def _connect_http(self, name: str, url: str, timeout: int) -> MCPServerInfo:
        """Connect via HTTP (Streamable HTTP transport)."""
        client = _HTTPMCPClient(url, timeout=timeout)
        server_meta = client.initialize()
        raw_tools = client.list_tools()
        tools = [_parse_tool(t, server_name=name) for t in raw_tools]

        info = MCPServerInfo(
            name=name,
            url=url,
            tools=tools,
            transport="http",
            server_info=server_meta,
        )
        self._servers[name] = info
        self._http_clients[name] = client
        for tool in tools:
            self._tool_map[tool.name] = name
        logger.info("Connected to MCP server '%s' (HTTP): %d tools", name, len(tools))
        return info

    def _connect_stdio(self, name: str, command: str, args: List[str]) -> MCPServerInfo:
        """Connect via stdio transport (requires ``mcp`` SDK)."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "stdio MCP transport requires the 'mcp' package. "
                "Install it with: pip install 'mcp[cli]'"
            )

        params = StdioServerParameters(command=command, args=args)

        async def _connect():
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.list_tools()
                    return session, response.tools

        session, raw_tools = _run_sync(_connect())
        tools = [_parse_tool(t, server_name=name) for t in raw_tools]

        info = MCPServerInfo(
            name=name,
            command=command,
            args=args,
            tools=tools,
            transport="stdio",
        )
        self._servers[name] = info
        self._stdio_clients[name] = {"session": session, "params": params}
        for tool in tools:
            self._tool_map[tool.name] = name
        logger.info("Connected to MCP server '%s' (stdio): %d tools", name, len(tools))
        return info

    def disconnect(self, name: str) -> None:
        """Disconnect from a named server and remove its tools."""
        if name in self._servers:
            for tool in self._servers[name].tools:
                self._tool_map.pop(tool.name, None)
            del self._servers[name]
        self._http_clients.pop(name, None)
        self._stdio_clients.pop(name, None)
        logger.info("Disconnected from MCP server '%s'", name)

    def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._servers.keys()):
            self.disconnect(name)

    # -- introspection -------------------------------------------------------

    @property
    def connected_servers(self) -> List[str]:
        """Return names of currently connected servers."""
        return list(self._servers.keys())

    def server_info(self, name: str) -> Optional[MCPServerInfo]:
        """Return info for a specific server, or ``None``."""
        return self._servers.get(name)

    def list_tools(self, server_name: Optional[str] = None) -> List[MCPTool]:
        """List discovered tools, optionally filtered by server."""
        if server_name:
            info = self._servers.get(server_name)
            return info.tools if info else []
        return [
            tool
            for info in self._servers.values()
            for tool in info.tools
        ]

    def tools_for_llm_prompt(self, server_name: Optional[str] = None) -> str:
        """Format tool descriptions for injection into an LLM system prompt."""
        tools = self.list_tools(server_name)
        if not tools:
            return ""
        lines = ["Available MCP tools:\n"]
        current_server = ""
        for tool in tools:
            if tool.server_name != current_server:
                current_server = tool.server_name
                lines.append(f"\n### Server: {current_server}")
            lines.append(tool.prompt_block)
        return "\n".join(lines)

    # -- tool invocation -----------------------------------------------------

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool on a specific server.

        Parameters
        ----------
        server_name : str
            Name of the connected server.
        tool_name : str
            Name of the tool to call.
        arguments : dict
            Arguments to pass to the tool.

        Returns
        -------
        Any
            Tool result (parsed JSON or raw text).
        """
        if server_name not in self._servers:
            raise ValueError(f"Server '{server_name}' is not connected")

        if server_name in self._http_clients:
            return self._http_clients[server_name].call_tool(tool_name, arguments)
        elif server_name in self._stdio_clients:
            return self._call_stdio_tool(server_name, tool_name, arguments)
        else:
            raise RuntimeError(f"No client found for server '{server_name}'")

    def _call_stdio_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool via stdio transport."""
        session_info = self._stdio_clients.get(server_name)
        if not session_info:
            raise RuntimeError(f"No stdio session for server '{server_name}'")

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "stdio MCP transport requires the 'mcp' package."
            )

        params = session_info["params"]

        async def _call():
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    # Extract text content
                    texts = []
                    for item in (result.content or []):
                        if hasattr(item, "text"):
                            try:
                                texts.append(json.loads(item.text))
                            except (json.JSONDecodeError, TypeError):
                                texts.append(item.text)
                        else:
                            texts.append(str(item))
                    return texts[0] if len(texts) == 1 else texts

        return _run_sync(_call())

    def call(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Call a tool by name, auto-routing to the correct server.

        Parameters
        ----------
        tool_name : str
            Name of the tool (must be unique across connected servers).
        arguments : dict, optional
            Arguments to pass to the tool.

        Returns
        -------
        Any
            Tool result.

        Raises
        ------
        ValueError
            If the tool name is unknown or ambiguous.
        """
        arguments = arguments or {}
        server_name = self._tool_map.get(tool_name)
        if server_name is None:
            known = ", ".join(sorted(self._tool_map.keys()))
            raise ValueError(
                f"Unknown MCP tool '{tool_name}'. Available tools: {known or '(none)'}"
            )
        return self.call_tool(server_name, tool_name, arguments)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def make_cache_key(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Generate a deterministic cache key for an MCP tool call."""
    args_str = json.dumps(arguments, sort_keys=True, default=str)
    digest = hashlib.md5(args_str.encode()).hexdigest()[:12]
    return f"mcp_{tool_name}_{digest}"
