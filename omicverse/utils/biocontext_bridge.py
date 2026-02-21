"""
BioContextBridge - Pre-configured MCP bridge for BioContext.ai databases.

Wraps :class:`MCPClientManager` with BioContext-specific defaults and
integrates with :class:`FilesystemContextManager` for result caching.

Zero-config remote usage
------------------------
>>> from omicverse.utils.biocontext_bridge import BioContextBridge
>>> bc = BioContextBridge()
>>> bc.connect()
>>> result = bc.query("string_interaction_partners",
...     {"identifiers": "TP53", "species": 9606})

Agent integration (caching)
---------------------------
When a ``FilesystemContextManager`` is provided, query results are
automatically cached under the ``"mcp_cache"`` category.  Repeated
queries with the same parameters return the cached result without a
network call until the ``cache_ttl`` expires.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .mcp_client import MCPClientManager, MCPServerInfo, MCPTool, make_cache_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BioContext constants
# ---------------------------------------------------------------------------

BIOCONTEXT_REMOTE_URL = "https://mcp.biocontext.ai/mcp/"
BIOCONTEXT_LOCAL_COMMAND = "uvx"
BIOCONTEXT_LOCAL_ARGS = ["biocontext_kb@latest"]

# ---------------------------------------------------------------------------
# BioContextBridge
# ---------------------------------------------------------------------------


class BioContextBridge:
    """High-level wrapper around BioContext MCP with optional caching.

    Parameters
    ----------
    mode : str
        Connection mode: ``"remote"`` (hosted endpoint), ``"local"``
        (stdio via ``uvx``), or ``"auto"`` (try remote first, fall back
        to local).
    context_manager : FilesystemContextManager, optional
        When provided, query results are cached to disk.
    cache_ttl : int
        Seconds before a cached result expires (default: 3600).
    timeout : int
        HTTP timeout in seconds (default: 30).
    """

    def __init__(
        self,
        mode: str = "remote",
        context_manager: Optional[Any] = None,
        cache_ttl: int = 3600,
        timeout: int = 30,
    ):
        if mode not in ("remote", "local", "auto"):
            raise ValueError(f"Invalid mode '{mode}'; use 'remote', 'local', or 'auto'")
        self._mode = mode
        self._context_manager = context_manager
        self._cache_ttl = cache_ttl
        self._timeout = timeout

        self._manager: Optional[MCPClientManager] = None
        self._server_info: Optional[MCPServerInfo] = None
        self._connected = False

    # -- connection ----------------------------------------------------------

    def connect(self) -> MCPServerInfo:
        """Establish the MCP connection and discover tools.

        Returns
        -------
        MCPServerInfo
            Metadata about the connected BioContext server.
        """
        self._manager = MCPClientManager(default_timeout=self._timeout)

        if self._mode == "remote":
            return self._connect_remote()
        elif self._mode == "local":
            return self._connect_local()
        else:  # auto
            try:
                return self._connect_remote()
            except Exception as exc:
                logger.warning("Remote BioContext failed (%s); trying local…", exc)
                return self._connect_local()

    def _connect_remote(self) -> MCPServerInfo:
        info = self._manager.connect(
            "biocontext",
            url=BIOCONTEXT_REMOTE_URL,
            timeout=self._timeout,
        )
        self._server_info = info
        self._connected = True
        return info

    def _connect_local(self) -> MCPServerInfo:
        info = self._manager.connect(
            "biocontext",
            command=BIOCONTEXT_LOCAL_COMMAND,
            args=BIOCONTEXT_LOCAL_ARGS,
        )
        self._server_info = info
        self._connected = True
        return info

    # -- properties ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def manager(self) -> Optional[MCPClientManager]:
        """Access the underlying :class:`MCPClientManager`."""
        return self._manager

    def available_tools(self) -> List[MCPTool]:
        """Return the list of discovered BioContext tools."""
        if not self._manager:
            return []
        return self._manager.list_tools("biocontext")

    # -- querying ------------------------------------------------------------

    def query(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> Any:
        """Query a BioContext MCP tool.

        Parameters
        ----------
        tool_name : str
            Name of the BioContext tool to call.
        arguments : dict, optional
            Arguments for the tool.
        use_cache : bool
            Check/update the filesystem cache (default: ``True``).

        Returns
        -------
        Any
            Parsed tool result (dict, list, or string).
        """
        if not self._connected or not self._manager:
            raise RuntimeError(
                "BioContext is not connected. Call .connect() first."
            )
        arguments = arguments or {}
        cache_key = make_cache_key(tool_name, arguments)

        # Cache check
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s", cache_key)
                return cached

        # Actual MCP call
        result = self._manager.call_tool("biocontext", tool_name, arguments)

        # Write to cache
        if use_cache:
            self._write_cache(cache_key, result)

        return result

    # -- convenience wrappers ------------------------------------------------

    def string_interactions(
        self, identifiers: str, species: int = 9606, limit: int = 50
    ) -> Any:
        """Query STRING for protein interaction partners."""
        return self.query("string_interaction_partners", {
            "identifiers": identifiers,
            "species": species,
            "limit": limit,
        })

    def uniprot_lookup(self, accession: str) -> Any:
        """Look up a protein in UniProt."""
        return self.query("uniprot_protein_lookup", {
            "accession": accession,
        })

    def kegg_pathway(self, pathway_id: str) -> Any:
        """Retrieve a KEGG pathway."""
        return self.query("kegg_pathway_info", {
            "pathway_id": pathway_id,
        })

    def panglao_markers(self, cell_type: str, species: str = "Hs") -> Any:
        """Look up cell-type marker genes from PanglaoDB."""
        return self.query("panglao_cell_markers", {
            "cell_type": cell_type,
            "species": species,
        })

    def europepmc_search(self, query: str, limit: int = 10) -> Any:
        """Search Europe PMC for literature."""
        return self.query("europepmc_search", {
            "query": query,
            "limit": limit,
        })

    def reactome_pathway(self, pathway_id: str) -> Any:
        """Retrieve Reactome pathway information."""
        return self.query("reactome_pathway_info", {
            "pathway_id": pathway_id,
        })

    def open_targets(self, target_id: str) -> Any:
        """Query Open Targets for a target."""
        return self.query("opentargets_target_info", {
            "target_id": target_id,
        })

    # -- caching internals ---------------------------------------------------

    def _cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Public helper for external callers that need a cache key."""
        return make_cache_key(tool_name, arguments)

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Return cached result if fresh, else ``None``."""
        if not self._context_manager:
            return None
        try:
            results = self._context_manager.search_context(cache_key, match_type="glob")
            if not results:
                return None
            # Get full note content
            note = self._context_manager.get_note(cache_key)
            if note is None:
                return None
            content = note.content if hasattr(note, "content") else note
            if isinstance(content, dict):
                ts = content.get("_cached_at", 0)
                if time.time() - ts > self._cache_ttl:
                    return None  # expired
                return content.get("data")
            return content
        except Exception:
            return None

    def _write_cache(self, cache_key: str, result: Any) -> None:
        """Write a result into the filesystem cache."""
        if not self._context_manager:
            return
        try:
            payload = {
                "_cached_at": time.time(),
                "data": result,
            }
            self._context_manager.write_note(
                key=cache_key,
                content=payload,
                category="mcp_cache",
            )
        except Exception as exc:
            logger.debug("Failed to cache MCP result: %s", exc)

    # -- cleanup -------------------------------------------------------------

    def disconnect(self) -> None:
        """Disconnect from BioContext."""
        if self._manager:
            self._manager.disconnect_all()
        self._connected = False
        self._server_info = None
