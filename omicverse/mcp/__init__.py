"""
OmicVerse Registry MCP Server.

Exposes ``@register_function`` entries as MCP tools, auto-generated from
the global registry.
"""

from __future__ import annotations

from typing import List, Optional

__all__ = ["build_default_manifest", "build_mcp_server", "get_manifest"]


def build_default_manifest(phase: str = "P0+P0.5") -> List[dict]:
    """Build the default manifest from the global registry."""
    from .manifest import build_registry_manifest
    return build_registry_manifest(phase=phase)


def build_mcp_server(phase: str = "P0+P0.5"):
    """Create and return a ``RegistryMcpServer`` instance."""
    from .server import RegistryMcpServer
    return RegistryMcpServer(phase=phase)


def get_manifest(phase: Optional[str] = None) -> List[dict]:
    """Read-only access to the manifest (without internal function refs)."""
    from .manifest import build_registry_manifest
    entries = build_registry_manifest(phase=phase)
    return [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries]
