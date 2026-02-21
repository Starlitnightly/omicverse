"""
Unit tests for MCP client manager and BioContext bridge.

All tests are fully mocked — no network access required.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Dict, List

import pytest

from omicverse.utils.mcp_client import (
    MCPClientManager,
    MCPTool,
    MCPToolParam,
    MCPServerInfo,
    _HTTPMCPClient,
    _parse_tool,
    _parse_tool_params,
    make_cache_key,
)
from omicverse.utils.biocontext_bridge import BioContextBridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_TOOLS_JSON = [
    {
        "name": "string_interaction_partners",
        "description": "Get protein interaction partners from STRING",
        "inputSchema": {
            "type": "object",
            "properties": {
                "identifiers": {"type": "string", "description": "Gene/protein name"},
                "species": {"type": "integer", "description": "NCBI species ID"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["identifiers"],
        },
    },
    {
        "name": "uniprot_protein_lookup",
        "description": "Look up a protein in UniProt",
        "inputSchema": {
            "type": "object",
            "properties": {
                "accession": {"type": "string", "description": "UniProt accession"},
            },
            "required": ["accession"],
        },
    },
]


def _make_init_response():
    return {
        "result": {
            "protocolVersion": "2025-03-26",
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        }
    }


def _make_list_tools_response():
    return {"result": {"tools": FAKE_TOOLS_JSON}}


def _make_call_tool_response(data):
    return {
        "result": {
            "content": [{"type": "text", "text": json.dumps(data)}]
        }
    }


# ---------------------------------------------------------------------------
# _parse_tool_params
# ---------------------------------------------------------------------------

class TestParseToolParams:
    def test_from_dict(self):
        schema = {
            "type": "object",
            "properties": {
                "identifiers": {"type": "string", "description": "Gene name"},
                "species": {"type": "integer", "description": "Species"},
            },
            "required": ["identifiers"],
        }
        params = _parse_tool_params(schema)
        assert len(params) == 2
        assert params[0].name == "identifiers"
        assert params[0].required is True
        assert params[1].name == "species"
        assert params[1].required is False

    def test_from_none(self):
        assert _parse_tool_params(None) == []

    def test_from_empty_dict(self):
        assert _parse_tool_params({}) == []


class TestParseTool:
    def test_from_dict(self):
        tool = _parse_tool(FAKE_TOOLS_JSON[0], server_name="test")
        assert tool.name == "string_interaction_partners"
        assert tool.server_name == "test"
        assert len(tool.parameters) == 3
        assert tool.parameters[0].required is True

    def test_signature_text(self):
        tool = _parse_tool(FAKE_TOOLS_JSON[0])
        sig = tool.signature_text
        assert "string_interaction_partners(" in sig
        assert "identifiers: string" in sig

    def test_prompt_block(self):
        tool = _parse_tool(FAKE_TOOLS_JSON[1])
        block = tool.prompt_block
        assert "uniprot_protein_lookup" in block
        assert "accession" in block


# ---------------------------------------------------------------------------
# _HTTPMCPClient
# ---------------------------------------------------------------------------

class TestHTTPMCPClient:
    @patch("omicverse.utils.mcp_client.urlopen")
    def test_initialize(self, mock_urlopen):
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps(_make_init_response()).encode()
        resp_mock.headers = {"Content-Type": "application/json", "Mcp-Session-Id": "sess-123"}
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp_mock

        client = _HTTPMCPClient("https://example.com/mcp/")
        result = client.initialize()
        assert "protocolVersion" in result or "serverInfo" in result

    @patch("omicverse.utils.mcp_client.urlopen")
    def test_list_tools(self, mock_urlopen):
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps(_make_list_tools_response()).encode()
        resp_mock.headers = {"Content-Type": "application/json"}
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp_mock

        client = _HTTPMCPClient("https://example.com/mcp/")
        tools = client.list_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "string_interaction_partners"

    @patch("omicverse.utils.mcp_client.urlopen")
    def test_call_tool(self, mock_urlopen):
        resp_mock = MagicMock()
        data = {"partners": [{"name": "MDM2"}]}
        resp_mock.read.return_value = json.dumps(_make_call_tool_response(data)).encode()
        resp_mock.headers = {"Content-Type": "application/json"}
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp_mock

        client = _HTTPMCPClient("https://example.com/mcp/")
        result = client.call_tool("string_interaction_partners", {"identifiers": "TP53"})
        assert result == data

    @patch("omicverse.utils.mcp_client.urlopen")
    def test_sse_response(self, mock_urlopen):
        sse_body = 'event: message\ndata: {"result": {"content": [{"type": "text", "text": "hello"}]}}\n\n'
        resp_mock = MagicMock()
        resp_mock.read.return_value = sse_body.encode()
        resp_mock.headers = {"Content-Type": "text/event-stream"}
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp_mock

        client = _HTTPMCPClient("https://example.com/mcp/")
        result = client.call_tool("some_tool", {})
        assert result == "hello"


# ---------------------------------------------------------------------------
# MCPClientManager
# ---------------------------------------------------------------------------

class TestMCPClientManager:
    def _connect_mock_http(self, mgr: MCPClientManager, name: str = "test"):
        """Helper: inject a fake HTTP server without network."""
        tools = [_parse_tool(t, server_name=name) for t in FAKE_TOOLS_JSON]
        info = MCPServerInfo(name=name, url="https://fake/", tools=tools, transport="http")
        mgr._servers[name] = info
        for tool in tools:
            mgr._tool_map[tool.name] = name
        # Mock HTTP client
        mock_client = MagicMock(spec=_HTTPMCPClient)
        mgr._http_clients[name] = mock_client
        return mock_client

    def test_connected_servers(self):
        mgr = MCPClientManager()
        assert mgr.connected_servers == []
        self._connect_mock_http(mgr, "srv1")
        assert mgr.connected_servers == ["srv1"]

    def test_list_tools_single_server(self):
        mgr = MCPClientManager()
        self._connect_mock_http(mgr, "srv1")
        tools = mgr.list_tools("srv1")
        assert len(tools) == 2
        assert tools[0].name == "string_interaction_partners"

    def test_list_tools_all(self):
        mgr = MCPClientManager()
        self._connect_mock_http(mgr, "srv1")
        self._connect_mock_http(mgr, "srv2")
        all_tools = mgr.list_tools()
        assert len(all_tools) == 4  # 2 per server

    def test_call_routes_correctly(self):
        mgr = MCPClientManager()
        mock_client = self._connect_mock_http(mgr, "srv1")
        mock_client.call_tool.return_value = {"data": "ok"}

        result = mgr.call("string_interaction_partners", {"identifiers": "TP53"})
        mock_client.call_tool.assert_called_once_with(
            "string_interaction_partners", {"identifiers": "TP53"}
        )
        assert result == {"data": "ok"}

    def test_call_unknown_tool_raises(self):
        mgr = MCPClientManager()
        with pytest.raises(ValueError, match="Unknown MCP tool"):
            mgr.call("nonexistent_tool")

    def test_disconnect(self):
        mgr = MCPClientManager()
        self._connect_mock_http(mgr, "srv1")
        assert "srv1" in mgr.connected_servers
        mgr.disconnect("srv1")
        assert "srv1" not in mgr.connected_servers
        assert "string_interaction_partners" not in mgr._tool_map

    def test_disconnect_all(self):
        mgr = MCPClientManager()
        self._connect_mock_http(mgr, "srv1")
        self._connect_mock_http(mgr, "srv2")
        mgr.disconnect_all()
        assert mgr.connected_servers == []

    def test_tools_for_llm_prompt(self):
        mgr = MCPClientManager()
        self._connect_mock_http(mgr, "biocontext")
        prompt = mgr.tools_for_llm_prompt()
        assert "string_interaction_partners" in prompt
        assert "uniprot_protein_lookup" in prompt
        assert "Server: biocontext" in prompt

    def test_tools_for_llm_prompt_empty(self):
        mgr = MCPClientManager()
        assert mgr.tools_for_llm_prompt() == ""

    def test_server_info(self):
        mgr = MCPClientManager()
        self._connect_mock_http(mgr, "srv1")
        info = mgr.server_info("srv1")
        assert info is not None
        assert info.name == "srv1"
        assert mgr.server_info("nonexistent") is None

    def test_connect_requires_url_or_command(self):
        mgr = MCPClientManager()
        with pytest.raises(ValueError, match="Either 'url'"):
            mgr.connect("test")


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self):
        k1 = make_cache_key("tool", {"a": 1, "b": 2})
        k2 = make_cache_key("tool", {"b": 2, "a": 1})
        assert k1 == k2  # sorted keys

    def test_different_tools(self):
        k1 = make_cache_key("tool_a", {"x": 1})
        k2 = make_cache_key("tool_b", {"x": 1})
        assert k1 != k2

    def test_prefix(self):
        k = make_cache_key("my_tool", {"x": 1})
        assert k.startswith("mcp_my_tool_")


# ---------------------------------------------------------------------------
# BioContextBridge
# ---------------------------------------------------------------------------

class TestBioContextBridge:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            BioContextBridge(mode="invalid")

    def test_not_connected_raises(self):
        bc = BioContextBridge()
        with pytest.raises(RuntimeError, match="not connected"):
            bc.query("some_tool")

    @patch("omicverse.utils.biocontext_bridge.MCPClientManager")
    def test_connect_remote(self, MockMgr):
        mock_mgr = MockMgr.return_value
        fake_info = MCPServerInfo(name="biocontext", url="https://fake/", tools=[])
        mock_mgr.connect.return_value = fake_info

        bc = BioContextBridge(mode="remote")
        info = bc.connect()
        assert bc.is_connected
        mock_mgr.connect.assert_called_once()
        assert info.name == "biocontext"

    @patch("omicverse.utils.biocontext_bridge.MCPClientManager")
    def test_query_with_cache_miss(self, MockMgr):
        mock_mgr = MockMgr.return_value
        fake_tools = [_parse_tool(FAKE_TOOLS_JSON[0], "biocontext")]
        fake_info = MCPServerInfo(name="biocontext", url="https://fake/", tools=fake_tools)
        mock_mgr.connect.return_value = fake_info
        mock_mgr.call_tool.return_value = {"partners": []}

        bc = BioContextBridge(mode="remote")
        bc.connect()
        result = bc.query("string_interaction_partners", {"identifiers": "TP53"})
        mock_mgr.call_tool.assert_called_once()
        assert result == {"partners": []}

    @patch("omicverse.utils.biocontext_bridge.MCPClientManager")
    def test_query_with_cache_hit(self, MockMgr):
        mock_mgr = MockMgr.return_value
        fake_info = MCPServerInfo(name="biocontext", url="https://fake/", tools=[])
        mock_mgr.connect.return_value = fake_info

        # Mock context manager with cache hit
        mock_ctx = MagicMock()
        mock_ctx.search_context.return_value = [MagicMock()]
        cached_data = {"_cached_at": time.time(), "data": {"cached": True}}
        mock_note = MagicMock()
        mock_note.content = cached_data
        mock_ctx.get_note.return_value = mock_note

        bc = BioContextBridge(mode="remote", context_manager=mock_ctx, cache_ttl=3600)
        bc.connect()
        result = bc.query("some_tool", {"key": "val"})

        # Should not call the actual MCP tool
        mock_mgr.call_tool.assert_not_called()
        assert result == {"cached": True}

    @patch("omicverse.utils.biocontext_bridge.MCPClientManager")
    def test_cache_expired(self, MockMgr):
        mock_mgr = MockMgr.return_value
        fake_info = MCPServerInfo(name="biocontext", url="https://fake/", tools=[])
        mock_mgr.connect.return_value = fake_info
        mock_mgr.call_tool.return_value = {"fresh": True}

        # Mock context manager with expired cache
        mock_ctx = MagicMock()
        mock_ctx.search_context.return_value = [MagicMock()]
        expired_data = {"_cached_at": time.time() - 7200, "data": {"old": True}}
        mock_note = MagicMock()
        mock_note.content = expired_data
        mock_ctx.get_note.return_value = mock_note

        bc = BioContextBridge(mode="remote", context_manager=mock_ctx, cache_ttl=3600)
        bc.connect()
        result = bc.query("tool", {"x": 1})

        # Should have called the actual tool (cache expired)
        mock_mgr.call_tool.assert_called_once()
        assert result == {"fresh": True}

    @patch("omicverse.utils.biocontext_bridge.MCPClientManager")
    def test_convenience_methods_exist(self, MockMgr):
        mock_mgr = MockMgr.return_value
        fake_info = MCPServerInfo(name="biocontext", url="https://fake/", tools=[])
        mock_mgr.connect.return_value = fake_info
        mock_mgr.call_tool.return_value = {}

        bc = BioContextBridge(mode="remote")
        bc.connect()

        # All convenience methods should call through to query
        bc.string_interactions("TP53")
        bc.uniprot_lookup("P04637")
        bc.kegg_pathway("hsa04110")
        bc.panglao_markers("T cells")
        bc.europepmc_search("CRISPR")
        bc.reactome_pathway("R-HSA-109582")
        bc.open_targets("ENSG00000141510")

        assert mock_mgr.call_tool.call_count == 7

    @patch("omicverse.utils.biocontext_bridge.MCPClientManager")
    def test_disconnect(self, MockMgr):
        mock_mgr = MockMgr.return_value
        fake_info = MCPServerInfo(name="biocontext", url="https://fake/", tools=[])
        mock_mgr.connect.return_value = fake_info

        bc = BioContextBridge(mode="remote")
        bc.connect()
        assert bc.is_connected
        bc.disconnect()
        assert not bc.is_connected

    def test_available_tools_not_connected(self):
        bc = BioContextBridge()
        assert bc.available_tools() == []


# ---------------------------------------------------------------------------
# AgentConfig MCPConfig
# ---------------------------------------------------------------------------

class TestMCPConfig:
    def test_default_config(self):
        from omicverse.utils.agent_config import MCPConfig
        cfg = MCPConfig()
        assert cfg.enable_biocontext == "auto"
        assert cfg.servers == []
        assert cfg.biocontext_mode == "remote"
        assert cfg.cache_ttl == 3600

    def test_agent_config_has_mcp(self):
        from omicverse.utils.agent_config import AgentConfig
        cfg = AgentConfig()
        assert hasattr(cfg, "mcp")
        assert cfg.mcp.enable_biocontext == "auto"

    def test_from_flat_kwargs_default_is_auto(self):
        from omicverse.utils.agent_config import AgentConfig
        cfg = AgentConfig.from_flat_kwargs()
        assert cfg.mcp.enable_biocontext == "auto"

    def test_from_flat_kwargs_with_mcp(self):
        from omicverse.utils.agent_config import AgentConfig
        cfg = AgentConfig.from_flat_kwargs(
            enable_biocontext=True,
            biocontext_mode="auto",
            mcp_servers=[{"name": "test", "url": "https://example.com/mcp/"}],
        )
        assert cfg.mcp.enable_biocontext is True
        assert cfg.mcp.biocontext_mode == "auto"
        assert len(cfg.mcp.servers) == 1


# ---------------------------------------------------------------------------
# Auto-detection logic
# ---------------------------------------------------------------------------

class TestBioContextAutoDetection:
    """Test the keyword-based auto-detection used by run_async.

    We import OmicVerseAgent only for its class-level constants and
    static helpers — no instantiation needed.
    """

    @staticmethod
    def _detect(request: str) -> bool:
        """Reproduce the detection logic without instantiating the agent."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        keywords = OmicVerseAgent._BIOCONTEXT_TRIGGER_KEYWORDS
        lower = request.lower()
        return any(kw in lower for kw in keywords)

    def test_detects_english_protein_interaction(self):
        assert self._detect("find protein interaction partners for TP53")

    def test_detects_chinese_protein_interaction(self):
        assert self._detect("查找TP53的蛋白互作伙伴")

    def test_detects_kegg_pathway(self):
        assert self._detect("look up KEGG pathway hsa04110")

    def test_detects_cell_markers(self):
        assert self._detect("get cell type markers for T cells from PanglaoDB")

    def test_detects_drug_target(self):
        assert self._detect("查询BRAF的药物靶点信息")

    def test_detects_literature_search(self):
        assert self._detect("search PubMed for CRISPR screen papers")

    def test_no_trigger_for_normal_analysis(self):
        assert not self._detect("quality control with nUMI>500")

    def test_no_trigger_for_clustering(self):
        assert not self._detect("leiden clustering with resolution 1.0")

    def test_no_trigger_for_deg(self):
        assert not self._detect("find differentially expressed genes between clusters")

    def test_biocontext_is_eager(self):
        from omicverse.utils.smart_agent import OmicVerseAgent
        assert OmicVerseAgent._biocontext_is_eager(True)
        assert OmicVerseAgent._biocontext_is_eager("yes")
        assert OmicVerseAgent._biocontext_is_eager("true")
        assert not OmicVerseAgent._biocontext_is_eager(False)
        assert not OmicVerseAgent._biocontext_is_eager("auto")
        assert not OmicVerseAgent._biocontext_is_eager("no")

    def test_biocontext_is_disabled(self):
        from omicverse.utils.smart_agent import OmicVerseAgent
        assert OmicVerseAgent._biocontext_is_disabled(False)
        assert OmicVerseAgent._biocontext_is_disabled("no")
        assert OmicVerseAgent._biocontext_is_disabled("false")
        assert not OmicVerseAgent._biocontext_is_disabled("auto")
        assert not OmicVerseAgent._biocontext_is_disabled(True)
