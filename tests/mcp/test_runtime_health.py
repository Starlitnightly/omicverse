"""Tests for ov.get_limits, ov.get_health, and new meta tools presence."""

from omicverse.mcp.session_store import SessionStore, RuntimeLimits
from omicverse.mcp.server import RegistryMcpServer, META_TOOLS
from omicverse.mcp.manifest import build_registry_manifest
from omicverse.mcp.executor import McpExecutor
from tests.mcp.conftest import build_mock_registry


def _make_mock_server(limits=None, **kwargs):
    reg = build_mock_registry()
    manifest = build_registry_manifest(registry=reg)
    srv = RegistryMcpServer.__new__(RegistryMcpServer)
    srv._phase = "P0+P0.5"
    srv._store = SessionStore(limits=limits, **kwargs)
    srv._manifest = manifest
    srv._executor = McpExecutor(manifest, srv._store)
    return srv


class TestGetLimits:
    def test_get_limits_returns_config(self):
        srv = _make_mock_server(limits=RuntimeLimits(max_adata_per_session=10))
        result = srv.call_tool("ov.get_limits", {})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["quotas"]["max_adata_per_session"] == 10
        assert data["quotas"]["max_artifacts_per_session"] == 200
        assert data["ttl"]["session_ttl_seconds"] is None
        assert data["usage"]["adata_count"] == 0

    def test_get_limits_reflects_usage(self):
        srv = _make_mock_server(limits=RuntimeLimits(max_adata_per_session=10))

        class _Fake:
            shape = (10, 5)

        srv._store.create_adata(_Fake())
        result = srv.call_tool("ov.get_limits", {})
        data = result["outputs"][0]["data"]
        assert data["usage"]["adata_count"] == 1


class TestGetHealth:
    def test_get_health_returns_summary(self):
        srv = _make_mock_server()
        result = srv.call_tool("ov.get_health", {})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert "session_id" in data
        assert "session_age_seconds" in data
        assert data["session_expired"] is False
        assert "handles" in data
        assert "warnings" in data

    def test_get_health_quota_warnings(self):
        srv = _make_mock_server(limits=RuntimeLimits(max_adata_per_session=5))

        class _Fake:
            shape = (10, 5)

        # Fill to >80% (5 slots, need 5 to be at 100%)
        for _ in range(5):
            srv._store.create_adata(_Fake())
        result = srv.call_tool("ov.get_health", {})
        data = result["outputs"][0]["data"]
        assert len(data["warnings"]) >= 1
        assert "adata" in data["warnings"][0]

    def test_get_health_no_warnings_under_threshold(self):
        srv = _make_mock_server(limits=RuntimeLimits(max_adata_per_session=50))
        result = srv.call_tool("ov.get_health", {})
        data = result["outputs"][0]["data"]
        assert data["warnings"] == []

    def test_get_health_shows_session_expired(self):
        import time
        srv = _make_mock_server(limits=RuntimeLimits(session_ttl_seconds=1))
        srv._store._created_at = time.time() - 10
        # Note: call_tool will reject with session_expired before reaching handler.
        # But we can test check_session_expired directly.
        assert srv._store.check_session_expired() is True


class TestCleanupRuntime:
    def test_cleanup_runtime_via_server(self):
        srv = _make_mock_server(limits=RuntimeLimits(event_ttl_seconds=60))
        result = srv.call_tool("ov.cleanup_runtime", {"dry_run": True})
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["dry_run"] is True
        assert "results" in data

    def test_cleanup_runtime_target_events(self):
        srv = _make_mock_server(limits=RuntimeLimits(event_ttl_seconds=60))
        result = srv.call_tool("ov.cleanup_runtime", {
            "target": "events", "dry_run": True
        })
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["target"] == "events"


class TestNewMetaToolsPresent:
    def test_new_tools_in_list_tools(self):
        srv = _make_mock_server()
        tools = srv.list_tools()
        names = {t["name"] for t in tools}
        assert "ov.get_limits" in names
        assert "ov.cleanup_runtime" in names
        assert "ov.get_health" in names

    def test_new_tools_have_schema(self):
        for name in ("ov.get_limits", "ov.cleanup_runtime", "ov.get_health"):
            assert name in META_TOOLS
            assert "inputSchema" in META_TOOLS[name]
            assert "description" in META_TOOLS[name]
