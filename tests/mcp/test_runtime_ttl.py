"""Tests for TTL-based expiration and session TTL."""

import time

import pytest

from omicverse.mcp.session_store import (
    SessionStore, RuntimeLimits, TraceRecord,
)
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


class TestSessionTTL:
    def test_session_not_expired_by_default(self):
        store = SessionStore()
        assert store.check_session_expired() is False

    def test_session_not_expired_when_ttl_none(self):
        store = SessionStore(limits=RuntimeLimits(session_ttl_seconds=None))
        assert store.check_session_expired() is False

    def test_session_expired_after_ttl(self):
        store = SessionStore(limits=RuntimeLimits(session_ttl_seconds=1))
        # Backdate creation
        store._created_at = time.time() - 10
        assert store.check_session_expired() is True

    def test_session_not_expired_within_ttl(self):
        store = SessionStore(limits=RuntimeLimits(session_ttl_seconds=3600))
        assert store.check_session_expired() is False

    def test_session_expired_blocks_call_tool(self):
        srv = _make_mock_server(limits=RuntimeLimits(session_ttl_seconds=1))
        srv._store._created_at = time.time() - 10
        result = srv.call_tool("ov.list_tools", {})
        assert result["ok"] is False
        assert result["error_code"] == "session_expired"


class TestEventTraceTTL:
    def test_cleanup_events_with_ttl(self):
        store = SessionStore(limits=RuntimeLimits(event_ttl_seconds=60))
        # Add old events
        for i in range(5):
            store.record_event("old_event", {"i": i})
        # Backdate them
        for e in store._events:
            e.timestamp = time.time() - 120
        # Add fresh events
        for i in range(3):
            store.record_event("new_event", {"i": i})

        # Dry run
        result = store.cleanup_expired_events(dry_run=True)
        assert result["matched"] == 5
        assert result["deleted"] == 0
        assert len(store._events) == 8  # unchanged

        # Execute
        result = store.cleanup_expired_events(dry_run=False)
        assert result["matched"] == 5
        assert result["deleted"] == 5
        assert len(store._events) == 3  # only fresh remain

    def test_cleanup_events_no_ttl(self):
        store = SessionStore(limits=RuntimeLimits(event_ttl_seconds=None))
        store.record_event("test", {})
        result = store.cleanup_expired_events(dry_run=False)
        assert result["matched"] == 0
        assert result["deleted"] == 0

    def test_cleanup_traces_with_ttl(self):
        store = SessionStore(limits=RuntimeLimits(trace_ttl_seconds=60))
        now = time.time()
        # Old traces
        for i in range(4):
            store.record_trace(TraceRecord(
                trace_id=f"old{i}", session_id="default",
                tool_name="test", tool_type="meta",
                started_at=now - 120, finished_at=now - 120,
                duration_ms=1.0, ok=True,
            ))
        # Fresh trace
        store.record_trace(TraceRecord(
            trace_id="new0", session_id="default",
            tool_name="test", tool_type="meta",
            started_at=now, finished_at=now,
            duration_ms=1.0, ok=True,
        ))

        result = store.cleanup_expired_traces(dry_run=False)
        assert result["matched"] == 4
        assert result["deleted"] == 4
        assert len(store._traces) == 1
        assert store._traces[0].trace_id == "new0"

    def test_cleanup_traces_no_ttl(self):
        store = SessionStore(limits=RuntimeLimits(trace_ttl_seconds=None))
        result = store.cleanup_expired_traces(dry_run=False)
        assert result["matched"] == 0
