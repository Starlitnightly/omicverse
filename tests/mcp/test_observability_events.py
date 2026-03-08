"""Tests for structured event recording."""

from __future__ import annotations

import tempfile

from omicverse.mcp.session_store import SessionStore
from omicverse.mcp.executor import McpExecutor
from omicverse.mcp.server import RegistryMcpServer


def _make_mock_server(mock_registry, **kwargs):
    from omicverse.mcp.manifest import build_registry_manifest

    manifest = build_registry_manifest(registry=mock_registry)
    srv = RegistryMcpServer.__new__(RegistryMcpServer)
    srv._phase = "P0+P0.5"
    srv._store = SessionStore(**kwargs)
    srv._manifest = manifest
    srv._executor = McpExecutor(manifest, srv._store)
    return srv


# ---------------------------------------------------------------------------
# Event recording basics
# ---------------------------------------------------------------------------


class TestEventRecording:
    def test_record_event_returns_event_id(self):
        store = SessionStore()
        eid = store.record_event("test_event", {"key": "val"})
        assert eid.startswith("evt_")

    def test_event_has_required_fields(self):
        store = SessionStore()
        store.record_event("test_event", {"key": "val"})
        events = store.list_events()
        assert len(events) == 1
        e = events[0]
        assert "event_id" in e
        assert "event_type" in e
        assert "session_id" in e
        assert "timestamp" in e
        assert "details" in e

    def test_adata_created_event(self):
        store = SessionStore()
        from tests.mcp.conftest import _make_mock_adata
        store.create_adata(_make_mock_adata(5, 5))
        events = store.list_events(event_type="adata_created")
        assert len(events) == 1
        assert "adata_id" in events[0]["details"]

    def test_artifact_registered_event(self):
        store = SessionStore()
        store.create_artifact("/tmp/x.png", "image/png")
        events = store.list_events(event_type="artifact_registered")
        assert len(events) == 1
        assert events[0]["details"]["path"] == "/tmp/x.png"

    def test_instance_created_event(self):
        store = SessionStore()
        store.create_instance({"x": 1}, "FakeClass")
        events = store.list_events(event_type="instance_created")
        assert len(events) == 1
        assert events[0]["details"]["class_name"] == "FakeClass"

    def test_adata_persisted_event(self):
        persist_dir = tempfile.mkdtemp(prefix="ov_test_")
        store = SessionStore(persist_dir=persist_dir)
        from tests.mcp.conftest import _make_mock_adata
        adata_id = store.create_adata(_make_mock_adata(5, 5))
        store.persist_adata(adata_id)
        events = store.list_events(event_type="adata_persisted")
        assert len(events) == 1
        assert events[0]["details"]["adata_id"] == adata_id

    def test_adata_restored_event(self):
        from tests.mcp.conftest import _make_mock_adata
        from unittest.mock import patch
        import types

        persist_dir = tempfile.mkdtemp(prefix="ov_test_")
        store = SessionStore(persist_dir=persist_dir)
        adata_id = store.create_adata(_make_mock_adata(5, 5))
        result = store.persist_adata(adata_id)

        mock_ad = types.ModuleType("anndata")
        mock_ad.read_h5ad = lambda p: _make_mock_adata(5, 5)
        with patch.dict("sys.modules", {"anndata": mock_ad}):
            store.restore_adata(result["path"])

        events = store.list_events(event_type="adata_restored")
        assert len(events) >= 1

    def test_instance_destroyed_event(self):
        store = SessionStore()
        from tests.mcp.conftest import _make_mock_adata
        adata_id = store.create_adata(_make_mock_adata(5, 5))
        store.delete_handle(adata_id)
        events = store.list_events(event_type="instance_destroyed")
        assert len(events) == 1
        assert events[0]["details"]["handle_id"] == adata_id


# ---------------------------------------------------------------------------
# Event filtering
# ---------------------------------------------------------------------------


class TestEventFiltering:
    def test_list_events_default_limit(self):
        store = SessionStore()
        for i in range(60):
            store.record_event("test_event", {"i": i})
        events = store.list_events()
        assert len(events) == 50  # default limit

    def test_list_events_by_type(self):
        store = SessionStore()
        store.record_event("type_a", {"x": 1})
        store.record_event("type_b", {"x": 2})
        store.record_event("type_a", {"x": 3})
        events = store.list_events(event_type="type_a")
        assert len(events) == 2
        assert all(e["event_type"] == "type_a" for e in events)

    def test_list_events_by_tool_name(self):
        store = SessionStore()
        store.record_event("tool_called", {"tool_name": "ov.pp.pca", "ok": True})
        store.record_event("tool_called", {"tool_name": "ov.pp.qc", "ok": True})
        events = store.list_events(tool_name="ov.pp.pca")
        assert len(events) == 1
        assert events[0]["details"]["tool_name"] == "ov.pp.pca"

    def test_list_events_most_recent_first(self):
        store = SessionStore()
        store.record_event("evt", {"order": 1})
        store.record_event("evt", {"order": 2})
        store.record_event("evt", {"order": 3})
        events = store.list_events()
        assert events[0]["details"]["order"] == 3
        assert events[2]["details"]["order"] == 1


# ---------------------------------------------------------------------------
# Event isolation
# ---------------------------------------------------------------------------


class TestEventIsolation:
    def test_events_scoped_to_store_instance(self):
        store_a = SessionStore(session_id="A")
        store_b = SessionStore(session_id="B")
        store_a.record_event("test", {"from": "A"})
        assert len(store_a.list_events()) == 1
        assert len(store_b.list_events()) == 0
