"""Tests for SessionStore lifecycle."""

import time
import pytest
from omicverse.mcp.session_store import SessionStore


class TestAdataLifecycle:
    def test_create_and_get(self, session_store, mock_adata):
        adata_id = session_store.create_adata(mock_adata)
        assert adata_id.startswith("adata_")
        retrieved = session_store.get_adata(adata_id)
        assert retrieved is mock_adata

    def test_unknown_id_raises(self, session_store):
        with pytest.raises(KeyError):
            session_store.get_adata("adata_nonexistent")

    def test_update(self, session_store, mock_adata):
        adata_id = session_store.create_adata(mock_adata)
        from tests.mcp.conftest import _make_mock_adata
        new_adata = _make_mock_adata(50, 200)
        session_store.update_adata(adata_id, new_adata)
        retrieved = session_store.get_adata(adata_id)
        assert retrieved is new_adata

    def test_update_unknown_raises(self, session_store, mock_adata):
        with pytest.raises(KeyError):
            session_store.update_adata("adata_nonexistent", mock_adata)

    def test_delete(self, session_store, mock_adata):
        adata_id = session_store.create_adata(mock_adata)
        session_store.delete_handle(adata_id)
        with pytest.raises(KeyError):
            session_store.get_adata(adata_id)

    def test_list_adata(self, session_store, mock_adata):
        session_store.create_adata(mock_adata)
        listing = session_store.list_adata()
        assert len(listing) == 1
        assert "adata_id" in listing[0]
        assert "metadata" in listing[0]

    def test_metadata_auto_shape(self, session_store, mock_adata):
        adata_id = session_store.create_adata(mock_adata)
        listing = session_store.list_adata()
        assert listing[0]["metadata"]["shape"] == [100, 500]


class TestArtifactLifecycle:
    def test_create_and_get(self, session_store):
        art_id = session_store.create_artifact("/tmp/test.png", "image/png")
        assert art_id.startswith("artifact_")
        handle = session_store.get_artifact(art_id)
        assert handle.path == "/tmp/test.png"
        assert handle.content_type == "image/png"

    def test_unknown_artifact_raises(self, session_store):
        with pytest.raises(KeyError):
            session_store.get_artifact("artifact_nonexistent")


class TestInstanceLifecycle:
    def test_create_and_get(self, session_store):
        obj = {"type": "mock_instance"}
        inst_id = session_store.create_instance(obj, "MockClass")
        assert inst_id.startswith("inst_")
        retrieved = session_store.get_instance(inst_id)
        assert retrieved is obj

    def test_unknown_instance_raises(self, session_store):
        with pytest.raises(KeyError):
            session_store.get_instance("inst_nonexistent")


class TestCleanup:
    def test_cleanup_removes_expired(self, session_store, mock_adata):
        adata_id = session_store.create_adata(mock_adata)
        # Manually backdate (access nested dict via session_id)
        session_store._adata[session_store.session_id][adata_id].last_accessed = time.time() - 7200
        removed = session_store.cleanup_expired(max_age_seconds=3600)
        assert removed == 1
        with pytest.raises(KeyError):
            session_store.get_adata(adata_id)

    def test_cleanup_keeps_recent(self, session_store, mock_adata):
        session_store.create_adata(mock_adata)
        removed = session_store.cleanup_expired(max_age_seconds=3600)
        assert removed == 0
        assert session_store.stats["adata_count"] == 1


class TestDeleteHandle:
    def test_delete_unknown_raises(self, session_store):
        with pytest.raises(KeyError):
            session_store.delete_handle("unknown_id")


class TestStats:
    def test_empty_stats(self, session_store):
        s = session_store.stats
        assert s == {"session_id": "default", "adata_count": 0, "artifact_count": 0, "instance_count": 0}

    def test_stats_after_create(self, session_store, mock_adata):
        session_store.create_adata(mock_adata)
        session_store.create_artifact("/tmp/x.png", "image/png")
        s = session_store.stats
        assert s["adata_count"] == 1
        assert s["artifact_count"] == 1
