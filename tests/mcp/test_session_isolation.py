"""Tests for session isolation in SessionStore."""

from __future__ import annotations

import time
import pytest

from omicverse.mcp.session_store import (
    SessionStore, SessionError, AdataHandle, ArtifactHandle, InstanceHandle,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSessionConstruction:
    def test_default_session_id_is_default(self):
        store = SessionStore()
        assert store.session_id == "default"

    def test_custom_session_id(self):
        store = SessionStore(session_id="my-session")
        assert store.session_id == "my-session"

    def test_persist_dir_optional(self):
        store = SessionStore()
        assert store._persist_dir is None
        store2 = SessionStore(persist_dir="/tmp/test")
        assert store2._persist_dir == "/tmp/test"


# ---------------------------------------------------------------------------
# Cross-session access
# ---------------------------------------------------------------------------


class TestCrossSessionAccess:
    """Inject handles into a foreign session and verify cross-session errors."""

    def _make_store_with_foreign(self):
        """Create a store with a handle in a foreign session."""
        store = SessionStore(session_id="session-A")
        # Inject a handle into session-B
        now = time.time()
        store._adata["session-B"] = {
            "adata_foreign": AdataHandle(
                adata_id="adata_foreign", obj="fake",
                created_at=now, last_accessed=now,
                metadata={"shape": [10, 20]},
            ),
        }
        store._artifacts["session-B"] = {
            "artifact_foreign": ArtifactHandle(
                artifact_id="artifact_foreign", path="/tmp/x.png",
                content_type="image/png", created_at=now,
            ),
        }
        store._instances["session-B"] = {
            "inst_foreign": InstanceHandle(
                instance_id="inst_foreign", obj="fake_instance",
                class_name="FakeClass", created_at=now,
            ),
        }
        return store

    def test_adata_cross_session_raises_session_error(self):
        store = self._make_store_with_foreign()
        with pytest.raises(SessionError) as exc_info:
            store.get_adata("adata_foreign")
        assert exc_info.value.error_code == "cross_session_access"

    def test_artifact_cross_session_raises(self):
        store = self._make_store_with_foreign()
        with pytest.raises(SessionError) as exc_info:
            store.get_artifact("artifact_foreign")
        assert exc_info.value.error_code == "cross_session_access"

    def test_instance_cross_session_raises(self):
        store = self._make_store_with_foreign()
        with pytest.raises(SessionError) as exc_info:
            store.get_instance("inst_foreign")
        assert exc_info.value.error_code == "cross_session_access"

    def test_session_error_has_error_code(self):
        store = self._make_store_with_foreign()
        with pytest.raises(SessionError) as exc_info:
            store.get_adata("adata_foreign")
        assert hasattr(exc_info.value, "error_code")
        assert exc_info.value.error_code == "cross_session_access"

    def test_session_error_has_details(self):
        store = self._make_store_with_foreign()
        with pytest.raises(SessionError) as exc_info:
            store.get_adata("adata_foreign")
        details = exc_info.value.details
        assert details["owner_session"] == "session-B"
        assert details["current_session"] == "session-A"

    def test_delete_cross_session_raises(self):
        store = self._make_store_with_foreign()
        with pytest.raises(SessionError) as exc_info:
            store.delete_handle("adata_foreign")
        assert exc_info.value.error_code == "cross_session_access"

    def test_update_cross_session_raises(self):
        store = self._make_store_with_foreign()
        with pytest.raises(SessionError) as exc_info:
            store.update_adata("adata_foreign", "new_obj")
        assert exc_info.value.error_code == "cross_session_access"


# ---------------------------------------------------------------------------
# Session scoping
# ---------------------------------------------------------------------------


class TestSessionScoping:
    def test_list_adata_scoped_to_session(self):
        store = self._make_two_session_store()
        listing = store.list_adata()
        assert len(listing) == 1
        assert listing[0]["adata_id"] == "adata_own"

    def test_list_handles_scoped_to_session(self):
        store = self._make_two_session_store()
        handles = store.list_handles()
        handle_ids = [h["handle_id"] for h in handles]
        assert "adata_own" in handle_ids
        assert "adata_foreign" not in handle_ids

    def test_stats_scoped_to_session(self):
        store = self._make_two_session_store()
        s = store.stats
        assert s["adata_count"] == 1
        assert s["session_id"] == "session-A"

    def test_cleanup_works_across_sessions(self):
        store = self._make_two_session_store()
        # Backdate both sessions
        for handles in store._adata.values():
            for h in handles.values():
                h.last_accessed = time.time() - 7200
        removed = store.cleanup_expired(max_age_seconds=3600)
        assert removed == 2  # one from each session

    def test_handles_independent_between_sessions(self):
        """Two separate SessionStore instances have fully independent state."""
        store_a = SessionStore(session_id="A")
        store_b = SessionStore(session_id="B")
        from tests.mcp.conftest import _make_mock_adata
        adata = _make_mock_adata(10, 20)

        id_a = store_a.create_adata(adata)
        assert store_a.stats["adata_count"] == 1
        assert store_b.stats["adata_count"] == 0

        with pytest.raises(KeyError):
            store_b.get_adata(id_a)

    def _make_two_session_store(self):
        store = SessionStore(session_id="session-A")
        now = time.time()
        store._adata["session-A"]["adata_own"] = AdataHandle(
            adata_id="adata_own", obj="mine",
            created_at=now, last_accessed=now,
        )
        store._adata["session-B"] = {
            "adata_foreign": AdataHandle(
                adata_id="adata_foreign", obj="theirs",
                created_at=now, last_accessed=now,
            ),
        }
        return store


# ---------------------------------------------------------------------------
# Session info
# ---------------------------------------------------------------------------


class TestSessionInfo:
    def test_session_info_returns_correct_structure(self):
        store = SessionStore(session_id="test-info")
        info = store.session_info()
        assert info["session_id"] == "test-info"
        assert "handles" in info
        assert "stats" in info

    def test_session_info_includes_handles(self):
        store = SessionStore()
        from tests.mcp.conftest import _make_mock_adata
        adata_id = store.create_adata(_make_mock_adata(5, 5))
        info = store.session_info()
        assert adata_id in info["handles"]["adata"]

    def test_session_info_includes_persist_dir(self):
        store = SessionStore(persist_dir="/tmp/persist")
        info = store.session_info()
        assert info["persist_dir"] == "/tmp/persist"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_parameterless_constructor_works(self):
        store = SessionStore()
        assert store.session_id == "default"
        assert store.stats["adata_count"] == 0

    def test_create_get_cycle_unchanged(self):
        store = SessionStore()
        from tests.mcp.conftest import _make_mock_adata
        adata = _make_mock_adata(10, 20)
        adata_id = store.create_adata(adata)
        assert adata_id.startswith("adata_")
        retrieved = store.get_adata(adata_id)
        assert retrieved is adata

    def test_stats_shape_has_session_id(self):
        store = SessionStore()
        s = store.stats
        assert "session_id" in s
        assert "adata_count" in s
        assert "artifact_count" in s
        assert "instance_count" in s

    def test_existing_api_contract(self):
        """All public methods from the original API still exist."""
        store = SessionStore()
        assert callable(store.create_adata)
        assert callable(store.get_adata)
        assert callable(store.update_adata)
        assert callable(store.list_adata)
        assert callable(store.create_artifact)
        assert callable(store.get_artifact)
        assert callable(store.create_instance)
        assert callable(store.get_instance)
        assert callable(store.delete_handle)
        assert callable(store.cleanup_expired)

    def test_session_error_is_key_error(self):
        """SessionError is a subclass of KeyError for backward compat."""
        exc = SessionError("test_code", "test message")
        assert isinstance(exc, KeyError)
