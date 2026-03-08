"""Tests for artifact deletion and cleanup."""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from omicverse.mcp.session_store import SessionStore


# ---------------------------------------------------------------------------
# delete_artifact
# ---------------------------------------------------------------------------


class TestDeleteArtifact:
    def test_delete_handle_only(self):
        store = SessionStore()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"keep me")
            path = f.name
        try:
            aid = store.create_artifact(path, "text/plain")
            result = store.delete_artifact(aid, delete_file=False)
            assert result["deleted_handle"] is True
            assert result["deleted_file"] is False
            assert os.path.isfile(path)  # file still exists
            assert aid not in store._session_artifacts()
        finally:
            os.unlink(path)

    def test_delete_with_file(self):
        store = SessionStore()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"delete me")
            path = f.name
        aid = store.create_artifact(path, "text/plain")
        result = store.delete_artifact(aid, delete_file=True)
        assert result["deleted_handle"] is True
        assert result["deleted_file"] is True
        assert not os.path.isfile(path)

    def test_delete_file_not_found_graceful(self):
        store = SessionStore()
        aid = store.create_artifact("/tmp/no_such_file_12345.txt", "text/plain")
        result = store.delete_artifact(aid, delete_file=True)
        assert result["deleted_handle"] is True
        assert result["deleted_file"] is False  # file didn't exist

    def test_delete_unknown_raises(self):
        store = SessionStore()
        with pytest.raises(KeyError):
            store.delete_artifact("artifact_nonexistent")

    def test_delete_records_event(self):
        store = SessionStore()
        aid = store.create_artifact("/tmp/x.txt", "text/plain")
        store.delete_artifact(aid)
        events = store.list_events(event_type="artifact_deleted")
        assert len(events) >= 1
        assert events[0]["details"]["artifact_id"] == aid


# ---------------------------------------------------------------------------
# cleanup_artifacts
# ---------------------------------------------------------------------------


class TestCleanupArtifacts:
    def test_dry_run_preview(self):
        store = SessionStore()
        store.create_artifact("/tmp/a.txt", "text/plain")
        store.create_artifact("/tmp/b.txt", "text/plain")
        result = store.cleanup_artifacts(dry_run=True)
        assert result["dry_run"] is True
        assert result["matched"] == 2
        assert result["deleted"] == 0
        # Artifacts still exist
        assert len(store.list_artifacts()) == 2

    def test_actual_cleanup(self):
        store = SessionStore()
        store.create_artifact("/tmp/a.txt", "text/plain")
        store.create_artifact("/tmp/b.txt", "text/plain")
        result = store.cleanup_artifacts(dry_run=False)
        assert result["dry_run"] is False
        assert result["matched"] == 2
        assert result["deleted"] == 2
        assert len(store.list_artifacts()) == 0

    def test_filter_by_type(self):
        store = SessionStore()
        store.create_artifact("/tmp/a.png", "image/png", artifact_type="image")
        store.create_artifact("/tmp/b.csv", "text/csv", artifact_type="table")
        result = store.cleanup_artifacts(artifact_type="image", dry_run=False)
        assert result["matched"] == 1
        assert result["deleted"] == 1
        remaining = store.list_artifacts()
        assert len(remaining) == 1
        assert remaining[0]["artifact_type"] == "table"

    def test_filter_by_age(self):
        store = SessionStore()
        # Create one artifact and make it "old"
        aid_old = store.create_artifact("/tmp/old.txt", "text/plain")
        h = store.get_artifact(aid_old)
        h.created_at = time.time() - 200
        # Create one fresh
        store.create_artifact("/tmp/new.txt", "text/plain")
        result = store.cleanup_artifacts(older_than_seconds=100, dry_run=False)
        assert result["matched"] == 1
        assert result["deleted"] == 1
        remaining = store.list_artifacts()
        assert len(remaining) == 1

    def test_cleanup_with_file_deletion(self):
        store = SessionStore()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"bye")
            path = f.name
        store.create_artifact(path, "text/plain")
        result = store.cleanup_artifacts(delete_files=True, dry_run=False)
        assert result["deleted"] == 1
        assert result["items"][0]["deleted_file"] is True
        assert not os.path.isfile(path)

    def test_cleanup_records_event_and_metrics(self):
        store = SessionStore()
        store.create_artifact("/tmp/x.txt", "text/plain")
        store.cleanup_artifacts(dry_run=True)
        events = store.list_events(event_type="artifact_cleanup")
        assert len(events) >= 1
        metrics = store.get_metrics()
        assert metrics["artifact_cleanup_runs"] >= 1

    def test_cleanup_no_match(self):
        store = SessionStore()
        store.create_artifact("/tmp/a.png", "image/png", artifact_type="image")
        result = store.cleanup_artifacts(artifact_type="table", dry_run=False)
        assert result["matched"] == 0
        assert result["deleted"] == 0
        assert len(store.list_artifacts()) == 1
