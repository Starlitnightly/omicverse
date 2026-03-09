"""Tests for cleanup_runtime unified dispatcher."""

import os
import tempfile
import time

import pytest

from omicverse.mcp.session_store import (
    SessionStore, RuntimeLimits, TraceRecord,
)


class TestCleanupRuntime:
    def test_cleanup_all_dry_run(self):
        store = SessionStore(limits=RuntimeLimits(
            event_ttl_seconds=60,
            trace_ttl_seconds=60,
        ))
        # Add old events
        store.record_event("old", {})
        for e in store._events:
            e.timestamp = time.time() - 120
        # Add old trace
        now = time.time()
        store.record_trace(TraceRecord(
            trace_id="t1", session_id="default",
            tool_name="x", tool_type="meta",
            started_at=now - 120, finished_at=now - 120,
            duration_ms=1.0, ok=True,
        ))

        result = store.cleanup_runtime(target="all", dry_run=True)
        assert result["dry_run"] is True
        assert result["total_deleted"] == 0
        assert "events" in result["results"]
        assert "traces" in result["results"]

    def test_cleanup_all_execute(self):
        store = SessionStore(limits=RuntimeLimits(
            event_ttl_seconds=60,
            trace_ttl_seconds=60,
        ))
        # Add old events
        for _ in range(3):
            store.record_event("old", {})
        for e in store._events:
            e.timestamp = time.time() - 120

        result = store.cleanup_runtime(target="all", dry_run=False)
        assert result["dry_run"] is False
        assert result["total_deleted"] >= 3
        assert store._obs_metrics["cleanup_runs_total"] >= 1
        assert store._last_cleanup_at is not None

    def test_cleanup_events_only(self):
        store = SessionStore(limits=RuntimeLimits(event_ttl_seconds=60))
        store.record_event("old", {})
        for e in store._events:
            e.timestamp = time.time() - 120
        result = store.cleanup_runtime(target="events", dry_run=False)
        assert "events" in result["results"]
        assert "traces" not in result["results"]

    def test_cleanup_traces_only(self):
        store = SessionStore(limits=RuntimeLimits(trace_ttl_seconds=60))
        result = store.cleanup_runtime(target="traces", dry_run=False)
        assert "traces" in result["results"]
        assert "events" not in result["results"]

    def test_cleanup_all_skips_artifacts_when_no_ttl(self):
        store = SessionStore(limits=RuntimeLimits(
            event_ttl_seconds=60,
            artifact_ttl_seconds=None,
        ))
        store.create_artifact("/some/file.txt")
        result = store.cleanup_runtime(target="all", dry_run=False)
        # Artifacts should NOT be in results (no TTL set)
        assert "artifacts" not in result["results"]
        # Artifact should still exist
        assert len(store._session_artifacts()) == 1

    def test_cleanup_artifacts_explicit_target(self):
        store = SessionStore(limits=RuntimeLimits(artifact_ttl_seconds=60))
        store.create_artifact("/a.txt")
        # Backdate
        for a in store._session_artifacts().values():
            a.created_at = time.time() - 120
        result = store.cleanup_runtime(target="artifacts", dry_run=False)
        assert "artifacts" in result["results"]
        assert result["results"]["artifacts"]["deleted"] >= 1

    def test_cleanup_records_event_and_metrics(self):
        store = SessionStore(limits=RuntimeLimits(event_ttl_seconds=60))
        store.cleanup_runtime(target="events", dry_run=False)
        events = store.list_events(event_type="runtime_cleanup")
        assert len(events) >= 1
        m = store.get_metrics()
        assert m["cleanup_runs_total"] >= 1


class TestCleanupArtifactFiles:
    def test_cleanup_does_not_delete_files_by_default(self, tmp_path):
        f = tmp_path / "keep.txt"
        f.write_text("data")
        store = SessionStore(limits=RuntimeLimits(artifact_ttl_seconds=60))
        store.create_artifact(str(f))
        for a in store._session_artifacts().values():
            a.created_at = time.time() - 120
        store.cleanup_runtime(target="artifacts", dry_run=False, delete_files=False)
        assert f.exists()

    def test_cleanup_deletes_files_when_requested(self, tmp_path):
        f = tmp_path / "delete_me.txt"
        f.write_text("data")
        store = SessionStore(limits=RuntimeLimits(artifact_ttl_seconds=60))
        store.create_artifact(str(f))
        for a in store._session_artifacts().values():
            a.created_at = time.time() - 120
        store.cleanup_runtime(target="artifacts", dry_run=False, delete_files=True)
        assert not f.exists()
