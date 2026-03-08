"""Tests for artifact manifest export."""

from __future__ import annotations

import os
import tempfile

from omicverse.mcp.session_store import SessionStore


class TestExportManifest:
    def test_empty_session(self):
        store = SessionStore()
        manifest = store.export_artifacts_manifest()
        assert manifest["artifact_count"] == 0
        assert manifest["artifacts"] == []

    def test_manifest_structure(self):
        store = SessionStore()
        store.create_artifact("/tmp/a.png", "image/png", artifact_type="image")
        manifest = store.export_artifacts_manifest()
        assert "session_id" in manifest
        assert "exported_at" in manifest
        assert "artifact_count" in manifest
        assert "artifacts" in manifest

    def test_artifact_count_correct(self):
        store = SessionStore()
        store.create_artifact("/tmp/a.png", "image/png")
        store.create_artifact("/tmp/b.csv", "text/csv")
        store.create_artifact("/tmp/c.json", "application/json")
        manifest = store.export_artifacts_manifest()
        assert manifest["artifact_count"] == 3
        assert len(manifest["artifacts"]) == 3

    def test_file_exists_field(self):
        store = SessionStore()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"real file")
            real_path = f.name
        try:
            store.create_artifact(real_path, "text/plain")
            store.create_artifact("/tmp/fake_12345.txt", "text/plain")
            manifest = store.export_artifacts_manifest()
            by_path = {a["path"]: a for a in manifest["artifacts"]}
            assert by_path[real_path]["file_exists"] is True
            assert by_path["/tmp/fake_12345.txt"]["file_exists"] is False
        finally:
            os.unlink(real_path)

    def test_records_event(self):
        store = SessionStore()
        store.create_artifact("/tmp/a.png", "image/png")
        store.export_artifacts_manifest()
        events = store.list_events(event_type="artifact_manifest_exported")
        assert len(events) == 1
        assert events[0]["details"]["artifact_count"] == 1

    def test_session_id_in_manifest(self):
        store = SessionStore(session_id="my_session")
        store.create_artifact("/tmp/a.png", "image/png")
        manifest = store.export_artifacts_manifest()
        assert manifest["session_id"] == "my_session"
