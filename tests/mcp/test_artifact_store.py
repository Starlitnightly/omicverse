"""Tests for enhanced artifact model and SessionStore artifact methods."""

from __future__ import annotations

import os
import tempfile

import pytest

from omicverse.mcp.session_store import SessionStore, SessionError


# ---------------------------------------------------------------------------
# Artifact model
# ---------------------------------------------------------------------------


class TestArtifactModel:
    def test_default_fields(self):
        store = SessionStore()
        aid = store.create_artifact("/tmp/test.png", "image/png")
        h = store.get_artifact(aid)
        assert h.artifact_type == "file"
        assert h.source_tool == ""
        assert h.updated_at > 0
        assert h.session_id == "default"

    def test_create_with_artifact_type(self):
        store = SessionStore()
        aid = store.create_artifact("/tmp/test.png", "image/png", artifact_type="image")
        h = store.get_artifact(aid)
        assert h.artifact_type == "image"

    def test_create_with_source_tool(self):
        store = SessionStore()
        aid = store.create_artifact(
            "/tmp/test.png", "image/png",
            source_tool="ov.pl.umap",
        )
        h = store.get_artifact(aid)
        assert h.source_tool == "ov.pl.umap"

    def test_create_backward_compat(self):
        """Old call signature (path, content_type) still works."""
        store = SessionStore()
        aid = store.create_artifact("/tmp/test.csv", "text/csv")
        h = store.get_artifact(aid)
        assert h.artifact_type == "file"
        assert h.source_tool == ""

    def test_updated_at_equals_created_at_on_creation(self):
        store = SessionStore()
        aid = store.create_artifact("/tmp/test.txt", "text/plain")
        h = store.get_artifact(aid)
        assert h.updated_at == h.created_at


# ---------------------------------------------------------------------------
# list_artifacts
# ---------------------------------------------------------------------------


class TestListArtifacts:
    def test_list_empty(self):
        store = SessionStore()
        assert store.list_artifacts() == []

    def test_list_returns_all(self):
        store = SessionStore()
        store.create_artifact("/a.png", "image/png", artifact_type="image")
        store.create_artifact("/b.csv", "text/csv", artifact_type="table")
        result = store.list_artifacts()
        assert len(result) == 2

    def test_filter_by_artifact_type(self):
        store = SessionStore()
        store.create_artifact("/a.png", "image/png", artifact_type="image")
        store.create_artifact("/b.csv", "text/csv", artifact_type="table")
        store.create_artifact("/c.png", "image/png", artifact_type="image")
        result = store.list_artifacts(artifact_type="image")
        assert len(result) == 2
        assert all(a["artifact_type"] == "image" for a in result)

    def test_filter_by_content_type(self):
        store = SessionStore()
        store.create_artifact("/a.png", "image/png")
        store.create_artifact("/b.csv", "text/csv")
        result = store.list_artifacts(content_type="text/csv")
        assert len(result) == 1
        assert result[0]["content_type"] == "text/csv"

    def test_filter_by_source_tool(self):
        store = SessionStore()
        store.create_artifact("/a.png", "image/png", source_tool="ov.pl.umap")
        store.create_artifact("/b.png", "image/png", source_tool="ov.pl.pca")
        result = store.list_artifacts(source_tool="ov.pl.umap")
        assert len(result) == 1
        assert result[0]["source_tool"] == "ov.pl.umap"

    def test_limit(self):
        store = SessionStore()
        for i in range(10):
            store.create_artifact(f"/tmp/{i}.txt", "text/plain")
        result = store.list_artifacts(limit=3)
        assert len(result) == 3

    def test_most_recent_first(self):
        store = SessionStore()
        aid1 = store.create_artifact("/a.txt", "text/plain")
        aid2 = store.create_artifact("/b.txt", "text/plain")
        result = store.list_artifacts()
        assert result[0]["artifact_id"] == aid2
        assert result[1]["artifact_id"] == aid1


# ---------------------------------------------------------------------------
# describe_artifact
# ---------------------------------------------------------------------------


class TestDescribeArtifact:
    def test_describe_returns_full_metadata(self):
        store = SessionStore()
        aid = store.create_artifact(
            "/tmp/nonexistent.png", "image/png",
            artifact_type="image", source_tool="ov.pl.umap",
            metadata={"dpi": 150},
        )
        desc = store.describe_artifact(aid)
        assert desc["artifact_id"] == aid
        assert desc["artifact_type"] == "image"
        assert desc["source_tool"] == "ov.pl.umap"
        assert desc["metadata"]["dpi"] == 150
        assert "file_exists" in desc
        assert "file_size_bytes" in desc

    def test_describe_file_exists(self):
        store = SessionStore()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            aid = store.create_artifact(path, "text/plain")
            desc = store.describe_artifact(aid)
            assert desc["file_exists"] is True
            assert desc["file_size_bytes"] == 5
        finally:
            os.unlink(path)

    def test_describe_file_not_found(self):
        store = SessionStore()
        aid = store.create_artifact("/tmp/definitely_not_here_12345.txt", "text/plain")
        desc = store.describe_artifact(aid)
        assert desc["file_exists"] is False
        assert desc["file_size_bytes"] == -1

    def test_describe_unknown_artifact_raises(self):
        store = SessionStore()
        with pytest.raises(KeyError):
            store.describe_artifact("artifact_nonexistent")

    def test_describe_cross_session_raises(self):
        store = SessionStore(session_id="A")
        aid = store.create_artifact("/tmp/x.png", "image/png")
        store._session_id = "B"
        store._artifacts.setdefault("B", {})
        with pytest.raises(SessionError) as exc_info:
            store.describe_artifact(aid)
        assert exc_info.value.error_code == "cross_session_access"
