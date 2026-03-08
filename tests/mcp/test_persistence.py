"""Tests for adata persistence and restore."""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest
from unittest.mock import patch, MagicMock

from omicverse.mcp.session_store import (
    SessionStore, SessionError, AdataHandle, _sidecar_path,
)


# ---------------------------------------------------------------------------
# persist_adata
# ---------------------------------------------------------------------------


class TestPersistAdata:
    def test_persist_creates_file(self):
        store = SessionStore(persist_dir=tempfile.mkdtemp(prefix="ov_test_"))
        from tests.mcp.conftest import _make_mock_adata
        adata = _make_mock_adata(10, 20)
        adata_id = store.create_adata(adata)

        result = store.persist_adata(adata_id)
        assert os.path.isfile(result["path"])

    def test_persist_creates_sidecar_json(self):
        store = SessionStore(persist_dir=tempfile.mkdtemp(prefix="ov_test_"))
        from tests.mcp.conftest import _make_mock_adata
        adata = _make_mock_adata(10, 20)
        adata_id = store.create_adata(adata)

        result = store.persist_adata(adata_id)
        meta_path = result["metadata_path"]
        assert os.path.isfile(meta_path)
        with open(meta_path) as f:
            sidecar = json.load(f)
        assert sidecar["adata_id"] == adata_id

    def test_sidecar_has_required_fields(self):
        store = SessionStore(persist_dir=tempfile.mkdtemp(prefix="ov_test_"))
        from tests.mcp.conftest import _make_mock_adata
        adata_id = store.create_adata(_make_mock_adata(10, 20))

        result = store.persist_adata(adata_id)
        with open(result["metadata_path"]) as f:
            sidecar = json.load(f)
        assert "adata_id" in sidecar
        assert "session_id" in sidecar
        assert "file_path" in sidecar
        assert "content_type" in sidecar
        assert "created_at" in sidecar
        assert "persisted_at" in sidecar
        assert "original_metadata" in sidecar

    def test_persist_with_explicit_path(self):
        store = SessionStore()
        from tests.mcp.conftest import _make_mock_adata
        adata_id = store.create_adata(_make_mock_adata(10, 20))

        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = f.name

        try:
            result = store.persist_adata(adata_id, path=path)
            assert result["path"] == path
            assert os.path.isfile(path)
        finally:
            os.unlink(path)
            meta = _sidecar_path(path)
            if os.path.isfile(meta):
                os.unlink(meta)

    def test_persist_auto_creates_persist_dir(self):
        store = SessionStore()  # no persist_dir
        from tests.mcp.conftest import _make_mock_adata
        adata_id = store.create_adata(_make_mock_adata(10, 20))

        result = store.persist_adata(adata_id)
        assert store._persist_dir is not None
        assert os.path.isdir(store._persist_dir)
        assert os.path.isfile(result["path"])

    def test_persist_unknown_adata_raises(self):
        store = SessionStore()
        with pytest.raises(KeyError):
            store.persist_adata("adata_nonexistent")

    def test_persist_cross_session_raises(self):
        store = SessionStore(session_id="A")
        store._adata["B"] = {
            "adata_foreign": AdataHandle(
                adata_id="adata_foreign", obj="fake",
                created_at=time.time(), last_accessed=time.time(),
            ),
        }
        with pytest.raises(SessionError) as exc_info:
            store.persist_adata("adata_foreign")
        assert exc_info.value.error_code == "cross_session_access"


# ---------------------------------------------------------------------------
# restore_adata
# ---------------------------------------------------------------------------


class TestRestoreAdata:
    def test_restore_nonexistent_file_raises(self):
        store = SessionStore()
        with pytest.raises(FileNotFoundError):
            store.restore_adata("/nonexistent/path.h5ad")

    def test_restore_roundtrip(self):
        """Persist then restore using MockAnnData."""
        store = SessionStore(persist_dir=tempfile.mkdtemp(prefix="ov_test_"))
        from tests.mcp.conftest import _make_mock_adata
        adata = _make_mock_adata(10, 20)
        adata_id = store.create_adata(adata)

        result = store.persist_adata(adata_id)
        path = result["path"]

        # restore_adata requires anndata.read_h5ad — mock it
        mock_restored = _make_mock_adata(10, 20)
        with patch("omicverse.mcp.session_store.SessionStore.restore_adata") as mock_restore:
            # Don't mock — test the real path with anndata mocked
            pass

        # Actually test with a direct mock of anndata import
        import types
        mock_ad = types.ModuleType("anndata")
        mock_ad.read_h5ad = lambda p: _make_mock_adata(10, 20)

        with patch.dict("sys.modules", {"anndata": mock_ad}):
            new_id = store.restore_adata(path)

        assert new_id.startswith("adata_")
        assert new_id != adata_id  # new handle generated

    def test_restore_reads_sidecar_metadata(self):
        store = SessionStore(persist_dir=tempfile.mkdtemp(prefix="ov_test_"))
        from tests.mcp.conftest import _make_mock_adata
        adata_id = store.create_adata(_make_mock_adata(10, 20))
        result = store.persist_adata(adata_id)
        path = result["path"]

        import types
        mock_ad = types.ModuleType("anndata")
        mock_ad.read_h5ad = lambda p: _make_mock_adata(10, 20)

        with patch.dict("sys.modules", {"anndata": mock_ad}):
            new_id = store.restore_adata(path)

        handle = store._session_adata()[new_id]
        assert "restored_from" in handle.metadata
        assert handle.metadata["restored_from"] == path

    def test_restore_without_sidecar(self):
        """Restore a file that has no .meta.json sidecar."""
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            # Write a dummy file
            f.write(b"dummy")
            path = f.name

        try:
            store = SessionStore()
            from tests.mcp.conftest import _make_mock_adata
            import types
            mock_ad = types.ModuleType("anndata")
            mock_ad.read_h5ad = lambda p: _make_mock_adata(10, 20)

            with patch.dict("sys.modules", {"anndata": mock_ad}):
                new_id = store.restore_adata(path)

            assert new_id.startswith("adata_")
            handle = store._session_adata()[new_id]
            assert handle.metadata.get("restored_from") == path
        finally:
            os.unlink(path)

    def test_restore_generates_new_id(self):
        """Without explicit adata_id, generates a new one."""
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            f.write(b"dummy")
            path = f.name

        try:
            store = SessionStore()
            from tests.mcp.conftest import _make_mock_adata
            import types
            mock_ad = types.ModuleType("anndata")
            mock_ad.read_h5ad = lambda p: _make_mock_adata(10, 20)

            with patch.dict("sys.modules", {"anndata": mock_ad}):
                id1 = store.restore_adata(path)
                id2 = store.restore_adata(path)

            assert id1 != id2
            assert store.stats["adata_count"] == 2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Instance non-persistence
# ---------------------------------------------------------------------------


class TestInstanceNonPersistence:
    def test_instance_not_in_persist_scope(self):
        """persist_adata rejects instance handles."""
        store = SessionStore()
        inst_id = store.create_instance({"obj": True}, "FakeClass")
        with pytest.raises(KeyError):
            store.persist_adata(inst_id)  # not an adata handle

    def test_instance_id_documented_as_ephemeral(self):
        """Instance handles have no write_h5ad or persistence method."""
        store = SessionStore()
        inst_id = store.create_instance({"obj": True}, "FakeClass")
        obj = store.get_instance(inst_id)
        assert not hasattr(obj, "write_h5ad")


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


class TestArtifactMetadata:
    def test_artifact_has_metadata_fields(self):
        store = SessionStore()
        art_id = store.create_artifact(
            "/tmp/plot.png", "image/png",
            metadata={"tool": "ov.pl.embedding"},
        )
        handle = store.get_artifact(art_id)
        assert handle.artifact_id == art_id
        assert handle.path == "/tmp/plot.png"
        assert handle.content_type == "image/png"
        assert handle.created_at > 0
        assert handle.metadata["tool"] == "ov.pl.embedding"

    def test_artifact_path_and_content_type_preserved(self):
        store = SessionStore()
        art_id = store.create_artifact("/data/table.csv", "text/csv")
        handle = store.get_artifact(art_id)
        assert handle.path == "/data/table.csv"
        assert handle.content_type == "text/csv"


# ---------------------------------------------------------------------------
# Sidecar path helper
# ---------------------------------------------------------------------------


class TestSidecarPath:
    def test_h5ad_extension(self):
        assert _sidecar_path("/tmp/data.h5ad") == "/tmp/data.meta.json"

    def test_no_extension(self):
        assert _sidecar_path("/tmp/data") == "/tmp/data.meta.json"
