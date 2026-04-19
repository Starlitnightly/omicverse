"""Unit tests for ov.alignment._db reference-DB helpers."""
from __future__ import annotations

import io
import os
import pytest


def test_resolve_db_dir_rejects_none_and_empty(monkeypatch):
    from omicverse.alignment._db import _resolve_db_dir
    # Clear env var
    monkeypatch.delenv("OMICVERSE_DB_DIR", raising=False)
    with pytest.raises(ValueError, match="db_dir must be specified"):
        _resolve_db_dir(None)
    with pytest.raises(ValueError):
        _resolve_db_dir("")
    with pytest.raises(ValueError):
        _resolve_db_dir("   ")


def test_resolve_db_dir_env_fallback(tmp_path, monkeypatch):
    from omicverse.alignment._db import _resolve_db_dir
    monkeypatch.setenv("OMICVERSE_DB_DIR", str(tmp_path))
    out = _resolve_db_dir(None)
    assert out == tmp_path.resolve()


def test_resolve_db_dir_explicit_wins(tmp_path, monkeypatch):
    from omicverse.alignment._db import _resolve_db_dir
    monkeypatch.setenv("OMICVERSE_DB_DIR", "/tmp/should-not-win")
    out = _resolve_db_dir(str(tmp_path))
    assert out == tmp_path.resolve()


def test_fetch_silva_emits_deprecation_warning(monkeypatch, tmp_path):
    """fetch_silva must emit DeprecationWarning BEFORE any download attempt."""
    import io
    from omicverse.alignment import _db

    # Stub urlopen to return an empty-bytes response — just enough to let
    # the download finish so we can inspect what ended up on disk.
    class _FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_urlopen(url, timeout=None):
        return _FakeResp(b"")

    monkeypatch.setattr(_db.urllib.request, "urlopen", fake_urlopen)

    with pytest.warns(DeprecationWarning, match="v138"):
        _db.fetch_silva(db_dir=str(tmp_path))

    # Even though we stubbed urlopen, the function wrote an empty file (no
    # sha256 is configured for the silva source so that path succeeds).
    # Assert the stale `.part` was cleaned up and the final file exists.
    target = tmp_path / "silva_16s_v123" / "silva_16s_v123.fa.gz"
    assert target.exists()
    assert not target.with_suffix(target.suffix + ".part").exists()


def test_download_cleans_up_part_on_error(monkeypatch, tmp_path):
    """Network error mid-download must not leave a stale .part behind."""
    from omicverse.alignment import _db

    class _Boom(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, *_, **__):
            raise ConnectionResetError("simulated mid-download failure")

    def fake_urlopen(url, timeout=None):
        return _Boom()

    monkeypatch.setattr(_db.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "file.gz"
    with pytest.raises(ConnectionResetError):
        _db._download("http://example.com/file.gz", dest)
    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".part").exists()
