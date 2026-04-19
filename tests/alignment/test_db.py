"""Unit tests for ov.alignment._db reference-DB helpers."""
from __future__ import annotations

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
    """Do not actually download; patch urlopen to short-circuit."""
    from omicverse.alignment import _db
    # Prevent real HTTP
    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, *_, **__): return b""

    def fake_urlopen(url, timeout=None):
        return _FakeResp()

    monkeypatch.setattr(_db.urllib.request, "urlopen", fake_urlopen)
    # But we also need the file to appear to exist to short-circuit normally,
    # so call fetch_silva and catch the warning only.
    with pytest.warns(DeprecationWarning, match="v138"):
        try:
            _db.fetch_silva(db_dir=str(tmp_path))
        except Exception:
            pass  # the fake urlopen writes 0 bytes and verification may fail
