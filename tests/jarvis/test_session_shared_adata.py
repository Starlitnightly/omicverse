from __future__ import annotations

from pathlib import Path

from omicverse.jarvis.session import SessionManager


def test_shared_adata_propagates_to_existing_and_new_sessions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(SessionManager, "_build_agent", lambda self, kernel_root: object())

    sm = SessionManager(session_dir=str(tmp_path))

    first = sm.get_or_create(1)
    assert first.adata is None

    shared = object()
    sm.set_shared_adata(shared)

    assert sm.get_shared_adata() is shared
    assert first.adata is shared

    second = sm.get_or_create(2)
    assert second.adata is shared
