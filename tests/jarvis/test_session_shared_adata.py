from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_shared_kernel_reuses_one_session_across_users(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(SessionManager, "_build_agent", lambda self, kernel_root: object())

    sm = SessionManager(session_dir=str(tmp_path), shared_kernel=True)

    first = sm.get_or_create(101)
    second = sm.get_or_create(202)

    assert first is second
    assert sm.get_active_kernel(101) == "main"
    assert sm.get_active_kernel(202) == "main"


def test_shared_kernel_executor_seeds_shared_adata(tmp_path: Path, monkeypatch) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.use_notebook_execution = False
            self._notebook_executor = None
            self.notebook_timeout = 222

    class FakeExecutor:
        def __init__(self) -> None:
            self.shell = SimpleNamespace(user_ns={"odata": object()})

        def _ensure_kernel(self) -> None:
            return

        def sync_adata(self, adata) -> None:
            self.shell.user_ns["odata"] = adata
            self.shell.user_ns["adata"] = adata

    monkeypatch.setattr(SessionManager, "_build_agent", lambda self, kernel_root: FakeAgent())
    executor = FakeExecutor()
    expected = executor.shell.user_ns["odata"]

    sm = SessionManager(session_dir=str(tmp_path), shared_kernel=True)
    sm.attach_shared_kernel_executor(executor)
    session = sm.get_or_create(101)

    assert sm.get_shared_adata() is expected
    assert session.adata is expected


def test_shared_kernel_executor_is_attached_to_agent(tmp_path: Path, monkeypatch) -> None:
    class FakeExecutor:
        def __init__(self) -> None:
            self.shell = SimpleNamespace(user_ns={})
            self.calls = []

        def _ensure_kernel(self) -> None:
            return

        def execute(self, code, adata=None, timeout=300):
            self.calls.append((code, adata, timeout))
            self.shell.user_ns["odata"] = adata
            self.shell.user_ns["adata"] = adata
            return {"adata": adata}

        def sync_adata(self, adata) -> None:
            self.shell.user_ns["odata"] = adata
            self.shell.user_ns["adata"] = adata

    class FakeAgent:
        def __init__(self) -> None:
            self.use_notebook_execution = False
            self._notebook_executor = None
            self.notebook_timeout = 321

    monkeypatch.setattr(SessionManager, "_build_agent", lambda self, kernel_root: FakeAgent())
    executor = FakeExecutor()
    shared = object()

    sm = SessionManager(session_dir=str(tmp_path), shared_kernel=True, shared_kernel_executor=executor)
    sm.set_shared_adata(shared)
    session = sm.get_or_create(101)

    assert session.agent.use_notebook_execution is True
    assert session.agent._notebook_executor is not None
    assert session.agent._notebook_executor.execute("x = 1", shared) is shared
    assert executor.calls == [("x = 1", shared, 321)]
