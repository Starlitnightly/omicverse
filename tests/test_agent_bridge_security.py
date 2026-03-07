from pathlib import Path
from types import SimpleNamespace

from omicverse.jarvis.agent_bridge import AgentBridge


class _DummyAgent:
    def __init__(self, workspace_dir=None, session_dir=None):
        self._filesystem_context = SimpleNamespace(_workspace_dir=workspace_dir)
        self._notebook_executor = SimpleNamespace(current_session={"session_dir": session_dir} if session_dir else None)


def test_read_file_if_safe_rejects_symlink_and_out_of_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("top-secret", encoding="utf-8")

    symlink = workspace / "link.txt"
    symlink.symlink_to(outside)

    allowed_roots = AgentBridge._resolve_roots([workspace])
    assert AgentBridge._read_file_if_safe(symlink, allowed_roots, 1024) == b""


def test_read_file_if_safe_rejects_large_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    large = workspace / "large.txt"
    large.write_bytes(b"a" * 100)

    allowed_roots = AgentBridge._resolve_roots([workspace])
    assert AgentBridge._read_file_if_safe(large, allowed_roots, 32) == b""


def test_harvest_artifacts_only_scans_trusted_roots(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    untrusted = tmp_path / "untrusted"
    untrusted.mkdir()

    # File in cwd-like untrusted directory should not be considered after hardening.
    leaked = untrusted / "secret.txt"
    leaked.write_text("do-not-send", encoding="utf-8")

    safe = trusted / "report.txt"
    safe.write_text("send-me", encoding="utf-8")

    monkeypatch.chdir(untrusted)

    bridge = AgentBridge(_DummyAgent(workspace_dir=str(trusted)))
    bridge._run_started_at = 0.0

    artifacts = bridge._harvest_artifacts()
    names = [a.filename for a in artifacts]

    assert "report.txt" in names
    assert "secret.txt" not in names
