from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from omicverse.jarvis.agent_bridge import AgentBridge


def test_agent_bridge_harvests_png_from_workspace_outputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outputs = workspace / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    expected = b"png-from-outputs"
    (outputs / "final_plot.png").write_bytes(expected)

    agent = SimpleNamespace(
        _notebook_executor=None,
        _filesystem_context=SimpleNamespace(_workspace_dir=workspace),
    )
    bridge = AgentBridge(agent)
    bridge._run_started_at = 0.0

    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        figures = bridge._harvest_file_figures(set())
    finally:
        os.chdir(old_cwd)

    assert figures == [expected]
