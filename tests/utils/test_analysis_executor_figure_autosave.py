from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor


def test_analysis_executor_injects_figure_autosave_into_context_figures(tmp_path: Path) -> None:
    figure_root = tmp_path / "context_session"
    ctx = SimpleNamespace(
        _filesystem_context=SimpleNamespace(workspace_dir=figure_root),
    )
    executor = AnalysisExecutor(ctx)

    wrapped = executor._inject_figure_autosave("plt.plot([1, 2], [3, 4])")

    assert str(figure_root / "figures") in wrapped
    assert "auto_figure_" in wrapped
    assert "_ov_fig_plt.get_fignums()" in wrapped
