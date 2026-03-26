from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from omicverse.utils.ovagent.turn_controller import TurnController


def test_persist_tool_debug_output_writes_results_file(tmp_path: Path) -> None:
    ctx = SimpleNamespace(
        _filesystem_context=SimpleNamespace(workspace_dir=tmp_path),
    )
    controller = TurnController(
        ctx=ctx,
        prompt_builder=MagicMock(),
        tool_runtime=MagicMock(),
    )

    path = controller._persist_tool_debug_output(
        "execute_code",
        "stdout:\nfull debug output",
        turn_index=3,
        tool_index=1,
        description="Plot T-cell marker UMAP",
    )

    assert path is not None
    assert path.exists()
    assert path.parent == tmp_path / "results"
    text = path.read_text(encoding="utf-8")
    assert "tool=execute_code" in text
    assert "turn=3" in text
    assert "tool_index=1" in text
    assert "description=Plot T-cell marker UMAP" in text
    assert text.endswith("stdout:\nfull debug output")


def test_log_tool_debug_output_chunks_full_content(caplog) -> None:
    long_output = "B" * 4500

    with caplog.at_level(logging.INFO):
        TurnController._log_tool_debug_output(
            tool_name="execute_code",
            output=long_output,
            turn_index=2,
            tool_index=1,
            path=Path("/tmp/fake.log"),
            chunk_size=4000,
        )

    messages = [record.message for record in caplog.records]
    assert any("execute_code_result_saved" in message for message in messages)
    assert any("chunk=1/2" in message for message in messages)
    assert any("chunk=2/2" in message for message in messages)


def test_persist_execute_code_source_writes_python_file(tmp_path: Path) -> None:
    ctx = SimpleNamespace(
        _filesystem_context=SimpleNamespace(workspace_dir=tmp_path),
    )
    controller = TurnController(
        ctx=ctx,
        prompt_builder=MagicMock(),
        tool_runtime=MagicMock(),
    )

    path = controller._persist_execute_code_source(
        "import scanpy as sc\nsc.pl.umap(adata, color='CD3D')\n",
        turn_index=2,
        tool_index=1,
        description="Plot CD3D UMAP",
    )

    assert path is not None
    assert path.exists()
    assert path.suffix == ".py"
    assert path.parent == tmp_path / "results"
    assert "sc.pl.umap" in path.read_text(encoding="utf-8")


def test_log_execute_code_source_chunks_full_content(caplog) -> None:
    long_code = "print('x')\n" * 500

    with caplog.at_level(logging.INFO):
        TurnController._log_execute_code_source(
            code=long_code,
            turn_index=2,
            tool_index=1,
            path=Path("/tmp/fake.py"),
            chunk_size=4000,
        )

    messages = [record.message for record in caplog.records]
    assert any("execute_code_source_saved" in message for message in messages)
    assert any("chunk=1/" in message for message in messages)


def test_persist_execute_code_stdout_writes_results_file(tmp_path: Path) -> None:
    ctx = SimpleNamespace(
        _filesystem_context=SimpleNamespace(workspace_dir=tmp_path),
    )
    controller = TurnController(
        ctx=ctx,
        prompt_builder=MagicMock(),
        tool_runtime=MagicMock(),
    )

    path = controller._persist_execute_code_stdout(
        "CD3D UMAP completed\ncells=700\n",
        turn_index=2,
        tool_index=1,
        description="Plot CD3D UMAP",
    )

    assert path is not None
    assert path.exists()
    assert path.parent == tmp_path / "results"
    text = path.read_text(encoding="utf-8")
    assert "tool=execute_code_stdout" in text
    assert "turn=2" in text
    assert text.endswith("CD3D UMAP completed\ncells=700\n")


def test_log_execute_code_stdout_chunks_full_content(caplog) -> None:
    long_stdout = "stdout line\n" * 500

    with caplog.at_level(logging.INFO):
        TurnController._log_execute_code_stdout(
            stdout=long_stdout,
            turn_index=2,
            tool_index=1,
            path=Path("/tmp/stdout.log"),
            chunk_size=4000,
        )

    messages = [record.message for record in caplog.records]
    assert any("execute_code_stdout_saved" in message for message in messages)
    assert any("chunk=1/" in message for message in messages)
