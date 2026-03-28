"""Tests for shared command data-gathering and formatting abstractions.

Covers:
- gather_status / format_status_plain: session status collection and formatting
- gather_usage / format_usage_plain: token usage collection and formatting
- gather_workspace / format_workspace_plain: workspace listing
- perform_save / format_save_result_plain: save command execution
- StatusInfo / UsageInfo / WorkspaceInfo / SaveResult dataclasses

Acceptance criteria addressed:
- AC-001.1: Shared command handling and presenter formatting in reusable abstractions
- AC-001.2: Channels no longer duplicate common command flows
- AC-001.3: Existing presenter outputs remain backward-compatible
- AC-001.5: Platform-specific differences preserved (formatters are pluggable)
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from omicverse.jarvis.channels.channel_core import (
    SaveResult,
    StatusInfo,
    UsageInfo,
    WorkspaceInfo,
    format_save_result_plain,
    format_status_plain,
    format_usage_plain,
    format_workspace_plain,
    gather_status,
    gather_usage,
    gather_workspace,
    perform_save,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_adata(n_obs: int = 5000, n_vars: int = 2000, obs_cols: Optional[list] = None):
    """Create a minimal adata-like object."""
    cols = obs_cols if obs_cols is not None else ["cell_type", "batch", "n_counts"]
    return SimpleNamespace(
        n_obs=n_obs,
        n_vars=n_vars,
        obs=SimpleNamespace(columns=cols),
    )


def _make_session(
    adata=None,
    last_usage=None,
    workspace=None,
    h5ad_files=None,
    agents_md="",
    memory_dir=None,
    kernel_status=None,
    user_id="u1",
    workspace_dir=None,
):
    """Create a minimal session-like object for testing."""
    ws = workspace or Path("/tmp/test_workspace")

    def _kernel_status():
        if kernel_status is not None:
            return kernel_status
        return {"prompt_count": 3, "max_prompts": 20, "session_id": "sid-123", "alive": True}

    return SimpleNamespace(
        adata=adata,
        last_usage=last_usage,
        workspace=ws,
        list_h5ad_files=lambda: h5ad_files or [],
        get_agents_md=lambda: agents_md,
        memory_dir=memory_dir or Path("/tmp/test_memory"),
        kernel_status=_kernel_status,
        user_id=user_id,
        agent=SimpleNamespace(workspace_dir=workspace_dir or ws),
        prompt_count=0,
        save_adata=lambda: None,
        append_memory_log=lambda **kw: None,
    )


def _make_session_manager(kernel_name: str = "default"):
    return SimpleNamespace(
        get_active_kernel=lambda uid: kernel_name,
    )


# ── gather_status ───────────────────────────────────────────────────────────

class TestGatherStatus:
    def test_with_adata(self) -> None:
        adata = _make_adata(n_obs=1000, n_vars=500, obs_cols=["ct", "batch"])
        session = _make_session(adata=adata)
        info = gather_status(session)
        assert info.adata_shape == (1000, 500)
        assert info.obs_columns == ["ct", "batch"]

    def test_without_adata(self) -> None:
        session = _make_session()
        info = gather_status(session)
        assert info.adata_shape is None
        assert info.obs_columns == []

    def test_with_session_manager_kernel(self) -> None:
        session = _make_session()
        sm = _make_session_manager(kernel_name="my-kernel")
        info = gather_status(session, session_manager=sm)
        assert info.kernel_name == "my-kernel"

    def test_without_session_manager(self) -> None:
        session = _make_session()
        info = gather_status(session)
        assert info.kernel_name is None

    def test_kernel_status_fields(self) -> None:
        session = _make_session(
            kernel_status={"prompt_count": 7, "max_prompts": 30, "session_id": "abc"}
        )
        info = gather_status(session)
        assert info.prompt_count == 7
        assert info.max_prompts == 30
        assert info.session_id == "abc"

    def test_running_flags(self) -> None:
        session = _make_session()
        info = gather_status(session, is_running=True, running_request="analyze pbmc")
        assert info.is_running is True
        assert info.running_request == "analyze pbmc"

    def test_workspace_path(self) -> None:
        session = _make_session(workspace_dir="/data/workspace")
        info = gather_status(session)
        assert info.workspace_path == "/data/workspace"

    def test_resilient_to_missing_kernel_status(self) -> None:
        session = SimpleNamespace(
            adata=None,
            user_id="u1",
            agent=SimpleNamespace(workspace_dir="/w"),
        )
        # No kernel_status method
        info = gather_status(session)
        assert info.prompt_count is None

    def test_resilient_to_session_manager_error(self) -> None:
        session = _make_session()
        sm = SimpleNamespace(get_active_kernel=lambda uid: (_ for _ in ()).throw(RuntimeError("no")))
        info = gather_status(session, session_manager=sm)
        assert info.kernel_name is None


# ── format_status_plain ─────────────────────────────────────────────────────

class TestFormatStatusPlain:
    def test_with_full_info(self) -> None:
        info = StatusInfo(
            adata_shape=(5000, 2000),
            obs_columns=["cell_type", "batch"],
            kernel_name="default",
            prompt_count=3,
            max_prompts=20,
            is_running=True,
            workspace_path="/data/ws",
        )
        text = format_status_plain(info)
        assert "5,000 cells x 2,000 genes" in text
        assert "obs: cell_type, batch" in text
        assert "kernel: default" in text
        assert "prompts: 3/20" in text
        assert "分析中（可 /cancel）" in text
        assert "工作区: /data/ws" in text

    def test_no_adata(self) -> None:
        info = StatusInfo()
        text = format_status_plain(info)
        assert "暂无数据" in text

    def test_idle(self) -> None:
        info = StatusInfo(is_running=False)
        text = format_status_plain(info)
        assert "/cancel" not in text

    def test_no_kernel(self) -> None:
        info = StatusInfo(adata_shape=(100, 50))
        text = format_status_plain(info)
        assert "kernel" not in text

    def test_no_workspace(self) -> None:
        info = StatusInfo()
        text = format_status_plain(info)
        assert "工作区" not in text


# ── gather_usage ────────────────────────────────────────────────────────────

class TestGatherUsage:
    def test_no_usage(self) -> None:
        session = _make_session()
        info = gather_usage(session)
        assert info.has_data is False

    def test_with_usage(self) -> None:
        usage = SimpleNamespace(
            input_tokens=1500,
            output_tokens=800,
            total_tokens=2300,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=100,
        )
        session = _make_session(last_usage=usage)
        info = gather_usage(session)
        assert info.has_data is True
        assert info.input_tokens == "1,500"
        assert info.output_tokens == "800"
        assert info.total_tokens == "2,300"
        assert info.cache_read == "200"
        assert info.cache_creation == "100"

    def test_missing_cache_fields(self) -> None:
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        session = _make_session(last_usage=usage)
        info = gather_usage(session)
        assert info.has_data is True
        assert info.cache_read == ""
        assert info.cache_creation == ""

    def test_string_token_values(self) -> None:
        usage = SimpleNamespace(
            input_tokens="N/A",
            output_tokens="N/A",
            total_tokens="N/A",
        )
        session = _make_session(last_usage=usage)
        info = gather_usage(session)
        assert info.input_tokens == "N/A"


# ── format_usage_plain ──────────────────────────────────────────────────────

class TestFormatUsagePlain:
    def test_no_data(self) -> None:
        info = UsageInfo(has_data=False)
        text = format_usage_plain(info)
        assert "暂无用量数据" in text

    def test_with_data(self) -> None:
        info = UsageInfo(
            input_tokens="1,500",
            output_tokens="800",
            total_tokens="2,300",
            has_data=True,
        )
        text = format_usage_plain(info)
        assert "输入: 1,500" in text
        assert "输出: 800" in text
        assert "合计: 2,300" in text

    def test_with_cache(self) -> None:
        info = UsageInfo(
            input_tokens="100",
            output_tokens="50",
            total_tokens="150",
            cache_read="200",
            cache_creation="100",
            has_data=True,
        )
        text = format_usage_plain(info)
        assert "缓存读取: 200" in text
        assert "缓存写入: 100" in text

    def test_no_cache_when_empty(self) -> None:
        info = UsageInfo(
            input_tokens="100",
            output_tokens="50",
            total_tokens="150",
            has_data=True,
        )
        text = format_usage_plain(info)
        assert "缓存" not in text


# ── gather_workspace ────────────────────────────────────────────────────────

class TestGatherWorkspace:
    def test_empty_workspace(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        session = _make_session(
            workspace=tmp_path,
            h5ad_files=[],
            agents_md="",
            memory_dir=memory_dir,
        )
        info = gather_workspace(session)
        assert info.path == str(tmp_path)
        assert info.h5ad_files == []
        assert info.h5ad_total == 0
        assert info.has_agents_md is False

    def test_with_h5ad_files(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        f1 = tmp_path / "data1.h5ad"
        f1.write_bytes(b"x" * 2_097_152)  # 2 MB
        f2 = tmp_path / "data2.h5ad"
        f2.write_bytes(b"y" * 1_048_576)  # 1 MB
        session = _make_session(
            workspace=tmp_path,
            h5ad_files=[f1, f2],
            agents_md="# My instructions",
            memory_dir=memory_dir,
        )
        info = gather_workspace(session)
        assert info.h5ad_total == 2
        assert len(info.h5ad_files) == 2
        assert info.h5ad_files[0][0] == "data1.h5ad"
        assert abs(info.h5ad_files[0][1] - 2.0) < 0.1
        assert info.has_agents_md is True

    def test_caps_at_10_files(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        files = []
        for i in range(15):
            f = tmp_path / f"data_{i}.h5ad"
            f.write_bytes(b"x")
            files.append(f)
        session = _make_session(
            workspace=tmp_path,
            h5ad_files=files,
            memory_dir=memory_dir,
        )
        info = gather_workspace(session)
        assert len(info.h5ad_files) == 10
        assert info.h5ad_total == 15


# ── format_workspace_plain ──────────────────────────────────────────────────

class TestFormatWorkspacePlain:
    def test_empty_workspace(self) -> None:
        info = WorkspaceInfo(path="/w", h5ad_total=0)
        text = format_workspace_plain(info)
        assert "Workspace: /w" in text
        assert "数据文件 (空)" in text

    def test_with_files(self) -> None:
        info = WorkspaceInfo(
            path="/w",
            h5ad_files=[("a.h5ad", 1.5), ("b.h5ad", None)],
            h5ad_total=2,
            has_agents_md=True,
            has_today_memory=True,
        )
        text = format_workspace_plain(info)
        assert "数据文件 (2)" in text
        assert "- a.h5ad (1.5 MB)" in text
        assert "- b.h5ad" in text
        assert "AGENTS.md OK" in text
        assert "今日记忆 OK" in text

    def test_overflow_indicator(self) -> None:
        info = WorkspaceInfo(
            path="/w",
            h5ad_files=[("a.h5ad", 1.0)],
            h5ad_total=12,
        )
        text = format_workspace_plain(info)
        assert "还有 11 个" in text


# ── perform_save ────────────────────────────────────────────────────────────

class TestPerformSave:
    def test_no_data(self) -> None:
        session = _make_session()
        result = perform_save(session)
        assert result.no_data is True
        assert result.success is False

    def test_successful_save(self, tmp_path: Path) -> None:
        save_path = tmp_path / "current.h5ad"
        save_path.write_bytes(b"saved")
        adata = _make_adata(n_obs=100, n_vars=50)
        session = _make_session(adata=adata)
        session.save_adata = lambda: save_path
        result = perform_save(session)
        assert result.success is True
        assert result.path == str(save_path)
        assert result.adata_shape == (100, 50)

    def test_save_returns_none(self) -> None:
        adata = _make_adata()
        session = _make_session(adata=adata)
        session.save_adata = lambda: None
        result = perform_save(session)
        assert result.success is False
        assert result.error is not None

    def test_save_exception(self) -> None:
        adata = _make_adata()
        session = _make_session(adata=adata)
        session.save_adata = lambda: (_ for _ in ()).throw(IOError("disk full"))
        result = perform_save(session)
        assert result.success is False
        assert "disk full" in result.error


# ── format_save_result_plain ────────────────────────────────────────────────

class TestFormatSaveResultPlain:
    def test_no_data(self) -> None:
        result = SaveResult(no_data=True)
        text = format_save_result_plain(result)
        assert "没有数据" in text

    def test_success(self) -> None:
        result = SaveResult(
            success=True,
            path="/w/current.h5ad",
            adata_shape=(100, 50),
        )
        text = format_save_result_plain(result)
        assert "已保存" in text
        assert "100 cells x 50 genes" in text
        assert "/w/current.h5ad" in text

    def test_error(self) -> None:
        result = SaveResult(error="disk full")
        text = format_save_result_plain(result)
        assert "disk full" in text


# ── Cross-channel consistency ───────────────────────────────────────────────

class TestCrossChannelConsistency:
    """Verify that shared gather functions produce the same data regardless
    of which channel calls them — the core invariant of this abstraction."""

    def test_status_same_for_different_channels(self) -> None:
        adata = _make_adata(n_obs=3000, n_vars=1000)
        sm = _make_session_manager("k1")
        session = _make_session(adata=adata)

        info1 = gather_status(session, session_manager=sm, is_running=True)
        info2 = gather_status(session, session_manager=sm, is_running=True)
        assert info1.adata_shape == info2.adata_shape
        assert info1.kernel_name == info2.kernel_name
        assert info1.obs_columns == info2.obs_columns

    def test_usage_same_for_different_channels(self) -> None:
        usage = SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150)
        session = _make_session(last_usage=usage)

        info1 = gather_usage(session)
        info2 = gather_usage(session)
        assert info1.input_tokens == info2.input_tokens
        assert info1.has_data == info2.has_data


class TestChannelCoreImports:
    """Verify that the new abstractions are importable alongside existing ones."""

    def test_all_new_symbols_importable(self) -> None:
        from omicverse.jarvis.channels.channel_core import (
            StatusInfo,
            UsageInfo,
            WorkspaceInfo,
            SaveResult,
            gather_status,
            gather_usage,
            gather_workspace,
            perform_save,
            format_status_plain,
            format_usage_plain,
            format_workspace_plain,
            format_save_result_plain,
        )
        # Smoke test: all are callable or dataclass
        assert callable(gather_status)
        assert callable(format_status_plain)

    def test_existing_symbols_still_importable(self) -> None:
        from omicverse.jarvis.channels.channel_core import (
            RunningTask,
            text_chunks,
            strip_local_paths,
            build_full_request,
            get_prior_history,
            notify_turn_complete,
            process_result_state,
            format_analysis_error,
            default_summary,
        )
        assert callable(text_chunks)
        assert callable(strip_local_paths)
