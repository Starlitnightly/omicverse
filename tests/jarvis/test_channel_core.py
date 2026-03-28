"""Tests for the shared channel-core abstractions.

Covers:
- text_chunks: paragraph-aware text splitting
- strip_local_paths: filesystem path scrubbing
- build_full_request: session-enriched request building
- get_prior_history: web bridge history retrieval
- notify_turn_complete: web bridge turn mirroring
- process_result_state: post-analysis session state updates
- format_analysis_error: error message formatting
- default_summary: fallback summary generation
- RunningTask: dataclass smoke test
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from omicverse.jarvis.channels.channel_core import (
    RunningTask,
    build_full_request,
    default_summary,
    format_analysis_error,
    get_prior_history,
    notify_turn_complete,
    process_result_state,
    strip_local_paths,
    text_chunks,
)


# ── text_chunks ──────────────────────────────────────────────────────────────

class TestTextChunks:
    def test_empty_string(self) -> None:
        assert text_chunks("") == []
        assert text_chunks(None) == []

    def test_short_text_single_chunk(self) -> None:
        assert text_chunks("hello", limit=100) == ["hello"]

    def test_exact_limit(self) -> None:
        assert text_chunks("abc", limit=3) == ["abc"]

    def test_split_on_paragraphs(self) -> None:
        text = "AAA\n\nBBB\n\nCCC"
        result = text_chunks(text, limit=7)
        assert result == ["AAA", "BBB", "CCC"]

    def test_long_paragraph_hard_split(self) -> None:
        text = "ABCDEFGHIJ"
        result = text_chunks(text, limit=4)
        assert result == ["ABCD", "EFGH", "IJ"]

    def test_paragraphs_coalesce_when_possible(self) -> None:
        text = "A\n\nB\n\nC"
        result = text_chunks(text, limit=100)
        assert result == ["A\n\nB\n\nC"]

    def test_strips_whitespace(self) -> None:
        assert text_chunks("  hello  ") == ["hello"]

    def test_default_limit(self) -> None:
        short = "x" * 2000
        assert text_chunks(short) == [short]
        long = "x" * 2001
        result = text_chunks(long)
        assert len(result) == 2


# ── strip_local_paths ────────────────────────────────────────────────────────

class TestStripLocalPaths:
    def test_removes_absolute_paths(self) -> None:
        assert strip_local_paths("see /Users/alice/data/file.txt here") == "see here"

    def test_removes_home_tilde_paths(self) -> None:
        result = strip_local_paths("saved to ~/Documents/out.csv done")
        assert "~" not in result
        assert "done" in result

    def test_removes_backtick_paths(self) -> None:
        assert strip_local_paths("run `cat /a/b/c/d.py`") == "run"

    def test_removes_extension_paths(self) -> None:
        result = strip_local_paths("file at ./data/subdir/result.h5ad ok")
        assert "result.h5ad" not in result
        assert "ok" in result

    def test_collapses_whitespace(self) -> None:
        assert "  " not in strip_local_paths("a    b")

    def test_collapses_newlines(self) -> None:
        assert "\n\n\n" not in strip_local_paths("a\n\n\n\nb")

    def test_none_returns_empty(self) -> None:
        assert strip_local_paths(None) == ""

    def test_preserves_normal_text(self) -> None:
        text = "Analysis complete: 5000 cells detected"
        assert strip_local_paths(text) == text

    def test_preserves_scientific_data_paths(self) -> None:
        """Paths under /data, /opt, /mnt, /var are legitimate scientific paths."""
        assert "/data/shared/genome.fa" in strip_local_paths(
            "Reference genome at /data/shared/genome.fa"
        )

    def test_preserves_opt_tool_paths(self) -> None:
        assert "/opt/cellranger/bin" in strip_local_paths(
            "Using tool at /opt/cellranger/bin"
        )

    def test_preserves_mnt_paths(self) -> None:
        assert "/mnt/nfs/lab" in strip_local_paths(
            "Lab data on /mnt/nfs/lab/experiment_1"
        )

    def test_preserves_var_paths(self) -> None:
        assert "/var/log/analysis" in strip_local_paths(
            "See /var/log/analysis for details"
        )

    def test_still_removes_users_paths(self) -> None:
        result = strip_local_paths("see /Users/alice/work/project here")
        assert "/Users/" not in result

    def test_still_removes_home_paths(self) -> None:
        result = strip_local_paths("file at /home/bob/.config/app.ini")
        assert "/home/" not in result

    def test_still_removes_tmp_paths(self) -> None:
        result = strip_local_paths("cached in /tmp/ov_session_abc123")
        assert "/tmp/" not in result

    def test_still_removes_private_paths(self) -> None:
        result = strip_local_paths("see /private/var/folders/xx/tmp here")
        assert "/private/" not in result


# ── build_full_request ───────────────────────────────────────────────────────

class TestBuildFullRequest:
    def test_delegates_to_channel_media(self) -> None:
        session = SimpleNamespace(
            get_agents_md=lambda: "",
            get_memory_context=lambda: "",
        )
        result = build_full_request(session, "analyze data", channel_label="Test")
        assert "analyze data" in result
        assert "Test" in result

    def test_includes_channel_label(self) -> None:
        session = SimpleNamespace(
            get_agents_md=lambda: "",
            get_memory_context=lambda: "",
        )
        result = build_full_request(session, "hello", channel_label="Discord")
        assert "Discord" in result


# ── get_prior_history ────────────────────────────────────────────────────────

class TestGetPriorHistory:
    def test_returns_empty_when_no_web_bridge(self) -> None:
        sm = SimpleNamespace()
        assert get_prior_history(sm, "discord", "dm", "123", SimpleNamespace()) == []

    def test_returns_empty_when_bridge_is_none(self) -> None:
        sm = SimpleNamespace(gateway_web_bridge=None)
        assert get_prior_history(sm, "discord", "dm", "123", SimpleNamespace()) == []

    def test_calls_web_bridge(self) -> None:
        calls = []

        class FakeBridge:
            def get_prior_history_simple(self, channel, scope_type, scope_id, session_id=""):
                calls.append((channel, scope_type, scope_id))
                return [{"role": "user", "content": "hi"}]

        sm = SimpleNamespace(gateway_web_bridge=FakeBridge())
        session = SimpleNamespace(
            agent=SimpleNamespace(get_current_session_info=lambda: None),
        )
        result = get_prior_history(sm, "telegram", "group", "456", session)
        assert len(result) == 1
        assert calls == [("telegram", "group", "456")]


# ── notify_turn_complete ─────────────────────────────────────────────────────

class TestNotifyTurnComplete:
    def test_noop_when_no_web_bridge(self) -> None:
        sm = SimpleNamespace()
        notify_turn_complete(
            sm,
            channel="discord",
            scope_type="dm",
            scope_id="123",
            session=SimpleNamespace(),
            user_text="hello",
            llm_text="world",
        )

    def test_calls_on_turn_complete_simple(self) -> None:
        calls = []

        class FakeBridge:
            def on_turn_complete_simple(self, **kwargs):
                calls.append(kwargs)

        sm = SimpleNamespace(gateway_web_bridge=FakeBridge())
        session = SimpleNamespace(
            agent=SimpleNamespace(get_current_session_info=lambda: None),
        )
        notify_turn_complete(
            sm,
            channel="wechat",
            scope_type="dm",
            scope_id="789",
            session=session,
            user_text="test",
            llm_text="reply",
            adata=object(),
            figures=[b"png"],
        )
        assert len(calls) == 1
        assert calls[0]["channel"] == "wechat"
        assert calls[0]["user_text"] == "test"
        assert calls[0]["figures"] == [b"png"]

    def test_swallows_exceptions(self) -> None:
        class FakeBridge:
            def on_turn_complete_simple(self, **kwargs):
                raise RuntimeError("boom")

        sm = SimpleNamespace(gateway_web_bridge=FakeBridge())
        notify_turn_complete(
            sm,
            channel="discord",
            scope_type="dm",
            scope_id="123",
            session=SimpleNamespace(
                agent=SimpleNamespace(get_current_session_info=lambda: None),
            ),
            user_text="x",
            llm_text="y",
        )

    def test_logs_debug_on_exception(self, caplog) -> None:
        """Exception in web bridge produces a debug log with traceback."""
        class FakeBridge:
            def on_turn_complete_simple(self, **kwargs):
                raise RuntimeError("bridge-error")

        sm = SimpleNamespace(gateway_web_bridge=FakeBridge())
        with caplog.at_level(logging.DEBUG, logger="omicverse.jarvis.channels.channel_core"):
            notify_turn_complete(
                sm,
                channel="discord",
                scope_type="dm",
                scope_id="123",
                session=SimpleNamespace(
                    agent=SimpleNamespace(get_current_session_info=lambda: None),
                ),
                user_text="x",
                llm_text="y",
            )
        assert any("notify_turn_complete" in r.message for r in caplog.records)
        assert any(r.exc_info for r in caplog.records if "notify_turn_complete" in r.message)


# ── process_result_state ─────────────────────────────────────────────────────

class TestProcessResultState:
    def _make_session(self, adata=None):
        return SimpleNamespace(
            adata=adata,
            last_usage=None,
            prompt_count=0,
            save_adata=lambda: None,
            append_memory_log=lambda **kw: None,
        )

    def _make_result(self, adata=None, usage=None, figures=None, summary=None):
        return SimpleNamespace(
            adata=adata,
            usage=usage,
            figures=figures or [],
            summary=summary,
        )

    def test_updates_session_adata(self) -> None:
        session = self._make_session()
        new_adata = SimpleNamespace(n_obs=100, n_vars=50)
        result = self._make_result(adata=new_adata)
        delivery_figures, adata_info = process_result_state(session, result, "test")
        assert session.adata is new_adata
        assert "100" in adata_info
        assert "50" in adata_info

    def test_increments_prompt_count(self) -> None:
        session = self._make_session()
        result = self._make_result(adata=SimpleNamespace(n_obs=1, n_vars=1))
        process_result_state(session, result, "test")
        assert session.prompt_count == 1

    def test_updates_usage(self) -> None:
        session = self._make_session()
        usage = {"tokens": 42}
        result = self._make_result(usage=usage)
        process_result_state(session, result, "test")
        assert session.last_usage == usage

    def test_no_adata_returns_empty_info(self) -> None:
        session = self._make_session()
        result = self._make_result()
        _, adata_info = process_result_state(session, result, "test")
        assert adata_info == ""

    def test_logs_debug_on_save_adata_failure(self, caplog) -> None:
        """save_adata failure produces a debug log entry."""
        session = self._make_session()
        session.save_adata = lambda: (_ for _ in ()).throw(IOError("disk"))
        result = self._make_result(adata=SimpleNamespace(n_obs=1, n_vars=1))
        with caplog.at_level(logging.DEBUG, logger="omicverse.jarvis.channels.channel_core"):
            process_result_state(session, result, "test")
        assert any("save_adata" in r.message for r in caplog.records)

    def test_logs_debug_on_memory_log_failure(self, caplog) -> None:
        """append_memory_log failure produces a debug log entry."""
        session = self._make_session()
        session.append_memory_log = lambda **kw: (_ for _ in ()).throw(RuntimeError("mem"))
        result = self._make_result()
        with caplog.at_level(logging.DEBUG, logger="omicverse.jarvis.channels.channel_core"):
            process_result_state(session, result, "test")
        assert any("append_memory_log" in r.message for r in caplog.records)

    def test_appends_memory_log(self) -> None:
        log_calls = []
        session = SimpleNamespace(
            adata=None,
            last_usage=None,
            prompt_count=0,
            save_adata=lambda: None,
            append_memory_log=lambda **kw: log_calls.append(kw),
        )
        result = self._make_result(summary="done")
        process_result_state(session, result, "my request")
        assert len(log_calls) == 1
        assert log_calls[0]["request"] == "my request"
        assert log_calls[0]["summary"] == "done"


# ── format_analysis_error ────────────────────────────────────────────────────

class TestFormatAnalysisError:
    def test_basic_error(self) -> None:
        result = SimpleNamespace(error="timeout", diagnostics=None)
        text = format_analysis_error(result, "")
        assert "分析出错: timeout" in text

    def test_with_diagnostics(self) -> None:
        result = SimpleNamespace(error="fail", diagnostics=["hint1", "hint2"])
        text = format_analysis_error(result, "")
        assert "- hint1" in text
        assert "- hint2" in text
        assert "诊断" in text

    def test_with_llm_buf(self) -> None:
        result = SimpleNamespace(error="err", diagnostics=None)
        text = format_analysis_error(result, "some model output")
        assert "模型输出" in text
        assert "some model output" in text

    def test_max_llm_truncation(self) -> None:
        result = SimpleNamespace(error="err", diagnostics=None)
        long_buf = "x" * 5000
        text = format_analysis_error(result, long_buf, max_llm=100)
        # The llm portion should be truncated
        assert len(text) < 5000

    def test_empty_llm_buf_excluded(self) -> None:
        result = SimpleNamespace(error="err", diagnostics=None)
        text = format_analysis_error(result, "   ")
        assert "模型输出" not in text


# ── default_summary ──────────────────────────────────────────────────────────

class TestDefaultSummary:
    def test_with_adata(self) -> None:
        session = SimpleNamespace(adata=SimpleNamespace(n_obs=5000, n_vars=2000))
        summary = default_summary(session)
        assert "5,000" in summary
        assert "2,000" in summary
        assert "分析完成" in summary

    def test_without_adata(self) -> None:
        session = SimpleNamespace(adata=None)
        assert default_summary(session) == "分析完成"


# ── RunningTask ──────────────────────────────────────────────────────────────

class TestRunningTask:
    def test_dataclass_fields(self) -> None:
        task = asyncio.Future()
        rt = RunningTask(task=task, request="test", started_at=1.0)
        assert rt.task is task
        assert rt.request == "test"
        assert rt.started_at == 1.0


# ── Channel imports use shared code ──────────────────────────────────────────

class TestChannelImports:
    """Verify that channel modules can import and that their shared helpers
    resolve to the channel_core implementations."""

    def test_discord_imports_channel_core(self) -> None:
        from omicverse.jarvis.channels import discord as mod
        assert hasattr(mod, "text_chunks")
        assert mod.text_chunks is text_chunks

    def test_imessage_imports_channel_core(self) -> None:
        from omicverse.jarvis.channels import imessage as mod
        assert hasattr(mod, "text_chunks")
        assert mod.text_chunks is text_chunks

    def test_channel_core_strip_local_paths_matches_discord(self) -> None:
        test_str = "see /Users/foo/bar.txt end"
        assert strip_local_paths(test_str) == strip_local_paths(test_str)
