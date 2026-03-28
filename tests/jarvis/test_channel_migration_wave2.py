"""Tests for channel migration wave 2: QQ, WeChat, Feishu, iMessage.

Verifies that all four channels adopt the shared session/media/presenter
abstractions without keeping parallel copies of the same boilerplate.

Acceptance criteria addressed:
- AC-001.1: QQ, WeChat, Feishu, and iMessage adopt the shared abstractions.
- AC-001.2: Platform-specific transport logic remains local to each channel.
- AC-001.5: No parallel copies of command/session boilerplate.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from omicverse.jarvis.channels.channel_core import (
    RunningTask,
    coalesce_pending_requests,
    command_parts,
    default_summary,
    format_analysis_error,
    format_status_plain,
    gather_status,
    text_chunks,
)


# ── Shared helpers unit tests ───────────────────────────────────────────────


class TestCoalescePendingRequests:
    """Verify the shared coalesce function that was extracted from channel duplicates."""

    def test_empty_list(self) -> None:
        text, content = coalesce_pending_requests([])
        assert text == ""
        assert content == []

    def test_single_item(self) -> None:
        items = [{"text": "hello", "request_content": [{"type": "image", "url": "x"}]}]
        text, content = coalesce_pending_requests(items)
        assert text == "hello"
        assert len(content) == 1
        assert content[0]["url"] == "x"

    def test_multiple_items_joined(self) -> None:
        items = [
            {"text": "first request"},
            {"text": "second request"},
        ]
        text, content = coalesce_pending_requests(items)
        assert "first request" in text
        assert "second request" in text
        assert "\n\n" in text
        assert content == []

    def test_blank_text_skipped(self) -> None:
        items = [
            {"text": "   "},
            {"text": "real text"},
        ]
        text, content = coalesce_pending_requests(items)
        assert text == "real text"

    def test_request_content_merged(self) -> None:
        items = [
            {"text": "a", "request_content": [{"type": "img"}]},
            {"text": "b", "request_content": [{"type": "file"}, {"type": "img2"}]},
        ]
        text, content = coalesce_pending_requests(items)
        assert len(content) == 3

    def test_missing_keys(self) -> None:
        items = [{"other": "data"}, {}]
        text, content = coalesce_pending_requests(items)
        assert text == ""
        assert content == []


class TestCommandParts:
    """Verify the shared command-parsing function."""

    def test_simple_command(self) -> None:
        cmd, tail = command_parts("/help")
        assert cmd == "/help"
        assert tail == ""

    def test_command_with_tail(self) -> None:
        cmd, tail = command_parts("/model gpt-4o")
        assert cmd == "/model"
        assert tail == "gpt-4o"

    def test_command_case_insensitive(self) -> None:
        cmd, tail = command_parts("/RESET")
        assert cmd == "/reset"

    def test_empty_string(self) -> None:
        cmd, tail = command_parts("")
        assert cmd == ""
        assert tail == ""

    def test_command_with_multiword_tail(self) -> None:
        cmd, tail = command_parts("/shell ls -la /tmp")
        assert cmd == "/shell"
        assert tail == "ls -la /tmp"


# ── QQ channel uses shared abstractions ─────────────────────────────────────


class TestQQUsesSharedAbstractions:
    """Verify QQ imports and delegates to shared channel_core functions."""

    def test_qq_imports_running_task(self) -> None:
        from omicverse.jarvis.channels import qq as mod
        # QQRuntime should store RunningTask instances
        assert hasattr(mod, "RunningTask")
        assert mod.RunningTask is RunningTask

    def test_qq_imports_coalesce(self) -> None:
        from omicverse.jarvis.channels import qq as mod
        assert hasattr(mod, "coalesce_pending_requests")
        assert mod.coalesce_pending_requests is coalesce_pending_requests

    def test_qq_imports_format_analysis_error(self) -> None:
        from omicverse.jarvis.channels import qq as mod
        assert hasattr(mod, "format_analysis_error")
        assert mod.format_analysis_error is format_analysis_error

    def test_qq_imports_shared_text_utils(self) -> None:
        from omicverse.jarvis.channels import qq as mod
        assert mod.text_chunks is text_chunks

    def test_qq_runtime_coalesce_delegates(self) -> None:
        """QQRuntime._coalesce_pending_requests delegates to shared function."""
        from omicverse.jarvis.channels.qq import QQRuntime
        items = [{"text": "a", "request_content": [{"t": 1}]}, {"text": "b"}]
        text, content = QQRuntime._coalesce_pending_requests(items)
        expected_text, expected_content = coalesce_pending_requests(items)
        assert text == expected_text
        assert content == expected_content


# ── WeChat channel uses shared abstractions ─────────────────────────────────


class TestWeChatUsesSharedAbstractions:
    """Verify WeChat imports and delegates to shared channel_core functions."""

    def test_wechat_no_duplicate_running_task(self) -> None:
        """WeChat must NOT define its own RunningTask class."""
        import inspect
        from omicverse.jarvis.channels import wechat as mod
        source = inspect.getsource(mod)
        # The module should import RunningTask, not define a class
        lines = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith("class RunningTask")
        ]
        assert len(lines) == 0, "WeChat still defines its own RunningTask class"

    def test_wechat_imports_running_task_from_core(self) -> None:
        from omicverse.jarvis.channels import wechat as mod
        assert hasattr(mod, "RunningTask")
        assert mod.RunningTask is RunningTask

    def test_wechat_imports_coalesce(self) -> None:
        from omicverse.jarvis.channels import wechat as mod
        assert hasattr(mod, "coalesce_pending_requests")
        assert mod.coalesce_pending_requests is coalesce_pending_requests

    def test_wechat_imports_command_parts(self) -> None:
        from omicverse.jarvis.channels import wechat as mod
        assert hasattr(mod, "command_parts")
        assert mod.command_parts is command_parts

    def test_wechat_imports_format_analysis_error(self) -> None:
        from omicverse.jarvis.channels import wechat as mod
        assert hasattr(mod, "format_analysis_error")
        assert mod.format_analysis_error is format_analysis_error

    def test_wechat_coalesce_delegates(self) -> None:
        from omicverse.jarvis.channels.wechat import WeChatJarvisBot
        items = [{"text": "x", "request_content": []}, {"text": "y"}]
        text, content = WeChatJarvisBot._coalesce_pending_requests(items)
        expected_text, expected_content = coalesce_pending_requests(items)
        assert text == expected_text
        assert content == expected_content

    def test_wechat_command_parts_delegates(self) -> None:
        from omicverse.jarvis.channels.wechat import WeChatJarvisBot
        cmd, tail = WeChatJarvisBot._command_parts("/model gpt-4o")
        expected_cmd, expected_tail = command_parts("/model gpt-4o")
        assert cmd == expected_cmd
        assert tail == expected_tail


# ── Feishu channel uses shared abstractions ─────────────────────────────────


class TestFeishuUsesSharedAbstractions:
    """Verify Feishu imports shared functions from channel_core."""

    def test_feishu_imports_format_analysis_error(self) -> None:
        from omicverse.jarvis.channels import feishu as mod
        assert hasattr(mod, "format_analysis_error")
        assert mod.format_analysis_error is format_analysis_error

    def test_feishu_imports_default_summary(self) -> None:
        from omicverse.jarvis.channels import feishu as mod
        assert hasattr(mod, "default_summary")
        assert mod.default_summary is default_summary

    def test_feishu_imports_format_status_plain(self) -> None:
        from omicverse.jarvis.channels import feishu as mod
        assert hasattr(mod, "format_status_plain")
        assert mod.format_status_plain is format_status_plain

    def test_feishu_imports_running_task(self) -> None:
        from omicverse.jarvis.channels import feishu as mod
        assert hasattr(mod, "RunningTask")
        assert mod.RunningTask is RunningTask

    def test_feishu_imports_gather_status(self) -> None:
        from omicverse.jarvis.channels import feishu as mod
        assert hasattr(mod, "gather_status")
        assert mod.gather_status is gather_status

    def test_feishu_imports_text_chunks(self) -> None:
        from omicverse.jarvis.channels import feishu as mod
        assert mod.text_chunks is text_chunks


# ── iMessage channel uses shared abstractions ───────────────────────────────


class TestIMessageUsesSharedAbstractions:
    """Verify iMessage imports and uses shared channel_core functions."""

    def test_imessage_imports_running_task(self) -> None:
        from omicverse.jarvis.channels import imessage as mod
        assert hasattr(mod, "RunningTask")
        assert mod.RunningTask is RunningTask

    def test_imessage_imports_gather_status(self) -> None:
        from omicverse.jarvis.channels import imessage as mod
        assert hasattr(mod, "gather_status")
        assert mod.gather_status is gather_status

    def test_imessage_imports_format_status_plain(self) -> None:
        from omicverse.jarvis.channels import imessage as mod
        assert hasattr(mod, "format_status_plain")
        assert mod.format_status_plain is format_status_plain

    def test_imessage_imports_command_parts(self) -> None:
        from omicverse.jarvis.channels import imessage as mod
        assert hasattr(mod, "command_parts")
        assert mod.command_parts is command_parts

    def test_imessage_uses_running_task_in_tasks_dict(self) -> None:
        """IMessageJarvisBot._tasks should be typed as Dict[str, RunningTask]."""
        import typing
        from omicverse.jarvis.channels.imessage import IMessageJarvisBot
        hints = typing.get_type_hints(IMessageJarvisBot.__init__)
        # Even if type hints don't propagate at runtime, we can verify via
        # the annotation on the class attribute or via instance construction.
        # Alternative: check that the __init__ code creates the right type.
        import inspect
        source = inspect.getsource(IMessageJarvisBot.__init__)
        assert "RunningTask" in source

    def test_imessage_has_status_command(self) -> None:
        """IMessageJarvisBot._handle_command should handle /status."""
        import inspect
        from omicverse.jarvis.channels.imessage import IMessageJarvisBot
        source = inspect.getsource(IMessageJarvisBot._handle_command)
        assert '"/status"' in source or "gather_status" in source


# ── Cross-channel consistency ───────────────────────────────────────────────


class TestCrossChannelConsistency:
    """Verify that no channel module defines its own RunningTask or
    coalesce_pending_requests from scratch (only delegations allowed)."""

    @pytest.mark.parametrize("channel_name", ["qq", "wechat", "feishu", "imessage", "discord"])
    def test_all_channels_importable(self, channel_name: str) -> None:
        """Every channel module can be imported without error."""
        __import__(f"omicverse.jarvis.channels.{channel_name}")

    def test_no_duplicate_running_task_definitions(self) -> None:
        """Only channel_core should define RunningTask."""
        import inspect
        modules_to_check = []
        for name in ("qq", "wechat", "feishu", "imessage", "discord"):
            mod = __import__(f"omicverse.jarvis.channels.{name}", fromlist=[name])
            modules_to_check.append((name, mod))

        for name, mod in modules_to_check:
            source = inspect.getsource(mod)
            class_defs = [
                line.strip() for line in source.splitlines()
                if line.strip().startswith("class RunningTask")
            ]
            assert len(class_defs) == 0, (
                f"Channel {name} defines its own RunningTask class. "
                f"It should import from channel_core instead."
            )

    def test_format_analysis_error_used_by_all_wave2_channels(self) -> None:
        """QQ, WeChat, Feishu, iMessage should all import format_analysis_error."""
        for name in ("qq", "wechat", "feishu", "imessage"):
            mod = __import__(f"omicverse.jarvis.channels.{name}", fromlist=[name])
            assert hasattr(mod, "format_analysis_error"), (
                f"Channel {name} does not import format_analysis_error"
            )
            assert mod.format_analysis_error is format_analysis_error, (
                f"Channel {name} has a different format_analysis_error"
            )

    def test_running_task_used_by_all_wave2_channels(self) -> None:
        """All wave 2 channels should use RunningTask from channel_core."""
        for name in ("qq", "wechat", "feishu", "imessage"):
            mod = __import__(f"omicverse.jarvis.channels.{name}", fromlist=[name])
            assert hasattr(mod, "RunningTask"), (
                f"Channel {name} does not import RunningTask"
            )
            assert mod.RunningTask is RunningTask, (
                f"Channel {name} has a different RunningTask"
            )
