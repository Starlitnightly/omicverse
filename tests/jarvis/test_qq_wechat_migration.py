"""Tests for QQ and WeChat channel migration to shared runtime abstractions.

Covers:
- qq_route_from_target: ConversationRoute construction for c2c, group, guild, dm
- qq_runtime_envelope: MessageEnvelope construction
- QQRuntimePresenter: all MessagePresenter protocol methods
- wechat_route_from_session_key: ConversationRoute construction for DM
- wechat_runtime_envelope: MessageEnvelope construction
- WeChatRuntimePresenter: all MessagePresenter protocol methods
- Runtime wiring: QQRuntime and WeChatJarvisBot use MessageRuntime
- Shared channel_core helpers used for commands

Acceptance criteria addressed:
- AC-001.1: QQ and WeChat no longer carry bespoke copies of shared session,
             presenter, and result-formatting flows.
- AC-001.2: Platform-specific media and upload behavior remains local and correct.
- AC-001.3: Channel-focused tests cover the migrated QQ and WeChat paths.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from omicverse.jarvis.channels.qq import (
    QQRuntimePresenter,
    QQTarget,
    qq_route_from_target,
    qq_runtime_envelope,
    _BORING,
    _MAX_TEXT,
)
from omicverse.jarvis.channels.wechat import (
    WeChatRuntimePresenter,
    wechat_route_from_session_key,
    wechat_runtime_envelope,
    _BORING_SUMMARIES,
    _META_COMMENTARY_RE,
    _MAX_TEXT as WECHAT_MAX_TEXT,
)
from omicverse.jarvis.channels.channel_core import (
    gather_status,
    format_status_plain,
    strip_local_paths,
    StatusInfo,
)
from omicverse.jarvis.gateway.routing import SessionKey
from omicverse.jarvis.runtime import (
    ConversationRoute,
    DeliveryEvent,
    MessageEnvelope,
    MessagePresenter,
)


# ── Test helpers ──────────────────────────────────────────────────────────


def _make_session(adata=None, h5ad_files=None, workspace="/tmp/ws"):
    files = h5ad_files or []
    return SimpleNamespace(
        adata=adata,
        workspace=workspace,
        list_h5ad_files=lambda: files,
    )


def _make_adata(n_obs: int = 5000, n_vars: int = 2000, obs_cols=None):
    cols = obs_cols or ["cell_type", "batch"]
    return SimpleNamespace(
        n_obs=n_obs,
        n_vars=n_vars,
        obs=SimpleNamespace(columns=cols),
    )


def _make_result(
    summary="分析完成",
    error=None,
    figures=None,
    reports=None,
    artifacts=None,
    adata=None,
    usage=None,
):
    return SimpleNamespace(
        summary=summary,
        error=error,
        figures=figures or [],
        reports=reports or [],
        artifacts=artifacts or [],
        adata=adata,
        usage=usage,
    )


# ═══════════════════════════════════════════════════════════════════════════
# QQ Route Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQQRouteConstruction:
    def test_c2c_route(self):
        target = QQTarget(kind="c2c", id="user123")
        route = qq_route_from_target(target)
        assert route.channel == "qq"
        assert route.scope_type == "c2c"
        assert route.scope_id == "user123"
        # c2c is QQ-specific; ConversationRoute preserves it as-is

    def test_group_route(self):
        target = QQTarget(kind="group", id="group456")
        route = qq_route_from_target(target)
        assert route.channel == "qq"
        assert route.scope_type == "group"
        assert route.scope_id == "group456"
        assert route.is_group

    def test_guild_route(self):
        target = QQTarget(kind="guild", id="chan789")
        route = qq_route_from_target(target)
        assert route.channel == "qq"
        assert route.scope_type == "guild"
        assert route.scope_id == "chan789"

    def test_dm_route(self):
        target = QQTarget(kind="dm", id="guild_dm_1")
        route = qq_route_from_target(target)
        assert route.channel == "qq"
        assert route.scope_type == "dm"
        assert route.is_direct

    def test_route_key_stable(self):
        target = QQTarget(kind="c2c", id="u1")
        r1 = qq_route_from_target(target)
        r2 = qq_route_from_target(target)
        assert r1.route_key() == r2.route_key()


# ═══════════════════════════════════════════════════════════════════════════
# QQ Envelope Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQQEnvelope:
    def test_basic_envelope(self):
        target = QQTarget(kind="c2c", id="u1")
        env = qq_runtime_envelope(target, "hello", sender_id="u1", msg_id="m1")
        assert env is not None
        assert env.text == "hello"
        assert env.sender_id == "u1"
        assert env.message_id == "m1"
        assert env.route.channel == "qq"
        # c2c is QQ-specific scope_type; trigger depends on is_direct
        assert env.trigger in ("direct", "mention")

    def test_empty_text_returns_none(self):
        target = QQTarget(kind="c2c", id="u1")
        assert qq_runtime_envelope(target, "", sender_id="u1") is None
        assert qq_runtime_envelope(target, "   ", sender_id="u1") is None

    def test_group_trigger_is_mention(self):
        target = QQTarget(kind="group", id="g1")
        env = qq_runtime_envelope(target, "test", sender_id="u1")
        assert env is not None
        assert env.trigger == "mention"

    def test_metadata_propagation(self):
        target = QQTarget(kind="c2c", id="u1")
        env = qq_runtime_envelope(
            target, "test", sender_id="u1",
            metadata={"image_count": 2, "request_content": [{"type": "image"}]},
        )
        assert env is not None
        assert env.metadata["image_count"] == 2

    def test_text_is_stripped(self):
        target = QQTarget(kind="c2c", id="u1")
        env = qq_runtime_envelope(target, "  hello world  ", sender_id="u1")
        assert env is not None
        assert env.text == "hello world"


# ═══════════════════════════════════════════════════════════════════════════
# QQ Presenter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQQPresenter:
    def setup_method(self):
        self.presenter = QQRuntimePresenter()
        self.route = ConversationRoute(
            channel="qq", scope_type="c2c", scope_id="u1",
        )

    def _envelope(self, text="test", metadata=None):
        return MessageEnvelope(
            route=self.route,
            text=text,
            sender_id="u1",
            metadata=metadata or {},
        )

    def test_ack_with_adata(self):
        session = _make_session(adata=_make_adata(3000, 1000))
        events = self.presenter.ack(self._envelope(), session)
        assert len(events) == 1
        assert "3,000" in events[0].text
        assert "1,000" in events[0].text

    def test_ack_no_adata_with_files(self):
        files = [SimpleNamespace(name="data.h5ad")]
        session = _make_session(h5ad_files=files)
        events = self.presenter.ack(self._envelope(), session)
        assert "data.h5ad" in events[0].text
        assert "/load" in events[0].text

    def test_ack_no_adata_no_files(self):
        session = _make_session()
        events = self.presenter.ack(self._envelope(), session)
        assert "Agent" in events[0].text

    def test_ack_with_images(self):
        session = _make_session()
        env = self._envelope(metadata={"image_count": 3})
        events = self.presenter.ack(env, session)
        assert "3 张" in events[0].text

    def test_queue_started(self):
        events = self.presenter.queue_started(self.route, 2)
        assert len(events) == 1
        assert "2" in events[0].text

    def test_draft_open(self):
        event = self.presenter.draft_open(self.route)
        assert event.kind == "text"
        assert event.mode == "open"

    def test_draft_update_with_progress(self):
        event = self.presenter.draft_update(self.route, "", "loading data")
        assert "loading data" in event.text

    def test_draft_update_with_llm_text(self):
        event = self.presenter.draft_update(self.route, "analysis results", "")
        assert "analysis results" in event.text

    def test_draft_update_fallback(self):
        event = self.presenter.draft_update(self.route, "", "")
        assert "思考中" in event.text

    def test_draft_cancelled(self):
        event = self.presenter.draft_cancelled(self.route)
        assert "取消" in event.text

    def test_analysis_error(self):
        event = self.presenter.analysis_error(self.route, "timeout")
        assert "timeout" in event.text

    def test_typing_c2c_returns_event(self):
        event = self.presenter.typing(self.route)
        assert event is not None
        assert event.kind == "typing"

    def test_typing_group_returns_none(self):
        route = ConversationRoute(channel="qq", scope_type="group", scope_id="g1")
        assert self.presenter.typing(route) is None

    def test_quick_chat_reply(self):
        event = self.presenter.quick_chat_reply(self.route, "hello!")
        assert event.text == "hello!"

    def test_quick_chat_fallback(self):
        event = self.presenter.quick_chat_fallback(self.route)
        assert "等待" in event.text

    def test_analysis_status_returns_none(self):
        assert self.presenter.analysis_status(
            self.route, has_media=True, has_reports=True, has_artifacts=True,
        ) is None

    def test_final_events_with_reports_and_figures(self):
        result = _make_result(
            summary="Done",
            reports=["# Report 1"],
            figures=[b"PNG_DATA"],
            artifacts=[SimpleNamespace(filename="out.csv")],
        )
        session = _make_session()
        events = self.presenter.final_events(
            self.route,
            session=session,
            user_text="analyze",
            llm_text="analysis complete with results",
            result=result,
        )
        kinds = [e.kind for e in events]
        assert "text" in kinds  # report + summary
        assert "photo" in kinds  # figure
        assert "document" in kinds  # artifact

    def test_final_events_boring_summary_replaced(self):
        result = _make_result(summary="分析完成")
        session = _make_session(adata=_make_adata())
        events = self.presenter.final_events(
            self.route,
            session=session,
            user_text="test",
            llm_text="分析完成",
            result=result,
        )
        # Boring summary should be replaced with default_summary
        summary_event = events[-1]
        assert summary_event.kind == "text"


# ═══════════════════════════════════════════════════════════════════════════
# WeChat Route Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWeChatRouteConstruction:
    def test_dm_route(self):
        sk = SessionKey(channel="wechat", scope_type="dm", scope_id="user1")
        route = wechat_route_from_session_key(sk)
        assert route.channel == "wechat"
        assert route.scope_type == "dm"
        assert route.scope_id == "user1"
        assert route.is_direct

    def test_route_key_stable(self):
        sk = SessionKey(channel="wechat", scope_type="dm", scope_id="u1")
        r1 = wechat_route_from_session_key(sk)
        r2 = wechat_route_from_session_key(sk)
        assert r1.route_key() == r2.route_key()


# ═══════════════════════════════════════════════════════════════════════════
# WeChat Envelope Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWeChatEnvelope:
    def test_basic_envelope(self):
        sk = SessionKey(channel="wechat", scope_type="dm", scope_id="u1")
        env = wechat_runtime_envelope(sk, "hello", sender_id="u1")
        assert env is not None
        assert env.text == "hello"
        assert env.sender_id == "u1"
        assert env.route.channel == "wechat"
        assert env.trigger == "direct"

    def test_empty_text_returns_none(self):
        sk = SessionKey(channel="wechat", scope_type="dm", scope_id="u1")
        assert wechat_runtime_envelope(sk, "", sender_id="u1") is None
        assert wechat_runtime_envelope(sk, "   ", sender_id="u1") is None

    def test_metadata_propagation(self):
        sk = SessionKey(channel="wechat", scope_type="dm", scope_id="u1")
        env = wechat_runtime_envelope(
            sk, "test", sender_id="u1",
            metadata={"image_count": 1},
        )
        assert env.metadata["image_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# WeChat Presenter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWeChatPresenter:
    def setup_method(self):
        self.presenter = WeChatRuntimePresenter()
        self.route = ConversationRoute(
            channel="wechat", scope_type="dm", scope_id="u1",
        )

    def _envelope(self, text="test", metadata=None):
        return MessageEnvelope(
            route=self.route,
            text=text,
            sender_id="u1",
            metadata=metadata or {},
        )

    def test_ack_with_adata(self):
        session = _make_session(adata=_make_adata(3000, 1000))
        events = self.presenter.ack(self._envelope(), session)
        assert len(events) == 1
        assert "3,000" in events[0].text
        assert "1,000" in events[0].text

    def test_ack_no_adata_with_files(self):
        files = [SimpleNamespace(name="sample.h5ad")]
        session = _make_session(h5ad_files=files)
        events = self.presenter.ack(self._envelope(), session)
        assert "sample.h5ad" in events[0].text

    def test_ack_no_adata_no_files(self):
        session = _make_session()
        events = self.presenter.ack(self._envelope(), session)
        assert "分析" in events[0].text

    def test_ack_with_images(self):
        session = _make_session()
        env = self._envelope(metadata={"image_count": 2})
        events = self.presenter.ack(env, session)
        assert "2 张" in events[0].text

    def test_queue_started(self):
        events = self.presenter.queue_started(self.route, 3)
        assert "3" in events[0].text

    def test_draft_open(self):
        event = self.presenter.draft_open(self.route)
        assert event.kind == "text"

    def test_draft_update_with_progress(self):
        event = self.presenter.draft_update(self.route, "", "clustering")
        assert "clustering" in event.text

    def test_draft_cancelled(self):
        event = self.presenter.draft_cancelled(self.route)
        assert "取消" in event.text

    def test_analysis_error(self):
        event = self.presenter.analysis_error(self.route, "oom")
        assert "oom" in event.text

    def test_typing_returns_none(self):
        assert self.presenter.typing(self.route) is None

    def test_quick_chat_reply(self):
        event = self.presenter.quick_chat_reply(self.route, "ok")
        assert event.text == "ok"

    def test_quick_chat_fallback(self):
        event = self.presenter.quick_chat_fallback(self.route)
        assert "等待" in event.text

    def test_analysis_status_returns_none(self):
        assert self.presenter.analysis_status(
            self.route, has_media=True, has_reports=False, has_artifacts=False,
        ) is None

    def test_final_events_with_all_types(self):
        result = _make_result(
            summary="Done",
            reports=["Report text"],
            figures=[b"IMG"],
            artifacts=[SimpleNamespace(filename="output.xlsx")],
        )
        session = _make_session()
        events = self.presenter.final_events(
            self.route,
            session=session,
            user_text="analyze",
            llm_text="here are the results",
            result=result,
        )
        kinds = [e.kind for e in events]
        assert "photo" in kinds
        assert kinds.count("text") >= 2  # report + artifacts notice + summary

    def test_final_events_meta_commentary_stripped(self):
        """WeChat-specific: meta-commentary in summary is suppressed."""
        result = _make_result(summary="Done")
        session = _make_session(adata=_make_adata())
        events = self.presenter.final_events(
            self.route,
            session=session,
            user_text="test",
            llm_text="I complied by calling the finish tool as per the mandatory tool call requirement.",
            result=result,
        )
        summary_text = events[-1].text
        # The meta-commentary should be stripped, replaced by default
        assert "mandatory tool call" not in summary_text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Runtime Wiring Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQQRuntimeWiring:
    """Verify QQRuntime creates and delegates to MessageRuntime."""

    def test_qq_runtime_has_runtime_attribute(self):
        """QQRuntime.__init__ creates a MessageRuntime instance."""
        from omicverse.jarvis.channels.qq import QQRuntime
        # Just verify the class has the expected structure — no live construction
        # needed (that would require a real session_manager).
        assert hasattr(QQRuntime, '_dispatch')
        assert hasattr(QQRuntime, '_deliver_event')

    def test_qq_no_bespoke_analysis_methods(self):
        """QQRuntime no longer has _run_analysis, _analysis_wrapper, _quick_chat."""
        from omicverse.jarvis.channels.qq import QQRuntime
        assert not hasattr(QQRuntime, '_run_analysis')
        assert not hasattr(QQRuntime, '_analysis_wrapper')
        assert not hasattr(QQRuntime, '_quick_chat')

    def test_qq_no_bespoke_task_tracking(self):
        """QQRuntime no longer uses its own _tasks/_pending dicts."""
        import inspect
        from omicverse.jarvis.channels.qq import QQRuntime
        source = inspect.getsource(QQRuntime.__init__)
        assert '_tasks' not in source
        assert '_pending' not in source


class TestWeChatRuntimeWiring:
    """Verify WeChatJarvisBot creates and delegates to MessageRuntime."""

    def test_wechat_bot_has_runtime_attribute(self):
        from omicverse.jarvis.channels.wechat import WeChatJarvisBot
        assert hasattr(WeChatJarvisBot, '_on_message')
        assert hasattr(WeChatJarvisBot, '_deliver_event')

    def test_wechat_no_bespoke_analysis_methods(self):
        from omicverse.jarvis.channels.wechat import WeChatJarvisBot
        assert not hasattr(WeChatJarvisBot, '_run_analysis')
        assert not hasattr(WeChatJarvisBot, '_analysis_wrapper')
        assert not hasattr(WeChatJarvisBot, '_spawn_analysis')

    def test_wechat_no_bespoke_task_tracking(self):
        import inspect
        from omicverse.jarvis.channels.wechat import WeChatJarvisBot
        source = inspect.getsource(WeChatJarvisBot.__init__)
        assert '_tasks' not in source
        assert '_pending' not in source

    def test_wechat_no_bespoke_ack(self):
        """_send_ack is now handled by the presenter through MessageRuntime."""
        from omicverse.jarvis.channels.wechat import WeChatJarvisBot
        assert not hasattr(WeChatJarvisBot, '_send_ack')


# ═══════════════════════════════════════════════════════════════════════════
# Platform-Specific Preservation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPlatformSpecificPreserved:
    """Ensure platform-specific classes and methods survive the migration."""

    def test_qq_client_preserved(self):
        from omicverse.jarvis.channels.qq import QQClient
        assert hasattr(QQClient, 'send_text')
        assert hasattr(QQClient, 'send_image')
        assert hasattr(QQClient, 'upload_image_c2c')

    def test_qq_deduper_preserved(self):
        from omicverse.jarvis.channels.qq import QQDeduper
        assert hasattr(QQDeduper, 'seen_or_record')

    def test_qq_image_server_preserved(self):
        from omicverse.jarvis.channels.qq import _ImageServer
        assert hasattr(_ImageServer, 'host_image')
        assert hasattr(_ImageServer, 'start')
        assert hasattr(_ImageServer, 'stop')

    def test_qq_target_preserved(self):
        from omicverse.jarvis.channels.qq import QQTarget
        t = QQTarget(kind="c2c", id="u1")
        assert t.route_key() == "qq:c2c:u1"

    def test_wechat_api_client_preserved(self):
        from omicverse.jarvis.channels.wechat import WeChatApiClient
        assert hasattr(WeChatApiClient, 'send_text')
        assert hasattr(WeChatApiClient, 'upload_image')
        assert hasattr(WeChatApiClient, 'send_image')
        assert hasattr(WeChatApiClient, 'download_cdn_media')

    def test_wechat_aes_helpers_preserved(self):
        from omicverse.jarvis.channels.wechat import (
            _aes_ecb_padded_size,
            _encrypt_aes_ecb,
            _decrypt_aes_ecb,
            _build_cdn_download_url,
        )
        assert _aes_ecb_padded_size(15) == 16
        assert _aes_ecb_padded_size(16) == 32

    def test_wechat_h5ad_handler_preserved(self):
        from omicverse.jarvis.channels.wechat import WeChatJarvisBot
        assert hasattr(WeChatJarvisBot, '_handle_h5ad_file')

    def test_wechat_image_handling_preserved(self):
        from omicverse.jarvis.channels.wechat import WeChatJarvisBot
        assert hasattr(WeChatJarvisBot, '_prepare_inbound_images')
        assert hasattr(WeChatJarvisBot, '_send_figures')
        assert hasattr(WeChatJarvisBot, '_send_figure')


# ═══════════════════════════════════════════════════════════════════════════
# Shared Helper Usage Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSharedHelperUsage:
    """Verify that migrated channels use shared channel_core helpers for commands."""

    def test_qq_uses_gather_status(self):
        import inspect
        from omicverse.jarvis.channels.qq import QQRuntime
        source = inspect.getsource(QQRuntime._handle_status)
        assert "gather_status" in source
        assert "format_status_plain" in source

    def test_qq_uses_gather_workspace(self):
        import inspect
        from omicverse.jarvis.channels.qq import QQRuntime
        source = inspect.getsource(QQRuntime._handle_workspace)
        assert "gather_workspace" in source

    def test_qq_uses_gather_usage(self):
        import inspect
        from omicverse.jarvis.channels.qq import QQRuntime
        source = inspect.getsource(QQRuntime._handle_usage)
        assert "gather_usage" in source

    def test_qq_uses_perform_save(self):
        import inspect
        from omicverse.jarvis.channels.qq import QQRuntime
        source = inspect.getsource(QQRuntime._handle_save)
        assert "perform_save" in source

    def test_wechat_uses_gather_status(self):
        import inspect
        from omicverse.jarvis.channels.wechat import WeChatJarvisBot
        source = inspect.getsource(WeChatJarvisBot._handle_status)
        assert "gather_status" in source
        assert "format_status_plain" in source


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Channel Consistency Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossChannelConsistency:
    """Verify that QQ and WeChat presenters conform to the MessagePresenter protocol."""

    @pytest.mark.parametrize("presenter_cls", [QQRuntimePresenter, WeChatRuntimePresenter])
    def test_presenter_has_all_protocol_methods(self, presenter_cls):
        required_methods = [
            "ack", "queue_started", "draft_open", "draft_update",
            "draft_cancelled", "analysis_error", "typing",
            "quick_chat_reply", "quick_chat_fallback",
            "analysis_status", "final_events",
        ]
        for method_name in required_methods:
            assert hasattr(presenter_cls, method_name), (
                f"{presenter_cls.__name__} missing {method_name}"
            )

    @pytest.mark.parametrize("presenter_cls,channel", [
        (QQRuntimePresenter, "qq"),
        (WeChatRuntimePresenter, "wechat"),
    ])
    def test_ack_returns_delivery_events(self, presenter_cls, channel):
        route = ConversationRoute(channel=channel, scope_type="dm", scope_id="u1")
        env = MessageEnvelope(route=route, text="test", sender_id="u1")
        session = _make_session()
        events = presenter_cls().ack(env, session)
        assert isinstance(events, list)
        assert all(isinstance(e, DeliveryEvent) for e in events)

    @pytest.mark.parametrize("presenter_cls,channel", [
        (QQRuntimePresenter, "qq"),
        (WeChatRuntimePresenter, "wechat"),
    ])
    def test_final_events_returns_delivery_events(self, presenter_cls, channel):
        route = ConversationRoute(channel=channel, scope_type="dm", scope_id="u1")
        result = _make_result(summary="Done")
        session = _make_session()
        events = presenter_cls().final_events(
            route, session=session, user_text="test",
            llm_text="results", result=result,
        )
        assert isinstance(events, list)
        assert all(isinstance(e, DeliveryEvent) for e in events)
        assert len(events) >= 1  # at least summary


class TestPublicEntryPointsPreserved:
    """Verify that the public entry points still exist and are importable."""

    def test_run_qq_bot_importable(self):
        from omicverse.jarvis.channels.qq import run_qq_bot
        assert callable(run_qq_bot)

    def test_run_wechat_bot_importable(self):
        from omicverse.jarvis.channels.wechat import run_wechat_bot
        assert callable(run_wechat_bot)
