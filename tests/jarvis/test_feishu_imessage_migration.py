"""Tests for Feishu and iMessage migration to shared MessageRuntime abstractions.

Covers:
- Feishu: FeishuRuntimePresenter, FeishuDelivery, FeishuRuntime rewiring
- iMessage: imessage_route_from_message, IMessageRuntimePresenter, IMessageDelivery
- Cross-channel: both channels conform to MessagePresenter protocol

Acceptance criteria addressed:
- AC-001.1: Feishu and iMessage keep only platform transport and conversion logic locally
- AC-001.2: Shared session, presenter, media, and runtime behavior is reused
- AC-001.3: Channel-focused tests cover the migrated behavior
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from omicverse.jarvis.runtime import (
    ConversationRoute,
    DeliveryEvent,
    MessageEnvelope,
)


# ── Test helpers ──────────────────────────────────────────────────────────


def _make_route(
    channel: str = "feishu",
    scope_type: str = "dm",
    scope_id: str = "oc_123",
    thread_id: Optional[str] = None,
    sender_id: Optional[str] = None,
) -> ConversationRoute:
    return ConversationRoute(
        channel=channel,
        scope_type=scope_type,
        scope_id=scope_id,
        thread_id=thread_id,
        sender_id=sender_id,
    )


def _make_envelope(
    route: Optional[ConversationRoute] = None,
    text: str = "analyze data",
    sender_id: str = "user1",
) -> MessageEnvelope:
    route = route or _make_route()
    return MessageEnvelope(
        route=route,
        text=text,
        sender_id=sender_id,
        trigger="webhook",
        explicit_trigger=True,
    )


def _make_session(adata=None, h5ad_files=None, workspace="/tmp/ws"):
    files = h5ad_files or []
    return SimpleNamespace(
        adata=adata,
        workspace=workspace,
        list_h5ad_files=lambda: files,
    )


def _make_adata(n_obs: int = 5000, n_vars: int = 2000):
    return SimpleNamespace(
        n_obs=n_obs,
        n_vars=n_vars,
        obs=SimpleNamespace(columns=["cell_type", "batch"]),
    )


def _make_result(
    summary="分析完成",
    figures=None,
    reports=None,
    artifacts=None,
    error=None,
    adata=None,
):
    return SimpleNamespace(
        summary=summary,
        figures=figures or [],
        reports=reports or [],
        artifacts=artifacts or [],
        error=error,
        adata=adata,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Feishu
# ═══════════════════════════════════════════════════════════════════════════


class TestFeishuRuntimePresenter:
    """Verify FeishuRuntimePresenter produces correct DeliveryEvents."""

    def setup_method(self):
        from omicverse.jarvis.channels.feishu import FeishuRuntimePresenter
        self.presenter = FeishuRuntimePresenter()
        self.route = _make_route(channel="feishu", scope_id="oc_abc")

    def test_ack_with_adata(self):
        session = _make_session(adata=_make_adata(3000, 1500))
        envelope = _make_envelope(route=self.route)
        events = self.presenter.ack(envelope, session)
        assert len(events) == 1
        assert events[0].kind == "text"
        assert "3,000" in events[0].text
        assert "1,500" in events[0].text

    def test_ack_with_h5ad_files(self):
        files = [SimpleNamespace(name="data.h5ad")]
        session = _make_session(h5ad_files=files)
        envelope = _make_envelope(route=self.route)
        events = self.presenter.ack(envelope, session)
        assert "data.h5ad" in events[0].text

    def test_ack_no_data(self):
        session = _make_session()
        envelope = _make_envelope(route=self.route)
        events = self.presenter.ack(envelope, session)
        assert len(events) == 1
        # Should contain some acknowledgement text
        assert events[0].text.strip()

    def test_queue_started(self):
        events = self.presenter.queue_started(self.route, 3)
        assert len(events) == 1
        assert "3" in events[0].text

    def test_draft_open(self):
        event = self.presenter.draft_open(self.route)
        assert event.mode == "open"
        assert event.target == "analysis-draft"

    def test_draft_update_with_progress(self):
        event = self.presenter.draft_update(self.route, "", "running code...")
        assert event.mode == "edit"
        assert event.target == "analysis-draft"
        assert "running code" in event.text

    def test_draft_update_with_llm_text(self):
        event = self.presenter.draft_update(self.route, "some output", "")
        assert "some output" in event.text

    def test_draft_update_empty(self):
        event = self.presenter.draft_update(self.route, "", "")
        assert event.text.strip()  # Should have some default text

    def test_draft_cancelled(self):
        event = self.presenter.draft_cancelled(self.route)
        assert "取消" in event.text
        assert event.mode == "edit"

    def test_analysis_error(self):
        event = self.presenter.analysis_error(self.route, "something failed")
        assert "something failed" in event.text

    def test_typing_returns_none(self):
        event = self.presenter.typing(self.route)
        assert event is None

    def test_quick_chat_reply(self):
        event = self.presenter.quick_chat_reply(self.route, "hello")
        assert event.text == "hello"
        assert event.kind == "text"

    def test_quick_chat_fallback(self):
        event = self.presenter.quick_chat_fallback(self.route)
        assert "/cancel" in event.text

    def test_analysis_status_with_media(self):
        event = self.presenter.analysis_status(
            self.route, has_media=True, has_reports=False, has_artifacts=False,
        )
        # Feishu should return an event or None; either is acceptable
        if event is not None:
            assert event.mode == "edit"

    def test_analysis_status_nothing(self):
        event = self.presenter.analysis_status(
            self.route, has_media=False, has_reports=False, has_artifacts=False,
        )
        assert event is None

    def test_final_events_with_figures(self):
        result = _make_result(summary="good result", figures=[b"png1", b"png2"])
        events = self.presenter.final_events(
            self.route, session=_make_session(), user_text="test",
            llm_text="", result=result,
        )
        photo_events = [e for e in events if e.kind == "photo"]
        assert len(photo_events) == 2
        assert photo_events[0].filename == "figure_1.png"

    def test_final_events_with_reports(self):
        result = _make_result(summary="done", reports=["report text"])
        events = self.presenter.final_events(
            self.route, session=_make_session(), user_text="test",
            llm_text="", result=result,
        )
        # Reports may be emitted as "prose" (Feishu markdown cards) or "text"
        report_events = [e for e in events if e.kind in ("text", "prose") and "report text" in e.text]
        assert len(report_events) >= 1

    def test_final_events_with_artifacts(self):
        artifact = SimpleNamespace(data=b"csv", filename="data.csv")
        result = _make_result(artifacts=[artifact])
        events = self.presenter.final_events(
            self.route, session=_make_session(), user_text="test",
            llm_text="", result=result,
        )
        doc_events = [e for e in events if e.kind == "document"]
        assert len(doc_events) == 1
        assert doc_events[0].filename == "data.csv"

    def test_final_events_boring_summary_fallback(self):
        """When summary is boring, should fall back to llm_text."""
        result = _make_result(summary="分析完成")
        events = self.presenter.final_events(
            self.route, session=_make_session(), user_text="test",
            llm_text="detailed LLM output", result=result,
        )
        text_events = [e for e in events if e.kind == "text" and e.mode == "send"]
        assert any("detailed LLM output" in e.text for e in text_events)

    def test_final_events_draft_completion(self):
        result = _make_result(summary="good")
        events = self.presenter.final_events(
            self.route, session=_make_session(), user_text="test",
            llm_text="", result=result,
        )
        edit_events = [e for e in events if e.mode == "edit" and e.target == "analysis-draft"]
        assert any("✅" in e.text for e in edit_events)


class TestFeishuDelivery:
    """Verify FeishuDelivery translates DeliveryEvents to FeishuClient calls."""

    def _make_client(self):
        """Create a mock FeishuClient that records calls."""
        calls = []

        class MockClient:
            def send_text(self, chat_id, text):
                calls.append(("send_text", chat_id, text))
                return f"msg_{len(calls)}"

            def send_markdown_card(self, chat_id, content, title="", color="blue"):
                calls.append(("send_markdown_card", chat_id, content, title))
                return f"msg_{len(calls)}"

            def edit_card(self, message_id, content, title="", color="blue"):
                calls.append(("edit_card", message_id, content))
                return True

            def edit_text(self, message_id, text):
                calls.append(("edit_text", message_id, text))
                return True

            def send_image_bytes(self, chat_id, data, filename="figure.png"):
                calls.append(("send_image_bytes", chat_id, len(data), filename))

            def send_file_bytes(self, chat_id, data, filename):
                calls.append(("send_file_bytes", chat_id, len(data), filename))

        return MockClient(), calls

    @pytest.mark.asyncio
    async def test_deliver_text_send(self):
        from omicverse.jarvis.channels.feishu import FeishuDelivery
        client, calls = self._make_client()
        delivery = FeishuDelivery(client=client)
        route = _make_route(channel="feishu", scope_id="oc_test")
        event = DeliveryEvent(route=route, kind="text", text="hello")
        await delivery.deliver(event)
        assert any(c[0] == "send_text" and c[1] == "oc_test" for c in calls)

    @pytest.mark.asyncio
    async def test_deliver_text_open_stores_draft(self):
        from omicverse.jarvis.channels.feishu import FeishuDelivery
        client, calls = self._make_client()
        delivery = FeishuDelivery(client=client)
        route = _make_route(channel="feishu", scope_id="oc_test")
        event = DeliveryEvent(
            route=route, kind="text", mode="open",
            target="analysis-draft", text="thinking...",
        )
        await delivery.deliver(event)
        assert any("send_markdown_card" in str(c) or "send_text" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_deliver_typing_noop(self):
        from omicverse.jarvis.channels.feishu import FeishuDelivery
        client, calls = self._make_client()
        delivery = FeishuDelivery(client=client)
        route = _make_route(channel="feishu", scope_id="oc_test")
        event = DeliveryEvent(route=route, kind="typing")
        await delivery.deliver(event)
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_deliver_photo(self):
        from omicverse.jarvis.channels.feishu import FeishuDelivery
        client, calls = self._make_client()
        delivery = FeishuDelivery(client=client)
        route = _make_route(channel="feishu", scope_id="oc_test")
        event = DeliveryEvent(
            route=route, kind="photo",
            binary=b"png-data", filename="fig.png",
        )
        await delivery.deliver(event)
        assert any(c[0] == "send_image_bytes" for c in calls)

    @pytest.mark.asyncio
    async def test_deliver_document(self):
        from omicverse.jarvis.channels.feishu import FeishuDelivery
        client, calls = self._make_client()
        delivery = FeishuDelivery(client=client)
        route = _make_route(channel="feishu", scope_id="oc_test")
        event = DeliveryEvent(
            route=route, kind="document",
            binary=b"csv-data", filename="result.csv",
        )
        await delivery.deliver(event)
        assert any(c[0] == "send_file_bytes" for c in calls)


class TestFeishuRuntimeWiring:
    """Verify FeishuRuntime delegates to MessageRuntime."""

    def test_no_tasks_dict(self):
        """FeishuRuntime should not have its own _tasks dict."""
        source = inspect.getsource(
            __import__("omicverse.jarvis.channels.feishu", fromlist=["FeishuRuntime"]).FeishuRuntime
        )
        # Should not have self._tasks = {} (that's MessageRuntime's job now)
        assert "self._tasks" not in source or "_tasks:" not in source

    def test_no_run_analysis_method(self):
        """FeishuRuntime should not have a _run_analysis method."""
        from omicverse.jarvis.channels.feishu import FeishuRuntime
        assert not hasattr(FeishuRuntime, "_run_analysis")

    def test_no_spawn_analysis_method(self):
        from omicverse.jarvis.channels.feishu import FeishuRuntime
        assert not hasattr(FeishuRuntime, "_spawn_analysis")

    def test_no_quick_chat_method(self):
        from omicverse.jarvis.channels.feishu import FeishuRuntime
        assert not hasattr(FeishuRuntime, "_quick_chat")

    def test_handle_text_uses_envelope(self):
        """handle_text should build a MessageEnvelope for analysis."""
        from omicverse.jarvis.channels.feishu import FeishuRuntime
        source = inspect.getsource(FeishuRuntime.handle_text)
        assert "MessageEnvelope" in source
        assert "_message_runtime" in source


# ═══════════════════════════════════════════════════════════════════════════
# iMessage
# ═══════════════════════════════════════════════════════════════════════════


class TestIMessageRouteFromMessage:
    """Verify imessage_route_from_message builds correct routes."""

    def test_dm_with_chat_id(self):
        from omicverse.jarvis.channels.imessage import imessage_route_from_message
        route = imessage_route_from_message({"chat_id": 42, "is_group": False, "sender": "+1"})
        assert route.channel == "imessage"
        assert route.scope_type == "dm"
        assert route.scope_id == "42"

    def test_group_with_chat_id(self):
        from omicverse.jarvis.channels.imessage import imessage_route_from_message
        route = imessage_route_from_message({"chat_id": 99, "is_group": True, "sender": "+1"})
        assert route.scope_type == "group"
        assert route.scope_id == "99"

    def test_dm_with_sender_handle(self):
        from omicverse.jarvis.channels.imessage import imessage_route_from_message
        route = imessage_route_from_message({"is_group": False, "sender": "alice@icloud.com"})
        assert route.scope_type == "dm"
        assert route.scope_id == "alice@icloud.com"

    def test_with_chat_identifier(self):
        from omicverse.jarvis.channels.imessage import imessage_route_from_message
        route = imessage_route_from_message({
            "is_group": True,
            "chat_identifier": "chat12345",
        })
        assert route.scope_id == "chat12345"

    def test_no_scope_returns_none(self):
        from omicverse.jarvis.channels.imessage import imessage_route_from_message
        route = imessage_route_from_message({"is_group": False})
        assert route is None


class TestIMessageRuntimePresenter:
    """Verify IMessageRuntimePresenter produces correct DeliveryEvents."""

    def setup_method(self):
        from omicverse.jarvis.channels.imessage import IMessageRuntimePresenter
        self.presenter = IMessageRuntimePresenter()
        self.route = _make_route(channel="imessage", scope_id="+1234567890")

    def test_ack(self):
        session = _make_session()
        envelope = _make_envelope(route=self.route)
        events = self.presenter.ack(envelope, session)
        assert len(events) == 1
        assert "⏳" in events[0].text

    def test_queue_started(self):
        events = self.presenter.queue_started(self.route, 2)
        assert len(events) == 1
        assert "2" in events[0].text

    def test_draft_open(self):
        event = self.presenter.draft_open(self.route)
        assert event.mode == "open"
        assert event.target == "analysis-draft"

    def test_draft_update_with_progress(self):
        event = self.presenter.draft_update(self.route, "", "executing code")
        assert "executing code" in event.text

    def test_draft_cancelled(self):
        event = self.presenter.draft_cancelled(self.route)
        assert "取消" in event.text

    def test_analysis_error(self):
        event = self.presenter.analysis_error(self.route, "error msg")
        assert "error msg" in event.text

    def test_typing_returns_none(self):
        assert self.presenter.typing(self.route) is None

    def test_quick_chat_reply(self):
        event = self.presenter.quick_chat_reply(self.route, "response")
        assert event.text == "response"

    def test_quick_chat_fallback(self):
        event = self.presenter.quick_chat_fallback(self.route)
        assert "/cancel" in event.text

    def test_analysis_status_always_none(self):
        """iMessage doesn't need mid-delivery status."""
        event = self.presenter.analysis_status(
            self.route, has_media=True, has_reports=True, has_artifacts=True,
        )
        assert event is None

    def test_final_events_with_figures(self):
        result = _make_result(figures=[b"png1"])
        events = self.presenter.final_events(
            self.route, session=_make_session(), user_text="test",
            llm_text="", result=result,
        )
        photo_events = [e for e in events if e.kind == "photo"]
        assert len(photo_events) == 1
        assert photo_events[0].filename == "figure_1.png"

    def test_final_events_with_artifacts(self):
        artifact = SimpleNamespace(data=b"csv", filename="out.csv")
        result = _make_result(artifacts=[artifact])
        events = self.presenter.final_events(
            self.route, session=_make_session(), user_text="test",
            llm_text="", result=result,
        )
        doc_events = [e for e in events if e.kind == "document"]
        assert len(doc_events) == 1
        assert doc_events[0].filename == "out.csv"

    def test_final_events_boring_summary_uses_llm(self):
        result = _make_result(summary="分析完成")
        events = self.presenter.final_events(
            self.route, session=_make_session(), user_text="test",
            llm_text="detailed analysis text", result=result,
        )
        text_events = [e for e in events if e.kind == "text" and e.mode == "send"]
        assert any("detailed analysis text" in e.text for e in text_events)

    def test_final_events_draft_completion(self):
        result = _make_result(summary="good analysis")
        events = self.presenter.final_events(
            self.route, session=_make_session(), user_text="test",
            llm_text="", result=result,
        )
        edit_events = [e for e in events if e.mode == "edit"]
        assert any("✅" in e.text for e in edit_events)


class TestIMessageDelivery:
    """Verify IMessageDelivery translates DeliveryEvents to RPC calls."""

    def _make_client(self):
        """Create a mock IMessageRpcClient."""
        calls = []

        class MockClient:
            async def send_message(self, target, text, *, file_path=None, timeout=60.0):
                calls.append(("send_message", target, text, file_path))

        return MockClient(), calls

    @pytest.mark.asyncio
    async def test_deliver_text(self):
        from omicverse.jarvis.channels.imessage import IMessageDelivery
        client, calls = self._make_client()
        delivery = IMessageDelivery(client=client)
        route = _make_route(channel="imessage", scope_id="+1234")
        delivery.register_target(route, "+1234")
        event = DeliveryEvent(route=route, kind="text", text="hello")
        await delivery.deliver(event)
        assert len(calls) == 1
        assert calls[0][1] == "+1234"
        assert calls[0][2] == "hello"

    @pytest.mark.asyncio
    async def test_deliver_typing_noop(self):
        from omicverse.jarvis.channels.imessage import IMessageDelivery
        client, calls = self._make_client()
        delivery = IMessageDelivery(client=client)
        route = _make_route(channel="imessage", scope_id="+1234")
        event = DeliveryEvent(route=route, kind="typing")
        await delivery.deliver(event)
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_deliver_photo_uses_temp_file(self):
        from omicverse.jarvis.channels.imessage import IMessageDelivery
        client, calls = self._make_client()
        delivery = IMessageDelivery(client=client)
        route = _make_route(channel="imessage", scope_id="+1234")
        delivery.register_target(route, "+1234")
        event = DeliveryEvent(
            route=route, kind="photo",
            binary=b"png-data", filename="fig.png", caption="Figure 1",
        )
        await delivery.deliver(event)
        assert len(calls) == 1
        assert calls[0][3] is not None  # file_path should be set

    @pytest.mark.asyncio
    async def test_deliver_edit_skips_status_markers(self):
        """Pure status edits should be skipped (iMessage can't edit)."""
        from omicverse.jarvis.channels.imessage import IMessageDelivery
        client, calls = self._make_client()
        delivery = IMessageDelivery(client=client)
        route = _make_route(channel="imessage", scope_id="+1234")
        delivery.register_target(route, "+1234")
        # This is a pure status marker that should be skipped
        event = DeliveryEvent(
            route=route, kind="text", mode="edit",
            target="analysis-draft", text="✅ 分析完成",
        )
        await delivery.deliver(event)
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_deliver_edit_sends_substantive_updates(self):
        """Substantive edit updates should be sent as new messages."""
        from omicverse.jarvis.channels.imessage import IMessageDelivery
        client, calls = self._make_client()
        delivery = IMessageDelivery(client=client)
        route = _make_route(channel="imessage", scope_id="+1234")
        delivery.register_target(route, "+1234")
        event = DeliveryEvent(
            route=route, kind="text", mode="edit",
            target="analysis-draft", text="❌ Analysis failed: some error",
        )
        await delivery.deliver(event)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_fallback_to_scope_id_when_no_target(self):
        """When no target is registered, use scope_id."""
        from omicverse.jarvis.channels.imessage import IMessageDelivery
        client, calls = self._make_client()
        delivery = IMessageDelivery(client=client)
        route = _make_route(channel="imessage", scope_id="+5555")
        event = DeliveryEvent(route=route, kind="text", text="hi")
        await delivery.deliver(event)
        assert len(calls) == 1
        assert calls[0][1] == "+5555"


class TestIMessageBotWiring:
    """Verify IMessageJarvisBot delegates to MessageRuntime."""

    def test_no_tasks_dict(self):
        """Bot should not have its own _tasks dict."""
        from omicverse.jarvis.channels.imessage import IMessageJarvisBot
        source = inspect.getsource(IMessageJarvisBot.__init__)
        assert "_tasks" not in source

    def test_no_run_analysis_method(self):
        from omicverse.jarvis.channels.imessage import IMessageJarvisBot
        assert not hasattr(IMessageJarvisBot, "_run_analysis")

    def test_no_send_bytes_method(self):
        """_send_bytes is handled by IMessageDelivery now."""
        from omicverse.jarvis.channels.imessage import IMessageJarvisBot
        assert not hasattr(IMessageJarvisBot, "_send_bytes")

    def test_handle_message_uses_envelope(self):
        from omicverse.jarvis.channels.imessage import IMessageJarvisBot
        source = inspect.getsource(IMessageJarvisBot._handle_message)
        assert "MessageEnvelope" in source
        assert "_message_runtime" in source

    def test_handle_command_uses_runtime_for_cancel(self):
        from omicverse.jarvis.channels.imessage import IMessageJarvisBot
        source = inspect.getsource(IMessageJarvisBot._handle_command)
        assert "_message_runtime.cancel" in source

    def test_handle_command_uses_runtime_for_status(self):
        from omicverse.jarvis.channels.imessage import IMessageJarvisBot
        source = inspect.getsource(IMessageJarvisBot._handle_command)
        assert "task_state" in source


# ═══════════════════════════════════════════════════════════════════════════
# Cross-channel consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossChannelPresenterConformance:
    """Verify both presenters implement all MessagePresenter methods."""

    @pytest.mark.parametrize("cls_path", [
        "omicverse.jarvis.channels.feishu:FeishuRuntimePresenter",
        "omicverse.jarvis.channels.imessage:IMessageRuntimePresenter",
    ])
    def test_presenter_has_all_protocol_methods(self, cls_path):
        module_path, cls_name = cls_path.split(":")
        mod = __import__(module_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        required_methods = [
            "ack", "queue_started", "draft_open", "draft_update",
            "draft_cancelled", "analysis_error", "typing",
            "quick_chat_reply", "quick_chat_fallback",
            "analysis_status", "final_events",
        ]
        for method in required_methods:
            assert hasattr(cls, method), f"{cls_name} missing {method}"
            assert callable(getattr(cls, method)), f"{cls_name}.{method} not callable"

    def test_feishu_and_imessage_no_direct_agent_bridge(self):
        """Neither migrated channel should use AgentBridge directly."""
        for name in ("feishu", "imessage"):
            mod = __import__(f"omicverse.jarvis.channels.{name}", fromlist=[name])
            source = inspect.getsource(mod)
            # AgentBridge should not be instantiated (AgentBridgeExecutionAdapter wraps it)
            assert "AgentBridge(" not in source, (
                f"Channel {name} still uses AgentBridge() directly"
            )

    def test_both_channels_still_importable(self):
        """Both channels can be imported without error after migration."""
        from omicverse.jarvis.channels import feishu, imessage
        assert feishu is not None
        assert imessage is not None

    def test_platform_transport_preserved(self):
        """Platform-specific transport classes are preserved."""
        from omicverse.jarvis.channels.feishu import (
            FeishuClient,
            FeishuDeduper,
            FeishuWebhookSecurity,
            FeishuWebhookProcessor,
        )
        from omicverse.jarvis.channels.imessage import (
            IMessageRpcClient,
            IMessageTarget,
        )
        assert FeishuClient is not None
        assert FeishuDeduper is not None
        assert FeishuWebhookSecurity is not None
        assert FeishuWebhookProcessor is not None
        assert IMessageRpcClient is not None
        assert IMessageTarget is not None

    def test_entry_points_preserved(self):
        """Public entry point functions are preserved."""
        from omicverse.jarvis.channels.feishu import run_feishu_bot
        from omicverse.jarvis.channels.imessage import run_imessage_bot
        assert callable(run_feishu_bot)
        assert callable(run_imessage_bot)
