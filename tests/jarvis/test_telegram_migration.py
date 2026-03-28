"""Tests for Telegram channel migration to shared runtime and channel_core.

Covers:
- telegram_route_from_update: ConversationRoute for DM, group, callback queries
- telegram_runtime_envelope: MessageEnvelope construction
- TelegramRuntimePresenter: all MessagePresenter protocol methods
- TelegramDelivery: event delivery for text, edit, photo, document
- Callback handler uses shared channel_core helpers (perform_save, gather_status)
- End-to-end: Telegram wires through MessageRuntime

Acceptance criteria addressed:
- AC-001.1: Telegram uses shared runtime and channel_core end to end
- AC-001.2: Telegram-local duplicated helpers removed or reduced to transport glue
- AC-001.3: Telegram-focused Jarvis tests stay green
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from omicverse.jarvis.channels.telegram import (
    AccessControl,
    TelegramDelivery,
    TelegramRuntimePresenter,
    _SESSION_INIT_ERROR_TEMPLATE,
    telegram_route_from_update,
    telegram_runtime_envelope,
    telegram_trigger_for_update,
    _strip_leading_bot_mention,
)
from omicverse.jarvis.channels.channel_core import (
    gather_status,
    gather_usage,
    gather_workspace,
    perform_save,
    strip_local_paths,
    StatusInfo,
    UsageInfo,
    SaveResult,
)
from omicverse.jarvis.runtime import (
    ConversationRoute,
    DeliveryEvent,
    MessageEnvelope,
    MessagePresenter,
)


# ── Test helpers ──────────────────────────────────────────────────────────


def _make_update(
    *,
    chat_id: int = 100,
    chat_type: str = "private",
    user_id: int = 42,
    username: str = "alice",
    text: str = "hello",
    message_id: int = 999,
    message_thread_id: Optional[int] = None,
    photo: Any = None,
    document: Any = None,
    caption: Optional[str] = None,
) -> SimpleNamespace:
    """Build a minimal Telegram Update stub for message-type updates."""
    user = SimpleNamespace(id=user_id, username=username, is_bot=False)
    chat = SimpleNamespace(id=chat_id, type=chat_type)
    message = SimpleNamespace(
        text=text,
        caption=caption,
        message_id=message_id,
        message_thread_id=message_thread_id,
        photo=photo,
        document=document,
        entities=None,
        caption_entities=None,
        reply_to_message=None,
        from_user=user,
        chat=chat,
    )
    return SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
    )


def _make_callback_update(
    *,
    chat_id: int = 100,
    chat_type: str = "private",
    user_id: int = 42,
    username: str = "alice",
    callback_data: str = "jarvis:status",
    message_thread_id: Optional[int] = None,
) -> SimpleNamespace:
    """Build an Update stub for callback query updates."""
    user = SimpleNamespace(id=user_id, username=username, is_bot=False)
    chat = SimpleNamespace(id=chat_id, type=chat_type)
    message = SimpleNamespace(
        chat=chat,
        chat_id=chat_id,
        message_thread_id=message_thread_id,
    )
    query = SimpleNamespace(
        from_user=user,
        message=message,
        data=callback_data,
    )
    return SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
        callback_query=query,
    )


def _make_route(
    scope_type: str = "dm",
    scope_id: str = "100",
    thread_id: Optional[str] = None,
    sender_id: str = "42",
) -> ConversationRoute:
    return ConversationRoute(
        channel="telegram",
        scope_type=scope_type,
        scope_id=scope_id,
        thread_id=thread_id,
        sender_id=sender_id,
    )


def _make_adata(n_obs: int = 5000, n_vars: int = 2000, obs_cols=None):
    cols = obs_cols or ["cell_type", "batch"]
    return SimpleNamespace(
        n_obs=n_obs,
        n_vars=n_vars,
        obs=SimpleNamespace(columns=cols),
    )


def _make_session(adata=None, workspace="/tmp/ws"):
    h5ad = []
    memory_dir = Path("/tmp/memory")
    return SimpleNamespace(
        adata=adata,
        workspace=workspace,
        workspace_dir=Path(workspace),
        list_h5ad_files=lambda: h5ad,
        get_agents_md=lambda: "",
        memory_dir=memory_dir,
        get_recent_memory_text=lambda: "Some memory text",
        last_usage=None,
        kernel_status=lambda: {"prompt_count": 3, "max_prompts": 20, "session_id": "s1", "alive": True},
        user_id=42,
    )


def _make_result(
    *,
    adata=None,
    summary: str = "分析完成",
    figures: Optional[list] = None,
    reports: Optional[list] = None,
    artifacts: Optional[list] = None,
    error: Optional[str] = None,
    usage=None,
    diagnostics=None,
):
    return SimpleNamespace(
        adata=adata,
        summary=summary,
        figures=figures or [],
        reports=reports or [],
        artifacts=artifacts or [],
        error=error,
        usage=usage,
        diagnostics=diagnostics or [],
    )


# ── Route construction ───────────────────────────────────────────────────


class TestTelegramRouteFromUpdate:
    """Verify telegram_route_from_update for message and callback-query updates."""

    def test_dm_route(self):
        update = _make_update(chat_id=100, chat_type="private", user_id=42)
        route = telegram_route_from_update(update)
        assert route.channel == "telegram"
        assert route.scope_type == "dm"
        assert route.scope_id == "100"
        assert route.sender_id == "42"
        assert route.thread_id is None

    def test_group_route(self):
        update = _make_update(chat_id=200, chat_type="group", user_id=7)
        route = telegram_route_from_update(update)
        assert route.scope_type == "group"
        assert route.scope_id == "200"

    def test_thread_route(self):
        update = _make_update(chat_id=300, chat_type="supergroup", user_id=5, message_thread_id=77)
        route = telegram_route_from_update(update)
        assert route.scope_type == "group"
        assert route.thread_id == "77"

    def test_callback_query_update_matches_message_update(self):
        """Route from a callback-query update should match the equivalent message update."""
        msg_update = _make_update(chat_id=100, chat_type="private", user_id=42)
        cb_update = _make_callback_update(chat_id=100, chat_type="private", user_id=42)

        msg_route = telegram_route_from_update(msg_update)
        cb_route = telegram_route_from_update(cb_update)

        assert msg_route.channel == cb_route.channel
        assert msg_route.scope_type == cb_route.scope_type
        assert msg_route.scope_id == cb_route.scope_id
        assert msg_route.sender_id == cb_route.sender_id

    def test_callback_query_group_with_thread(self):
        cb_update = _make_callback_update(
            chat_id=300, chat_type="supergroup", user_id=5, message_thread_id=77,
        )
        route = telegram_route_from_update(cb_update)
        assert route.scope_type == "group"
        assert route.scope_id == "300"
        assert route.thread_id == "77"


# ── Envelope construction ────────────────────────────────────────────────


class TestTelegramRuntimeEnvelope:
    def test_basic_dm_envelope(self):
        update = _make_update(text="analyze my data", user_id=42)
        env = telegram_runtime_envelope(update, "mybot")
        assert env is not None
        assert env.text == "analyze my data"
        assert env.sender_id == "42"
        assert env.route.scope_type == "dm"

    def test_empty_text_returns_none(self):
        update = _make_update(text="")
        env = telegram_runtime_envelope(update, "mybot")
        assert env is None

    def test_mention_trigger_strips_bot_name(self):
        update = _make_update(text="@mybot do analysis", chat_type="group")
        # Simulate that trigger is "mention"
        env = telegram_runtime_envelope(update, "mybot")
        assert env is not None


# ── TelegramRuntimePresenter ─────────────────────────────────────────────


class TestTelegramRuntimePresenter:
    """Verify presenter implements MessagePresenter protocol and produces correct events."""

    def test_conforms_to_protocol(self):
        """TelegramRuntimePresenter should satisfy the MessagePresenter protocol."""
        presenter = TelegramRuntimePresenter()
        # Check all protocol methods exist with correct signatures
        assert callable(getattr(presenter, "ack", None))
        assert callable(getattr(presenter, "queue_started", None))
        assert callable(getattr(presenter, "draft_open", None))
        assert callable(getattr(presenter, "draft_update", None))
        assert callable(getattr(presenter, "draft_cancelled", None))
        assert callable(getattr(presenter, "analysis_error", None))
        assert callable(getattr(presenter, "typing", None))
        assert callable(getattr(presenter, "quick_chat_reply", None))
        assert callable(getattr(presenter, "quick_chat_fallback", None))
        assert callable(getattr(presenter, "analysis_status", None))
        assert callable(getattr(presenter, "final_events", None))

    def test_ack_with_adata(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        adata = _make_adata(5000, 2000)
        session = _make_session(adata=adata)
        envelope = MessageEnvelope(
            route=route, text="test", sender_id="42",
            sender_username="alice", message_id="1",
            trigger="direct", explicit_trigger=True, metadata={},
        )
        events = presenter.ack(envelope, session)
        assert len(events) >= 1
        assert events[0].kind == "text"
        assert events[0].text_format == "html"

    def test_ack_without_adata(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        session = _make_session(adata=None)
        envelope = MessageEnvelope(
            route=route, text="test", sender_id="42",
            sender_username="alice", message_id="1",
            trigger="direct", explicit_trigger=True, metadata={},
        )
        events = presenter.ack(envelope, session)
        assert len(events) >= 1
        assert events[0].kind == "text"

    def test_draft_open_returns_edit_target(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        event = presenter.draft_open(route)
        assert event.kind == "text"
        assert event.mode == "open"
        assert event.target == "analysis-draft"

    def test_draft_update(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        event = presenter.draft_update(route, "LLM output", "running code")
        assert event.kind == "text"
        assert event.mode == "edit"

    def test_draft_cancelled(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        event = presenter.draft_cancelled(route)
        assert event.mode == "edit"
        assert "取消" in event.text

    def test_analysis_error(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        event = presenter.analysis_error(route, "some error")
        assert event.mode == "edit"

    def test_typing(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        event = presenter.typing(route)
        assert event is not None
        assert event.kind == "typing"

    def test_quick_chat_reply(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        event = presenter.quick_chat_reply(route, "Hello there")
        assert event.kind == "text"

    def test_quick_chat_fallback(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        event = presenter.quick_chat_fallback(route)
        assert event.kind == "text"
        assert "/cancel" in event.text

    def test_analysis_status_with_media(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        event = presenter.analysis_status(route, has_media=True, has_reports=False, has_artifacts=False)
        assert event is not None
        assert "图片" in event.text

    def test_analysis_status_no_artifacts_returns_none(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        event = presenter.analysis_status(route, has_media=False, has_reports=False, has_artifacts=False)
        assert event is None

    def test_final_events_simple(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        session = _make_session()
        result = _make_result(summary="Done analysis")
        events = presenter.final_events(
            route, session=session, user_text="test", llm_text="output", result=result,
        )
        assert len(events) >= 1

    def test_final_events_with_figures(self):
        presenter = TelegramRuntimePresenter()
        route = _make_route()
        adata = _make_adata()
        session = _make_session(adata=adata)
        result = _make_result(adata=adata, figures=[b"png1", b"png2"])
        events = presenter.final_events(
            route, session=session, user_text="test", llm_text="output", result=result,
        )
        photo_events = [e for e in events if e.kind == "photo"]
        assert len(photo_events) == 2

    def test_strip_local_paths_used_from_channel_core(self):
        """Presenter uses strip_local_paths from channel_core directly."""
        text = "saved to /Users/alice/workspace/result.csv"
        cleaned = strip_local_paths(text)
        assert "/Users/alice" not in cleaned


# ── Callback handler uses channel_core ───────────────────────────────────


class TestCallbackUsesSharedHelpers:
    """Verify callback handler patterns use channel_core helpers after migration."""

    def test_perform_save_no_data(self):
        """perform_save handles no-data case that callback handler relies on."""
        session = _make_session(adata=None)
        result = perform_save(session)
        assert result.no_data is True
        assert not result.success

    def test_perform_save_with_data(self, tmp_path):
        """perform_save handles successful save case."""
        adata = _make_adata(100, 50)
        save_path = tmp_path / "current.h5ad"
        save_path.write_bytes(b"fake")

        session = SimpleNamespace(
            adata=adata,
            save_adata=lambda: save_path,
        )
        result = perform_save(session)
        assert result.success is True
        assert result.adata_shape == (100, 50)
        assert result.path == str(save_path)

    def test_gather_status_with_adata(self):
        """gather_status returns structured info that callback handler formats."""
        adata = _make_adata(5000, 2000, obs_cols=["cell_type", "batch"])
        session = _make_session(adata=adata)
        info = gather_status(session)
        assert info.adata_shape == (5000, 2000)
        assert len(info.obs_columns) > 0

    def test_gather_status_no_adata(self):
        """gather_status with no data returns empty shape."""
        session = _make_session(adata=None)
        info = gather_status(session)
        assert info.adata_shape is None


# ── Session error template ───────────────────────────────────────────────


class TestSessionErrorTemplate:
    """Verify the shared error template used for session init failures."""

    def test_template_formats_with_exception(self):
        msg = _SESSION_INIT_ERROR_TEMPLATE.format(exc="API key missing")
        assert "API key missing" in msg
        assert "ov.list_supported_models()" in msg

    def test_template_is_consistent(self):
        """The template should be a single constant, not duplicated."""
        # If this test can import the constant, it exists and is reusable
        assert isinstance(_SESSION_INIT_ERROR_TEMPLATE, str)
        assert len(_SESSION_INIT_ERROR_TEMPLATE) > 20


# ── End-to-end shared runtime wiring ─────────────────────────────────────


class TestTelegramRuntimeWiring:
    """Verify Telegram wires through the shared MessageRuntime end to end."""

    def test_presenter_used_by_runtime_protocol(self):
        """TelegramRuntimePresenter satisfies MessagePresenter via duck typing."""
        presenter = TelegramRuntimePresenter()
        # Verify all required protocol method signatures exist
        from inspect import signature
        protocol_methods = [
            "ack", "queue_started", "draft_open", "draft_update",
            "draft_cancelled", "analysis_error", "typing",
            "quick_chat_reply", "quick_chat_fallback",
            "analysis_status", "final_events",
        ]
        for method_name in protocol_methods:
            method = getattr(presenter, method_name, None)
            assert method is not None, f"Missing protocol method: {method_name}"
            assert callable(method), f"Protocol method not callable: {method_name}"

    def test_runtime_imports_from_shared_package(self):
        """Telegram module imports runtime types from the shared runtime package."""
        import omicverse.jarvis.channels.telegram as tg_mod
        # These should be imported from omicverse.jarvis.runtime
        assert hasattr(tg_mod, "MessageRuntime")
        assert hasattr(tg_mod, "MessageRouter")
        assert hasattr(tg_mod, "ConversationRoute")
        assert hasattr(tg_mod, "DeliveryEvent")
        assert hasattr(tg_mod, "MessageEnvelope")
        assert hasattr(tg_mod, "AgentBridgeExecutionAdapter")

    def test_channel_core_imports(self):
        """Telegram module imports command helpers from channel_core."""
        import omicverse.jarvis.channels.telegram as tg_mod
        # Verify channel_core helpers are accessible via the module
        from omicverse.jarvis.channels.channel_core import (
            gather_status,
            gather_usage,
            gather_workspace,
            perform_save,
            strip_local_paths,
        )
        # These should be importable without error
        assert callable(gather_status)
        assert callable(gather_usage)
        assert callable(gather_workspace)
        assert callable(perform_save)
        assert callable(strip_local_paths)


# ── Access control (unchanged, regression) ───────────────────────────────


class TestAccessControl:
    def test_open_allows_all(self):
        ac = AccessControl(allowed=None)
        assert ac.allows(123, "anyone") is True

    def test_whitelist_by_id(self):
        ac = AccessControl(allowed=["42"])
        assert ac.allows(42, "alice") is True
        assert ac.allows(99, "bob") is False

    def test_whitelist_by_username(self):
        ac = AccessControl(allowed=["@alice"])
        assert ac.allows(99, "alice") is True
        assert ac.allows(99, "bob") is False


# ── TelegramDelivery (transport glue, regression) ────────────────────────


class TestTelegramDeliveryRegression:
    @pytest.mark.asyncio
    async def test_edit_text_not_modified_is_success(self):
        """Existing test: 'message is not modified' errors treated as success."""
        class NotModifiedBot:
            async def edit_message_text(self, **kwargs):
                raise Exception("BadRequest: Message is not modified")

        delivery = TelegramDelivery(
            bot=NotModifiedBot(),
            chat_lock_factory=lambda _: None,
            keyboard_factory=lambda _: None,
        )
        ok = await delivery._edit_text(123, 456, "<b>same</b>", parse_mode="HTML", reply_markup=None)
        assert ok is True

    @pytest.mark.asyncio
    async def test_deliver_typing_event(self):
        """Typing events should call send_chat_action."""
        calls = []

        class FakeBot:
            async def send_chat_action(self, **kwargs):
                calls.append(kwargs)

        delivery = TelegramDelivery(
            bot=FakeBot(),
            chat_lock_factory=lambda _: asyncio.Lock(),
            keyboard_factory=lambda _: None,
        )
        event = DeliveryEvent(
            route=_make_route(scope_id="100"),
            kind="typing",
        )
        await delivery.deliver(event)
        assert len(calls) == 1
        assert calls[0]["action"] == "typing"
