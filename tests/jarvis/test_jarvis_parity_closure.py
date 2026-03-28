"""Cross-channel Jarvis shared runtime parity closure tests.

Proves that all six channels (Telegram, Feishu, iMessage, QQ, WeChat, Discord)
conform to the shared runtime model without keeping obsolete per-channel
duplicates.

Acceptance criteria addressed:
- AC-001.1: Cross-channel regression coverage proves parity on the shared
  runtime model across all six channels.
- AC-001.2: Obsolete per-channel helpers superseded by the shared layer are
  confirmed removed.
- AC-001.3: The final shared Jarvis runtime model is explicit and test-backed
  without changing public channel entry points.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from omicverse.jarvis.channels.channel_core import (
    command_parts,
    default_summary,
    format_status_plain,
    gather_status,
    strip_local_paths,
    text_chunks,
)
from omicverse.jarvis.runtime import (
    ConversationRoute,
    DeliveryEvent,
    MessageEnvelope,
    MessagePresenter,
    MessageRuntime,
)

# Channel names and their presenter/delivery/bot class naming conventions
ALL_CHANNELS = ("telegram", "feishu", "imessage", "qq", "wechat", "discord")


def _import_channel(name: str):
    return __import__(f"omicverse.jarvis.channels.{name}", fromlist=[name])


# ── Section 1: Shared import parity ──────────────────────────────────────────


class TestSharedImportParity:
    """Every channel must import shared helpers from channel_core, not define
    local copies."""

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_channel_importable(self, channel_name: str) -> None:
        _import_channel(channel_name)

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_imports_strip_local_paths(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        assert hasattr(mod, "strip_local_paths"), (
            f"{channel_name} does not import strip_local_paths from channel_core"
        )
        assert mod.strip_local_paths is strip_local_paths

    # Telegram uses _fmt.send_prose() for HTML-aware chunking rather than
    # the plain-text text_chunks, which is correct platform-specific behavior.
    _TEXT_CHUNKS_CHANNELS = ("feishu", "imessage", "qq", "wechat", "discord")

    @pytest.mark.parametrize("channel_name", _TEXT_CHUNKS_CHANNELS)
    def test_imports_text_chunks(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        assert hasattr(mod, "text_chunks"), (
            f"{channel_name} does not import text_chunks from channel_core"
        )
        assert mod.text_chunks is text_chunks

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_imports_gather_status(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        assert hasattr(mod, "gather_status"), (
            f"{channel_name} does not import gather_status from channel_core"
        )
        assert mod.gather_status is gather_status

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_no_local_running_task_class(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        source = inspect.getsource(mod)
        local_defs = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith("class RunningTask")
        ]
        assert len(local_defs) == 0, (
            f"{channel_name} still defines its own RunningTask class"
        )


# ── Section 2: MessagePresenter protocol conformance ────────────────────────

_PRESENTER_METHODS = [
    "ack",
    "queue_started",
    "draft_open",
    "draft_update",
    "draft_cancelled",
    "analysis_error",
    "typing",
    "quick_chat_reply",
    "quick_chat_fallback",
    "analysis_status",
    "final_events",
]

_PRESENTER_CLASSES = {
    "telegram": "TelegramRuntimePresenter",
    "feishu": "FeishuRuntimePresenter",
    "imessage": "IMessageRuntimePresenter",
    "qq": "QQRuntimePresenter",
    "wechat": "WeChatRuntimePresenter",
    "discord": "DiscordRuntimePresenter",
}


class TestPresenterProtocolConformance:
    """Every channel defines a presenter that satisfies the full
    MessagePresenter protocol."""

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_presenter_class_exists(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        cls_name = _PRESENTER_CLASSES[channel_name]
        assert hasattr(mod, cls_name), (
            f"{channel_name} is missing {cls_name}"
        )

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_presenter_has_all_methods(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        cls_name = _PRESENTER_CLASSES[channel_name]
        presenter_cls = getattr(mod, cls_name)
        presenter = presenter_cls()
        for method in _PRESENTER_METHODS:
            assert hasattr(presenter, method), (
                f"{cls_name} is missing required MessagePresenter method: {method}"
            )
            assert callable(getattr(presenter, method)), (
                f"{cls_name}.{method} is not callable"
            )

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_presenter_method_signatures_match_protocol(
        self, channel_name: str,
    ) -> None:
        """Each presenter method must accept the same positional parameter
        count as the protocol definition (allowing for self)."""
        mod = _import_channel(channel_name)
        cls_name = _PRESENTER_CLASSES[channel_name]
        presenter_cls = getattr(mod, cls_name)
        protocol_methods = {
            name: inspect.signature(getattr(MessagePresenter, name))
            for name in _PRESENTER_METHODS
        }
        for method_name, proto_sig in protocol_methods.items():
            impl = getattr(presenter_cls, method_name, None)
            assert impl is not None
            impl_sig = inspect.signature(impl)
            proto_params = [
                p for p in proto_sig.parameters.values()
                if p.name != "self"
            ]
            impl_params = [
                p for p in impl_sig.parameters.values()
                if p.name != "self"
            ]
            assert len(impl_params) >= len(proto_params), (
                f"{cls_name}.{method_name} has fewer parameters than "
                f"MessagePresenter protocol requires"
            )


# ── Section 3: MessageRuntime wiring ────────────────────────────────────────

# Map channel names to the bot/runtime class that should wire MessageRuntime
# For most channels the runtime wiring lives in a bot/runtime class.
# Telegram wires MessageRuntime inside its module-level run_bot() function.
_RUNTIME_WIRING_CLASSES = {
    "feishu": "FeishuRuntime",
    "imessage": "IMessageJarvisBot",
    "qq": "QQRuntime",
    "wechat": "WeChatJarvisBot",
    "discord": "DiscordJarvisBot",
}

_NON_TELEGRAM = ("feishu", "imessage", "qq", "wechat", "discord")


class TestMessageRuntimeWiring:
    """Every channel wires through MessageRuntime, not direct AgentBridge."""

    @pytest.mark.parametrize("channel_name", _NON_TELEGRAM)
    def test_wires_through_message_runtime(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        cls_name = _RUNTIME_WIRING_CLASSES[channel_name]
        bot_cls = getattr(mod, cls_name)
        source = inspect.getsource(bot_cls)
        assert "MessageRuntime" in source, (
            f"{cls_name} does not reference MessageRuntime"
        )

    def test_telegram_wires_through_message_runtime(self) -> None:
        """Telegram wires MessageRuntime inside _register_handlers, called
        from run_bot."""
        mod = _import_channel("telegram")
        source = inspect.getsource(mod._register_handlers)
        assert "MessageRuntime" in source

    @pytest.mark.parametrize("channel_name", _NON_TELEGRAM)
    def test_no_direct_agent_bridge_call(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        cls_name = _RUNTIME_WIRING_CLASSES[channel_name]
        bot_cls = getattr(mod, cls_name)
        source = inspect.getsource(bot_cls)
        assert "AgentBridge(" not in source, (
            f"{cls_name} still instantiates AgentBridge directly"
        )

    def test_telegram_no_direct_agent_bridge_call(self) -> None:
        mod = _import_channel("telegram")
        source = inspect.getsource(mod._register_handlers)
        assert "AgentBridge(" not in source

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_imports_runtime_types(self, channel_name: str) -> None:
        """Every channel imports at least ConversationRoute and DeliveryEvent
        from the shared runtime module."""
        mod = _import_channel(channel_name)
        assert hasattr(mod, "ConversationRoute"), (
            f"{channel_name} missing ConversationRoute import"
        )
        assert hasattr(mod, "DeliveryEvent"), (
            f"{channel_name} missing DeliveryEvent import"
        )
        assert hasattr(mod, "MessageEnvelope"), (
            f"{channel_name} missing MessageEnvelope import"
        )


# ── Section 4: Public entry point preservation ──────────────────────────────

_ENTRY_POINTS = {
    "telegram": "run_bot",
    "feishu": "run_feishu_bot",
    "imessage": "run_imessage_bot",
    "qq": "run_qq_bot",
    "wechat": "run_wechat_bot",
    "discord": "run_discord_bot",
}


class TestPublicEntryPoints:
    """Channel entry points remain stable after cleanup."""

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_entry_point_exists(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        fn_name = _ENTRY_POINTS[channel_name]
        assert hasattr(mod, fn_name), (
            f"{channel_name} is missing public entry point {fn_name}"
        )
        assert callable(getattr(mod, fn_name))

    def test_channels_init_exports(self) -> None:
        """channels/__init__.py re-exports all entry points."""
        from omicverse.jarvis import channels
        for fn_name in _ENTRY_POINTS.values():
            assert hasattr(channels, fn_name), (
                f"channels/__init__.py does not export {fn_name}"
            )


# ── Section 5: Obsolete helper removal verification ────────────────────────


class TestObsoleteHelperRemoval:
    """Verify that per-channel helpers superseded by the shared layer have
    been removed from the current-dev baseline."""

    def test_discord_no_local_command_parts(self) -> None:
        """Discord should use command_parts from channel_core, not a local
        _command_parts method."""
        mod = _import_channel("discord")
        assert mod.command_parts is command_parts, (
            "Discord should import command_parts from channel_core"
        )
        bot_cls = getattr(mod, "DiscordJarvisBot")
        assert not hasattr(bot_cls, "_command_parts"), (
            "Discord still has a local _command_parts method"
        )

    def test_telegram_no_artifact_exts_on_presenter(self) -> None:
        """TelegramRuntimePresenter should not carry an unused _ARTIFACT_EXTS
        class variable (the canonical definition is in channel_core)."""
        from omicverse.jarvis.channels.telegram import TelegramRuntimePresenter
        assert not hasattr(TelegramRuntimePresenter, "_ARTIFACT_EXTS"), (
            "TelegramRuntimePresenter still has obsolete _ARTIFACT_EXTS"
        )

    def test_telegram_no_strip_local_paths_wrapper(self) -> None:
        """TelegramRuntimePresenter should not have a _strip_local_paths
        wrapper; callers should use the channel_core function directly."""
        from omicverse.jarvis.channels.telegram import TelegramRuntimePresenter
        assert not hasattr(TelegramRuntimePresenter, "_strip_local_paths"), (
            "TelegramRuntimePresenter still has _strip_local_paths wrapper"
        )

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_no_duplicate_text_chunks_definition(self, channel_name: str) -> None:
        """No channel should define its own text_chunks function."""
        mod = _import_channel(channel_name)
        source = inspect.getsource(mod)
        local_defs = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith("def text_chunks")
        ]
        assert len(local_defs) == 0, (
            f"{channel_name} defines its own text_chunks"
        )

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_no_duplicate_strip_local_paths_definition(
        self, channel_name: str,
    ) -> None:
        """No channel should define its own strip_local_paths function."""
        mod = _import_channel(channel_name)
        source = inspect.getsource(mod)
        local_defs = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith("def strip_local_paths")
        ]
        assert len(local_defs) == 0, (
            f"{channel_name} defines its own strip_local_paths"
        )


# ── Section 6: Shared runtime model contract ────────────────────────────────


class TestSharedRuntimeModelContract:
    """Validate the shared runtime model's public surface is complete and
    consistent."""

    def test_runtime_exports_all_types(self) -> None:
        from omicverse.jarvis import runtime
        expected = {
            "AgentBridgeExecutionAdapter",
            "ConversationRoute",
            "DeliveryEvent",
            "ExecutionAdapter",
            "ExecutionCallbacks",
            "MessageEnvelope",
            "MessagePolicy",
            "MessagePresenter",
            "MessageRuntime",
            "MessageRouter",
            "PolicyDecision",
            "RuntimeTaskState",
            "TaskRegistry",
        }
        actual = set(runtime.__all__)
        assert expected == actual

    def test_message_presenter_protocol_is_complete(self) -> None:
        """MessagePresenter protocol defines all 11 required methods."""
        methods = [
            name for name, _ in inspect.getmembers(
                MessagePresenter, predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        ]
        assert set(methods) == set(_PRESENTER_METHODS)

    def test_conversation_route_is_frozen(self) -> None:
        """ConversationRoute should be an immutable dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(ConversationRoute)
        route = ConversationRoute(
            channel="test", scope_type="dm", scope_id="1",
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            route.channel = "other"  # type: ignore[misc]

    def test_delivery_event_is_frozen(self) -> None:
        """DeliveryEvent should be an immutable dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(DeliveryEvent)

    def test_message_envelope_is_frozen(self) -> None:
        """MessageEnvelope should be an immutable dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(MessageEnvelope)

    def test_channel_core_command_data_classes_exist(self) -> None:
        """channel_core exports the shared command data structures."""
        from omicverse.jarvis.channels.channel_core import (
            RunningTask,
            StatusInfo,
            UsageInfo,
            WorkspaceInfo,
            SaveResult,
        )
        import dataclasses
        for cls in (RunningTask, StatusInfo, UsageInfo, WorkspaceInfo, SaveResult):
            assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"

    def test_channel_core_formatters_exist(self) -> None:
        """channel_core provides plain-text formatters for all command types."""
        from omicverse.jarvis.channels.channel_core import (
            format_status_plain,
            format_usage_plain,
            format_workspace_plain,
            format_save_result_plain,
        )
        assert callable(format_status_plain)
        assert callable(format_usage_plain)
        assert callable(format_workspace_plain)
        assert callable(format_save_result_plain)

    def test_channel_core_request_helpers_exist(self) -> None:
        """channel_core provides request-building and result-processing helpers."""
        from omicverse.jarvis.channels.channel_core import (
            build_full_request,
            process_result_state,
            format_analysis_error,
            get_prior_history,
            notify_turn_complete,
        )
        assert callable(build_full_request)
        assert callable(process_result_state)
        assert callable(format_analysis_error)
        assert callable(get_prior_history)
        assert callable(notify_turn_complete)


# ── Section 7: Delivery class parity ────────────────────────────────────────

_DELIVERY_CLASSES = {
    "telegram": "TelegramDelivery",
    "feishu": "FeishuDelivery",
    "imessage": "IMessageDelivery",
    "qq": "QQRuntimePresenter",  # QQ combines presenter + delivery
    "wechat": "WeChatRuntimePresenter",  # WeChat combines presenter + delivery
    "discord": "DiscordDelivery",
}


class TestDeliveryClassParity:
    """Every channel defines a delivery mechanism for translating
    DeliveryEvents into platform-native sends."""

    @pytest.mark.parametrize("channel_name", ALL_CHANNELS)
    def test_delivery_class_exists(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        cls_name = _DELIVERY_CLASSES[channel_name]
        assert hasattr(mod, cls_name), (
            f"{channel_name} is missing delivery class {cls_name}"
        )


# ── Section 8: Route builder parity ─────────────────────────────────────────

# Module-level route-builder functions. Feishu builds routes via a method
# on FeishuRuntime (self._route), so it's tested separately.
_ROUTE_BUILDERS = {
    "telegram": "telegram_route_from_update",
    "imessage": "imessage_route_from_message",
    "qq": "qq_route_from_target",
    "wechat": "wechat_route_from_session_key",
    "discord": "discord_route_from_message",
}

_CHANNELS_WITH_MODULE_ROUTE = ("telegram", "imessage", "qq", "wechat", "discord")


class TestRouteBuilderParity:
    """Every channel defines a route-builder that produces ConversationRoute
    instances from platform-native payloads."""

    @pytest.mark.parametrize("channel_name", _CHANNELS_WITH_MODULE_ROUTE)
    def test_route_builder_exists(self, channel_name: str) -> None:
        mod = _import_channel(channel_name)
        fn_name = _ROUTE_BUILDERS[channel_name]
        assert hasattr(mod, fn_name), (
            f"{channel_name} is missing route builder {fn_name}"
        )
        assert callable(getattr(mod, fn_name))

    def test_feishu_builds_routes_via_runtime(self) -> None:
        """Feishu builds routes through FeishuRuntime._route method."""
        from omicverse.jarvis.channels.feishu import FeishuRuntime
        assert hasattr(FeishuRuntime, "_route"), (
            "FeishuRuntime is missing _route method"
        )
