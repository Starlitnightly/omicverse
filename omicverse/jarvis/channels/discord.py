"""
Discord channel for OmicVerse Jarvis.

Behavior mirrors OpenClaw's default Discord mode:
- direct messages are handled directly
- guild messages require @bot mention
- analysis replies are streamed back as text and file attachments

Migrated to shared MessageRuntime / MessagePresenter abstractions so that
task management, analysis orchestration, web-bridge mirroring, and result
persistence are handled by the common runtime layer.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

try:
    import discord as _discord
except ImportError:  # pragma: no cover - optional dependency
    _discord = None

from .channel_core import (
    command_parts,
    text_chunks,
    strip_local_paths,
    gather_status,
    format_status_plain,
)
from ..channel_media import (
    MAX_INBOUND_IMAGES,
    format_h5ad_load_result,
    is_image_attachment,
    load_h5ad_to_session,
    prepare_inbound_image,
)
from ..media_ingest import (
    PreparedImage,
    build_workspace_note,
    compose_multimodal_user_text,
)
from ..model_help import render_model_help
from ..runtime import (
    AgentBridgeExecutionAdapter,
    ConversationRoute,
    DeliveryEvent,
    MessageEnvelope,
    MessageRuntime,
    MessageRouter,
)

logger = logging.getLogger("omicverse.jarvis.discord")
logger.setLevel(logging.INFO)

discord = _discord
_MAX_TEXT = 1900
_BORING_SUMMARIES = {"分析完成", "分析完成。", "task completed", "done", "完成"}


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def discord_route_from_message(message: Any) -> ConversationRoute:
    """Build a :class:`ConversationRoute` from a ``discord.Message``."""
    channel = getattr(message, "channel", None)
    author = getattr(message, "author", None)
    sender_id = str(getattr(author, "id", "")) if author else None

    if _discord is not None and isinstance(channel, _discord.Thread):
        return ConversationRoute(
            channel="discord",
            scope_type="thread",
            scope_id=str(getattr(channel, "parent_id", None) or getattr(channel, "id", "")),
            thread_id=str(getattr(channel, "id", "")),
            sender_id=sender_id,
        )
    if getattr(channel, "guild", None) is not None:
        return ConversationRoute(
            channel="discord",
            scope_type="channel",
            scope_id=str(getattr(channel, "id", "")),
            sender_id=sender_id,
        )
    return ConversationRoute(
        channel="discord",
        scope_type="dm",
        scope_id=str(getattr(channel, "id", "")),
        sender_id=sender_id,
    )


def discord_runtime_envelope(
    message: Any,
    text: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[MessageEnvelope]:
    """Build a :class:`MessageEnvelope` from a Discord message."""
    author = getattr(message, "author", None)
    if author is None or not text.strip():
        return None
    route = discord_route_from_message(message)
    return MessageEnvelope(
        route=route,
        text=text.strip(),
        sender_id=str(getattr(author, "id", "")),
        sender_username=getattr(author, "name", None),
        message_id=str(getattr(message, "id", "")) or None,
        trigger="direct" if route.is_direct else "mention",
        explicit_trigger=True,
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Presenter  (implements ``MessagePresenter`` protocol)
# ---------------------------------------------------------------------------


class DiscordRuntimePresenter:
    """Produce :class:`DeliveryEvent` instances for Discord rendering."""

    _BORING = _BORING_SUMMARIES

    def ack(self, envelope: MessageEnvelope, session: Any) -> List[DeliveryEvent]:
        if session.adata is not None:
            adata = session.adata
            text = f"⏳ 已收到，开始分析。\n当前数据: {adata.n_obs:,} cells x {adata.n_vars:,} genes"
        else:
            try:
                h5ad_files = session.list_h5ad_files()
            except Exception:
                h5ad_files = []
            if h5ad_files:
                names = ", ".join(item.name for item in h5ad_files[:5])
                text = f"⏳ 已收到，开始分析。\n检测到工作区数据文件: {names}"
            else:
                text = "⏳ 已收到，开始分析。"
        image_count = len(list((envelope.metadata or {}).get("request_content") or []))
        if image_count:
            text += f"\n检测到图片: {image_count} 张（已保存到 workspace/uploads/discord）"
        return [
            DeliveryEvent(
                route=envelope.route,
                kind="text",
                text=text,
                metadata={"reply_to": True},
            )
        ]

    def queue_started(self, route: ConversationRoute, queued_count: int) -> List[DeliveryEvent]:
        return [
            DeliveryEvent(
                route=route,
                kind="text",
                text=f"开始执行队列中的 {queued_count} 条请求...",
            )
        ]

    def draft_open(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="open",
            target="analysis-draft",
            text="💭 分析中…",
        )

    def draft_update(self, route: ConversationRoute, llm_text: str, progress: str) -> DeliveryEvent:
        if progress:
            text = f"⚙️ {progress[:200]}"
        elif llm_text.strip():
            trimmed = llm_text.strip()[-200:]
            text = f"💭 {trimmed}"
        else:
            text = "💭 分析中…"
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text=text,
        )

    def draft_cancelled(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text="🚫 分析已取消。",
        )

    def analysis_error(self, route: ConversationRoute, error_text: str) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text=f"❌ {error_text}",
        )

    def typing(self, route: ConversationRoute) -> Optional[DeliveryEvent]:
        return DeliveryEvent(route=route, kind="typing")

    def quick_chat_reply(self, route: ConversationRoute, text: str) -> DeliveryEvent:
        return DeliveryEvent(route=route, kind="text", text=text)

    def quick_chat_fallback(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            text="⏳ 后台分析进行中，请等待完成。使用 /cancel 取消。",
        )

    def analysis_status(
        self,
        route: ConversationRoute,
        *,
        has_media: bool,
        has_reports: bool,
        has_artifacts: bool,
    ) -> Optional[DeliveryEvent]:
        if has_media:
            status = "✅ 正在发送图片…"
        elif has_reports:
            status = "✅ 正在发送报告…"
        elif has_artifacts:
            status = "✅ 结果如下"
        else:
            return None
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text=status,
        )

    def final_events(
        self,
        route: ConversationRoute,
        *,
        session: Any,
        user_text: str,
        llm_text: str,
        result: Any,
    ) -> List[DeliveryEvent]:
        del user_text
        events: List[DeliveryEvent] = []

        # Reports
        for report in list(result.reports or []):
            for chunk in text_chunks(report, limit=_MAX_TEXT):
                events.append(DeliveryEvent(route=route, kind="text", text=chunk))

        # Figures (already converted to delivery bytes by the runtime)
        for index, figure_data in enumerate(result.figures or [], start=1):
            data = figure_data if isinstance(figure_data, bytes) else b""
            events.append(
                DeliveryEvent(
                    route=route,
                    kind="photo",
                    binary=data,
                    filename=f"figure_{index}.png",
                    caption=f"图 {index}",
                    metadata={"reply_first": index == 1},
                )
            )

        # Artifacts
        for artifact in list(result.artifacts or []):
            events.append(
                DeliveryEvent(
                    route=route,
                    kind="document",
                    binary=getattr(artifact, "data", b""),
                    filename=getattr(artifact, "filename", None) or "artifact.bin",
                    caption=f"附件: {getattr(artifact, 'filename', 'artifact.bin')}",
                )
            )

        # Summary text
        summary = strip_local_paths((result.summary or "").strip())
        if not summary or summary.lower() in self._BORING:
            summary = strip_local_paths(llm_text.strip()) if llm_text.strip() else ""
        if not summary:
            if session.adata is not None:
                adata = session.adata
                summary = f"分析完成\n{adata.n_obs:,} cells x {adata.n_vars:,} genes"
            else:
                summary = "分析完成"
        for chunk in text_chunks(summary, limit=_MAX_TEXT):
            events.append(DeliveryEvent(route=route, kind="text", text=chunk))

        # Mark draft as complete
        events.append(
            DeliveryEvent(
                route=route,
                kind="text",
                mode="edit",
                target="analysis-draft",
                text="✅ 分析完成",
            )
        )

        return events


# ---------------------------------------------------------------------------
# Delivery  (Discord transport layer)
# ---------------------------------------------------------------------------


class DiscordDelivery:
    """Translate :class:`DeliveryEvent` instances into Discord API calls."""

    def __init__(self, *, client: Any) -> None:
        self._client = client
        self._channels: Dict[str, Any] = {}
        self._source_messages: Dict[str, Any] = {}
        self._targets: Dict[str, Any] = {}

    def register_channel(
        self,
        route: ConversationRoute,
        channel: Any,
        source_message: Any = None,
    ) -> None:
        """Cache the Discord channel (and optional source message) for a route."""
        key = route.route_key()
        self._channels[key] = channel
        if source_message is not None:
            self._source_messages[key] = source_message

    async def deliver(self, event: DeliveryEvent) -> None:
        key = event.route.route_key()
        channel = self._channels.get(key)
        if channel is None:
            channel_id = int(event.route.thread_id or event.route.scope_id)
            channel = self._client.get_channel(channel_id)
        if channel is None:
            logger.warning("Cannot resolve Discord channel for %s", key)
            return

        source = self._source_messages.get(key)

        if event.kind == "typing":
            try:
                await channel.typing()
            except Exception:
                pass
            return

        if event.kind == "photo":
            reply_to = source if event.metadata.get("reply_first") else None
            await self._send_file(
                channel,
                event.binary or b"",
                filename=event.filename or "figure.png",
                caption=event.caption,
                reply_to=reply_to,
            )
            return

        if event.kind == "document":
            await self._send_file(
                channel,
                event.binary or b"",
                filename=event.filename or "artifact.bin",
                caption=event.caption,
            )
            return

        if event.kind != "text":
            return

        reply_to = source if event.metadata.get("reply_to") else None

        if event.mode == "open":
            msg = await self._send_text_msg(channel, event.text, reply_to=reply_to)
            if msg is not None and event.target:
                self._targets[self._target_key(event)] = msg
            return

        if event.mode == "edit":
            target_msg = self._targets.get(self._target_key(event))
            if target_msg is not None:
                ok = await self._edit_message(target_msg, event.text)
                if ok:
                    return
            msg = await self._send_text_msg(channel, event.text, reply_to=reply_to)
            if msg is not None and event.target:
                self._targets[self._target_key(event)] = msg
            return

        # Default: send new message(s), chunked if necessary
        first = True
        for chunk in text_chunks(event.text, limit=_MAX_TEXT):
            kwargs: Dict[str, Any] = {}
            if first and reply_to is not None:
                kwargs["reference"] = reply_to
                kwargs["mention_author"] = False
            await channel.send(chunk, **kwargs)
            first = False

    @staticmethod
    def _target_key(event: DeliveryEvent) -> str:
        return f"{event.route.route_key()}::{event.target or ''}"

    @staticmethod
    async def _send_text_msg(channel: Any, text: str, *, reply_to: Any = None) -> Any:
        try:
            first = True
            sent = None
            for chunk in text_chunks(text, limit=_MAX_TEXT):
                kwargs: Dict[str, Any] = {}
                if first and reply_to is not None:
                    kwargs["reference"] = reply_to
                    kwargs["mention_author"] = False
                msg = await channel.send(chunk, **kwargs)
                if first:
                    sent = msg
                first = False
            return sent
        except Exception as exc:
            logger.warning("Failed to send Discord message: %s", exc)
            return None

    @staticmethod
    async def _edit_message(message: Any, text: str) -> bool:
        try:
            trimmed = text[:_MAX_TEXT] if len(text) > _MAX_TEXT else text
            await message.edit(content=trimmed)
            return True
        except Exception:
            return False

    @staticmethod
    async def _send_file(
        channel: Any,
        data: bytes,
        *,
        filename: str,
        caption: str = "",
        reply_to: Any = None,
    ) -> None:
        file_obj = (
            discord.File(fp=io.BytesIO(data), filename=filename)
            if discord is not None
            else io.BytesIO(data)
        )
        kwargs: Dict[str, Any] = {
            "file": file_obj,
        }
        if caption:
            kwargs["content"] = caption[:_MAX_TEXT]
        if reply_to is not None:
            kwargs["reference"] = reply_to
            kwargs["mention_author"] = False
        try:
            await channel.send(**kwargs)
        except Exception as exc:
            logger.warning("Failed to send Discord file %s: %s", filename, exc)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


class DiscordJarvisBot:
    def __init__(
        self,
        *,
        token: str,
        session_manager: Any,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._token = token
        self._sm = session_manager
        self._stop_event = stop_event or threading.Event()
        self._client = self._build_client()
        self._delivery = DiscordDelivery(client=self._client)
        self._runtime = MessageRuntime(
            router=MessageRouter(session_manager),
            presenter=DiscordRuntimePresenter(),
            execution_adapter=AgentBridgeExecutionAdapter(),
            deliver=self._delivery.deliver,
            web_bridge=getattr(session_manager, "gateway_web_bridge", None),
        )

    def _build_client(self):
        if discord is None:
            raise RuntimeError(
                "Discord support requires `discord.py`. "
                'Install it with `pip install -U "omicverse[jarvis]"` or `pip install discord.py`.'
            )

        intents = discord.Intents.default()
        intents.guilds = True
        intents.guild_messages = True
        intents.dm_messages = True
        intents.message_content = True
        owner = self

        class _Client(discord.Client):
            async def on_connect(self_inner) -> None:
                logger.info("Discord websocket connected")

            async def on_ready(self_inner) -> None:
                user = self_inner.user
                logger.info(
                    "Discord bot connected as %s (%s)",
                    getattr(user, "name", "unknown"),
                    getattr(user, "id", "unknown"),
                )

            async def on_message(self_inner, message) -> None:
                try:
                    await owner._on_message(message)
                except Exception:
                    logger.exception("Discord on_message handler failed")

            async def on_disconnect(self_inner) -> None:
                logger.warning("Discord gateway disconnected")

            async def on_resumed(self_inner) -> None:
                logger.info("Discord gateway session resumed")

            async def on_error(self_inner, event_method, *args, **kwargs) -> None:
                logger.exception("Discord client event error in %s", event_method)

        return _Client(intents=intents)

    async def run(self) -> None:
        watcher = asyncio.create_task(self._watch_stop())
        try:
            logger.info("Starting Discord client")
            await self._client.start(self._token)
            logger.warning("Discord client.start() exited without reconnecting")
        except Exception:
            logger.exception("Discord client exited with error")
            raise
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    async def _watch_stop(self) -> None:
        await asyncio.to_thread(self._stop_event.wait)
        if not self._client.is_closed():
            await self._client.close()

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _on_message(self, message) -> None:
        author = getattr(message, "author", None)
        if author is None or getattr(author, "bot", False):
            return

        # Handle .h5ad file attachments before any text normalisation so that
        # a file-only message (empty content) is not silently dropped.
        attachments = list(getattr(message, "attachments", None) or [])
        h5ad_list = [
            a for a in attachments
            if (getattr(a, "filename", "") or "").lower().endswith(".h5ad")
        ]
        if h5ad_list:
            route = discord_route_from_message(message)
            session = self._runtime.get_session(route)
            await self._handle_h5ad_attachment(message, session, h5ad_list[0])
            return

        route = discord_route_from_message(message)
        session = self._runtime.get_session(route)
        inbound_images = await self._prepare_inbound_images(message, session)
        text = self._normalize_message_text(message)
        logger.info(
            "Discord message received: channel_type=%s author=%s guild=%s content_len=%s normalized_len=%s",
            type(getattr(message, "channel", None)).__name__,
            getattr(author, "id", None),
            getattr(getattr(message, "guild", None), "id", None),
            len((getattr(message, "content", None) or "")),
            len(text),
        )
        if not text and not inbound_images:
            logger.warning(
                "Discord message ignored because normalized content is empty. "
                "If this was a guild message, mention the bot. "
                "If this was a DM, check Discord Message Content intent."
            )
            return

        cmd, tail = command_parts(text)
        image_note = build_workspace_note(
            session.workspace,
            inbound_images,
            header="[Attached Discord images saved in workspace]",
        )
        user_text = compose_multimodal_user_text(text, image_note)
        request_content = [item.request_block for item in inbound_images]

        # ── Command routing ──────────────────────────────────────────
        if cmd == "/help":
            await self._send_text(
                message.channel,
                "OmicVerse Jarvis (Discord)\n"
                "/help 查看帮助\n"
                "/status 查看当前状态\n"
                "/reset 重置当前会话\n"
                "/model [名称] 查看或切换模型\n"
                "/cancel 取消当前分析",
                reply_to=message,
            )
            return

        if cmd == "/status":
            await self._handle_status(message.channel, message, session, route)
            return

        if cmd == "/reset":
            await self._runtime.cancel(route)
            session.reset()
            await self._send_text(message.channel, "✅ 已重置当前会话。", reply_to=message)
            return

        if cmd == "/cancel":
            cancelled = await self._runtime.cancel(route)
            if cancelled:
                await self._send_text(message.channel, "🚫 已发送取消信号。", reply_to=message)
            else:
                await self._send_text(message.channel, "当前没有正在运行的分析任务。", reply_to=message)
            return

        if cmd == "/model":
            if not tail:
                for chunk in text_chunks(render_model_help(getattr(self._sm, "_model", "unknown")), limit=_MAX_TEXT):
                    await self._send_text(message.channel, chunk, reply_to=message)
                return
            self._sm._model = tail
            await self._send_text(
                message.channel,
                f"✅ 模型已切换为 {tail}\n请 /reset 重启 kernel 使新模型生效。",
                reply_to=message,
            )
            return

        if cmd.startswith("/"):
            await self._send_text(message.channel, f"未知命令: {cmd}\n发送 /help 查看帮助。", reply_to=message)
            return

        # ── Analysis: route through the shared runtime ───────────────
        self._delivery.register_channel(route, message.channel, source_message=message)
        envelope = discord_runtime_envelope(
            message,
            user_text,
            metadata={"request_content": request_content} if request_content else {},
        )
        if envelope is None:
            return
        try:
            await self._runtime.handle_message(envelope)
        except Exception as exc:
            logger.exception("Failed to handle Discord analysis message")
            await self._send_text(message.channel, f"分析异常: {exc}")

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------

    async def _handle_status(self, channel, source_message, session: Any, route: ConversationRoute) -> None:
        state = self._runtime.task_state(route)
        info = gather_status(
            session,
            is_running=state.running,
            running_request=state.request[:300] if state.running else "",
        )
        await self._send_text(channel, format_status_plain(info), reply_to=source_message)

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _normalize_message_text(self, message) -> str:
        raw = (getattr(message, "content", None) or "").strip()
        if not raw:
            return ""
        if getattr(message.channel, "guild", None) is None:
            logger.info("Discord DM accepted without mention requirement")
            return raw
        user = self._client.user
        if user is None or not user.mentioned_in(message) or message.mention_everyone:
            return ""
        pattern = rf"<@!?{int(user.id)}>\s*"
        logger.info("Discord guild mention accepted for processing")
        return re.sub(pattern, "", raw).strip()


    async def _send_text(self, channel, text: str, *, reply_to=None) -> None:
        first = True
        for chunk in text_chunks(text, limit=_MAX_TEXT):
            kwargs: Dict[str, Any] = {}
            if first and reply_to is not None:
                kwargs["reference"] = reply_to
                kwargs["mention_author"] = False
            await channel.send(chunk, **kwargs)
            first = False

    # ------------------------------------------------------------------
    # Attachment helpers
    # ------------------------------------------------------------------

    async def _handle_h5ad_attachment(self, message, session: Any, attachment) -> None:
        """Download a .h5ad attachment sent by the user and load it into the session."""
        filename = getattr(attachment, "filename", None) or "upload.h5ad"
        await self._send_text(message.channel, "⏳ 正在下载并加载…", reply_to=message)
        try:
            data = await attachment.read()
            loaded = await asyncio.to_thread(load_h5ad_to_session, session, data, filename)
            await self._send_text(
                message.channel,
                format_h5ad_load_result(loaded, filename),
                reply_to=message,
            )
        except Exception as exc:
            logger.exception("Discord failed to load h5ad attachment")
            await self._send_text(message.channel, f"❌ 文件处理失败: {exc}", reply_to=message)

    async def _prepare_inbound_images(self, message, session: Any) -> List[PreparedImage]:
        attachments = list(getattr(message, "attachments", None) or [])
        if not attachments:
            return []
        prepared: List[PreparedImage] = []
        for attachment in attachments[:MAX_INBOUND_IMAGES]:
            filename = getattr(attachment, "filename", None) or ""
            content_type = str(getattr(attachment, "content_type", None) or "").strip().lower()
            if filename.lower().endswith(".h5ad"):
                continue
            if not is_image_attachment(filename, content_type):
                continue
            try:
                data = await attachment.read()
                prepared.append(
                    prepare_inbound_image(
                        data,
                        workspace_root=session.workspace,
                        channel_name="discord",
                        filename=filename,
                        mime_type=content_type,
                    )
                )
            except Exception:
                logger.warning(
                    "Discord inbound image preparation failed filename=%s",
                    filename or "unknown",
                    exc_info=True,
                )
        return prepared


def run_discord_bot(
    *,
    token: str,
    session_manager: Any,
    stop_event: Optional[threading.Event] = None,
) -> None:
    bot = DiscordJarvisBot(
        token=token,
        session_manager=session_manager,
        stop_event=stop_event,
    )
    asyncio.run(bot.run())
