"""
Discord channel for OmicVerse Jarvis.

Behavior mirrors OpenClaw's default Discord mode:
- direct messages are handled directly
- guild messages require @bot mention
- analysis replies are streamed back as text and file attachments
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

try:
    import discord as _discord
except ImportError:  # pragma: no cover - optional dependency
    _discord = None

from ..agent_bridge import AgentBridge
from ..gateway.routing import GatewaySessionRegistry, SessionKey
from .channel_core import (
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

logger = logging.getLogger("omicverse.jarvis.discord")
logger.setLevel(logging.INFO)

discord = _discord
_MAX_TEXT = 1900
_PROGRESS_GAP = 12.0
_BORING_SUMMARIES = {"分析完成", "分析完成。", "task completed", "done", "完成"}


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
        self._registry = GatewaySessionRegistry(session_manager)
        self._tasks: Dict[str, RunningTask] = {}
        self._pending: Dict[str, List[Dict[str, Any]]] = {}
        self._client = self._build_client()

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
            session_key = self._session_key(message)
            session = self._registry.get_or_create(session_key)
            await self._handle_h5ad_attachment(message, session, h5ad_list[0])
            return

        session_key = self._session_key(message)
        session = self._registry.get_or_create(session_key)
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

        route = session_key.as_key()
        cmd, tail = self._command_parts(text)
        image_note = build_workspace_note(
            session.workspace,
            inbound_images,
            header="[Attached Discord images saved in workspace]",
        )
        user_text = compose_multimodal_user_text(text, image_note)
        request_content = [item.request_block for item in inbound_images]

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
            session.reset()
            await self._send_text(message.channel, "✅ 已重置当前会话。", reply_to=message)
            return

        if cmd == "/cancel":
            await self._handle_cancel(message.channel, message, route)
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

        running = self._tasks.get(route)
        if running and not running.task.done():
            self._pending.setdefault(route, []).append(
                {
                    "text": user_text,
                    "request_content": request_content,
                }
            )
            await self._send_text(
                message.channel,
                "⏭ 已加入当前会话队列，当前分析完成后继续处理。",
                reply_to=message,
            )
            return

        await self._send_ack(message.channel, message, session, image_count=len(inbound_images))
        await self._spawn_analysis(
            message.channel,
            message,
            session_key,
            session,
            user_text,
            request_content=request_content,
        )

    async def _handle_status(self, channel, source_message, session: Any, route: str) -> None:
        parts: List[str] = []
        running = self._tasks.get(route)
        if running and not running.task.done():
            parts.append(f"当前状态: 运行中\n请求: {running.request[:300]}")
        else:
            parts.append("当前状态: 空闲")
        if session.adata is not None:
            adata = session.adata
            parts.append(f"当前数据: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        try:
            workspace = session.agent.workspace_dir
        except Exception:
            workspace = None
        if workspace:
            parts.append(f"工作区: {workspace}")
        await self._send_text(channel, "\n".join(parts), reply_to=source_message)

    async def _handle_cancel(self, channel, source_message, route: str) -> None:
        self._pending.pop(route, None)
        running = self._tasks.get(route)
        if not running or running.task.done():
            await self._send_text(channel, "当前没有正在运行的分析任务。", reply_to=source_message)
            return
        running.task.cancel()
        await self._send_text(channel, "🚫 已发送取消信号。", reply_to=source_message)

    async def _send_ack(self, channel, source_message, session: Any, *, image_count: int = 0) -> None:
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
        if image_count:
            text += f"\n检测到图片: {image_count} 张（已保存到 workspace/uploads/discord）"
        await self._send_text(channel, text, reply_to=source_message)

    async def _spawn_analysis(
        self,
        channel,
        source_message,
        session_key: SessionKey,
        session: Any,
        user_text: str,
        request_content: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        route = session_key.as_key()
        task = asyncio.create_task(
            self._analysis_wrapper(
                channel,
                source_message,
                session_key,
                session,
                user_text,
                request_content or [],
            )
        )
        self._tasks[route] = RunningTask(task=task, request=user_text, started_at=time.time())

    async def _analysis_wrapper(
        self,
        channel,
        source_message,
        session_key: SessionKey,
        session: Any,
        user_text: str,
        request_content: List[Dict[str, Any]],
    ) -> None:
        route = session_key.as_key()
        try:
            await self._run_analysis(
                channel,
                source_message,
                session_key,
                session,
                user_text,
                request_content,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Discord analysis wrapper failed")
            await self._send_text(channel, f"分析异常: {exc}")
        finally:
            self._tasks.pop(route, None)
            queued = self._pending.pop(route, [])
            if queued:
                await self._send_text(channel, f"开始执行队列中的 {len(queued)} 条请求...")
                next_session = self._registry.get_or_create(session_key)
                coalesced, request_content = self._coalesce_pending_requests(queued)
                await self._spawn_analysis(
                    channel,
                    None,
                    session_key,
                    next_session,
                    coalesced,
                    request_content=request_content,
                )

    async def _run_analysis(
        self,
        channel,
        source_message,
        session_key: SessionKey,
        session: Any,
        user_text: str,
        request_content: List[Dict[str, Any]],
    ) -> None:
        llm_buf = ""
        last_progress_sent = 0.0
        full_request = self._build_full_request(session, user_text)

        async def progress_cb(msg: str) -> None:
            nonlocal last_progress_sent
            now = asyncio.get_running_loop().time()
            if now - last_progress_sent < _PROGRESS_GAP:
                return
            last_progress_sent = now
            await self._send_text(channel, f"⚙️ {msg[:200]}")

        async def llm_chunk_cb(chunk: str) -> None:
            nonlocal llm_buf
            if chunk:
                llm_buf += chunk

        prior_history = get_prior_history(
            self._sm, "discord", session_key.scope_type, session_key.scope_id, session,
        )

        bridge = AgentBridge(session.agent, progress_cb=progress_cb, llm_chunk_cb=llm_chunk_cb)
        try:
            result = await bridge.run(
                full_request,
                session.adata,
                history=prior_history,
                request_content=request_content,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Discord analysis failed")
            await self._send_text(channel, f"分析失败: {exc}", reply_to=source_message)
            return

        delivery_figures, _adata_info = process_result_state(session, result, user_text)

        notify_turn_complete(
            self._sm,
            channel="discord",
            scope_type=session_key.scope_type,
            scope_id=session_key.scope_id,
            session=session,
            user_text=user_text,
            llm_text=llm_buf,
            adata=result.adata,
        )

        if result.error:
            await self._send_text(
                channel, format_analysis_error(result, llm_buf), reply_to=source_message,
            )
            return

        for report in list(result.reports or []):
            for chunk in text_chunks(report, limit=_MAX_TEXT):
                await self._send_text(channel, chunk)

        for index, figure in enumerate(delivery_figures, start=1):
            await self._send_file(
                channel,
                figure,
                filename=f"figure_{index}.png",
                caption=f"图 {index}",
                reply_to=source_message if index == 1 else None,
            )

        for artifact in list(result.artifacts or []):
            await self._send_file(
                channel,
                artifact.data,
                filename=artifact.filename or "artifact.bin",
                caption=f"附件: {artifact.filename}",
            )

        summary = strip_local_paths(bridge.pick_reply_text(result, llm_buf))
        if not summary:
            summary = default_summary(session)
        for chunk in text_chunks(summary, limit=_MAX_TEXT):
            await self._send_text(channel, chunk)

    def _build_full_request(self, session: Any, text: str) -> str:
        return build_full_request(session, text, channel_label="Discord")

    def _session_key(self, message) -> SessionKey:
        channel = message.channel
        if discord is not None and isinstance(channel, discord.Thread):
            return SessionKey(
                channel="discord",
                scope_type="thread",
                scope_id=str(channel.parent_id or channel.id),
                thread_id=str(channel.id),
            )
        if getattr(channel, "guild", None) is not None:
            return SessionKey(channel="discord", scope_type="channel", scope_id=str(channel.id))
        return SessionKey(channel="discord", scope_type="dm", scope_id=str(channel.id))

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

    @staticmethod
    def _command_parts(text: str) -> tuple[str, str]:
        tokens = text.split()
        cmd = tokens[0].lower() if tokens else ""
        tail = text.split(None, 1)[1].strip() if len(tokens) > 1 else ""
        return cmd, tail

    async def _send_text(self, channel, text: str, *, reply_to=None) -> None:
        first = True
        for chunk in text_chunks(text, limit=_MAX_TEXT):
            kwargs: Dict[str, Any] = {}
            if first and reply_to is not None:
                kwargs["reference"] = reply_to
                kwargs["mention_author"] = False
            await channel.send(chunk, **kwargs)
            first = False

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

    @staticmethod
    def _coalesce_pending_requests(items: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        parts: List[str] = []
        request_content: List[Dict[str, Any]] = []
        for item in items:
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
            request_content.extend(list(item.get("request_content") or []))
        return "\n\n".join(parts).strip(), request_content

    async def _send_file(
        self,
        channel,
        data: bytes,
        *,
        filename: str,
        caption: str = "",
        reply_to=None,
    ) -> None:
        kwargs: Dict[str, Any] = {
            "file": discord.File(fp=io.BytesIO(data), filename=filename),
        }
        if caption:
            kwargs["content"] = caption[:_MAX_TEXT]
        if reply_to is not None:
            kwargs["reference"] = reply_to
            kwargs["mention_author"] = False
        await channel.send(**kwargs)


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
