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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import discord as _discord
except ImportError:  # pragma: no cover - optional dependency
    _discord = None

from ..agent_bridge import AgentBridge
from ..gateway.routing import GatewaySessionRegistry, SessionKey
from ..model_help import render_model_help

logger = logging.getLogger("omicverse.jarvis.discord")
logger.setLevel(logging.INFO)

discord = _discord
_MAX_TEXT = 1900
_PROGRESS_GAP = 12.0
_BORING_SUMMARIES = {"分析完成", "分析完成。", "task completed", "done", "完成"}


@dataclass
class RunningTask:
    task: asyncio.Task
    request: str
    started_at: float


def _text_chunks(text: str, limit: int = _MAX_TEXT) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    buf = ""
    for para in text.split("\n\n"):
        cand = f"{buf}\n\n{para}".strip() if buf else para
        if len(cand) <= limit:
            buf = cand
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= limit:
            buf = para
            continue
        pos = 0
        while pos < len(para):
            chunks.append(para[pos : pos + limit])
            pos += limit
        buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _strip_local_paths(text: str) -> str:
    value = text or ""
    value = re.sub(r"`[^`\n]*(?:/[^`\n]*){2,}`", "", value)
    value = re.sub(r"/(?:Users|home|tmp|var|opt|root|data|mnt|private)/\S+", "", value)
    value = re.sub(r"~[/\\]\S+", "", value)
    ext = r"pdf|csv|tsv|txt|xlsx|html|json|h5ad|png|jpg|svg"
    value = re.sub(rf"\.?/?(?:\w[\w/-]*/)+\w[\w.-]*\.(?:{ext})", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


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
        self._pending: Dict[str, List[str]] = {}
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

        text = self._normalize_message_text(message)
        logger.info(
            "Discord message received: channel_type=%s author=%s guild=%s content_len=%s normalized_len=%s",
            type(getattr(message, "channel", None)).__name__,
            getattr(author, "id", None),
            getattr(getattr(message, "guild", None), "id", None),
            len((getattr(message, "content", None) or "")),
            len(text),
        )
        if not text:
            logger.warning(
                "Discord message ignored because normalized content is empty. "
                "If this was a guild message, mention the bot. "
                "If this was a DM, check Discord Message Content intent."
            )
            return

        session_key = self._session_key(message)
        session = self._registry.get_or_create(session_key)
        route = session_key.as_key()
        cmd, tail = self._command_parts(text)

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
                for chunk in _text_chunks(render_model_help(getattr(self._sm, "_model", "unknown"))):
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
            self._pending.setdefault(route, []).append(text)
            await self._send_text(
                message.channel,
                "⏭ 已加入当前会话队列，当前分析完成后继续处理。",
                reply_to=message,
            )
            return

        await self._send_ack(message.channel, message, session)
        await self._spawn_analysis(message.channel, message, session_key, session, text)

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

    async def _send_ack(self, channel, source_message, session: Any) -> None:
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
        await self._send_text(channel, text, reply_to=source_message)

    async def _spawn_analysis(
        self,
        channel,
        source_message,
        session_key: SessionKey,
        session: Any,
        user_text: str,
    ) -> None:
        route = session_key.as_key()
        task = asyncio.create_task(
            self._analysis_wrapper(channel, source_message, session_key, session, user_text)
        )
        self._tasks[route] = RunningTask(task=task, request=user_text, started_at=time.time())

    async def _analysis_wrapper(
        self,
        channel,
        source_message,
        session_key: SessionKey,
        session: Any,
        user_text: str,
    ) -> None:
        route = session_key.as_key()
        try:
            await self._run_analysis(channel, source_message, session_key, session, user_text)
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
                await self._spawn_analysis(channel, None, session_key, next_session, "\n\n".join(queued))

    async def _run_analysis(
        self,
        channel,
        source_message,
        session_key: SessionKey,
        session: Any,
        user_text: str,
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

        bridge = AgentBridge(session.agent, progress_cb=progress_cb, llm_chunk_cb=llm_chunk_cb)
        try:
            result = await bridge.run(full_request, session.adata)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Discord analysis failed")
            await self._send_text(channel, f"分析失败: {exc}", reply_to=source_message)
            return

        if result.adata is not None:
            session.adata = result.adata
            try:
                session.save_adata()
                session.prompt_count += 1
            except Exception:
                pass
        if result.usage is not None:
            session.last_usage = result.usage
        try:
            adata = session.adata
            adata_info = f"{adata.n_obs:,} cells x {adata.n_vars:,} genes" if adata is not None else ""
            session.append_memory_log(
                request=user_text,
                summary=(result.summary or "分析完成"),
                adata_info=adata_info,
            )
        except Exception:
            pass

        web_bridge = getattr(self._sm, "gateway_web_bridge", None)
        if web_bridge is not None:
            try:
                web_bridge.on_turn_complete_simple(
                    channel="discord",
                    scope_type=session_key.scope_type,
                    scope_id=session_key.scope_id,
                    user_text=user_text,
                    llm_text=llm_buf,
                    adata=result.adata,
                )
            except Exception:
                pass

        if result.error:
            err_text = f"分析出错: {result.error}"
            if result.diagnostics:
                hints = "\n".join(f"- {item}" for item in result.diagnostics[:4])
                err_text += f"\n\n诊断:\n{hints}"
            if llm_buf.strip():
                err_text += f"\n\n模型输出:\n{llm_buf[:1200]}"
            await self._send_text(channel, err_text, reply_to=source_message)
            return

        for report in list(result.reports or []):
            for chunk in _text_chunks(report):
                await self._send_text(channel, chunk)

        for index, figure in enumerate(list(result.figures or []), start=1):
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

        summary = _strip_local_paths((result.summary or "").strip())
        if not summary or summary.lower() in _BORING_SUMMARIES:
            if llm_buf.strip():
                summary = llm_buf[:1800]
            elif session.adata is not None:
                adata = session.adata
                summary = f"分析完成\n{adata.n_obs:,} cells x {adata.n_vars:,} genes"
            else:
                summary = "分析完成"
        for chunk in _text_chunks(summary):
            await self._send_text(channel, chunk)

    def _build_full_request(self, session: Any, text: str) -> str:
        ctx_parts: List[str] = []
        try:
            agents_md = session.get_agents_md()
            if agents_md:
                ctx_parts.append(f"[User instructions]\n{agents_md}")
        except Exception:
            pass
        try:
            memory_ctx = session.get_memory_context()
            if memory_ctx:
                ctx_parts.append(f"[Analysis history]\n{memory_ctx}")
        except Exception:
            pass
        return "\n\n".join(ctx_parts) + f"\n\n[Current request]\n{text}" if ctx_parts else text

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
        for chunk in _text_chunks(text):
            kwargs: Dict[str, Any] = {}
            if first and reply_to is not None:
                kwargs["reference"] = reply_to
                kwargs["mention_author"] = False
            await channel.send(chunk, **kwargs)
            first = False

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
