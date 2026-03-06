"""
Telegram bot handlers for OmicVerse Jarvis.

Architecture (OpenClaw-inspired):
  • handle_analysis() immediately acknowledges and spawns a background asyncio.Task
  • The background task streams LLM tokens by editing a single "thinking" message
  • Progress (code execution) is sent as separate messages
  • On completion: result header → figures with captions → summary → inline keyboard
  • /cancel stops the running task gracefully
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Set

from .. import _fmt
from ..gateway.routing import GatewaySessionRegistry, SessionKey

logger = logging.getLogger("omicverse.jarvis")


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

class AccessControl:
    """Allow all users when *allowed* is empty; otherwise whitelist."""

    def __init__(self, allowed: Optional[List[str]] = None) -> None:
        self._ids: Set[int] = set()
        self._usernames: Set[str] = set()
        for entry in (allowed or []):
            entry = entry.strip()
            if entry.lstrip("-").isdigit():
                self._ids.add(int(entry))
            else:
                self._usernames.add(entry.lstrip("@").lower())

    @property
    def _open(self) -> bool:
        return not self._ids and not self._usernames

    def allows(self, user_id: int, username: Optional[str]) -> bool:
        if self._open:
            return True
        if user_id in self._ids:
            return True
        if username and username.lower() in self._usernames:
            return True
        return False


# ---------------------------------------------------------------------------
# Bot builder
# ---------------------------------------------------------------------------

def run_bot(
    token: str,
    session_manager: Any,
    access_control: AccessControl,
    verbose: bool = False,
) -> None:
    """Build and start the Telegram application (blocking)."""
    try:
        from telegram.ext import Application
    except ImportError as exc:
        raise ImportError(
            "python-telegram-bot is required.  "
            "Install with: pip install omicverse[jarvis]"
        ) from exc

    app = Application.builder().token(token).concurrent_updates(True).build()
    _register_handlers(app, session_manager, access_control, verbose)
    logger.info("OmicVerse Jarvis bot starting (polling)...")
    app.run_polling(drop_pending_updates=True)


def _register_handlers(app: Any, sm: Any, ac: AccessControl, verbose: bool) -> None:
    from telegram import (
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Update,
    )
    from telegram.ext import (
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    # Per-user background analysis tasks
    _tasks: Dict[int, asyncio.Task] = {}
    # Human-readable description of each user's running analysis (for concurrent chat)
    _task_requests: Dict[int, str] = {}
    # OpenClaw Collect mode: messages queued while analysis runs, coalesced after
    _pending: Dict[int, List[str]] = {}
    # Per-chat outbound lock (OpenClaw-style channel sequencing)
    _chat_locks: Dict[int, asyncio.Lock] = {}
    _route_registry = GatewaySessionRegistry(sm)

    # ------------------------------------------------------------------
    # Guard / session helpers
    # ------------------------------------------------------------------

    async def _guard(update: Update) -> bool:
        user = update.effective_user
        if user is None:
            return False
        if not ac.allows(user.id, user.username):
            await update.message.reply_text("⛔ 您没有访问权限。")
            return False
        return True

    async def _get_session(update: Update):
        try:
            chat = update.effective_chat
            msg = update.effective_message
            scope_type = "dm" if (chat and chat.type == "private") else "group"
            scope_id = str(chat.id if chat else update.effective_user.id)
            thread_id = None
            if msg is not None:
                thread_id = getattr(msg, "message_thread_id", None)
            sk = SessionKey(
                channel="telegram",
                scope_type=scope_type,
                scope_id=scope_id,
                thread_id=(str(thread_id) if thread_id else None),
            )
            return _route_registry.get_or_create(sk)
        except Exception as exc:
            logger.exception("Failed to create session")
            await update.message.reply_text(
                _fmt.error_message(
                    f"Agent 初始化失败：{exc}\n"
                    "请检查 --model 参数，运行 ov.list_supported_models() 查看可用模型。"
                ),
                parse_mode="HTML",
            )
            return None

    async def _cancel_user_task(user_id: int) -> bool:
        """Cancel running analysis task and flush pending queue. Returns True if a task existed."""
        _pending.pop(user_id, None)
        task = _tasks.get(user_id)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except Exception:
                pass
            return True
        return False

    def _analysis_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[
            InlineKeyboardButton("💾 保存",  callback_data="jarvis:save"),
            InlineKeyboardButton("📊 状态",  callback_data="jarvis:status"),
            InlineKeyboardButton("🧠 历史",  callback_data="jarvis:memory"),
        ]])

    def _chat_lock(chat_id: int) -> asyncio.Lock:
        lock = _chat_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            _chat_locks[chat_id] = lock
        return lock

    async def _quick_chat(
        session: Any,
        user_text: str,
        chat_id: int,
        bot: Any,
        running_request: str = "",
        queued: bool = False,
    ) -> None:
        """OpenClaw-style concurrent chat: typing indicator + fast LLM reply while analysis runs.

        queued=True hints the system prompt that this message has also been enqueued so the
        LLM can mention it to the user naturally.
        """
        # Typing indicator — instant UX feedback (OpenClaw pattern)
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            pass

        try:
            system_lines = [
                "You are OmicVerse Jarvis, a bioinformatics AI assistant.",
                "The user is chatting with you while a background analysis is running in the background.",
                "Answer concisely and helpfully. Do NOT execute code or call tools.",
                "Reply in the same language the user uses.",
            ]
            if running_request:
                system_lines.append(f"\nCurrently running analysis: {running_request[:300]}")
            if queued:
                system_lines.append(
                    "If the user's message looks like a new analysis request, "
                    "inform them it has been queued and will start automatically after the current analysis finishes."
                )
            if session.adata is not None:
                a = session.adata
                system_lines.append(f"Loaded data: {a.n_obs:,} cells × {a.n_vars:,} genes")
            memory_ctx = session.get_memory_context()
            if memory_ctx:
                system_lines.append(f"\nRecent analysis history:\n{memory_ctx[:600]}")

            messages = [
                {"role": "system", "content": "\n".join(system_lines)},
                {"role": "user",   "content": user_text},
            ]
            response = await session.agent._llm.chat(messages, tools=None, tool_choice=None)
            reply = (response.content or "").strip() or "💬  分析进行中，稍后再试。"
            async with _chat_lock(chat_id):
                await _safe_send_message(
                    bot, chat_id, _fmt.md_to_html(reply), parse_mode="HTML"
                )
        except Exception as exc:
            logger.warning("Quick chat failed: %s", exc)
            try:
                async with _chat_lock(chat_id):
                    await _safe_send_message(
                        bot, chat_id,
                        "⏳  后台分析进行中，请等待完成。使用 <code>/cancel</code> 取消。",
                        parse_mode="HTML",
                    )
            except Exception:
                pass

    async def _send_photo_or_file(bot: Any, chat_id: int, png_bytes: bytes, caption: str) -> None:
        try:
            await bot.send_photo(chat_id=chat_id, photo=BytesIO(png_bytes), caption=caption)
            return
        except Exception:
            pass
        # Fallback: send as document if Telegram photo pipeline rejects bytes.
        try:
            await bot.send_document(
                chat_id=chat_id,
                document=BytesIO(png_bytes),
                filename="figure.png",
                caption=caption,
            )
        except Exception as exc:
            logger.warning("Failed to send figure as photo/document: %s", exc)

    async def _send_artifact_document(
        bot: Any,
        chat_id: int,
        filename: str,
        data: bytes,
    ) -> None:
        try:
            await bot.send_document(
                chat_id=chat_id,
                document=BytesIO(data),
                filename=filename,
                caption=f"📎  {filename}",
            )
        except Exception as exc:
            logger.warning("Failed to send artifact %s: %s", filename, exc)

    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text or "")

    # File extensions harvested as artifacts (must stay in sync with agent_bridge.py).
    _ARTIFACT_EXTS = r"pdf|csv|tsv|txt|xlsx|html|json|h5ad|png|jpg|svg"

    def _strip_local_paths(text: str) -> str:
        """Remove local filesystem path references so Telegram doesn't show dead links.

        Files are always delivered as sendDocument instead of clickable local links.
        """
        t = text or ""
        # Backtick-wrapped paths with ≥2 directory levels
        t = re.sub(r'`[^`\n]*(?:/[^`\n]*){2,}`', '', t)
        # Absolute Unix paths starting with common root prefixes
        t = re.sub(r'/(?:Users|home|tmp|var|opt|root|data|mnt|private)/\S+', '', t)
        # ~/paths
        t = re.sub(r'~[/\\]\S+', '', t)
        # Relative paths ending with a known artifact extension:
        #   ./output/report.html  or  output/report.html  (with/without leading ./)
        _ext = _ARTIFACT_EXTS
        t = re.sub(
            rf'\.?/?(?:\w[\w/-]*/)+\w[\w.-]*\.(?:{_ext})',
            '', t, flags=re.IGNORECASE,
        )
        # Collapse whitespace artifacts left by removals
        t = re.sub(r'[ \t]{2,}', ' ', t)
        t = re.sub(r'\n{3,}', '\n\n', t)
        return t.strip()

    async def _safe_send_message(
        bot: Any,
        chat_id: int,
        text: str,
        *,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Any = None,
    ) -> None:
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            return
        except Exception:
            pass
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=_strip_html(text),
                reply_markup=reply_markup,
            )
        except Exception as exc:
            logger.warning("Failed to send message: %s", exc)

    async def _safe_edit_message(
        bot: Any,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Any = None,
    ) -> bool:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            return True
        except Exception:
            pass
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=_strip_html(text),
                reply_markup=reply_markup,
            )
            return True
        except Exception:
            return False

    async def _send_prose_locked(
        bot: Any,
        chat_id: int,
        raw_text: str,
        *,
        header: str = "",
        always_expand: bool = False,
    ) -> None:
        async with _chat_lock(chat_id):
            await _fmt.send_prose(
                bot,
                chat_id,
                raw_text,
                header=header,
                always_expand=always_expand,
            )

    # ------------------------------------------------------------------
    # /start
    # ------------------------------------------------------------------

    async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        text = (
            "👋  <b>OmicVerse Jarvis</b>\n"
            f"{_fmt._DIV}\n"
            "移动端单细胞分析助手，支持中英文自然语言指令。\n\n"
            "<b>快速开始</b>\n"
            "1. 通过 <code>scp</code>/<code>sftp</code> 上传 .h5ad 到 workspace\n"
            "2. <code>/workspace</code> 查看文件 → <code>/load</code> 加载\n"
            "3. 直接发送分析需求，例如：\n"
            "   <i>质控 nUMI&gt;500 mito&lt;0.2，然后标准化、UMAP</i>\n\n"
            "<b>数据命令</b>\n"
            "<code>/workspace</code>  查看 workspace\n"
            "<code>/ls</code>         列出文件\n"
            "<code>/find</code>       搜索文件\n"
            "<code>/load</code>       加载数据\n"
            "<code>/shell</code>      执行 shell 命令\n\n"
            "<b>会话命令</b>\n"
            "<code>/kernel</code>     当前 kernel 状态\n"
            "<code>/kernel ls</code>  列出所有 kernel\n"
            "<code>/kernel new x</code> 新建并切换 kernel\n"
            "<code>/kernel use x</code> 切换 kernel\n"
            "<code>/memory</code>     分析历史\n"
            "<code>/usage</code>      token 用量\n"
            "<code>/model</code>      切换模型\n"
            "<code>/status</code>     数据状态\n"
            "<code>/save</code>       下载 h5ad\n"
            "<code>/cancel</code>     取消当前分析\n"
            "<code>/reset</code>      重置会话\n"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ------------------------------------------------------------------
    # /help
    # ------------------------------------------------------------------

    async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        text = (
            "📚  <b>使用指南</b>\n"
            f"{_fmt._DIV}\n"
            "<b>分析示例</b>\n"
            "• <code>质控 nUMI&gt;500 mito&lt;0.2</code>\n"
            "• <code>标准化、高变基因、PCA、UMAP</code>\n"
            "• <code>Leiden 聚类 resolution=0.5</code>\n"
            "• <code>差异表达 cluster 1 vs 2</code>\n"
            "• <code>GO 富集分析 cluster 0</code>\n\n"
            "<b>数据管理</b>\n"
            "• <code>/workspace</code> — workspace 概览\n"
            "• <code>/ls [路径]</code> — 列出文件\n"
            "• <code>/find &lt;模式&gt;</code> — 搜索文件\n"
            "• <code>/load &lt;文件名&gt;</code> — 加载数据\n"
            "• <code>/shell &lt;命令&gt;</code> — 执行 shell"
            "（白名单：ls find cat head wc file du pwd tree）\n\n"
            "<b>会话管理</b>\n"
            "• <code>/kernel</code> — 当前 kernel 健康 + prompt 余量\n"
            "• <code>/kernel ls</code> — 列出可用 kernels\n"
            "• <code>/kernel new 名称</code> — 新建并切换 kernel\n"
            "• <code>/kernel use 名称</code> — 切换到指定 kernel\n"
            "• <code>/memory</code> — 近两天分析日志\n"
            "• <code>/usage</code> — 最近一次 token 用量\n"
            "• <code>/model [名称]</code> — 查看/切换 LLM 模型\n"
            "• <code>/status</code> — 当前数据信息\n"
            "• <code>/save</code> — 下载 .h5ad\n"
            "• <code>/cancel</code> — 取消正在运行的分析\n"
            "• <code>/reset</code> — 清空会话并重启 kernel\n\n"
            "<b>自定义指令</b>\n"
            "在 workspace 创建 <code>AGENTS.md</code>，写入偏好（语言、分析风格等），"
            "每次请求自动注入。"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ------------------------------------------------------------------
    # /status
    # ------------------------------------------------------------------

    async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        user    = update.effective_user
        session = await _get_session(update)
        if session is None:
            return

        lines = [f"👤  <b>用户</b>  <code>{user.id}</code>"]
        try:
            kname = sm.get_active_kernel(user.id)
            lines.append(f"🧩  Kernel：<code>{_fmt.esc(kname)}</code>")
        except Exception:
            pass
        if session.adata is not None:
            a = session.adata
            lines.append(f"🔬  {a.n_obs:,} cells × {a.n_vars:,} genes")
            if a.obs.columns.tolist():
                cols = ", ".join(a.obs.columns.tolist()[:8])
                lines.append(f"📋  obs: <code>{_fmt.esc(cols)}</code>")
        else:
            lines.append("📭  暂无数据  ·  使用 <code>/load</code> 加载")

        # Task status
        task = _tasks.get(user.id)
        if task and not task.done():
            lines.append("⚙️  分析中…  ·  <code>/cancel</code> 取消")

        try:
            info = session.agent.get_current_session_info()
            if info:
                p  = info.get("prompt_count", 0)
                mp = info.get("max_prompts", "?")
                if getattr(session, "max_prompts_setting", 0) <= 0:
                    mp = "∞"
                lines.append(f"💬  会话  {p}/{mp}")
        except Exception:
            pass

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ------------------------------------------------------------------
    # /reset
    # ------------------------------------------------------------------

    async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        user_id = update.effective_user.id
        await _cancel_user_task(user_id)
        session = await _get_session(update)
        if session is None:
            return
        session.reset()
        await update.message.reply_text(
            "✅  会话已重置，kernel 已重启。\n"
            "<i>变量已清空，adata 仍可通过 /load 重新加载。</i>",
            parse_mode="HTML",
        )

    # ------------------------------------------------------------------
    # /cancel
    # ------------------------------------------------------------------

    async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        user_id = update.effective_user.id
        cancelled = await _cancel_user_task(user_id)
        if cancelled:
            await update.message.reply_text("🚫  分析已取消。")
        else:
            await update.message.reply_text("ℹ️  当前没有正在运行的分析。")

    # ------------------------------------------------------------------
    # /save
    # ------------------------------------------------------------------

    async def handle_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        if session.adata is None:
            await update.message.reply_text("❌  没有数据，请先 <code>/load</code> 加载。", parse_mode="HTML")
            return

        await update.message.reply_text("⏳  正在保存…")
        path = session.save_adata()
        if path and path.exists():
            a = session.adata
            with open(str(path), "rb") as fh:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=fh,
                    filename="current.h5ad",
                    caption=f"💾  {a.n_obs:,} cells × {a.n_vars:,} genes",
                )
        else:
            await update.message.reply_text("❌  保存失败，请重试。")

    # ------------------------------------------------------------------
    # /usage
    # ------------------------------------------------------------------

    async def handle_usage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return

        usage = session.last_usage
        if usage is None:
            await update.message.reply_text("ℹ️  暂无用量数据，请先进行一次分析。")
            return

        def _attr(obj: Any, *names: str, default: str = "?") -> str:
            for name in names:
                v = getattr(obj, name, None)
                if v is not None:
                    return f"{v:,}" if isinstance(v, int) else str(v)
            return default

        lines = [
            "📊  <b>Token 用量</b>  （最近一次）",
            _fmt._DIV,
            f"输入：<code>{_attr(usage, 'input_tokens')}</code>",
            f"输出：<code>{_attr(usage, 'output_tokens')}</code>",
            f"合计：<code>{_attr(usage, 'total_tokens')}</code>",
        ]
        # cache_read / cache_creation if present (Anthropic prompt caching)
        cr = _attr(usage, "cache_read_input_tokens", default="")
        cc = _attr(usage, "cache_creation_input_tokens", default="")
        if cr and cr != "?":
            lines.append(f"缓存读取：<code>{cr}</code>")
        if cc and cc != "?":
            lines.append(f"缓存写入：<code>{cc}</code>")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ------------------------------------------------------------------
    # /model [name]
    # ------------------------------------------------------------------

    async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        args = context.args or []

        if not args:
            current = sm._model
            await update.message.reply_text(
                f"🤖  当前模型：<code>{_fmt.esc(current)}</code>\n"
                f"{_fmt._DIV}\n"
                f"切换示例：\n"
                f"• <code>/model claude-sonnet-4-6</code>\n"
                f"• <code>/model claude-opus-4-6</code>\n"
                f"• <code>/model claude-haiku-4-5-20251001</code>\n\n"
                f"<i>切换后请 /reset 重启 kernel 使新模型生效。</i>",
                parse_mode="HTML",
            )
            return

        new_model = args[0]
        sm._model = new_model
        await update.message.reply_text(
            f"✅  模型已切换为 <code>{_fmt.esc(new_model)}</code>\n"
            f"<i>请 /reset 重启 kernel 使新模型生效（当前 kernel 不受影响）。</i>",
            parse_mode="HTML",
        )

    # ------------------------------------------------------------------
    # /kernel
    # ------------------------------------------------------------------

    async def handle_kernel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        user_id = update.effective_user.id
        args = context.args or []

        # /kernel  -> status of active kernel
        if not args:
            session = await _get_session(update)
            if session is None:
                return
            st    = session.kernel_status()
            alive = st["alive"]
            p     = st["prompt_count"]
            mp    = st["max_prompts"]
            sid   = st["session_id"] or "—"
            kname = sm.get_active_kernel(user_id)

            icon      = "🟢" if alive else "🔴"
            remaining = (mp - p) if isinstance(mp, int) else "∞"

            lines = [
                "⚙️  <b>Kernel 状态</b>",
                _fmt._DIV,
                f"🧩  当前：<code>{_fmt.esc(kname)}</code>",
                f"{icon}  {'运行中' if alive else '未启动 / 已关闭'}",
                f"💬  Prompts：<code>{p}</code> / <code>{mp}</code>（剩余 {remaining}）",
                f"🆔  Session：<code>{_fmt.esc(str(sid))}</code>",
                "",
                "子命令：<code>/kernel ls</code> · <code>/kernel new 名称</code> · <code>/kernel use 名称</code>",
            ]
            if isinstance(mp, int) and p >= mp * 0.8:
                lines += [
                    "",
                    "⚠️  即将到达上限，下次分析后 kernel 将重启（变量清空）。",
                    "   可用 <code>/reset</code> 手动重启，或启动时增大 <code>--max-prompts</code>。",
                ]
            task = _tasks.get(user_id)
            if task and not task.done():
                lines += ["", "⚙️  当前有分析正在运行  ·  <code>/cancel</code> 取消"]
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            return

        sub = args[0].lower()
        if sub in {"ls", "list"}:
            active = sm.get_active_kernel(user_id)
            names = sm.list_kernels(user_id)
            lines = [
                "🧩  <b>Kernel 列表</b>",
                _fmt._DIV,
            ]
            for name in names:
                mark = "✅" if name == active else "•"
                lines.append(f"{mark}  <code>{_fmt.esc(name)}</code>")
            lines += [
                "",
                "切换：<code>/kernel use 名称</code>",
                "新建：<code>/kernel new 名称</code>",
            ]
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            return

        if sub in {"new", "create", "use", "switch"}:
            task = _tasks.get(user_id)
            if task and not task.done():
                await update.message.reply_text(
                    "⏳  当前有分析正在运行，请先等待完成或使用 <code>/cancel</code>。",
                    parse_mode="HTML",
                )
                return
            if len(args) < 2:
                await update.message.reply_text(
                    "用法：<code>/kernel new 名称</code> 或 <code>/kernel use 名称</code>\n"
                    "名称规则：字母/数字/._-，长度 1-32。",
                    parse_mode="HTML",
                )
                return
            name = args[1]
            try:
                if sub in {"new", "create"}:
                    sm.create_kernel(user_id, name, switch=True)
                    action = "新建并切换"
                else:
                    sm.switch_kernel(user_id, name, create=False)
                    action = "切换"
            except Exception as exc:
                await update.message.reply_text(
                    _fmt.error_message(str(exc)), parse_mode="HTML"
                )
                return
            await update.message.reply_text(
                f"✅  已{action}到 kernel：<code>{_fmt.esc(sm.get_active_kernel(user_id))}</code>",
                parse_mode="HTML",
            )
            return

        await update.message.reply_text(
            "用法：<code>/kernel</code> | <code>/kernel ls</code> | "
            "<code>/kernel new 名称</code> | <code>/kernel use 名称</code>",
            parse_mode="HTML",
        )

    # ------------------------------------------------------------------
    # /workspace
    # ------------------------------------------------------------------

    async def handle_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return

        from datetime import datetime
        ws         = session.workspace
        h5ad_files = session.list_h5ad_files()
        agents_md  = session.get_agents_md()
        today_log  = session.memory_dir / f"{datetime.now().date()}.md"

        lines = [
            "📁  <b>Workspace</b>",
            _fmt._DIV,
            f"<code>{ws}</code>",
            "",
        ]
        if h5ad_files:
            lines.append(f"📊  <b>数据文件</b>  ({len(h5ad_files)})")
            for f in h5ad_files[:10]:
                try:
                    mb = f.stat().st_size / 1_048_576
                    lines.append(f"  • <code>{_fmt.esc(f.name)}</code>  <i>{mb:.1f} MB</i>")
                except OSError:
                    lines.append(f"  • <code>{_fmt.esc(f.name)}</code>")
            if len(h5ad_files) > 10:
                lines.append(f"  <i>… 还有 {len(h5ad_files) - 10} 个</i>")
        else:
            lines.append("📊  <b>数据文件</b>  (空)")
            lines.append(f"  <i>scp *.h5ad user@host:{ws}</i>")

        lines += [
            "",
            f"📋  AGENTS.md  {'✅' if agents_md else '—'}",
            f"🧠  今日记忆  {'✅' if today_log.exists() else '—'}",
            "",
            "<code>/load &lt;文件名&gt;</code>  ·  <code>/ls</code>  ·  <code>/memory</code>",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ------------------------------------------------------------------
    # /ls [subpath]
    # ------------------------------------------------------------------

    async def handle_ls(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        args    = context.args or []
        subpath = args[0] if args else ""
        cmd     = f"ls -lh {subpath}".strip() if subpath else "ls -lh"
        out     = session.shell.exec(cmd, cwd=session.workspace)
        await _fmt.send_code(context.bot, update.effective_chat.id, out, header=f"$ {cmd}")

    # ------------------------------------------------------------------
    # /find <pattern>
    # ------------------------------------------------------------------

    async def handle_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        args = context.args or []
        if not args:
            await update.message.reply_text(
                "用法：<code>/find &lt;模式&gt;</code>，例如 <code>/find *.h5ad</code>",
                parse_mode="HTML",
            )
            return
        pattern = args[0]
        cmd     = f"find . -name {pattern}"
        out     = session.shell.exec(cmd, cwd=session.workspace)
        await _fmt.send_code(context.bot, update.effective_chat.id, out, header=f"$ {cmd}")

    # ------------------------------------------------------------------
    # /load <filename>
    # ------------------------------------------------------------------

    async def handle_load(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        args = context.args or []
        if not args:
            await update.message.reply_text(
                "用法：<code>/load &lt;文件名&gt;</code>，例如 <code>/load pbmc3k.h5ad</code>\n"
                "<code>/workspace</code> 查看可用文件。",
                parse_mode="HTML",
            )
            return

        filename = args[0]
        await update.message.reply_text(
            f"⏳  加载 <code>{_fmt.esc(filename)}</code>…", parse_mode="HTML"
        )
        try:
            adata = session.load_from_workspace(filename)
        except Exception as exc:
            logger.exception("Failed to load from workspace")
            await update.message.reply_text(_fmt.error_message(str(exc)), parse_mode="HTML")
            return

        if adata is None:
            h5ad_files = session.list_h5ad_files()
            hint = ""
            if h5ad_files:
                names = "  ".join(f.name for f in h5ad_files[:5])
                hint  = f"\n可用文件：<code>{_fmt.esc(names)}</code>"
            await update.message.reply_text(
                f"❌  未找到 <code>{_fmt.esc(filename)}</code>{hint}", parse_mode="HTML"
            )
            return

        await update.message.reply_text(
            f"✅  加载成功\n{_fmt._DIV}\n"
            f"🔬  <b>{adata.n_obs:,} cells × {adata.n_vars:,} genes</b>\n"
            f"📁  <code>{_fmt.esc(filename)}</code>",
            parse_mode="HTML",
        )

    # ------------------------------------------------------------------
    # /shell <cmd>
    # ------------------------------------------------------------------

    async def handle_shell(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        raw   = update.message.text or ""
        parts = raw.split(None, 1)
        if len(parts) < 2:
            await update.message.reply_text(
                "用法：<code>/shell &lt;命令&gt;</code>\n"
                "允许：<code>ls  find  cat  head  wc  file  du  pwd  tree</code>",
                parse_mode="HTML",
            )
            return
        cmd = parts[1].strip()
        out = session.shell.exec(cmd, cwd=session.workspace)
        await _fmt.send_code(context.bot, update.effective_chat.id, out, header=f"$ {cmd}")

    # ------------------------------------------------------------------
    # /memory
    # ------------------------------------------------------------------

    async def handle_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        text = session.get_recent_memory_text()
        await _send_prose_locked(
            context.bot,
            update.effective_chat.id,
            text,
            header="🧠  <b>分析历史</b>（近两天）",
            always_expand=True,
        )

    # ------------------------------------------------------------------
    # Document handler (.h5ad upload)
    # ------------------------------------------------------------------

    async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        doc = update.message.document
        if doc is None:
            return
        filename = doc.file_name or ""
        if not filename.endswith(".h5ad"):
            await update.message.reply_text("⚠️  请发送 <code>.h5ad</code> 格式的文件。", parse_mode="HTML")
            return

        await update.message.reply_text("⏳  正在下载并加载…")
        session = await _get_session(update)
        if session is None:
            return
        try:
            tg_file = await context.bot.get_file(doc.file_id)
            dest    = str(session.workspace_dir / "current.h5ad")
            await tg_file.download_to_drive(dest)
            import scanpy as sc
            session.adata = sc.read_h5ad(dest)
            a = session.adata
            await update.message.reply_text(
                f"✅  加载成功\n{_fmt._DIV}\n"
                f"🔬  <b>{a.n_obs:,} cells × {a.n_vars:,} genes</b>\n"
                f"📁  <code>{_fmt.esc(filename)}</code>",
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.exception("Failed to load h5ad")
            await update.message.reply_text(_fmt.error_message(str(exc)), parse_mode="HTML")

    # ------------------------------------------------------------------
    # Inline keyboard callback handler
    # ------------------------------------------------------------------

    async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()

        user    = query.from_user
        chat_id = query.message.chat_id
        if not ac.allows(user.id, user.username):
            return

        data = query.data or ""
        try:
            session = sm.get_or_create(user.id)
        except Exception:
            return

        if data == "jarvis:save":
            if session.adata is None:
                await context.bot.send_message(chat_id=chat_id, text="❌  没有数据。")
                return
            path = session.save_adata()
            if path and path.exists():
                a = session.adata
                with open(str(path), "rb") as fh:
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=fh,
                        filename="current.h5ad",
                        caption=f"💾  {a.n_obs:,} cells × {a.n_vars:,} genes",
                    )
            else:
                await context.bot.send_message(chat_id=chat_id, text="❌  保存失败。")

        elif data == "jarvis:status":
            if session.adata is not None:
                a = session.adata
                lines = [f"🔬  {a.n_obs:,} cells × {a.n_vars:,} genes"]
                if a.obs.columns.tolist():
                    cols = ", ".join(a.obs.columns.tolist()[:8])
                    lines.append(f"📋  obs: <code>{_fmt.esc(cols)}</code>")
                await context.bot.send_message(
                    chat_id=chat_id, text="\n".join(lines), parse_mode="HTML"
                )
            else:
                await context.bot.send_message(chat_id=chat_id, text="📭  暂无数据。")

        elif data == "jarvis:memory":
            text = session.get_recent_memory_text()
            await _send_prose_locked(
                context.bot,
                chat_id,
                text,
                header="🧠  <b>分析历史</b>",
                always_expand=True,
            )

    # ------------------------------------------------------------------
    # OpenClaw buffered-block dispatcher
    # ------------------------------------------------------------------

    async def _dispatch_final_blocks(
        bot: Any,
        chat_id: int,
        *,
        reports: List[str],
        figures: List[bytes],
        artifacts: List[Any],
        explain: str,
        final_text: str,
        keyboard: Any,
    ) -> None:
        """Deliver all reply blocks in sequence (OpenClaw lane-delivery pattern).

        Block order:  reports → figures (sendPhoto) → artifacts (sendDocument) → summary+keyboard

        Each block type goes through the correct Telegram API.
        Files are ALWAYS sent via sendDocument — never as local path links.
        """
        n_figs = len(figures)

        # Block 1 – Reports (long markdown/text blocks)
        for i, rep in enumerate(reports, start=1):
            header = "📝  <b>分析报告</b>" if i == 1 else f"📝  <b>分析报告（续 {i}）</b>"
            await _send_prose_locked(bot, chat_id, rep, header=header, always_expand=False)

        # Block 2 – Figures → sendPhoto (sendDocument fallback on error)
        for i, png_bytes in enumerate(figures, start=1):
            async with _chat_lock(chat_id):
                await _send_photo_or_file(
                    bot, chat_id, png_bytes, _fmt.figure_caption(i, n_figs)
                )

        # Block 3 – Artifacts → sendDocument (never local path links)
        for art in artifacts:
            async with _chat_lock(chat_id):
                await _send_artifact_document(bot, chat_id, art.filename, art.data)

        # Block 4 – Summary + keyboard (final text block)
        if explain:
            if len(explain) > 1200:
                await _send_prose_locked(bot, chat_id, explain)
                async with _chat_lock(chat_id):
                    await _safe_send_message(bot, chat_id, _fmt._DIV, reply_markup=keyboard)
            else:
                async with _chat_lock(chat_id):
                    await _safe_send_message(
                        bot, chat_id,
                        _fmt.md_to_html(explain),
                        parse_mode="HTML",
                        reply_markup=keyboard,
                    )
        else:
            async with _chat_lock(chat_id):
                await _safe_send_message(
                    bot, chat_id, final_text, parse_mode="HTML", reply_markup=keyboard
                )

    # ------------------------------------------------------------------
    # Background analysis runner  (OpenClaw sub-agent pattern)
    # ------------------------------------------------------------------

    async def _run_analysis_bg(
        session:      Any,
        user_text:    str,
        chat_id:      int,
        full_request: str,
        bot:          Any,
    ) -> None:
        """OpenClaw-style: draft stream first, then finalize via one outbound sequence."""
        stream_msg_id: Optional[int] = None
        last_edit = 0.0
        llm_buf = ""
        last_progress = ""
        EDIT_GAP = 1.5
        DRAFT_MAX = 2800

        def _trim_for_draft(text: str, max_len: int = DRAFT_MAX) -> str:
            """Keep draft readable without cutting from a random middle position."""
            if len(text) <= max_len:
                return text
            head = int(max_len * 0.55)
            tail = max_len - head - 40
            if tail < 200:
                tail = 200
            return (
                text[:head].rstrip()
                + "\n\n[...内容较长，已省略中间部分...]\n\n"
                + text[-tail:].lstrip()
            )

        def _draft_text() -> str:
            body = _fmt.md_to_html(_trim_for_draft(llm_buf)) if llm_buf.strip() else "<i>思考中…</i>"
            if last_progress:
                return f"🔄  <code>{_fmt.esc(last_progress[:180])}</code>\n\n💭  {body}"
            return f"💭  {body}"

        async def _edit_draft(html: str, force: bool = False) -> None:
            nonlocal last_edit
            if stream_msg_id is None:
                return
            now = time.monotonic()
            if (not force) and (now - last_edit < EDIT_GAP):
                return
            async with _chat_lock(chat_id):
                try:
                    ok = await _safe_edit_message(
                        bot,
                        chat_id,
                        stream_msg_id,
                        html,
                        parse_mode="HTML",
                    )
                    if ok:
                        last_edit = now
                except Exception:
                    last_edit = now
                    pass

        async def llm_chunk_cb(chunk: str) -> None:
            nonlocal llm_buf
            if not chunk:
                return
            llm_buf += chunk
            await _edit_draft(_draft_text(), force=False)

        async def progress_cb(msg: str) -> None:
            nonlocal last_progress
            last_progress = msg
            await _edit_draft(_draft_text(), force=True)

        # Draft placeholder
        async with _chat_lock(chat_id):
            try:
                stream_msg = await bot.send_message(
                    chat_id=chat_id,
                    text="💭  <b>思考中…</b>",
                    parse_mode="HTML",
                )
                stream_msg_id = stream_msg.message_id
            except Exception:
                stream_msg_id = None

        from ..agent_bridge import AgentBridge
        bridge = AgentBridge(session.agent, progress_cb, llm_chunk_cb)

        try:
            result = await bridge.run(full_request, session.adata)
        except asyncio.CancelledError:
            await _edit_draft("🚫  分析已取消。", force=True)
            raise

        # Persist state
        if result.adata is not None:
            session.adata = result.adata
            session.prompt_count += 1
            try:
                session.save_adata()
            except Exception:
                pass
        if result.usage is not None:
            session.last_usage = result.usage

        a_cur = result.adata or session.adata
        a_info = f"{a_cur.n_obs:,} cells × {a_cur.n_vars:,} genes" if a_cur else ""
        try:
            session.append_memory_log(
                request=user_text,
                summary=result.summary or "分析完成",
                adata_info=a_info,
            )
        except Exception:
            pass

        keyboard = _analysis_keyboard()

        # Error finalization
        if result.error:
            err_text = _fmt.error_message(result.error)
            edited = False
            if stream_msg_id is not None:
                async with _chat_lock(chat_id):
                    edited = await _safe_edit_message(
                        bot,
                        chat_id,
                        stream_msg_id,
                        err_text,
                        parse_mode="HTML",
                    )
            if not edited:
                async with _chat_lock(chat_id):
                    await _safe_send_message(
                        bot,
                        chat_id,
                        err_text,
                        parse_mode="HTML",
                    )
            return

        # Build final text payload
        # Strip local path references: OpenClaw pattern — files are sent via
        # sendDocument, not as clickable local links.
        _BORING = {"分析完成", "分析完成。", "task completed", "done", "完成"}
        summary = _strip_local_paths((result.summary or "").strip())
        has_summary = bool(summary and summary.lower() not in _BORING)
        long_summary = has_summary and len(summary) > 1200
        final_text = _fmt.md_to_html(summary) if has_summary and not long_summary else ""
        if (not final_text) and a_info:
            final_text = f"📊  <code>{_fmt.esc(a_info)}</code>"
        if not final_text:
            final_text = _fmt._DIV

        has_media    = bool(result.figures)
        has_reports  = bool(getattr(result, "reports", None))
        artifacts    = list(getattr(result, "artifacts", []) or [])
        has_artifacts = bool(artifacts)

        # OpenClaw lane delivery:
        #   Draft = intermediate streaming state only.
        #   Any complex reply (media / reports / artifacts / long text) is routed
        #   through _dispatch_final_blocks — the unified block dispatcher:
        #     reports → figures (sendPhoto) → artifacts (sendDocument) → summary+keyboard
        is_complex = has_media or has_reports or has_artifacts or long_summary
        if is_complex:
            if has_media:
                status = "正在发送图片…"
            elif has_reports:
                status = "正在发送报告…"
            else:
                status = "结果如下"
            await _edit_draft(f"✅  {status}", force=True)

            # Prefer agent summary; fall back to stripped LLM stream buffer.
            explain = summary if has_summary else _strip_local_paths(llm_buf.strip())
            await _dispatch_final_blocks(
                bot, chat_id,
                reports=list(result.reports) if has_reports else [],
                figures=list(result.figures) if has_media else [],
                artifacts=artifacts,
                explain=explain,
                final_text=final_text,
                keyboard=keyboard,
            )
            # OpenClaw: all final blocks delivered — clean up draft → tombstone.
            await _edit_draft("✅  分析完成", force=True)
            return

        # Text-only, no complex blocks: draft IS the streaming result.
        # Update draft to final LLM text; keyboard goes to a separate fresh message.
        if llm_buf.strip():
            final_html = _fmt.md_to_html(llm_buf.strip())
            if len(final_html) > 3200 or "<pre>" in final_html:
                # Avoid oversized editMessageText failures; send full prose as new blocks.
                await _edit_draft("✅  分析完成，正文如下。", force=True)
                await _send_prose_locked(bot, chat_id, llm_buf.strip(), always_expand=False)
            else:
                await _edit_draft(final_html, force=True)
        elif stream_msg_id is not None:
            await _edit_draft("✅  分析完成", force=True)
        async with _chat_lock(chat_id):
            await _safe_send_message(
                bot, chat_id, final_text, parse_mode="HTML", reply_markup=keyboard
            )

    # ------------------------------------------------------------------
    # Text analysis handler  — launches background task immediately
    # ------------------------------------------------------------------

    async def handle_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        user_id = update.effective_user.id
        session = await _get_session(update)
        if session is None:
            return

        # OpenClaw Collect mode: if analysis is running, queue message + respond conversationally
        existing = _tasks.get(user_id)
        if existing and not existing.done():
            _pending.setdefault(user_id, []).append(user_text)
            running_req = _task_requests.get(user_id, "")
            asyncio.create_task(
                _quick_chat(
                    session, user_text, update.effective_chat.id, context.bot,
                    running_request=running_req, queued=True,
                )
            )
            return

        chat_id = update.effective_chat.id
        bot = context.bot

        # Acknowledge immediately
        if session.adata is not None:
            a = session.adata
            await update.message.reply_text(
                _fmt.ack_message(user_text, adata_info=f"{a.n_obs:,} cells × {a.n_vars:,} genes"),
                parse_mode="HTML",
            )
        else:
            h5ad_files = session.list_h5ad_files()
            if h5ad_files:
                names = "\n".join(
                    f"  • <code>{_fmt.esc(f.name)}</code>" for f in h5ad_files[:5]
                )
                hint = (
                    f"💡  workspace 中检测到 {len(h5ad_files)} 个文件：\n"
                    f"{names}\n使用 <code>/load &lt;文件名&gt;</code> 加载"
                )
            else:
                hint = "💡  未检测到已加载数据，Agent 将自行加载数据"
            await update.message.reply_text(
                _fmt.ack_message(user_text, workspace_hint=hint),
                parse_mode="HTML",
            )

        await _spawn_analysis(session, user_text, chat_id, bot, user_id)

    async def _spawn_analysis(
        session: Any,
        user_text: str,
        chat_id: int,
        bot: Any,
        user_id: int,
    ) -> None:
        """Build context, spawn background analysis task, and drain pending queue when done."""
        # Build context: AGENTS.md + memory + request
        ctx_parts  = []
        agents_md  = session.get_agents_md()
        memory_ctx = session.get_memory_context()
        if agents_md:
            ctx_parts.append(f"[User instructions]\n{agents_md}")
        if memory_ctx:
            ctx_parts.append(f"[Analysis history]\n{memory_ctx}")
        full_request = (
            "\n\n".join(ctx_parts) + f"\n\n[Current request]\n{user_text}"
            if ctx_parts else user_text
        )

        async def _wrapper() -> None:
            try:
                await _run_analysis_bg(session, user_text, chat_id, full_request, bot)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.exception("Analysis task failed")
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=_fmt.error_message(str(exc)),
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
            finally:
                _tasks.pop(user_id, None)
                _task_requests.pop(user_id, None)
                # OpenClaw Collect: drain queued messages into a single followup run
                queued = _pending.pop(user_id, [])
                if queued:
                    coalesced = "\n\n".join(queued)
                    n = len(queued)
                    try:
                        await _safe_send_message(
                            bot, chat_id,
                            f"⏭  开始执行队列中的 {n} 条请求…",
                            parse_mode="HTML",
                        )
                    except Exception:
                        pass
                    asyncio.create_task(
                        _spawn_analysis(session, coalesced, chat_id, bot, user_id)
                    )

        task = asyncio.create_task(_wrapper())
        _tasks[user_id] = task
        _task_requests[user_id] = user_text

    # ------------------------------------------------------------------
    # Register all handlers
    # ------------------------------------------------------------------

    app.add_handler(CommandHandler("start",     handle_start))
    app.add_handler(CommandHandler("help",      handle_help))
    app.add_handler(CommandHandler("status",    handle_status))
    app.add_handler(CommandHandler("reset",     handle_reset))
    app.add_handler(CommandHandler("cancel",    handle_cancel))
    app.add_handler(CommandHandler("save",      handle_save))
    app.add_handler(CommandHandler("usage",     handle_usage))
    app.add_handler(CommandHandler("model",     handle_model))
    app.add_handler(CommandHandler("kernel",    handle_kernel))
    app.add_handler(CommandHandler("workspace", handle_workspace))
    app.add_handler(CommandHandler("ls",        handle_ls))
    app.add_handler(CommandHandler("find",      handle_find))
    app.add_handler(CommandHandler("load",      handle_load))
    app.add_handler(CommandHandler("shell",     handle_shell))
    app.add_handler(CommandHandler("memory",    handle_memory))
    app.add_handler(CallbackQueryHandler(handle_callback, pattern=r"^jarvis:"))
    app.add_handler(MessageHandler(filters.Document.ALL,             handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,  handle_analysis))
