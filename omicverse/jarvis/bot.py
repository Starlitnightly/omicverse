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
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Set

from . import _fmt

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

    app = Application.builder().token(token).build()
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
            return sm.get_or_create(update.effective_user.id)
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
        """Cancel running analysis task for user. Returns True if one existed."""
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
            "<code>/kernel</code>     kernel 状态\n"
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
            "• <code>/kernel</code> — kernel 健康 + prompt 余量\n"
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
        session = await _get_session(update)
        if session is None:
            return

        st    = session.kernel_status()
        alive = st["alive"]
        p     = st["prompt_count"]
        mp    = st["max_prompts"]
        sid   = st["session_id"] or "—"

        icon      = "🟢" if alive else "🔴"
        remaining = (mp - p) if isinstance(mp, int) else "?"

        lines = [
            "⚙️  <b>Kernel 状态</b>",
            _fmt._DIV,
            f"{icon}  {'运行中' if alive else '未启动 / 已关闭'}",
            f"💬  Prompts：<code>{p}</code> / <code>{mp}</code>（剩余 {remaining}）",
            f"🆔  Session：<code>{_fmt.esc(str(sid))}</code>",
        ]
        if isinstance(mp, int) and p >= mp * 0.8:
            lines += [
                "",
                "⚠️  即将到达上限，下次分析后 kernel 将重启（变量清空）。",
                "   可用 <code>/reset</code> 手动重启，或启动时增大 <code>--max-prompts</code>。",
            ]

        # Running task indicator
        task = _tasks.get(update.effective_user.id)
        if task and not task.done():
            lines += ["", "⚙️  当前有分析正在运行  ·  <code>/cancel</code> 取消"]

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

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
        await _fmt.send_prose(
            context.bot, update.effective_chat.id, text,
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
            await _fmt.send_prose(
                context.bot, chat_id, text,
                header="🧠  <b>分析历史</b>",
                always_expand=True,
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
        """
        Runs entirely in background.  Streams LLM tokens by editing a single
        "thinking" placeholder message (OpenClaw edit-in-place pattern).
        """
        # 1. Send streaming placeholder
        try:
            stream_msg    = await bot.send_message(
                chat_id=chat_id,
                text="💭  <b>思考中…</b>",
                parse_mode="HTML",
            )
            stream_msg_id = stream_msg.message_id
        except Exception:
            stream_msg_id = None

        llm_buf   = ""
        last_edit = 0.0
        EDIT_GAP  = 1.5   # seconds between edits (Telegram rate limit ~20 edits/min)

        async def _edit_stream(html: str) -> None:
            """Edit the single placeholder message, rate-limited."""
            nonlocal last_edit
            if not stream_msg_id:
                return
            now = time.monotonic()
            if now - last_edit < EDIT_GAP:
                return
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=stream_msg_id,
                    text=html,
                    parse_mode="HTML",
                )
                last_edit = now
            except Exception:
                pass

        async def _force_edit(html: str) -> None:
            """Edit the placeholder immediately, ignoring the rate-limit timer."""
            nonlocal last_edit
            if not stream_msg_id:
                return
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=stream_msg_id,
                    text=html,
                    parse_mode="HTML",
                )
                last_edit = time.monotonic()
            except Exception:
                pass

        async def llm_chunk_cb(chunk: str) -> None:
            """Accumulate LLM reasoning text; rate-limited edit of placeholder."""
            nonlocal llm_buf
            llm_buf += chunk
            if llm_buf.strip():
                preview = _fmt.md_to_html(llm_buf[-2500:])
                await _edit_stream(f"💭  {preview}")

        async def progress_cb(msg: str) -> None:
            """Code-execution event — always updates immediately (discrete, not streamed).
            Resets the LLM buffer so the next llm_chunk starts fresh text.
            """
            nonlocal llm_buf
            llm_buf = ""
            snippet = _fmt.esc(msg[:200])
            await _force_edit(f"🔄  <code>{snippet}</code>")

        from .agent_bridge import AgentBridge
        bridge = AgentBridge(session.agent, progress_cb, llm_chunk_cb)

        try:
            result = await bridge.run(full_request, session.adata)
        except asyncio.CancelledError:
            if stream_msg_id:
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=stream_msg_id,
                        text="🚫  分析已取消。",
                    )
                except Exception:
                    pass
            raise   # propagate so the task wrapper can clean up

        # 2. Persist adata + usage + memory  (before touching the stream message)
        if result.adata is not None:
            session.adata = result.adata
            session.prompt_count += 1
            try:
                session.save_adata()
            except Exception:
                pass

        if result.usage is not None:
            session.last_usage = result.usage

        try:
            a_cur  = result.adata or session.adata
            a_info = f"{a_cur.n_obs:,} cells × {a_cur.n_vars:,} genes" if a_cur else ""
            session.append_memory_log(
                request=user_text,
                summary=result.summary or "分析完成",
                adata_info=a_info,
            )
        except Exception:
            pass

        # 3. Error — edit placeholder to show error (no delete)
        if result.error:
            err_text = _fmt.error_message(result.error)
            edited = False
            if stream_msg_id:
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id, message_id=stream_msg_id,
                        text=err_text, parse_mode="HTML",
                    )
                    edited = True
                except Exception:
                    pass
            if not edited:
                await bot.send_message(chat_id=chat_id, text=err_text, parse_mode="HTML")
            return

        # 4. Figures — send each as a photo.
        #    The streaming placeholder (with agent reasoning) stays untouched above.
        n_figs  = len(result.figures)
        a_final = result.adata or session.adata
        a_info  = f"{a_final.n_obs:,} cells × {a_final.n_vars:,} genes" if a_final else ""

        for i, png_bytes in enumerate(result.figures, start=1):
            try:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=BytesIO(png_bytes),
                    caption=_fmt.figure_caption(i, n_figs),
                )
            except Exception as exc:
                logger.warning("Failed to send figure %d: %s", i, exc)

        # 5. Keyboard + optional adata info line.
        #    Skip generic "分析完成" summaries — the LLM reasoning in the stream
        #    message above is already the real answer.  Only show the summary when
        #    it carries new information not already visible.
        keyboard = _analysis_keyboard()
        _BORING = {"分析完成", "分析完成。", "task completed", "done", "完成"}
        summary  = (result.summary or "").strip()
        is_boring = summary.lower() in _BORING or not summary

        if is_boring:
            # Just send the keyboard with adata info (no redundant "分析完成" text)
            footer = f"📊  <code>{_fmt.esc(a_info)}</code>" if a_info else _fmt._DIV
            await bot.send_message(
                chat_id=chat_id,
                text=footer,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
        elif len(summary) <= 600:
            body = _fmt.md_to_html(summary)
            await bot.send_message(
                chat_id=chat_id,
                text=body,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
        else:
            # Long summary → expandable blockquote, keyboard on a follow-up message
            await _fmt.send_prose(bot, chat_id, summary)
            await bot.send_message(
                chat_id=chat_id,
                text=f"{_fmt._DIV}",
                reply_markup=keyboard,
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

        # Reject if analysis already running
        existing = _tasks.get(user_id)
        if existing and not existing.done():
            await update.message.reply_text(
                "⏳  当前有分析正在运行。\n"
                "请等待完成，或使用 <code>/cancel</code> 取消。",
                parse_mode="HTML",
            )
            return

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

        chat_id = update.effective_chat.id

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

        # Spawn background task
        bot = context.bot

        async def _wrapper() -> None:
            try:
                await _run_analysis_bg(
                    session, user_text, chat_id, full_request, bot
                )
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

        task = asyncio.create_task(_wrapper())
        _tasks[user_id] = task

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
