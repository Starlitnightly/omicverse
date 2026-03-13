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

import atexit
import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import socket
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .. import _fmt
from ..channel_language import response_language_instruction, tr
from ..channel_shared import render_help_text, render_start_text
from ..config import default_state_dir
from ..gateway.routing import GatewaySessionRegistry, SessionKey
from ..model_help import render_model_help

logger = logging.getLogger("omicverse.jarvis")

_POLLING_RESTART_DELAY_SECONDS = 1.0
_POLLING_MAX_ATTEMPTS = 2


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

def telegram_route_from_update(update: Any) -> ConversationRoute:
    chat = getattr(update, "effective_chat", None)
    msg = getattr(update, "effective_message", None)
    user = getattr(update, "effective_user", None)
    scope_type = "dm" if (chat and chat.type == "private") else "group"
    scope_id = str(chat.id if chat else getattr(user, "id", ""))
    thread_id = getattr(msg, "message_thread_id", None) if msg is not None else None
    return ConversationRoute(
        channel="telegram",
        scope_type=scope_type,
        scope_id=scope_id,
        thread_id=(str(thread_id) if thread_id else None),
        sender_id=(str(user.id) if user else None),
    )


def _normalize_bot_username(username: Optional[str]) -> str:
    return (username or "").lstrip("@").strip().lower()


def _message_mentions_bot(message: Any, bot_username: Optional[str]) -> bool:
    name = _normalize_bot_username(bot_username)
    text = (getattr(message, "text", None) or "").strip()
    if not name or not text:
        return False

    mention_token = f"@{name}"
    if mention_token in text.lower():
        return True

    entities = list(getattr(message, "entities", None) or [])
    for entity in entities:
        entity_type = str(getattr(entity, "type", "")).lower()
        if "mention" not in entity_type:
            continue
        offset = int(getattr(entity, "offset", 0) or 0)
        length = int(getattr(entity, "length", 0) or 0)
        token = text[offset : offset + length].strip().lower()
        if token == mention_token:
            return True
    return False


def _message_replies_to_bot(message: Any, bot_username: Optional[str]) -> bool:
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return False
    from_user = getattr(reply, "from_user", None)
    if from_user is None:
        return False
    if getattr(from_user, "is_bot", False):
        name = _normalize_bot_username(bot_username)
        reply_name = _normalize_bot_username(getattr(from_user, "username", None))
        return not name or name == reply_name
    return False


def telegram_trigger_for_update(update: Any, bot_username: Optional[str]) -> Optional[str]:
    route = telegram_route_from_update(update)
    if route.is_direct:
        return "direct"

    message = getattr(update, "effective_message", None)
    text = (getattr(message, "text", None) or "").strip()
    if not text:
        return None
    if _message_mentions_bot(message, bot_username):
        return "mention"
    if _message_replies_to_bot(message, bot_username):
        return "reply"
    return None


def _strip_leading_bot_mention(text: str, bot_username: Optional[str]) -> str:
    name = _normalize_bot_username(bot_username)
    if not name:
        return text.strip()
    pattern = rf"^\s*@{re.escape(name)}[\s,:-]*"
    return re.sub(pattern, "", text.strip(), count=1, flags=re.IGNORECASE)


def telegram_runtime_envelope(update: Any, bot_username: Optional[str]) -> Optional[MessageEnvelope]:
    message = getattr(update, "effective_message", None)
    user = getattr(update, "effective_user", None)
    text = (getattr(message, "text", None) or "").strip()
    if not text or user is None:
        return None

    route = telegram_route_from_update(update)
    trigger = telegram_trigger_for_update(update, bot_username) or "implicit"
    normalized = _strip_leading_bot_mention(text, bot_username) if trigger == "mention" else text
    normalized = normalized.strip()
    if not normalized:
        return None

    return MessageEnvelope(
        route=route,
        text=normalized,
        sender_id=str(user.id),
        sender_username=getattr(user, "username", None),
        message_id=str(getattr(message, "message_id", "")) or None,
        trigger=trigger,
        explicit_trigger=(trigger != "implicit"),
    )


class TelegramRuntimePresenter:
    _BORING = {"分析完成", "分析完成。", "task completed", "done", "完成"}
    _ARTIFACT_EXTS = r"pdf|csv|tsv|txt|xlsx|html|json|h5ad|png|jpg|svg"
    _DRAFT_MAX = 2800

    def ack(self, envelope: MessageEnvelope, session: Any) -> List[DeliveryEvent]:
        if session.adata is not None:
            a = session.adata
            text = _fmt.ack_message(
                envelope.text,
                adata_info=f"{a.n_obs:,} cells × {a.n_vars:,} genes",
            )
        else:
            h5ad_files = session.list_h5ad_files()
            if h5ad_files:
                names = "\n".join(
                    f"  • <code>{_fmt.esc(f.name)}</code>" for f in h5ad_files[:5]
                )
                hint = tr(
                    envelope.text,
                    en=(
                        f"💡  Detected {len(h5ad_files)} file(s) in the workspace:\n"
                        f"{names}\nUse <code>/load &lt;filename&gt;</code> to load one."
                    ),
                    zh=(
                        f"💡  workspace 中检测到 {len(h5ad_files)} 个文件：\n"
                        f"{names}\n使用 <code>/load &lt;文件名&gt;</code> 加载"
                    ),
                )
            else:
                hint = tr(
                    envelope.text,
                    en="💡  No loaded dataset detected. The agent will load data if needed.",
                    zh="💡  未检测到已加载数据，Agent 将自行加载数据",
                )
            text = _fmt.ack_message(envelope.text, workspace_hint=hint)
        return [
            DeliveryEvent(
                route=envelope.route,
                kind="text",
                text=text,
                text_format="html",
            )
        ]

    def queue_started(self, route: ConversationRoute, queued_count: int) -> List[DeliveryEvent]:
        return [
            DeliveryEvent(
                route=route,
                kind="text",
                text=f"⏭  Starting {queued_count} queued request(s)...",
                text_format="html",
            )
        ]

    def draft_open(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="open",
            target="analysis-draft",
            text="💭  <b>Thinking...</b>",
            text_format="html",
        )

    def draft_update(self, route: ConversationRoute, llm_text: str, progress: str) -> DeliveryEvent:
        body = _fmt.md_to_html(self._trim_for_draft(llm_text)) if llm_text.strip() else "<i>Thinking...</i>"
        if progress:
            text = f"🔄  <code>{_fmt.esc(progress[:180])}</code>\n\n💭  {body}"
        else:
            text = f"💭  {body}"
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text=text,
            text_format="html",
        )

    def draft_cancelled(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text="🚫  Analysis cancelled.",
            text_format="html",
        )

    def analysis_error(self, route: ConversationRoute, error_text: str) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text=_fmt.error_message(error_text),
            text_format="html",
        )

    def typing(self, route: ConversationRoute) -> Optional[DeliveryEvent]:
        return DeliveryEvent(route=route, kind="typing")

    def quick_chat_reply(self, route: ConversationRoute, text: str) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            text=_fmt.md_to_html(text),
            text_format="html",
        )

    def quick_chat_fallback(self, route: ConversationRoute, *, user_text: str = "") -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            text=tr(
                user_text,
                en="⏳ Background analysis is still running. Please wait or use <code>/cancel</code>.",
                zh="⏳ 后台分析进行中，请等待完成。使用 <code>/cancel</code> 取消。",
            ),
            text_format="html",
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
            status = "✅  Sending images..."
        elif has_reports:
            status = "✅  Sending reports..."
        elif has_artifacts:
            status = "✅  Sending results..."
        else:
            return None
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text=status,
            text_format="html",
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
        events: List[DeliveryEvent] = []

        a_cur = result.adata or session.adata
        a_info = f"{a_cur.n_obs:,} cells × {a_cur.n_vars:,} genes" if a_cur else ""

        summary = self._strip_local_paths((result.summary or "").strip())
        has_summary = bool(summary and summary.lower() not in self._BORING)
        long_summary = has_summary and len(summary) > 1200
        final_text = _fmt.md_to_html(summary) if has_summary and not long_summary else ""
        if (not final_text) and a_info:
            final_text = f"📊  <code>{_fmt.esc(a_info)}</code>"
        if not final_text:
            final_text = _fmt._DIV

        has_media = bool(result.figures)
        has_reports = bool(result.reports)
        artifacts = list(getattr(result, "artifacts", []) or [])
        has_artifacts = bool(artifacts)
        is_complex = has_media or has_reports or has_artifacts or long_summary

        if is_complex:
            explain = summary if has_summary else self._strip_local_paths(llm_text.strip())
            for i, rep in enumerate(result.reports or [], start=1):
                header = (
                    tr(user_text, en="📝  <b>Analysis Report</b>", zh="📝  <b>分析报告</b>")
                    if i == 1 else
                    tr(user_text, en=f"📝  <b>Analysis Report (cont. {i})</b>", zh=f"📝  <b>分析报告（续 {i}）</b>")
                )
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="prose",
                        text=rep,
                        metadata={"header": header, "always_expand": False},
                    )
                )
            n_figs = len(result.figures or [])
            for i, png_bytes in enumerate(result.figures or [], start=1):
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="photo",
                        binary=png_bytes,
                        caption=_fmt.figure_caption(i, n_figs),
                    )
                )
            for art in artifacts:
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="document",
                        binary=art.data,
                        filename=art.filename,
                        caption=f"📎  {art.filename}",
                    )
                )

            if explain:
                if len(explain) > 1200:
                    events.append(
                        DeliveryEvent(
                            route=route,
                            kind="prose",
                            text=explain,
                            metadata={"always_expand": False},
                        )
                    )
                    events.append(
                        DeliveryEvent(
                            route=route,
                            kind="text",
                            text=_fmt._DIV,
                            text_format="html",
                            controls=("save", "status", "memory"),
                        )
                    )
                else:
                    events.append(
                        DeliveryEvent(
                            route=route,
                            kind="text",
                            text=_fmt.md_to_html(explain),
                            text_format="html",
                            controls=("save", "status", "memory"),
                        )
                    )
            else:
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="text",
                        text=final_text,
                        text_format="html",
                        controls=("save", "status", "memory"),
                    )
                )
            events.append(
                DeliveryEvent(
                    route=route,
                    kind="text",
                    mode="edit",
                    target="analysis-draft",
                    text=tr(user_text, en="✅  Analysis complete", zh="✅  分析完成"),
                    text_format="html",
                )
            )
            return events

        if llm_text.strip():
            final_html = _fmt.md_to_html(llm_text.strip())
            if len(final_html) > 3200 or "<pre>" in final_html:
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="text",
                        mode="edit",
                        target="analysis-draft",
                        text=tr(user_text, en="✅  Analysis complete. Full response below.", zh="✅  分析完成，正文如下。"),
                        text_format="html",
                    )
                )
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="prose",
                        text=llm_text.strip(),
                        metadata={"always_expand": False},
                    )
                )
            else:
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="text",
                        mode="edit",
                        target="analysis-draft",
                        text=final_html,
                        text_format="html",
                        controls=("save", "status", "memory"),
                    )
                )
                return events
        else:
            events.append(
                DeliveryEvent(
                    route=route,
                    kind="text",
                    mode="edit",
                    target="analysis-draft",
                    text=tr(user_text, en="✅  Analysis complete", zh="✅  分析完成"),
                    text_format="html",
                )
            )

        events.append(
            DeliveryEvent(
                route=route,
                kind="text",
                text=final_text,
                text_format="html",
                controls=("save", "status", "memory"),
            )
        )
        return events

    @classmethod
    def _trim_for_draft(cls, text: str, max_len: int = _DRAFT_MAX) -> str:
        if len(text) <= max_len:
            return text
        head = int(max_len * 0.55)
        tail = max_len - head - 40
        if tail < 200:
            tail = 200
        return (
            text[:head].rstrip()
            + "\n\n[...long response truncated in the middle...]\n\n"
            + text[-tail:].lstrip()
        )

    @classmethod
    def _strip_local_paths(cls, text: str) -> str:
        t = text or ""
        t = re.sub(r'`[^`\n]*(?:/[^`\n]*){2,}`', '', t)
        t = re.sub(r'/(?:Users|home|tmp|var|opt|root|data|mnt|private)/\S+', '', t)
        t = re.sub(r'~[/\\]\S+', '', t)
        t = re.sub(
            rf'\.?/?(?:\w[\w/-]*/)+\w[\w.-]*\.(?:{cls._ARTIFACT_EXTS})',
            '',
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(r'[ \t]{2,}', ' ', t)
        t = re.sub(r'\n{3,}', '\n\n', t)
        return t.strip()


class TelegramDelivery:
    def __init__(
        self,
        *,
        bot: Any,
        chat_lock_factory: Callable[[int], asyncio.Lock],
        keyboard_factory: Callable[[Tuple[str, ...]], Any],
    ) -> None:
        self._bot = bot
        self._chat_lock_factory = chat_lock_factory
        self._keyboard_factory = keyboard_factory
        self._targets: Dict[str, int] = {}

    async def deliver(self, event: DeliveryEvent) -> None:
        chat_id = int(event.route.scope_id)
        thread_id = int(event.route.thread_id) if event.route.thread_id else None
        if event.kind == "typing":
            try:
                kwargs = {"chat_id": chat_id, "action": "typing"}
                if thread_id is not None:
                    kwargs["message_thread_id"] = thread_id
                await self._bot.send_chat_action(**kwargs)
            except Exception:
                pass
            return

        if event.kind == "prose":
            async with self._chat_lock_factory(chat_id):
                await _fmt.send_prose(
                    self._bot,
                    chat_id,
                    event.text,
                    header=str(event.metadata.get("header") or ""),
                    always_expand=bool(event.metadata.get("always_expand")),
                    message_thread_id=thread_id,
                )
            return

        if event.kind == "photo":
            async with self._chat_lock_factory(chat_id):
                await self._send_photo_or_file(
                    chat_id,
                    event.binary or b"",
                    event.caption,
                    thread_id=thread_id,
                )
            return

        if event.kind == "document":
            async with self._chat_lock_factory(chat_id):
                await self._send_document(
                    chat_id,
                    filename=event.filename or "artifact.bin",
                    data=event.binary or b"",
                    caption=event.caption,
                    thread_id=thread_id,
                )
            return

        if event.kind != "text":
            return

        reply_markup = self._keyboard_factory(event.controls) if event.controls else None
        parse_mode = "HTML" if event.text_format == "html" else None
        async with self._chat_lock_factory(chat_id):
            if event.mode == "open":
                msg_id = await self._send_text(
                    chat_id,
                    event.text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    thread_id=thread_id,
                )
                if msg_id is not None and event.target:
                    self._targets[self._target_key(event)] = msg_id
                return

            if event.mode == "edit":
                target_id = self._targets.get(self._target_key(event))
                if target_id is not None:
                    ok = await self._edit_text(
                        chat_id,
                        target_id,
                        event.text,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
                    if ok:
                        return
                msg_id = await self._send_text(
                    chat_id,
                    event.text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    thread_id=thread_id,
                )
                if msg_id is not None and event.target:
                    self._targets[self._target_key(event)] = msg_id
                return

            await self._send_text(
                chat_id,
                event.text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                thread_id=thread_id,
            )

    @staticmethod
    def _target_key(event: DeliveryEvent) -> str:
        return f"{event.route.route_key()}::{event.target or ''}"

    async def _send_text(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Any = None,
        thread_id: Optional[int] = None,
    ) -> Optional[int]:
        try:
            kwargs = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
            }
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            message = await self._bot.send_message(**kwargs)
            return getattr(message, "message_id", None)
        except Exception:
            pass
        try:
            kwargs = {
                "chat_id": chat_id,
                "text": self._strip_html(text),
                "reply_markup": reply_markup,
            }
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            message = await self._bot.send_message(**kwargs)
            return getattr(message, "message_id", None)
        except Exception as exc:
            logger.warning("Failed to send message: %s", exc)
            return None

    async def _edit_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Any = None,
    ) -> bool:
        try:
            await self._bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            return True
        except Exception as exc:
            if self._is_not_modified_error(exc):
                return True
            pass
        try:
            await self._bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=self._strip_html(text),
                reply_markup=reply_markup,
            )
            return True
        except Exception as exc:
            if self._is_not_modified_error(exc):
                return True
            return False

    async def _send_photo_or_file(
        self,
        chat_id: int,
        png_bytes: bytes,
        caption: str,
        *,
        thread_id: Optional[int] = None,
    ) -> None:
        try:
            kwargs = {
                "chat_id": chat_id,
                "photo": BytesIO(png_bytes),
                "caption": caption,
            }
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            await self._bot.send_photo(**kwargs)
            return
        except Exception:
            pass
        await self._send_document(
            chat_id,
            filename="figure.png",
            data=png_bytes,
            caption=caption,
            thread_id=thread_id,
        )

    async def _send_document(
        self,
        chat_id: int,
        *,
        filename: str,
        data: bytes,
        caption: str,
        thread_id: Optional[int] = None,
    ) -> None:
        try:
            kwargs = {
                "chat_id": chat_id,
                "document": BytesIO(data),
                "filename": filename,
                "caption": caption,
            }
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            await self._bot.send_document(**kwargs)
        except Exception as exc:
            logger.warning("Failed to send artifact %s: %s", filename, exc)

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text or "")

    @staticmethod
    def _is_not_modified_error(exc: Exception) -> bool:
        return "message is not modified" in str(exc).strip().lower()
# ---------------------------------------------------------------------------
# Bot builder
# ---------------------------------------------------------------------------


@dataclass
class _PollingState:
    conflict_detected: bool = False
    conflict_message: str = ""


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _telegram_runtime_dir() -> Path:
    runtime_dir = default_state_dir() / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _telegram_lock_path(token: str) -> Path:
    return _telegram_runtime_dir() / f"telegram-{_token_fingerprint(token)}.json"


def _load_runtime_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_runtime_record(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "token_hash": _token_fingerprint(token),
        "hostname": socket.gethostname(),
        "started_at": time.time(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cleanup_runtime_record(path: Path) -> None:
    record = _load_runtime_record(path)
    if int(record.get("pid") or 0) != os.getpid():
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.debug("Failed to remove Telegram runtime record %s", path, exc_info=True)


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process(pid: int, *, label: str, grace_seconds: float = 5.0) -> bool:
    if pid <= 0 or pid == os.getpid():
        return True
    if not _process_exists(pid):
        return True

    logger.warning("Stopping %s (pid=%s) before Telegram polling takeover.", label, pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        logger.warning("Cannot stop %s (pid=%s): permission denied.", label, pid)
        return False

    deadline = time.time() + max(grace_seconds, 0.0)
    while time.time() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        logger.warning("Cannot force stop %s (pid=%s): permission denied.", label, pid)
        return False

    for _ in range(10):
        if not _process_exists(pid):
            return True
        time.sleep(0.1)
    return not _process_exists(pid)


def _claim_telegram_instance(token: str) -> Path:
    path = _telegram_lock_path(token)
    record = _load_runtime_record(path)
    previous_pid = int(record.get("pid") or 0)
    if (
        previous_pid
        and previous_pid != os.getpid()
        and str(record.get("token_hash") or "") == _token_fingerprint(token)
    ):
        stopped = _terminate_process(
            previous_pid,
            label="existing OmicVerse Jarvis Telegram bot",
        )
        if not stopped:
            logger.warning(
                "Tracked Telegram bot instance pid=%s is still running; polling will retry once if Telegram returns 409 Conflict.",
                previous_pid,
            )

    _write_runtime_record(path, token)
    atexit.register(_cleanup_runtime_record, path)
    return path


def _register_polling_error_handler(app: Any, *, polling_state: _PollingState, conflict_type: type[BaseException]) -> None:
    async def _handle_error(update: object, context: Any) -> None:
        error = getattr(context, "error", None)
        if isinstance(error, conflict_type):
            polling_state.conflict_detected = True
            polling_state.conflict_message = str(error)
            logger.warning(
                "Telegram returned 409 Conflict because another getUpdates consumer is active. "
                "Stopping this polling loop and retrying once."
            )
            context.application.stop_running()
            return

        if error is not None:
            logger.error(
                "Unhandled Telegram bot error: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    app.add_error_handler(_handle_error)


def _telegram_conflict_error(conflict_message: str) -> str:
    detail = f" ({conflict_message})" if conflict_message else ""
    return (
        "Telegram polling still conflicts with another bot instance after Jarvis retried once"
        f"{detail}. Stop the other process using this bot token and start `omicverse jarvis` again."
    )


def _run_polling_with_restart(
    *,
    application_factory: Any,
    conflict_type: type[BaseException],
    restart_delay_seconds: float = _POLLING_RESTART_DELAY_SECONDS,
    max_attempts: int = _POLLING_MAX_ATTEMPTS,
) -> None:
    last_conflict_message = ""
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        app = application_factory()
        polling_state = _PollingState()
        _register_polling_error_handler(
            app,
            polling_state=polling_state,
            conflict_type=conflict_type,
        )
        logger.info("OmicVerse Jarvis bot starting (polling)...")
        app.run_polling(drop_pending_updates=True)
        if not polling_state.conflict_detected:
            return
        last_conflict_message = polling_state.conflict_message
        if attempt >= attempts:
            break
        time.sleep(max(restart_delay_seconds, 0.0))

    raise RuntimeError(_telegram_conflict_error(last_conflict_message))


def run_bot(
    token: str,
    session_manager: Any,
    access_control: AccessControl,
    verbose: bool = False,
) -> None:
    """Build and start the Telegram application (blocking)."""
    try:
        from telegram.error import Conflict
        from telegram.ext import Application
    except ImportError as exc:
        raise ImportError(
            "python-telegram-bot is required.  "
            "Install with: pip install omicverse[jarvis]"
        ) from exc

    _claim_telegram_instance(token)

    def _build_application() -> Any:
        app = Application.builder().token(token).concurrent_updates(True).build()
        _register_handlers(app, session_manager, access_control, verbose)
        return app

    _run_polling_with_restart(
        application_factory=_build_application,
        conflict_type=Conflict,
    )


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
            await update.message.reply_text("⛔ You do not have access.")
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
                    f"Agent initialization failed: {exc}\n"
                    "Check the <code>--model</code> argument, or run "
                    "<code>import omicverse as ov; print(ov.list_supported_models())</code> "
                    "to see the available models."
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
            InlineKeyboardButton("💾 Save",   callback_data="jarvis:save"),
            InlineKeyboardButton("📊 Status", callback_data="jarvis:status"),
            InlineKeyboardButton("🧠 Memory", callback_data="jarvis:memory"),
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
                response_language_instruction(user_text),
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
            reply = (response.content or "").strip() or tr(
                user_text,
                en="💬  Analysis in progress. Please try again shortly.",
                zh="💬  分析进行中，请稍后再试。",
            )
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
                        tr(
                            user_text,
                            en="⏳  Background analysis is still running. Please wait or use <code>/cancel</code>.",
                            zh="⏳  后台分析进行中，请等待完成。使用 <code>/cancel</code> 取消。",
                        ),
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
        text = render_start_text(getattr(update.message, "text", "") or "", html=True)
        await update.message.reply_text(text, parse_mode="HTML")

    # ------------------------------------------------------------------
    # /help
    # ------------------------------------------------------------------

    async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        text = render_help_text(getattr(update.message, "text", "") or "", html=True)
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

        lines = [f"👤  <b>User</b>  <code>{user.id}</code>"]
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
            lines.append("📭  No dataset loaded  ·  use <code>/load</code>")

        # Task status
        if runtime.task_state(route).running:
            lines.append("⚙️  Analysis running...  ·  <code>/cancel</code>")

        try:
            info = session.agent.get_current_session_info()
            if info:
                p  = info.get("prompt_count", 0)
                mp = info.get("max_prompts", "?")
                if getattr(session, "max_prompts_setting", 0) <= 0:
                    mp = "∞"
                lines.append(f"💬  Session  {p}/{mp}")
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
            "✅  Session reset. Kernel restarted.\n"
            "<i>Variables were cleared. You can reload adata with /load.</i>",
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
            await update.message.reply_text("🚫  Analysis cancelled.")
        else:
            await update.message.reply_text("ℹ️  No analysis is currently running.")

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
            await update.message.reply_text(
                "❌  No dataset loaded. Use <code>/load</code> first.",
                parse_mode="HTML",
            )
            return

        await update.message.reply_text("⏳  Saving...")
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
            await update.message.reply_text("❌  Save failed. Please try again.")

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
            await update.message.reply_text("ℹ️  No usage data yet. Run an analysis first.")
            return

        def _attr(obj: Any, *names: str, default: str = "?") -> str:
            for name in names:
                v = getattr(obj, name, None)
                if v is not None:
                    return f"{v:,}" if isinstance(v, int) else str(v)
            return default

        lines = [
            "📊  <b>Token Usage</b>  (most recent run)",
            _fmt._DIV,
            f"Input: <code>{_attr(usage, 'input_tokens')}</code>",
            f"Output: <code>{_attr(usage, 'output_tokens')}</code>",
            f"Total: <code>{_attr(usage, 'total_tokens')}</code>",
        ]
        # cache_read / cache_creation if present (Anthropic prompt caching)
        cr = _attr(usage, "cache_read_input_tokens", default="")
        cc = _attr(usage, "cache_creation_input_tokens", default="")
        if cr and cr != "?":
            lines.append(f"Cache read: <code>{cr}</code>")
        if cc and cc != "?":
            lines.append(f"Cache write: <code>{cc}</code>")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ------------------------------------------------------------------
    # /model [name]
    # ------------------------------------------------------------------

    async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        args = context.args or []

        if not args:
            await update.message.reply_text(
                render_model_help(sm._model, html=True),
                parse_mode="HTML",
            )
            return

        new_model = args[0]
        sm._model = new_model
        await update.message.reply_text(
            f"✅  Model switched to <code>{_fmt.esc(new_model)}</code>\n"
            "<i>Use /reset to recreate the kernel and apply it.</i>",
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
                "⚙️  <b>Kernel Status</b>",
                _fmt._DIV,
                f"🧩  Current: <code>{_fmt.esc(kname)}</code>",
                f"{icon}  {'Running' if alive else 'Stopped / Closed'}",
                f"💬  Prompts: <code>{p}</code> / <code>{mp}</code> (remaining {remaining})",
                f"🆔  Session: <code>{_fmt.esc(str(sid))}</code>",
                "",
                "Subcommands: <code>/kernel ls</code> · <code>/kernel new name</code> · <code>/kernel use name</code>",
            ]
            if isinstance(mp, int) and p >= mp * 0.8:
                lines += [
                    "",
                    "⚠️  The prompt budget is almost exhausted. The kernel will restart after the next analysis.",
                    "   Use <code>/reset</code> to restart manually, or increase <code>--max-prompts</code>.",
                ]
            if runtime.task_state(route).running:
                lines += ["", "⚙️  Analysis running  ·  <code>/cancel</code>"]
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            return

        sub = args[0].lower()
        if sub in {"ls", "list"}:
            active = sm.get_active_kernel(user_id)
            names = sm.list_kernels(user_id)
            lines = [
                "🧩  <b>Kernels</b>",
                _fmt._DIV,
            ]
            for name in names:
                mark = "✅" if name == active else "•"
                lines.append(f"{mark}  <code>{_fmt.esc(name)}</code>")
            lines += [
                "",
                "Switch: <code>/kernel use name</code>",
                "Create: <code>/kernel new name</code>",
            ]
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            return

        if sub in {"new", "create", "use", "switch"}:
            task = _tasks.get(user_id)
            if task and not task.done():
                await update.message.reply_text(
                    "⏳  An analysis is currently running. Wait for it to finish or use <code>/cancel</code>.",
                    parse_mode="HTML",
                )
                return
            if len(args) < 2:
                await update.message.reply_text(
                    "Usage: <code>/kernel new name</code> or <code>/kernel use name</code>\n"
                    "Names may contain letters, numbers, <code>.</code>, <code>_</code>, or <code>-</code>, with length 1-32.",
                    parse_mode="HTML",
                )
                return
            name = args[1]
            try:
                if sub in {"new", "create"}:
                    sm.create_kernel(user_id, name, switch=True)
                    action = "Created and switched"
                else:
                    sm.switch_kernel(user_id, name, create=False)
                    action = "Switched"
            except Exception as exc:
                await update.message.reply_text(
                    _fmt.error_message(str(exc)), parse_mode="HTML"
                )
                return
            await update.message.reply_text(
                f"✅  {action} to kernel: <code>{_fmt.esc(sm.get_active_kernel(user_id))}</code>",
                parse_mode="HTML",
            )
            return

        await update.message.reply_text(
            "Usage: <code>/kernel</code> | <code>/kernel ls</code> | "
            "<code>/kernel new name</code> | <code>/kernel use name</code>",
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
            lines.append(f"📊  <b>Data Files</b>  ({len(h5ad_files)})")
            for f in h5ad_files[:10]:
                try:
                    mb = f.stat().st_size / 1_048_576
                    lines.append(f"  • <code>{_fmt.esc(f.name)}</code>  <i>{mb:.1f} MB</i>")
                except OSError:
                    lines.append(f"  • <code>{_fmt.esc(f.name)}</code>")
            if len(h5ad_files) > 10:
                lines.append(f"  <i>… and {len(h5ad_files) - 10} more</i>")
        else:
            lines.append("📊  <b>Data Files</b>  (empty)")
            lines.append(f"  <i>scp *.h5ad user@host:{ws}</i>")

        lines += [
            "",
            f"📋  AGENTS.md  {'✅' if agents_md else '—'}",
            f"🧠  Today's Memory  {'✅' if today_log.exists() else '—'}",
            "",
            "<code>/load &lt;filename&gt;</code>  ·  <code>/ls</code>  ·  <code>/memory</code>",
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
                "Usage: <code>/find &lt;pattern&gt;</code>, for example <code>/find *.h5ad</code>",
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
                "Usage: <code>/load &lt;filename&gt;</code>, for example <code>/load pbmc3k.h5ad</code>\n"
                "Use <code>/workspace</code> to inspect available files.",
                parse_mode="HTML",
            )
            return

        filename = args[0]
        await update.message.reply_text(
            f"⏳  Loading <code>{_fmt.esc(filename)}</code>...", parse_mode="HTML"
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
                hint  = f"\nAvailable files: <code>{_fmt.esc(names)}</code>"
            await update.message.reply_text(
                f"❌  File not found: <code>{_fmt.esc(filename)}</code>{hint}", parse_mode="HTML"
            )
            return

        await update.message.reply_text(
            f"✅  Loaded successfully\n{_fmt._DIV}\n"
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
                "Usage: <code>/shell &lt;command&gt;</code>\n"
                "Allowed: <code>ls  find  cat  head  wc  file  du  pwd  tree</code>",
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
        async with _chat_lock(update.effective_chat.id):
            await _fmt.send_prose(
                context.bot,
                update.effective_chat.id,
                text,
                header="🧠  <b>Analysis History</b> (last two days)",
                always_expand=True,
                message_thread_id=getattr(update.effective_message, "message_thread_id", None),
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
            await update.message.reply_text(
                "⚠️  Please upload a <code>.h5ad</code> file.",
                parse_mode="HTML",
            )
            return

        await update.message.reply_text("⏳  Downloading and loading...")
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
                f"✅  Loaded successfully\n{_fmt._DIV}\n"
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
                await context.bot.send_message(chat_id=chat_id, text="❌  No dataset loaded.")
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
                await context.bot.send_message(chat_id=chat_id, text="❌  Save failed.")

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
                await context.bot.send_message(chat_id=chat_id, text="📭  No dataset loaded.")

        elif data == "jarvis:memory":
            text = session.get_recent_memory_text()
            await _send_prose_locked(
                context.bot,
                chat_id,
                text,
                header="🧠  <b>Analysis History</b>",
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
