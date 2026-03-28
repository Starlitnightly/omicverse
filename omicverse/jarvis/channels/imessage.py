"""
iMessage channel for OmicVerse Jarvis via ``imsg rpc``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..model_help import render_model_help
from ..runtime import (
    AgentBridgeExecutionAdapter,
    ConversationRoute,
    DeliveryEvent,
    MessageEnvelope,
    MessageRuntime,
    MessageRouter,
)
from .channel_core import (
    command_parts,
    default_summary,
    format_status_plain,
    gather_status,
    strip_local_paths,
    text_chunks,
)

logger = logging.getLogger("omicverse.jarvis.imessage")

_BORING_SUMMARIES = {"分析完成", "分析完成。", "task completed", "done", "完成"}
_MAX_TEXT = 3800
_PROGRESS_GAP = 5.0
_DEFAULT_IMESSAGE_PROBE_TIMEOUT = 10.0
_IMSG_PERMISSION_DENIED_RE = re.compile(
    r'permissionDenied\(path:\s*"(?P<path>[^"]+)"(?:,\s*underlying:\s*(?P<detail>.+?))?\)$',
    re.IGNORECASE,
)


@dataclass
class IMessageTarget:
    kind: str
    value: str
    service: str = "auto"


def _normalize_handle(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    lower = value.lower()
    for prefix in ("imessage:", "sms:", "auto:"):
        if lower.startswith(prefix):
            return _normalize_handle(value[len(prefix):])
    if value.startswith("+"):
        return re.sub(r"\s+", "", value)
    if "@" in value:
        return value.lower()
    return re.sub(r"\s+", "", value)


def _parse_target(raw: str) -> IMessageTarget:
    text = (raw or "").strip()
    if not text:
        raise ValueError("iMessage target is required")
    lower = text.lower()
    for prefix, service in (("imessage:", "imessage"), ("sms:", "sms"), ("auto:", "auto")):
        if lower.startswith(prefix):
            nested = _parse_target(text[len(prefix):].strip())
            if nested.kind == "handle":
                nested.service = service
            return nested
    if lower.startswith("chat_id:"):
        return IMessageTarget(kind="chat_id", value=str(int(text.split(":", 1)[1].strip())))
    if lower.startswith("chat_guid:"):
        return IMessageTarget(kind="chat_guid", value=text.split(":", 1)[1].strip())
    if lower.startswith("chat_identifier:"):
        return IMessageTarget(kind="chat_identifier", value=text.split(":", 1)[1].strip())
    return IMessageTarget(kind="handle", value=text, service="auto")


def _message_target(message: Dict[str, Any]) -> Optional[str]:
    chat_id = message.get("chat_id")
    if isinstance(chat_id, int):
        return f"chat_id:{chat_id}"
    chat_guid = str(message.get("chat_guid") or "").strip()
    if chat_guid:
        return f"chat_guid:{chat_guid}"
    chat_identifier = str(message.get("chat_identifier") or "").strip()
    if chat_identifier:
        return f"chat_identifier:{chat_identifier}"
    sender = _normalize_handle(str(message.get("sender") or ""))
    return sender or None


def _resolve_cli_path(cli_path: str) -> Optional[str]:
    value = str(cli_path or "imsg").strip()
    if not value:
        return None
    expanded = os.path.expanduser(value)
    if os.path.sep in expanded:
        return expanded if os.path.isfile(expanded) and os.access(expanded, os.X_OK) else None
    return shutil.which(expanded)


async def _probe_rpc_support(cli_path: str, timeout: float) -> None:
    resolved_cli = _resolve_cli_path(cli_path)
    if resolved_cli is None:
        raise RuntimeError(
            f"无法找到 imsg 可执行文件（当前路径: {cli_path}）。"
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            resolved_cli,
            "rpc",
            "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"无法启动 imsg。请先安装它，例如：brew install steipete/tap/imsg（当前路径: {cli_path}）"
        ) from exc

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError("imsg rpc --help 超时，无法确认当前 imsg 是否支持 rpc 子命令。")

    combined = "\n".join(
        part.decode("utf-8", errors="ignore").strip()
        for part in (stdout, stderr)
        if part
    ).strip()
    normalized = combined.lower()
    if "unknown command" in normalized and "rpc" in normalized:
        raise RuntimeError('当前 imsg 不支持 "rpc" 子命令，请先升级 imsg。')
    if proc.returncode not in (0, None):
        raise RuntimeError(
            combined or f"imsg rpc --help failed (code {proc.returncode})"
        )


async def probe_imessage(
    *,
    cli_path: str = "imsg",
    db_path: Optional[str] = None,
    timeout: float = _DEFAULT_IMESSAGE_PROBE_TIMEOUT,
) -> None:
    await _probe_rpc_support(cli_path, timeout)
    client = IMessageRpcClient(cli_path=cli_path, db_path=db_path)
    try:
        await client.request("chats.list", {"limit": 1}, timeout=timeout)
    finally:
        await client.stop()


class IMessageRpcClient:
    def __init__(
        self,
        *,
        cli_path: str = "imsg",
        db_path: Optional[str] = None,
        on_notification: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        self._cli_path = cli_path or "imsg"
        self._db_path = os.path.expanduser(db_path) if db_path else None
        self._on_notification = on_notification
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._pending: Dict[str, asyncio.Future] = {}
        self._writer_lock = asyncio.Lock()
        self._next_id = 1
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._terminal_error: Optional[RuntimeError] = None

    async def start(self) -> None:
        if self._proc is not None:
            return
        args = [self._cli_path, "rpc"]
        if self._db_path:
            args.extend(["--db", self._db_path])
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"无法启动 imsg。请先安装它，例如：brew install steipete/tap/imsg（当前路径: {self._cli_path}）"
            ) from exc
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdin:
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            proc.terminate()
        for task in (self._stdout_task, self._stderr_task):
            if task is not None:
                task.cancel()
        self._fail_all(self._terminal_error or RuntimeError("imsg rpc closed"))

    async def wait_closed(self) -> None:
        if self._proc is None:
            return
        await self._proc.wait()

    async def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout: float = 30.0,
    ) -> Any:
        await self.start()
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("imsg rpc not running")
        request_id = str(self._next_id)
        self._next_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[request_id] = fut
        payload = {
            "jsonrpc": "2.0",
            "id": int(request_id),
            "method": method,
            "params": params or {},
        }
        async with self._writer_lock:
            self._proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            await self._proc.stdin.drain()
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)

    async def send_message(
        self,
        target: str,
        text: str,
        *,
        file_path: Optional[str] = None,
        timeout: float = 60.0,
    ) -> Any:
        parsed = _parse_target(target)
        params: Dict[str, Any] = {"text": text, "service": parsed.service or "auto", "region": "US"}
        if file_path:
            params["file"] = file_path
        if parsed.kind == "chat_id":
            params["chat_id"] = int(parsed.value)
        elif parsed.kind == "chat_guid":
            params["chat_guid"] = parsed.value
        elif parsed.kind == "chat_identifier":
            params["chat_identifier"] = parsed.value
        else:
            params["to"] = parsed.value
        return await self.request("send", params, timeout=timeout)

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                raw = await self._proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    parsed_error = self._parse_terminal_error(line)
                    if parsed_error is not None:
                        self._terminal_error = parsed_error
                        self._fail_all(parsed_error)
                        continue
                    logger.warning("Failed to parse imsg rpc line: %s", line[:200])
                    continue
                if payload.get("id") is not None:
                    request_id = str(payload["id"])
                    fut = self._pending.get(request_id)
                    if fut is None or fut.done():
                        continue
                    if payload.get("error"):
                        message = str((payload.get("error") or {}).get("message") or "imsg rpc error")
                        data = (payload.get("error") or {}).get("data")
                        if data is not None:
                            message = f"{message}: {data}"
                        fut.set_exception(RuntimeError(message))
                    else:
                        fut.set_result(payload.get("result"))
                    continue
                method = str(payload.get("method") or "").strip()
                params = payload.get("params")
                if method and isinstance(params, dict) and self._on_notification is not None:
                    await self._on_notification(method, params)
        finally:
            self._fail_all(self._terminal_error or RuntimeError("imsg rpc closed"))

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        while True:
            raw = await self._proc.stderr.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if line:
                parsed_error = self._parse_terminal_error(line)
                if parsed_error is not None:
                    self._terminal_error = parsed_error
                    self._fail_all(parsed_error)
                logger.warning("imsg rpc: %s", line)

    def _fail_all(self, error: Exception) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    @staticmethod
    def _parse_terminal_error(line: str) -> Optional[RuntimeError]:
        text = str(line or "").strip()
        if not text:
            return None

        match = _IMSG_PERMISSION_DENIED_RE.search(text)
        if match:
            denied_path = match.group("path") or "~/Library/Messages/chat.db"
            detail = str(match.group("detail") or "authorization denied").strip()
            return RuntimeError(
                "imsg cannot access the Messages database at "
                f"{denied_path}. macOS denied permission ({detail}). "
                "Grant Full Disk Access to the app that launches Jarvis "
                "(for example Terminal, iTerm, or Python) and retry."
            )

        return None


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def imessage_route_from_message(message: Dict[str, Any]) -> Optional[ConversationRoute]:
    """Build a :class:`ConversationRoute` from an imsg RPC message dict."""
    is_group = bool(message.get("is_group"))
    chat_id = message.get("chat_id")
    sender = _normalize_handle(str(message.get("sender") or ""))
    chat_identifier = str(message.get("chat_identifier") or "").strip()
    scope_id = ""
    if isinstance(chat_id, int):
        scope_id = str(chat_id)
    elif not is_group and sender:
        scope_id = sender
    elif chat_identifier:
        scope_id = chat_identifier
    if not scope_id:
        return None
    return ConversationRoute(
        channel="imessage",
        scope_type="group" if is_group else "dm",
        scope_id=scope_id,
    )


# ---------------------------------------------------------------------------
# Presenter  (implements ``MessagePresenter`` protocol)
# ---------------------------------------------------------------------------


class IMessageRuntimePresenter:
    """Produce :class:`DeliveryEvent` instances for iMessage rendering (plain text)."""

    _BORING = _BORING_SUMMARIES

    def ack(self, envelope: MessageEnvelope, session: Any) -> List[DeliveryEvent]:
        return [
            DeliveryEvent(
                route=envelope.route,
                kind="text",
                text="⏳ 已收到，开始分析。",
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
            text="⏳ 已收到，开始分析。",
        )

    def draft_update(self, route: ConversationRoute, llm_text: str, progress: str) -> DeliveryEvent:
        if progress:
            text = f"⚙️ {progress[:200]}"
        elif llm_text.strip():
            text = f"💭 {llm_text.strip()[-200:]}"
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
            text="🚫 已取消当前分析。",
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
        return None

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
        return None  # iMessage is plain text; no mid-delivery status needed

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

        # Figures
        for index, figure_data in enumerate(result.figures or [], start=1):
            data = figure_data if isinstance(figure_data, bytes) else b""
            events.append(
                DeliveryEvent(
                    route=route,
                    kind="photo",
                    binary=data,
                    filename=f"figure_{index}.png",
                    caption=f"图 {index}",
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
            summary = default_summary(session)
        for chunk in text_chunks(summary, limit=_MAX_TEXT):
            events.append(DeliveryEvent(route=route, kind="text", text=chunk))

        # Mark draft complete
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
# Delivery  (iMessage transport layer)
# ---------------------------------------------------------------------------


class IMessageDelivery:
    """Translate :class:`DeliveryEvent` instances into iMessage RPC calls."""

    # Edit-mode texts that are pure status markers; skip sending as new
    # messages to avoid noise (iMessage cannot edit sent messages).
    _SKIP_EDITS = {"✅ 分析完成", "💭 分析中…", "⏳ 已收到，开始分析。"}

    def __init__(self, *, client: IMessageRpcClient) -> None:
        self._client = client
        self._targets: Dict[str, str] = {}

    def register_target(self, route: ConversationRoute, target: str) -> None:
        """Cache the iMessage target (handle / chat_id) for a route."""
        self._targets[route.route_key()] = target

    async def deliver(self, event: DeliveryEvent) -> None:
        target = self._targets.get(event.route.route_key()) or event.route.scope_id
        if not target:
            logger.warning("Cannot resolve iMessage target for %s", event.route.route_key())
            return

        try:
            if event.kind == "typing":
                return

            if event.kind in ("photo", "document"):
                await self._send_binary(target, event)
                return

            # Text events (all modes)
            text = (event.text or "").strip()
            if not text:
                return

            # For edit-mode events: iMessage cannot edit, so skip pure status
            # markers but send substantive updates as new messages.
            if event.mode == "edit" and text in self._SKIP_EDITS:
                return

            for chunk in text_chunks(text, limit=_MAX_TEXT):
                await self._client.send_message(target, chunk)
        except Exception:
            logger.warning(
                "iMessage delivery failed for %s kind=%s",
                event.route.route_key(),
                event.kind,
                exc_info=True,
            )

    async def _send_binary(self, target: str, event: DeliveryEvent) -> None:
        data = event.binary or b""
        if not data:
            return
        filename = event.filename or "attachment.bin"
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(
            prefix="jarvis-imessage-", suffix=suffix, delete=False,
        ) as handle:
            handle.write(data)
            temp_path = handle.name
        try:
            caption = event.caption or filename
            await self._client.send_message(target, caption, file_path=temp_path, timeout=120.0)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


class IMessageJarvisBot:
    def __init__(
        self,
        *,
        session_manager: Any,
        cli_path: str = "imsg",
        db_path: Optional[str] = None,
        include_attachments: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._sm = session_manager
        self._cli_path = cli_path or "imsg"
        self._db_path = db_path
        self._include_attachments = include_attachments
        self._stop_event = stop_event
        self._client = IMessageRpcClient(
            cli_path=self._cli_path,
            db_path=self._db_path,
            on_notification=self._handle_notification,
        )
        self._delivery = IMessageDelivery(client=self._client)
        self._message_runtime = MessageRuntime(
            router=MessageRouter(session_manager),
            presenter=IMessageRuntimePresenter(),
            execution_adapter=AgentBridgeExecutionAdapter(),
            deliver=self._delivery.deliver,
            web_bridge=getattr(session_manager, "gateway_web_bridge", None),
        )

    async def run(self) -> None:
        await probe_imessage(
            cli_path=self._cli_path,
            db_path=self._db_path,
        )
        await self._client.start()
        logger.info("OmicVerse Jarvis iMessage channel starting via imsg rpc")
        await self._client.request(
            "watch.subscribe",
            {"attachments": bool(self._include_attachments)},
            timeout=60.0,
        )
        try:
            if self._stop_event is None:
                await self._client.wait_closed()
            else:
                wait_closed_task = asyncio.create_task(self._client.wait_closed())
                stop_task = asyncio.create_task(asyncio.to_thread(self._stop_event.wait))
                done, pending = await asyncio.wait(
                    {wait_closed_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if stop_task in done and self._stop_event.is_set():
                    await self._client.stop()
                elif wait_closed_task in done:
                    await wait_closed_task
        finally:
            await self._client.stop()

    async def _handle_notification(self, method: str, params: Dict[str, Any]) -> None:
        if method != "message":
            if method == "error":
                logger.warning("imessage watch error: %s", params)
            return
        message = params.get("message")
        if not isinstance(message, dict):
            return
        await self._handle_message(message)

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        sender = _normalize_handle(str(message.get("sender") or ""))
        if not sender or bool(message.get("is_from_me")):
            return

        route = imessage_route_from_message(message)
        target = _message_target(message)
        if route is None or not target:
            return

        text = str(message.get("text") or "").strip()
        attachments = message.get("attachments") or []

        # Check for .h5ad file attachments; the imsg RPC exposes local file
        # paths so we can read them directly without a network download.
        if isinstance(attachments, list):
            for _att in attachments:
                _path = str(
                    (_att.get("path") if isinstance(_att, dict) else None) or ""
                ).strip()
                _fname = str(
                    (_att.get("filename") if isinstance(_att, dict) else None)
                    or (Path(_path).name if _path else "")
                ).strip()
                if _fname.lower().endswith(".h5ad") and _path:
                    session = self._message_runtime.get_session(route)
                    await self._handle_h5ad_attachment(route, session, target, _path, _fname)
                    return

        if not text and attachments:
            text = "<media:attachment>"
        if not text:
            return

        if text.startswith("/"):
            await self._handle_command(route, target, text)
            return

        # Register the delivery target and delegate to the shared runtime.
        self._delivery.register_target(route, target)
        envelope = MessageEnvelope(
            route=route,
            text=text,
            sender_id=sender,
            trigger="direct" if route.is_direct else "group",
            explicit_trigger=True,
        )
        try:
            await self._message_runtime.handle_message(envelope)
        except Exception as exc:
            logger.exception("iMessage analysis dispatch failed")
            await self._send_text(target, f"分析异常: {exc}")

    async def _handle_command(self, route: ConversationRoute, target: str, text: str) -> None:
        session = self._message_runtime.get_session(route)
        cmd, tail = command_parts(text)

        if cmd == "/help":
            await self._send_text(
                target,
                "可用命令:\n"
                "/help 查看帮助\n"
                "/reset 重置当前会话\n"
                "/model [名称] 查看或切换模型\n"
                "/cancel 取消当前分析",
            )
            return

        if cmd == "/reset":
            await self._message_runtime.cancel(route)
            session.reset()
            await self._send_text(target, "✅ 已重置当前会话。")
            return

        if cmd == "/cancel":
            cancelled = await self._message_runtime.cancel(route)
            if cancelled:
                await self._send_text(target, "🚫 已取消当前分析。")
            else:
                await self._send_text(target, "当前没有正在运行的分析任务。")
            return

        if cmd == "/status":
            state = self._message_runtime.task_state(route)
            info = gather_status(
                session,
                is_running=state.running,
                running_request=state.request[:300] if state.running else "",
            )
            await self._send_text(target, format_status_plain(info))
            return

        if cmd == "/model":
            if not tail:
                for chunk in text_chunks(render_model_help(getattr(self._sm, "_model", "unknown"))):
                    await self._send_text(target, chunk)
                return
            self._sm._model = tail
            await self._send_text(
                target,
                f"✅ 模型已切换为 {tail}\n请 /reset 重启 kernel 使新模型生效。",
            )
            return

        await self._send_text(target, f"未知命令: {cmd}\n发送 /help 查看帮助。")

    async def _send_text(self, target: str, text: str) -> None:
        for chunk in text_chunks(text):
            await self._client.send_message(target, chunk)

    async def _handle_h5ad_attachment(
        self,
        route: ConversationRoute,
        session: Any,
        target: str,
        src_path: str,
        file_name: str,
    ) -> None:
        """Copy a local .h5ad attachment into the session workspace and load it."""
        await self._send_text(target, "⏳ 正在加载文件…")
        try:
            src = Path(src_path)
            if not src.exists():
                await self._send_text(target, f"❌ 找不到文件: {src_path}")
                return
            dest = session.workspace / file_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            await asyncio.to_thread(shutil.copy2, str(src), str(dest))
            loaded = await asyncio.to_thread(session.load_from_workspace, file_name)
            if loaded is not None:
                a = loaded
                await self._send_text(
                    target,
                    f"✅ 加载成功\n🔬 {a.n_obs:,} cells × {a.n_vars:,} genes\n📁 {file_name}",
                )
            else:
                await self._send_text(target, f"✅ 已接收 {file_name}，但自动加载失败，请检查文件格式。")
        except Exception as exc:
            logger.exception("iMessage failed to load h5ad attachment")
            await self._send_text(target, f"❌ 文件处理失败: {exc}")

def run_imessage_bot(
    *,
    session_manager: Any,
    cli_path: str = "imsg",
    db_path: Optional[str] = None,
    include_attachments: bool = False,
    stop_event: Optional[threading.Event] = None,
) -> None:
    bot = IMessageJarvisBot(
        session_manager=session_manager,
        cli_path=cli_path,
        db_path=db_path,
        include_attachments=include_attachments,
        stop_event=stop_event,
    )
    asyncio.run(bot.run())
