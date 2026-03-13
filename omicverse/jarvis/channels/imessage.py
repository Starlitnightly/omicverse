"""
iMessage channel for OmicVerse Jarvis via ``imsg rpc``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..agent_bridge import AgentBridge
from ..channel_language import response_language_instruction, tr
from ..channel_shared import render_help_text, render_start_text
from ..gateway.routing import GatewaySessionRegistry, SessionKey
from ..model_help import render_model_help

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


@dataclass
class RunningTask:
    task: asyncio.Task
    request: str
    started_at: float


def _strip_local_paths(text: str) -> str:
    t = text or ""
    t = re.sub(r'`[^`\n]*(?:/[^`\n]*){2,}`', '', t)
    t = re.sub(r'/(?:Users|home|tmp|var|opt|root|data|mnt|private)/\S+', '', t)
    t = re.sub(r'~[/\\]\S+', '', t)
    _ext = r"pdf|csv|tsv|txt|xlsx|html|json|h5ad|png|jpg|svg"
    t = re.sub(rf'\.?/?(?:\w[\w/-]*/)+\w[\w.-]*\.(?:{_ext})', '', t, flags=re.IGNORECASE)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


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
            chunks.append(para[pos:pos + limit])
            pos += limit
        buf = ""
    if buf:
        chunks.append(buf)
    return chunks


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


def _attachment_candidates(message: Dict[str, Any]) -> List[str]:
    raw = message.get("attachments")
    if not raw:
        return []
    entries = raw if isinstance(raw, list) else [raw]
    candidates: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            value = entry.strip()
            if value:
                candidates.append(value)
            continue
        if not isinstance(entry, dict):
            continue
        for key in ("path", "file_path", "file", "filename", "name", "transfer_name"):
            value = str(entry.get(key) or "").strip()
            if value:
                candidates.append(value)
    seen = set()
    result: List[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


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
            f"Could not find the imsg executable (current path: {cli_path})."
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
            f"Could not start imsg. Install it first, for example with: brew install steipete/tap/imsg (current path: {cli_path})"
        ) from exc

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError("Timed out while running `imsg rpc --help`; could not verify whether this imsg build supports the `rpc` subcommand.")

    combined = "\n".join(
        part.decode("utf-8", errors="ignore").strip()
        for part in (stdout, stderr)
        if part
    ).strip()
    normalized = combined.lower()
    if "unknown command" in normalized and "rpc" in normalized:
        raise RuntimeError('The current imsg build does not support the "rpc" subcommand. Upgrade imsg first.')
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
                f"Could not start imsg. Install it first, for example with: brew install steipete/tap/imsg (current path: {self._cli_path})"
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


class IMessageJarvisBot:
    def __init__(
        self,
        *,
        session_manager: Any,
        cli_path: str = "imsg",
        db_path: Optional[str] = None,
        include_attachments: bool = False,
    ) -> None:
        self._sm = session_manager
        self._cli_path = cli_path or "imsg"
        self._db_path = db_path
        self._include_attachments = include_attachments
        self._route_registry = GatewaySessionRegistry(session_manager)
        self._tasks: Dict[str, RunningTask] = {}
        self._pending: Dict[str, List[str]] = {}
        self._client = IMessageRpcClient(
            cli_path=self._cli_path,
            db_path=self._db_path,
            on_notification=self._handle_notification,
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
            await self._client.wait_closed()
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

        session_key = self._session_key_for_message(message)
        target = _message_target(message)
        if session_key is None or not target:
            return

        text = str(message.get("text") or "").strip()
        attachments = _attachment_candidates(message)
        if not text and attachments:
            if await self._handle_attachment_import(session_key, target, attachments):
                return
            names = ", ".join(Path(item).name or item for item in attachments[:3])
            suffix = "..." if len(attachments) > 3 else ""
            await self._send_text(
                target,
                "Attachment received"
                + (f": {names}{suffix}" if names else ".")
                + "\nPlace the file in the workspace and use /load <filename> if auto-import is not available.",
            )
            return
        if not text:
            return

        if text.startswith("/"):
            await self._handle_command(session_key, target, text)
            return

        task_key = session_key.as_key()
        running = self._tasks.get(task_key)
        session = self._route_registry.get_or_create(session_key)
        if running is not None and not running.task.done():
            self._pending.setdefault(task_key, []).append(text)
            asyncio.create_task(
                self._quick_chat(
                    target,
                    session,
                    text,
                    running_request=running.request,
                    queued=True,
                )
            )
            return

        await self._spawn_analysis(task_key, session_key, target, session, text)

    async def _handle_attachment_import(
        self,
        session_key: SessionKey,
        target: str,
        attachments: List[str],
    ) -> bool:
        session = self._route_registry.get_or_create(session_key)
        for raw in attachments:
            path = Path(os.path.expanduser(str(raw or "").strip()))
            if not path.name.lower().endswith(".h5ad"):
                continue
            if not path.is_absolute() or not path.exists() or not path.is_file():
                continue
            try:
                dest = session.workspace / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if path != dest:
                    shutil.copy2(path, dest)
                adata = await asyncio.to_thread(session.load_from_workspace, path.name)
                if adata is not None:
                    await self._send_text(
                        target,
                        f"✅ Uploaded and loaded {path.name}\n🔬 {adata.n_obs:,} cells x {adata.n_vars:,} genes",
                    )
                    return True
            except Exception as exc:
                logger.warning("iMessage attachment import failed for %s: %s", path, exc)
                await self._send_text(target, f"Attachment import failed for {path.name}: {exc}")
                return True
        return False

    def _session_key_for_message(self, message: Dict[str, Any]) -> Optional[SessionKey]:
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
        return SessionKey(
            channel="imessage",
            scope_type="group" if is_group else "dm",
            scope_id=scope_id,
        )

    def _build_full_request(self, session: Any, text: str) -> str:
        ctx_parts: List[str] = [f"[Response language]\n{response_language_instruction(text)}"]
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
        return "\n\n".join(ctx_parts + [f"[Current request]\n{text}"]) if ctx_parts else text

    async def _handle_command(self, session_key: SessionKey, target: str, text: str) -> None:
        session = self._route_registry.get_or_create(session_key)
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        tail = parts[1].strip() if len(parts) > 1 else ""
        route = session_key.as_key()

        if cmd == "/start":
            await self._send_text(target, render_start_text(text))
            return

        if cmd == "/help":
            await self._send_text(target, render_help_text(text))
            return

        if cmd == "/cancel":
            await self._handle_cancel(target, route)
            return

        if cmd == "/status":
            await self._handle_status(target, session, route)
            return

        if cmd == "/reset":
            self._pending.pop(route, None)
            running = self._tasks.get(route)
            if running and not running.task.done():
                running.task.cancel()
            session.reset()
            await self._send_text(target, "✅ Current session reset.")
            return

        if cmd == "/kernel":
            await self._handle_kernel(target, session, route, parts[1].split() if len(parts) > 1 else [])
            return

        if cmd == "/workspace":
            await self._handle_workspace(target, session)
            return

        if cmd == "/ls":
            await self._handle_ls(target, session, tail)
            return

        if cmd == "/find":
            await self._handle_find(target, session, tail)
            return

        if cmd == "/load":
            await self._handle_load(target, session, tail)
            return

        if cmd == "/shell":
            await self._handle_shell(target, session, tail)
            return

        if cmd == "/memory":
            await self._handle_memory(target, session)
            return

        if cmd == "/usage":
            await self._handle_usage(target, session)
            return

        if cmd == "/save":
            await self._handle_save(target, session)
            return

        if cmd == "/model":
            if not tail:
                for chunk in _text_chunks(render_model_help(getattr(self._sm, "_model", "unknown"))):
                    await self._send_text(target, chunk)
                return
            self._sm._model = tail
            await self._send_text(
                target,
                f"✅ Model switched to {tail}\nUse /reset to recreate the kernel and apply it.",
            )
            return

        await self._send_text(target, f"Unknown command: {cmd}\nSend /help for usage.")

    async def _quick_chat(
        self,
        target: str,
        session: Any,
        user_text: str,
        running_request: str = "",
        queued: bool = False,
    ) -> None:
        try:
            system_lines = [
                "You are OmicVerse Jarvis, a bioinformatics AI assistant.",
                "The user is chatting with you while a background analysis is running.",
                "Answer concisely. Do NOT execute code or call tools.",
                response_language_instruction(user_text),
            ]
            if running_request:
                system_lines.append(f"\nCurrently running: {running_request[:300]}")
            if queued:
                system_lines.append(
                    "The user's message has been queued and will start after the current analysis finishes. Mention this naturally."
                )
            if session.adata is not None:
                a = session.adata
                system_lines.append(f"Loaded data: {a.n_obs:,} cells x {a.n_vars:,} genes")
            try:
                memory_ctx = session.get_memory_context()
                if memory_ctx:
                    system_lines.append(f"\nRecent history:\n{memory_ctx[:600]}")
            except Exception:
                pass
            response = await session.agent._llm.chat(
                [
                    {"role": "system", "content": "\n".join(system_lines)},
                    {"role": "user", "content": user_text},
                ],
                tools=None,
                tool_choice=None,
            )
            reply = (response.content or "").strip() or tr(
                user_text,
                en="⏳ Background analysis is still running. Please wait.",
                zh="⏳ 后台分析进行中，请等待完成。",
            )
            await self._send_text(target, reply)
        except Exception as exc:
            logger.warning("iMessage quick_chat failed: %s", exc)
            await self._send_text(
                target,
                tr(
                    user_text,
                    en="⏳ Background analysis is still running. Please wait or use /cancel.",
                    zh="⏳ 后台分析进行中，请等待完成。使用 /cancel 取消。",
                ),
            )

    async def _spawn_analysis(
        self,
        task_key: str,
        session_key: SessionKey,
        target: str,
        session: Any,
        user_text: str,
    ) -> None:
        if session.adata is not None:
            ack = tr(
                user_text,
                en=(
                    "⏳ Request received. Starting analysis.\n"
                    f"Loaded data: {session.adata.n_obs:,} cells x {session.adata.n_vars:,} genes"
                ),
                zh=(
                    f"⏳ 已收到，开始分析。\n"
                    f"已加载数据: {session.adata.n_obs:,} cells x {session.adata.n_vars:,} genes"
                ),
            )
        else:
            try:
                h5ad_files = session.list_h5ad_files()
            except Exception:
                h5ad_files = []
            if h5ad_files:
                names = "  ".join(file.name for file in h5ad_files[:5])
                ack = tr(
                    user_text,
                    en=(
                        "⏳ Request received. Starting analysis.\n"
                        f"Detected files in the workspace: {names}\n"
                        "Use /load <filename> to load one."
                    ),
                    zh=(
                        f"⏳ 已收到，开始分析。\n"
                        f"workspace 中检测到文件: {names}\n"
                        f"使用 /load <文件名> 加载。"
                    ),
                )
            else:
                ack = tr(
                    user_text,
                    en="⏳ Request received. Starting analysis.\nNo dataset detected. The agent will load data if needed.",
                    zh="⏳ 已收到，开始分析。\n未检测到已加载数据，Agent 将自行加载数据。",
                )
        await self._send_text(target, ack)
        task = asyncio.create_task(self._analysis_wrapper(task_key, session_key, target, user_text))
        self._tasks[task_key] = RunningTask(task=task, request=user_text, started_at=time.time())

    async def _analysis_wrapper(
        self,
        task_key: str,
        session_key: SessionKey,
        target: str,
        user_text: str,
    ) -> None:
        try:
            await self._run_analysis(task_key, session_key, target, user_text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("iMessage analysis wrapper failed")
            await self._send_text(
                target,
                tr(user_text, en=f"Analysis failed: {exc}", zh=f"分析失败: {exc}"),
            )
        finally:
            self._tasks.pop(task_key, None)
            queued = self._pending.pop(task_key, [])
            if queued:
                coalesced = "\n\n".join(queued)
                await self._send_text(target, f"⏭ Starting {len(queued)} queued request(s)...")
                session = self._route_registry.get_or_create(session_key)
                await self._spawn_analysis(task_key, session_key, target, session, coalesced)

    async def _run_analysis(
        self,
        task_key: str,
        session_key: SessionKey,
        target: str,
        user_text: str,
    ) -> None:
        session = self._route_registry.get_or_create(session_key)
        full_request = self._build_full_request(session, user_text)
        llm_buf = ""
        last_progress_sent = 0.0

        async def progress_cb(msg: str) -> None:
            nonlocal last_progress_sent
            now = asyncio.get_running_loop().time()
            if now - last_progress_sent < _PROGRESS_GAP:
                return
            last_progress_sent = now
            await self._send_text(target, f"⚙️ {msg[:200]}")

        async def llm_chunk_cb(chunk: str) -> None:
            nonlocal llm_buf
            if chunk:
                llm_buf += chunk

        bridge = AgentBridge(session.agent, progress_cb=progress_cb, llm_chunk_cb=llm_chunk_cb)

        try:
            result = await bridge.run(full_request, session.adata)
        except asyncio.CancelledError:
            await self._send_text(
                target,
                tr(user_text, en="🚫 Analysis cancelled.", zh="🚫 已取消当前分析。"),
            )
            raise
        except Exception as exc:
            logger.exception("iMessage analysis failed")
            await self._send_text(
                target,
                tr(user_text, en=f"Analysis failed: {exc}", zh=f"分析失败: {exc}"),
            )
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
            a = session.adata
            adata_info = f"{a.n_obs:,} cells x {a.n_vars:,} genes" if a is not None else ""
            session.append_memory_log(
                request=user_text,
                summary=(result.summary or "分析完成"),
                adata_info=adata_info,
            )
        except Exception:
            pass

        if result.error:
            err_text = tr(
                user_text,
                en=f"Analysis error: {result.error}",
                zh=f"分析出错: {result.error}",
            )
            if result.diagnostics:
                hints = "\n".join(f"- {item}" for item in result.diagnostics[:4])
                err_text += tr(
                    user_text,
                    en=f"\n\nDiagnostics:\n{hints}",
                    zh=f"\n\n诊断:\n{hints}",
                )
            if llm_buf.strip():
                err_text += tr(
                    user_text,
                    en=f"\n\nModel output:\n{llm_buf[:1200]}",
                    zh=f"\n\n模型输出:\n{llm_buf[:1200]}",
                )
            await self._send_text(target, err_text)
            return

        for report in list(result.reports or []):
            for chunk in _text_chunks(report):
                await self._send_text(target, chunk)

        for index, figure in enumerate(list(result.figures or []), start=1):
            await self._send_bytes(
                target,
                figure,
                filename=f"figure_{index}.png",
                caption=f"Figure {index}",
            )

        for artifact in list(result.artifacts or []):
            await self._send_bytes(
                target,
                artifact.data,
                filename=artifact.filename or "artifact.bin",
                caption=f"Attachment: {artifact.filename}",
            )

        summary = _strip_local_paths((result.summary or "").strip())
        if not summary or summary.lower() in _BORING_SUMMARIES:
            if llm_buf.strip():
                summary = llm_buf[:1800]
            elif session.adata is not None:
                a = session.adata
                summary = tr(
                    user_text,
                    en=f"Analysis complete\n{a.n_obs:,} cells x {a.n_vars:,} genes",
                    zh=f"分析完成\n{a.n_obs:,} cells x {a.n_vars:,} genes",
                )
            else:
                summary = tr(user_text, en="Analysis complete", zh="分析完成")
        for chunk in _text_chunks(summary):
            await self._send_text(target, chunk)

    async def _handle_cancel(self, target: str, route: str) -> None:
        self._pending.pop(route, None)
        running = self._tasks.get(route)
        if not running or running.task.done():
            await self._send_text(target, "No analysis is currently running.")
            return
        running.task.cancel()
        await self._send_text(target, "Cancellation requested.")

    async def _handle_status(self, target: str, session: Any, route: str) -> None:
        lines: List[str] = []
        if session.adata is not None:
            a = session.adata
            lines.append(f"{a.n_obs:,} cells x {a.n_vars:,} genes")
            try:
                cols = ", ".join(list(a.obs.columns[:8]))
                if cols:
                    lines.append(f"obs: {cols}")
            except Exception:
                pass
        else:
            lines.append("No dataset loaded")
        try:
            kname = self._sm.get_active_kernel(session.user_id)
            lines.append(f"kernel: {kname}")
        except Exception:
            pass
        kst = session.kernel_status()
        if kst:
            lines.append(f"prompts: {kst.get('prompt_count', 0)}/{kst.get('max_prompts', '?')}")
        running = self._tasks.get(route)
        if running and not running.task.done():
            lines.append("Analysis running (/cancel available)")
        await self._send_text(target, "\n".join(lines))

    async def _handle_workspace(self, target: str, session: Any) -> None:
        ws = session.workspace
        h5ad_files = session.list_h5ad_files()
        agents_md = session.get_agents_md()
        today_log = session.memory_dir / f"{datetime.now().date()}.md"
        lines = [f"Workspace: {ws}", ""]
        if h5ad_files:
            lines.append(f"Data files ({len(h5ad_files)})")
            for file in h5ad_files[:10]:
                try:
                    mb = file.stat().st_size / 1_048_576
                    lines.append(f"- {file.name} ({mb:.1f} MB)")
                except OSError:
                    lines.append(f"- {file.name}")
        else:
            lines.append("Data files (empty)")
        lines += [
            "",
            f"AGENTS.md {'OK' if agents_md else '-'}",
            f"Today's memory {'OK' if today_log.exists() else '-'}",
        ]
        await self._send_text(target, "\n".join(lines))

    async def _handle_ls(self, target: str, session: Any, subpath: str) -> None:
        cmd = f"ls -lh {subpath}".strip() if subpath else "ls -lh"
        out = session.shell.exec(cmd, cwd=session.workspace)
        await self._send_text(target, f"$ {cmd}\n{out}")

    async def _handle_find(self, target: str, session: Any, pattern: str) -> None:
        pattern = (pattern or "").strip()
        if not pattern:
            await self._send_text(target, "Usage: /find <pattern>")
            return
        cmd = f"find . -name {pattern}"
        out = session.shell.exec(cmd, cwd=session.workspace)
        await self._send_text(target, f"$ {cmd}\n{out}")

    async def _handle_load(self, target: str, session: Any, filename: str) -> None:
        filename = (filename or "").strip()
        if not filename:
            await self._send_text(target, "Usage: /load <filename>")
            return
        await self._send_text(target, f"Loading {filename}...")
        try:
            adata = await asyncio.to_thread(session.load_from_workspace, filename)
        except Exception as exc:
            await self._send_text(target, f"Load failed: {exc}")
            return
        if adata is None:
            files = session.list_h5ad_files()
            hint = f"\nAvailable files: {'  '.join(f.name for f in files[:5])}" if files else ""
            await self._send_text(target, f"File not found: {filename}{hint}")
            return
        await self._send_text(
            target,
            f"Loaded successfully\n{adata.n_obs:,} cells x {adata.n_vars:,} genes\n{filename}",
        )

    async def _handle_shell(self, target: str, session: Any, cmd: str) -> None:
        cmd = (cmd or "").strip()
        if not cmd:
            await self._send_text(
                target,
                "Usage: /shell <command>\nAllowed: ls find cat head wc file du pwd tree",
            )
            return
        out = session.shell.exec(cmd, cwd=session.workspace)
        await self._send_text(target, f"$ {cmd}\n{out}")

    async def _handle_memory(self, target: str, session: Any) -> None:
        text = session.get_recent_memory_text()
        await self._send_text(target, f"Analysis history (last two days)\n\n{text}")

    async def _handle_usage(self, target: str, session: Any) -> None:
        usage = session.last_usage
        if usage is None:
            await self._send_text(target, "No usage data yet. Run an analysis first.")
            return

        def _attr(obj: Any, *names: str, default: str = "?") -> str:
            for name in names:
                value = getattr(obj, name, None)
                if value is not None:
                    return f"{value:,}" if isinstance(value, int) else str(value)
            return default

        lines = [
            "Token usage (most recent run)",
            f"Input: {_attr(usage, 'input_tokens')}",
            f"Output: {_attr(usage, 'output_tokens')}",
            f"Total: {_attr(usage, 'total_tokens')}",
        ]
        await self._send_text(target, "\n".join(lines))

    async def _handle_save(self, target: str, session: Any) -> None:
        if session.adata is None:
            await self._send_text(target, "No dataset loaded. Use /load or run an analysis first.")
            return
        await self._send_text(target, "Saving current.h5ad...")
        try:
            path = await asyncio.to_thread(session.save_adata)
            if not path or not Path(path).exists():
                await self._send_text(target, "Save failed. Please try again.")
                return
            await self._send_bytes(
                target,
                Path(path).read_bytes(),
                filename="current.h5ad",
                caption="current.h5ad",
            )
            a = session.adata
            await self._send_text(
                target,
                f"Saved current.h5ad\n{a.n_obs:,} cells x {a.n_vars:,} genes",
            )
        except Exception as exc:
            await self._send_text(target, f"Save failed: {exc}")

    async def _handle_kernel(self, target: str, session: Any, route: str, args: List[str]) -> None:
        if not args:
            kname = self._sm.get_active_kernel(session.user_id)
            kst = session.kernel_status()
            alive = "running" if kst.get("alive") else "stopped"
            await self._send_text(
                target,
                "Kernel status\n"
                f"Current: {kname}\n"
                f"State: {alive}\n"
                f"Prompts: {kst.get('prompt_count', 0)}/{kst.get('max_prompts', '?')}\n\n"
                "Subcommands: /kernel ls | /kernel new <name> | /kernel use <name>",
            )
            return
        sub = args[0].lower()
        if sub == "ls":
            names = self._sm.list_kernels(session.user_id)
            active = self._sm.get_active_kernel(session.user_id)
            await self._send_text(target, "\n".join(["kernels:"] + [f"{'*' if n == active else '-'} {n}" for n in names]))
            return
        if sub in {"new", "use"}:
            if len(args) < 2:
                await self._send_text(target, "Usage: /kernel new <name> or /kernel use <name>")
                return
            running = self._tasks.get(route)
            if running and not running.task.done():
                await self._send_text(target, "An analysis is currently running. Use /cancel or wait for it to finish.")
                return
            try:
                if sub == "new":
                    self._sm.create_kernel(session.user_id, args[1], switch=True)
                else:
                    self._sm.switch_kernel(session.user_id, args[1], create=False)
                await self._send_text(
                    target,
                    f"Switched to kernel: {self._sm.get_active_kernel(session.user_id)}",
                )
            except Exception as exc:
                await self._send_text(target, f"Kernel operation failed: {exc}")
            return
        await self._send_text(target, "Usage: /kernel | /kernel ls | /kernel new <name> | /kernel use <name>")

    async def _send_text(self, target: str, text: str) -> None:
        for chunk in _text_chunks(text):
            await self._client.send_message(target, chunk)

    async def _send_bytes(self, target: str, data: bytes, *, filename: str, caption: str) -> None:
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(prefix="jarvis-imessage-", suffix=suffix, delete=False) as handle:
            handle.write(data)
            temp_path = handle.name
        try:
            await self._client.send_message(target, caption or filename, file_path=temp_path, timeout=120.0)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def run_imessage_bot(
    *,
    session_manager: Any,
    cli_path: str = "imsg",
    db_path: Optional[str] = None,
    include_attachments: bool = False,
) -> None:
    bot = IMessageJarvisBot(
        session_manager=session_manager,
        cli_path=cli_path,
        db_path=db_path,
        include_attachments=include_attachments,
    )
    asyncio.run(bot.run())
