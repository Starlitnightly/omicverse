"""
Feishu webhook channel for OmicVerse Jarvis (feature-parity focused).

Capabilities:
- URL verification handshake
- Deterministic session routing by chat/thread
- Streaming draft updates (edit message)
- Send images/files from analysis artifacts
- Receive .h5ad file messages, save into workspace, and auto-load
- Commands: /status /reset /cancel /kernel [ls|new|use]
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..agent_bridge import AgentBridge
from ..gateway.routing import GatewaySessionRegistry, SessionKey

logger = logging.getLogger("omicverse.jarvis.feishu")

_MAX_TEXT = 4800
_BORING = {"分析完成", "分析完成。", "task completed", "done", "完成"}


@dataclass
class RunningTask:
    task: asyncio.Task
    request: str
    started_at: float


class FeishuClient:
    def __init__(self, app_id: str, app_secret: str) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def _tenant_access_token(self) -> str:
        now = time.time()
        if self._token and now < self._expires_at - 60:
            return self._token
        resp = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu auth failed: {data}")
        self._token = data["tenant_access_token"]
        self._expires_at = now + int(data.get("expire", 7200))
        return self._token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._tenant_access_token()}"}

    def send_text(self, chat_id: str, text: str) -> Optional[str]:
        payload = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text[:_MAX_TEXT]}, ensure_ascii=False),
        }
        resp = requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            headers=self._headers(),
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu send text failed: {data}")
        return ((data.get("data") or {}).get("message_id"))

    def edit_text(self, message_id: str, text: str) -> bool:
        payload = {
            "msg_type": "text",
            "content": json.dumps({"text": text[:_MAX_TEXT]}, ensure_ascii=False),
        }
        resp = requests.patch(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}",
            headers=self._headers(),
            json=payload,
            timeout=20,
        )
        if resp.status_code >= 400:
            return False
        data = resp.json()
        return data.get("code") == 0

    def _send_message(self, chat_id: str, msg_type: str, content: dict) -> Optional[str]:
        payload = {
            "receive_id": chat_id,
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
        }
        resp = requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu send {msg_type} failed: {data}")
        return ((data.get("data") or {}).get("message_id"))

    def upload_image(self, image_bytes: bytes, filename: str = "figure.png") -> str:
        files = {
            "image_type": (None, "message"),
            "image": (filename, image_bytes, "application/octet-stream"),
        }
        resp = requests.post(
            "https://open.feishu.cn/open-apis/im/v1/images",
            headers=self._headers(),
            files=files,
            timeout=40,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu upload image failed: {data}")
        image_key = ((data.get("data") or {}).get("image_key"))
        if not image_key:
            raise RuntimeError("Feishu upload image missing image_key")
        return image_key

    def send_image_bytes(self, chat_id: str, image_bytes: bytes, filename: str = "figure.png") -> None:
        image_key = self.upload_image(image_bytes, filename=filename)
        self._send_message(chat_id, "image", {"image_key": image_key})

    def upload_file(self, data: bytes, filename: str) -> str:
        files = {
            "file_type": (None, "stream"),
            "file_name": (None, filename),
            "file": (filename, data, "application/octet-stream"),
        }
        resp = requests.post(
            "https://open.feishu.cn/open-apis/im/v1/files",
            headers=self._headers(),
            files=files,
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"Feishu upload file failed: {body}")
        file_key = ((body.get("data") or {}).get("file_key"))
        if not file_key:
            raise RuntimeError("Feishu upload file missing file_key")
        return file_key

    def send_file_bytes(self, chat_id: str, data: bytes, filename: str) -> None:
        file_key = self.upload_file(data, filename=filename)
        self._send_message(chat_id, "file", {"file_key": file_key})

    def download_file(self, file_key: str) -> bytes:
        resp = requests.get(
            f"https://open.feishu.cn/open-apis/im/v1/files/{file_key}/download",
            headers=self._headers(),
            timeout=120,
        )
        resp.raise_for_status()
        return resp.content


class FeishuRuntime:
    def __init__(self, client: FeishuClient, session_manager: Any) -> None:
        self._client = client
        self._registry = GatewaySessionRegistry(session_manager)
        self._sm = session_manager
        self._tasks: Dict[str, RunningTask] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: asyncio.Future) -> None:
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    @staticmethod
    def _parse_command(text: str) -> List[str]:
        return (text or "").strip().split()

    def _session_key(self, chat_id: str, thread_id: Optional[str]) -> SessionKey:
        return SessionKey(
            channel="feishu",
            scope_type="chat",
            scope_id=str(chat_id),
            thread_id=(str(thread_id) if thread_id else None),
        )

    async def handle_text(self, chat_id: str, thread_id: Optional[str], text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        sk = self._session_key(chat_id, thread_id)
        route = sk.as_key()
        session = self._registry.get_or_create(sk)

        tokens = self._parse_command(text)
        cmd = tokens[0].lower() if tokens else ""

        if cmd == "/cancel":
            await self._handle_cancel(chat_id, route)
            return
        if cmd == "/status":
            await self._handle_status(chat_id, session, route)
            return
        if cmd == "/reset":
            session.reset()
            self._client.send_text(chat_id, "✅ 会话已重置，kernel 将在下一次请求时重建。")
            return
        if cmd == "/kernel":
            await self._handle_kernel(chat_id, session, route, tokens[1:])
            return

        running = self._tasks.get(route)
        if running and not running.task.done():
            self._client.send_text(
                chat_id,
                "⏳ 当前已有分析在运行。发送 /cancel 可取消，或等待完成后再发新任务。",
            )
            return

        draft_id = self._client.send_text(chat_id, "💭 思考中…")
        task = asyncio.create_task(self._run_analysis(chat_id, route, session, text, draft_id))
        self._tasks[route] = RunningTask(task=task, request=text, started_at=time.time())
        task.add_done_callback(lambda _: self._tasks.pop(route, None))

    async def handle_file(
        self,
        chat_id: str,
        thread_id: Optional[str],
        file_key: str,
        file_name: str,
    ) -> None:
        sk = self._session_key(chat_id, thread_id)
        session = self._registry.get_or_create(sk)
        safe_name = Path(file_name or "uploaded.bin").name
        if not safe_name.lower().endswith(".h5ad"):
            self._client.send_text(chat_id, f"⚠️ 仅支持 .h5ad 文件，已收到：{safe_name}")
            return
        try:
            data = await asyncio.to_thread(self._client.download_file, file_key)
            target = session.workspace / safe_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            loaded = await asyncio.to_thread(session.load_from_workspace, safe_name)
            if loaded is not None:
                self._client.send_text(
                    chat_id,
                    f"✅ 已上传并加载 {safe_name}\n🔬 {loaded.n_obs:,} cells × {loaded.n_vars:,} genes",
                )
            else:
                self._client.send_text(chat_id, f"✅ 已上传 {safe_name}，但自动加载失败。")
        except Exception as exc:
            logger.exception("Feishu file handling failed")
            self._client.send_text(chat_id, f"❌ 文件处理失败: {exc}")

    async def _run_analysis(
        self,
        chat_id: str,
        route: str,
        session: Any,
        user_text: str,
        draft_id: Optional[str],
    ) -> None:
        llm_buf = ""
        last_progress = ""
        last_edit = 0.0
        edit_gap = 1.2

        def _trim(text: str, max_len: int = 2200) -> str:
            if len(text) <= max_len:
                return text
            head = int(max_len * 0.6)
            tail = max_len - head - 30
            return text[:head] + "\n...\n" + text[-max(200, tail):]

        def _draft_text() -> str:
            if llm_buf.strip():
                if last_progress:
                    return f"🔄 {last_progress[:140]}\n\n💭 {_trim(llm_buf)}"
                return f"💭 {_trim(llm_buf)}"
            return "💭 思考中…"

        async def _edit(force: bool = False) -> None:
            nonlocal last_edit
            if not draft_id:
                return
            now = time.monotonic()
            if (not force) and (now - last_edit < edit_gap):
                return
            text = _draft_text()
            ok = await asyncio.to_thread(self._client.edit_text, draft_id, text)
            if ok:
                last_edit = now

        async def llm_chunk_cb(chunk: str) -> None:
            nonlocal llm_buf
            if not chunk:
                return
            llm_buf += chunk
            await _edit(force=False)

        async def progress_cb(msg: str) -> None:
            nonlocal last_progress
            last_progress = msg
            await _edit(force=True)

        bridge = AgentBridge(session.agent, progress_cb=progress_cb, llm_chunk_cb=llm_chunk_cb)
        try:
            result = await bridge.run(user_text, session.adata)
        except asyncio.CancelledError:
            if draft_id:
                await asyncio.to_thread(self._client.edit_text, draft_id, "🚫 已取消当前分析。")
            raise
        except Exception as exc:
            logger.exception("Feishu analysis failed")
            if draft_id:
                await asyncio.to_thread(self._client.edit_text, draft_id, f"❌ 分析失败: {exc}")
            else:
                await asyncio.to_thread(self._client.send_text, chat_id, f"❌ 分析失败: {exc}")
            return

        if result.adata is not None:
            session.adata = result.adata
            try:
                session.save_adata()
            except Exception:
                pass
        if result.usage is not None:
            session.last_usage = result.usage
        try:
            a = session.adata
            adata_info = f"{a.n_obs:,} cells × {a.n_vars:,} genes" if a is not None else ""
            session.append_memory_log(
                request=user_text,
                summary=(result.summary or "分析完成"),
                adata_info=adata_info,
            )
        except Exception:
            pass

        if result.error:
            if draft_id:
                await asyncio.to_thread(self._client.edit_text, draft_id, f"❌ {result.error}")
            else:
                await asyncio.to_thread(self._client.send_text, chat_id, f"❌ {result.error}")
            return

        if draft_id:
            await asyncio.to_thread(self._client.edit_text, draft_id, "✅ 分析完成，正在发送结果…")

        for rep in list(result.reports or []):
            for chunk in self._text_chunks(rep):
                await asyncio.to_thread(self._client.send_text, chat_id, chunk)

        for i, fig in enumerate(list(result.figures or []), start=1):
            try:
                await asyncio.to_thread(
                    self._client.send_image_bytes, chat_id, fig, f"figure_{i}.png"
                )
            except Exception:
                try:
                    await asyncio.to_thread(
                        self._client.send_file_bytes, chat_id, fig, f"figure_{i}.png"
                    )
                except Exception:
                    logger.warning("Failed to send figure %s", i)

        for art in list(result.artifacts or []):
            try:
                await asyncio.to_thread(
                    self._client.send_file_bytes, chat_id, art.data, art.filename
                )
            except Exception:
                logger.warning("Failed to send artifact %s", art.filename)

        summary = (result.summary or "").strip()
        if not summary or summary.lower() in _BORING:
            if session.adata is not None:
                a = session.adata
                summary = f"✅ 分析完成\n🔬 {a.n_obs:,} cells × {a.n_vars:,} genes"
            else:
                summary = "✅ 分析完成"
        for chunk in self._text_chunks(summary):
            await asyncio.to_thread(self._client.send_text, chat_id, chunk)
        if draft_id:
            await asyncio.to_thread(self._client.edit_text, draft_id, "✅ 分析完成")

    async def _handle_cancel(self, chat_id: str, route: str) -> None:
        running = self._tasks.get(route)
        if not running or running.task.done():
            self._client.send_text(chat_id, "ℹ️ 当前没有正在运行的分析。")
            return
        running.task.cancel()
        self._client.send_text(chat_id, "⏹ 已发送取消信号。")

    async def _handle_status(self, chat_id: str, session: Any, route: str) -> None:
        lines: List[str] = []
        if session.adata is not None:
            a = session.adata
            lines.append(f"🔬 {a.n_obs:,} cells × {a.n_vars:,} genes")
        else:
            lines.append("📭 暂无数据")
        try:
            kname = self._sm.get_active_kernel(session.user_id)
            lines.append(f"🧩 kernel: {kname}")
        except Exception:
            pass
        kst = session.kernel_status()
        if kst:
            lines.append(f"💬 prompts: {kst.get('prompt_count', 0)}/{kst.get('max_prompts', '?')}")
            if kst.get("session_id"):
                lines.append(f"🆔 session: {kst.get('session_id')}")
        running = self._tasks.get(route)
        if running and not running.task.done():
            lines.append("⚙️ 分析中（可 /cancel）")
        self._client.send_text(chat_id, "\n".join(lines))

    async def _handle_kernel(self, chat_id: str, session: Any, route: str, args: List[str]) -> None:
        if not args:
            kname = self._sm.get_active_kernel(session.user_id)
            kst = session.kernel_status()
            alive = "🟢" if kst.get("alive") else "🔴"
            text = (
                "⚙️ Kernel 状态\n"
                "--------------------\n"
                f"🧩 当前: {kname}\n"
                f"{alive} {'运行中' if kst.get('alive') else '未启动/已关闭'}\n"
                f"💬 Prompts: {kst.get('prompt_count', 0)} / {kst.get('max_prompts', '?')}\n"
                f"🆔 Session: {kst.get('session_id') or '-'}\n\n"
                "子命令: /kernel ls | /kernel new 名称 | /kernel use 名称"
            )
            self._client.send_text(chat_id, text)
            return
        sub = args[0].lower()
        if sub == "ls":
            names = self._sm.list_kernels(session.user_id)
            active = self._sm.get_active_kernel(session.user_id)
            lines = ["🧩 kernels:"]
            for n in names:
                lines.append(f"{'*' if n == active else '-'} {n}")
            self._client.send_text(chat_id, "\n".join(lines))
            return
        if sub in {"new", "use"}:
            if len(args) < 2:
                self._client.send_text(chat_id, "用法: /kernel new 名称 或 /kernel use 名称")
                return
            target = args[1]
            running = self._tasks.get(route)
            if running and not running.task.done():
                self._client.send_text(chat_id, "⏳ 当前有分析在运行，请先 /cancel 或等待完成。")
                return
            try:
                if sub == "new":
                    self._sm.create_kernel(session.user_id, target, switch=True)
                else:
                    self._sm.switch_kernel(session.user_id, target, create=False)
                self._client.send_text(
                    chat_id, f"✅ 已切换到 kernel: {self._sm.get_active_kernel(session.user_id)}"
                )
            except Exception as exc:
                self._client.send_text(chat_id, f"❌ kernel 操作失败: {exc}")
            return
        self._client.send_text(chat_id, "用法: /kernel | /kernel ls | /kernel new 名称 | /kernel use 名称")

    @staticmethod
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
            else:
                pos = 0
                while pos < len(para):
                    chunks.append(para[pos:pos + limit])
                    pos += limit
                buf = ""
        if buf:
            chunks.append(buf)
        return chunks


def run_feishu_bot(
    *,
    app_id: str,
    app_secret: str,
    session_manager: Any,
    host: str = "0.0.0.0",
    port: int = 8080,
    path: str = "/feishu/events",
) -> None:
    client = FeishuClient(app_id, app_secret)
    runtime = FeishuRuntime(client, session_manager)
    seen_message_ids: set[str] = set()
    seen_lock = threading.Lock()

    class FeishuHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != path:
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                body = json.loads(raw)
            except Exception:
                self.send_response(400)
                self.end_headers()
                return

            if body.get("type") == "url_verification":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"challenge": body.get("challenge", "")}).encode("utf-8")
                )
                return

            event = body.get("event", {})
            header = body.get("header", {})
            if header.get("event_type") != "im.message.receive_v1":
                self.send_response(200)
                self.end_headers()
                return

            msg = event.get("message", {})
            message_id = msg.get("message_id")
            with seen_lock:
                if message_id and message_id in seen_message_ids:
                    self.send_response(200)
                    self.end_headers()
                    return
                if message_id:
                    seen_message_ids.add(message_id)

            message_type = (msg.get("message_type") or "").strip().lower()
            chat_id = (event.get("chat_id") or msg.get("chat_id") or "").strip()
            thread_id = event.get("root_id") or msg.get("root_id")
            if not chat_id:
                self.send_response(200)
                self.end_headers()
                return

            try:
                content = json.loads(msg.get("content", "{}"))
            except Exception:
                content = {}

            if message_type == "text":
                text = (content.get("text") or "").strip()
                if text:
                    runtime.submit(runtime.handle_text(chat_id, thread_id, text))
            elif message_type == "file":
                file_key = (content.get("file_key") or "").strip()
                file_name = (content.get("file_name") or content.get("name") or "upload.bin").strip()
                if file_key:
                    runtime.submit(runtime.handle_file(chat_id, thread_id, file_key, file_name))
            elif message_type == "image":
                runtime.submit(
                    runtime.handle_text(chat_id, thread_id, "收到图片。若需分析请上传 .h5ad 或发送文字指令。")
                )

            self.send_response(200)
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            logger.debug("feishu_http " + fmt, *args)

    server = ThreadingHTTPServer((host, port), FeishuHandler)
    logger.info("Feishu Jarvis webhook listening on http://%s:%s%s", host, port, path)
    try:
        server.serve_forever()
    finally:
        server.server_close()
