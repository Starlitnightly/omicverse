"""
QQ Bot channel for OmicVerse Jarvis.

Capabilities:
- C2C private chat messages
- Group AT messages (group chat where @bot is mentioned)
- Guild (channel) AT messages
- Long-connection WebSocket event subscription (no public URL needed)
- Streaming status updates during analysis
- Send images from analysis artifacts (requires --qq-image-host)
- Commands: /status /reset /cancel /kernel /workspace /ls /find /load /memory /usage /help

Authentication:
  POST https://bots.qq.com/app/getAppAccessToken
  Body: {"appId": "...", "clientSecret": "..."}

WebSocket opcodes:
  0  Dispatch (receive events)
  1  Heartbeat (send)
  2  Identify (send on new connection)
  6  Resume (send on reconnect)
  7  Reconnect (server requests reconnect)
  9  Invalid Session (server rejects identify)
  10 Hello (server sends heartbeat interval)
  11 Heartbeat ACK

Intents:
  PUBLIC_GUILD_MESSAGES = 1 << 30
  DIRECT_MESSAGE        = 1 << 12
  GROUP_AND_C2C         = 1 << 25
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..agent_bridge import AgentBridge
from ..gateway.routing import GatewaySessionRegistry, SessionKey

logger = logging.getLogger("omicverse.jarvis.qq")

# ── API constants ─────────────────────────────────────────────────────────────
_API_BASE = "https://api.sgroup.qq.com"
_TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"

# QQ message limits
_MAX_TEXT = 2000
_BORING = {"分析完成", "分析完成。", "task completed", "done", "完成"}

# Intents bitmask (full: private + group + guild)
_INTENT_PUBLIC_GUILD_MESSAGES = 1 << 30
_INTENT_DIRECT_MESSAGE = 1 << 12
_INTENT_GROUP_AND_C2C = 1 << 25
_INTENT_FULL = _INTENT_PUBLIC_GUILD_MESSAGES | _INTENT_DIRECT_MESSAGE | _INTENT_GROUP_AND_C2C
_INTENT_NO_DM = _INTENT_PUBLIC_GUILD_MESSAGES | _INTENT_GROUP_AND_C2C
_INTENT_GUILD_ONLY = _INTENT_PUBLIC_GUILD_MESSAGES | (1 << 1)

_INTENT_LEVELS = [_INTENT_FULL, _INTENT_NO_DM, _INTENT_GUILD_ONLY]

# WebSocket reconnect delays (seconds)
_RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60]

# WS opcodes
_OP_DISPATCH = 0
_OP_HEARTBEAT = 1
_OP_IDENTIFY = 2
_OP_RESUME = 6
_OP_RECONNECT = 7
_OP_INVALID_SESSION = 9
_OP_HELLO = 10
_OP_HEARTBEAT_ACK = 11


@dataclass
class RunningTask:
    task: asyncio.Task
    request: str
    started_at: float


# ── QQClient ──────────────────────────────────────────────────────────────────

class QQClient:
    """QQ Bot REST API client with token caching."""

    def __init__(self, app_id: str, client_secret: str, markdown: bool = False) -> None:
        self._app_id = app_id
        self._client_secret = client_secret
        self._markdown = markdown  # enable msg_type=2 markdown (requires bot permission)
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = threading.Lock()

    def _get_access_token(self) -> str:
        with self._lock:
            now = time.time()
            if self._token and now < self._expires_at - 60:
                return self._token
            resp = requests.post(
                _TOKEN_URL,
                json={"appId": self._app_id, "clientSecret": self._client_secret},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            token = data.get("access_token")
            if not token:
                raise RuntimeError(f"QQ Bot auth failed: {data}")
            expires_in = int(data.get("expires_in", 7200))
            self._token = token
            self._expires_at = now + expires_in
            return token

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"QQBot {self._get_access_token()}",
            "Content-Type": "application/json",
        }

    def _api(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{_API_BASE}{path}"
        resp = requests.request(method, url, headers=self._headers(), timeout=20, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def get_gateway_url(self) -> str:
        data = self._api("GET", "/gateway")
        return data["url"]

    # ── Send text ─────────────────────────────────────────────────────────────

    def _build_body(
        self,
        text: str,
        msg_id: Optional[str],
        msg_seq: int,
        *,
        use_markdown: bool = False,
    ) -> Dict[str, Any]:
        """Build a C2C/group message body, switching between plain text and markdown."""
        if use_markdown and self._markdown:
            body: Dict[str, Any] = {
                "markdown": {"content": text[:_MAX_TEXT]},
                "msg_type": 2,
                "msg_seq": msg_seq,
            }
        else:
            body = {"content": text[:_MAX_TEXT], "msg_type": 0, "msg_seq": msg_seq}
        if msg_id:
            body["msg_id"] = msg_id
        return body

    def _post_with_fallback(self, url: str, body: Dict[str, Any], msg_id: Optional[str]) -> Optional[str]:
        """POST a message body; if it fails with a msg_id, retry as proactive (no msg_id)."""
        try:
            data = self._api("POST", url, json=body)
            return data.get("id") or data.get("message_id")
        except requests.HTTPError as exc:
            if msg_id:
                logger.debug("QQ reply failed (%s), retrying as proactive", exc)
                body_p = {k: v for k, v in body.items() if k != "msg_id"}
                # If markdown also failed proactively, fall back to plain text
                if body_p.get("msg_type") == 2:
                    body_p["msg_type"] = 0
                    body_p.pop("markdown", None)
                    body_p["content"] = body.get("markdown", {}).get("content", "")
                try:
                    data = self._api("POST", url, json=body_p)
                    return data.get("id") or data.get("message_id")
                except Exception as e2:
                    logger.warning("QQ proactive send failed: %s", e2)
            else:
                logger.warning("QQ send failed: %s", exc)
        except Exception as exc:
            logger.warning("QQ send failed: %s", exc)
        return None

    def send_c2c(
        self,
        openid: str,
        text: str,
        msg_id: Optional[str] = None,
        msg_seq: int = 1,
        use_markdown: bool = False,
    ) -> Optional[str]:
        body = self._build_body(text, msg_id, msg_seq, use_markdown=use_markdown)
        return self._post_with_fallback(f"/v2/users/{openid}/messages", body, msg_id)

    def send_group(
        self,
        group_openid: str,
        text: str,
        msg_id: Optional[str] = None,
        msg_seq: int = 1,
        use_markdown: bool = False,
    ) -> Optional[str]:
        body = self._build_body(text, msg_id, msg_seq, use_markdown=use_markdown)
        return self._post_with_fallback(f"/v2/groups/{group_openid}/messages", body, msg_id)

    def send_guild(self, channel_id: str, text: str, msg_id: Optional[str] = None) -> Optional[str]:
        """Guild messages always support markdown natively."""
        body: Dict[str, Any] = {"content": text[:_MAX_TEXT]}
        if msg_id:
            body["msg_id"] = msg_id
        try:
            data = self._api("POST", f"/channels/{channel_id}/messages", json=body)
            return data.get("id") or data.get("message_id")
        except Exception as exc:
            logger.warning("QQ send_guild failed: %s", exc)
            return None

    def send_text(
        self,
        target: "QQTarget",
        text: str,
        msg_id: Optional[str] = None,
        msg_seq: int = 1,
    ) -> Optional[str]:
        """Send plain text to any target type."""
        if target.kind == "c2c":
            return self.send_c2c(target.id, text, msg_id=msg_id, msg_seq=msg_seq)
        elif target.kind == "group":
            return self.send_group(target.id, text, msg_id=msg_id, msg_seq=msg_seq)
        elif target.kind in {"guild", "dm"}:
            return self.send_guild(target.id, text, msg_id=msg_id)
        return None

    def send_markdown(
        self,
        target: "QQTarget",
        text: str,
        msg_id: Optional[str] = None,
        msg_seq: int = 1,
    ) -> Optional[str]:
        """Send markdown-formatted text (msg_type=2 for C2C/group; native for guild).
        Falls back to plain text if markdown is disabled or fails."""
        if target.kind == "c2c":
            return self.send_c2c(target.id, text, msg_id=msg_id, msg_seq=msg_seq, use_markdown=True)
        elif target.kind == "group":
            return self.send_group(target.id, text, msg_id=msg_id, msg_seq=msg_seq, use_markdown=True)
        elif target.kind in {"guild", "dm"}:
            return self.send_guild(target.id, text, msg_id=msg_id)
        return None

    # ── Upload & send image ───────────────────────────────────────────────────

    def upload_image_c2c(self, openid: str, image_url: str) -> Optional[str]:
        """Upload image for C2C. Returns file_info string."""
        try:
            data = self._api(
                "POST", f"/v2/users/{openid}/files",
                json={"file_type": 1, "url": image_url, "srv_send_msg": False},
            )
            return (data.get("file_info") or (data.get("data") or {}).get("file_info"))
        except Exception as exc:
            logger.warning("QQ upload_image_c2c failed: %s", exc)
            return None

    def upload_image_group(self, group_openid: str, image_url: str) -> Optional[str]:
        """Upload image for group. Returns file_info string."""
        try:
            data = self._api(
                "POST", f"/v2/groups/{group_openid}/files",
                json={"file_type": 1, "url": image_url, "srv_send_msg": False},
            )
            return (data.get("file_info") or (data.get("data") or {}).get("file_info"))
        except Exception as exc:
            logger.warning("QQ upload_image_group failed: %s", exc)
            return None

    def send_image_c2c(self, openid: str, file_info: str, msg_id: Optional[str] = None, msg_seq: int = 1) -> None:
        body: Dict[str, Any] = {"media": {"file_info": file_info}, "msg_type": 7, "msg_seq": msg_seq}
        if msg_id:
            body["msg_id"] = msg_id
        try:
            self._api("POST", f"/v2/users/{openid}/messages", json=body)
        except Exception as exc:
            logger.warning("QQ send_image_c2c failed: %s", exc)

    def send_image_group(self, group_openid: str, file_info: str, msg_id: Optional[str] = None, msg_seq: int = 1) -> None:
        body: Dict[str, Any] = {"media": {"file_info": file_info}, "msg_type": 7, "msg_seq": msg_seq}
        if msg_id:
            body["msg_id"] = msg_id
        try:
            self._api("POST", f"/v2/groups/{group_openid}/messages", json=body)
        except Exception as exc:
            logger.warning("QQ send_image_group failed: %s", exc)

    def send_image(self, target: "QQTarget", image_url: str, msg_id: Optional[str] = None, msg_seq: int = 1) -> None:
        """Upload and send image. image_url must be publicly accessible."""
        if target.kind == "c2c":
            fi = self.upload_image_c2c(target.id, image_url)
            if fi:
                self.send_image_c2c(target.id, fi, msg_id=msg_id, msg_seq=msg_seq)
        elif target.kind == "group":
            fi = self.upload_image_group(target.id, image_url)
            if fi:
                self.send_image_group(target.id, fi, msg_id=msg_id, msg_seq=msg_seq)
        elif target.kind in {"guild", "dm"}:
            # Guild supports markdown image embedding
            self.send_guild(target.id, f"![figure]({image_url})", msg_id=msg_id)

    def send_typing_c2c(self, openid: str) -> None:
        """Show 'bot is typing' indicator for C2C."""
        try:
            self._api("POST", f"/v2/users/{openid}/messages", json={"event_type": 1})
        except Exception:
            pass


# ── QQTarget ──────────────────────────────────────────────────────────────────

@dataclass
class QQTarget:
    """Represents a message destination: c2c, group, guild, or dm."""
    kind: str   # "c2c" | "group" | "guild" | "dm"
    id: str     # openid / group_openid / channel_id

    def route_key(self) -> str:
        return f"qq:{self.kind}:{self.id}"


# ── Image hosting server ──────────────────────────────────────────────────────

class _ImageServer:
    """Minimal HTTP server to temporarily serve image bytes for QQ upload."""

    def __init__(self, host: str, port: int, public_base: str) -> None:
        self._host = host
        self._port = port
        self._public_base = public_base.rstrip("/")
        self._images: Dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        store = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                name = self.path.lstrip("/")
                with store._lock:
                    data = store._images.get(name)
                if data is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt: str, *args: object) -> None:
                logger.debug("qq_image_server " + fmt, *args)

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("QQ image server started on %s:%s (public: %s)", self._host, self._port, self._public_base)

    def host_image(self, data: bytes, name: str) -> str:
        """Store image bytes and return the public URL."""
        with self._lock:
            self._images[name] = data
        return f"{self._public_base}/{name}"

    def remove(self, name: str) -> None:
        with self._lock:
            self._images.pop(name, None)


# ── QQRuntime ─────────────────────────────────────────────────────────────────

class QQRuntime:
    def __init__(
        self,
        client: QQClient,
        session_manager: Any,
        image_server: Optional[_ImageServer] = None,
    ) -> None:
        self._client = client
        self._registry = GatewaySessionRegistry(session_manager)
        self._sm = session_manager
        self._tasks: Dict[str, RunningTask] = {}
        self._pending: Dict[str, List[str]] = {}
        self._image_server = image_server
        # msg_seq tracking: QQ requires a unique, incrementing seq per msg_id
        # The ack message and all follow-up messages for the same msg_id share one counter
        self._msg_seqs: Dict[str, int] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Any) -> None:
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _session_key(self, target: QQTarget) -> SessionKey:
        return SessionKey(
            channel="qq",
            scope_type=target.kind,
            scope_id=target.id,
            thread_id=None,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
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
        return (
            "\n\n".join(ctx_parts) + f"\n\n[Current request]\n{text}"
            if ctx_parts else text
        )

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

    def _send_markdown(self, target: QQTarget, text: str, msg_id: Optional[str] = None) -> Optional[str]:
        """Send markdown content (mirrors Feishu send_markdown_card).
        Uses msg_type=2 when bot has markdown permission; falls back to plain text."""
        seq = self._alloc_seq(msg_id)
        return self._client.send_markdown(target, text, msg_id=msg_id, msg_seq=seq)

    def _alloc_seq(self, msg_id: Optional[str]) -> int:
        """Return the next msg_seq for this msg_id, incrementing the counter.
        QQ requires a unique incrementing seq for every message referencing the same msg_id.
        Without this, the second+ messages are silently dropped by QQ."""
        if not msg_id:
            return 1
        seq = self._msg_seqs.get(msg_id, 0) + 1
        self._msg_seqs[msg_id] = seq
        # Prevent unbounded growth (keep at most 500 tracked msg_ids)
        if len(self._msg_seqs) > 500:
            for k in list(self._msg_seqs.keys())[:100]:
                del self._msg_seqs[k]
        return seq

    def _send_text(self, target: QQTarget, text: str, msg_id: Optional[str] = None) -> Optional[str]:
        """Send text, automatically allocating the next msg_seq for this msg_id."""
        seq = self._alloc_seq(msg_id)
        return self._client.send_text(target, text, msg_id=msg_id, msg_seq=seq)

    # ── Quick chat ────────────────────────────────────────────────────────────

    async def _quick_chat(
        self,
        target: QQTarget,
        msg_id: Optional[str],
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
                "Reply in the same language the user uses.",
            ]
            if running_request:
                system_lines.append(f"\nCurrently running: {running_request[:300]}")
            if queued:
                system_lines.append(
                    "The user's message has been queued and will start after current analysis. Mention this."
                )
            if session.adata is not None:
                a = session.adata
                system_lines.append(f"Loaded data: {a.n_obs:,} cells x {a.n_vars:,} genes")
            try:
                mem = session.get_memory_context()
                if mem:
                    system_lines.append(f"\nHistory:\n{mem[:600]}")
            except Exception:
                pass
            messages = [
                {"role": "system", "content": "\n".join(system_lines)},
                {"role": "user", "content": user_text},
            ]
            response = await session.agent._llm.chat(messages, tools=None, tool_choice=None)
            reply = (response.content or "").strip() or "正在分析，请稍候..."
            for chunk in self._text_chunks(reply):
                await asyncio.to_thread(self._send_text, target, chunk, msg_id)
        except Exception as exc:
            logger.warning("QQ quick_chat failed: %s", exc)
            try:
                await asyncio.to_thread(self._send_text, target, "正在后台分析，请等待完成。", msg_id)
            except Exception:
                pass

    # ── Analysis pipeline ─────────────────────────────────────────────────────

    async def _spawn_analysis(
        self,
        target: QQTarget,
        msg_id: Optional[str],
        session: Any,
        user_text: str,
    ) -> None:
        full_request = self._build_full_request(session, user_text)
        route = target.route_key()
        task = asyncio.create_task(
            self._analysis_wrapper(target, msg_id, session, user_text, full_request)
        )
        self._tasks[route] = RunningTask(task=task, request=user_text, started_at=time.time())

    async def _analysis_wrapper(
        self,
        target: QQTarget,
        msg_id: Optional[str],
        session: Any,
        user_text: str,
        full_request: str,
    ) -> None:
        route = target.route_key()
        try:
            await self._run_analysis(target, msg_id, session, user_text, full_request)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("QQ analysis wrapper error")
            try:
                await asyncio.to_thread(self._send_text, target, f"分析异常: {exc}", msg_id)
            except Exception:
                pass
        finally:
            self._tasks.pop(route, None)
            queued = self._pending.pop(route, [])
            if queued:
                coalesced = "\n\n".join(queued)
                n = len(queued)
                try:
                    # Use None as msg_id for queued follow-ups: the original msg_id has
                    # almost certainly expired (>5 min) by now. Proactive send or new msg_id
                    # from the queued message will be used in the next _dispatch.
                    await asyncio.to_thread(
                        self._send_text, target, f"开始执行队列中的 {n} 条请求...", None
                    )
                except Exception:
                    pass
                asyncio.create_task(
                    self._spawn_analysis(target, None, session, coalesced)
                )

    async def _run_analysis(
        self,
        target: QQTarget,
        msg_id: Optional[str],
        session: Any,
        user_text: str,
        full_request: str,
    ) -> None:
        llm_buf = ""

        def _trim(text: str, max_len: int = 1800) -> str:
            if len(text) <= max_len:
                return text
            head = int(max_len * 0.6)
            tail = max_len - head - 20
            return text[:head] + "\n...\n" + text[-max(150, tail):]

        last_progress_send: float = 0.0
        _PROGRESS_GAP = 12.0  # throttle: send progress at most every 12 s

        async def llm_chunk_cb(chunk: str) -> None:
            nonlocal llm_buf
            if chunk:
                llm_buf += chunk

        async def progress_cb(msg: str) -> None:
            nonlocal last_progress_send
            # QQ can't edit messages; send throttled progress snapshots instead
            # (mirrors Feishu's edit_card live updates, just less frequent)
            now = time.monotonic()
            if now - last_progress_send < _PROGRESS_GAP:
                return
            last_progress_send = now
            try:
                await asyncio.to_thread(self._send_text, target, f"⚙️ {msg[:200]}", msg_id)
            except Exception:
                pass

        bridge = AgentBridge(session.agent, progress_cb=progress_cb, llm_chunk_cb=llm_chunk_cb)
        try:
            result = await bridge.run(full_request, session.adata)
        except asyncio.CancelledError:
            await asyncio.to_thread(self._send_text, target, "已取消当前分析。", msg_id)
            raise
        except Exception as exc:
            logger.exception("QQ analysis failed")
            await asyncio.to_thread(self._send_text, target, f"分析失败: {exc}", msg_id)
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
            err_text = f"分析出错: {result.error}"
            if result.diagnostics:
                hints = "\n".join(f"- {x}" for x in result.diagnostics[:4])
                err_text += f"\n\n诊断:\n{hints}"
            if llm_buf.strip():
                err_text += f"\n\n模型输出:\n{_trim(llm_buf, 1200)}"
            # Use markdown for error detail (mirrors Feishu edit_card red)
            await asyncio.to_thread(self._send_markdown, target, err_text, msg_id)
            return

        # Send reports — markdown rendering (mirrors Feishu send_markdown_card "📊 分析报告")
        for rep in list(result.reports or []):
            for chunk in self._text_chunks(rep, limit=_MAX_TEXT):
                await asyncio.to_thread(self._send_markdown, target, chunk, msg_id)

        # Send figures
        for i, fig in enumerate(list(result.figures or []), start=1):
            if self._image_server is None:
                logger.debug("QQ: no image server configured, skipping figure %s", i)
                continue
            try:
                img_name = f"fig_{int(time.time())}_{i}.png"
                img_url = self._image_server.host_image(fig, img_name)
                seq = self._alloc_seq(msg_id)
                await asyncio.to_thread(self._client.send_image, target, img_url, msg_id, seq)
                await asyncio.sleep(10)
                self._image_server.remove(img_name)
            except Exception:
                logger.warning("QQ: failed to send figure %s", i)

        # Send artifacts (file send not directly supported by QQ Bot API; log only)
        for art in list(result.artifacts or []):
            logger.info("QQ: artifact '%s' generated (file send not supported in QQ Bot API)", art.filename)

        # Send summary — markdown rendering (mirrors Feishu send_markdown_card green "✅ 分析完成")
        summary = self._strip_local_paths((result.summary or "").strip())
        if not summary or summary.lower() in _BORING:
            if llm_buf.strip():
                summary = _trim(llm_buf, max_len=1800)
            elif result.diagnostics:
                hints = "\n".join(f"- {x}" for x in result.diagnostics[:5])
                summary = f"未生成有效答复\n{hints}"
            elif session.adata is not None:
                a = session.adata
                summary = f"分析完成\n{a.n_obs:,} cells x {a.n_vars:,} genes"
            else:
                summary = "分析完成"
        for chunk in self._text_chunks(summary, limit=_MAX_TEXT):
            await asyncio.to_thread(self._send_markdown, target, chunk, msg_id)

    # ── Message dispatcher ────────────────────────────────────────────────────

    def handle_message(
        self,
        *,
        kind: str,          # "c2c" | "group" | "guild" | "dm"
        sender_id: str,
        content: str,
        msg_id: str,
        group_id: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> None:
        """Entry point called by the WebSocket gateway for each received message."""
        # Build target
        if kind == "group" and group_id:
            target = QQTarget(kind="group", id=group_id)
        elif kind == "guild" and channel_id:
            target = QQTarget(kind="guild", id=channel_id)
        elif kind == "dm" and channel_id:
            target = QQTarget(kind="dm", id=channel_id)
        else:
            target = QQTarget(kind="c2c", id=sender_id)

        self.submit(self._dispatch(target, msg_id, content))

    async def _dispatch(self, target: QQTarget, msg_id: str, raw_text: str) -> None:
        text = (raw_text or "").strip()
        # Strip @bot mention prefix that QQ injects (e.g. "<@!botid> ")
        text = re.sub(r'^<@!?\w+>\s*', '', text).strip()
        if not text:
            return

        sk = self._session_key(target)
        session = self._registry.get_or_create(sk)
        route = target.route_key()
        tokens = text.split()
        cmd = tokens[0].lower() if tokens else ""
        tail = text.split(None, 1)[1].strip() if len(tokens) > 1 else ""

        if cmd == "/cancel":
            await self._handle_cancel(target, msg_id, route)
            return
        if cmd in {"/start", "/help"}:
            await self._handle_help(target, msg_id)
            return
        if cmd == "/status":
            await self._handle_status(target, msg_id, session, route)
            return
        if cmd == "/reset":
            session.reset()
            await asyncio.to_thread(self._send_text, target, "会话已重置，kernel 将在下次请求时重建。", msg_id)
            return
        if cmd == "/kernel":
            await self._handle_kernel(target, msg_id, session, route, tokens[1:])
            return
        if cmd == "/workspace":
            await self._handle_workspace(target, msg_id, session)
            return
        if cmd == "/ls":
            await self._handle_ls(target, msg_id, session, tail)
            return
        if cmd == "/find":
            await self._handle_find(target, msg_id, session, tail)
            return
        if cmd == "/load":
            await self._handle_load(target, msg_id, session, tail)
            return
        if cmd == "/shell":
            await self._handle_shell(target, msg_id, session, tail)
            return
        if cmd == "/memory":
            await self._handle_memory(target, msg_id, session)
            return
        if cmd == "/usage":
            await self._handle_usage(target, msg_id, session)
            return
        if cmd == "/model":
            await self._handle_model(target, msg_id, tail)
            return
        if cmd == "/save":
            await self._handle_save(target, msg_id, session)
            return

        running = self._tasks.get(route)
        if running and not running.task.done():
            self._pending.setdefault(route, []).append(text)
            asyncio.create_task(
                self._quick_chat(target, msg_id, session, text,
                                 running_request=running.request, queued=True)
            )
            return

        # Ack before analysis
        if session.adata is not None:
            a = session.adata
            ack = f"收到请求，开始分析...\n当前数据: {a.n_obs:,} cells x {a.n_vars:,} genes"
        else:
            try:
                h5ad_files = session.list_h5ad_files()
            except Exception:
                h5ad_files = []
            if h5ad_files:
                names = "  ".join(f.name for f in h5ad_files[:5])
                ack = f"收到请求，开始分析...\n检测到文件: {names}\n使用 /load <文件名> 加载"
            else:
                ack = "收到请求，开始分析...\n未检测到数据，Agent 将自行加载"
        await asyncio.to_thread(self._send_text, target, ack, msg_id)

        await self._spawn_analysis(target, msg_id, session, text)

    # ── Command handlers ──────────────────────────────────────────────────────

    async def _handle_cancel(self, target: QQTarget, msg_id: Optional[str], route: str) -> None:
        self._pending.pop(route, None)
        running = self._tasks.get(route)
        if not running or running.task.done():
            await asyncio.to_thread(self._send_text, target, "当前没有正在运行的分析。", msg_id)
            return
        running.task.cancel()
        await asyncio.to_thread(self._send_text, target, "已发送取消信号。", msg_id)

    async def _handle_help(self, target: QQTarget, msg_id: Optional[str]) -> None:
        text = (
            "OmicVerse Jarvis\n"
            "----------------\n"
            "数据命令:\n"
            "/workspace - 查看工作区\n"
            "/ls [路径] - 列出文件\n"
            "/find <模式> - 搜索文件\n"
            "/load <文件名> - 加载数据\n"
            "/shell <命令> - 执行白名单命令\n\n"
            "会话命令:\n"
            "/kernel | /kernel ls | /kernel new 名称 | /kernel use 名称\n"
            "/memory - 分析历史\n"
            "/usage - token 用量\n"
            "/model [名称] - 查看/切换模型\n"
            "/status - 当前状态\n"
            "/save - 导出 current.h5ad\n"
            "/cancel - 取消分析\n"
            "/reset - 重置会话"
        )
        await asyncio.to_thread(self._send_text, target, text, msg_id)

    async def _handle_status(self, target: QQTarget, msg_id: Optional[str], session: Any, route: str) -> None:
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
            lines.append("暂无数据")
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
            lines.append("分析中（可 /cancel）")
        await asyncio.to_thread(self._send_text, target, "\n".join(lines), msg_id)

    async def _handle_workspace(self, target: QQTarget, msg_id: Optional[str], session: Any) -> None:
        ws = session.workspace
        h5ad_files = session.list_h5ad_files()
        agents_md = session.get_agents_md()
        today_log = session.memory_dir / f"{datetime.now().date()}.md"
        lines = [f"Workspace: {ws}", ""]
        if h5ad_files:
            lines.append(f"数据文件 ({len(h5ad_files)})")
            for f in h5ad_files[:10]:
                try:
                    mb = f.stat().st_size / 1_048_576
                    lines.append(f"- {f.name} ({mb:.1f} MB)")
                except OSError:
                    lines.append(f"- {f.name}")
        else:
            lines.append("数据文件 (空)")
        lines += [
            "",
            f"AGENTS.md {'OK' if agents_md else '-'}",
            f"今日记忆 {'OK' if today_log.exists() else '-'}",
        ]
        await asyncio.to_thread(self._send_text, target, "\n".join(lines), msg_id)

    async def _handle_ls(self, target: QQTarget, msg_id: Optional[str], session: Any, subpath: str) -> None:
        cmd = f"ls -lh {subpath}".strip() if subpath else "ls -lh"
        out = session.shell.exec(cmd, cwd=session.workspace)
        for chunk in self._text_chunks(f"$ {cmd}\n{out}", limit=1800):
            await asyncio.to_thread(self._send_text, target, chunk, msg_id)

    async def _handle_find(self, target: QQTarget, msg_id: Optional[str], session: Any, pattern: str) -> None:
        pattern = (pattern or "").strip()
        if not pattern:
            await asyncio.to_thread(self._send_text, target, "用法: /find <模式>", msg_id)
            return
        cmd = f"find . -name {pattern}"
        out = session.shell.exec(cmd, cwd=session.workspace)
        for chunk in self._text_chunks(f"$ {cmd}\n{out}", limit=1800):
            await asyncio.to_thread(self._send_text, target, chunk, msg_id)

    async def _handle_load(self, target: QQTarget, msg_id: Optional[str], session: Any, filename: str) -> None:
        filename = (filename or "").strip()
        if not filename:
            await asyncio.to_thread(self._send_text, target, "用法: /load <文件名>", msg_id)
            return
        await asyncio.to_thread(self._send_text, target, f"正在加载 {filename}...", msg_id)
        try:
            adata = await asyncio.to_thread(session.load_from_workspace, filename)
        except Exception as exc:
            await asyncio.to_thread(self._send_text, target, f"加载失败: {exc}", msg_id)
            return
        if adata is None:
            files = session.list_h5ad_files()
            hint = ""
            if files:
                hint = "\n可用文件: " + "  ".join(f.name for f in files[:5])
            await asyncio.to_thread(self._send_text, target, f"未找到 {filename}{hint}", msg_id)
            return
        await asyncio.to_thread(
            self._send_text, target,
            f"加载成功\n{adata.n_obs:,} cells x {adata.n_vars:,} genes\n{filename}",
            msg_id,
        )

    async def _handle_shell(self, target: QQTarget, msg_id: Optional[str], session: Any, cmd: str) -> None:
        cmd = (cmd or "").strip()
        if not cmd:
            await asyncio.to_thread(
                self._send_text, target,
                "用法: /shell <命令>\n允许: ls find cat head wc file du pwd tree",
                msg_id,
            )
            return
        out = session.shell.exec(cmd, cwd=session.workspace)
        for chunk in self._text_chunks(f"$ {cmd}\n{out}", limit=1800):
            await asyncio.to_thread(self._send_text, target, chunk, msg_id)

    async def _handle_memory(self, target: QQTarget, msg_id: Optional[str], session: Any) -> None:
        text = session.get_recent_memory_text()
        for chunk in self._text_chunks(f"分析历史（近两天）\n\n{text}", limit=1800):
            await asyncio.to_thread(self._send_text, target, chunk, msg_id)

    async def _handle_usage(self, target: QQTarget, msg_id: Optional[str], session: Any) -> None:
        usage = session.last_usage
        if usage is None:
            await asyncio.to_thread(self._send_text, target, "暂无用量数据，请先进行一次分析。", msg_id)
            return

        def _attr(obj: Any, *names: str) -> str:
            for n in names:
                v = getattr(obj, n, None)
                if v is not None:
                    return f"{v:,}" if isinstance(v, int) else str(v)
            return "?"

        lines = [
            "Token 用量（最近一次）",
            f"输入: {_attr(usage, 'input_tokens')}",
            f"输出: {_attr(usage, 'output_tokens')}",
            f"合计: {_attr(usage, 'total_tokens')}",
        ]
        await asyncio.to_thread(self._send_text, target, "\n".join(lines), msg_id)

    async def _handle_model(self, target: QQTarget, msg_id: Optional[str], model_name: str) -> None:
        model_name = (model_name or "").strip()
        if not model_name:
            cur = getattr(self._sm, "_model", "unknown")
            await asyncio.to_thread(
                self._send_text, target,
                f"当前模型: {cur}\n使用 /model <名称> 切换，切换后请 /reset 使新模型生效。",
                msg_id,
            )
            return
        self._sm._model = model_name
        await asyncio.to_thread(
            self._send_text, target,
            f"模型已切换为 {model_name}\n请 /reset 重启 kernel 使新模型生效。",
            msg_id,
        )

    async def _handle_save(self, target: QQTarget, msg_id: Optional[str], session: Any) -> None:
        if session.adata is None:
            await asyncio.to_thread(self._send_text, target, "没有数据，请先 /load 或完成分析。", msg_id)
            return
        await asyncio.to_thread(self._send_text, target, "正在保存 current.h5ad...", msg_id)
        try:
            path = await asyncio.to_thread(session.save_adata)
            if not path or not Path(path).exists():
                await asyncio.to_thread(self._send_text, target, "保存失败，请重试。", msg_id)
                return
            a = session.adata
            # QQ Bot doesn't support raw file uploads; note location only
            await asyncio.to_thread(
                self._send_text, target,
                f"已保存 current.h5ad\n{a.n_obs:,} cells x {a.n_vars:,} genes\n路径: {path}",
                msg_id,
            )
        except Exception as exc:
            await asyncio.to_thread(self._send_text, target, f"保存失败: {exc}", msg_id)

    async def _handle_kernel(
        self,
        target: QQTarget,
        msg_id: Optional[str],
        session: Any,
        route: str,
        args: List[str],
    ) -> None:
        if not args:
            kname = self._sm.get_active_kernel(session.user_id)
            kst = session.kernel_status()
            alive = "运行中" if kst.get("alive") else "未启动"
            text = (
                f"Kernel 状态\n"
                f"当前: {kname}\n"
                f"状态: {alive}\n"
                f"Prompts: {kst.get('prompt_count', 0)}/{kst.get('max_prompts', '?')}\n\n"
                "子命令: /kernel ls | /kernel new 名称 | /kernel use 名称"
            )
            await asyncio.to_thread(self._send_text, target, text, msg_id)
            return
        sub = args[0].lower()
        if sub == "ls":
            names = self._sm.list_kernels(session.user_id)
            active = self._sm.get_active_kernel(session.user_id)
            lines = ["kernels:"] + [f"{'*' if n == active else '-'} {n}" for n in names]
            await asyncio.to_thread(self._send_text, target, "\n".join(lines), msg_id)
            return
        if sub in {"new", "use"}:
            if len(args) < 2:
                await asyncio.to_thread(self._send_text, target, "用法: /kernel new 名称 或 /kernel use 名称", msg_id)
                return
            target_name = args[1]
            running = self._tasks.get(route)
            if running and not running.task.done():
                await asyncio.to_thread(self._send_text, target, "当前有分析在运行，请先 /cancel 或等待完成。", msg_id)
                return
            try:
                if sub == "new":
                    self._sm.create_kernel(session.user_id, target_name, switch=True)
                else:
                    self._sm.switch_kernel(session.user_id, target_name, create=False)
                await asyncio.to_thread(
                    self._send_text, target,
                    f"已切换到 kernel: {self._sm.get_active_kernel(session.user_id)}",
                    msg_id,
                )
            except Exception as exc:
                await asyncio.to_thread(self._send_text, target, f"kernel 操作失败: {exc}", msg_id)
            return
        await asyncio.to_thread(
            self._send_text, target,
            "用法: /kernel | /kernel ls | /kernel new 名称 | /kernel use 名称",
            msg_id,
        )


# ── WebSocket Gateway ─────────────────────────────────────────────────────────

def _run_gateway(
    *,
    app_id: str,
    client_secret: str,
    runtime: QQRuntime,
    client: QQClient,
    stop_event: threading.Event,
) -> None:
    """
    Run the QQ Bot WebSocket gateway in a blocking loop.
    Automatically reconnects with exponential backoff.
    Handles Identify / Resume, heartbeat, and all message events.
    """
    try:
        import websocket  # websocket-client
    except ImportError:
        raise RuntimeError(
            "QQ Bot channel requires 'websocket-client'. "
            "Install with: pip install websocket-client"
        )

    session_id: Optional[str] = None
    last_seq: Optional[int] = None
    intent_index = 0
    reconnect_attempt = 0

    while not stop_event.is_set():
        try:
            access_token = client._get_access_token()
            gateway_url = client.get_gateway_url()
            logger.info("QQ Bot connecting to %s", gateway_url)

            heartbeat_timer: Optional[threading.Timer] = None
            _ws_ref: list = []

            def _send_heartbeat() -> None:
                ws = _ws_ref[0] if _ws_ref else None
                if ws:
                    try:
                        ws.send(json.dumps({"op": _OP_HEARTBEAT, "d": last_seq}))
                        logger.debug("QQ heartbeat sent (seq=%s)", last_seq)
                    except Exception:
                        pass

            def on_open(ws: Any) -> None:
                _ws_ref.clear()
                _ws_ref.append(ws)
                logger.info("QQ Bot WebSocket connected")

            def on_message(ws: Any, raw: str) -> None:
                nonlocal session_id, last_seq, intent_index, heartbeat_timer, reconnect_attempt

                try:
                    payload = json.loads(raw)
                except Exception:
                    return

                op = payload.get("op")
                s = payload.get("s")
                t = payload.get("t")
                d = payload.get("d") or {}

                if s is not None:
                    last_seq = s

                if op == _OP_HELLO:
                    interval_ms = (d.get("heartbeat_interval") or 41250) / 1000.0
                    logger.info("QQ Hello received, heartbeat interval=%.1fs", interval_ms)

                    def _hb_loop() -> None:
                        nonlocal heartbeat_timer
                        _send_heartbeat()
                        heartbeat_timer = threading.Timer(interval_ms, _hb_loop)
                        heartbeat_timer.daemon = True
                        heartbeat_timer.start()

                    if heartbeat_timer:
                        heartbeat_timer.cancel()
                    heartbeat_timer = threading.Timer(interval_ms, _hb_loop)
                    heartbeat_timer.daemon = True
                    heartbeat_timer.start()

                    # Identify or Resume
                    if session_id and last_seq is not None:
                        logger.info("QQ Resuming session %s", session_id)
                        ws.send(json.dumps({
                            "op": _OP_RESUME,
                            "d": {
                                "token": f"QQBot {access_token}",
                                "session_id": session_id,
                                "seq": last_seq,
                            },
                        }))
                    else:
                        intent = _INTENT_LEVELS[min(intent_index, len(_INTENT_LEVELS) - 1)]
                        logger.info("QQ Identify with intents=%d", intent)
                        ws.send(json.dumps({
                            "op": _OP_IDENTIFY,
                            "d": {
                                "token": f"QQBot {access_token}",
                                "intents": intent,
                                "shard": [0, 1],
                            },
                        }))

                elif op == _OP_DISPATCH:
                    reconnect_attempt = 0  # successful dispatch resets backoff

                    if t == "READY":
                        session_id = d.get("session_id")
                        logger.info("QQ Ready, session_id=%s", session_id)

                    elif t == "C2C_MESSAGE_CREATE":
                        openid = (d.get("author") or {}).get("user_openid", "")
                        content = d.get("content", "")
                        msg_id = d.get("id", "")
                        if openid and content.strip():
                            runtime.handle_message(
                                kind="c2c",
                                sender_id=openid,
                                content=content,
                                msg_id=msg_id,
                            )

                    elif t == "GROUP_AT_MESSAGE_CREATE":
                        author = d.get("author") or {}
                        member_openid = author.get("member_openid", "")
                        group_openid = d.get("group_openid", "")
                        content = d.get("content", "")
                        msg_id = d.get("id", "")
                        if content.strip():
                            runtime.handle_message(
                                kind="group",
                                sender_id=member_openid,
                                content=content,
                                msg_id=msg_id,
                                group_id=group_openid,
                            )

                    elif t == "AT_MESSAGE_CREATE":
                        author = d.get("author") or {}
                        sender_id = author.get("id", "")
                        channel_id = d.get("channel_id", "")
                        content = d.get("content", "")
                        msg_id = d.get("id", "")
                        if content.strip():
                            runtime.handle_message(
                                kind="guild",
                                sender_id=sender_id,
                                content=content,
                                msg_id=msg_id,
                                channel_id=channel_id,
                            )

                    elif t == "DIRECT_MESSAGE_CREATE":
                        author = d.get("author") or {}
                        sender_id = author.get("id", "")
                        guild_id = d.get("guild_id", "")
                        content = d.get("content", "")
                        msg_id = d.get("id", "")
                        if content.strip():
                            runtime.handle_message(
                                kind="dm",
                                sender_id=sender_id,
                                content=content,
                                msg_id=msg_id,
                                channel_id=guild_id,
                            )

                elif op == _OP_HEARTBEAT_ACK:
                    logger.debug("QQ Heartbeat ACK")

                elif op == _OP_RECONNECT:
                    logger.info("QQ Server requested reconnect")
                    ws.close()

                elif op == _OP_INVALID_SESSION:
                    can_resume = bool(d)
                    logger.warning("QQ Invalid session, can_resume=%s", can_resume)
                    if not can_resume:
                        session_id = None
                        last_seq = None
                        # Downgrade intents on repeated failures
                        intent_index = min(intent_index + 1, len(_INTENT_LEVELS) - 1)
                    ws.close()

            def on_error(ws: Any, error: Any) -> None:
                logger.warning("QQ WebSocket error: %s", error)

            def on_close(ws: Any, code: Any, reason: Any) -> None:
                nonlocal heartbeat_timer
                logger.info("QQ WebSocket closed (code=%s reason=%s)", code, reason)
                if heartbeat_timer:
                    heartbeat_timer.cancel()
                    heartbeat_timer = None

            ws_app = websocket.WebSocketApp(
                gateway_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws_app.run_forever()

        except Exception as exc:
            logger.error("QQ gateway error: %s", exc)

        if stop_event.is_set():
            break

        delay = _RECONNECT_DELAYS[min(reconnect_attempt, len(_RECONNECT_DELAYS) - 1)]
        reconnect_attempt += 1
        logger.info("QQ reconnecting in %ss (attempt %s)...", delay, reconnect_attempt)
        stop_event.wait(timeout=delay)


# ── Public entry point ────────────────────────────────────────────────────────

def run_qq_bot(
    *,
    app_id: str,
    client_secret: str,
    session_manager: Any,
    markdown: bool = False,
    image_host: Optional[str] = None,
    image_server_bind: str = "0.0.0.0",
    image_server_port: int = 8081,
) -> None:
    """
    Start the QQ Bot channel and block until interrupted.

    Parameters
    ----------
    app_id : str
        QQ Bot AppID from the QQ Open Platform.
    client_secret : str
        QQ Bot ClientSecret / AppSecret.
    session_manager : SessionManager
        OmicVerse Jarvis session manager.
    markdown : bool
        Enable markdown reply format (msg_type=2). Requires the bot to have
        markdown message permission on the QQ Open Platform (default: False).
        Analysis reports and summaries use markdown; plain text is the fallback.
    image_host : str, optional
        Public base URL for the image hosting server (e.g. ``http://1.2.3.4:8081``).
        Required if you want analysis figures to be sent as QQ images.
        If omitted, figures are generated but not forwarded to QQ.
    image_server_bind : str
        Local interface to bind the image HTTP server (default: ``0.0.0.0``).
    image_server_port : int
        Port for the image HTTP server (default: 8081).
    """
    client = QQClient(app_id, client_secret, markdown=markdown)

    img_server: Optional[_ImageServer] = None
    if image_host:
        img_server = _ImageServer(
            host=image_server_bind,
            port=image_server_port,
            public_base=image_host,
        )
        img_server.start()

    runtime = QQRuntime(client, session_manager, image_server=img_server)

    stop_event = threading.Event()
    try:
        logger.info("QQ Bot Jarvis starting (app_id=%s)", app_id)
        _run_gateway(
            app_id=app_id,
            client_secret=client_secret,
            runtime=runtime,
            client=client,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        logger.info("QQ Bot shutting down")
        stop_event.set()
