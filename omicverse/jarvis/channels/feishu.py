"""
Feishu webhook channel for OmicVerse Jarvis.

Minimal first implementation:
- URL verification handshake
- Receive text messages
- Deterministic session routing by chat/thread
- Send final text reply back to Feishu chat
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import requests

from ..agent_bridge import AgentBridge
from ..gateway.routing import GatewaySessionRegistry, SessionKey

logger = logging.getLogger("omicverse.jarvis.feishu")


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

    def send_text(self, chat_id: str, text: str) -> None:
        token = self._tenant_access_token()
        payload = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        resp = requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu send message failed: {data}")


def run_feishu_bot(
    *,
    app_id: str,
    app_secret: str,
    verification_token: str,
    session_manager: Any,
    host: str = "0.0.0.0",
    port: int = 8080,
    path: str = "/feishu/events",
) -> None:
    client = FeishuClient(app_id, app_secret)
    registry = GatewaySessionRegistry(session_manager)
    seen_message_ids: set[str] = set()
    seen_lock = threading.Lock()

    async def _process_text(chat_id: str, thread_id: Optional[str], text: str) -> None:
        key = SessionKey(
            channel="feishu",
            scope_type="chat",
            scope_id=str(chat_id),
            thread_id=(str(thread_id) if thread_id else None),
        )
        session = registry.get_or_create(key)

        lower = text.strip().lower()
        if lower == "/status":
            if session.adata is not None:
                a = session.adata
                client.send_text(chat_id, f"🔬 {a.n_obs:,} cells × {a.n_vars:,} genes")
            else:
                client.send_text(chat_id, "📭 暂无数据")
            return
        if lower == "/reset":
            session.reset()
            client.send_text(chat_id, "✅ 会话已重置，kernel 将在下一次请求时重建。")
            return

        bridge = AgentBridge(session.agent)
        result = await bridge.run(text, session.adata)
        if result.adata is not None:
            session.adata = result.adata
            try:
                session.save_adata()
            except Exception:
                pass
        if result.error:
            client.send_text(chat_id, f"❌ {result.error}")
            return
        summary = (result.summary or "").strip()
        if not summary:
            if session.adata is not None:
                a = session.adata
                summary = f"✅ 分析完成\n🔬 {a.n_obs:,} cells × {a.n_vars:,} genes"
            else:
                summary = "✅ 分析完成"
        client.send_text(chat_id, summary[:5000])

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

            # URL verification
            if body.get("type") == "url_verification":
                if body.get("token") != verification_token:
                    self.send_response(403)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"challenge": body.get("challenge", "")}).encode("utf-8"))
                return

            if body.get("token") != verification_token:
                self.send_response(403)
                self.end_headers()
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

            if msg.get("message_type") != "text":
                self.send_response(200)
                self.end_headers()
                return

            try:
                content = json.loads(msg.get("content", "{}"))
            except Exception:
                content = {}
            text = (content.get("text") or "").strip()
            chat_id = (event.get("chat_id") or msg.get("chat_id") or "").strip()
            thread_id = event.get("root_id") or msg.get("root_id")
            if text and chat_id:
                threading.Thread(
                    target=lambda: asyncio.run(_process_text(chat_id, thread_id, text)),
                    daemon=True,
                ).start()

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
