"""
Feishu webhook channel for OmicVerse Jarvis (feature-parity focused).

Capabilities:
- URL verification handshake
- Websocket long-connection event subscription
- Deterministic session routing by chat/thread
- Streaming draft updates (edit message)
- Send images/files from analysis artifacts
- Receive .h5ad file messages, save into workspace, and auto-load
- Commands: /status /reset /cancel /kernel [ls|new|use]
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests

from ..agent_bridge import AgentBridge
from ..gateway.routing import GatewaySessionRegistry, SessionKey
from ..model_help import render_model_help
from ..runtime import ConversationRoute

logger = logging.getLogger("omicverse.jarvis.feishu")

_MAX_TEXT = 4800
_BORING = {"分析完成", "分析完成。", "task completed", "done", "完成"}
_MAX_BODY_BYTES = 2 * 1024 * 1024
_DEDUP_TTL_SECONDS = 24 * 60 * 60
_DEDUP_MAX_ENTRIES = 10000
_FEISHU_EVENT_MESSAGE_RECEIVE = "im.message.receive_v1"


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

    def send_post(self, chat_id: str, text: str, title: str = "") -> Optional[str]:
        """Send a post message with basic markdown (bold, italic, links, inline code)."""
        zh_content: Dict[str, Any] = {
            "content": [[{"tag": "md", "text": text[:3000]}]]
        }
        if title:
            zh_content["title"] = title[:80]
        return self._send_message(chat_id, "post", {"zh_cn": zh_content})

    def send_markdown_card(
        self,
        chat_id: str,
        content: str,
        title: str = "",
        color: str = "blue",
    ) -> Optional[str]:
        """Send an interactive card with full markdown rendering (code blocks, tables)."""
        elements = [{"tag": "markdown", "content": content[:5000]}]
        card: Dict[str, Any] = {"schema": "2.0", "body": {"elements": elements}}
        if title:
            card["header"] = {
                "title": {"tag": "plain_text", "content": title[:60]},
                "template": color,
            }
        return self._send_message(chat_id, "interactive", card)

    def edit_card(
        self,
        message_id: str,
        content: str,
        title: str = "",
        color: str = "blue",
    ) -> bool:
        """Edit an existing interactive card message with updated markdown content."""
        elements = [{"tag": "markdown", "content": content[:5000]}]
        card: Dict[str, Any] = {"schema": "2.0", "body": {"elements": elements}}
        if title:
            card["header"] = {
                "title": {"tag": "plain_text", "content": title[:60]},
                "template": color,
            }
        payload = {
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        }
        resp = requests.patch(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}",
            headers=self._headers(),
            json=payload,
            timeout=20,
        )
        if resp.status_code >= 400:
            return False
        return resp.json().get("code") == 0


class FeishuDeduper:
    """Memory + disk dedupe cache for webhook event/message IDs."""

    def __init__(
        self,
        store_path: Path,
        *,
        ttl_seconds: int = _DEDUP_TTL_SECONDS,
        max_entries: int = _DEDUP_MAX_ENTRIES,
    ) -> None:
        self._store_path = store_path
        self._ttl_seconds = max(0, int(ttl_seconds))
        self._max_entries = max(100, int(max_entries))
        self._lock = threading.Lock()
        self._cache: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        try:
            if not self._store_path.exists():
                return
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                now = time.time()
                for k, v in raw.items():
                    if isinstance(k, str) and isinstance(v, (int, float)):
                        if self._ttl_seconds <= 0 or (now - float(v)) < self._ttl_seconds:
                            self._cache[k] = float(v)
                self._prune_locked(now)
        except Exception:
            logger.debug("Failed to load Feishu dedupe cache", exc_info=True)

    def _prune_locked(self, now: Optional[float] = None) -> None:
        ts_now = now if now is not None else time.time()
        if self._ttl_seconds > 0:
            expired = [
                key for key, ts in self._cache.items() if (ts_now - ts) >= float(self._ttl_seconds)
            ]
            for key in expired:
                self._cache.pop(key, None)
        if len(self._cache) > self._max_entries:
            keep = sorted(self._cache.items(), key=lambda item: item[1], reverse=True)[: self._max_entries]
            self._cache = dict(keep)

    def _persist_locked(self) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._store_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._store_path)
        except Exception:
            logger.debug("Failed to persist Feishu dedupe cache", exc_info=True)

    def seen_or_record(self, key: str) -> bool:
        token = (key or "").strip()
        if not token:
            return False
        with self._lock:
            now = time.time()
            self._prune_locked(now)
            seen = token in self._cache
            self._cache[token] = now
            self._persist_locked()
            return seen


class FeishuWebhookSecurity:
    def __init__(
        self,
        *,
        verification_token: Optional[str] = None,
        encrypt_key: Optional[str] = None,
    ) -> None:
        self._verification_token = (verification_token or "").strip()
        self._encrypt_key = (encrypt_key or "").strip()

    @staticmethod
    def _canonical_json(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _sha_digest(text: str, algo: str) -> str:
        h = hashlib.new(algo)
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: str,
        parsed_body: Any,
    ) -> bool:
        if not self._encrypt_key:
            return True
        timestamp = (headers.get("x-lark-request-timestamp") or "").strip()
        nonce = (headers.get("x-lark-request-nonce") or "").strip()
        signature = (headers.get("x-lark-signature") or "").strip().lower()
        if not timestamp or not nonce or not signature:
            return False
        canonical = self._canonical_json(parsed_body)
        expected_canonical = self._sha_digest(
            f"{timestamp}{nonce}{self._encrypt_key}{canonical}",
            "sha256",
        )
        if signature == expected_canonical:
            return True
        expected_raw = self._sha_digest(
            f"{timestamp}{nonce}{self._encrypt_key}{raw_body}",
            "sha256",
        )
        return signature == expected_raw

    def verify_token(self, payload: Any) -> bool:
        if not self._verification_token:
            return True
        if not isinstance(payload, dict):
            return False
        token = (payload.get("token") or "").strip()
        if not token:
            token = ((payload.get("header") or {}).get("token") or "").strip()
        if not token:
            return True
        return token == self._verification_token

    def decrypt_payload(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        encrypt = payload.get("encrypt")
        if not encrypt:
            return payload
        if not self._encrypt_key:
            raise RuntimeError("Encrypted callback received but no encrypt key configured")
        plain = self._aes_decrypt_json(str(encrypt), self._encrypt_key)
        if not isinstance(plain, dict):
            raise RuntimeError("Invalid decrypted callback payload")
        rest = {k: v for k, v in payload.items() if k != "encrypt"}
        if rest:
            plain.update(rest)
        return plain

    @staticmethod
    def _aes_decrypt_json(encrypt: str, encrypt_key: str) -> Any:
        try:
            from cryptography.hazmat.primitives import hashes, padding
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as exc:
            raise RuntimeError(
                "Encrypted Feishu callbacks require 'cryptography' package"
            ) from exc
        digest = hashes.Hash(hashes.SHA256())
        digest.update(encrypt_key.encode("utf-8"))
        key = digest.finalize()
        raw = base64.b64decode(encrypt)
        if len(raw) < 17:
            raise RuntimeError("Invalid encrypted callback payload")
        iv, ciphertext = raw[:16], raw[16:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_plain = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        plain = unpadder.update(padded_plain) + unpadder.finalize()
        return json.loads(plain.decode("utf-8"))


class FeishuWebhookProcessor:
    def __init__(
        self,
        *,
        runtime: Any,
        path: str,
        deduper: FeishuDeduper,
        verification_token: Optional[str] = None,
        encrypt_key: Optional[str] = None,
        max_body_bytes: int = _MAX_BODY_BYTES,
    ) -> None:
        self._runtime = runtime
        self._path = path
        self._deduper = deduper
        self._security = FeishuWebhookSecurity(
            verification_token=verification_token,
            encrypt_key=encrypt_key,
        )
        self._max_body_bytes = max(1024, int(max_body_bytes))

    @staticmethod
    def _as_json(data: Any) -> bytes:
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    def _route_payload(self, payload: Any) -> None:
        _process_feishu_event_payload(
            runtime=self._runtime,
            deduper=self._deduper,
            payload=payload,
            expected_event_type=_FEISHU_EVENT_MESSAGE_RECEIVE,
        )

    def process_http(
        self,
        request_path: str,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> Tuple[int, Dict[str, str], bytes]:
        if request_path.split("?", 1)[0] != self._path:
            return 404, {}, b""
        content_type = (headers.get("Content-Type") or headers.get("content-type") or "").lower()
        if "application/json" not in content_type:
            return 415, {"Content-Type": "text/plain; charset=utf-8"}, b"Unsupported Media Type"
        if len(raw_body) > self._max_body_bytes:
            return 413, {"Content-Type": "text/plain; charset=utf-8"}, b"Payload Too Large"
        raw_text = raw_body.decode("utf-8", errors="replace") if raw_body else "{}"
        try:
            body = json.loads(raw_text)
        except Exception:
            return 400, {}, b""
        if not self._security.verify_signature(headers, raw_text, body):
            return 401, {"Content-Type": "text/plain; charset=utf-8"}, b"Invalid Signature"
        try:
            payload = self._security.decrypt_payload(body)
        except Exception as exc:
            logger.warning("Failed to decrypt Feishu payload: %s", exc)
            return 400, {"Content-Type": "text/plain; charset=utf-8"}, b"Invalid Encrypted Payload"
        if not self._security.verify_token(payload):
            return 403, {"Content-Type": "text/plain; charset=utf-8"}, b"Invalid Verification Token"
        if (payload.get("type") or "").strip() == "url_verification":
            return 200, {"Content-Type": "application/json"}, self._as_json(
                {"challenge": payload.get("challenge", "")}
            )
        self._route_payload(payload)
        return 200, {}, b""


def _json_loads_safely(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _process_feishu_event_payload(
    *,
    runtime: Any,
    deduper: FeishuDeduper,
    payload: Any,
    expected_event_type: str = _FEISHU_EVENT_MESSAGE_RECEIVE,
) -> bool:
    if not isinstance(payload, dict):
        return False
    header = payload.get("header") or {}
    event = payload.get("event") or {}
    event_type = (header.get("event_type") or (event or {}).get("type") or "").strip()
    if event_type != expected_event_type:
        return False
    event_id = (header.get("event_id") or "").strip()
    if event_id and deduper.seen_or_record(f"event:{event_id}"):
        return True
    if not isinstance(event, dict):
        return True
    msg = event.get("message") or {}
    if not isinstance(msg, dict):
        return True
    message_id = (msg.get("message_id") or "").strip()
    if message_id and deduper.seen_or_record(f"message:{message_id}"):
        return True
    message_type = (msg.get("message_type") or "").strip().lower()
    chat_id = (event.get("chat_id") or msg.get("chat_id") or "").strip()
    chat_type = (
        event.get("chat_type")
        or msg.get("chat_type")
        or event.get("conversation_type")
        or msg.get("conversation_type")
    )
    thread_id = (
        event.get("root_id")
        or msg.get("root_id")
        or event.get("thread_id")
        or msg.get("thread_id")
    )
    if not chat_id:
        return True
    content = _json_loads_safely(msg.get("content"))

    if message_type == "text":
        text = (content.get("text") or "").strip()
        if text:
            runtime.submit(runtime.handle_text(chat_id, thread_id, text, chat_type=chat_type))
    elif message_type in {"file", "media"}:
        file_key = (content.get("file_key") or "").strip()
        file_name = (content.get("file_name") or content.get("name") or "upload.bin").strip()
        if file_key:
            runtime.submit(
                runtime.handle_file(chat_id, thread_id, file_key, file_name, chat_type=chat_type)
            )
    elif message_type == "image":
        runtime.submit(
            runtime.handle_text(
                chat_id,
                thread_id,
                "收到图片。若需分析请上传 .h5ad 或发送文字指令。",
                chat_type=chat_type,
            )
        )
    return True


class FeishuRuntime:
    def __init__(self, client: FeishuClient, session_manager: Any) -> None:
        self._client = client
        self._registry = GatewaySessionRegistry(session_manager)
        self._sm = session_manager
        self._tasks: Dict[str, RunningTask] = {}
        self._pending: Dict[str, List[str]] = {}
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

    @staticmethod
    def _route(
        chat_id: str,
        thread_id: Optional[str],
        chat_type: Optional[str] = None,
    ) -> ConversationRoute:
        norm = (chat_type or "").strip().lower()
        if norm in {"p2p", "private", "direct", "dm"}:
            scope_type = "dm"
        elif norm in {"group", "chat", "room"}:
            scope_type = "group"
        else:
            scope_type = "chat"
        return ConversationRoute(
            channel="feishu",
            scope_type=scope_type,
            scope_id=str(chat_id),
            thread_id=(str(thread_id) if thread_id else None),
        )

    def _session_key(
        self,
        chat_id: str,
        thread_id: Optional[str],
        chat_type: Optional[str] = None,
    ) -> SessionKey:
        return self._route(chat_id, thread_id, chat_type).to_session_key()

    def _route_key(
        self,
        chat_id: str,
        thread_id: Optional[str],
        chat_type: Optional[str] = None,
    ) -> str:
        return self._route(chat_id, thread_id, chat_type).route_key()

    # ------------------------------------------------------------------
    # Helpers mirroring Telegram feature set
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_local_paths(text: str) -> str:
        """Remove local filesystem path references from result text."""
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
        """Build context-injected request (AGENTS.md + memory + current request)."""
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

    async def _spawn_analysis(
        self, chat_id: str, route: str, session: Any, user_text: str
    ) -> None:
        """Build full_request, create draft card, and launch _analysis_wrapper task."""
        full_request = self._build_full_request(session, user_text)
        try:
            draft_id = self._client.send_markdown_card(
                chat_id, "💭 思考中…", title="OmicVerse Jarvis", color="grey"
            )
        except Exception:
            draft_id = self._client.send_text(chat_id, "💭 思考中…")
        task = asyncio.create_task(
            self._analysis_wrapper(chat_id, route, session, user_text, draft_id, full_request)
        )
        self._tasks[route] = RunningTask(task=task, request=user_text, started_at=time.time())

    async def _analysis_wrapper(
        self,
        chat_id: str,
        route: str,
        session: Any,
        user_text: str,
        draft_id: Optional[str],
        full_request: str,
    ) -> None:
        """Wrap _run_analysis; drain pending queue when done (OpenClaw Collect pattern)."""
        try:
            await self._run_analysis(chat_id, route, session, user_text, draft_id, full_request)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Feishu analysis wrapper error")
            try:
                self._client.send_text(chat_id, f"❌ 分析异常: {exc}")
            except Exception:
                pass
        finally:
            self._tasks.pop(route, None)
            queued = self._pending.pop(route, [])
            if queued:
                coalesced = "\n\n".join(queued)
                n = len(queued)
                try:
                    self._client.send_text(chat_id, f"⏭ 开始执行队列中的 {n} 条请求…")
                except Exception:
                    pass
                asyncio.create_task(
                    self._spawn_analysis(chat_id, route, session, coalesced)
                )

    async def _quick_chat(
        self,
        chat_id: str,
        session: Any,
        user_text: str,
        running_request: str = "",
        queued: bool = False,
    ) -> None:
        """Respond conversationally (no tools) while background analysis runs."""
        try:
            system_lines = [
                "You are OmicVerse Jarvis, a bioinformatics AI assistant.",
                "The user is chatting with you while a background analysis is running.",
                "Answer concisely and helpfully. Do NOT execute code or call tools.",
                "Reply in the same language the user uses.",
            ]
            if running_request:
                system_lines.append(f"\nCurrently running analysis: {running_request[:300]}")
            if queued:
                system_lines.append(
                    "The user's message has been queued and will start automatically "
                    "after the current analysis finishes. Mention this naturally."
                )
            if session.adata is not None:
                a = session.adata
                system_lines.append(f"Loaded data: {a.n_obs:,} cells × {a.n_vars:,} genes")
            try:
                memory_ctx = session.get_memory_context()
                if memory_ctx:
                    system_lines.append(f"\nRecent analysis history:\n{memory_ctx[:600]}")
            except Exception:
                pass
            messages = [
                {"role": "system", "content": "\n".join(system_lines)},
                {"role": "user", "content": user_text},
            ]
            response = await session.agent._llm.chat(messages, tools=None, tool_choice=None)
            reply = (response.content or "").strip() or "⏳ 后台分析进行中，请等待完成。"
            for chunk in self._text_chunks(reply):
                await asyncio.to_thread(self._client.send_text, chat_id, chunk)
        except Exception as exc:
            logger.warning("Feishu quick_chat failed: %s", exc)
            try:
                await asyncio.to_thread(
                    self._client.send_text,
                    chat_id,
                    "⏳ 后台分析进行中，请等待完成。使用 /cancel 取消。",
                )
            except Exception:
                pass

    async def handle_text(
        self,
        chat_id: str,
        thread_id: Optional[str],
        text: str,
        *,
        chat_type: Optional[str] = None,
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        sk = self._session_key(chat_id, thread_id, chat_type)
        route = self._route_key(chat_id, thread_id, chat_type)
        session = self._registry.get_or_create(sk)

        tokens = self._parse_command(text)
        cmd = tokens[0].lower() if tokens else ""
        tail = text.split(None, 1)[1].strip() if len(tokens) > 1 else ""

        if cmd == "/cancel":
            await self._handle_cancel(chat_id, route)
            return
        if cmd in {"/start", "/help"}:
            await self._handle_help(chat_id)
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
        if cmd == "/workspace":
            await self._handle_workspace(chat_id, session)
            return
        if cmd == "/ls":
            await self._handle_ls(chat_id, session, tail)
            return
        if cmd == "/find":
            await self._handle_find(chat_id, session, tail)
            return
        if cmd == "/load":
            await self._handle_load(chat_id, session, tail)
            return
        if cmd == "/shell":
            await self._handle_shell(chat_id, session, tail)
            return
        if cmd == "/memory":
            await self._handle_memory(chat_id, session)
            return
        if cmd == "/usage":
            await self._handle_usage(chat_id, session)
            return
        if cmd == "/model":
            await self._handle_model(chat_id, tail)
            return
        if cmd == "/save":
            await self._handle_save(chat_id, session)
            return

        running = self._tasks.get(route)
        if running and not running.task.done():
            # OpenClaw Collect mode: queue message + respond conversationally
            self._pending.setdefault(route, []).append(text)
            asyncio.create_task(
                self._quick_chat(
                    chat_id, session, text,
                    running_request=running.request, queued=True,
                )
            )
            return

        # Ack message before analysis (mirrors Telegram handle_analysis)
        if session.adata is not None:
            a = session.adata
            ack = (
                f"⚙️ 收到请求，开始分析…\n"
                f"🔬 当前数据: {a.n_obs:,} cells × {a.n_vars:,} genes"
            )
        else:
            try:
                h5ad_files = session.list_h5ad_files()
            except Exception:
                h5ad_files = []
            if h5ad_files:
                names = "  ".join(f.name for f in h5ad_files[:5])
                ack = (
                    f"⚙️ 收到请求，开始分析…\n"
                    f"💡 workspace 中检测到文件: {names}\n"
                    f"使用 /load <文件名> 加载"
                )
            else:
                ack = "⚙️ 收到请求，开始分析…\n💡 未检测到已加载数据，Agent 将自行加载数据"
        self._client.send_text(chat_id, ack)

        await self._spawn_analysis(chat_id, route, session, text)

    async def handle_file(
        self,
        chat_id: str,
        thread_id: Optional[str],
        file_key: str,
        file_name: str,
        *,
        chat_type: Optional[str] = None,
    ) -> None:
        sk = self._session_key(chat_id, thread_id, chat_type)
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
        full_request: Optional[str] = None,
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
            content = _draft_text()
            ok = await asyncio.to_thread(
                self._client.edit_card, draft_id, content, "💭 分析中…", "grey"
            )
            if not ok:
                ok = await asyncio.to_thread(self._client.edit_text, draft_id, content)
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
            result = await bridge.run(full_request or user_text, session.adata)
        except asyncio.CancelledError:
            if draft_id:
                ok = await asyncio.to_thread(
                    self._client.edit_card, draft_id, "🚫 已取消当前分析。", "取消", "red"
                )
                if not ok:
                    await asyncio.to_thread(self._client.edit_text, draft_id, "🚫 已取消当前分析。")
            raise
        except Exception as exc:
            logger.exception("Feishu analysis failed")
            err_msg = f"❌ 分析失败: {exc}"
            if draft_id:
                ok = await asyncio.to_thread(
                    self._client.edit_card, draft_id, err_msg, "❌ 分析失败", "red"
                )
                if not ok:
                    await asyncio.to_thread(self._client.edit_text, draft_id, err_msg)
            else:
                await asyncio.to_thread(self._client.send_text, chat_id, err_msg)
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
            adata_info = f"{a.n_obs:,} cells × {a.n_vars:,} genes" if a is not None else ""
            session.append_memory_log(
                request=user_text,
                summary=(result.summary or "分析完成"),
                adata_info=adata_info,
            )
        except Exception:
            pass

        if result.error:
            err_text = f"❌ {result.error}"
            if result.diagnostics:
                hints = "\n".join(f"- {x}" for x in result.diagnostics[:4])
                err_text += f"\n\n诊断信息:\n{hints}"
            if llm_buf.strip():
                err_text += f"\n\n最后模型输出片段:\n{_trim(llm_buf, max_len=1200)}"
            if draft_id:
                ok = await asyncio.to_thread(
                    self._client.edit_card, draft_id, err_text, "❌ 分析失败", "red"
                )
                if not ok:
                    await asyncio.to_thread(self._client.edit_text, draft_id, err_text)
            else:
                await asyncio.to_thread(self._client.send_text, chat_id, err_text)
            return

        if draft_id:
            ok = await asyncio.to_thread(
                self._client.edit_card, draft_id, "✅ 分析完成，正在发送结果…", "💭 分析中…", "grey"
            )
            if not ok:
                await asyncio.to_thread(self._client.edit_text, draft_id, "✅ 分析完成，正在发送结果…")

        for rep in list(result.reports or []):
            if len(rep) <= 4800:
                await asyncio.to_thread(
                    self._client.send_markdown_card, chat_id, rep, "📊 分析报告"
                )
            else:
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

        summary = self._strip_local_paths((result.summary or "").strip())
        if not summary or summary.lower() in _BORING:
            if llm_buf.strip():
                summary = _trim(llm_buf, max_len=3600)
            elif result.diagnostics:
                hints = "\n".join(f"- {x}" for x in result.diagnostics[:5])
                summary = f"⚠️ 未生成有效最终答复\n{hints}"
            elif session.adata is not None:
                a = session.adata
                summary = f"✅ 分析完成\n🔬 {a.n_obs:,} cells × {a.n_vars:,} genes"
            else:
                summary = "✅ 分析完成"
        if len(summary) <= 4800:
            await asyncio.to_thread(
                self._client.send_markdown_card, chat_id, summary, "✅ 分析完成", "green"
            )
        else:
            for chunk in self._text_chunks(summary):
                await asyncio.to_thread(self._client.send_text, chat_id, chunk)
        if draft_id:
            ok = await asyncio.to_thread(
                self._client.edit_card, draft_id, "✅ 分析完成", "✅ 分析完成", "green"
            )
            if not ok:
                await asyncio.to_thread(self._client.edit_text, draft_id, "✅ 分析完成")

    async def _handle_cancel(self, chat_id: str, route: str) -> None:
        self._pending.pop(route, None)  # clear queued messages on cancel
        running = self._tasks.get(route)
        if not running or running.task.done():
            self._client.send_text(chat_id, "ℹ️ 当前没有正在运行的分析。")
            return
        running.task.cancel()
        self._client.send_text(chat_id, "⏹ 已发送取消信号。")

    async def _handle_help(self, chat_id: str) -> None:
        text = (
            "👋 OmicVerse Jarvis\n"
            "--------------------\n"
            "数据命令:\n"
            "/workspace 查看工作区\n"
            "/ls [路径] 列出文件\n"
            "/find <模式> 搜索文件\n"
            "/load <文件名> 加载数据\n"
            "/shell <命令> 执行白名单 shell\n\n"
            "会话命令:\n"
            "/kernel | /kernel ls | /kernel new 名称 | /kernel use 名称\n"
            "/memory 分析历史\n"
            "/usage token 用量\n"
            "/model [名称] 查看/切换模型\n"
            "/status 当前状态\n"
            "/save 导出 current.h5ad\n"
            "/cancel 取消分析\n"
            "/reset 重置会话"
        )
        self._client.send_text(chat_id, text)

    async def _handle_status(self, chat_id: str, session: Any, route: str) -> None:
        lines: List[str] = []
        if session.adata is not None:
            a = session.adata
            lines.append(f"🔬 {a.n_obs:,} cells × {a.n_vars:,} genes")
            try:
                cols = ", ".join(list(a.obs.columns[:8]))
                if cols:
                    lines.append(f"📋 obs: {cols}")
            except Exception:
                pass
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

    async def _handle_workspace(self, chat_id: str, session: Any) -> None:
        ws = session.workspace
        h5ad_files = session.list_h5ad_files()
        agents_md = session.get_agents_md()
        today_log = session.memory_dir / f"{datetime.now().date()}.md"
        lines = [
            "📁 Workspace",
            "--------------------",
            str(ws),
            "",
        ]
        if h5ad_files:
            lines.append(f"📊 数据文件 ({len(h5ad_files)})")
            for f in h5ad_files[:10]:
                try:
                    mb = f.stat().st_size / 1_048_576
                    lines.append(f"- {f.name} ({mb:.1f} MB)")
                except OSError:
                    lines.append(f"- {f.name}")
            if len(h5ad_files) > 10:
                lines.append(f"... 还有 {len(h5ad_files) - 10} 个")
        else:
            lines.append("📊 数据文件 (空)")
        lines += [
            "",
            f"📋 AGENTS.md {'✅' if agents_md else '—'}",
            f"🧠 今日记忆 {'✅' if today_log.exists() else '—'}",
            "",
            "可用: /load <文件名> | /ls | /memory",
        ]
        self._client.send_text(chat_id, "\n".join(lines))

    async def _handle_ls(self, chat_id: str, session: Any, subpath: str) -> None:
        cmd = f"ls -lh {subpath}".strip() if subpath else "ls -lh"
        out = session.shell.exec(cmd, cwd=session.workspace)
        for chunk in self._text_chunks(f"$ {cmd}\n{out}", limit=3200):
            self._client.send_text(chat_id, chunk)

    async def _handle_find(self, chat_id: str, session: Any, pattern: str) -> None:
        pattern = (pattern or "").strip()
        if not pattern:
            self._client.send_text(chat_id, "用法: /find <模式>，例如 /find *.h5ad")
            return
        cmd = f"find . -name {pattern}"
        out = session.shell.exec(cmd, cwd=session.workspace)
        for chunk in self._text_chunks(f"$ {cmd}\n{out}", limit=3200):
            self._client.send_text(chat_id, chunk)

    async def _handle_load(self, chat_id: str, session: Any, filename: str) -> None:
        filename = (filename or "").strip()
        if not filename:
            self._client.send_text(chat_id, "用法: /load <文件名>，例如 /load pbmc3k.h5ad")
            return
        self._client.send_text(chat_id, f"⏳ 正在加载 {filename} ...")
        try:
            adata = await asyncio.to_thread(session.load_from_workspace, filename)
        except Exception as exc:
            logger.exception("Feishu /load failed")
            self._client.send_text(chat_id, f"❌ 加载失败: {exc}")
            return
        if adata is None:
            files = session.list_h5ad_files()
            hint = ""
            if files:
                hint = "\n可用文件: " + "  ".join(f.name for f in files[:5])
            self._client.send_text(chat_id, f"❌ 未找到 {filename}{hint}")
            return
        self._client.send_text(
            chat_id,
            f"✅ 加载成功\n🔬 {adata.n_obs:,} cells × {adata.n_vars:,} genes\n📁 {filename}",
        )

    async def _handle_shell(self, chat_id: str, session: Any, cmd: str) -> None:
        cmd = (cmd or "").strip()
        if not cmd:
            self._client.send_text(
                chat_id,
                "用法: /shell <命令>\n允许: ls find cat head wc file du pwd tree",
            )
            return
        out = session.shell.exec(cmd, cwd=session.workspace)
        for chunk in self._text_chunks(f"$ {cmd}\n{out}", limit=3200):
            self._client.send_text(chat_id, chunk)

    async def _handle_memory(self, chat_id: str, session: Any) -> None:
        text = session.get_recent_memory_text()
        for chunk in self._text_chunks(f"🧠 分析历史（近两天）\n\n{text}", limit=3200):
            self._client.send_text(chat_id, chunk)

    async def _handle_usage(self, chat_id: str, session: Any) -> None:
        usage = session.last_usage
        if usage is None:
            self._client.send_text(chat_id, "ℹ️ 暂无用量数据，请先进行一次分析。")
            return

        def _attr(obj: Any, *names: str, default: str = "?") -> str:
            for name in names:
                v = getattr(obj, name, None)
                if v is not None:
                    return f"{v:,}" if isinstance(v, int) else str(v)
            return default

        lines = [
            "📊 Token 用量（最近一次）",
            "--------------------",
            f"输入: {_attr(usage, 'input_tokens')}",
            f"输出: {_attr(usage, 'output_tokens')}",
            f"合计: {_attr(usage, 'total_tokens')}",
        ]
        cr = _attr(usage, "cache_read_input_tokens", default="")
        cc = _attr(usage, "cache_creation_input_tokens", default="")
        if cr and cr != "?":
            lines.append(f"缓存读取: {cr}")
        if cc and cc != "?":
            lines.append(f"缓存写入: {cc}")
        self._client.send_text(chat_id, "\n".join(lines))

    async def _handle_model(self, chat_id: str, model_name: str) -> None:
        model_name = (model_name or "").strip()
        if not model_name:
            text = render_model_help(getattr(self._sm, "_model", "unknown"))
            for chunk in self._text_chunks(text):
                self._client.send_text(chat_id, chunk)
            return
        self._sm._model = model_name
        self._client.send_text(
            chat_id,
            f"✅ 模型已切换为 {model_name}\n请 /reset 重启 kernel 使新模型生效。",
        )

    async def _handle_save(self, chat_id: str, session: Any) -> None:
        if session.adata is None:
            self._client.send_text(chat_id, "❌ 没有数据，请先 /load 或完成分析。")
            return
        self._client.send_text(chat_id, "⏳ 正在保存 current.h5ad ...")
        try:
            path = await asyncio.to_thread(session.save_adata)
            if not path or not Path(path).exists():
                self._client.send_text(chat_id, "❌ 保存失败，请重试。")
                return
            data = Path(path).read_bytes()
            await asyncio.to_thread(self._client.send_file_bytes, chat_id, data, "current.h5ad")
            a = session.adata
            self._client.send_text(chat_id, f"💾 已发送 current.h5ad\n🔬 {a.n_obs:,} cells × {a.n_vars:,} genes")
        except Exception as exc:
            logger.exception("Feishu /save failed")
            self._client.send_text(chat_id, f"❌ 保存失败: {exc}")

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
    verification_token: Optional[str] = None,
    encrypt_key: Optional[str] = None,
) -> None:
    client = FeishuClient(app_id, app_secret)
    runtime = FeishuRuntime(client, session_manager)
    base_dir = getattr(session_manager, "_base", Path(os.path.expanduser("~/.ovjarvis")))
    deduper = FeishuDeduper(Path(base_dir) / "feishu" / "dedup" / "global.json")
    processor = FeishuWebhookProcessor(
        runtime=runtime,
        path=path,
        deduper=deduper,
        verification_token=verification_token,
        encrypt_key=encrypt_key,
    )

    class FeishuHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            status, extra_headers, payload = processor.process_http(self.path, self.headers, raw)
            self.send_response(status)
            for key, value in extra_headers.items():
                self.send_header(key, value)
            self.end_headers()
            if payload:
                self.wfile.write(payload)

        def log_message(self, fmt: str, *args: object) -> None:
            logger.debug("feishu_http " + fmt, *args)

    server = ThreadingHTTPServer((host, port), FeishuHandler)
    logger.info("Feishu Jarvis webhook listening on http://%s:%s%s", host, port, path)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def run_feishu_ws_bot(
    *,
    app_id: str,
    app_secret: str,
    session_manager: Any,
    verification_token: Optional[str] = None,
    encrypt_key: Optional[str] = None,
) -> None:
    try:
        import lark_oapi as lark
    except ImportError as exc:
        raise RuntimeError(
            "Feishu websocket mode requires 'lark-oapi'. Install with: pip install lark-oapi"
        ) from exc

    client = FeishuClient(app_id, app_secret)
    runtime = FeishuRuntime(client, session_manager)
    base_dir = getattr(session_manager, "_base", Path(os.path.expanduser("~/.ovjarvis")))
    deduper = FeishuDeduper(Path(base_dir) / "feishu" / "dedup" / "global.json")

    def _on_message(data: Any) -> None:
        try:
            raw = lark.JSON.marshal(data)
            payload = json.loads(raw) if isinstance(raw, str) else {}
            _process_feishu_event_payload(
                runtime=runtime,
                deduper=deduper,
                payload=payload,
                expected_event_type=_FEISHU_EVENT_MESSAGE_RECEIVE,
            )
        except Exception:
            logger.exception("Feishu websocket event handling failed")

    builder = lark.EventDispatcherHandler.builder(
        encrypt_key or "",
        verification_token or "",
    )
    event_handler = builder.register_p2_im_message_receive_v1(_on_message).build()
    logger.info("Feishu Jarvis long connection starting (websocket mode)")
    sdk_log_level = (
        lark.LogLevel.DEBUG
        if logger.isEnabledFor(logging.DEBUG)
        else lark.LogLevel.INFO
    )
    ws_client = lark.ws.Client(
        app_id,
        app_secret,
        event_handler=event_handler,
        log_level=sdk_log_level,
    )
    try:
        ws_client.start()
    except Exception as exc:
        raise RuntimeError(
            "Feishu websocket connection failed. Check app_id/app_secret, "
            "Feishu event subscription mode, and network reachability."
        ) from exc
