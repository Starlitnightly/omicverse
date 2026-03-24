"""
WeChat channel for OmicVerse Jarvis via Tencent iLink long-polling.

Supports:
- official iLink `getupdates` long-poll for inbound messages
- official iLink `sendmessage` for outbound text and image replies
- image upload via AES-128-ECB + CDN (mirrors @tencent-weixin/openclaw-weixin)
- basic commands: /help /status /reset /cancel /model
- group chats are ignored
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import re
import secrets
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding as _aes_padding
    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRYPTO_AVAILABLE = False

from ..agent_bridge import AgentBridge
from .._bridge_session import resolve_bridge_session_id
from ..config import default_state_dir
from ..gateway.routing import GatewaySessionRegistry, SessionKey
from ..model_help import render_model_help

logger = logging.getLogger("omicverse.jarvis.wechat")
logger.setLevel(logging.INFO)

_DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"

_WECHAT_CHANNEL_HINT = """\
[Channel: WeChat — image and text delivery rules]
1. To produce a figure/plot: write and execute Python code that creates a matplotlib \
figure (e.g. sc.pl.umap, sc.pl.violin, plt.plot). End the cell with plt.show() or \
plt.savefig("output.png"). The gateway captures every matplotlib figure created during \
code execution and sends it as a WeChat image message automatically — you do NOT need \
to do anything else.
2. NEVER provide image download links, sandbox:// paths, or Markdown image syntax. \
Any link you output will NOT be visible in WeChat.
3. After code runs, reply in plain text only: briefly describe what was plotted. \
No Markdown tables, fences, bold, or hyperlinks.
4. Never narrate your own tool usage ("I called finish", "mandatory tool call", etc.).\
"""
_DEFAULT_LONG_POLL_TIMEOUT_MS = 35_000
_DEFAULT_API_TIMEOUT_MS = 15_000
_MAX_TEXT = 3800
_PROGRESS_GAP = 12.0
_BORING_SUMMARIES = {"分析完成", "分析完成。", "task completed", "done", "完成"}

# Fragments that indicate the agent leaked internal tool-call meta-commentary.
# Any summary containing one of these is suppressed (replaced with llm_buf or empty).
_META_COMMENTARY_FRAGMENTS = (
    "mandatory tool call",
    "mandatory tool-use",
    "complied by calling",
    "no further computation",
    "forced a tool call",
    "forced tool",
    "satisfy.*tool.use requirement",
    "tool-use requirement",
    "calling finish",
    "user requested mandatory",
)
_META_COMMENTARY_RE = re.compile(
    "|".join(_META_COMMENTARY_FRAGMENTS), re.IGNORECASE
)

# --------------------------------------------------------------------------- #
# AES-128-ECB helpers (mirrors cdn/aes-ecb.ts)                                #
# --------------------------------------------------------------------------- #

def _aes_ecb_padded_size(plaintext_size: int) -> int:
    """PKCS7-padded ciphertext size for AES-128-ECB."""
    return math.ceil((plaintext_size + 1) / 16) * 16


def _encrypt_aes_ecb(plaintext: bytes, key: bytes) -> bytes:
    """AES-128-ECB encrypt with PKCS7 padding."""
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError(
            "cryptography package is required for WeChat image upload. "
            "Install it with: pip install cryptography"
        )
    padder = _aes_padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    enc = cipher.encryptor()
    return enc.update(padded) + enc.finalize()


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
    # Markdown links/images whose href is a local or sandbox path:
    #   [label](sandbox:/mnt/...)  ![img](./output/foo.png)
    value = re.sub(r"!?\[[^\]]*\]\((?:sandbox:|file:)?[./~]?[^\)]*\)", "", value)
    # Inline code containing multi-segment paths
    value = re.sub(r"`[^`\n]*(?:/[^`\n]*){2,}`", "", value)
    # Absolute system paths
    value = re.sub(r"/(?:Users|home|tmp|var|opt|root|data|mnt|private)/\S+", "", value)
    # Tilde-relative paths
    value = re.sub(r"~[/\\]\S+", "", value)
    # Relative file references with known extensions
    ext = r"pdf|csv|tsv|txt|xlsx|html|json|h5ad|png|jpg|svg"
    value = re.sub(rf"\.?/?(?:\w[\w/-]*/)+\w[\w.-]*\.(?:{ext})", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _extract_text(item_list: Any) -> str:
    if not isinstance(item_list, list):
        return ""
    parts: List[str] = []
    for item in item_list:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == 1:
            text = str(((item.get("text_item") or {}).get("text") or "")).strip()
            if text:
                parts.append(text)
        elif item_type == 3:
            text = str(((item.get("voice_item") or {}).get("text") or "")).strip()
            if text:
                parts.append(text)
        elif item_type == 2:
            parts.append("[图片]")
        elif item_type == 4:
            fname = str(((item.get("file_item") or {}).get("file_name") or "")).strip()
            parts.append(f"[文件] {fname}" if fname else "[文件]")
        elif item_type == 5:
            parts.append("[视频]")
    return "\n".join(parts).strip()


class WeChatApiClient:
    def __init__(
        self,
        *,
        token: str,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._token = str(token or "").strip()
        self._base_url = str(base_url or _DEFAULT_BASE_URL).rstrip("/")

    @staticmethod
    def _random_wechat_uin() -> str:
        return base64.b64encode(str(secrets.randbits(32)).encode("utf-8")).decode("ascii")

    def _headers(self, body: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "Authorization": f"Bearer {self._token}",
            "X-WECHAT-UIN": self._random_wechat_uin(),
            "Content-Length": str(len(body.encode("utf-8"))),
        }

    def _post_json(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        *,
        timeout: tuple[float, float],
    ) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False)
        resp = requests.post(
            f"{self._base_url}{endpoint}",
            data=body.encode("utf-8"),
            headers=self._headers(body),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"WeChat API returned non-object response for {endpoint}")
        return data

    def probe(self) -> None:
        payload = {
            "get_updates_buf": "",
            "base_info": {"channel_version": "jarvis"},
        }
        data = self._post_json(
            "/ilink/bot/getupdates",
            payload,
            timeout=(5.0, 2.0),
        )
        errcode = data.get("errcode")
        ret = data.get("ret")
        if (isinstance(errcode, int) and errcode != 0) or (isinstance(ret, int) and ret != 0):
            raise RuntimeError(data.get("errmsg") or f"WeChat auth failed (ret={ret}, errcode={errcode})")

    def get_updates(
        self,
        *,
        get_updates_buf: str,
        timeout_ms: int = _DEFAULT_LONG_POLL_TIMEOUT_MS,
    ) -> Dict[str, Any]:
        payload = {
            "get_updates_buf": get_updates_buf or "",
            "base_info": {"channel_version": "jarvis"},
        }
        read_timeout = max(5.0, (float(timeout_ms) / 1000.0) + 5.0)
        return self._post_json(
            "/ilink/bot/getupdates",
            payload,
            timeout=(10.0, read_timeout),
        )

    def send_text(self, *, to_user_id: str, text: str, context_token: str) -> None:
        payload = {
            "msg": {
                "from_user_id": "",
                "to_user_id": to_user_id,
                "client_id": uuid.uuid4().hex,
                "message_type": 2,
                "message_state": 2,
                "context_token": context_token,
                "item_list": [
                    {
                        "type": 1,
                        "text_item": {"text": text},
                    }
                ],
            },
            "base_info": {"channel_version": "jarvis"},
        }
        data = self._post_json(
            "/ilink/bot/sendmessage",
            payload,
            timeout=(10.0, 20.0),
        )
        errcode = data.get("errcode")
        ret = data.get("ret")
        if (isinstance(errcode, int) and errcode != 0) or (isinstance(ret, int) and ret != 0):
            raise RuntimeError(data.get("errmsg") or f"WeChat send failed (ret={ret}, errcode={errcode})")

    # ------------------------------------------------------------------ #
    # Image upload + send (mirrors cdn/upload.ts + messaging/send.ts)     #
    # ------------------------------------------------------------------ #

    def _get_upload_url(
        self,
        *,
        filekey: str,
        to_user_id: str,
        rawsize: int,
        rawfilemd5: str,
        filesize: int,
        aeskey_hex: str,
    ) -> str:
        """Call getuploadurl and return upload_param."""
        payload = {
            "filekey": filekey,
            "media_type": 1,  # IMAGE
            "to_user_id": to_user_id,
            "rawsize": rawsize,
            "rawfilemd5": rawfilemd5,
            "filesize": filesize,
            "no_need_thumb": True,
            "aeskey": aeskey_hex,
            "base_info": {"channel_version": "jarvis"},
        }
        data = self._post_json("/ilink/bot/getuploadurl", payload, timeout=(10.0, 20.0))
        upload_param = data.get("upload_param")
        if not upload_param:
            raise RuntimeError(f"getuploadurl returned no upload_param: {data}")
        return str(upload_param)

    @staticmethod
    def _cdn_upload(
        *,
        cdn_base_url: str,
        upload_param: str,
        filekey: str,
        ciphertext: bytes,
    ) -> str:
        """POST encrypted bytes to CDN. Returns x-encrypted-param (downloadParam)."""
        url = (
            f"{cdn_base_url}/upload"
            f"?encrypted_query_param={urllib.parse.quote(upload_param, safe='')}"
            f"&filekey={urllib.parse.quote(filekey, safe='')}"
        )
        for attempt in range(1, 4):
            resp = requests.post(
                url,
                data=ciphertext,
                headers={"Content-Type": "application/octet-stream"},
                timeout=(10.0, 30.0),
            )
            if 400 <= resp.status_code < 500:
                err = resp.headers.get("x-error-message") or resp.text
                raise RuntimeError(f"CDN upload client error {resp.status_code}: {err}")
            if resp.status_code == 200:
                download_param = resp.headers.get("x-encrypted-param")
                if not download_param:
                    raise RuntimeError("CDN upload response missing x-encrypted-param header")
                return download_param
            # server error — retry
            if attempt == 3:
                err = resp.headers.get("x-error-message") or f"status {resp.status_code}"
                raise RuntimeError(f"CDN upload failed after 3 attempts: {err}")
        raise RuntimeError("CDN upload: unreachable")

    def upload_image(
        self,
        *,
        file_path: Path,
        to_user_id: str,
        cdn_base_url: str = _CDN_BASE_URL,
    ) -> Tuple[str, str, int]:
        """Upload a local image file to WeChat CDN.

        Returns (download_encrypted_query_param, aeskey_hex, ciphertext_size).
        Mirrors upload.ts uploadMediaToCdn.
        """
        plaintext = file_path.read_bytes()
        rawsize = len(plaintext)
        rawfilemd5 = hashlib.md5(plaintext).hexdigest()
        filesize = _aes_ecb_padded_size(rawsize)
        filekey = secrets.token_hex(16)
        aeskey = secrets.token_bytes(16)
        aeskey_hex = aeskey.hex()

        ciphertext = _encrypt_aes_ecb(plaintext, aeskey)

        upload_param = self._get_upload_url(
            filekey=filekey,
            to_user_id=to_user_id,
            rawsize=rawsize,
            rawfilemd5=rawfilemd5,
            filesize=filesize,
            aeskey_hex=aeskey_hex,
        )
        download_param = self._cdn_upload(
            cdn_base_url=cdn_base_url,
            upload_param=upload_param,
            filekey=filekey,
            ciphertext=ciphertext,
        )
        return download_param, aeskey_hex, filesize

    def send_image(
        self,
        *,
        to_user_id: str,
        context_token: str,
        download_param: str,
        aeskey_hex: str,
        ciphertext_size: int,
        caption: str = "",
    ) -> None:
        """Send an uploaded image (and optional text caption) via sendmessage.

        Mirrors send.ts sendImageMessageWeixin:
          aes_key in JSON = base64(utf8_bytes_of_hex_string)  ← not base64(raw_key)
          mid_size        = ciphertext file size
          encrypt_type    = 1
        """
        # aes_key encoding: base64 of the hex-string bytes (matches TS Buffer.from(hex).toString("base64"))
        aes_key_b64 = base64.b64encode(aeskey_hex.encode("ascii")).decode("ascii")

        items: List[Dict[str, Any]] = []
        if caption:
            items.append({"type": 1, "text_item": {"text": caption}})
        items.append({
            "type": 2,
            "image_item": {
                "media": {
                    "encrypt_query_param": download_param,
                    "aes_key": aes_key_b64,
                    "encrypt_type": 1,
                },
                "mid_size": ciphertext_size,
            },
        })

        for item in items:
            payload = {
                "msg": {
                    "from_user_id": "",
                    "to_user_id": to_user_id,
                    "client_id": uuid.uuid4().hex,
                    "message_type": 2,
                    "message_state": 2,
                    "context_token": context_token,
                    "item_list": [item],
                },
                "base_info": {"channel_version": "jarvis"},
            }
            data = self._post_json("/ilink/bot/sendmessage", payload, timeout=(10.0, 20.0))
            errcode = data.get("errcode")
            ret = data.get("ret")
            if (isinstance(errcode, int) and errcode != 0) or (isinstance(ret, int) and ret != 0):
                raise RuntimeError(
                    data.get("errmsg") or f"WeChat send_image failed (ret={ret}, errcode={errcode})"
                )

    def download_file(self, file_key: str) -> bytes:
        """Download a file from the WeChat iLink server by file_key.

        Calls ``/ilink/bot/getmediadownloadurl`` to obtain a temporary CDN URL,
        then fetches the raw bytes from that URL.  The exact endpoint name may
        differ across iLink deployments — adjust if the server returns an error.
        """
        payload = {
            "file_key": file_key,
            "base_info": {"channel_version": "jarvis"},
        }
        body = json.dumps(payload, ensure_ascii=False)
        resp = requests.post(
            f"{self._base_url}/ilink/bot/getmediadownloadurl",
            data=body.encode("utf-8"),
            headers=self._headers(body),
            timeout=(10.0, 30.0),
        )
        resp.raise_for_status()
        result = resp.json()
        download_url = (
            ((result.get("data") or {}).get("download_url"))
            or result.get("download_url")
            or result.get("url")
        )
        if not download_url:
            raise RuntimeError(f"WeChat file download URL not returned by server: {result}")
        file_resp = requests.get(download_url, timeout=(10.0, 120.0))
        file_resp.raise_for_status()
        return file_resp.content


class WeChatJarvisBot:
    def __init__(
        self,
        *,
        token: str,
        base_url: str,
        session_manager: Any,
        allow_from: Optional[List[str]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._client = WeChatApiClient(token=token, base_url=base_url)
        self._sm = session_manager
        self._stop_event = stop_event or threading.Event()
        self._allow_from = {str(item).strip() for item in (allow_from or []) if str(item).strip()}
        self._registry = GatewaySessionRegistry(session_manager)
        self._tasks: Dict[str, RunningTask] = {}
        self._pending: Dict[str, List[str]] = {}
        self._context_tokens: Dict[str, str] = {}
        self._cursor_path = self._cursor_store_path(token)

    @staticmethod
    def _cursor_store_path(token: str) -> Path:
        token_hash = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
        return default_state_dir() / "wechat" / f"{token_hash}_cursor.json"

    def _load_cursor(self) -> str:
        try:
            if not self._cursor_path.exists():
                return ""
            raw = json.loads(self._cursor_path.read_text(encoding="utf-8"))
            value = raw.get("get_updates_buf") if isinstance(raw, dict) else ""
            return str(value or "")
        except Exception:
            logger.debug("Failed to load WeChat cursor", exc_info=True)
            return ""

    def _save_cursor(self, value: str) -> None:
        try:
            self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cursor_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"get_updates_buf": value}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._cursor_path)
        except Exception:
            logger.debug("Failed to persist WeChat cursor", exc_info=True)

    async def run(self) -> None:
        try:
            await self._probe_connection()
            logger.info("WeChat long-poll loop starting")
            await self._poll_loop()
        except Exception:
            logger.exception("WeChat channel exited with error")
            raise

    async def _probe_connection(self) -> None:
        logger.info("Probing WeChat iLink endpoint")
        try:
            await asyncio.to_thread(self._client.probe)
        except requests.exceptions.ReadTimeout:
            logger.info("WeChat probe timed out during long-poll; endpoint reachable, continuing")
            return
        logger.info("WeChat iLink auth OK")

    async def _poll_loop(self) -> None:
        get_updates_buf = self._load_cursor()
        timeout_ms = _DEFAULT_LONG_POLL_TIMEOUT_MS
        consecutive_failures = 0
        last_error = ""
        while not self._stop_event.is_set():
            try:
                data = await asyncio.to_thread(
                    self._client.get_updates,
                    get_updates_buf=get_updates_buf,
                    timeout_ms=timeout_ms,
                )
                errcode = data.get("errcode")
                ret = data.get("ret")
                if (isinstance(errcode, int) and errcode != 0) or (isinstance(ret, int) and ret != 0):
                    last_error = str(data.get("errmsg") or f"ret={ret}, errcode={errcode}")
                    consecutive_failures += 1
                    logger.warning(
                        "WeChat getupdates failed (%s/3): %s",
                        consecutive_failures,
                        last_error,
                    )
                    if consecutive_failures >= 3:
                        raise RuntimeError(f"WeChat getupdates failed repeatedly: {last_error}")
                    await self._sleep_or_stop(2.0)
                    continue

                consecutive_failures = 0
                next_buf = str(data.get("get_updates_buf") or "")
                if next_buf:
                    get_updates_buf = next_buf
                    self._save_cursor(get_updates_buf)
                next_timeout = data.get("longpolling_timeout_ms")
                if isinstance(next_timeout, int) and next_timeout > 0:
                    timeout_ms = next_timeout

                msgs = data.get("msgs") or []
                for raw in msgs:
                    if not isinstance(raw, dict):
                        continue
                    await self._on_message(raw)
            except requests.exceptions.ReadTimeout:
                consecutive_failures = 0
                continue
            except Exception as exc:
                last_error = str(exc)
                consecutive_failures += 1
                logger.warning(
                    "WeChat polling error (%s/3): %s",
                    consecutive_failures,
                    exc,
                )
                if consecutive_failures >= 3:
                    raise RuntimeError(f"WeChat polling failed repeatedly: {last_error}") from exc
                await self._sleep_or_stop(2.0)

    async def _sleep_or_stop(self, seconds: float) -> None:
        await asyncio.to_thread(self._stop_event.wait, seconds)

    async def _on_message(self, raw: Dict[str, Any]) -> None:
        from_user_id = str(raw.get("from_user_id") or "").strip()
        context_token = str(raw.get("context_token") or "").strip()
        message_type = raw.get("message_type")
        group_id = str(raw.get("group_id") or "").strip()
        if not from_user_id:
            return
        if from_user_id.endswith("@im.bot"):
            return
        if self._allow_from and from_user_id not in self._allow_from:
            logger.info("WeChat message ignored because sender is not in allow_from: %s", from_user_id)
            return
        if group_id:
            logger.info("WeChat group message ignored for MVP: group_id=%s from=%s", group_id, from_user_id)
            return
        if not context_token:
            logger.warning("WeChat message ignored because context_token is missing: from=%s", from_user_id)
            return

        # Intercept .h5ad file uploads (item type 4) before passing text to the LLM.
        # The file_item dict is expected to carry a file_key and file_name.
        item_list = raw.get("item_list") or []
        for _item in item_list if isinstance(item_list, list) else []:
            if not isinstance(_item, dict) or _item.get("type") != 4:
                continue
            _file_item = _item.get("file_item") or {}
            _fname = str(_file_item.get("file_name") or "").strip()
            _fkey = str(
                _file_item.get("file_key") or _file_item.get("media_id") or ""
            ).strip()
            if _fname.lower().endswith(".h5ad") and _fkey:
                self._context_tokens[from_user_id] = context_token
                _sk = SessionKey(channel="wechat", scope_type="dm", scope_id=from_user_id)
                _session = self._registry.get_or_create(_sk)
                await self._handle_h5ad_file(_sk, _session, context_token, _fkey, _fname)
                return

        text = _extract_text(raw.get("item_list"))
        logger.info(
            "WeChat message received: from=%s message_type=%s content_len=%s",
            from_user_id,
            message_type,
            len(text),
        )
        if not text:
            logger.info("WeChat message ignored because no text content was extracted")
            return

        self._context_tokens[from_user_id] = context_token
        session_key = SessionKey(channel="wechat", scope_type="dm", scope_id=from_user_id)
        session = self._registry.get_or_create(session_key)
        route = session_key.as_key()
        cmd, tail = self._command_parts(text)

        if cmd == "/help":
            await self._send_session_text(
                session_key,
                "OmicVerse Jarvis (WeChat)\n"
                "/help 查看帮助\n"
                "/status 查看当前状态\n"
                "/reset 重置当前会话\n"
                "/model [名称] 查看或切换模型\n"
                "/cancel 取消当前分析",
                context_token=context_token,
            )
            return

        if cmd == "/status":
            await self._handle_status(session_key, session, route, context_token)
            return

        if cmd == "/reset":
            session.reset()
            await self._send_session_text(session_key, "✅ 已重置当前会话。", context_token=context_token)
            return

        if cmd == "/cancel":
            await self._handle_cancel(session_key, route, context_token)
            return

        if cmd == "/model":
            if not tail:
                for chunk in _text_chunks(render_model_help(getattr(self._sm, "_model", "unknown"))):
                    await self._send_session_text(session_key, chunk, context_token=context_token)
                return
            self._sm._model = tail
            await self._send_session_text(
                session_key,
                f"✅ 模型已切换为 {tail}\n请 /reset 重启 kernel 使新模型生效。",
                context_token=context_token,
            )
            return

        if cmd.startswith("/"):
            await self._send_session_text(
                session_key,
                f"未知命令: {cmd}\n发送 /help 查看帮助。",
                context_token=context_token,
            )
            return

        running = self._tasks.get(route)
        if running and not running.task.done():
            self._pending.setdefault(route, []).append(text)
            await self._send_session_text(
                session_key,
                "⏭ 已加入当前会话队列，当前分析完成后继续处理。",
                context_token=context_token,
            )
            return

        await self._send_ack(session_key, session, context_token)
        await self._spawn_analysis(session_key, session, text)

    async def _handle_status(
        self,
        session_key: SessionKey,
        session: Any,
        route: str,
        context_token: str,
    ) -> None:
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
        await self._send_session_text(session_key, "\n".join(parts), context_token=context_token)

    async def _handle_cancel(self, session_key: SessionKey, route: str, context_token: str) -> None:
        self._pending.pop(route, None)
        running = self._tasks.get(route)
        if not running or running.task.done():
            await self._send_session_text(
                session_key,
                "当前没有正在运行的分析任务。",
                context_token=context_token,
            )
            return
        running.task.cancel()
        await self._send_session_text(session_key, "🚫 已发送取消信号。", context_token=context_token)

    async def _send_ack(self, session_key: SessionKey, session: Any, context_token: str) -> None:
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
        await self._send_session_text(session_key, text, context_token=context_token)

    async def _handle_h5ad_file(
        self,
        session_key: SessionKey,
        session: Any,
        context_token: str,
        file_key: str,
        file_name: str,
    ) -> None:
        """Download a .h5ad file sent by the user and load it into the session."""
        self._context_tokens[session_key.scope_id] = context_token
        await self._send_session_text(session_key, "⏳ 正在下载并加载…", context_token=context_token)
        try:
            data = await asyncio.to_thread(self._client.download_file, file_key)
            target = session.workspace / file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            loaded = await asyncio.to_thread(session.load_from_workspace, file_name)
            if loaded is not None:
                a = loaded
                await self._send_session_text(
                    session_key,
                    f"✅ 加载成功\n🔬 {a.n_obs:,} cells × {a.n_vars:,} genes\n📁 {file_name}",
                    context_token=context_token,
                )
            else:
                await self._send_session_text(
                    session_key,
                    f"✅ 已接收 {file_name}，但自动加载失败，请检查文件格式。",
                    context_token=context_token,
                )
        except Exception as exc:
            logger.exception("WeChat failed to handle h5ad file")
            await self._send_session_text(
                session_key,
                f"❌ 文件处理失败: {exc}",
                context_token=context_token,
            )

    async def _spawn_analysis(self, session_key: SessionKey, session: Any, user_text: str) -> None:
        route = session_key.as_key()
        task = asyncio.create_task(self._analysis_wrapper(session_key, session, user_text))
        self._tasks[route] = RunningTask(task=task, request=user_text, started_at=time.time())

    async def _analysis_wrapper(self, session_key: SessionKey, session: Any, user_text: str) -> None:
        route = session_key.as_key()
        try:
            await self._run_analysis(session_key, session, user_text)
        except asyncio.CancelledError:
            try:
                await self._send_session_text(session_key, "已取消当前分析。")
            except Exception:
                pass
            raise
        except Exception as exc:
            logger.exception("WeChat analysis wrapper failed")
            await self._send_session_text(session_key, f"分析异常: {exc}")
        finally:
            self._tasks.pop(route, None)
            queued = self._pending.pop(route, [])
            if queued:
                await self._send_session_text(session_key, f"开始执行队列中的 {len(queued)} 条请求...")
                next_session = self._registry.get_or_create(session_key)
                await self._spawn_analysis(session_key, next_session, "\n\n".join(queued))

    async def _run_analysis(self, session_key: SessionKey, session: Any, user_text: str) -> None:
        llm_buf = ""
        last_progress_sent = 0.0
        full_request = self._build_full_request(session, user_text)

        async def progress_cb(msg: str) -> None:
            nonlocal last_progress_sent
            now = asyncio.get_running_loop().time()
            if now - last_progress_sent < _PROGRESS_GAP:
                return
            last_progress_sent = now
            await self._send_session_text(session_key, f"⚙️ {msg[:200]}")

        async def llm_chunk_cb(chunk: str) -> None:
            nonlocal llm_buf
            if chunk:
                llm_buf += chunk

        _wb = getattr(self._sm, "gateway_web_bridge", None)
        _prior_history = _wb.get_prior_history_simple(
            "wechat",
            session_key.scope_type,
            session_key.scope_id,
            session_id=resolve_bridge_session_id(session),
        ) if _wb else []

        bridge = AgentBridge(session.agent, progress_cb=progress_cb, llm_chunk_cb=llm_chunk_cb)
        try:
            result = await bridge.run(full_request, session.adata, history=_prior_history)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("WeChat analysis failed")
            await self._send_session_text(session_key, f"分析失败: {exc}")
            return

        logger.info(
            "WeChat: bridge result figures=%d artifacts=%d error=%s",
            len(result.figures),
            len(result.artifacts),
            result.error,
        )

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
                    channel="wechat",
                    scope_type=session_key.scope_type,
                    scope_id=session_key.scope_id,
                    user_text=user_text,
                    llm_text=llm_buf,
                    adata=result.adata,
                    figures=result.figures or [],
                    session_id=resolve_bridge_session_id(session),
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
            await self._send_session_text(session_key, err_text)
            return

        for report in list(result.reports or []):
            for chunk in _text_chunks(report):
                await self._send_session_text(session_key, chunk)

        if result.figures:
            await self._send_figures(session_key, result.figures)
        if result.artifacts:
            await self._send_session_text(
                session_key,
                f"已生成 {len(result.artifacts)} 个附件，请在 Web 界面下载。",
            )

        # pick_reply_text handles has_artifacts + llm_buf priority; WeChat then
        # additionally strips agent meta-commentary fragments.
        summary = _strip_local_paths(bridge.pick_reply_text(result, llm_buf))
        if summary and _META_COMMENTARY_RE.search(summary):
            logger.debug("WeChat: suppressing meta-commentary summary: %.120s", summary)
            summary = ""
        if not summary:
            if session.adata is not None:
                adata = session.adata
                summary = f"分析完成\n{adata.n_obs:,} cells x {adata.n_vars:,} genes"
            else:
                summary = "分析完成"
        for chunk in _text_chunks(summary):
            await self._send_session_text(session_key, chunk)

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
        # Channel hint goes last so task context takes priority over delivery rules.
        ctx_parts.append(_WECHAT_CHANNEL_HINT)
        ctx_parts.append(f"[Current request]\n{text}")
        return "\n\n".join(ctx_parts)

    @staticmethod
    def _command_parts(text: str) -> tuple[str, str]:
        tokens = text.split()
        cmd = tokens[0].lower() if tokens else ""
        tail = text.split(None, 1)[1].strip() if len(tokens) > 1 else ""
        return cmd, tail

    async def _send_figures(self, session_key: SessionKey, figures: List[Any]) -> None:
        """Upload and send each figure as a WeChat image message."""
        for fig in figures:
            try:
                await self._send_figure(session_key, fig)
            except Exception as exc:
                logger.warning("WeChat: failed to send figure %s: %s", getattr(fig, "name", fig), exc)
                await self._send_session_text(
                    session_key,
                    f"图像发送失败，请在 Web 界面查看。",
                )

    async def _send_figure(self, session_key: SessionKey, figure: Any) -> None:
        """Upload one figure and send it as a WeChat image message.

        ``figure`` may be:
        - ``bytes``  — raw PNG data (what AgentBridge.result.figures contains)
        - ``Path`` / ``str`` — path to an existing image file
        - object with a ``.path`` attribute
        """
        import tempfile

        context_token = self._context_tokens.get(session_key.scope_id, "")
        if not context_token:
            logger.warning("WeChat: skipping figure send, no context_token for %s", session_key.scope_id)
            return

        tmp_file: Optional[Any] = None
        try:
            if isinstance(figure, bytes):
                # Write raw bytes to a temp PNG file, then upload from disk.
                tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_file.write(figure)
                tmp_file.flush()
                tmp_file.close()
                fig_path = Path(tmp_file.name)
            elif isinstance(figure, (str, Path)):
                fig_path = Path(figure)
            elif hasattr(figure, "path"):
                fig_path = Path(figure.path)
            else:
                logger.warning("WeChat: unrecognised figure type %s, skipping", type(figure))
                return

            if not fig_path.exists():
                logger.warning("WeChat: figure file not found: %s", fig_path)
                return

            logger.info("WeChat: uploading figure %s (%d bytes) to CDN", fig_path.name, fig_path.stat().st_size)
            download_param, aeskey_hex, ciphertext_size = await asyncio.to_thread(
                self._client.upload_image,
                file_path=fig_path,
                to_user_id=session_key.scope_id,
            )
            logger.info("WeChat: figure upload done, sending image message")
            await asyncio.to_thread(
                self._client.send_image,
                to_user_id=session_key.scope_id,
                context_token=context_token,
                download_param=download_param,
                aeskey_hex=aeskey_hex,
                ciphertext_size=ciphertext_size,
            )
        finally:
            if tmp_file is not None:
                try:
                    Path(tmp_file.name).unlink(missing_ok=True)
                except Exception:
                    pass

    async def _send_session_text(
        self,
        session_key: SessionKey,
        text: str,
        *,
        context_token: Optional[str] = None,
    ) -> None:
        token = str(context_token or self._context_tokens.get(session_key.scope_id) or "").strip()
        if not token:
            logger.warning("Skipping WeChat send because context_token is missing: scope_id=%s", session_key.scope_id)
            return
        for chunk in _text_chunks(text):
            await asyncio.to_thread(
                self._client.send_text,
                to_user_id=session_key.scope_id,
                text=chunk,
                context_token=token,
            )


def run_wechat_bot(
    *,
    token: str,
    session_manager: Any,
    base_url: str = _DEFAULT_BASE_URL,
    allow_from: Optional[List[str]] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    bot = WeChatJarvisBot(
        token=token,
        base_url=base_url,
        session_manager=session_manager,
        allow_from=allow_from,
        stop_event=stop_event,
    )
    asyncio.run(bot.run())
