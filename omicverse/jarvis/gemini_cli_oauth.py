"""
Gemini CLI OAuth helpers for Jarvis.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from .config import default_auth_path, load_auth, save_auth


GOOGLE_OAUTH_CLIENT_ID_KEYS = (
    "OPENCLAW_GEMINI_OAUTH_CLIENT_ID",
    "GEMINI_CLI_OAUTH_CLIENT_ID",
)
GOOGLE_OAUTH_CLIENT_SECRET_KEYS = (
    "OPENCLAW_GEMINI_OAUTH_CLIENT_SECRET",
    "GEMINI_CLI_OAUTH_CLIENT_SECRET",
)
GOOGLE_REDIRECT_URI = "http://localhost:8085/oauth2callback"
GOOGLE_CALLBACK_PORT = 8085
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"
GOOGLE_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
)
GOOGLE_CODE_ASSIST_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"
GOOGLE_CODE_ASSIST_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
GOOGLE_CODE_ASSIST_ENDPOINT_AUTOPUSH = "https://autopush-cloudcode-pa.sandbox.googleapis.com"
GOOGLE_CODE_ASSIST_ENDPOINTS = (
    GOOGLE_CODE_ASSIST_ENDPOINT_PROD,
    GOOGLE_CODE_ASSIST_ENDPOINT_DAILY,
    GOOGLE_CODE_ASSIST_ENDPOINT_AUTOPUSH,
)
GOOGLE_TIER_FREE = "free-tier"
GOOGLE_TIER_LEGACY = "legacy-tier"
GOOGLE_TIER_STANDARD = "standard-tier"

_GOOGLE_CLIENT_ID_RE = re.compile(r"(\d+-[a-z0-9]+\.apps\.googleusercontent\.com)")
_GOOGLE_CLIENT_SECRET_RE = re.compile(r"(GOCSPX-[A-Za-z0-9_-]+)")

logger = logging.getLogger(__name__)


class GeminiCliOAuthError(RuntimeError):
    """Raised when Gemini CLI OAuth login or refresh fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _pkce_pair() -> Tuple[str, str]:
    verifier = secrets.token_bytes(32).hex()
    challenge = _b64url(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def _normalize_expires_at(value: object) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        epoch = float(value)
        if epoch > 1_000_000_000_000:
            epoch /= 1000.0
        return int(epoch)
    text = str(value).strip()
    if not text:
        return None
    try:
        epoch = float(text)
        if epoch > 1_000_000_000_000:
            epoch /= 1000.0
        return int(epoch)
    except Exception:
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return int(datetime.fromisoformat(text).timestamp())
    except Exception:
        return None


def token_expired(expires_at: object, skew_seconds: int = 300) -> bool:
    epoch = _normalize_expires_at(expires_at)
    if not epoch:
        return True
    return time.time() >= (float(epoch) - skew_seconds)


def _resolve_env(keys: Tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = str(os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def _find_binary_on_path(name: str) -> Optional[Path]:
    path_env = str(os.environ.get("PATH") or "")
    if not path_env:
        return None
    exts = (".cmd", ".bat", ".exe", "") if os.name == "nt" else ("",)
    for directory in path_env.split(os.pathsep):
        if not directory:
            continue
        for ext in exts:
            candidate = Path(directory) / f"{name}{ext}"
            if candidate.exists():
                try:
                    return candidate.resolve()
                except Exception:
                    return candidate
    return None


def _candidate_gemini_cli_dirs(binary_path: Path) -> list[Path]:
    resolved = binary_path.resolve()
    binary_dir = resolved.parent
    candidates = [
        resolved.parent.parent,
        resolved.parent / "node_modules" / "@google" / "gemini-cli",
        binary_dir / "node_modules" / "@google" / "gemini-cli",
        binary_dir.parent / "node_modules" / "@google" / "gemini-cli",
        binary_dir.parent / "lib" / "node_modules" / "@google" / "gemini-cli",
    ]
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).replace("\\", "/").lower() if os.name == "nt" else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _find_file(root: Path, filename: str, depth: int = 10) -> Optional[Path]:
    if depth <= 0 or not root.exists() or not root.is_dir():
        return None
    try:
        for entry in root.iterdir():
            if entry.is_file() and entry.name == filename:
                return entry
            if entry.is_dir() and not entry.name.startswith("."):
                found = _find_file(entry, filename, depth - 1)
                if found is not None:
                    return found
    except Exception:
        return None
    return None


def extract_gemini_cli_credentials() -> Optional[Tuple[str, str]]:
    gemini_path = _find_binary_on_path("gemini")
    if gemini_path is None:
        return None

    content = ""
    for root in _candidate_gemini_cli_dirs(gemini_path):
        search_paths = [
            root / "node_modules" / "@google" / "gemini-cli-core" / "dist" / "src" / "code_assist" / "oauth2.js",
            root / "node_modules" / "@google" / "gemini-cli-core" / "dist" / "code_assist" / "oauth2.js",
        ]
        for search_path in search_paths:
            if search_path.exists():
                try:
                    content = search_path.read_text(encoding="utf-8")
                except Exception:
                    content = ""
                if content:
                    break
        if content:
            break
        found = _find_file(root, "oauth2.js", depth=10)
        if found is not None:
            try:
                content = found.read_text(encoding="utf-8")
            except Exception:
                content = ""
            if content:
                break

    if not content:
        return None

    client_id_match = _GOOGLE_CLIENT_ID_RE.search(content)
    client_secret_match = _GOOGLE_CLIENT_SECRET_RE.search(content)
    if not client_id_match or not client_secret_match:
        return None
    return client_id_match.group(1), client_secret_match.group(1)


def resolve_oauth_client_config() -> Tuple[str, Optional[str]]:
    env_client_id = _resolve_env(GOOGLE_OAUTH_CLIENT_ID_KEYS)
    env_client_secret = _resolve_env(GOOGLE_OAUTH_CLIENT_SECRET_KEYS)
    if env_client_id:
        return env_client_id, env_client_secret

    extracted = extract_gemini_cli_credentials()
    if extracted is not None:
        return extracted[0], extracted[1]

    raise GeminiCliOAuthError(
        "Gemini CLI not found. Install it first, or set GEMINI_CLI_OAUTH_CLIENT_ID."
    )


def build_authorize_url(*, code_challenge: str, state: str) -> str:
    client_id, _ = resolve_oauth_client_config()
    query = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "scope": " ".join(GOOGLE_SCOPES),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(query)}"


def _parse_manual_callback(value: str, expected_state: str) -> Dict[str, str]:
    text = str(value or "").strip()
    if not text:
        raise GeminiCliOAuthError("Missing OAuth callback URL or code")
    try:
        parsed = urlparse(text)
        if parsed.scheme and parsed.netloc:
            params = {key: values[-1] for key, values in parse_qs(parsed.query).items() if values}
            code = str(params.get("code") or "").strip()
            state = str(params.get("state") or expected_state).strip()
            if not code:
                raise GeminiCliOAuthError("Missing 'code' parameter in redirect URL")
            if not state:
                raise GeminiCliOAuthError("Missing 'state' parameter in redirect URL")
            return {"code": code, "state": state}
    except GeminiCliOAuthError:
        raise
    except Exception:
        pass
    if not expected_state:
        raise GeminiCliOAuthError("Paste the full redirect URL, not just the code.")
    return {"code": text, "state": expected_state}


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    server_version = "JarvisGeminiOAuth/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/oauth2callback":
            self.send_error(404)
            return
        params = {key: values[-1] for key, values in parse_qs(parsed.query).items() if values}
        self.server.result = params  # type: ignore[attr-defined]
        self.server.event.set()  # type: ignore[attr-defined]
        body = (
            "<html><body><h3>Gemini CLI OAuth complete</h3>"
            "<p>You can close this window and return to Jarvis.</p></body></html>"
        )
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _fetch_user_email(access_token: str) -> str:
    try:
        resp = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if resp.ok:
            payload = resp.json()
            if isinstance(payload, dict):
                return str(payload.get("email") or "").strip()
    except Exception:
        pass
    return ""


def _proxy_style_headers(access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "x-goog-api-client": "gl-python/omicverse",
        "Accept": "application/json",
    }


def _load_code_assist_proxy_style(access_token: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{GOOGLE_CODE_ASSIST_ENDPOINT_PROD}/v1internal:loadCodeAssist",
        headers=_proxy_style_headers(access_token),
        json={
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        },
        timeout=10,
    )
    if not resp.ok:
        raise GeminiCliOAuthError(f"loadCodeAssist failed: {resp.status_code} {resp.reason}")
    payload = resp.json()
    return payload if isinstance(payload, dict) else {}


def _resolve_env_project() -> str:
    return str(
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
        or ""
    ).strip()


def _resolve_platform() -> str:
    if os.name == "nt":
        return "WINDOWS"
    if os.uname().sysname.lower() == "darwin":  # type: ignore[attr-defined]
        return "MACOS"
    return "PLATFORM_UNSPECIFIED"


def _is_vpc_sc_affected(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    details = error.get("details")
    if not isinstance(details, list):
        return False
    for item in details:
        if isinstance(item, dict) and item.get("reason") == "SECURITY_POLICY_VIOLATED":
            return True
    return False


def _default_tier(allowed_tiers: Any) -> str:
    if not isinstance(allowed_tiers, list) or not allowed_tiers:
        return GOOGLE_TIER_LEGACY
    for tier in allowed_tiers:
        if isinstance(tier, dict) and tier.get("isDefault"):
            return str(tier.get("id") or GOOGLE_TIER_LEGACY)
    return GOOGLE_TIER_LEGACY


def _poll_operation(endpoint: str, operation_name: str, headers: Dict[str, str]) -> Dict[str, Any]:
    for _ in range(24):
        time.sleep(5)
        resp = requests.get(
            f"{endpoint}/v1internal/{operation_name}",
            headers=headers,
            timeout=10,
        )
        if not resp.ok:
            continue
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("done"):
            return payload
    raise GeminiCliOAuthError("Operation polling timeout")


def resolve_google_oauth_identity(access_token: str) -> Dict[str, str]:
    email = _fetch_user_email(access_token)
    result: Dict[str, str] = {}
    if email:
        result["email"] = email
    try:
        project_id = discover_google_project(access_token)
    except Exception as exc:
        project_id = _resolve_env_project()
        logger.warning(
            "gemini_cli_oauth_project_discovery_failed error=%s fallback_project=%s",
            exc,
            bool(project_id),
        )
    if project_id:
        result["project_id"] = project_id
    return result


def discover_google_project(access_token: str) -> str:
    env_project = _resolve_env_project()
    try:
        load_assist = _load_code_assist_proxy_style(access_token)
        gcp_managed = bool(load_assist.get("gcpManaged"))
        project_value = load_assist.get("cloudaicompanionProject")
        if not gcp_managed:
            if isinstance(project_value, str) and project_value.strip():
                return project_value.strip()
            if isinstance(project_value, dict):
                project_id = str(project_value.get("id") or "").strip()
                if project_id:
                    return project_id
        elif isinstance(project_value, str) and project_value.strip():
            return project_value.strip()
        elif isinstance(project_value, dict):
            project_id = str(project_value.get("id") or "").strip()
            if project_id:
                return project_id
    except Exception as exc:
        logger.warning("gemini_cli_oauth_proxy_load_code_assist_failed error=%s", exc)

    platform_name = _resolve_platform()
    metadata = {
        "ideType": "ANTIGRAVITY",
        "platform": platform_name,
        "pluginType": "GEMINI",
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "gl-python/omicverse",
        "Client-Metadata": json.dumps(metadata),
    }
    load_body: Dict[str, Any] = {"metadata": dict(metadata)}
    if env_project:
        load_body["cloudaicompanionProject"] = env_project
        load_body["metadata"]["duetProject"] = env_project

    data: Dict[str, Any] = {}
    active_endpoint = GOOGLE_CODE_ASSIST_ENDPOINT_PROD
    load_error: Optional[Exception] = None
    for endpoint in GOOGLE_CODE_ASSIST_ENDPOINTS:
        try:
            resp = requests.post(
                f"{endpoint}/v1internal:loadCodeAssist",
                headers=headers,
                json=load_body,
                timeout=10,
            )
            if not resp.ok:
                payload = None
                try:
                    payload = resp.json()
                except Exception:
                    payload = None
                if _is_vpc_sc_affected(payload):
                    data = {"currentTier": {"id": GOOGLE_TIER_STANDARD}}
                    active_endpoint = endpoint
                    load_error = None
                    break
                load_error = GeminiCliOAuthError(
                    f"loadCodeAssist failed: {resp.status_code} {resp.reason}"
                )
                continue
            payload = resp.json()
            if isinstance(payload, dict):
                data = payload
            active_endpoint = endpoint
            load_error = None
            break
        except Exception as exc:
            load_error = exc if isinstance(exc, Exception) else GeminiCliOAuthError("loadCodeAssist failed")

    has_load_code_assist_data = bool(
        data.get("currentTier")
        or data.get("cloudaicompanionProject")
        or data.get("allowedTiers")
    )
    if not has_load_code_assist_data and load_error is not None:
        if env_project:
            return env_project
        raise GeminiCliOAuthError(str(load_error))

    current_tier = data.get("currentTier")
    current_project = data.get("cloudaicompanionProject")
    if current_tier:
        if isinstance(current_project, str) and current_project:
            return current_project
        if isinstance(current_project, dict):
            project_id = str(current_project.get("id") or "").strip()
            if project_id:
                return project_id
        if env_project:
            return env_project
        raise GeminiCliOAuthError(
            "This account requires GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID to be set."
        )

    tier_id = _default_tier(data.get("allowedTiers")) or GOOGLE_TIER_FREE
    if tier_id != GOOGLE_TIER_FREE and not env_project:
        raise GeminiCliOAuthError(
            "This account requires GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID to be set."
        )

    onboard_body: Dict[str, Any] = {
        "tierId": tier_id,
        "metadata": dict(metadata),
    }
    if tier_id != GOOGLE_TIER_FREE and env_project:
        onboard_body["cloudaicompanionProject"] = env_project
        onboard_body["metadata"]["duetProject"] = env_project

    onboard_resp = requests.post(
        f"{active_endpoint}/v1internal:onboardUser",
        headers=headers,
        json=onboard_body,
        timeout=10,
    )
    if not onboard_resp.ok:
        raise GeminiCliOAuthError(
            f"onboardUser failed: {onboard_resp.status_code} {onboard_resp.reason}"
        )

    operation = onboard_resp.json()
    if not isinstance(operation, dict):
        operation = {}
    if not operation.get("done") and operation.get("name"):
        operation = _poll_operation(active_endpoint, str(operation["name"]), headers)

    project_id = str((((operation.get("response") or {}).get("cloudaicompanionProject") or {}).get("id")) or "").strip()
    if project_id:
        return project_id
    if env_project:
        return env_project
    raise GeminiCliOAuthError(
        "Could not discover or provision a Google Cloud project. "
        "Set GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID."
    )


def exchange_code_for_tokens(*, code: str, code_verifier: str) -> Dict[str, Any]:
    client_id, client_secret = resolve_oauth_client_config()
    body = {
        "client_id": client_id,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "code_verifier": code_verifier,
    }
    if client_secret:
        body["client_secret"] = client_secret

    resp = requests.post(
        GOOGLE_TOKEN_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Accept": "*/*",
            "User-Agent": "google-api-nodejs-client/9.15.1",
        },
        timeout=30,
    )
    if not resp.ok:
        raise GeminiCliOAuthError(f"Token exchange failed: {resp.text[:300]}")
    data = resp.json()
    access_token = str(data.get("access_token") or "").strip()
    refresh_token = str(data.get("refresh_token") or "").strip()
    expires_in = data.get("expires_in")
    if not refresh_token:
        raise GeminiCliOAuthError("No refresh token received. Please try again.")
    if not access_token:
        raise GeminiCliOAuthError("OAuth token exchange returned no access token")
    try:
        expires_seconds = int(expires_in)
    except Exception as exc:
        raise GeminiCliOAuthError("OAuth token exchange returned invalid expiry") from exc
    identity = resolve_google_oauth_identity(access_token)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(time.time()) + expires_seconds - 300,
        "email": str(identity.get("email") or "").strip(),
        "project_id": str(identity.get("project_id") or "").strip(),
    }


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    client_id, client_secret = resolve_oauth_client_config()
    body = {
        "client_id": client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    if client_secret:
        body["client_secret"] = client_secret

    resp = requests.post(
        GOOGLE_TOKEN_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Accept": "*/*",
            "User-Agent": "google-api-nodejs-client/9.15.1",
        },
        timeout=30,
    )
    if not resp.ok:
        raise GeminiCliOAuthError(f"Google token refresh failed: {resp.text[:300]}")
    data = resp.json()
    access_token = str(data.get("access_token") or "").strip()
    next_refresh = str(data.get("refresh_token") or refresh_token).strip()
    if not access_token:
        raise GeminiCliOAuthError("Google token refresh returned no access token")
    try:
        expires_seconds = int(data.get("expires_in"))
    except Exception as exc:
        raise GeminiCliOAuthError("Google token refresh returned invalid expiry") from exc
    return {
        "access_token": access_token,
        "refresh_token": next_refresh,
        "expires_at": int(time.time()) + expires_seconds - 300,
    }


def _normalize_auth_record(record: Dict[str, Any]) -> Dict[str, Any]:
    auth = dict(record)
    tokens = dict(auth.get("tokens") or {})
    expires_at = _normalize_expires_at(
        tokens.get("expires_at")
        or tokens.get("expires")
        or tokens.get("expiry")
    )
    if expires_at:
        tokens["expires_at"] = expires_at
    access_token = str(tokens.get("access_token") or tokens.get("access") or "").strip()
    refresh_token = str(tokens.get("refresh_token") or tokens.get("refresh") or "").strip()
    email = str(tokens.get("email") or "").strip()
    project_id = str(tokens.get("project_id") or tokens.get("projectId") or "").strip()
    auth["tokens"] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "email": email,
        "project_id": project_id,
    }
    auth["provider"] = "google-gemini-cli"
    if "last_refresh" not in auth or not auth.get("last_refresh"):
        auth["last_refresh"] = _utc_now()
    return auth


class GeminiCliOAuthManager:
    """Manage Jarvis Gemini CLI OAuth state."""

    def __init__(self, auth_path: Optional[Path] = None) -> None:
        self.auth_path = auth_path or default_auth_path()

    def _load_root(self) -> Dict[str, Any]:
        return dict(load_auth(self.auth_path))

    def _save_root(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        save_auth(payload, self.auth_path)
        return payload

    def load(self) -> Dict[str, Any]:
        root = self._load_root()
        providers = dict(root.get("oauth_providers") or {})
        return _normalize_auth_record(dict(providers.get("gemini_cli") or {}))

    def save(self, auth: Dict[str, Any]) -> Dict[str, Any]:
        root = self._load_root()
        providers = dict(root.get("oauth_providers") or {})
        normalized = _normalize_auth_record(auth)
        providers["gemini_cli"] = normalized
        root["oauth_providers"] = providers
        self._save_root(root)
        return normalized

    def import_gemini_cli_auth(self, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        auth_path = path or (Path.home() / ".gemini" / "oauth_creds.json")
        if not auth_path.exists():
            return None
        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        record = {
            "provider": "google-gemini-cli",
            "tokens": {
                "access_token": payload.get("access_token") or payload.get("access") or payload.get("accessToken"),
                "refresh_token": payload.get("refresh_token") or payload.get("refresh") or payload.get("refreshToken"),
                "expires_at": payload.get("expires_at") or payload.get("expires") or payload.get("expiresAt"),
                "email": payload.get("email"),
                "project_id": payload.get("project_id") or payload.get("projectId"),
            },
            "last_refresh": _utc_now(),
            "source": str(auth_path),
        }
        normalized = _normalize_auth_record(record)
        refresh_token = str((normalized.get("tokens") or {}).get("refresh_token") or "").strip()
        access_token = str((normalized.get("tokens") or {}).get("access_token") or "").strip()
        if not refresh_token and not access_token:
            return None
        if refresh_token and token_expired((normalized.get("tokens") or {}).get("expires_at")):
            refreshed = self.refresh(record=normalized)
            return refreshed
        return self.save(normalized)

    def login(
        self,
        *,
        open_browser: bool = True,
        timeout_seconds: int = 300,
        prompt_for_redirect: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        verifier, challenge = _pkce_pair()
        state = verifier
        auth_url = build_authorize_url(code_challenge=challenge, state=state)

        params: Dict[str, str]
        if prompt_for_redirect is not None:
            if open_browser:
                try:
                    webbrowser.open(auth_url)
                except Exception:
                    pass
            params = _parse_manual_callback(prompt_for_redirect(auth_url), verifier)
        else:
            server = self._create_callback_server()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                if open_browser:
                    webbrowser.open(auth_url)
                if not server.event.wait(timeout_seconds):  # type: ignore[attr-defined]
                    raise GeminiCliOAuthError("Timed out waiting for Gemini CLI OAuth callback")
                params = dict(getattr(server, "result", {}) or {})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        if str(params.get("state") or "").strip() != state:
            raise GeminiCliOAuthError("Gemini CLI OAuth callback state mismatch")
        if params.get("error"):
            raise GeminiCliOAuthError(
                f"Gemini CLI OAuth failed: {params.get('error_description') or params['error']}"
            )
        code = str(params.get("code") or "").strip()
        if not code:
            raise GeminiCliOAuthError("Gemini CLI OAuth callback did not include a code")

        tokens = exchange_code_for_tokens(code=code, code_verifier=verifier)
        record = {
            "provider": "google-gemini-cli",
            "tokens": tokens,
            "last_refresh": _utc_now(),
        }
        return self.save(record)

    def refresh(self, record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        auth = _normalize_auth_record(record or self.load())
        tokens = dict(auth.get("tokens") or {})
        refresh_token = str(tokens.get("refresh_token") or "").strip()
        if not refresh_token:
            raise GeminiCliOAuthError("No refresh token is available for Gemini CLI OAuth")
        refreshed = refresh_access_token(refresh_token)
        project_id = str(tokens.get("project_id") or "").strip()
        email = str(tokens.get("email") or "").strip()
        if not project_id or not email:
            identity = resolve_google_oauth_identity(refreshed["access_token"])
            project_id = project_id or str(identity.get("project_id") or "").strip()
            email = email or str(identity.get("email") or "").strip()
        auth["provider"] = "google-gemini-cli"
        auth["tokens"] = {
            **refreshed,
            "project_id": project_id,
            "email": email,
        }
        auth["last_refresh"] = _utc_now()
        return self.save(auth)

    def ensure_access_token(self, refresh_if_needed: bool = True) -> Optional[str]:
        auth = self.load()
        tokens = dict(auth.get("tokens") or {})
        access_token = str(tokens.get("access_token") or "").strip()
        refresh_token = str(tokens.get("refresh_token") or "").strip()
        expires_at = tokens.get("expires_at")
        if refresh_if_needed and refresh_token and (not access_token or token_expired(expires_at)):
            auth = self.refresh(auth)
            tokens = dict(auth.get("tokens") or {})
            access_token = str(tokens.get("access_token") or "").strip()
        return access_token or None

    def ensure_access_token_with_import_fallback(
        self,
        *,
        refresh_if_needed: bool = True,
        import_if_missing: bool = True,
    ) -> Optional[str]:
        access_token = self.ensure_access_token(refresh_if_needed=refresh_if_needed)
        if access_token or not import_if_missing:
            return access_token
        imported = self.import_gemini_cli_auth()
        if not imported:
            return None
        return self.ensure_access_token(refresh_if_needed=refresh_if_needed)

    def build_api_key_payload(
        self,
        *,
        refresh_if_needed: bool = True,
        import_if_missing: bool = True,
    ) -> Optional[str]:
        access_token = self.ensure_access_token_with_import_fallback(
            refresh_if_needed=refresh_if_needed,
            import_if_missing=import_if_missing,
        )
        if not access_token:
            return None
        auth = self.load()
        tokens = dict((auth.get("tokens") or {}))
        project_id = str(tokens.get("project_id") or "").strip()
        if not project_id:
            try:
                identity = resolve_google_oauth_identity(access_token)
            except Exception as exc:
                logger.warning("gemini_cli_oauth_project_resolution_failed error=%s", exc)
                identity = {}
            resolved_project_id = str(identity.get("project_id") or "").strip()
            resolved_email = str(identity.get("email") or tokens.get("email") or "").strip()
            if resolved_project_id or resolved_email:
                auth["tokens"] = {
                    **tokens,
                    "access_token": access_token,
                    "project_id": resolved_project_id or project_id,
                    "email": resolved_email,
                }
                auth["last_refresh"] = _utc_now()
                auth = self.save(auth)
                tokens = dict((auth.get("tokens") or {}))
                project_id = str(tokens.get("project_id") or "").strip()
        payload = {"token": access_token}
        if project_id:
            payload["projectId"] = project_id
        return json.dumps(payload, separators=(",", ":"))

    @staticmethod
    def _create_callback_server() -> ThreadingHTTPServer:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", GOOGLE_CALLBACK_PORT), _OAuthCallbackHandler)
        except OSError as exc:
            raise GeminiCliOAuthError(
                f"Could not start Gemini CLI OAuth callback server on port {GOOGLE_CALLBACK_PORT}"
            ) from exc
        server.event = threading.Event()  # type: ignore[attr-defined]
        server.result = {}  # type: ignore[attr-defined]
        return server
