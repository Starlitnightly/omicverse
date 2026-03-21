"""
OpenAI OAuth helpers for Jarvis.
"""
from __future__ import annotations

import base64
import hashlib
import json
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

from .config import default_auth_path, load_auth, load_codex_auth, save_auth


OPENAI_AUTH_ISSUER = "https://auth.openai.com"
OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_ORIGINATOR = "pi"
OPENAI_CALLBACK_PORT = 1455
OPENAI_SCOPE = "openid profile email offline_access"
OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
OPENAI_CODEX_API_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
OPENAI_DEVICE_AUTH_URL = f"{OPENAI_AUTH_ISSUER}/api/accounts/deviceauth"
OPENAI_DEVICE_AUTH_CALLBACK = f"{OPENAI_AUTH_ISSUER}/deviceauth/callback"
OPENAI_DEVICE_AUTH_PAGE = f"{OPENAI_AUTH_ISSUER}/codex/device"
OAUTH_POLLING_SAFETY_MARGIN_S = 3


class OpenAIOAuthError(RuntimeError):
    """Raised when OpenAI OAuth login or refresh fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _pkce_pair() -> Tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    parts = (token or "").split(".")
    if len(parts) != 3 or not parts[1]:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def jwt_auth_claims(token: str) -> Dict[str, Any]:
    payload = _decode_jwt_payload(token)
    nested = payload.get("https://api.openai.com/auth")
    return nested if isinstance(nested, dict) else {}


def _extract_account_id_from_claims(payload: Dict[str, Any]) -> Optional[str]:
    """Extract chatgpt_account_id from JWT claims, checking 3 locations."""
    # Location 1: top-level field
    account_id = str(payload.get("chatgpt_account_id") or "").strip()
    if account_id:
        return account_id

    # Location 2: nested under https://api.openai.com/auth namespace
    nested = payload.get("https://api.openai.com/auth")
    if isinstance(nested, dict):
        account_id = str(nested.get("chatgpt_account_id") or "").strip()
        if account_id:
            return account_id

    # Location 3: first entry in organizations array
    orgs = payload.get("organizations")
    if isinstance(orgs, list) and orgs:
        first_org = orgs[0]
        if isinstance(first_org, dict):
            org_id = str(first_org.get("id") or "").strip()
            if org_id:
                return org_id

    return None


def extract_account_id(tokens: Dict[str, str]) -> Optional[str]:
    """Extract account ID from tokens, preferring id_token over access_token."""
    for key in ("id_token", "access_token"):
        token = str(tokens.get(key) or "").strip()
        if token:
            payload = _decode_jwt_payload(token)
            account_id = _extract_account_id_from_claims(payload)
            if account_id:
                return account_id
    return None


def jwt_org_context(token: str) -> Dict[str, str]:
    payload = _decode_jwt_payload(token)
    nested = payload.get("https://api.openai.com/auth")
    claims = nested if isinstance(nested, dict) else {}
    context: Dict[str, str] = {}
    for key in ("organization_id", "project_id"):
        value = str(claims.get(key) or "").strip()
        if value:
            context[key] = value
    # Use the improved 3-location extraction for account_id
    account_id = _extract_account_id_from_claims(payload)
    if account_id:
        context["chatgpt_account_id"] = account_id
    return context


def token_expired(token: str, skew_seconds: int = 300) -> bool:
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)):
        return True
    return time.time() >= (float(exp) - skew_seconds)


def build_authorize_url(
    *,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    workspace_id: Optional[str] = None,
) -> str:
    query = {
        "response_type": "code",
        "client_id": OPENAI_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OPENAI_SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": OPENAI_ORIGINATOR,
    }
    if workspace_id:
        query["allowed_workspace_id"] = workspace_id
    return f"{OPENAI_AUTH_ISSUER}/oauth/authorize?{urlencode(query)}"


def _token_endpoint() -> str:
    return f"{OPENAI_AUTH_ISSUER}/oauth/token"


def exchange_code_for_tokens(
    *,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> Dict[str, str]:
    resp = requests.post(
        _token_endpoint(),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": OPENAI_CLIENT_ID,
            "code_verifier": code_verifier,
        },
        timeout=30,
    )
    if not resp.ok:
        raise OpenAIOAuthError(f"OAuth token exchange failed: HTTP {resp.status_code} {resp.text[:300]}")
    data = resp.json()
    if not all(data.get(key) for key in ("id_token", "access_token", "refresh_token")):
        raise OpenAIOAuthError("OAuth token exchange returned incomplete credentials")
    return {
        "id_token": str(data["id_token"]),
        "access_token": str(data["access_token"]),
        "refresh_token": str(data["refresh_token"]),
    }


def refresh_chatgpt_tokens(refresh_token: str) -> Dict[str, str]:
    resp = requests.post(
        _token_endpoint(),
        data={
            "client_id": OPENAI_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    if not resp.ok:
        raise OpenAIOAuthError(f"OpenAI token refresh failed: HTTP {resp.status_code} {resp.text[:300]}")
    data = resp.json()
    access_token = str(data.get("access_token") or "").strip()
    id_token = str(data.get("id_token") or "").strip()
    next_refresh = str(data.get("refresh_token") or refresh_token).strip()
    if not access_token or not id_token:
        raise OpenAIOAuthError("OpenAI token refresh returned incomplete credentials")
    return {
        "id_token": id_token,
        "access_token": access_token,
        "refresh_token": next_refresh,
    }


def _normalize_auth_record(record: Dict[str, Any]) -> Dict[str, Any]:
    auth = dict(record)
    tokens = dict(auth.get("tokens") or {})
    if tokens:
        token_for_claims = str(tokens.get("id_token") or tokens.get("access_token") or "")
        context = jwt_org_context(token_for_claims)
        account_id = str(tokens.get("account_id") or context.get("chatgpt_account_id") or "").strip()
        if account_id:
            tokens["account_id"] = account_id
        organization_id = str(tokens.get("organization_id") or context.get("organization_id") or "").strip()
        if organization_id:
            tokens["organization_id"] = organization_id
        project_id = str(tokens.get("project_id") or context.get("project_id") or "").strip()
        if project_id:
            tokens["project_id"] = project_id
        auth["tokens"] = tokens
        auth["provider"] = "openai-codex"
    if "last_refresh" not in auth or not auth.get("last_refresh"):
        auth["last_refresh"] = _utc_now()
    return auth


def _parse_manual_callback(value: str) -> Dict[str, str]:
    text = (value or "").strip()
    if not text:
        raise OpenAIOAuthError("Missing OAuth callback URL or code/state pair")
    if "://" in text:
        parsed = urlparse(text)
        params = parse_qs(parsed.query)
        flat = {key: values[-1] for key, values in params.items() if values}
        if flat:
            return flat
    if "#" in text and "://" not in text:
        code, state = text.split("#", 1)
        return {"code": code.strip(), "state": state.strip()}
    raise OpenAIOAuthError("Could not parse OAuth callback input")


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    server_version = "JarvisOAuth/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_error(404)
            return
        params = {key: values[-1] for key, values in parse_qs(parsed.query).items() if values}
        self.server.result = params  # type: ignore[attr-defined]
        self.server.event.set()  # type: ignore[attr-defined]
        body = (
            "<html><body><h3>OpenAI OAuth complete</h3>"
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


class OpenAIOAuthManager:
    """Manage Jarvis OpenAI OAuth state."""

    def __init__(self, auth_path: Optional[Path] = None) -> None:
        self.auth_path = auth_path or default_auth_path()

    def load(self) -> Dict[str, Any]:
        return _normalize_auth_record(load_auth(self.auth_path))

    def save(self, auth: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(load_auth(self.auth_path))
        merged.update(auth)
        normalized = _normalize_auth_record(merged)
        save_auth(normalized, self.auth_path)
        return normalized

    def import_codex_auth(self) -> Optional[Dict[str, Any]]:
        codex_auth = load_codex_auth()
        tokens = dict(codex_auth.get("tokens") or {})
        if not tokens:
            return None
        access_token = str(tokens.get("access_token") or "").strip()
        id_token = str(tokens.get("id_token") or "").strip()
        if not access_token and not id_token:
            return None
        refresh_token = str(tokens.get("refresh_token") or "").strip()
        record = {
            "provider": "openai-codex",
            "tokens": tokens,
            "last_refresh": codex_auth.get("last_refresh") or _utc_now(),
        }
        token_for_expiry = access_token or id_token
        if refresh_token and token_expired(token_for_expiry):
            refreshed = refresh_chatgpt_tokens(refresh_token)
            claims = jwt_org_context(refreshed["id_token"])
            record["tokens"] = {
                **refreshed,
                "account_id": claims.get("chatgpt_account_id"),
                "organization_id": claims.get("organization_id"),
                "project_id": claims.get("project_id"),
            }
            record["last_refresh"] = _utc_now()
        return self.save(record)

    def login(
        self,
        *,
        workspace_id: Optional[str] = None,
        open_browser: bool = True,
        timeout_seconds: int = 300,
        prompt_for_redirect: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        verifier, challenge = _pkce_pair()
        state = _b64url(secrets.token_bytes(24))

        event = threading.Event()
        server = self._create_callback_server(event)
        host, port = server.server_address
        redirect_uri = f"http://localhost:{port}/auth/callback"
        auth_url = build_authorize_url(
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=challenge,
            workspace_id=workspace_id,
        )

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            if open_browser:
                webbrowser.open(auth_url)
            if not event.wait(timeout_seconds):
                if prompt_for_redirect is None:
                    raise OpenAIOAuthError("Timed out waiting for OpenAI OAuth callback")
                manual = prompt_for_redirect(auth_url)
                params = _parse_manual_callback(manual)
            else:
                params = dict(getattr(server, "result", {}) or {})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        if params.get("state") != state:
            raise OpenAIOAuthError("OpenAI OAuth callback state mismatch")
        if params.get("error"):
            detail = str(params.get("error_description") or params["error"])
            raise OpenAIOAuthError(f"OpenAI OAuth failed: {detail}")

        code = str(params.get("code") or "").strip()
        if not code:
            raise OpenAIOAuthError("OpenAI OAuth callback did not include a code")

        tokens = exchange_code_for_tokens(
            code=code,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
        )
        claims = jwt_org_context(tokens["id_token"])
        record = {
            "provider": "openai-codex",
            "tokens": {
                **tokens,
                "account_id": claims.get("chatgpt_account_id"),
                "organization_id": claims.get("organization_id"),
                "project_id": claims.get("project_id"),
            },
            "last_refresh": _utc_now(),
        }
        return self.save(record)

    def login_device(
        self,
        *,
        timeout_seconds: int = 300,
        on_user_code: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """Device Flow login for headless environments (SSH, etc.).

        Parameters
        ----------
        timeout_seconds : int
            Maximum seconds to wait for the user to complete authorization.
        on_user_code : callable, optional
            Called with ``(url, user_code)`` so the caller can display them.
            If *None*, the URL and code are printed to stdout.
        """
        # Step 1: Request device authorization code
        resp = requests.post(
            f"{OPENAI_DEVICE_AUTH_URL}/usercode",
            json={"client_id": OPENAI_CLIENT_ID},
            timeout=30,
        )
        if not resp.ok:
            raise OpenAIOAuthError(
                f"Device auth usercode request failed: HTTP {resp.status_code} {resp.text[:300]}"
            )
        data = resp.json()
        device_auth_id = str(data.get("device_auth_id") or "").strip()
        user_code = str(data.get("user_code") or "").strip()
        server_interval = max(int(data.get("interval") or 1), 1)
        if not device_auth_id or not user_code:
            raise OpenAIOAuthError("Device auth returned incomplete response")

        # Step 2: Display code to user
        if on_user_code is not None:
            on_user_code(OPENAI_DEVICE_AUTH_PAGE, user_code)
        else:
            print(f"\nOpen this URL: {OPENAI_DEVICE_AUTH_PAGE}")
            print(f"Enter code:    {user_code}\n")

        # Step 3: Poll for authorization completion
        poll_interval = server_interval + OAUTH_POLLING_SAFETY_MARGIN_S
        deadline = time.time() + timeout_seconds
        authorization_code: Optional[str] = None
        code_verifier: Optional[str] = None

        while time.time() < deadline:
            time.sleep(poll_interval)
            poll_resp = requests.post(
                f"{OPENAI_DEVICE_AUTH_URL}/token",
                json={"device_auth_id": device_auth_id, "user_code": user_code},
                timeout=30,
            )
            if poll_resp.status_code in (403, 404):
                # User hasn't completed authorization yet
                continue
            if not poll_resp.ok:
                raise OpenAIOAuthError(
                    f"Device auth polling failed: HTTP {poll_resp.status_code} {poll_resp.text[:300]}"
                )
            poll_data = poll_resp.json()
            authorization_code = str(poll_data.get("authorization_code") or "").strip()
            code_verifier = str(poll_data.get("code_verifier") or "").strip()
            if authorization_code:
                break

        if not authorization_code or not code_verifier:
            raise OpenAIOAuthError("Timed out waiting for device authorization")

        # Step 4: Exchange authorization code for tokens
        tokens = exchange_code_for_tokens(
            code=authorization_code,
            redirect_uri=OPENAI_DEVICE_AUTH_CALLBACK,
            code_verifier=code_verifier,
        )
        claims = jwt_org_context(tokens["id_token"])
        record = {
            "provider": "openai-codex",
            "tokens": {
                **tokens,
                "account_id": claims.get("chatgpt_account_id"),
                "organization_id": claims.get("organization_id"),
                "project_id": claims.get("project_id"),
            },
            "last_refresh": _utc_now(),
        }
        return self.save(record)

    def refresh(self) -> Dict[str, Any]:
        auth = self.load()
        tokens = dict(auth.get("tokens") or {})
        refresh_token = str(tokens.get("refresh_token") or "").strip()
        if not refresh_token:
            raise OpenAIOAuthError("No refresh token is available for OpenAI OAuth")
        refreshed = refresh_chatgpt_tokens(refresh_token)
        claims = jwt_org_context(refreshed["id_token"])
        auth["provider"] = "openai-codex"
        auth["tokens"] = {
            **refreshed,
            "account_id": claims.get("chatgpt_account_id"),
            "organization_id": claims.get("organization_id"),
            "project_id": claims.get("project_id"),
        }
        auth["last_refresh"] = _utc_now()
        return self.save(auth)

    def ensure_access_token(self, refresh_if_needed: bool = True) -> Optional[str]:
        auth = self.load()
        tokens = dict(auth.get("tokens") or {})
        access_token = str(tokens.get("access_token") or "").strip()
        refresh_token = str(tokens.get("refresh_token") or "").strip()

        if refresh_if_needed and refresh_token and (not access_token or token_expired(access_token)):
            auth = self.refresh()
            tokens = dict(auth.get("tokens") or {})
            access_token = str(tokens.get("access_token") or "").strip()

        return access_token or None

    @staticmethod
    def _create_callback_server(event: threading.Event) -> ThreadingHTTPServer:
        for port in (OPENAI_CALLBACK_PORT, 0):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), _OAuthCallbackHandler)
                server.event = event  # type: ignore[attr-defined]
                server.result = {}  # type: ignore[attr-defined]
                return server
            except OSError:
                continue
        raise OpenAIOAuthError("Could not start a local OAuth callback server")


class CodexAPIClient:
    """HTTP client wrapper for Codex API with auto token refresh and URL rewriting.

    Intercepts requests to standard OpenAI endpoints and rewrites them to the
    ChatGPT Codex backend, injecting the correct Bearer token and account headers.
    """

    def __init__(self, oauth_manager: Optional[OpenAIOAuthManager] = None) -> None:
        self._oauth = oauth_manager or OpenAIOAuthManager()
        self._session = requests.Session()

    def _ensure_auth_headers(self) -> Dict[str, str]:
        """Return headers with a valid Bearer token, refreshing if needed."""
        access_token = self._oauth.ensure_access_token(refresh_if_needed=True)
        if not access_token:
            raise OpenAIOAuthError("No valid access token available; run login first")

        auth = self._oauth.load()
        tokens = dict(auth.get("tokens") or {})
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {access_token}",
            "originator": OPENAI_ORIGINATOR,
        }
        account_id = str(tokens.get("account_id") or "").strip()
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id
        return headers

    @staticmethod
    def _rewrite_url(url: str) -> str:
        """Rewrite standard OpenAI API URLs to the Codex endpoint."""
        parsed = urlparse(url)
        if "/v1/responses" in parsed.path or "/chat/completions" in parsed.path:
            return OPENAI_CODEX_API_ENDPOINT
        return url

    def request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 120,
        **kwargs: Any,
    ) -> requests.Response:
        """Send a request with Codex auth, URL rewriting, and auto token refresh."""
        final_url = self._rewrite_url(url)
        auth_headers = self._ensure_auth_headers()
        merged_headers = {**(headers or {}), **auth_headers}

        resp = self._session.request(
            method,
            final_url,
            json=json_body,
            headers=merged_headers,
            timeout=timeout,
            **kwargs,
        )

        # If 401, try one token refresh and retry
        if resp.status_code == 401:
            try:
                self._oauth.refresh()
            except OpenAIOAuthError:
                return resp
            auth_headers = self._ensure_auth_headers()
            merged_headers = {**(headers or {}), **auth_headers}
            resp = self._session.request(
                method,
                final_url,
                json=json_body,
                headers=merged_headers,
                timeout=timeout,
                **kwargs,
            )
        return resp

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", url, **kwargs)
