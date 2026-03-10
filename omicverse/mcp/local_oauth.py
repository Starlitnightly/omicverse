"""
Minimal localhost OAuth provider for Streamable HTTP MCP development.

This provider is intentionally simple:
- dynamic client registration is stored in memory
- /authorize auto-approves and redirects immediately
- access/refresh tokens are stored in memory only

It is suitable for local MCP development on 127.0.0.1/localhost, not for
internet-facing deployment.
"""

from __future__ import annotations

import secrets
import time
from typing import Dict, Optional

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    ProviderTokenVerifier,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


LOCAL_AUTH_SCOPES = ["openid", "profile", "offline_access", "mcp"]


class LocalOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """In-memory OAuth provider for localhost MCP servers."""

    def __init__(self):
        self._clients: Dict[str, OAuthClientInformationFull] = {}
        self._auth_codes: Dict[str, AuthorizationCode] = {}
        self._refresh_tokens: Dict[str, RefreshToken] = {}
        self._access_tokens: Dict[str, AccessToken] = {}

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        now = time.time()
        code = secrets.token_urlsafe(32)
        scopes = params.scopes
        if scopes is None:
            scopes = client.scope.split(" ") if client.scope else []

        self._auth_codes[code] = AuthorizationCode(
            code=code,
            scopes=scopes,
            expires_at=now + 300,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        return construct_redirect_uri(
            str(params.redirect_uri),
            code=code,
            state=params.state,
        )

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> Optional[AuthorizationCode]:
        auth_code = self._auth_codes.get(authorization_code)
        if auth_code is None:
            return None
        if auth_code.client_id != client.client_id:
            return None
        return auth_code

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        self._auth_codes.pop(authorization_code.code, None)
        now = int(time.time())
        access_token_str = secrets.token_urlsafe(32)
        refresh_token_str = secrets.token_urlsafe(32)

        access_token = AccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=now + 3600,
            resource=authorization_code.resource,
        )
        refresh_token = RefreshToken(
            token=refresh_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=now + 30 * 24 * 3600,
        )
        self._access_tokens[access_token_str] = access_token
        self._refresh_tokens[refresh_token_str] = refresh_token

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=refresh_token_str,
            scope=" ".join(authorization_code.scopes) if authorization_code.scopes else None,
        )

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> Optional[RefreshToken]:
        token = self._refresh_tokens.get(refresh_token)
        if token is None:
            return None
        if token.client_id != client.client_id:
            return None
        return token

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        self._refresh_tokens.pop(refresh_token.token, None)
        now = int(time.time())
        access_token_str = secrets.token_urlsafe(32)
        refresh_token_str = secrets.token_urlsafe(32)

        access_token = AccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=now + 3600,
        )
        new_refresh_token = RefreshToken(
            token=refresh_token_str,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=now + 30 * 24 * 3600,
        )
        self._access_tokens[access_token_str] = access_token
        self._refresh_tokens[refresh_token_str] = new_refresh_token

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=refresh_token_str,
            scope=" ".join(scopes) if scopes else None,
        )

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        access_token = self._access_tokens.get(token)
        if access_token is None:
            return None
        if access_token.expires_at is not None and access_token.expires_at < int(time.time()):
            self._access_tokens.pop(token, None)
            return None
        return access_token

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self._access_tokens.pop(token.token, None)
            return
        self._refresh_tokens.pop(token.token, None)

    def build_token_verifier(self) -> ProviderTokenVerifier:
        return ProviderTokenVerifier(self)
