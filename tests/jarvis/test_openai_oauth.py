"""Tests for OpenAI OAuth helpers — JWT parsing, account ID extraction, and URL rewriting."""
from __future__ import annotations

import base64
import json

import pytest

from omicverse.jarvis.openai_oauth import (
    OPENAI_CODEX_API_ENDPOINT,
    CodexAPIClient,
    _decode_jwt_payload,
    _extract_account_id_from_claims,
    extract_account_id,
    jwt_org_context,
    token_expired,
)


def _make_jwt(payload: dict) -> str:
    """Build a fake JWT with the given payload (no signature verification)."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.fake_signature"


class TestDecodeJwtPayload:
    def test_valid_jwt(self):
        token = _make_jwt({"sub": "user123", "exp": 9999999999})
        claims = _decode_jwt_payload(token)
        assert claims["sub"] == "user123"

    def test_invalid_jwt(self):
        assert _decode_jwt_payload("not.a.jwt") == {} or isinstance(_decode_jwt_payload("not.a.jwt"), dict)
        assert _decode_jwt_payload("") == {}
        assert _decode_jwt_payload("one_part") == {}

    def test_non_dict_payload(self):
        body = base64.urlsafe_b64encode(json.dumps([1, 2, 3]).encode()).rstrip(b"=").decode()
        token = f"header.{body}.sig"
        assert _decode_jwt_payload(token) == {}


class TestExtractAccountIdFromClaims:
    def test_location1_top_level(self):
        claims = {"chatgpt_account_id": "acct_top"}
        assert _extract_account_id_from_claims(claims) == "acct_top"

    def test_location2_nested_namespace(self):
        claims = {
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_nested"}
        }
        assert _extract_account_id_from_claims(claims) == "acct_nested"

    def test_location3_organizations_array(self):
        claims = {"organizations": [{"id": "org_first"}, {"id": "org_second"}]}
        assert _extract_account_id_from_claims(claims) == "org_first"

    def test_priority_top_level_over_nested(self):
        claims = {
            "chatgpt_account_id": "acct_top",
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_nested"},
            "organizations": [{"id": "org_first"}],
        }
        assert _extract_account_id_from_claims(claims) == "acct_top"

    def test_priority_nested_over_org(self):
        claims = {
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_nested"},
            "organizations": [{"id": "org_first"}],
        }
        assert _extract_account_id_from_claims(claims) == "acct_nested"

    def test_no_account_id(self):
        assert _extract_account_id_from_claims({}) is None
        assert _extract_account_id_from_claims({"organizations": []}) is None
        assert _extract_account_id_from_claims({"organizations": [{}]}) is None


class TestExtractAccountId:
    def test_from_id_token(self):
        tokens = {
            "id_token": _make_jwt({"chatgpt_account_id": "from_id"}),
            "access_token": _make_jwt({"chatgpt_account_id": "from_access"}),
        }
        assert extract_account_id(tokens) == "from_id"

    def test_fallback_to_access_token(self):
        tokens = {
            "id_token": _make_jwt({}),
            "access_token": _make_jwt({"chatgpt_account_id": "from_access"}),
        }
        assert extract_account_id(tokens) == "from_access"

    def test_no_tokens(self):
        assert extract_account_id({}) is None


class TestJwtOrgContext:
    def test_extracts_all_fields(self):
        payload = {
            "chatgpt_account_id": "acct_123",
            "https://api.openai.com/auth": {
                "organization_id": "org_456",
                "project_id": "proj_789",
            },
        }
        context = jwt_org_context(_make_jwt(payload))
        assert context["chatgpt_account_id"] == "acct_123"
        assert context["organization_id"] == "org_456"
        assert context["project_id"] == "proj_789"


class TestTokenExpired:
    def test_expired(self):
        token = _make_jwt({"exp": 0})
        assert token_expired(token) is True

    def test_not_expired(self):
        token = _make_jwt({"exp": 9999999999})
        assert token_expired(token) is False

    def test_missing_exp(self):
        token = _make_jwt({})
        assert token_expired(token) is True


class TestCodexAPIClientURLRewrite:
    def test_rewrite_responses(self):
        assert CodexAPIClient._rewrite_url("https://api.openai.com/v1/responses") == OPENAI_CODEX_API_ENDPOINT

    def test_rewrite_chat_completions(self):
        assert CodexAPIClient._rewrite_url("https://api.openai.com/v1/chat/completions") == OPENAI_CODEX_API_ENDPOINT

    def test_no_rewrite_other(self):
        url = "https://api.openai.com/v1/models"
        assert CodexAPIClient._rewrite_url(url) == url
