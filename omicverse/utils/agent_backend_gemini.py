"""Google Gemini adapter helpers for OmicVerseLLMBackend.

Internal module — import from ``omicverse.utils.agent_backend`` instead.
"""
from __future__ import annotations

import json
import logging
import ssl
import uuid
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError
from urllib.parse import quote

from .agent_backend_common import (
    Usage, ToolCall, ChatResponse,
    _coerce_int, _compute_total, _request_timeout_seconds,
)
from .model_config import get_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini CLI OAuth constants
# ---------------------------------------------------------------------------

_GOOGLE_GEMINI_CLI_BASE_URL = "https://cloudcode-pa.googleapis.com"
_GOOGLE_GEMINI_CLI_UNSUPPORTED_SCHEMA_KEYS = {
    "default",
    "patternProperties",
    "additionalProperties",
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "definitions",
    "examples",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "multipleOf",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
}


# ---------------------------------------------------------------------------
# Schema / message conversion utilities
# ---------------------------------------------------------------------------

def _json_schema_to_gemini_schema(schema: Dict) -> Any:
    """Convert JSON Schema to Gemini proto Schema (best-effort)."""
    try:
        import google.generativeai as genai  # type: ignore
        Type = genai.protos.Type

        type_map = {
            "string": Type.STRING,
            "number": Type.NUMBER,
            "integer": Type.INTEGER,
            "boolean": Type.BOOLEAN,
            "array": Type.ARRAY,
            "object": Type.OBJECT,
        }

        props = {}
        required = schema.get("required", [])
        for pname, pschema in schema.get("properties", {}).items():
            ptype = type_map.get(pschema.get("type", "string"), Type.STRING)
            enum_vals = pschema.get("enum")
            props[pname] = genai.protos.Schema(
                type=ptype,
                description=pschema.get("description", ""),
                **({"enum": enum_vals} if enum_vals else {}),
            )

        return genai.protos.Schema(
            type=Type.OBJECT,
            properties=props,
            required=required,
        )
    except (ImportError, AttributeError, KeyError, TypeError) as exc:
        logger.debug("Gemini schema conversion failed, returning None: %s", exc)
        return None


def _messages_to_gemini_contents(messages: List[Dict]) -> List[Any]:
    """Convert OpenAI-style messages to Gemini Content objects."""
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return []

    contents = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue  # system handled via system_instruction
        gemini_role = "model" if role == "assistant" else "user"

        content = m.get("content", "")
        if isinstance(content, str):
            parts = []
            if content:
                parts.append(genai.protos.Part(text=content))
            if role == "assistant":
                for tool_call in m.get("tool_calls") or []:
                    function = dict(tool_call.get("function") or {})
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, ValueError, TypeError) as exc:
                            logger.debug("Gemini SDK tool-call argument parse failed, wrapping raw: %s", exc)
                            arguments = {"raw": arguments}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    parts.append(genai.protos.Part(
                        function_call=genai.protos.FunctionCall(
                            name=function.get("name", tool_call.get("name", "unknown")),
                            args=arguments,
                        )
                    ))
            if parts:
                contents.append(genai.protos.Content(
                    role=gemini_role,
                    parts=parts,
                ))
        elif isinstance(content, list):
            # Anthropic-style tool_result content blocks
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_result":
                        parts.append(genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=block.get("name", "unknown"),
                                response={"result": block.get("content", "")},
                            )
                        ))
                    elif block.get("type") == "text":
                        parts.append(genai.protos.Part(text=block.get("text", "")))
            if parts:
                contents.append(genai.protos.Content(role=gemini_role, parts=parts))

        # Handle OpenAI tool role
        if role == "tool":
            tool_call_id = m.get("tool_call_id", "")
            tool_name = m.get("name", "unknown")
            contents.append(genai.protos.Content(
                role="user",
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": m.get("content", "")},
                    )
                )],
            ))

    return contents


# ---------------------------------------------------------------------------
# Single-turn simple chat
# ---------------------------------------------------------------------------

def _chat_via_gemini(backend, user_prompt: str) -> str:
    """Synchronous single-turn Gemini chat (no tool calling)."""
    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY for Gemini provider")
    if _gemini_uses_oauth_bearer(api_key):
        return _chat_via_gemini_rest(backend, user_prompt, api_key)
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=backend.config.model.split("/", 1)[-1],
            system_instruction=backend.config.system_prompt
        )

        # Generate config for temperature control
        generation_config = genai.types.GenerationConfig(
            temperature=backend.config.temperature,
            max_output_tokens=backend.config.max_tokens,
        )

        # Wrap Gemini SDK call with retry logic
        def _make_gemini_call():
            resp = model.generate_content(
                user_prompt,
                generation_config=generation_config
            )

            # Capture usage information from Gemini
            if hasattr(resp, 'usage_metadata') and resp.usage_metadata is not None:
                usage = resp.usage_metadata
                input_tokens = _coerce_int(getattr(usage, 'prompt_token_count', None))
                output_tokens = _coerce_int(getattr(usage, 'candidates_token_count', None))
                total_tokens = _compute_total(input_tokens, output_tokens, _coerce_int(getattr(usage, 'total_token_count', None)))
                if total_tokens is not None:
                    backend.last_usage = Usage(
                        input_tokens=input_tokens or 0,
                        output_tokens=output_tokens or 0,
                        total_tokens=total_tokens,
                        model=backend.config.model,
                        provider=backend.config.provider
                    )

            # Extract text robustly; Gemini may omit parts when finish_reason != 1
            text = ""
            try:
                text = getattr(resp, "text", "") or ""
            except (ValueError, AttributeError) as exc:
                logger.debug("Gemini response .text extraction failed, falling back to candidate parts: %s", exc)
                text = ""
            if not text and getattr(resp, "candidates", None):
                for cand in resp.candidates or []:
                    parts = getattr(getattr(cand, "content", None), "parts", None) or []
                    piece = "".join(str(getattr(p, "text", "") or "") for p in parts)
                    if piece:
                        text += piece
                text = text.strip()
            return text

        return backend._retry(_make_gemini_call)

    except ImportError:
        raise RuntimeError(
            "google-generativeai package not installed. Install it or choose an OpenAI-compatible model."
        )


# ---------------------------------------------------------------------------
# Multi-turn chat with tool / function calling
# ---------------------------------------------------------------------------

def _chat_tools_gemini(
    backend,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    tool_choice: Optional[str],
) -> ChatResponse:
    """Multi-turn chat via Gemini API with function calling support."""
    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY for Gemini provider")

    if _gemini_uses_oauth_bearer(api_key):
        return _chat_tools_gemini_rest(backend, messages, tools, tool_choice, api_key)

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)

        # Convert tools to Gemini format
        gemini_tools = None
        if tools:
            func_decls = []
            for t in tools:
                fd = genai.protos.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=_json_schema_to_gemini_schema(t.get("parameters", {})),
                )
                func_decls.append(fd)
            gemini_tools = [genai.protos.Tool(function_declarations=func_decls)]

        model = genai.GenerativeModel(
            model_name=backend.config.model.split("/", 1)[-1],
            system_instruction=backend.config.system_prompt,
            tools=gemini_tools,
        )

        generation_config = genai.types.GenerationConfig(
            temperature=backend.config.temperature,
            max_output_tokens=backend.config.max_tokens,
        )

        # Convert messages to Gemini Content format
        gemini_contents = _messages_to_gemini_contents(messages)

        def _make_call():
            resp = model.generate_content(
                gemini_contents,
                generation_config=generation_config,
            )

            # Capture usage
            usage_obj = None
            if hasattr(resp, 'usage_metadata') and resp.usage_metadata is not None:
                usage = resp.usage_metadata
                it = _coerce_int(getattr(usage, 'prompt_token_count', None))
                ot = _coerce_int(getattr(usage, 'candidates_token_count', None))
                tt = _compute_total(it, ot, _coerce_int(getattr(usage, 'total_token_count', None)))
                if tt is not None:
                    usage_obj = Usage(
                        input_tokens=it or 0,
                        output_tokens=ot or 0,
                        total_tokens=tt,
                        model=backend.config.model,
                        provider=backend.config.provider
                    )
                    backend.last_usage = usage_obj

            # Extract content and function calls
            content_parts = []
            tc_list = []
            candidate = (resp.candidates or [None])[0] if hasattr(resp, 'candidates') else None
            if candidate and hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts or []:
                    if hasattr(part, 'text') and part.text:
                        content_parts.append(part.text)
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tc_list.append(ToolCall(
                            id=f"gemini_{fc.name}_{uuid.uuid4().hex[:12]}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        ))

            content = "\n".join(content_parts) or None
            stop = "tool_use" if tc_list else "end_turn"

            return ChatResponse(
                content=content,
                tool_calls=tc_list,
                stop_reason=stop,
                usage=usage_obj,
                raw_message={
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tc_list
                    ],
                } if (content or tc_list) else None,
            )

        return backend._retry(_make_call)

    except ImportError:
        raise RuntimeError(
            "google-generativeai package not installed. Install it for tool-calling support."
        )


# ---------------------------------------------------------------------------
# Gemini CLI OAuth bearer helpers
# ---------------------------------------------------------------------------

def _gemini_uses_oauth_bearer(api_key: str) -> bool:
    """Check if the API key is a JSON OAuth bearer token payload."""
    text = str(api_key or "")
    if not text.startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.debug("Gemini OAuth bearer check: JSON parse failed: %s", exc)
        return False
    token = str((payload or {}).get("token") or "").strip() if isinstance(payload, dict) else ""
    return bool(token)


def _gemini_auth_headers(api_key: str) -> Dict[str, str]:
    """Build auth headers: OAuth bearer if JSON payload, else x-goog-api-key."""
    text = str(api_key or "")
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.debug("Gemini auth header JSON parse failed, falling back to API key: %s", exc)
            payload = None
        if isinstance(payload, dict):
            token = str(payload.get("token") or "").strip()
            if token:
                return {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }
    return {
        "x-goog-api-key": text,
        "Content-Type": "application/json",
    }


def _gemini_oauth_payload(api_key: str) -> Optional[Dict[str, str]]:
    """Extract OAuth token and project ID from a JSON API key payload."""
    text = str(api_key or "")
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.debug("Gemini OAuth payload JSON parse failed: %s", exc)
        return None
    if not isinstance(payload, dict):
        return None
    token = str(payload.get("token") or "").strip()
    project_id = str(payload.get("projectId") or payload.get("project_id") or "").strip()
    if not token:
        return None
    return {
        "token": token,
        "projectId": project_id,
    }


def _gemini_base_url(backend) -> str:
    """Resolve the Gemini REST base URL from config or provider defaults."""
    info = get_provider(backend.config.provider)
    return str(
        backend.config.endpoint
        or (info.base_url if info else "https://generativelanguage.googleapis.com/v1beta")
    ).rstrip("/")


def _gemini_generate_content_url(backend) -> str:
    return f"{_gemini_base_url(backend)}/models/{quote(backend._wire_model_name(), safe='')}:generateContent"


def _gemini_cli_base_url(backend) -> str:
    endpoint = str(backend.config.endpoint or "").strip().rstrip("/")
    return endpoint or _GOOGLE_GEMINI_CLI_BASE_URL


def _gemini_cli_generate_content_url(backend, stream: bool = False) -> str:
    suffix = ":streamGenerateContent?alt=sse" if stream else ":generateContent"
    return f"{_gemini_cli_base_url(backend)}/v1internal{suffix}"


def _gemini_function_response_payload(result: Any) -> Dict[str, Any]:
    """Normalize a tool result into a Gemini-compatible function response dict."""
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {"output": result}
    if isinstance(result, str):
        stripped = result.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.debug("Gemini function response JSON parse failed, wrapping as output: %s", exc)
                parsed = None
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"output": parsed}
        return {"output": result}
    return {"output": result}


def _clean_schema_for_gemini_cli(schema: Any) -> Any:
    """Strip unsupported JSON Schema keys for the Gemini CLI REST API."""
    if isinstance(schema, list):
        return [_clean_schema_for_gemini_cli(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    cleaned: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in _GOOGLE_GEMINI_CLI_UNSUPPORTED_SCHEMA_KEYS:
            continue
        if key in {"anyOf", "oneOf"} and isinstance(value, list):
            preferred = None
            for item in value:
                if isinstance(item, dict) and item.get("type") == "array":
                    preferred = item
                    break
            if preferred is not None:
                merged = dict(preferred)
                if schema.get("description") and not merged.get("description"):
                    merged["description"] = schema.get("description")
                return _clean_schema_for_gemini_cli(merged)
            continue
        cleaned_value = _clean_schema_for_gemini_cli(value)
        if key == "type" and isinstance(cleaned_value, str):
            cleaned_value = cleaned_value.upper()
        cleaned[key] = cleaned_value
    return cleaned


def _messages_to_gemini_rest_contents(backend, messages: List[Dict]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages to Gemini REST JSON contents."""
    contents: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "system":
            continue
        parts: List[Dict[str, Any]] = []
        content = message.get("content", "")
        if isinstance(content, str) and content:
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    parts.append({"text": block.get("text", "")})
                elif block.get("type") == "tool_result":
                    parts.append({
                        "functionResponse": {
                            "name": block.get("name", "unknown"),
                            "response": _gemini_function_response_payload(block.get("content", "")),
                        }
                    })
        elif isinstance(content, dict) and content:
            parts.append({"text": json.dumps(content, ensure_ascii=False)})

        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = dict(tool_call.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, ValueError, TypeError) as exc:
                        logger.debug("Gemini REST tool-call argument parse failed, wrapping raw: %s", exc)
                        arguments = {"raw": arguments}
                if not isinstance(arguments, dict):
                    arguments = {}
                parts.append({
                    "functionCall": {
                        "name": function.get("name", tool_call.get("name", "unknown")),
                        "args": arguments,
                    }
                })
        elif role == "tool":
            parts = [{
                "functionResponse": {
                    "name": message.get("name", "unknown"),
                    "response": _gemini_function_response_payload(message.get("content", "")),
                }
            }]

        if parts:
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            })
    return contents


def _gemini_rest_generation_config(backend) -> Dict[str, Any]:
    return {
        "temperature": backend.config.temperature,
        "maxOutputTokens": backend.config.max_tokens,
    }


def _gemini_rest_request(backend, body: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Send a generateContent request to the standard Gemini REST endpoint."""
    url = _gemini_generate_content_url(backend)
    headers = _gemini_auth_headers(api_key)
    data = json.dumps(body).encode("utf-8")
    req = urllib_request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    with urllib_request.urlopen(req, context=ctx, timeout=_request_timeout_seconds()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _gemini_cli_request(backend, body: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Send a generateContent request to the Gemini CLI internal endpoint."""
    oauth = _gemini_oauth_payload(api_key)
    if not oauth:
        raise RuntimeError("Gemini CLI OAuth payload is missing bearer token")
    url = _gemini_cli_generate_content_url(backend)
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {oauth['token']}",
        "Content-Type": "application/json",
        "User-Agent": "GeminiCLI/v23.5.0 (darwin; arm64) google-api-nodejs-client/9.15.1",
        "x-goog-api-client": "gl-python/omicverse",
        "Accept": "application/json",
    }
    req = urllib_request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    try:
        with urllib_request.urlopen(req, context=ctx, timeout=_request_timeout_seconds()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="ignore").strip()
        except (OSError, UnicodeDecodeError) as read_exc:
            logger.debug("Failed to read Gemini CLI HTTPError body: %s", read_exc)
            body_text = ""
        detail = body_text[:2000] if body_text else str(exc)
        raise RuntimeError(
            f"Gemini CLI generateContent failed: HTTP {exc.code}: {detail}"
        ) from exc


def _capture_gemini_usage(backend, payload: Dict[str, Any]) -> Optional[Usage]:
    """Extract usage metadata from a Gemini REST response payload."""
    response_payload = payload.get("response") if isinstance(payload.get("response"), dict) else payload
    usage_data = response_payload.get("usageMetadata")
    if not isinstance(usage_data, dict):
        return None
    input_tokens = _coerce_int(usage_data.get("promptTokenCount"))
    output_tokens = _coerce_int(usage_data.get("candidatesTokenCount"))
    total_tokens = _compute_total(
        input_tokens,
        output_tokens,
        _coerce_int(usage_data.get("totalTokenCount")),
    )
    if total_tokens is None:
        return None
    usage = Usage(
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        total_tokens=total_tokens,
        model=backend.config.model,
        provider=backend.config.provider,
    )
    backend.last_usage = usage
    return usage


def _extract_gemini_text_and_tool_calls(payload: Dict[str, Any]):
    """Extract text, tool calls, raw_message, and stop reason from a Gemini REST response."""
    response_payload = payload.get("response") if isinstance(payload.get("response"), dict) else payload
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        logger.warning("No Gemini candidates returned: %s", response_payload)
        return None, [], None, "end_turn"

    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    content = candidate.get("content")
    parts = list((content or {}).get("parts") or []) if isinstance(content, dict) else []
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
        function_call = part.get("functionCall")
        if isinstance(function_call, dict):
            name = str(function_call.get("name") or "unknown").strip() or "unknown"
            args = function_call.get("args")
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(ToolCall(
                id=f"gemini_{name}_{uuid.uuid4().hex[:12]}",
                name=name,
                arguments=args,
            ))

    stop_reason = "tool_use" if tool_calls else "end_turn"
    finish_reason = str(candidate.get("finishReason") or "").upper()
    if finish_reason == "MAX_TOKENS":
        stop_reason = "max_tokens"
    content_text = "\n".join(text_parts).strip() or None
    raw_message = {
        "role": "assistant",
        "content": content_text,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ],
    } if (content_text or tool_calls) else None
    return content_text, tool_calls, raw_message, stop_reason


# ---------------------------------------------------------------------------
# Gemini CLI REST chat methods
# ---------------------------------------------------------------------------

def _chat_tools_gemini_rest(
    backend,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    tool_choice: Optional[str],
    api_key: str,
) -> ChatResponse:
    """Multi-turn chat via Gemini CLI REST API with function calling support."""
    body: Dict[str, Any] = {
        "model": backend._wire_model_name(),
        "request": {
            "contents": _messages_to_gemini_rest_contents(backend, messages),
            "generationConfig": _gemini_rest_generation_config(backend),
        },
    }
    oauth = _gemini_oauth_payload(api_key) or {}
    if oauth.get("projectId"):
        body["project"] = oauth["projectId"]
    if backend.config.system_prompt:
        body["request"]["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": backend.config.system_prompt}],
        }
    if tools:
        body["request"]["tools"] = [{
            "functionDeclarations": [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": _clean_schema_for_gemini_cli(tool.get("parameters", {}) or {}),
                }
                for tool in tools
            ]
        }]

    def _make_call() -> ChatResponse:
        payload = _gemini_cli_request(backend, body, api_key)
        usage = _capture_gemini_usage(backend, payload)
        content, tool_calls, raw_message, stop_reason = _extract_gemini_text_and_tool_calls(payload)
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            raw_message=raw_message,
        )

    return backend._retry(_make_call)


def _chat_via_gemini_rest(backend, user_prompt: str, api_key: str) -> str:
    """Single-turn chat via Gemini CLI REST API (no tool calling)."""
    body: Dict[str, Any] = {
        "model": backend._wire_model_name(),
        "request": {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": _gemini_rest_generation_config(backend),
        },
    }
    oauth = _gemini_oauth_payload(api_key) or {}
    if oauth.get("projectId"):
        body["project"] = oauth["projectId"]
    if backend.config.system_prompt:
        body["request"]["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": backend.config.system_prompt}],
        }

    def _make_call() -> str:
        payload = _gemini_cli_request(backend, body, api_key)
        _capture_gemini_usage(backend, payload)
        content, _tool_calls, _raw_message, _stop_reason = _extract_gemini_text_and_tool_calls(payload)
        return content or ""

    return backend._retry(_make_call)
