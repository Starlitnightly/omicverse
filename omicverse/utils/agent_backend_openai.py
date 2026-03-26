"""OpenAI-compatible and Responses API provider helpers for OmicVerseLLMBackend.

Internal module — import from ``omicverse.utils.agent_backend`` instead.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import platform
import re
import ssl
import warnings
from typing import Any, Callable, Dict, List, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from .agent_backend_common import (
    Usage,
    ToolCall,
    ChatResponse,
    _coerce_int,
    _compute_total,
    _request_timeout_seconds,
    _OPENAI_CODEX_BASE_URL,
    _OPENAI_CODEX_JWT_CLAIM_PATH,
)
from .model_config import ModelConfig, get_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI parameter / error adaptation helpers
# ---------------------------------------------------------------------------

def _apply_openai_chat_param_policy(backend, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Apply known provider/model-specific chat-completions parameter overrides.

    This mirrors OpenClaw's model-aware approach: treat provider/model
    capabilities as distinct from the generic OpenAI-compatible wire format.
    """
    adapted = dict(kwargs)
    wire_model = str(adapted.get("model") or backend._wire_model_name()).strip()
    wire_model_lower = wire_model.lower()

    # Moonshot Kimi K2.5 currently only accepts temperature=1.
    if backend.config.provider == "moonshot" and wire_model_lower.startswith("kimi-k2.5"):
        current_temperature = adapted.get("temperature")
        if current_temperature != 1:
            logger.info(
                "openai_chat_param_override model=%s provider=%s param=temperature from=%s to=1",
                backend.config.model,
                backend.config.provider,
                current_temperature,
            )
            adapted["temperature"] = 1

    return adapted


def _extract_openai_error_text(backend, exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="ignore").strip()
            if body:
                return body
        except Exception:
            pass
    text = str(exc or "").strip()
    if text:
        return text
    response = getattr(exc, "response", None)
    if response is not None:
        body = getattr(response, "text", None)
        if isinstance(body, str) and body.strip():
            return body.strip()
    return repr(exc)


def _adapt_openai_chat_kwargs_from_error(
    backend,
    kwargs: Dict[str, Any],
    exc: Exception,
) -> Optional[tuple]:
    """Best-effort one-shot adaptation for provider/model parameter mismatches."""
    text = _extract_openai_error_text(backend, exc)
    adapted = dict(kwargs)
    changes: List[str] = []

    for raw_param in re.findall(r"unsupported parameter:?\s*([a-zA-Z0-9_]+)", text, flags=re.IGNORECASE):
        target_key = next((key for key in adapted.keys() if key.lower() == raw_param.lower()), None)
        if target_key and target_key in adapted:
            adapted.pop(target_key, None)
            changes.append(f"removed {target_key}")

    if re.search(r"invalid temperature:.*only\s+1\s+is allowed", text, flags=re.IGNORECASE):
        if adapted.get("temperature") != 1:
            adapted["temperature"] = 1
            changes.append("set temperature=1")

    if re.search(
        r"tool_choice\s*['\"]?required['\"]?\s+is incompatible with thinking enabled",
        text,
        flags=re.IGNORECASE,
    ):
        if adapted.get("tool_choice") == "required":
            adapted["tool_choice"] = "auto"
            changes.append("downgraded tool_choice=auto")

    if not changes or adapted == kwargs:
        return None
    return adapted, ", ".join(changes)


def _call_openai_chat_with_adaptation(
    backend,
    request_fn: Callable[[Dict[str, Any]], Any],
    kwargs: Dict[str, Any],
    *,
    base_url: str,
) -> Any:
    """Execute an OpenAI-compatible chat request with one adaptive retry."""
    try:
        return request_fn(kwargs)
    except Exception as exc:
        adapted = _adapt_openai_chat_kwargs_from_error(backend, kwargs, exc)
        if adapted is None:
            raise
        retried_kwargs, change_summary = adapted
        logger.info(
            "openai_chat_retry_adapted model=%s provider=%s endpoint=%s changes=%s",
            backend.config.model,
            backend.config.provider,
            base_url,
            change_summary,
        )
        return request_fn(retried_kwargs)


# ---------------------------------------------------------------------------
# OpenAI Codex helpers
# ---------------------------------------------------------------------------

def _is_openai_codex_base_url(base_url: str) -> bool:
    normalized = str(base_url or "").strip().rstrip("/")
    return normalized.startswith(_OPENAI_CODEX_BASE_URL)


def _resolve_openai_codex_url(base_url: str) -> str:
    normalized = str(base_url or _OPENAI_CODEX_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _decode_openai_codex_jwt(token: str) -> Dict[str, Any]:
    parts = str(token or "").split(".")
    if len(parts) != 3 or not parts[1]:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_openai_codex_account_id(token: str) -> str:
    payload = _decode_openai_codex_jwt(token)
    auth = payload.get(_OPENAI_CODEX_JWT_CLAIM_PATH)
    if isinstance(auth, dict):
        account_id = str(auth.get("chatgpt_account_id") or "").strip()
        if account_id:
            return account_id
    # Fallback: env var injected by gateway when chatgpt_account_id is in
    # stored auth (e.g. ~/.ovjarvis/auth.json tokens.account_id) but not
    # embedded in the access_token JWT claims.
    return os.environ.get("CHATGPT_ACCOUNT_ID", "")


def _openai_codex_user_agent() -> str:
    try:
        system = platform.system().lower() or "unknown"
        release = platform.release() or "unknown"
        machine = platform.machine() or "unknown"
        return f"pi ({system} {release}; {machine})"
    except Exception:
        return "pi (python)"


def _is_ollama_endpoint(base_url: str) -> bool:
    normalized = str(base_url or "").strip().lower().rstrip("/")
    return (
        "127.0.0.1:11434" in normalized
        or "localhost:11434" in normalized
        or normalized.endswith(":11434/v1")
        or normalized.endswith(":11434")
    )


def _wrap_openai_connection_error(backend, exc: Exception, base_url: str) -> RuntimeError:
    exc_name = type(exc).__name__.lower()
    exc_text = str(exc).lower()
    if "connection" not in exc_name and "connection" not in exc_text and "refused" not in exc_text:
        return RuntimeError(str(exc))
    if _is_ollama_endpoint(base_url):
        return RuntimeError(
            f"Could not connect to Ollama at {base_url}. Start the Ollama server and verify the model is installed."
        )
    return RuntimeError(f"OpenAI-compatible connection failed for {base_url}: {exc}")


# ---------------------------------------------------------------------------
# OpenAI Responses helpers
# ---------------------------------------------------------------------------

def _responses_jsonable(value: Any) -> Any:
    """Convert SDK response objects into JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            k: _responses_jsonable(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _responses_jsonable(v)
            for v in value
        ]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _responses_jsonable(
                model_dump(exclude_none=True)
            )
        except TypeError:
            return _responses_jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _responses_jsonable(to_dict())
    if hasattr(value, "__dict__"):
        return {
            k: _responses_jsonable(v)
            for k, v in value.__dict__.items()
            if not k.startswith("_")
        }
    return str(value)


def _extract_responses_output_items(payload: Any) -> List[Dict[str, Any]]:
    """Return canonical output items from a Responses SDK object or dict."""
    if payload is None:
        return []
    if not isinstance(payload, dict):
        payload = _responses_jsonable(payload)
    output = payload.get("output")
    if output is None:
        return []
    if isinstance(output, list):
        return [
            item for item in (
                _responses_jsonable(item)
                for item in output
            )
            if isinstance(item, dict)
        ]
    if isinstance(output, dict):
        return [output]
    return []


def _extract_responses_text_from_items(items: List[Dict[str, Any]]) -> str:
    """Extract assistant-visible text from Responses output items."""
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("text"), str) and item.get("text"):
            return item["text"]
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, dict):
                    text = text.get("value") or text.get("text")
                if isinstance(text, str) and text:
                    return text
                if isinstance(part.get("output_text"), str) and part.get("output_text"):
                    return part["output_text"]
                if isinstance(part.get("content"), str) and part.get("content"):
                    return part["content"]
    return ""


def _extract_responses_tool_calls_from_items(items: List[Dict[str, Any]]) -> List[ToolCall]:
    """Extract function/tool calls from Responses output items."""
    tool_calls: List[ToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"function_call", "tool_call"}:
            continue
        args_raw = item.get("arguments", {})
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args_raw}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {"raw": str(args_raw)}
        call_id = item.get("call_id") or item.get("id") or f"call_{len(tool_calls) + 1}"
        tool_calls.append(
            ToolCall(
                id=str(call_id),
                name=item.get("name", "unknown"),
                arguments=args,
            )
        )
    return tool_calls


def _extract_responses_text_from_dict(payload: Dict[str, Any]) -> str:
    """Extract the assistant text from a Responses API JSON dict.

    Consolidates the duplicated extraction logic used in both SDK and
    HTTP paths for the OpenAI Responses API.
    """
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output = payload.get("output")
    if output is not None:
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            if isinstance(output.get("text"), str) and output.get("text"):
                return output["text"]
            content = output.get("content")
            if isinstance(content, list):
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    text_value = p.get("text")
                    if isinstance(text_value, dict):
                        text_value = text_value.get("value") or text_value.get("text")
                    if isinstance(text_value, str) and text_value:
                        return text_value
        if isinstance(output, list) and len(output) > 0:
            text = _extract_responses_text_from_items(
                _extract_responses_output_items(payload)
            )
            if text:
                return text

    text_field = payload.get("text")
    if isinstance(text_field, str) and text_field:
        return text_field

    raise RuntimeError(
        f"Unexpected Responses API response format. "
        f"Payload keys: {list(payload.keys())}"
    )


def _build_openai_responses_request(
    backend,
    *,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict]],
    tool_choice: Optional[str],
    include_system_messages: bool,
) -> Dict[str, Any]:
    input_items = _convert_messages_openai_responses(backend, messages)
    if not include_system_messages:
        input_items = [
            item for item in input_items
            if not (isinstance(item, dict) and item.get("role") == "system")
        ]

    kwargs: Dict[str, Any] = {
        "model": backend._wire_model_name(),
        "input": input_items,
        "max_output_tokens": backend.config.max_tokens,
        "reasoning": {"effort": "medium"},
    }
    if backend.config.system_prompt:
        kwargs["instructions"] = backend.config.system_prompt
    if tools:
        kwargs["tools"] = _convert_tools_openai_responses(tools)
        kwargs["tool_choice"] = tool_choice or "auto"
    return kwargs


def _iter_openai_codex_sse_events(raw_bytes: bytes) -> List[Dict[str, Any]]:
    text = raw_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
    events: List[Dict[str, Any]] = []
    for chunk in text.split("\n\n"):
        data_lines = [
            line[5:].strip()
            for line in chunk.split("\n")
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON Codex SSE chunk: %s", data[:200])
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _extract_openai_codex_final_response(backend, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_payload: Optional[Dict[str, Any]] = None
    for event in events:
        event_type = str(event.get("type") or "").strip()
        if event_type == "error":
            message = (
                str(event.get("message") or "").strip()
                or str(event.get("code") or "").strip()
                or json.dumps(event)
            )
            raise RuntimeError(f"Codex error: {message}")
        if event_type == "response.failed":
            response = event.get("response") or {}
            error = response.get("error") if isinstance(response, dict) else {}
            message = (
                str(error.get("message") or "").strip()
                if isinstance(error, dict)
                else ""
            )
            raise RuntimeError(message or "Codex response failed")
        if event_type in {"response.completed", "response.done"}:
            response = event.get("response")
            if isinstance(response, dict):
                final_payload = _responses_jsonable(response)

    if not final_payload:
        raise RuntimeError("Codex response stream completed without a final response payload")
    return final_payload


def _call_openai_codex_responses(
    backend,
    *,
    base_url: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict]],
    tool_choice: Optional[str],
) -> Dict[str, Any]:
    account_id = _extract_openai_codex_account_id(api_key)
    if not account_id:
        raise RuntimeError(
            "OpenAI OAuth access token is missing chatgpt_account_id; please rerun `omicverse jarvis --setup`."
        )

    payload = _build_openai_responses_request(
        backend,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        include_system_messages=False,
    )
    payload.pop("max_output_tokens", None)
    payload.pop("reasoning", None)
    payload.update(
        {
            "store": False,
            "stream": True,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
    )

    url = _resolve_openai_codex_url(base_url)
    timeout_s = _request_timeout_seconds()
    request = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "pi",
            "User-Agent": _openai_codex_user_agent(),
            "accept": "text/event-stream",
            "content-type": "application/json",
        },
        method="POST",
    )

    logger.info(
        "openai_codex_responses_start model=%s endpoint=%s tool_choice=%s messages=%d tools=%d timeout_s=%.1f",
        backend.config.model,
        url,
        tool_choice or "none",
        len(messages),
        len(tools or []),
        timeout_s,
    )

    try:
        with urllib_request.urlopen(
            request,
            timeout=timeout_s,
            context=ssl.create_default_context(),
        ) as response:
            raw_bytes = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI Codex Responses API failed: HTTP {exc.code} {detail[:400]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI Codex Responses API failed: {exc}") from exc

    events = _iter_openai_codex_sse_events(raw_bytes)
    final_payload = _extract_openai_codex_final_response(backend, events)
    logger.info(
        "openai_codex_responses_done model=%s output_items=%d tool_calls=%d",
        backend.config.model,
        len(_extract_responses_output_items(final_payload)),
        len(_extract_responses_tool_calls_from_items(_extract_responses_output_items(final_payload))),
    )
    return final_payload


def _convert_messages_openai_responses(backend, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize mixed conversation state into Responses API input items."""
    converted: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        item_type = message.get("type")
        if item_type in {"function_call", "function_call_output", "message", "reasoning"}:
            normalized = _responses_jsonable(message)
            if isinstance(normalized, dict):
                converted.append(normalized)
            continue

        role = message.get("role")
        if role == "tool":
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": message.get("content", ""),
                }
            )
            continue

        if role not in {"system", "user", "assistant"}:
            continue

        content = message.get("content", "")
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    parts.append({"type": "input_text", "text": str(block)})
                    continue
                block_type = block.get("type")
                if block_type in {"input_text", "output_text"}:
                    parts.append(
                        {
                            "type": "input_text",
                            "text": block.get("text", ""),
                        }
                    )
                elif block_type == "text":
                    parts.append(
                        {
                            "type": "input_text",
                            "text": block.get("text", ""),
                        }
                    )
                else:
                    normalized = _responses_jsonable(block)
                    if isinstance(normalized, dict):
                        parts.append(normalized)
            converted.append({"role": role, "content": parts})
        else:
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            converted.append({"role": role, "content": content})
    return converted


def _build_chat_response_from_responses_payload(backend, payload: Dict[str, Any]) -> ChatResponse:
    """Build the generic ChatResponse contract from a Responses payload."""
    output_items = _extract_responses_output_items(payload)
    tool_calls = _extract_responses_tool_calls_from_items(output_items)
    content = ""
    try:
        content = _extract_responses_text_from_dict(payload)
    except Exception:
        content = _extract_responses_text_from_items(output_items)

    usage_obj = None
    usage = payload.get("usage") or {}
    if usage:
        pt = _coerce_int(usage.get("input_tokens"))
        if pt is None:
            pt = _coerce_int(usage.get("prompt_tokens"))
        ct = _coerce_int(usage.get("output_tokens"))
        if ct is None:
            ct = _coerce_int(usage.get("completion_tokens"))
        tt = _compute_total(pt, ct, _coerce_int(usage.get("total_tokens")))
        if tt is not None:
            usage_obj = Usage(
                input_tokens=pt or 0,
                output_tokens=ct or 0,
                total_tokens=tt,
                model=backend.config.model,
                provider=backend.config.provider,
            )
            backend.last_usage = usage_obj

    raw_message: Optional[Any] = output_items if tool_calls else None
    stop_reason = "tool_use" if tool_calls else "end_turn"
    return ChatResponse(
        content=content or None,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage_obj,
        raw_message=raw_message,
    )


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

def _convert_tools_openai(tools: List[Dict]) -> List[Dict]:
    """Convert provider-agnostic tool defs to OpenAI format."""
    return [{"type": "function", "function": t} for t in tools]


def _convert_tools_openai_responses(tools: List[Dict]) -> List[Dict]:
    """Convert provider-agnostic tool defs to OpenAI Responses format."""
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
            "strict": False,
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Sync chat methods
# ---------------------------------------------------------------------------

def _chat_via_openai_compatible(backend, user_prompt: str) -> str:
    info = get_provider(backend.config.provider)
    base_url = backend.config.endpoint or (info.base_url if info else "https://api.openai.com/v1")
    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{backend.config.provider}'. Set the appropriate environment variable or pass api_key."
        )

    # Check if model requires Responses API (gpt-5 series)
    if ModelConfig.requires_responses_api(backend.config.model):
        return _chat_via_openai_responses(backend, base_url, api_key, user_prompt)

    # Otherwise use Chat Completions API (standard path for gpt-4o, gpt-4o-mini, etc.)
    # Try modern OpenAI SDK first, then fallback to raw HTTP
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=base_url, api_key=api_key)
        wire_model = backend._wire_model_name()

        # Wrap SDK call with retry logic
        def _make_sdk_call():
            kwargs = _apply_openai_chat_param_policy(backend, {
                "model": wire_model,
                "messages": [
                    {"role": "system", "content": backend.config.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": backend.config.temperature,
                "max_tokens": backend.config.max_tokens,
            })
            resp = _call_openai_chat_with_adaptation(
                backend,
                lambda payload: client.chat.completions.create(**payload),
                kwargs,
                base_url=base_url,
            )
            choice = (resp.choices or [None])[0]
            if not choice or not getattr(choice, "message", None):
                raise RuntimeError("No choices returned from the model")

            # Capture usage information (only if numeric values are present)
            if hasattr(resp, 'usage') and resp.usage is not None:
                usage = resp.usage
                pt = _coerce_int(getattr(usage, 'prompt_tokens', None))
                ct = _coerce_int(getattr(usage, 'completion_tokens', None))
                tt = _coerce_int(getattr(usage, 'total_tokens', None))
                tt = _compute_total(pt, ct, tt)
                if tt is not None:
                    backend.last_usage = Usage(
                        input_tokens=pt or 0,
                        output_tokens=ct or 0,
                        total_tokens=tt,
                        model=backend.config.model,
                        provider=backend.config.provider
                    )

            return choice.message.content or ""

        return backend._retry(_make_sdk_call)

    except ImportError:
        # OpenAI SDK not installed, fallback to HTTP
        return _chat_via_openai_http(backend, base_url, api_key, user_prompt)
    except Exception as exc:
        # Log SDK failure but try HTTP fallback as last resort
        logger.warning(
            "OpenAI SDK call failed (%s: %s), trying HTTP fallback",
            type(exc).__name__,
            exc
        )
        try:
            warnings.warn(
                f"OpenAI SDK call failed ({type(exc).__name__}: {exc}), trying HTTP fallback"
            )
        except Exception:
            pass
        return _chat_via_openai_http(backend, base_url, api_key, user_prompt)


def _chat_via_openai_http(backend, base_url: str, api_key: str, user_prompt: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = _apply_openai_chat_param_policy(backend, {
        "model": backend._wire_model_name(),
        "messages": [
            {"role": "system", "content": backend.config.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": backend.config.temperature,
        "max_tokens": backend.config.max_tokens,
    })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Wrap HTTP call with retry logic
    def _make_http_call():
        def _request_with_kwargs(payload: Dict[str, Any]) -> str:
            data = json.dumps(payload).encode("utf-8")
            req = urllib_request.Request(url, data=data, headers=headers, method="POST")
            ctx = ssl.create_default_context()
            with urllib_request.urlopen(req, context=ctx, timeout=90) as resp:
                content = resp.read().decode("utf-8")
                parsed = json.loads(content)
                choices = parsed.get("choices", [])
                if not choices:
                    raise RuntimeError(f"No choices in response: {parsed}")
                msg = choices[0].get("message", {})

                usage_data = parsed.get("usage", {})
                if usage_data:
                    backend.last_usage = Usage(
                        input_tokens=usage_data.get('prompt_tokens', 0),
                        output_tokens=usage_data.get('completion_tokens', 0),
                        total_tokens=usage_data.get('total_tokens', 0),
                        model=backend.config.model,
                        provider=backend.config.provider
                    )

                return msg.get("content", "")

        try:
            return _call_openai_chat_with_adaptation(
                backend,
                _request_with_kwargs,
                body,
                base_url=base_url,
            )
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI-compatible HTTP call failed: {type(exc).__name__}: {exc}"
            ) from exc

    return backend._retry(_make_http_call)


def _chat_via_openai_responses(backend, base_url: str, api_key: str, user_prompt: str) -> str:
    """Use OpenAI Responses API for models that require it (gpt-5 series).

    The Responses API uses:
    - Endpoint: /v1/responses (not /v1/chat/completions)
    - Request: instructions (system prompt) + input (content parts list)
      We send input as a message with content parts to maximize compatibility:
        [{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}]
      Additionally, we set response_format=text and modalities=["text"] to
      encourage plain-text responses suitable for code extraction.
    - Response: prefer output_text; otherwise inspect output.{text|content[*].text} or text
    """
    if _is_openai_codex_base_url(base_url):
        def _make_codex_call():
            payload = _call_openai_codex_responses(
                backend,
                base_url=base_url,
                api_key=api_key,
                messages=[{"role": "user", "content": user_prompt}],
                tools=None,
                tool_choice=None,
            )
            text = _extract_responses_text_from_dict(payload)
            if not text:
                text = _extract_responses_text_from_items(
                    _extract_responses_output_items(payload)
                )
            usage = payload.get("usage") or {}
            if usage:
                pt = _coerce_int(usage.get("input_tokens"))
                if pt is None:
                    pt = _coerce_int(usage.get("prompt_tokens"))
                ct = _coerce_int(usage.get("output_tokens"))
                if ct is None:
                    ct = _coerce_int(usage.get("completion_tokens"))
                tt = _compute_total(pt, ct, _coerce_int(usage.get("total_tokens")))
                if tt is not None:
                    backend.last_usage = Usage(
                        input_tokens=pt or 0,
                        output_tokens=ct or 0,
                        total_tokens=tt,
                        model=backend.config.model,
                        provider=backend.config.provider,
                    )
            return text

        return backend._retry(_make_codex_call)

    # Try OpenAI SDK first
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=base_url, api_key=api_key)

        # Wrap SDK call with retry logic
        def _make_responses_sdk_call():
            # Responses API uses 'instructions' for system prompt
            # and 'input' as a string (not message array)
            # Note: gpt-5 Responses API does not support temperature parameter
            logger.debug(f"GPT-5 Responses API call: model={backend.config.model}, max_tokens={backend.config.max_tokens}")
            logger.debug(f"User prompt length: {len(user_prompt)} chars")

            kwargs = _build_openai_responses_request(
                backend,
                messages=[{"role": "user", "content": user_prompt}],
                tools=None,
                tool_choice=None,
                include_system_messages=False,
            )

            # GPT-5 models use reasoning tokens - control effort for response quality
            # Set reasoning effort to 'high' for maximum reasoning capability and best quality responses
            logger.debug("Creating GPT-5 Responses API request with high reasoning effort...")
            kwargs["reasoning"] = {"effort": "high"}
            resp = client.responses.create(**kwargs)

            logger.debug(f"GPT-5 response received. Type: {type(resp).__name__}")
            logger.debug(f"Response attributes: {[attr for attr in dir(resp) if not attr.startswith('_')]}")

            # Capture usage information from Responses API (only if numeric)
            if hasattr(resp, 'usage') and resp.usage is not None:
                usage = resp.usage
                pt = _coerce_int(getattr(usage, 'input_tokens', None))
                if pt is None:
                    pt = _coerce_int(getattr(usage, 'prompt_tokens', None))
                ct = _coerce_int(getattr(usage, 'output_tokens', None))
                if ct is None:
                    ct = _coerce_int(getattr(usage, 'completion_tokens', None))
                tt = _coerce_int(getattr(usage, 'total_tokens', None))
                tt = _compute_total(pt, ct, tt)
                if tt is not None:
                    backend.last_usage = Usage(
                        input_tokens=pt or 0,
                        output_tokens=ct or 0,
                        total_tokens=tt,
                        model=backend.config.model,
                        provider=backend.config.provider
                    )

            # Extract text from Responses API format with fallback chain
            logger.debug("Attempting to extract text from GPT-5 response...")

            # Try output_text first (most common). Some SDK versions expose it as a method.
            output_text = getattr(resp, 'output_text', None)
            if callable(output_text):
                try:
                    output_text = output_text()
                except TypeError:
                    pass
                except Exception:
                    output_text = None
            if isinstance(output_text, str) and output_text:
                logger.debug(f"✓ Extracted via output_text (length: {len(output_text)} chars)")
                logger.debug(f"Response preview (first 200 chars): {output_text[:200]}")
                return output_text

            def _log_and_return(text: str, via: str) -> str:
                logger.debug(f"✓ Extracted via {via} (length: {len(text)} chars)")
                return text

            def _extract_from_parts(parts: Any, via_prefix: str) -> Optional[str]:
                try:
                    for i, p in enumerate(parts):
                        # p may be object or dict
                        t = getattr(p, 'text', None)
                        if isinstance(t, str) and t:
                            return _log_and_return(t, f"{via_prefix}[{i}].text")
                        if isinstance(p, dict):
                            t2 = p.get('text')
                            if isinstance(t2, str) and t2:
                                return _log_and_return(t2, f"{via_prefix}[{i}]['text']")
                except Exception as e:
                    logger.debug(f"Error iterating {via_prefix}: {e}")
                return None

            # Try resp.output (Responses API canonical structure)
            output = getattr(resp, 'output', None)
            if output is not None:
                logger.debug(f"Found output attribute: type={type(output).__name__}")

                if isinstance(output, str) and output:
                    return _log_and_return(output, "output (direct string)")

                if isinstance(output, dict):
                    t = output.get('text')
                    if isinstance(t, str) and t:
                        return _log_and_return(t, "output['text']")
                    parts = output.get('content')
                    if parts:
                        got = _extract_from_parts(parts, "output['content']")
                        if got is not None:
                            return got

                # Some SDK versions use output as a list of items; each item usually has .content[...].text
                if isinstance(output, list):
                    for oi, item in enumerate(output):
                        if isinstance(item, str) and item:
                            return _log_and_return(item, f"output[{oi}]")

                        if isinstance(item, dict):
                            t = item.get('text')
                            if isinstance(t, str) and t:
                                return _log_and_return(t, f"output[{oi}]['text']")
                            parts = item.get('content')
                            if parts:
                                got = _extract_from_parts(parts, f"output[{oi}]['content']")
                                if got is not None:
                                    return got

                        t = getattr(item, 'text', None)
                        if isinstance(t, str) and t:
                            return _log_and_return(t, f"output[{oi}].text")

                        parts = getattr(item, 'content', None)
                        if parts:
                            got = _extract_from_parts(parts, f"output[{oi}].content")
                            if got is not None:
                                return got

                # Object with .text or .content
                t = getattr(output, 'text', None)
                if isinstance(t, str) and t:
                    return _log_and_return(t, "output.text")

                parts = getattr(output, 'content', None)
                if parts:
                    got = _extract_from_parts(parts, "output.content")
                    if got is not None:
                        return got

            # Fallback: try text attribute directly
            text_attr = getattr(resp, 'text', None)
            if isinstance(text_attr, str) and text_attr:
                return _log_and_return(text_attr, "text attribute")

            # If nothing worked, provide diagnostic info
            logger.error("❌ Could not extract text from GPT-5 response")
            logger.error(f"Response type: {type(resp).__name__}")
            logger.error(f"Available attributes: {[attr for attr in dir(resp) if not attr.startswith('_')]}")
            raise RuntimeError(
                f"Unexpected Responses API response format. "
                f"Type: {type(resp).__name__}, "
                f"Available attributes: {dir(resp)}"
            )

        return backend._retry(_make_responses_sdk_call)

    except ImportError:
        # OpenAI SDK not installed, fallback to HTTP
        return _chat_via_openai_responses_http(backend, base_url, api_key, user_prompt)
    except Exception as exc:
        # Log SDK failure but try HTTP fallback as last resort
        logger.warning(
            "OpenAI Responses API SDK call failed (%s: %s), trying HTTP fallback",
            type(exc).__name__,
            exc
        )
        try:
            warnings.warn(
                f"OpenAI Responses API SDK call failed ({type(exc).__name__}: {exc}), trying HTTP fallback"
            )
        except Exception:
            pass
        return _chat_via_openai_responses_http(backend, base_url, api_key, user_prompt)


def _chat_via_openai_responses_http(backend, base_url: str, api_key: str, user_prompt: str) -> str:
    """HTTP fallback for OpenAI Responses API.

    Uses 'instructions' for system prompt and 'input' as string.
    Note: gpt-5 Responses API does not support temperature parameter.
    """
    # Wrap entire HTTP call logic with retry
    def _make_responses_http_call():
        url = base_url.rstrip("/") + "/responses"

        def make_body(parts_style: bool) -> Dict[str, Any]:
            if parts_style:
                return {
                    "model": backend.config.model,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": backend.config.system_prompt}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_prompt}
                            ],
                        },
                    ],
                    "instructions": backend.config.system_prompt,
                    "max_output_tokens": backend.config.max_tokens,
                    "reasoning": {"effort": "high"}  # Use high reasoning effort for better quality responses
                }
            else:
                return {
                    "model": backend.config.model,
                    "input": user_prompt,
                    "instructions": backend.config.system_prompt,
                    "max_output_tokens": backend.config.max_tokens,
                    "reasoning": {"effort": "high"}  # Use high reasoning effort for better quality responses
                }

        body = make_body(parts_style=True)
        data = json.dumps(body).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        req = urllib_request.Request(url, data=data, headers=headers, method="POST")

        # SSL certificate verification is enabled by default for security
        ctx = ssl.create_default_context()
        try:
            with urllib_request.urlopen(req, context=ctx, timeout=90) as resp:
                content = resp.read().decode("utf-8")
                payload = json.loads(content)

                # Capture usage information if present and numeric
                usage_data = payload.get("usage", {})
                if isinstance(usage_data, dict) and usage_data:
                    pt = usage_data.get('input_tokens') if 'input_tokens' in usage_data else usage_data.get('prompt_tokens')
                    ct = usage_data.get('output_tokens') if 'output_tokens' in usage_data else usage_data.get('completion_tokens')
                    tt = usage_data.get('total_tokens')
                    pt_i = _coerce_int(pt)
                    ct_i = _coerce_int(ct)
                    tt_i = _compute_total(pt_i, ct_i, _coerce_int(tt))
                    if tt_i is not None:
                        backend.last_usage = Usage(
                            input_tokens=pt_i or 0,
                            output_tokens=ct_i or 0,
                            total_tokens=tt_i,
                            model=backend.config.model,
                            provider=backend.config.provider
                        )

                # Extract text with fallback chain
                # Try output_text first (most common)
                if "output_text" in payload and payload["output_text"]:
                    return payload["output_text"]

                # Try output.text
                if "output" in payload:
                    output = payload["output"]
                    # Direct string
                    if isinstance(output, str):
                        return output
                    # Dict with text
                    if isinstance(output, dict):
                        if "text" in output and output["text"]:
                            return output["text"]
                        # content list of parts
                        content = output.get("content")
                        if isinstance(content, list):
                            for p in content:
                                if isinstance(p, dict) and p.get("text"):
                                    return p["text"]
                    # List of parts
                    if isinstance(output, list) and len(output) > 0:
                        first = output[0]
                        if isinstance(first, str):
                            return first
                        if isinstance(first, dict) and first.get("text"):
                            return first["text"]

                # Fallback: check for 'text' field directly
                if "text" in payload:
                    return payload["text"]

                # If nothing worked, return truncated payload for debugging
                payload_str = json.dumps(payload)[:500]
                raise RuntimeError(
                    f"Unexpected Responses API response format. Payload (first 500 chars): {payload_str}"
                )

        except urllib_request.HTTPError as exc:
            # If 400 Bad Request, retry with simpler string input format
            if exc.code == 400:
                try:
                    alt_body = make_body(parts_style=False)
                    alt_data = json.dumps(alt_body).encode("utf-8")
                    alt_req = urllib_request.Request(url, data=alt_data, headers=headers, method="POST")
                    with urllib_request.urlopen(alt_req, context=ctx, timeout=90) as resp:
                        content = resp.read().decode("utf-8")
                        payload = json.loads(content)

                        # Capture usage information (alternative path)
                        usage_data = payload.get("usage", {})
                        if usage_data:
                            backend.last_usage = Usage(
                                input_tokens=usage_data.get('input_tokens', 0) or usage_data.get('prompt_tokens', 0),
                                output_tokens=usage_data.get('output_tokens', 0) or usage_data.get('completion_tokens', 0),
                                total_tokens=usage_data.get('total_tokens', 0),
                                model=backend.config.model,
                                provider=backend.config.provider
                            )

                        # Same extraction as above
                        if "output_text" in payload and payload["output_text"]:
                            return payload["output_text"]
                        if "output" in payload:
                            output = payload["output"]
                            if isinstance(output, str):
                                return output
                            if isinstance(output, dict):
                                if "text" in output and output["text"]:
                                    return output["text"]
                                content_list = output.get("content")
                                if isinstance(content_list, list):
                                    for p in content_list:
                                        if isinstance(p, dict) and p.get("text"):
                                            return p["text"]
                            if isinstance(output, list) and len(output) > 0:
                                first = output[0]
                                if isinstance(first, str):
                                    return first
                                if isinstance(first, dict) and first.get("text"):
                                    return first["text"]
                        if "text" in payload:
                            return payload["text"]
                        payload_str = json.dumps(payload)[:500]
                        raise RuntimeError(
                            f"Unexpected Responses API response format (alt). Payload (first 500 chars): {payload_str}"
                        )
                except Exception:
                    # Fall through to include original error details below
                    pass
            # Include response body snippet in error message
            try:
                # Some environments require reading from exc.fp
                try:
                    raw = exc.read()
                except Exception:
                    raw = getattr(getattr(exc, 'fp', None), 'read', lambda: b'')()
                error_body = (raw or b"").decode("utf-8", errors="ignore")[:500]
                raise RuntimeError(
                    f"OpenAI Responses API HTTP {exc.code} error: {exc.reason}. "
                    f"Response body (first 500 chars): {error_body}"
                ) from exc
            except Exception:
                raise RuntimeError(
                    f"OpenAI Responses API HTTP {exc.code} error: {exc.reason}"
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI Responses API HTTP call failed: {type(exc).__name__}: {exc}"
            ) from exc

    return backend._retry(_make_responses_http_call)


def _chat_tools_openai(
    backend,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    tool_choice: Optional[str],
) -> ChatResponse:
    """Multi-turn chat via OpenAI-compatible API with tool support."""
    if backend.config.provider == "openai" and ModelConfig.requires_responses_api(backend.config.model):
        return _chat_tools_openai_responses(backend, messages, tools, tool_choice)

    info = get_provider(backend.config.provider)
    base_url = backend.config.endpoint or (info.base_url if info else "https://api.openai.com/v1")
    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{backend.config.provider}'."
        )

    try:
        from openai import OpenAI  # type: ignore
        timeout_s = _request_timeout_seconds()
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

        def _make_call():
            try:
                # GPT-5 series requires max_completion_tokens instead of max_tokens
                if ModelConfig.requires_responses_api(backend.config.model):
                    token_param = {"max_completion_tokens": backend.config.max_tokens}
                else:
                    token_param = {"max_tokens": backend.config.max_tokens}

                kwargs = _apply_openai_chat_param_policy(backend, {
                    "model": backend._wire_model_name(),
                    "messages": messages,
                    "temperature": backend.config.temperature,
                    **token_param,
                })
                if tools:
                    kwargs["tools"] = _convert_tools_openai(tools)
                    kwargs["tool_choice"] = tool_choice or "auto"

                logger.info(
                    "openai_tool_chat_start model=%s provider=%s endpoint=%s tool_choice=%s messages=%d tools=%d timeout_s=%.1f",
                    backend.config.model,
                    backend.config.provider,
                    base_url,
                    kwargs.get("tool_choice", "none"),
                    len(messages),
                    len(tools or []),
                    timeout_s,
                )
                resp = _call_openai_chat_with_adaptation(
                    backend,
                    lambda payload: client.chat.completions.create(**payload),
                    kwargs,
                    base_url=base_url,
                )
                logger.info(
                    "openai_tool_chat_done model=%s finish_reason=%s choices=%d",
                    backend.config.model,
                    getattr((resp.choices or [None])[0], "finish_reason", None),
                    len(resp.choices or []),
                )
                choice = (resp.choices or [None])[0]
                if not choice or not getattr(choice, "message", None):
                    raise RuntimeError("No choices returned from the model")

                # Capture usage
                usage_obj = None
                if hasattr(resp, 'usage') and resp.usage is not None:
                    usage = resp.usage
                    pt = _coerce_int(getattr(usage, 'prompt_tokens', None))
                    ct = _coerce_int(getattr(usage, 'completion_tokens', None))
                    tt = _compute_total(pt, ct, _coerce_int(getattr(usage, 'total_tokens', None)))
                    if tt is not None:
                        usage_obj = Usage(
                            input_tokens=pt or 0,
                            output_tokens=ct or 0,
                            total_tokens=tt,
                            model=backend.config.model,
                            provider=backend.config.provider
                        )
                        backend.last_usage = usage_obj

                # Extract content and tool calls
                msg = choice.message
                content = msg.content or None
                tc_list = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args_str = tc.function.arguments
                        try:
                            args = json.loads(args_str)
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": args_str}
                        tc_list.append(ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        ))

                stop = "tool_use" if tc_list else "end_turn"
                if choice.finish_reason == "length":
                    stop = "max_tokens"

                # Build raw message dict for re-injection into messages list
                raw_msg = {"role": "assistant", "content": content}
                if tc_list:
                    raw_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in tc_list
                    ]

                return ChatResponse(
                    content=content,
                    tool_calls=tc_list,
                    stop_reason=stop,
                    usage=usage_obj,
                    raw_message=raw_msg,
                )
            except Exception as exc:
                raise _wrap_openai_connection_error(backend, exc, base_url) from exc

        return backend._retry(_make_call)

    except ImportError as exc:
        if getattr(exc, "name", None) == "openai":
            raise RuntimeError(
                "openai package not installed. Install openai>=1.0 for tool-calling support."
            ) from exc
        raise RuntimeError(
            f"OpenAI import failed during tool-calling setup: {exc}"
        ) from exc


def _chat_tools_openai_responses(
    backend,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict]],
    tool_choice: Optional[str],
) -> ChatResponse:
    """Multi-turn tool chat via the OpenAI Responses API."""
    info = get_provider(backend.config.provider)
    base_url = backend.config.endpoint or (info.base_url if info else "https://api.openai.com/v1")
    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{backend.config.provider}'."
        )

    if _is_openai_codex_base_url(base_url):
        def _make_codex_call():
            payload = _call_openai_codex_responses(
                backend,
                base_url=base_url,
                api_key=api_key,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            return _build_chat_response_from_responses_payload(backend, payload)

        return backend._retry(_make_codex_call)

    try:
        from openai import OpenAI  # type: ignore

        timeout_s = _request_timeout_seconds()
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

        def _make_call():
            kwargs = _build_openai_responses_request(
                backend,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                include_system_messages=True,
            )

            logger.info(
                "openai_responses_tool_chat_start model=%s provider=%s endpoint=%s tool_choice=%s messages=%d tools=%d timeout_s=%.1f",
                backend.config.model,
                backend.config.provider,
                base_url,
                kwargs.get("tool_choice", "none"),
                len(messages),
                len(tools or []),
                timeout_s,
            )
            resp = client.responses.create(**kwargs)
            payload = _responses_jsonable(resp)
            logger.info(
                "openai_responses_tool_chat_done model=%s output_items=%d tool_calls=%d",
                backend.config.model,
                len(_extract_responses_output_items(payload)),
                len(_extract_responses_tool_calls_from_items(_extract_responses_output_items(payload))),
            )
            return _build_chat_response_from_responses_payload(backend, payload)

        return backend._retry(_make_call)

    except ImportError as exc:
        if getattr(exc, "name", None) == "openai":
            raise RuntimeError(
                "openai package not installed. Install openai>=1.0 for Responses API tool support."
            ) from exc
        raise RuntimeError(
            f"OpenAI import failed during Responses API setup: {exc}"
        ) from exc
