"""Anthropic Messages API adapter helpers for OmicVerseLLMBackend.

Internal module — import from ``omicverse.utils.agent_backend`` instead.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .agent_backend_common import (
    Usage,
    ToolCall,
    ChatResponse,
    _coerce_int,
    _compute_total,
    _request_timeout_seconds,
)
from .model_config import get_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

def _convert_tools_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert provider-agnostic tool defs to Anthropic format."""
    return [{
        "name": t["name"],
        "description": t["description"],
        "input_schema": t["parameters"],
    } for t in tools]


# ---------------------------------------------------------------------------
# Single-turn sync helper  (was OmicVerseLLMBackend._chat_via_anthropic)
# ---------------------------------------------------------------------------

def _chat_via_anthropic(backend: Any, user_prompt: str) -> str:
    api_key = backend._resolve_api_key()
    if not api_key:
        raise backend._missing_api_key_error(get_provider(backend.config.provider), "Anthropic")
    try:
        import anthropic  # type: ignore

        client_kwargs = {"api_key": api_key}
        if backend.config.endpoint:
            base = backend.config.endpoint.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            client_kwargs["base_url"] = base
        import httpx
        timeout_s = _request_timeout_seconds()
        client_kwargs["timeout"] = httpx.Timeout(timeout_s, connect=10.0)
        client = anthropic.Anthropic(**client_kwargs)

        wire_model = backend._wire_model_name()

        # Wrap Anthropic SDK call with retry logic
        def _make_anthropic_call():
            resp = client.messages.create(
                model=wire_model,
                max_tokens=backend.config.max_tokens,
                system=backend.config.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=backend.config.temperature,
            )

            # Capture usage information from Anthropic
            if hasattr(resp, 'usage') and resp.usage is not None:
                usage = resp.usage
                input_tokens = _coerce_int(getattr(usage, 'input_tokens', None))
                output_tokens = _coerce_int(getattr(usage, 'output_tokens', None))
                total_tokens = _compute_total(input_tokens, output_tokens, _coerce_int(getattr(usage, 'total_tokens', None)))
                if total_tokens is not None:
                    backend.last_usage = Usage(
                        input_tokens=input_tokens or 0,
                        output_tokens=output_tokens or 0,
                        total_tokens=total_tokens,
                        model=backend.config.model,
                        provider=backend.config.provider
                    )

            # Concatenate text chunks if present
            parts = []
            for block in getattr(resp, "content", []) or []:
                if getattr(block, "type", "") == "text":
                    parts.append(getattr(block, "text", ""))
            return "\n".join([p for p in parts if p])

        return backend._retry(_make_anthropic_call)

    except ImportError:
        raise RuntimeError(
            "anthropic package not installed. Install anthropic or choose an OpenAI-compatible model."
        )


# ---------------------------------------------------------------------------
# Multi-turn tool-calling helper  (was OmicVerseLLMBackend._chat_tools_anthropic)
# ---------------------------------------------------------------------------

def _chat_tools_anthropic(
    backend: Any,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    tool_choice: Optional[str],
) -> ChatResponse:
    """Multi-turn chat via Anthropic Messages API with tool support."""
    api_key = backend._resolve_api_key()
    if not api_key:
        raise backend._missing_api_key_error(get_provider(backend.config.provider), "Anthropic")

    try:
        import anthropic  # type: ignore
        client_kwargs = {"api_key": api_key}
        if backend.config.endpoint:
            # Proxy endpoints often use /v1 suffix for OpenAI compat.
            # The Anthropic SDK prepends /v1/messages itself, so strip
            # trailing /v1 to avoid /v1/v1/messages.
            base = backend.config.endpoint.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            client_kwargs["base_url"] = base
        import httpx
        timeout_s = _request_timeout_seconds()
        client_kwargs["timeout"] = httpx.Timeout(timeout_s, connect=10.0)
        client = anthropic.Anthropic(**client_kwargs)
        logger.info(
            "anthropic_tool_chat_start model=%s timeout_s=%.1f messages=%d tools=%d",
            backend.config.model, timeout_s, len(messages), len(tools or []),
        )

        # Separate system message from conversation messages
        system_text = backend.config.system_prompt
        conv_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_text = m.get("content", system_text)
            else:
                conv_messages.append(m)

        wire_model = backend._wire_model_name()

        def _make_call():
            kwargs = {
                "model": wire_model,
                "max_tokens": backend.config.max_tokens,
                "system": system_text,
                "messages": conv_messages,
                "temperature": backend.config.temperature,
            }
            if tools:
                kwargs["tools"] = _convert_tools_anthropic(tools)
                if tool_choice:
                    if tool_choice == "auto":
                        kwargs["tool_choice"] = {"type": "auto"}
                    elif tool_choice == "required":
                        kwargs["tool_choice"] = {"type": "any"}
                    elif tool_choice == "none":
                        pass  # Don't set tool_choice

            resp = client.messages.create(**kwargs)
            logger.info(
                "anthropic_tool_chat_done model=%s stop_reason=%s",
                backend.config.model, getattr(resp, "stop_reason", None),
            )

            # Capture usage
            usage_obj = None
            if hasattr(resp, 'usage') and resp.usage is not None:
                usage = resp.usage
                it = _coerce_int(getattr(usage, 'input_tokens', None))
                ot = _coerce_int(getattr(usage, 'output_tokens', None))
                tt = _compute_total(it, ot, None)
                if tt is not None:
                    usage_obj = Usage(
                        input_tokens=it or 0,
                        output_tokens=ot or 0,
                        total_tokens=tt,
                        model=backend.config.model,
                        provider=backend.config.provider
                    )
                    backend.last_usage = usage_obj

            # Extract content and tool calls
            content_parts = []
            tc_list = []
            raw_content = []
            for block in getattr(resp, "content", []) or []:
                if getattr(block, "type", "") == "text":
                    content_parts.append(getattr(block, "text", ""))
                    raw_content.append({"type": "text", "text": getattr(block, "text", "")})
                elif getattr(block, "type", "") == "tool_use":
                    tc_list.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    ))
                    raw_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            content = "\n".join(p for p in content_parts if p) or None
            stop = getattr(resp, "stop_reason", "end_turn")
            if stop == "tool_use":
                stop = "tool_use"
            elif stop == "max_tokens":
                stop = "max_tokens"
            else:
                stop = "end_turn"

            raw_msg = {"role": "assistant", "content": raw_content}

            return ChatResponse(
                content=content,
                tool_calls=tc_list,
                stop_reason=stop,
                usage=usage_obj,
                raw_message=raw_msg,
            )

        return backend._retry(_make_call)

    except ImportError:
        raise RuntimeError(
            "anthropic package not installed. Install it for tool-calling support."
        )
