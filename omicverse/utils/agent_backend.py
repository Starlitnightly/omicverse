"""
OmicVerse internal LLM backend (Pantheon replacement)

This module provides a lightweight, provider-agnostic backend to call LLMs
directly, removing the dependency on the Pantheon framework. It is intentionally
minimal: it sends a system prompt plus the user prompt to the selected model
and returns the text content of the response.

Design goals:
- Keep public behavior in smart_agent.py intact (system + user messages → text)
- Support OpenAI and OpenAI-compatible providers out of the box
- GPT-5 series uses OpenAI Responses API with 'instructions' + 'input' parameters
- Provide graceful fallbacks and clear error messages for other providers
- Allow configuration of model parameters (max_tokens, temperature)

Note: Network calls require the relevant provider API keys to be present in the
environment, or passed explicitly to the backend constructor. This file does not
introduce any runtime dependency on Pantheon.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import platform
import random
import re
import ssl
import textwrap
import traceback
import warnings
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from .model_config import ModelConfig, WireAPI, get_provider

# Type variable for retry decorator
T = TypeVar('T')

# Logger for retry operations
logger = logging.getLogger(__name__)

_OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
_OPENAI_CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"


def _request_timeout_seconds() -> float:
    """Request timeout used for blocking SDK calls in agentic tool loops."""
    raw = os.environ.get("OV_AGENT_CHAT_TIMEOUT_SECONDS", "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 120.0


# ---------------------------------------------------------------------------
# Wire-API → method dispatch tables  (P0-1: replaces if/elif chains)
# ---------------------------------------------------------------------------

_SYNC_DISPATCH = {
    WireAPI.CHAT_COMPLETIONS:   "_chat_via_openai_compatible",
    WireAPI.ANTHROPIC_MESSAGES: "_chat_via_anthropic",
    WireAPI.GEMINI_GENERATE:    "_chat_via_gemini",
    WireAPI.DASHSCOPE:          "_chat_via_dashscope",
    WireAPI.LOCAL:              "_run_python_local",
}

_STREAM_DISPATCH = {
    WireAPI.CHAT_COMPLETIONS:   "_stream_openai_compatible",
    WireAPI.ANTHROPIC_MESSAGES: "_stream_anthropic",
    WireAPI.GEMINI_GENERATE:    "_stream_gemini",
    WireAPI.DASHSCOPE:          "_stream_dashscope",
    # LOCAL handled specially (non-streaming fallback)
}

# ---------------------------------------------------------------------------
# Shared ThreadPoolExecutor  (P3-1: replaces per-call creation)
# ---------------------------------------------------------------------------

import atexit as _atexit
import concurrent.futures as _cf

_SHARED_EXECUTOR: Optional[_cf.ThreadPoolExecutor] = None

def _get_shared_executor() -> _cf.ThreadPoolExecutor:
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is None or _SHARED_EXECUTOR._shutdown:
        _SHARED_EXECUTOR = _cf.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="ovagent-stream",
        )
        _atexit.register(_SHARED_EXECUTOR.shutdown, wait=False)
    return _SHARED_EXECUTOR


@dataclass
class BackendConfig:
    model: str
    api_key: Optional[str]
    endpoint: Optional[str]
    provider: str
    system_prompt: str
    max_tokens: int = 8192
    temperature: float = 0.2
    # Retry configuration
    max_retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    retry_jitter: float = 0.5


@dataclass
class Usage:
    """Token usage statistics for a model completion.

    Attributes
    ----------
    input_tokens : int
        Number of tokens in the prompt/input
    output_tokens : int
        Number of tokens in the generated response
    total_tokens : int
        Total tokens used (input + output)
    model : str
        Model identifier that generated this usage
    provider : str
        Provider name (openai, anthropic, google, dashscope, etc.)
    """
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    provider: str


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatResponse:
    """Structured response from a multi-turn chat with tool-calling support."""
    content: Optional[str]
    tool_calls: List[ToolCall]
    stop_reason: str  # "end_turn", "tool_use", "max_tokens"
    usage: Optional[Usage] = None
    raw_message: Optional[Any] = None  # provider-specific message object for re-injection


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort conversion to int; returns None when not numeric.

    Handles ints, floats, and digit-only strings. Avoids treating generic mocks
    or arbitrary objects as numeric to keep tests robust when using Mock().
    """
    try:
        if value is None:
            return None
        # Explicitly ignore common mock objects or objects without a sensible int()
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    except Exception:
        return None
    return None


def _compute_total(input_tokens: Optional[int], output_tokens: Optional[int], total_tokens: Optional[int]) -> Optional[int]:
    """Compute total tokens preferring provided total, else sum of components."""
    if isinstance(total_tokens, int):
        return total_tokens
    if isinstance(input_tokens, int) or isinstance(output_tokens, int):
        return (input_tokens or 0) + (output_tokens or 0)
    return None


def _should_retry(exc: Exception) -> bool:
    """Determine if an exception is retryable.

    Retry on:
    - Timeouts (URLError with timeout, socket timeout)
    - 429 (rate limit)
    - 5xx (server errors)
    - Connection resets, refused, TLS/SSL errors
    - Provider SDK-specific transient errors (RateLimit, ServiceUnavailable, etc.)

    Do not retry on:
    - 4xx (except 429) - client errors
    - Auth errors (401, 403)
    """
    # HTTP errors
    if isinstance(exc, HTTPError):
        # Retry on rate limit and server errors
        if exc.code == 429 or exc.code >= 500:
            return True
        # Don't retry on client errors (400, 401, 403, 404, etc.)
        return False

    # URL errors (timeouts, connection issues)
    if isinstance(exc, URLError):
        return True

    # Timeout errors
    if isinstance(exc, TimeoutError):
        return True

    # Provider SDK-specific exception class names indicating transient issues
    exc_type_name = type(exc).__name__
    transient_exception_names = [
        'timeout', 'connection', 'ratelimit', 'serviceunavailable',
        'apierror', 'throttl', 'unavailable', 'overload'
    ]
    if any(keyword in exc_type_name.lower() for keyword in transient_exception_names):
        return True

    # Connection and transient error patterns in exception messages
    exc_str = str(exc).lower()
    transient_message_patterns = [
        'timeout', 'timed out', 'connection', 'reset', 'refused', 'broken pipe',
        'connection reset by peer', 'rate limit', 'rate_limit', 'too many requests',
        'service unavailable', 'temporarily unavailable', 'try again',
        'ssl', 'tls', 'handshake', 'certificate', 'overload', 'capacity'
    ]
    if any(pattern in exc_str for pattern in transient_message_patterns):
        return True

    # Check wrapped exceptions (e.g., RuntimeError wrapping HTTPError)
    if hasattr(exc, '__cause__') and exc.__cause__ is not None:
        return _should_retry(exc.__cause__)

    # Check for HTTP error codes in RuntimeError messages
    if isinstance(exc, RuntimeError):
        exc_msg = str(exc)
        # Check for 429 or 5xx in the message
        if '429' in exc_msg or any(f'HTTP {code}' in exc_msg or f'HTTP Error {code}' in exc_msg for code in range(500, 600)):
            return True

    # Default: don't retry unknown errors
    return False


def _retry_with_backoff(
    func: Callable[..., T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    factor: float = 2.0,
    jitter: float = 0.5,
    *args,
    **kwargs
) -> T:
    """Retry a function call with exponential backoff and jitter.

    Parameters
    ----------
    func : Callable
        Function to retry
    max_attempts : int
        Maximum number of attempts (default: 3)
    base_delay : float
        Base delay in seconds (default: 1.0)
    factor : float
        Exponential backoff factor (default: 2.0)
    jitter : float
        Maximum jitter factor as proportion of delay (default: 0.5)
    *args, **kwargs
        Arguments to pass to func

    Returns
    -------
    Result of func(*args, **kwargs)

    Raises
    ------
    Exception
        Last exception encountered if all retries fail
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exception = exc

            # Don't retry if this is the last attempt or if error is not retryable
            if attempt == max_attempts - 1 or not _should_retry(exc):
                break

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (factor ** attempt)
            jitter_amount = delay * jitter * random.random()
            total_delay = delay + jitter_amount

            # Log retry attempt
            logger.warning(
                "Retry attempt %d/%d: %s: %s. Retrying in %.2fs...",
                attempt + 1,
                max_attempts,
                type(exc).__name__,
                exc,
                total_delay
            )

            time.sleep(total_delay)

    # All retries exhausted, raise the last exception with context
    raise RuntimeError(
        f"All {max_attempts} attempts failed. Last error: {type(last_exception).__name__}: {last_exception}"
    ) from last_exception


class OmicVerseLLMBackend:
    """Simple, async LLM client used by the OmicVerse Agent.

    Parameters
    ----------
    system_prompt : str
        Instructions injected as the system message for each request
    model : str
        Model identifier (must be recognized by ModelConfig)
    api_key : str, optional
        Provider API key; falls back to environment variables if omitted
    endpoint : str, optional
        Custom base URL (used to override default OpenAI-compatible routes)
    max_tokens : int, optional
        Maximum tokens in model response (default: 8192)
    temperature : float, optional
        Sampling temperature for model output (default: 0.2)
    max_retry_attempts : int, optional
        Maximum number of retry attempts on transient failures (default: 3)
    retry_base_delay : float, optional
        Base delay in seconds for exponential backoff (default: 1.0)
    retry_backoff_factor : float, optional
        Exponential backoff multiplier (default: 2.0)
    retry_jitter : float, optional
        Maximum jitter factor as proportion of delay (default: 0.5)
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        model: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        max_retry_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        retry_jitter: float = 0.5
    ) -> None:
        try:
            provider = ModelConfig.get_provider_from_model(model, endpoint)
        except TypeError:
            # Some tests monkeypatch the older single-argument signature.
            provider = ModelConfig.get_provider_from_model(model)
        self.config = BackendConfig(
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            provider=provider,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retry_attempts=max_retry_attempts,
            retry_base_delay=retry_base_delay,
            retry_backoff_factor=retry_backoff_factor,
            retry_jitter=retry_jitter,
        )
        # Token usage tracking
        self.last_usage: Optional[Usage] = None

        # Validate GPT-5 model accessibility
        if 'gpt-5' in model.lower():
            api_key = self._resolve_api_key()
            if not api_key:
                warnings.warn(
                    f"GPT-5 model '{model}' requires OPENAI_API_KEY to be set. "
                    "Set the environment variable or pass api_key to the constructor.",
                    UserWarning
                )
            logger.debug(f"GPT-5 model detected: {model}, using Responses API")

    async def run(self, user_prompt: str) -> str:
        """Run an LLM completion and return the response text."""

        # Reset usage to avoid stale data from previous calls
        self.last_usage = None

        # Validate input
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")
        if len(user_prompt) > 200000:
            raise ValueError(
                f"user_prompt too long ({len(user_prompt)} chars, max 200000)"
            )

        # Many provider SDKs are synchronous; use a thread to avoid blocking the loop
        result = await asyncio.to_thread(self._run_sync, user_prompt)
        # Ensure downstream always receives a string
        text = "" if result is None else str(result)
        # Return raw text; do not wrap with code fences to keep behavior consistent
        # with provider expectations and tests.
        return text

    async def stream(self, user_prompt: str):
        """Stream LLM completion tokens as they arrive.

        Yields text deltas from the model response. For providers that don't support
        streaming, yields the complete response as a single chunk.

        Parameters
        ----------
        user_prompt : str
            The user's input prompt

        Yields
        ------
        str
            Text delta (chunk) from the streaming response

        Examples
        --------
        >>> backend = OmicVerseLLMBackend(system_prompt="...", model="gpt-4o")
        >>> async for chunk in backend.stream("Analyze this data"):
        ...     print(chunk, end="", flush=True)
        """

        # Reset usage to avoid stale data from previous calls
        self.last_usage = None

        # Validate input
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")
        if len(user_prompt) > 200000:
            raise ValueError(
                f"user_prompt too long ({len(user_prompt)} chars, max 200000)"
            )

        # Stream responses using provider-specific implementations
        async for chunk in self._stream_async(user_prompt):
            yield chunk

    async def _stream_async(self, user_prompt: str):
        """Internal async generator that dispatches to provider-specific streaming."""
        provider_info = get_provider(self.config.provider)
        if provider_info is None:
            raise RuntimeError(
                f"Provider '{self.config.provider}' is not registered. "
                "Use the non-streaming run() method or register the provider first."
            )

        wire = provider_info.wire_api

        # LOCAL has no streaming — fall back to sync
        if wire == WireAPI.LOCAL:
            yield await asyncio.to_thread(self._run_python_local, user_prompt)
            return

        method_name = _STREAM_DISPATCH.get(wire)
        if method_name is None:
            raise RuntimeError(
                f"Provider '{self.config.provider}' (wire={wire.value}) "
                "is not supported for streaming yet."
            )

        method = getattr(self, method_name)
        async for chunk in method(user_prompt):
            yield chunk

    # ---------------------------------------------------------------------
    # Multi-turn chat with tool-calling (agentic loop support)
    # ---------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        """Multi-turn chat with tool-calling support.

        Parameters
        ----------
        messages : list of dict
            Conversation messages (system, user, assistant, tool roles).
        tools : list of dict, optional
            Tool definitions in provider-agnostic format:
            [{"name": "...", "description": "...", "parameters": {...}}]
        tool_choice : str, optional
            "auto" (default), "required", or "none".

        Returns
        -------
        ChatResponse
            Structured response with content and/or tool_calls.
        """
        self.last_usage = None
        result = await asyncio.to_thread(
            self._chat_sync, messages, tools, tool_choice
        )
        return result

    def _wire_model_name(self) -> str:
        """Return the model name suitable for the wire API.

        Strips provider prefixes (``anthropic/``, ``gemini/``, etc.) that are
        used internally for routing but not accepted by the upstream APIs.
        """
        model = self.config.model
        try:
            model = ModelConfig.normalize_model_id(model)
        except Exception:
            # Fall back to the raw configured model if normalization fails.
            pass

        for prefix in (
            "openai/",
            "google/",
            "anthropic/",
            "gemini/",
            "deepseek/",
            "minimax/",
            "moonshot/",
            "qianfan/",
            "synthetic/",
            "together/",
            "xai/",
            "xiaomi/",
            "grok/",
            "zhipu/",
            "qwen/",
        ):
            if model.startswith(prefix):
                return model[len(prefix):]
        return model

    def _effective_wire_api(self, provider_info: Any) -> WireAPI:
        """Resolve the effective wire API for the configured provider."""
        wire = provider_info.wire_api if provider_info is not None else WireAPI.CHAT_COMPLETIONS
        if self.config.endpoint and wire == WireAPI.CHAT_COMPLETIONS:
            if "claude" in self.config.model.lower():
                return WireAPI.ANTHROPIC_MESSAGES
        return wire

    def _missing_api_key_error(self, provider_info: Any, fallback_display_name: str) -> RuntimeError:
        display_name = fallback_display_name
        env_key = ""
        if provider_info is not None:
            display_name = str(getattr(provider_info, "display_name", "") or display_name)
            env_key = str(getattr(provider_info, "env_key", "") or "")
        if env_key:
            return RuntimeError(f"Missing {env_key} for {display_name} provider")
        return RuntimeError(f"Missing API key for {display_name} provider")

    def _apply_openai_chat_param_policy(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply known provider/model-specific chat-completions parameter overrides.

        This mirrors OpenClaw's model-aware approach: treat provider/model
        capabilities as distinct from the generic OpenAI-compatible wire format.
        """
        adapted = dict(kwargs)
        wire_model = str(adapted.get("model") or self._wire_model_name()).strip()
        wire_model_lower = wire_model.lower()

        # Moonshot Kimi K2.5 currently only accepts temperature=1.
        if self.config.provider == "moonshot" and wire_model_lower.startswith("kimi-k2.5"):
            current_temperature = adapted.get("temperature")
            if current_temperature != 1:
                logger.info(
                    "openai_chat_param_override model=%s provider=%s param=temperature from=%s to=1",
                    self.config.model,
                    self.config.provider,
                    current_temperature,
                )
                adapted["temperature"] = 1

        return adapted

    def _extract_openai_error_text(self, exc: Exception) -> str:
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
        self,
        kwargs: Dict[str, Any],
        exc: Exception,
    ) -> Optional[tuple[Dict[str, Any], str]]:
        """Best-effort one-shot adaptation for provider/model parameter mismatches."""
        text = self._extract_openai_error_text(exc)
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
        self,
        request_fn: Callable[[Dict[str, Any]], Any],
        kwargs: Dict[str, Any],
        *,
        base_url: str,
    ) -> Any:
        """Execute an OpenAI-compatible chat request with one adaptive retry."""
        try:
            return request_fn(kwargs)
        except Exception as exc:
            adapted = self._adapt_openai_chat_kwargs_from_error(kwargs, exc)
            if adapted is None:
                raise
            retried_kwargs, change_summary = adapted
            logger.info(
                "openai_chat_retry_adapted model=%s provider=%s endpoint=%s changes=%s",
                self.config.model,
                self.config.provider,
                base_url,
                change_summary,
            )
            return request_fn(retried_kwargs)

    def _chat_sync(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
    ) -> ChatResponse:
        """Dispatch multi-turn chat to the correct provider."""
        provider_info = get_provider(self.config.provider)
        if provider_info is None:
            raise RuntimeError(
                f"Provider '{self.config.provider}' is not registered."
            )

        wire = self._effective_wire_api(provider_info)
        if wire == WireAPI.CHAT_COMPLETIONS:
            return self._chat_tools_openai(messages, tools, tool_choice)
        elif wire == WireAPI.ANTHROPIC_MESSAGES:
            return self._chat_tools_anthropic(messages, tools, tool_choice)
        elif wire == WireAPI.GEMINI_GENERATE:
            return self._chat_tools_gemini(messages, tools, tool_choice)
        elif wire == WireAPI.DASHSCOPE:
            return self._chat_tools_openai(messages, tools, tool_choice)
        else:
            raise RuntimeError(
                f"Tool-calling chat not supported for wire API '{wire.value}'"
            )

    def _convert_tools_openai(self, tools: List[Dict]) -> List[Dict]:
        """Convert provider-agnostic tool defs to OpenAI format."""
        return [{"type": "function", "function": t} for t in tools]

    def _convert_tools_openai_responses(self, tools: List[Dict]) -> List[Dict]:
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

    def _convert_tools_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """Convert provider-agnostic tool defs to Anthropic format."""
        return [{
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        } for t in tools]

    def _chat_tools_openai(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        tool_choice: Optional[str],
    ) -> ChatResponse:
        """Multi-turn chat via OpenAI-compatible API with tool support."""
        if self.config.provider == "openai" and ModelConfig.requires_responses_api(self.config.model):
            return self._chat_tools_openai_responses(messages, tools, tool_choice)

        info = get_provider(self.config.provider)
        base_url = self.config.endpoint or (info.base_url if info else "https://api.openai.com/v1")
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider '{self.config.provider}'."
            )

        try:
            from openai import OpenAI  # type: ignore
            timeout_s = _request_timeout_seconds()
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

            def _make_call():
                try:
                    # GPT-5 series requires max_completion_tokens instead of max_tokens
                    if ModelConfig.requires_responses_api(self.config.model):
                        token_param = {"max_completion_tokens": self.config.max_tokens}
                    else:
                        token_param = {"max_tokens": self.config.max_tokens}

                    kwargs = self._apply_openai_chat_param_policy({
                        "model": self._wire_model_name(),
                        "messages": messages,
                        "temperature": self.config.temperature,
                        **token_param,
                    })
                    if tools:
                        kwargs["tools"] = self._convert_tools_openai(tools)
                        kwargs["tool_choice"] = tool_choice or "auto"

                    logger.info(
                        "openai_tool_chat_start model=%s provider=%s endpoint=%s tool_choice=%s messages=%d tools=%d timeout_s=%.1f",
                        self.config.model,
                        self.config.provider,
                        base_url,
                        kwargs.get("tool_choice", "none"),
                        len(messages),
                        len(tools or []),
                        timeout_s,
                    )
                    resp = self._call_openai_chat_with_adaptation(
                        lambda payload: client.chat.completions.create(**payload),
                        kwargs,
                        base_url=base_url,
                    )
                    logger.info(
                        "openai_tool_chat_done model=%s finish_reason=%s choices=%d",
                        self.config.model,
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
                                model=self.config.model,
                                provider=self.config.provider
                            )
                            self.last_usage = usage_obj

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
                    raise self._wrap_openai_connection_error(exc, base_url) from exc

            return self._retry(_make_call)

        except ImportError:
            raise RuntimeError(
                "openai package not installed. Install openai>=1.0 for tool-calling support."
            )

    @staticmethod
    def _is_openai_codex_base_url(base_url: str) -> bool:
        normalized = str(base_url or "").strip().rstrip("/")
        return normalized.startswith(_OPENAI_CODEX_BASE_URL)

    @staticmethod
    def _resolve_openai_codex_url(base_url: str) -> str:
        normalized = str(base_url or _OPENAI_CODEX_BASE_URL).strip().rstrip("/")
        if normalized.endswith("/codex/responses"):
            return normalized
        if normalized.endswith("/codex"):
            return f"{normalized}/responses"
        return f"{normalized}/codex/responses"

    @staticmethod
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

    @classmethod
    def _extract_openai_codex_account_id(cls, token: str) -> str:
        payload = cls._decode_openai_codex_jwt(token)
        auth = payload.get(_OPENAI_CODEX_JWT_CLAIM_PATH)
        if not isinstance(auth, dict):
            return ""
        account_id = str(auth.get("chatgpt_account_id") or "").strip()
        return account_id

    @staticmethod
    def _openai_codex_user_agent() -> str:
        try:
            system = platform.system().lower() or "unknown"
            release = platform.release() or "unknown"
            machine = platform.machine() or "unknown"
            return f"pi ({system} {release}; {machine})"
        except Exception:
            return "pi (python)"

    @staticmethod
    def _is_ollama_endpoint(base_url: str) -> bool:
        normalized = str(base_url or "").strip().lower().rstrip("/")
        return (
            "127.0.0.1:11434" in normalized
            or "localhost:11434" in normalized
            or normalized.endswith(":11434/v1")
            or normalized.endswith(":11434")
        )

    def _wrap_openai_connection_error(self, exc: Exception, base_url: str) -> RuntimeError:
        exc_name = type(exc).__name__.lower()
        exc_text = str(exc).lower()
        if "connection" not in exc_name and "connection" not in exc_text and "refused" not in exc_text:
            return RuntimeError(str(exc))
        if self._is_ollama_endpoint(base_url):
            return RuntimeError(
                f"Could not connect to Ollama at {base_url}. Start the Ollama server and verify the model is installed."
            )
        return RuntimeError(f"OpenAI-compatible connection failed for {base_url}: {exc}")

    def _build_openai_responses_request(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]],
        tool_choice: Optional[str],
        include_system_messages: bool,
    ) -> Dict[str, Any]:
        input_items = self._convert_messages_openai_responses(messages)
        if not include_system_messages:
            input_items = [
                item for item in input_items
                if not (isinstance(item, dict) and item.get("role") == "system")
            ]

        kwargs: Dict[str, Any] = {
            "model": self._wire_model_name(),
            "input": input_items,
            "max_output_tokens": self.config.max_tokens,
            "reasoning": {"effort": "medium"},
        }
        if self.config.system_prompt:
            kwargs["instructions"] = self.config.system_prompt
        if tools:
            kwargs["tools"] = self._convert_tools_openai_responses(tools)
            kwargs["tool_choice"] = tool_choice or "auto"
        return kwargs

    @staticmethod
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

    def _extract_openai_codex_final_response(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                    final_payload = self._responses_jsonable(response)

        if not final_payload:
            raise RuntimeError("Codex response stream completed without a final response payload")
        return final_payload

    def _call_openai_codex_responses(
        self,
        *,
        base_url: str,
        api_key: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]],
        tool_choice: Optional[str],
    ) -> Dict[str, Any]:
        account_id = self._extract_openai_codex_account_id(api_key)
        if not account_id:
            raise RuntimeError(
                "OpenAI OAuth access token is missing chatgpt_account_id; please rerun `omicverse jarvis --setup`."
            )

        payload = self._build_openai_responses_request(
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

        url = self._resolve_openai_codex_url(base_url)
        timeout_s = _request_timeout_seconds()
        request = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "chatgpt-account-id": account_id,
                "OpenAI-Beta": "responses=experimental",
                "originator": "pi",
                "User-Agent": self._openai_codex_user_agent(),
                "accept": "text/event-stream",
                "content-type": "application/json",
            },
            method="POST",
        )

        logger.info(
            "openai_codex_responses_start model=%s endpoint=%s tool_choice=%s messages=%d tools=%d timeout_s=%.1f",
            self.config.model,
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

        events = self._iter_openai_codex_sse_events(raw_bytes)
        final_payload = self._extract_openai_codex_final_response(events)
        logger.info(
            "openai_codex_responses_done model=%s output_items=%d tool_calls=%d",
            self.config.model,
            len(self._extract_responses_output_items(final_payload)),
            len(self._extract_responses_tool_calls_from_items(self._extract_responses_output_items(final_payload))),
        )
        return final_payload

    @staticmethod
    def _responses_jsonable(value: Any) -> Any:
        """Convert SDK response objects into JSON-serializable structures."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                k: OmicVerseLLMBackend._responses_jsonable(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                OmicVerseLLMBackend._responses_jsonable(v)
                for v in value
            ]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return OmicVerseLLMBackend._responses_jsonable(
                    model_dump(exclude_none=True)
                )
            except TypeError:
                return OmicVerseLLMBackend._responses_jsonable(model_dump())
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return OmicVerseLLMBackend._responses_jsonable(to_dict())
        if hasattr(value, "__dict__"):
            return {
                k: OmicVerseLLMBackend._responses_jsonable(v)
                for k, v in value.__dict__.items()
                if not k.startswith("_")
            }
        return str(value)

    @staticmethod
    def _extract_responses_output_items(payload: Any) -> List[Dict[str, Any]]:
        """Return canonical output items from a Responses SDK object or dict."""
        if payload is None:
            return []
        if not isinstance(payload, dict):
            payload = OmicVerseLLMBackend._responses_jsonable(payload)
        output = payload.get("output")
        if output is None:
            return []
        if isinstance(output, list):
            return [
                item for item in (
                    OmicVerseLLMBackend._responses_jsonable(item)
                    for item in output
                )
                if isinstance(item, dict)
            ]
        if isinstance(output, dict):
            return [output]
        return []

    @staticmethod
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

    @staticmethod
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

    def _convert_messages_openai_responses(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize mixed conversation state into Responses API input items."""
        converted: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue

            item_type = message.get("type")
            if item_type in {"function_call", "function_call_output", "message", "reasoning"}:
                normalized = self._responses_jsonable(message)
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
                        normalized = self._responses_jsonable(block)
                        if isinstance(normalized, dict):
                            parts.append(normalized)
                converted.append({"role": role, "content": parts})
            else:
                if isinstance(content, dict):
                    content = json.dumps(content, ensure_ascii=False)
                converted.append({"role": role, "content": content})
        return converted

    def _build_chat_response_from_responses_payload(self, payload: Dict[str, Any]) -> ChatResponse:
        """Build the generic ChatResponse contract from a Responses payload."""
        output_items = self._extract_responses_output_items(payload)
        tool_calls = self._extract_responses_tool_calls_from_items(output_items)
        content = ""
        try:
            content = self._extract_responses_text_from_dict(payload)
        except Exception:
            content = self._extract_responses_text_from_items(output_items)

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
                    model=self.config.model,
                    provider=self.config.provider,
                )
                self.last_usage = usage_obj

        raw_message: Optional[Any] = output_items if tool_calls else None
        stop_reason = "tool_use" if tool_calls else "end_turn"
        return ChatResponse(
            content=content or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage_obj,
            raw_message=raw_message,
        )

    def _chat_tools_openai_responses(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]],
        tool_choice: Optional[str],
    ) -> ChatResponse:
        """Multi-turn tool chat via the OpenAI Responses API."""
        info = get_provider(self.config.provider)
        base_url = self.config.endpoint or (info.base_url if info else "https://api.openai.com/v1")
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider '{self.config.provider}'."
            )

        if self._is_openai_codex_base_url(base_url):
            def _make_codex_call():
                payload = self._call_openai_codex_responses(
                    base_url=base_url,
                    api_key=api_key,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                return self._build_chat_response_from_responses_payload(payload)

            return self._retry(_make_codex_call)

        try:
            from openai import OpenAI  # type: ignore

            timeout_s = _request_timeout_seconds()
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

            def _make_call():
                kwargs = self._build_openai_responses_request(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    include_system_messages=True,
                )

                logger.info(
                    "openai_responses_tool_chat_start model=%s provider=%s endpoint=%s tool_choice=%s messages=%d tools=%d timeout_s=%.1f",
                    self.config.model,
                    self.config.provider,
                    base_url,
                    kwargs.get("tool_choice", "none"),
                    len(messages),
                    len(tools or []),
                    timeout_s,
                )
                resp = client.responses.create(**kwargs)
                payload = self._responses_jsonable(resp)
                logger.info(
                    "openai_responses_tool_chat_done model=%s output_items=%d tool_calls=%d",
                    self.config.model,
                    len(self._extract_responses_output_items(payload)),
                    len(self._extract_responses_tool_calls_from_items(self._extract_responses_output_items(payload))),
                )
                return self._build_chat_response_from_responses_payload(payload)

            return self._retry(_make_call)

        except ImportError:
            raise RuntimeError(
                "openai package not installed. Install openai>=1.0 for Responses API tool support."
            )

    def _chat_tools_anthropic(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        tool_choice: Optional[str],
    ) -> ChatResponse:
        """Multi-turn chat via Anthropic Messages API with tool support."""
        api_key = self._resolve_api_key()
        if not api_key:
            raise self._missing_api_key_error(get_provider(self.config.provider), "Anthropic")

        try:
            import anthropic  # type: ignore
            client_kwargs = {"api_key": api_key}
            if self.config.endpoint:
                # Proxy endpoints often use /v1 suffix for OpenAI compat.
                # The Anthropic SDK prepends /v1/messages itself, so strip
                # trailing /v1 to avoid /v1/v1/messages.
                base = self.config.endpoint.rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]
                client_kwargs["base_url"] = base
            import httpx
            timeout_s = _request_timeout_seconds()
            client_kwargs["timeout"] = httpx.Timeout(timeout_s, connect=10.0)
            client = anthropic.Anthropic(**client_kwargs)
            logger.info(
                "anthropic_tool_chat_start model=%s timeout_s=%.1f messages=%d tools=%d",
                self.config.model, timeout_s, len(messages), len(tools or []),
            )

            # Separate system message from conversation messages
            system_text = self.config.system_prompt
            conv_messages = []
            for m in messages:
                if m.get("role") == "system":
                    system_text = m.get("content", system_text)
                else:
                    conv_messages.append(m)

            wire_model = self._wire_model_name()

            def _make_call():
                kwargs = {
                    "model": wire_model,
                    "max_tokens": self.config.max_tokens,
                    "system": system_text,
                    "messages": conv_messages,
                    "temperature": self.config.temperature,
                }
                if tools:
                    kwargs["tools"] = self._convert_tools_anthropic(tools)
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
                    self.config.model, getattr(resp, "stop_reason", None),
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
                            model=self.config.model,
                            provider=self.config.provider
                        )
                        self.last_usage = usage_obj

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

            return self._retry(_make_call)

        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Install it for tool-calling support."
            )

    def _chat_tools_gemini(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        tool_choice: Optional[str],
    ) -> ChatResponse:
        """Multi-turn chat via Gemini API with function calling support."""
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini provider")

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
                        parameters=self._json_schema_to_gemini_schema(t.get("parameters", {})),
                    )
                    func_decls.append(fd)
                gemini_tools = [genai.protos.Tool(function_declarations=func_decls)]

            model = genai.GenerativeModel(
                model_name=self.config.model.split("/", 1)[-1],
                system_instruction=self.config.system_prompt,
                tools=gemini_tools,
            )

            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

            # Convert messages to Gemini Content format
            gemini_contents = self._messages_to_gemini_contents(messages)

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
                            model=self.config.model,
                            provider=self.config.provider
                        )
                        self.last_usage = usage_obj

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
                                id=f"gemini_{fc.name}_{id(fc)}",
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
                    raw_message=None,  # Gemini history managed separately
                )

            return self._retry(_make_call)

        except ImportError:
            raise RuntimeError(
                "google-generativeai package not installed. Install it for tool-calling support."
            )

    @staticmethod
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
        except Exception:
            return None

    @staticmethod
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
                contents.append(genai.protos.Content(
                    role=gemini_role,
                    parts=[genai.protos.Part(text=content)],
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

    def format_tool_result_message(
        self, tool_call_id: str, tool_name: str, result: str
    ) -> Dict[str, Any]:
        """Format a tool result for appending to the messages list.

        Returns a message dict in the correct format for the current provider.
        """
        if self.config.provider == "openai" and ModelConfig.requires_responses_api(self.config.model):
            return {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": result,
            }

        provider_info = get_provider(self.config.provider)
        wire = self._effective_wire_api(provider_info)

        if wire == WireAPI.ANTHROPIC_MESSAGES:
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result,
                }],
            }
        else:
            # OpenAI-compatible format (also used by DashScope)
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _run_sync(self, user_prompt: str) -> str:
        """Dispatch to the correct sync provider method via the registry."""
        provider_info = get_provider(self.config.provider)
        if provider_info is None:
            raise RuntimeError(
                f"Provider '{self.config.provider}' is not registered. "
                "Please choose a supported model or register the provider first."
            )

        method_name = _SYNC_DISPATCH.get(self._effective_wire_api(provider_info))
        if method_name is None:
            raise RuntimeError(
                f"Provider '{self.config.provider}' (wire={provider_info.wire_api.value}) "
                "is not supported by the internal backend yet."
            )

        method = getattr(self, method_name)
        return method(user_prompt)

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve API key from explicit config, registry env_key, or alt_env_keys."""
        if self.config.api_key:
            return self.config.api_key

        info = get_provider(self.config.provider)
        if info is None:
            return None

        # Primary env key
        if info.env_key:
            val = os.getenv(info.env_key)
            if val:
                return val

        # Alternative env keys (e.g. ZHIPUAI_API_KEY)
        for alt in info.alt_env_keys:
            val = os.getenv(alt)
            if val:
                return val

        return None

    # ----------------------------- retry helper (P1-3) ----------------------------
    def _retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Retry *func* using the retry settings from self.config."""
        return _retry_with_backoff(
            func,
            max_attempts=self.config.max_retry_attempts,
            base_delay=self.config.retry_base_delay,
            factor=self.config.retry_backoff_factor,
            jitter=self.config.retry_jitter,
            *args,
            **kwargs,
        )

    # ---- GPT-5 Responses-API text extraction (P1-3: shared helper) -----------
    @staticmethod
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
                text = OmicVerseLLMBackend._extract_responses_text_from_items(
                    OmicVerseLLMBackend._extract_responses_output_items(payload)
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

    # ----------------------------- OpenAI compatible -----------------------------
    def _chat_via_openai_compatible(self, user_prompt: str) -> str:
        info = get_provider(self.config.provider)
        base_url = self.config.endpoint or (info.base_url if info else "https://api.openai.com/v1")
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider '{self.config.provider}'. Set the appropriate environment variable or pass api_key."
            )

        # Check if model requires Responses API (gpt-5 series)
        if ModelConfig.requires_responses_api(self.config.model):
            return self._chat_via_openai_responses(base_url, api_key, user_prompt)

        # Otherwise use Chat Completions API (standard path for gpt-4o, gpt-4o-mini, etc.)
        # Try modern OpenAI SDK first, then fallback to raw HTTP
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(base_url=base_url, api_key=api_key)
            wire_model = self._wire_model_name()

            # Wrap SDK call with retry logic
            def _make_sdk_call():
                kwargs = self._apply_openai_chat_param_policy({
                    "model": wire_model,
                    "messages": [
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                })
                resp = self._call_openai_chat_with_adaptation(
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
                        self.last_usage = Usage(
                            input_tokens=pt or 0,
                            output_tokens=ct or 0,
                            total_tokens=tt,
                            model=self.config.model,
                            provider=self.config.provider
                        )

                return choice.message.content or ""

            return self._retry(_make_sdk_call)

        except ImportError:
            # OpenAI SDK not installed, fallback to HTTP
            return self._chat_via_openai_http(base_url, api_key, user_prompt)
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
            return self._chat_via_openai_http(base_url, api_key, user_prompt)

    def _chat_via_openai_http(self, base_url: str, api_key: str, user_prompt: str) -> str:
        url = base_url.rstrip("/") + "/chat/completions"
        body = self._apply_openai_chat_param_policy({
            "model": self._wire_model_name(),
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
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
                        self.last_usage = Usage(
                            input_tokens=usage_data.get('prompt_tokens', 0),
                            output_tokens=usage_data.get('completion_tokens', 0),
                            total_tokens=usage_data.get('total_tokens', 0),
                            model=self.config.model,
                            provider=self.config.provider
                        )

                    return msg.get("content", "")

            try:
                return self._call_openai_chat_with_adaptation(
                    _request_with_kwargs,
                    body,
                    base_url=base_url,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"OpenAI-compatible HTTP call failed: {type(exc).__name__}: {exc}"
                ) from exc

        return self._retry(_make_http_call)

    # ----------------------------- OpenAI Responses API (gpt-5 series) -----------------------------
    def _chat_via_openai_responses(self, base_url: str, api_key: str, user_prompt: str) -> str:
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
        if self._is_openai_codex_base_url(base_url):
            def _make_codex_call():
                payload = self._call_openai_codex_responses(
                    base_url=base_url,
                    api_key=api_key,
                    messages=[{"role": "user", "content": user_prompt}],
                    tools=None,
                    tool_choice=None,
                )
                text = self._extract_responses_text_from_dict(payload)
                if not text:
                    text = self._extract_responses_text_from_items(
                        self._extract_responses_output_items(payload)
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
                        self.last_usage = Usage(
                            input_tokens=pt or 0,
                            output_tokens=ct or 0,
                            total_tokens=tt,
                            model=self.config.model,
                            provider=self.config.provider,
                        )
                return text

            return self._retry(_make_codex_call)

        # Try OpenAI SDK first
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(base_url=base_url, api_key=api_key)

            # Wrap SDK call with retry logic
            def _make_responses_sdk_call():
                # Responses API uses 'instructions' for system prompt
                # and 'input' as a string (not message array)
                # Note: gpt-5 Responses API does not support temperature parameter
                logger.debug(f"GPT-5 Responses API call: model={self.config.model}, max_tokens={self.config.max_tokens}")
                logger.debug(f"User prompt length: {len(user_prompt)} chars")

                kwargs = self._build_openai_responses_request(
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
                        self.last_usage = Usage(
                            input_tokens=pt or 0,
                            output_tokens=ct or 0,
                            total_tokens=tt,
                            model=self.config.model,
                            provider=self.config.provider
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

            return self._retry(_make_responses_sdk_call)

        except ImportError:
            # OpenAI SDK not installed, fallback to HTTP
            return self._chat_via_openai_responses_http(base_url, api_key, user_prompt)
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
            return self._chat_via_openai_responses_http(base_url, api_key, user_prompt)

    def _chat_via_openai_responses_http(self, base_url: str, api_key: str, user_prompt: str) -> str:
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
                        "model": self.config.model,
                        "input": [
                            {
                                "role": "system",
                                "content": [
                                    {"type": "input_text", "text": self.config.system_prompt}
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": user_prompt}
                                ],
                            },
                        ],
                        "instructions": self.config.system_prompt,
                        "max_output_tokens": self.config.max_tokens,
                        "reasoning": {"effort": "high"}  # Use high reasoning effort for better quality responses
                    }
                else:
                    return {
                        "model": self.config.model,
                        "input": user_prompt,
                        "instructions": self.config.system_prompt,
                        "max_output_tokens": self.config.max_tokens,
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
                            self.last_usage = Usage(
                                input_tokens=pt_i or 0,
                                output_tokens=ct_i or 0,
                                total_tokens=tt_i,
                                model=self.config.model,
                                provider=self.config.provider
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
                                self.last_usage = Usage(
                                    input_tokens=usage_data.get('input_tokens', 0) or usage_data.get('prompt_tokens', 0),
                                    output_tokens=usage_data.get('output_tokens', 0) or usage_data.get('completion_tokens', 0),
                                    total_tokens=usage_data.get('total_tokens', 0),
                                    model=self.config.model,
                                    provider=self.config.provider
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

        return self._retry(_make_responses_http_call)

    # ------------------------------- Local Python executor -------------------------------
    def _run_python_local(self, user_prompt: str) -> str:
        """Execute Python code locally when provider is set to 'python'."""

        code = (user_prompt or "").strip()

        # Strip simple markdown fences to mirror LLM responses
        if code.startswith("```"):
            lines = code.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines).strip()

        code = textwrap.dedent(code).strip()
        if not code:
            raise ValueError("No Python code provided for execution")

        # Pre-execution security scan
        from .agent_sandbox import CodeSecurityScanner
        from .agent_errors import SecurityViolationError
        scanner = CodeSecurityScanner()
        try:
            violations = scanner.scan(code)
            if scanner.has_critical(violations):
                report = scanner.format_report(violations)
                raise SecurityViolationError(
                    f"Code blocked by security scanner:\n{report}",
                    violations=violations,
                )
        except SyntaxError:
            pass  # Syntax errors handled downstream by compile()

        stdout = io.StringIO()
        stderr = io.StringIO()
        sandbox_globals: Dict[str, Any] = {"__name__": "__main__"}
        sandbox_locals: Dict[str, Any] = {}

        try:
            compiled = compile(code, "<ov-agent-python>", "exec")
        except SyntaxError as exc:
            raise RuntimeError(f"Python execution failed: {exc}") from exc

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compiled, sandbox_globals, sandbox_locals)
        except Exception as exc:
            tb = traceback.format_exc()
            raise RuntimeError(f"Python execution raised {type(exc).__name__}: {exc}\n{tb}") from exc

        stderr_output = stderr.getvalue().strip()
        stdout_output = stdout.getvalue().strip()
        if stderr_output:
            raise RuntimeError(f"Python execution wrote to stderr:\n{stderr_output}")

        if not stdout_output:
            for key in ("result", "output", "res"):
                if key in sandbox_locals:
                    stdout_output = repr(sandbox_locals[key])
                    break

        # Track a zero-usage record for parity with network providers
        self.last_usage = Usage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model=self.config.model,
            provider=self.config.provider
        )

        return stdout_output

    # ------------------------------- Anthropic SDK -------------------------------
    def _chat_via_anthropic(self, user_prompt: str) -> str:
        api_key = self._resolve_api_key()
        if not api_key:
            raise self._missing_api_key_error(get_provider(self.config.provider), "Anthropic")
        try:
            import anthropic  # type: ignore

            client_kwargs = {"api_key": api_key}
            if self.config.endpoint:
                base = self.config.endpoint.rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]
                client_kwargs["base_url"] = base
            import httpx
            timeout_s = _request_timeout_seconds()
            client_kwargs["timeout"] = httpx.Timeout(timeout_s, connect=10.0)
            client = anthropic.Anthropic(**client_kwargs)

            wire_model = self._wire_model_name()

            # Wrap Anthropic SDK call with retry logic
            def _make_anthropic_call():
                resp = client.messages.create(
                    model=wire_model,
                    max_tokens=self.config.max_tokens,
                    system=self.config.system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                )

                # Capture usage information from Anthropic
                if hasattr(resp, 'usage') and resp.usage is not None:
                    usage = resp.usage
                    input_tokens = _coerce_int(getattr(usage, 'input_tokens', None))
                    output_tokens = _coerce_int(getattr(usage, 'output_tokens', None))
                    total_tokens = _compute_total(input_tokens, output_tokens, _coerce_int(getattr(usage, 'total_tokens', None)))
                    if total_tokens is not None:
                        self.last_usage = Usage(
                            input_tokens=input_tokens or 0,
                            output_tokens=output_tokens or 0,
                            total_tokens=total_tokens,
                            model=self.config.model,
                            provider=self.config.provider
                        )

                # Concatenate text chunks if present
                parts = []
                for block in getattr(resp, "content", []) or []:
                    if getattr(block, "type", "") == "text":
                        parts.append(getattr(block, "text", ""))
                return "\n".join([p for p in parts if p])

            return self._retry(_make_anthropic_call)

        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Install anthropic or choose an OpenAI-compatible model."
            )

    # -------------------------------- Gemini SDK --------------------------------
    def _chat_via_gemini(self, user_prompt: str) -> str:
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini provider")
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=self.config.model.split("/", 1)[-1],
                system_instruction=self.config.system_prompt
            )

            # Generate config for temperature control
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
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
                        self.last_usage = Usage(
                            input_tokens=input_tokens or 0,
                            output_tokens=output_tokens or 0,
                            total_tokens=total_tokens,
                            model=self.config.model,
                            provider=self.config.provider
                        )

                # Extract text robustly; Gemini may omit parts when finish_reason != 1
                text = ""
                try:
                    text = getattr(resp, "text", "") or ""
                except Exception:
                    text = ""
                if not text and getattr(resp, "candidates", None):
                    for cand in resp.candidates or []:
                        parts = getattr(getattr(cand, "content", None), "parts", None) or []
                        piece = "".join(str(getattr(p, "text", "") or "") for p in parts)
                        if piece:
                            text += piece
                    text = text.strip()
                return text

            return self._retry(_make_gemini_call)

        except ImportError:
            raise RuntimeError(
                "google-generativeai package not installed. Install it or choose an OpenAI-compatible model."
            )

    # ------------------------------- DashScope SDK -------------------------------
    def _chat_via_dashscope(self, user_prompt: str) -> str:
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY for Qwen provider")
        try:
            from http import HTTPStatus
            from dashscope import Generation  # type: ignore

            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Wrap DashScope SDK call with retry logic
            def _make_dashscope_call():
                resp = Generation.call(
                    model=self.config.model.replace("/", ":"),
                    messages=messages,
                    api_key=api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                if resp.status_code != HTTPStatus.OK:
                    raise RuntimeError(f"DashScope call failed: {getattr(resp, 'message', resp.status_code)}")

                # Capture usage information from DashScope
                if hasattr(resp, 'usage') and resp.usage is not None:
                    usage = resp.usage
                    # DashScope uses input_tokens and output_tokens
                    input_tokens = _coerce_int(getattr(usage, 'input_tokens', None))
                    output_tokens = _coerce_int(getattr(usage, 'output_tokens', None))
                    total_tokens = _compute_total(input_tokens, output_tokens, _coerce_int(getattr(usage, 'total_tokens', None)))
                    if total_tokens is not None:
                        self.last_usage = Usage(
                            input_tokens=input_tokens or 0,
                            output_tokens=output_tokens or 0,
                            total_tokens=total_tokens,
                            model=self.config.model,
                            provider=self.config.provider
                        )

                # Qwen responses are OpenAI-like
                choices = getattr(resp, "output", {}).get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return ""

            return self._retry(_make_dashscope_call)

        except ImportError:
            raise RuntimeError(
                "dashscope package not installed. Install it or choose an OpenAI-compatible model."
            )

    # ----------------------------- Streaming Methods -----------------------------

    async def _run_generator_in_thread(self, generator_func):
        """Helper to run a synchronous generator in a thread and yield results asynchronously.

        This helper bridges synchronous SDK streaming generators with async generators by:
        1. Running the sync generator in a background thread
        2. Pushing chunks to an async queue
        3. Yielding chunks from the queue

        Parameters
        ----------
        generator_func : Callable
            A callable that returns a generator (sync)

        Yields
        ------
        Any
            Items yielded by the generator

        Raises
        ------
        Exception
            Any exception raised by the generator
        """
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        exception_holder = []

        def _run_stream():
            try:
                for item in generator_func():
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            except Exception as exc:
                exception_holder.append(exc)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # Start streaming in background thread (shared executor — P3-1)
        executor = _get_shared_executor()
        future = executor.submit(_run_stream)

        # Yield chunks as they arrive
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

        # Check for exceptions
        if exception_holder:
            raise exception_holder[0]

    async def _stream_openai_compatible(self, user_prompt: str):
        """Stream responses from OpenAI-compatible providers.

        Uses SDK streaming if available, falls back to non-streaming HTTP response.
        Includes retry logic for transient stream session creation failures.
        """
        info = get_provider(self.config.provider)
        base_url = self.config.endpoint or (info.base_url if info else "https://api.openai.com/v1")
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider '{self.config.provider}'. Set the appropriate environment variable or pass api_key."
            )

        # Check if model requires Responses API (gpt-5 series)
        if ModelConfig.requires_responses_api(self.config.model):
            # Use proper streaming for Responses API
            async for chunk in self._stream_openai_responses(base_url, api_key, user_prompt):
                yield chunk
            return

        # Try OpenAI SDK streaming first
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(base_url=base_url, api_key=api_key)
            wire_model = self._wire_model_name()

            # Generator function for streaming with retry on session creation
            def _stream_sdk():
                def _create_stream():
                    kwargs = self._apply_openai_chat_param_policy({
                        "model": wire_model,
                        "messages": [
                            {"role": "system", "content": self.config.system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "stream": True,
                    })
                    return self._call_openai_chat_with_adaptation(
                        lambda payload: client.chat.completions.create(**payload),
                        kwargs,
                        base_url=base_url,
                    )

                # Retry stream creation on transient failures
                stream = self._retry(_create_stream)

                for chunk in stream:
                    # Capture usage from streaming chunks (typically in final chunk)
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = chunk.usage
                        self.last_usage = Usage(
                            input_tokens=getattr(usage, 'prompt_tokens', 0),
                            output_tokens=getattr(usage, 'completion_tokens', 0),
                            total_tokens=getattr(usage, 'total_tokens', 0),
                            model=self.config.model,
                            provider=self.config.provider
                        )

                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield delta.content

            # Use helper to run generator in thread
            async for chunk in self._run_generator_in_thread(_stream_sdk):
                yield chunk

        except ImportError:
            # OpenAI SDK not installed, fall back to non-streaming HTTP
            logger.warning("OpenAI SDK not available for streaming, falling back to non-streaming HTTP")
            async for chunk in self._stream_openai_http_fallback(base_url, api_key, user_prompt):
                yield chunk
        except Exception as exc:
            # Log SDK failure and fall back to HTTP
            logger.warning(
                "OpenAI SDK streaming failed (%s: %s), falling back to non-streaming HTTP",
                type(exc).__name__,
                exc
            )
            async for chunk in self._stream_openai_http_fallback(base_url, api_key, user_prompt):
                yield chunk

    async def _stream_openai_http_fallback(self, base_url: str, api_key: str, user_prompt: str):
        """Non-streaming HTTP fallback for OpenAI-compatible providers."""
        # HTTP streaming is complex; just return full response
        result = await asyncio.to_thread(
            self._chat_via_openai_http,
            base_url,
            api_key,
            user_prompt
        )
        yield result

    async def _stream_openai_responses(self, base_url: str, api_key: str, user_prompt: str):
        """Stream responses from OpenAI Responses API (gpt-5 series) with proper streaming support."""
        if self._is_openai_codex_base_url(base_url):
            result = await asyncio.to_thread(
                self._chat_via_openai_responses,
                base_url,
                api_key,
                user_prompt,
            )
            yield result
            return

        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(base_url=base_url, api_key=api_key)

            # Generator function for streaming with retry on session creation
            def _stream_responses_sdk():
                def _create_stream():
                    logger.debug(f"Creating GPT-5 Responses API stream: model={self.config.model}")

                    kwargs = self._build_openai_responses_request(
                        messages=[{"role": "user", "content": user_prompt}],
                        tools=None,
                        tool_choice=None,
                        include_system_messages=False,
                    )
                    kwargs["reasoning"] = {"effort": "high"}
                    kwargs["stream"] = True
                    return client.responses.create(**kwargs)

                # Retry stream creation on transient failures
                stream = self._retry(_create_stream)

                # Process streaming chunks
                for chunk in stream:
                    # Capture usage from streaming chunks (typically in final chunk)
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = chunk.usage
                        pt = _coerce_int(getattr(usage, 'input_tokens', None))
                        if pt is None:
                            pt = _coerce_int(getattr(usage, 'prompt_tokens', None))
                        ct = _coerce_int(getattr(usage, 'output_tokens', None))
                        if ct is None:
                            ct = _coerce_int(getattr(usage, 'completion_tokens', None))
                        tt = _coerce_int(getattr(usage, 'total_tokens', None))
                        tt = _compute_total(pt, ct, tt)
                        if tt is not None:
                            self.last_usage = Usage(
                                input_tokens=pt or 0,
                                output_tokens=ct or 0,
                                total_tokens=tt,
                                model=self.config.model,
                                provider=self.config.provider
                            )

                    # Extract text delta from chunk
                    # Try multiple extraction paths for Responses API streaming
                    delta_text = None

                    # Try output_text_delta
                    if hasattr(chunk, 'output_text_delta') and chunk.output_text_delta:
                        delta_text = chunk.output_text_delta
                    # Try delta attribute
                    elif hasattr(chunk, 'delta'):
                        delta = chunk.delta
                        if isinstance(delta, str):
                            delta_text = delta
                        elif hasattr(delta, 'text') and delta.text:
                            delta_text = delta.text
                        elif hasattr(delta, 'content'):
                            content = delta.content
                            if isinstance(content, str):
                                delta_text = content
                            elif isinstance(content, list) and len(content) > 0:
                                for part in content:
                                    if hasattr(part, 'text') and part.text:
                                        delta_text = part.text
                                        break
                                    elif isinstance(part, dict) and part.get('text'):
                                        delta_text = part['text']
                                        break
                    # Try output attribute with delta
                    elif hasattr(chunk, 'output'):
                        output = chunk.output
                        if isinstance(output, str):
                            delta_text = output
                        elif hasattr(output, 'delta') and output.delta:
                            if isinstance(output.delta, str):
                                delta_text = output.delta
                            elif hasattr(output.delta, 'text'):
                                delta_text = output.delta.text

                    if delta_text:
                        yield delta_text

            # Use helper to run generator in thread
            async for chunk in self._run_generator_in_thread(_stream_responses_sdk):
                yield chunk

        except ImportError:
            # OpenAI SDK not installed, fall back to non-streaming HTTP
            logger.warning("OpenAI SDK not available for Responses API streaming, falling back to non-streaming")
            result = await asyncio.to_thread(
                self._chat_via_openai_responses,
                base_url,
                api_key,
                user_prompt
            )
            yield result
        except Exception as exc:
            # Log SDK failure and fall back to non-streaming
            logger.warning(
                "OpenAI Responses API streaming failed (%s: %s), falling back to non-streaming",
                type(exc).__name__,
                exc
            )
            result = await asyncio.to_thread(
                self._chat_via_openai_responses,
                base_url,
                api_key,
                user_prompt
            )
            yield result

    async def _stream_anthropic(self, user_prompt: str):
        """Stream responses from Anthropic Claude models with retry on session creation."""
        api_key = self._resolve_api_key()
        if not api_key:
            raise self._missing_api_key_error(get_provider(self.config.provider), "Anthropic")

        try:
            import anthropic  # type: ignore

            client_kwargs = {"api_key": api_key}
            if self.config.endpoint:
                client_kwargs["base_url"] = self.config.endpoint
            import httpx
            timeout_s = _request_timeout_seconds()
            client_kwargs["timeout"] = httpx.Timeout(timeout_s * 3, connect=10.0)  # streaming gets more time
            client = anthropic.Anthropic(**client_kwargs)
            wire_model = self._wire_model_name()

            # Generator function for streaming with retry on session creation
            def _stream_sdk():
                def _create_stream():
                    return client.messages.stream(
                        model=wire_model,
                        max_tokens=self.config.max_tokens,
                        system=self.config.system_prompt,
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.config.temperature,
                    )

                # Retry stream creation on transient failures
                stream_context = self._retry(_create_stream)

                with stream_context as stream:
                    for text in stream.text_stream:
                        yield text

                    # Capture usage information after stream completes
                    if hasattr(stream, 'get_final_message'):
                        final_message = stream.get_final_message()
                        if hasattr(final_message, 'usage') and final_message.usage:
                            usage = final_message.usage
                            input_tokens = getattr(usage, 'input_tokens', 0)
                            output_tokens = getattr(usage, 'output_tokens', 0)
                            self.last_usage = Usage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                total_tokens=input_tokens + output_tokens,
                                model=self.config.model,
                                provider=self.config.provider
                            )

            # Use helper to run generator in thread
            async for chunk in self._run_generator_in_thread(_stream_sdk):
                yield chunk

        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Install anthropic or choose an OpenAI-compatible model."
            )

    async def _stream_gemini(self, user_prompt: str):
        """Stream responses from Google Gemini models with retry on session creation."""
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini provider")

        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=self.config.model.split("/", 1)[-1],
                system_instruction=self.config.system_prompt
            )

            # Generate config for temperature control
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

            # Generator function for streaming with retry on session creation
            def _stream_sdk():
                def _create_stream():
                    return model.generate_content(
                        user_prompt,
                        generation_config=generation_config,
                        stream=True
                    )

                # Retry stream creation on transient failures
                response = self._retry(_create_stream)

                # Yield chunks as they arrive (true streaming) and track
                # the last chunk for usage metadata.
                last_chunk = None
                for chunk in response:
                    last_chunk = chunk
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text

                # Capture usage from the last chunk
                if last_chunk is not None and hasattr(last_chunk, 'usage_metadata') and last_chunk.usage_metadata is not None:
                    usage = last_chunk.usage_metadata
                    input_tokens = _coerce_int(getattr(usage, 'prompt_token_count', None))
                    output_tokens = _coerce_int(getattr(usage, 'candidates_token_count', None))
                    total_tokens = _compute_total(input_tokens, output_tokens, _coerce_int(getattr(usage, 'total_token_count', None)))
                    if total_tokens is not None:
                        self.last_usage = Usage(
                            input_tokens=input_tokens or 0,
                            output_tokens=output_tokens or 0,
                            total_tokens=total_tokens,
                            model=self.config.model,
                            provider=self.config.provider
                        )

            # Use helper to run generator in thread
            async for chunk in self._run_generator_in_thread(_stream_sdk):
                yield chunk

        except ImportError:
            raise RuntimeError(
                "google-generativeai package not installed. Install it or choose an OpenAI-compatible model."
            )

    async def _stream_dashscope(self, user_prompt: str):
        """Stream responses from Alibaba DashScope (Qwen) models with retry.

        Note: DashScope SDK supports streaming via incremental_output=True parameter.
        """
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY for Qwen provider")

        try:
            from http import HTTPStatus
            from dashscope import Generation  # type: ignore

            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Generator function for streaming with retry on session creation
            def _stream_sdk():
                def _create_stream():
                    return Generation.call(
                        model=self.config.model.replace("/", ":"),
                        messages=messages,
                        api_key=api_key,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        result_format='message',
                        stream=True,
                        incremental_output=True
                    )

                # Retry stream creation on transient failures
                responses = self._retry(_create_stream)

                for response in responses:
                    if response.status_code == HTTPStatus.OK:
                        # Capture usage information from streaming response
                        if hasattr(response, 'usage') and response.usage is not None:
                            usage = response.usage
                            input_tokens = _coerce_int(getattr(usage, 'input_tokens', None))
                            output_tokens = _coerce_int(getattr(usage, 'output_tokens', None))
                            total_tokens = _compute_total(input_tokens, output_tokens, _coerce_int(getattr(usage, 'total_tokens', None)))
                            if total_tokens is not None:
                                self.last_usage = Usage(
                                    input_tokens=input_tokens or 0,
                                    output_tokens=output_tokens or 0,
                                    total_tokens=total_tokens,
                                    model=self.config.model,
                                    provider=self.config.provider
                                )

                        # Extract text from streaming response
                        output = getattr(response, "output", {})
                        if isinstance(output, dict):
                            choices = output.get("choices", [])
                            if choices:
                                message = choices[0].get("message", {})
                                content = message.get("content", "")
                                if content:
                                    yield content
                    else:
                        # Error in streaming
                        raise RuntimeError(
                            f"DashScope streaming failed: {getattr(response, 'message', response.status_code)}"
                        )

            # Use helper to run generator in thread
            async for chunk in self._run_generator_in_thread(_stream_sdk):
                yield chunk

        except ImportError:
            raise RuntimeError(
                "dashscope package not installed. Install it or choose an OpenAI-compatible model."
            )


__all__ = ["OmicVerseLLMBackend", "Usage", "ChatResponse", "ToolCall"]
