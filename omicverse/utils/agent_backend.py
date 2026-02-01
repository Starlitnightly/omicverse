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
import contextlib
import io
import json
import logging
import os
import random
import ssl
import textwrap
import traceback
import warnings
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from .model_config import ModelConfig, WireAPI, get_provider

# Type variable for retry decorator
T = TypeVar('T')

# Logger for retry operations
logger = logging.getLogger(__name__)


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

        method_name = _SYNC_DISPATCH.get(provider_info.wire_api)
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
        if payload.get("output_text"):
            return payload["output_text"]

        output = payload.get("output")
        if output is not None:
            if isinstance(output, str):
                return output
            if isinstance(output, dict):
                if output.get("text"):
                    return output["text"]
                content = output.get("content")
                if isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("text"):
                            return p["text"]
            if isinstance(output, list) and len(output) > 0:
                first = output[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict) and first.get("text"):
                    return first["text"]

        if payload.get("text"):
            return payload["text"]

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

            # Wrap SDK call with retry logic
            def _make_sdk_call():
                resp = client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
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
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        data = json.dumps(body).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # Wrap HTTP call with retry logic
        def _make_http_call():
            req = urllib_request.Request(url, data=data, headers=headers, method="POST")
            # SSL certificate verification is enabled by default for security
            ctx = ssl.create_default_context()
            try:
                with urllib_request.urlopen(req, context=ctx, timeout=90) as resp:
                    content = resp.read().decode("utf-8")
                    payload = json.loads(content)
                    choices = payload.get("choices", [])
                    if not choices:
                        raise RuntimeError(f"No choices in response: {payload}")
                    msg = choices[0].get("message", {})

                    # Capture usage information
                    usage_data = payload.get("usage", {})
                    if usage_data:
                        self.last_usage = Usage(
                            input_tokens=usage_data.get('prompt_tokens', 0),
                            output_tokens=usage_data.get('completion_tokens', 0),
                            total_tokens=usage_data.get('total_tokens', 0),
                            model=self.config.model,
                            provider=self.config.provider
                        )

                    return msg.get("content", "")
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

                input_payload = [
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
                ]

                # GPT-5 models use reasoning tokens - control effort for response quality
                # Set reasoning effort to 'high' for maximum reasoning capability and best quality responses
                logger.debug("Creating GPT-5 Responses API request with high reasoning effort...")
                resp = client.responses.create(
                    model=self.config.model,
                    input=input_payload,
                    instructions=self.config.system_prompt,
                    max_output_tokens=self.config.max_tokens,
                    reasoning={"effort": "high"}  # Use high reasoning effort for better quality responses
                )

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

                # Try output_text first (most common)
                if hasattr(resp, 'output_text') and resp.output_text:
                    logger.debug(f"✓ Extracted via output_text (length: {len(resp.output_text)} chars)")
                    logger.debug(f"Response preview (first 200 chars): {resp.output_text[:200]}")
                    return resp.output_text

                # Try output.text
                if hasattr(resp, 'output'):
                    output = resp.output
                    logger.debug(f"Found output attribute: type={type(output).__name__}")

                    # Direct string
                    if isinstance(output, str):
                        logger.debug(f"✓ Extracted via output (direct string, length: {len(output)} chars)")
                        return output
                    # Object with .text
                    if hasattr(output, 'text') and getattr(output, 'text'):
                        text = getattr(output, 'text')
                        logger.debug(f"✓ Extracted via output.text (length: {len(text)} chars)")
                        return text
                    # Object with .content (list of parts)
                    if hasattr(output, 'content'):
                        parts = getattr(output, 'content')
                        logger.debug(f"Found output.content: type={type(parts)}, length={len(parts) if hasattr(parts, '__len__') else 'N/A'}")
                        try:
                            for i, p in enumerate(parts):
                                # p may be object or dict
                                if hasattr(p, 'text') and getattr(p, 'text'):
                                    text = getattr(p, 'text')
                                    logger.debug(f"✓ Extracted via output.content[{i}].text (length: {len(text)} chars)")
                                    return text
                                if isinstance(p, dict) and p.get('text'):
                                    text = p['text']
                                    logger.debug(f"✓ Extracted via output.content[{i}]['text'] (length: {len(text)} chars)")
                                    return text
                        except Exception as e:
                            logger.debug(f"Error iterating output.content: {e}")
                            pass
                    # List (first element may be dict or object)
                    if isinstance(output, list) and len(output) > 0:
                        first = output[0]
                        logger.debug(f"Output is a list, first element type: {type(first).__name__}")
                        if isinstance(first, str):
                            logger.debug(f"✓ Extracted via output[0] (direct string, length: {len(first)} chars)")
                            return first
                        if hasattr(first, 'text') and getattr(first, 'text'):
                            text = getattr(first, 'text')
                            logger.debug(f"✓ Extracted via output[0].text (length: {len(text)} chars)")
                            return text
                        if isinstance(first, dict) and first.get('text'):
                            text = first['text']
                            logger.debug(f"✓ Extracted via output[0]['text'] (length: {len(text)} chars)")
                            return text

                # Fallback: try text attribute directly
                if hasattr(resp, 'text') and resp.text:
                    logger.debug(f"✓ Extracted via text attribute (length: {len(resp.text)} chars)")
                    return resp.text

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
            raise RuntimeError("Missing ANTHROPIC_API_KEY for Anthropic provider")
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=api_key)

            # Wrap Anthropic SDK call with retry logic
            def _make_anthropic_call():
                resp = client.messages.create(
                    model=self.config.model,
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

            # Generator function for streaming with retry on session creation
            def _stream_sdk():
                def _create_stream():
                    return client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": self.config.system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        stream=True,
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
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(base_url=base_url, api_key=api_key)

            # Generator function for streaming with retry on session creation
            def _stream_responses_sdk():
                def _create_stream():
                    logger.debug(f"Creating GPT-5 Responses API stream: model={self.config.model}")

                    input_payload = [
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
                    ]

                    return client.responses.create(
                        model=self.config.model,
                        input=input_payload,
                        instructions=self.config.system_prompt,
                        max_output_tokens=self.config.max_tokens,
                        reasoning={"effort": "high"},
                        stream=True  # Enable streaming for GPT-5 Responses API
                    )

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
            raise RuntimeError("Missing ANTHROPIC_API_KEY for Anthropic provider")

        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=api_key)

            # Generator function for streaming with retry on session creation
            def _stream_sdk():
                def _create_stream():
                    return client.messages.stream(
                        model=self.config.model,
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


__all__ = ["OmicVerseLLMBackend", "Usage"]
