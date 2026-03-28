"""Shared types, configuration, retry logic, and utility helpers for the OmicVerse LLM backend.

This module is an internal implementation detail of the ``agent_backend`` package.
Public consumers should continue to import from ``omicverse.utils.agent_backend``.
"""
from __future__ import annotations

import concurrent.futures as _cf
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar
from urllib.error import HTTPError, URLError

from .model_config import WireAPI

__all__ = [
    "BackendConfig",
    "Usage",
    "ToolCall",
    "ChatResponse",
    "_coerce_int",
    "_compute_total",
    "_should_retry",
    "_retry_with_backoff",
    "_request_timeout_seconds",
    "_get_shared_executor",
]

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
    WireAPI.LOCAL:              "_run_python_local",  # non-streaming; handled specially in _stream_async
}

# ---------------------------------------------------------------------------
# Shared ThreadPoolExecutor  (P3-1: replaces per-call creation)
# ---------------------------------------------------------------------------

_SHARED_EXECUTOR: Optional[_cf.ThreadPoolExecutor] = None
_EXECUTOR_ATEXIT_REGISTERED: bool = False
_EXECUTOR_LOCK = threading.Lock()


def _shutdown_shared_executor() -> None:
    """Shut down the current shared executor at interpreter exit."""
    global _SHARED_EXECUTOR
    exc = _SHARED_EXECUTOR
    if exc is not None:
        try:
            exc.shutdown(wait=False)
        except (RuntimeError, OSError) as shutdown_exc:
            logger.debug("Shared executor shutdown error: %s", shutdown_exc)


def _get_shared_executor() -> _cf.ThreadPoolExecutor:
    """Return the module-level shared executor, creating it if needed.

    Uses double-checked locking to avoid duplicate construction under
    concurrent access from multiple threads.
    """
    global _SHARED_EXECUTOR, _EXECUTOR_ATEXIT_REGISTERED
    if _SHARED_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _SHARED_EXECUTOR is None:
                _SHARED_EXECUTOR = _cf.ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="ovagent-stream",
                )
                if not _EXECUTOR_ATEXIT_REGISTERED:
                    import atexit
                    atexit.register(_shutdown_shared_executor)
                    _EXECUTOR_ATEXIT_REGISTERED = True
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
    except (TypeError, ValueError, OverflowError) as exc:
        logger.debug("_coerce_int fallback for %r: %s", value, exc)
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
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    factor: float = 2.0,
    jitter: float = 0.5,
    **kwargs,
) -> T:
    """Retry a function call with exponential backoff and jitter.

    Parameters
    ----------
    func : Callable
        Function to retry
    *args
        Positional arguments forwarded to *func*
    max_attempts : int
        Maximum number of attempts (keyword-only, default: 3)
    base_delay : float
        Base delay in seconds (keyword-only, default: 1.0)
    factor : float
        Exponential backoff factor (keyword-only, default: 2.0)
    jitter : float
        Maximum jitter factor as proportion of delay (keyword-only, default: 0.5)
    **kwargs
        Keyword arguments forwarded to *func*

    Returns
    -------
    Result of func(*args, **kwargs)

    Raises
    ------
    RuntimeError
        If all retry attempts are exhausted
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
