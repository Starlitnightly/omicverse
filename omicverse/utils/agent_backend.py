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

Implementation note
-------------------
Provider-specific logic, streaming helpers, and shared types are extracted into
dedicated internal modules (``agent_backend_common``, ``agent_backend_openai``,
``agent_backend_anthropic``, ``agent_backend_gemini``, ``agent_backend_dashscope``,
``agent_backend_streaming``).  ``OmicVerseLLMBackend`` remains the stable public
facade; all other modules are implementation details.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import textwrap
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .model_config import ModelConfig, WireAPI, get_provider

# ---------------------------------------------------------------------------
# Re-export shared types and helpers from the common module
# ---------------------------------------------------------------------------
from .agent_backend_common import (  # noqa: F401 — public re-exports
    BackendConfig,
    ChatResponse,
    ToolCall,
    Usage,
    _OPENAI_CODEX_BASE_URL,
    _OPENAI_CODEX_JWT_CLAIM_PATH,
    _STREAM_DISPATCH,
    _SYNC_DISPATCH,
    _coerce_int,
    _compute_total,
    _get_shared_executor,
    _request_timeout_seconds,
    _retry_with_backoff,
    _should_retry,
)

# ---------------------------------------------------------------------------
# Provider adapter imports (internal)
# ---------------------------------------------------------------------------
from . import agent_backend_openai as _oai
from . import agent_backend_anthropic as _ant
from . import agent_backend_gemini as _gem
from .agent_backend_gemini import (  # noqa: F401 — public re-exports
    _GOOGLE_GEMINI_CLI_BASE_URL,
    _GOOGLE_GEMINI_CLI_UNSUPPORTED_SCHEMA_KEYS,
)
from . import agent_backend_dashscope as _ds
from . import agent_backend_streaming as _stream

# Type variable for retry decorator
T = TypeVar('T')

# Logger for retry operations
logger = logging.getLogger(__name__)


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

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

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
        timeout = _request_timeout_seconds()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, user_prompt),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"LLM request timed out after {timeout}s "
                f"(model={self.config.model}, provider={self.config.provider})"
            )
        # Ensure downstream always receives a string
        text = "" if result is None else str(result)
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
        timeout = _request_timeout_seconds()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._chat_sync, messages, tools, tool_choice),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"LLM chat request timed out after {timeout}s "
                f"(model={self.config.model}, provider={self.config.provider})"
            )
        return result

    # -----------------------------------------------------------------
    # Wire-model and provider helpers (kept on class for test compat)
    # -----------------------------------------------------------------

    def _wire_model_name(self) -> str:
        """Return the model name suitable for the wire API."""
        model = self.config.model
        try:
            model = ModelConfig.normalize_model_id(model)
        except (AttributeError, KeyError, ValueError, TypeError) as exc:
            logger.debug("normalize_model_id fallback for %r: %s", model, exc)

        for prefix in (
            "openai/", "google/", "anthropic/", "gemini/", "deepseek/",
            "minimax/", "moonshot/", "qianfan/", "synthetic/", "together/",
            "xai/", "xiaomi/", "grok/", "zhipu/", "qwen/",
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

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve API key from explicit config, registry env_key, or alt_env_keys."""
        if self.config.api_key:
            return self.config.api_key

        info = get_provider(self.config.provider)
        if info is None:
            return None

        if info.env_key:
            val = os.getenv(info.env_key)
            if val:
                return val

        for alt in info.alt_env_keys:
            val = os.getenv(alt)
            if val:
                return val

        return None

    def _retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Retry *func* with config retry settings, binding any args into a zero-arg closure."""
        return _retry_with_backoff(
            (lambda: func(*args, **kwargs)) if args or kwargs else func,
            max_attempts=self.config.max_retry_attempts,
            base_delay=self.config.retry_base_delay,
            factor=self.config.retry_backoff_factor,
            jitter=self.config.retry_jitter,
        )

    # -----------------------------------------------------------------
    # Sync dispatch
    # -----------------------------------------------------------------

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
        method_name = _CHAT_DISPATCH.get(wire)
        if method_name is None:
            raise RuntimeError(
                f"Tool-calling chat not supported for wire API '{wire.value}'"
            )
        return getattr(self, method_name)(messages, tools, tool_choice)

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

    # --- Local Python executor ---

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

        # Pre-execution security scan — SyntaxError is NOT swallowed so that
        # callers receive a diagnosable failure instead of silently falling
        # through to compile().
        from .agent_sandbox import CodeSecurityScanner, build_sandbox_globals
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
        except SyntaxError as exc:
            raise RuntimeError(f"Python execution failed: {exc}") from exc

        stdout = io.StringIO()
        stderr = io.StringIO()
        sandbox_globals = build_sandbox_globals()
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

    # --- Tool result formatting ---
    def format_tool_result_message(
        self, tool_call_id: str, tool_name: str, result: str
    ) -> Dict[str, Any]:
        """Format a tool result for appending to the messages list."""
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
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }


# ---------------------------------------------------------------------------
# Provider delegation registry
# ---------------------------------------------------------------------------
# Instead of ~60 handwritten forwarding methods on the class, the registry
# maps method names to their provider module and binding kind.
# _install_provider_delegates() materializes each entry as a late-bound
# method on OmicVerseLLMBackend so monkeypatching the module-level function
# is picked up at call time.
#
# kind: 'instance'  — func(backend, *args, **kwargs)
#       'static'    — func(*args, **kwargs), no backend arg
#       'async_gen' — async generator func(backend, *args, **kwargs)

_CHAT_DISPATCH = {
    WireAPI.CHAT_COMPLETIONS:   "_chat_tools_openai",
    WireAPI.ANTHROPIC_MESSAGES: "_chat_tools_anthropic",
    WireAPI.GEMINI_GENERATE:    "_chat_tools_gemini",
    WireAPI.DASHSCOPE:          "_chat_tools_openai",
}

_PROVIDER_DELEGATES = [
    # --- OpenAI: instance ---
    ("_apply_openai_chat_param_policy",          _oai, "instance"),
    ("_extract_openai_error_text",               _oai, "instance"),
    ("_adapt_openai_chat_kwargs_from_error",     _oai, "instance"),
    ("_call_openai_chat_with_adaptation",        _oai, "instance"),
    ("_wrap_openai_connection_error",            _oai, "instance"),
    ("_build_openai_responses_request",          _oai, "instance"),
    ("_extract_openai_codex_final_response",     _oai, "instance"),
    ("_call_openai_codex_responses",             _oai, "instance"),
    ("_convert_messages_openai_responses",       _oai, "instance"),
    ("_build_chat_response_from_responses_payload", _oai, "instance"),
    ("_chat_via_openai_compatible",              _oai, "instance"),
    ("_chat_via_openai_http",                    _oai, "instance"),
    ("_chat_via_openai_responses",               _oai, "instance"),
    ("_chat_via_openai_responses_http",          _oai, "instance"),
    ("_chat_tools_openai",                       _oai, "instance"),
    ("_chat_tools_openai_responses",             _oai, "instance"),
    # --- OpenAI: static ---
    ("_convert_tools_openai",                    _oai, "static"),
    ("_convert_tools_openai_responses",          _oai, "static"),
    ("_is_openai_codex_base_url",                _oai, "static"),
    ("_resolve_openai_codex_url",                _oai, "static"),
    ("_decode_openai_codex_jwt",                 _oai, "static"),
    ("_extract_openai_codex_account_id",         _oai, "static"),
    ("_openai_codex_user_agent",                 _oai, "static"),
    ("_is_ollama_endpoint",                      _oai, "static"),
    ("_responses_jsonable",                      _oai, "static"),
    ("_extract_responses_output_items",          _oai, "static"),
    ("_extract_responses_text_from_items",       _oai, "static"),
    ("_extract_responses_tool_calls_from_items", _oai, "static"),
    ("_extract_responses_text_from_dict",        _oai, "static"),
    ("_iter_openai_codex_sse_events",            _oai, "static"),
    # --- Anthropic: instance ---
    ("_chat_via_anthropic",                      _ant, "instance"),
    ("_chat_tools_anthropic",                    _ant, "instance"),
    # --- Anthropic: static ---
    ("_convert_tools_anthropic",                 _ant, "static"),
    # --- Gemini: instance ---
    ("_chat_via_gemini",                         _gem, "instance"),
    ("_chat_tools_gemini",                       _gem, "instance"),
    ("_gemini_base_url",                         _gem, "instance"),
    ("_gemini_generate_content_url",             _gem, "instance"),
    ("_gemini_cli_base_url",                     _gem, "instance"),
    ("_gemini_cli_generate_content_url",         _gem, "instance"),
    ("_messages_to_gemini_rest_contents",        _gem, "instance"),
    ("_gemini_rest_generation_config",           _gem, "instance"),
    ("_gemini_rest_request",                     _gem, "instance"),
    ("_gemini_cli_request",                      _gem, "instance"),
    ("_capture_gemini_usage",                    _gem, "instance"),
    ("_chat_tools_gemini_rest",                  _gem, "instance"),
    ("_chat_via_gemini_rest",                    _gem, "instance"),
    # --- Gemini: static ---
    ("_json_schema_to_gemini_schema",            _gem, "static"),
    ("_messages_to_gemini_contents",             _gem, "static"),
    ("_gemini_uses_oauth_bearer",                _gem, "static"),
    ("_gemini_auth_headers",                     _gem, "static"),
    ("_gemini_oauth_payload",                    _gem, "static"),
    ("_gemini_function_response_payload",        _gem, "static"),
    ("_clean_schema_for_gemini_cli",             _gem, "static"),
    ("_extract_gemini_text_and_tool_calls",      _gem, "static"),
    # --- DashScope: instance ---
    ("_chat_via_dashscope",                      _ds,  "instance"),
    # --- Streaming: async generators ---
    ("_run_generator_in_thread",                 _stream, "async_gen"),
    ("_stream_openai_compatible",                _stream, "async_gen"),
    ("_stream_openai_http_fallback",             _stream, "async_gen"),
    ("_stream_openai_responses",                 _stream, "async_gen"),
    ("_stream_anthropic",                        _stream, "async_gen"),
    ("_stream_gemini",                           _stream, "async_gen"),
    ("_stream_dashscope",                        _stream, "async_gen"),
]


def _install_provider_delegates(cls, delegates):
    """Materialize provider delegate methods onto *cls* from the registry.

    Each wrapper resolves the target function at call time via ``getattr``
    so that monkeypatching the module-level function is picked up.
    """
    for name, mod, kind in delegates:
        if kind == "instance":
            def _make(m, n):
                def _delegate(self, *args, **kwargs):
                    return getattr(m, n)(self, *args, **kwargs)
                _delegate.__name__ = n
                _delegate.__qualname__ = f"{cls.__name__}.{n}"
                return _delegate
            setattr(cls, name, _make(mod, name))
        elif kind == "static":
            def _make(m, n):
                def _delegate(*args, **kwargs):
                    return getattr(m, n)(*args, **kwargs)
                _delegate.__name__ = n
                _delegate.__qualname__ = f"{cls.__name__}.{n}"
                return staticmethod(_delegate)
            setattr(cls, name, _make(mod, name))
        elif kind == "async_gen":
            def _make(m, n):
                async def _delegate(self, *args, **kwargs):
                    async for chunk in getattr(m, n)(self, *args, **kwargs):
                        yield chunk
                _delegate.__name__ = n
                _delegate.__qualname__ = f"{cls.__name__}.{n}"
                return _delegate
            setattr(cls, name, _make(mod, name))
        else:
            raise ValueError(f"Unknown delegate kind: {kind!r} for {name}")


_install_provider_delegates(OmicVerseLLMBackend, _PROVIDER_DELEGATES)


__all__ = ["OmicVerseLLMBackend", "Usage", "ChatResponse", "ToolCall", "BackendConfig"]
