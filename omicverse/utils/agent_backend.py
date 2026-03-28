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
import builtins as _builtins_mod
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

    # -----------------------------------------------------------------
    # Provider-specific delegates (thin wrappers calling helper modules)
    # -----------------------------------------------------------------

    # --- OpenAI ---
    def _apply_openai_chat_param_policy(self, kwargs):
        return _oai._apply_openai_chat_param_policy(self, kwargs)

    def _extract_openai_error_text(self, exc):
        return _oai._extract_openai_error_text(self, exc)

    def _adapt_openai_chat_kwargs_from_error(self, kwargs, exc):
        return _oai._adapt_openai_chat_kwargs_from_error(self, kwargs, exc)

    def _call_openai_chat_with_adaptation(self, request_fn, kwargs, *, base_url):
        return _oai._call_openai_chat_with_adaptation(self, request_fn, kwargs, base_url=base_url)

    @staticmethod
    def _is_openai_codex_base_url(base_url):
        return _oai._is_openai_codex_base_url(base_url)

    @staticmethod
    def _resolve_openai_codex_url(base_url):
        return _oai._resolve_openai_codex_url(base_url)

    @staticmethod
    def _decode_openai_codex_jwt(token):
        return _oai._decode_openai_codex_jwt(token)

    @classmethod
    def _extract_openai_codex_account_id(cls, token):
        return _oai._extract_openai_codex_account_id(token)

    @staticmethod
    def _openai_codex_user_agent():
        return _oai._openai_codex_user_agent()

    @staticmethod
    def _is_ollama_endpoint(base_url):
        return _oai._is_ollama_endpoint(base_url)

    def _wrap_openai_connection_error(self, exc, base_url):
        return _oai._wrap_openai_connection_error(self, exc, base_url)

    @staticmethod
    def _responses_jsonable(value):
        return _oai._responses_jsonable(value)

    @staticmethod
    def _extract_responses_output_items(payload):
        return _oai._extract_responses_output_items(payload)

    @staticmethod
    def _extract_responses_text_from_items(items):
        return _oai._extract_responses_text_from_items(items)

    @staticmethod
    def _extract_responses_tool_calls_from_items(items):
        return _oai._extract_responses_tool_calls_from_items(items)

    @staticmethod
    def _extract_responses_text_from_dict(payload):
        return _oai._extract_responses_text_from_dict(payload)

    def _build_openai_responses_request(self, *, messages, tools, tool_choice, include_system_messages):
        return _oai._build_openai_responses_request(self, messages=messages, tools=tools, tool_choice=tool_choice, include_system_messages=include_system_messages)

    @staticmethod
    def _iter_openai_codex_sse_events(raw_bytes):
        return _oai._iter_openai_codex_sse_events(raw_bytes)

    def _extract_openai_codex_final_response(self, events):
        return _oai._extract_openai_codex_final_response(self, events)

    def _call_openai_codex_responses(self, *, base_url, api_key, messages, tools, tool_choice):
        return _oai._call_openai_codex_responses(self, base_url=base_url, api_key=api_key, messages=messages, tools=tools, tool_choice=tool_choice)

    def _convert_messages_openai_responses(self, messages):
        return _oai._convert_messages_openai_responses(self, messages)

    def _build_chat_response_from_responses_payload(self, payload):
        return _oai._build_chat_response_from_responses_payload(self, payload)

    def _convert_tools_openai(self, tools):
        return _oai._convert_tools_openai(tools)

    def _convert_tools_openai_responses(self, tools):
        return _oai._convert_tools_openai_responses(tools)

    def _chat_via_openai_compatible(self, user_prompt):
        return _oai._chat_via_openai_compatible(self, user_prompt)

    def _chat_via_openai_http(self, base_url, api_key, user_prompt):
        return _oai._chat_via_openai_http(self, base_url, api_key, user_prompt)

    def _chat_via_openai_responses(self, base_url, api_key, user_prompt):
        return _oai._chat_via_openai_responses(self, base_url, api_key, user_prompt)

    def _chat_via_openai_responses_http(self, base_url, api_key, user_prompt):
        return _oai._chat_via_openai_responses_http(self, base_url, api_key, user_prompt)

    def _chat_tools_openai(self, messages, tools, tool_choice):
        return _oai._chat_tools_openai(self, messages, tools, tool_choice)

    def _chat_tools_openai_responses(self, messages, tools, tool_choice):
        return _oai._chat_tools_openai_responses(self, messages, tools, tool_choice)

    # --- Anthropic ---
    def _convert_tools_anthropic(self, tools):
        return _ant._convert_tools_anthropic(tools)

    def _chat_via_anthropic(self, user_prompt):
        return _ant._chat_via_anthropic(self, user_prompt)

    def _chat_tools_anthropic(self, messages, tools, tool_choice):
        return _ant._chat_tools_anthropic(self, messages, tools, tool_choice)

    # --- Gemini ---
    @staticmethod
    def _json_schema_to_gemini_schema(schema):
        return _gem._json_schema_to_gemini_schema(schema)

    @staticmethod
    def _messages_to_gemini_contents(messages):
        return _gem._messages_to_gemini_contents(messages)

    def _chat_via_gemini(self, user_prompt):
        return _gem._chat_via_gemini(self, user_prompt)

    def _chat_tools_gemini(self, messages, tools, tool_choice):
        return _gem._chat_tools_gemini(self, messages, tools, tool_choice)

    # --- Gemini CLI OAuth helpers ---
    @staticmethod
    def _gemini_uses_oauth_bearer(api_key):
        return _gem._gemini_uses_oauth_bearer(api_key)

    @staticmethod
    def _gemini_auth_headers(api_key):
        return _gem._gemini_auth_headers(api_key)

    @staticmethod
    def _gemini_oauth_payload(api_key):
        return _gem._gemini_oauth_payload(api_key)

    def _gemini_base_url(self):
        return _gem._gemini_base_url(self)

    def _gemini_generate_content_url(self):
        return _gem._gemini_generate_content_url(self)

    def _gemini_cli_base_url(self):
        return _gem._gemini_cli_base_url(self)

    def _gemini_cli_generate_content_url(self, stream=False):
        return _gem._gemini_cli_generate_content_url(self, stream=stream)

    @staticmethod
    def _gemini_function_response_payload(result):
        return _gem._gemini_function_response_payload(result)

    @staticmethod
    def _clean_schema_for_gemini_cli(schema):
        return _gem._clean_schema_for_gemini_cli(schema)

    def _messages_to_gemini_rest_contents(self, messages):
        return _gem._messages_to_gemini_rest_contents(self, messages)

    def _gemini_rest_generation_config(self):
        return _gem._gemini_rest_generation_config(self)

    def _gemini_rest_request(self, body, api_key):
        return _gem._gemini_rest_request(self, body, api_key)

    def _gemini_cli_request(self, body, api_key):
        return _gem._gemini_cli_request(self, body, api_key)

    def _capture_gemini_usage(self, payload):
        return _gem._capture_gemini_usage(self, payload)

    @staticmethod
    def _extract_gemini_text_and_tool_calls(payload):
        return _gem._extract_gemini_text_and_tool_calls(payload)

    def _chat_tools_gemini_rest(self, messages, tools, tool_choice, api_key):
        return _gem._chat_tools_gemini_rest(self, messages, tools, tool_choice, api_key)

    def _chat_via_gemini_rest(self, user_prompt, api_key):
        return _gem._chat_via_gemini_rest(self, user_prompt, api_key)

    # --- DashScope ---
    def _chat_via_dashscope(self, user_prompt):
        return _ds._chat_via_dashscope(self, user_prompt)

    # --- Local Python executor ---

    # Builtins removed from the sandbox to prevent code-evaluation and
    # unchecked import bypass.  A safe __import__ replacement is injected
    # separately so that normal ``import`` statements still work.
    _DENIED_BUILTINS: frozenset = frozenset({
        "eval", "exec", "compile", "__import__", "breakpoint",
    })

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
        from .agent_sandbox import CodeSecurityScanner, SafeOsProxy
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

        # -- Restricted builtins surface --
        safe_builtins = {
            k: v for k, v in vars(_builtins_mod).items()
            if k not in self._DENIED_BUILTINS
        }

        # Inject a safe __import__ that returns SafeOsProxy for ``os`` while
        # delegating all other imports to the real implementation.
        _real_import = _builtins_mod.__import__
        _os_proxy = SafeOsProxy()

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "os":
                return _os_proxy
            if name.startswith("os."):
                if fromlist:
                    # from os.path import join — return the deepest submodule
                    parts = name.split(".")[1:]
                    mod = _os_proxy
                    for part in parts:
                        mod = getattr(mod, part)
                    return mod
                # import os.path — return proxy (bound to name 'os')
                return _os_proxy
            return _real_import(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = _safe_import

        sandbox_globals: Dict[str, Any] = {
            "__name__": "__main__",
            "__builtins__": safe_builtins,
            "os": _os_proxy,
        }
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

    # --- Streaming delegates ---
    async def _run_generator_in_thread(self, generator_func):
        async for chunk in _stream._run_generator_in_thread(self, generator_func):
            yield chunk

    async def _stream_openai_compatible(self, user_prompt):
        async for chunk in _stream._stream_openai_compatible(self, user_prompt):
            yield chunk

    async def _stream_openai_http_fallback(self, base_url, api_key, user_prompt):
        async for chunk in _stream._stream_openai_http_fallback(self, base_url, api_key, user_prompt):
            yield chunk

    async def _stream_openai_responses(self, base_url, api_key, user_prompt):
        async for chunk in _stream._stream_openai_responses(self, base_url, api_key, user_prompt):
            yield chunk

    async def _stream_anthropic(self, user_prompt):
        async for chunk in _stream._stream_anthropic(self, user_prompt):
            yield chunk

    async def _stream_gemini(self, user_prompt):
        async for chunk in _stream._stream_gemini(self, user_prompt):
            yield chunk

    async def _stream_dashscope(self, user_prompt):
        async for chunk in _stream._stream_dashscope(self, user_prompt):
            yield chunk


__all__ = ["OmicVerseLLMBackend", "Usage", "ChatResponse", "ToolCall", "BackendConfig"]
