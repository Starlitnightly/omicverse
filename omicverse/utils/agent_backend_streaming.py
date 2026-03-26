"""Streaming dispatch and provider-specific streaming helpers for OmicVerseLLMBackend.

Internal module — import from ``omicverse.utils.agent_backend`` instead.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .agent_backend_common import (
    Usage,
    _coerce_int,
    _compute_total,
    _get_shared_executor,
    _request_timeout_seconds,
)
from .model_config import ModelConfig, get_provider

if TYPE_CHECKING:
    from typing import Any, AsyncGenerator, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: bridge synchronous generator → async generator via thread
# ---------------------------------------------------------------------------

async def _run_generator_in_thread(backend, generator_func):
    """Helper to run a synchronous generator in a thread and yield results asynchronously.

    This helper bridges synchronous SDK streaming generators with async generators by:
    1. Running the sync generator in a background thread
    2. Pushing chunks to an async queue
    3. Yielding chunks from the queue

    Parameters
    ----------
    backend : OmicVerseLLMBackend
        The backend instance (used for shared state).
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


# ---------------------------------------------------------------------------
# OpenAI-compatible streaming
# ---------------------------------------------------------------------------

async def _stream_openai_compatible(backend, user_prompt: str):
    """Stream responses from OpenAI-compatible providers.

    Uses SDK streaming if available, falls back to non-streaming HTTP response.
    Includes retry logic for transient stream session creation failures.
    """
    from .agent_backend_openai import (
        _apply_openai_chat_param_policy,
        _call_openai_chat_with_adaptation,
    )

    info = get_provider(backend.config.provider)
    base_url = backend.config.endpoint or (info.base_url if info else "https://api.openai.com/v1")
    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{backend.config.provider}'. Set the appropriate environment variable or pass api_key."
        )

    # Check if model requires Responses API (gpt-5 series)
    if ModelConfig.requires_responses_api(backend.config.model):
        # Use proper streaming for Responses API
        async for chunk in _stream_openai_responses(backend, base_url, api_key, user_prompt):
            yield chunk
        return

    # Try OpenAI SDK streaming first
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=base_url, api_key=api_key)
        wire_model = backend._wire_model_name()

        # Generator function for streaming with retry on session creation
        def _stream_sdk():
            def _create_stream():
                kwargs = _apply_openai_chat_param_policy(backend, {
                    "model": wire_model,
                    "messages": [
                        {"role": "system", "content": backend.config.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": backend.config.temperature,
                    "max_tokens": backend.config.max_tokens,
                    "stream": True,
                })
                return _call_openai_chat_with_adaptation(
                    backend,
                    lambda payload: client.chat.completions.create(**payload),
                    kwargs,
                    base_url=base_url,
                )

            # Retry stream creation on transient failures
            stream = backend._retry(_create_stream)

            for chunk in stream:
                # Capture usage from streaming chunks (typically in final chunk)
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = chunk.usage
                    backend.last_usage = Usage(
                        input_tokens=getattr(usage, 'prompt_tokens', 0),
                        output_tokens=getattr(usage, 'completion_tokens', 0),
                        total_tokens=getattr(usage, 'total_tokens', 0),
                        model=backend.config.model,
                        provider=backend.config.provider
                    )

                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content

        # Use helper to run generator in thread
        async for chunk in _run_generator_in_thread(backend, _stream_sdk):
            yield chunk

    except ImportError:
        # OpenAI SDK not installed, fall back to non-streaming HTTP
        logger.warning("OpenAI SDK not available for streaming, falling back to non-streaming HTTP")
        async for chunk in _stream_openai_http_fallback(backend, base_url, api_key, user_prompt):
            yield chunk
    except Exception as exc:
        # Log SDK failure and fall back to HTTP
        logger.warning(
            "OpenAI SDK streaming failed (%s: %s), falling back to non-streaming HTTP",
            type(exc).__name__,
            exc
        )
        async for chunk in _stream_openai_http_fallback(backend, base_url, api_key, user_prompt):
            yield chunk


# ---------------------------------------------------------------------------
# OpenAI HTTP fallback (non-streaming)
# ---------------------------------------------------------------------------

async def _stream_openai_http_fallback(backend, base_url: str, api_key: str, user_prompt: str):
    """Non-streaming HTTP fallback for OpenAI-compatible providers."""
    from .agent_backend_openai import _chat_via_openai_http

    # HTTP streaming is complex; just return full response
    result = await asyncio.to_thread(
        _chat_via_openai_http,
        backend,
        base_url,
        api_key,
        user_prompt
    )
    yield result


# ---------------------------------------------------------------------------
# OpenAI Responses API streaming (gpt-5 series)
# ---------------------------------------------------------------------------

async def _stream_openai_responses(backend, base_url: str, api_key: str, user_prompt: str):
    """Stream responses from OpenAI Responses API (gpt-5 series) with proper streaming support."""
    from .agent_backend_openai import (
        _build_openai_responses_request,
        _chat_via_openai_responses,
        _is_openai_codex_base_url,
    )

    if _is_openai_codex_base_url(base_url):
        result = await asyncio.to_thread(
            _chat_via_openai_responses,
            backend,
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
                logger.debug(f"Creating GPT-5 Responses API stream: model={backend.config.model}")

                kwargs = _build_openai_responses_request(
                    backend,
                    messages=[{"role": "user", "content": user_prompt}],
                    tools=None,
                    tool_choice=None,
                    include_system_messages=False,
                )
                kwargs["reasoning"] = {"effort": "high"}
                kwargs["stream"] = True
                return client.responses.create(**kwargs)

            # Retry stream creation on transient failures
            stream = backend._retry(_create_stream)

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
                        backend.last_usage = Usage(
                            input_tokens=pt or 0,
                            output_tokens=ct or 0,
                            total_tokens=tt,
                            model=backend.config.model,
                            provider=backend.config.provider
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
        async for chunk in _run_generator_in_thread(backend, _stream_responses_sdk):
            yield chunk

    except ImportError:
        # OpenAI SDK not installed, fall back to non-streaming HTTP
        logger.warning("OpenAI SDK not available for Responses API streaming, falling back to non-streaming")
        result = await asyncio.to_thread(
            _chat_via_openai_responses,
            backend,
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
            _chat_via_openai_responses,
            backend,
            base_url,
            api_key,
            user_prompt
        )
        yield result


# ---------------------------------------------------------------------------
# Anthropic streaming
# ---------------------------------------------------------------------------

async def _stream_anthropic(backend, user_prompt: str):
    """Stream responses from Anthropic Claude models with retry on session creation."""
    api_key = backend._resolve_api_key()
    if not api_key:
        raise backend._missing_api_key_error(get_provider(backend.config.provider), "Anthropic")

    try:
        import anthropic  # type: ignore

        client_kwargs = {"api_key": api_key}
        if backend.config.endpoint:
            client_kwargs["base_url"] = backend.config.endpoint
        import httpx
        timeout_s = _request_timeout_seconds()
        client_kwargs["timeout"] = httpx.Timeout(timeout_s * 3, connect=10.0)  # streaming gets more time
        client = anthropic.Anthropic(**client_kwargs)
        wire_model = backend._wire_model_name()

        # Generator function for streaming with retry on session creation
        def _stream_sdk():
            def _create_stream():
                return client.messages.stream(
                    model=wire_model,
                    max_tokens=backend.config.max_tokens,
                    system=backend.config.system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=backend.config.temperature,
                )

            # Retry stream creation on transient failures
            stream_context = backend._retry(_create_stream)

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
                        backend.last_usage = Usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
                            model=backend.config.model,
                            provider=backend.config.provider
                        )

        # Use helper to run generator in thread
        async for chunk in _run_generator_in_thread(backend, _stream_sdk):
            yield chunk

    except ImportError:
        raise RuntimeError(
            "anthropic package not installed. Install anthropic or choose an OpenAI-compatible model."
        )


# ---------------------------------------------------------------------------
# Google Gemini streaming
# ---------------------------------------------------------------------------

async def _stream_gemini(backend, user_prompt: str):
    """Stream responses from Google Gemini models with retry on session creation."""
    import asyncio as _asyncio
    from .agent_backend_gemini import _gemini_uses_oauth_bearer, _chat_via_gemini_rest

    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY for Gemini provider")
    if _gemini_uses_oauth_bearer(api_key):
        result = await _asyncio.to_thread(_chat_via_gemini_rest, backend, user_prompt, api_key)
        if result:
            yield result
        return

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

        # Generator function for streaming with retry on session creation
        def _stream_sdk():
            def _create_stream():
                return model.generate_content(
                    user_prompt,
                    generation_config=generation_config,
                    stream=True
                )

            # Retry stream creation on transient failures
            response = backend._retry(_create_stream)

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
                    backend.last_usage = Usage(
                        input_tokens=input_tokens or 0,
                        output_tokens=output_tokens or 0,
                        total_tokens=total_tokens,
                        model=backend.config.model,
                        provider=backend.config.provider
                    )

        # Use helper to run generator in thread
        async for chunk in _run_generator_in_thread(backend, _stream_sdk):
            yield chunk

    except ImportError:
        raise RuntimeError(
            "google-generativeai package not installed. Install it or choose an OpenAI-compatible model."
        )


# ---------------------------------------------------------------------------
# Alibaba DashScope (Qwen) streaming
# ---------------------------------------------------------------------------

async def _stream_dashscope(backend, user_prompt: str):
    """Stream responses from Alibaba DashScope (Qwen) models with retry.

    Note: DashScope SDK supports streaming via incremental_output=True parameter.
    """
    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY for Qwen provider")

    try:
        from http import HTTPStatus
        from dashscope import Generation  # type: ignore

        messages = [
            {"role": "system", "content": backend.config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generator function for streaming with retry on session creation
        def _stream_sdk():
            def _create_stream():
                return Generation.call(
                    model=backend.config.model.replace("/", ":"),
                    messages=messages,
                    api_key=api_key,
                    temperature=backend.config.temperature,
                    max_tokens=backend.config.max_tokens,
                    result_format='message',
                    stream=True,
                    incremental_output=True
                )

            # Retry stream creation on transient failures
            responses = backend._retry(_create_stream)

            for response in responses:
                if response.status_code == HTTPStatus.OK:
                    # Capture usage information from streaming response
                    if hasattr(response, 'usage') and response.usage is not None:
                        usage = response.usage
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
        async for chunk in _run_generator_in_thread(backend, _stream_sdk):
            yield chunk

    except ImportError:
        raise RuntimeError(
            "dashscope package not installed. Install it or choose an OpenAI-compatible model."
        )
