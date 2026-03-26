"""Google Gemini adapter helpers for OmicVerseLLMBackend.

Internal module — import from ``omicverse.utils.agent_backend`` instead.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .agent_backend_common import Usage, ToolCall, ChatResponse, _coerce_int, _compute_total

logger = logging.getLogger(__name__)


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
    except Exception:
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


# ---------------------------------------------------------------------------
# Single-turn simple chat
# ---------------------------------------------------------------------------

def _chat_via_gemini(backend, user_prompt: str) -> str:
    """Synchronous single-turn Gemini chat (no tool calling)."""
    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY for Gemini provider")
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

        return backend._retry(_make_call)

    except ImportError:
        raise RuntimeError(
            "google-generativeai package not installed. Install it for tool-calling support."
        )
