"""
Agent Service - OmicVerse Agent Management
==========================================
Manages OmicVerse AI agent for code generation and chat.
"""

import json
import re
import time
import queue
import asyncio
import threading
import logging
import uuid
from typing import Any, Optional, Generator

from omicverse.utils.harness import STREAM_EVENT_TYPES as AGENT_EVENT_TYPES
from omicverse.utils.harness import make_turn_id
from omicverse.utils.harness.runtime_state import runtime_state
from omicverse.utils.harness.trace_store import RunTraceStore

from .agent_session_service import ApprovalRequest, QuestionRequest, session_manager


logger = logging.getLogger("omicverse_web.agent")

_TRACE_STORE = RunTraceStore()

# ---------------------------------------------------------------------------
# Log sanitization — redact API keys from log output
# ---------------------------------------------------------------------------

# Patterns: sk-..., key-..., bearer tokens, generic long hex/base64 secrets
_KEY_PATTERN = re.compile(
    r'(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]*'       # OpenAI-style sk-...
    r'|(key-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]*'       # key-...
    r'|(Bearer\s+[A-Za-z0-9_-]{8})[A-Za-z0-9_.-]*'  # Bearer tokens
)


def _redact_keys(text: str) -> str:
    """Replace API key patterns with redacted versions."""
    def _sub(m):
        if m.group(1):
            return m.group(1) + '***'
        if m.group(2):
            return m.group(2) + '***'
        if m.group(3):
            return m.group(3) + '***'
        return '***'
    return _KEY_PATTERN.sub(_sub, text)


class _KeyRedactingFilter(logging.Filter):
    """Logging filter that redacts API key patterns from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _redact_keys(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _redact_keys(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    _redact_keys(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        return True


logger.addFilter(_KeyRedactingFilter())

# ---------------------------------------------------------------------------
# Global agent instance cache
# ---------------------------------------------------------------------------
agent_instance = None
agent_config_signature = None


def get_agent_instance(config):
    """Get or create OmicVerse agent instance with caching.

    Args:
        config: Agent configuration dictionary with model, apiKey, apiBase

    Returns:
        OmicVerse Agent instance
    """
    import omicverse as ov
    global agent_instance, agent_config_signature

    if config is None:
        config = {}

    model = config.get('model') or 'gpt-5'
    api_key = config.get('apiKey') or ''
    endpoint = config.get('apiBase') or None

    # Use a hash of the API key for the signature so the full key is never
    # stored in a string that could appear in logs / tracebacks.
    import hashlib
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else ''
    signature = json.dumps({
        'model': model,
        'key_hash': key_hash,
        'endpoint': endpoint,
    }, sort_keys=True)

    # Only recreate agent if configuration changed
    if agent_instance is None or signature != agent_config_signature:
        agent_instance = ov.Agent(
            model=model,
            api_key=api_key or None,
            endpoint=endpoint or None,
            use_notebook_execution=False
        )
        agent_config_signature = signature

    return agent_instance


def get_harness_capabilities() -> dict:
    """Return the web-visible harness capability handshake payload."""
    return {
        "version": 1,
        "event_types": list(AGENT_EVENT_TYPES) + [
            "question_request",
            "question_resolved",
        ],
        "supports": {
            "sse_streaming": True,
            "trace_replay": True,
            "session_history": True,
            "turn_cancel": True,
            "item_lifecycle": True,
            "approval_requests": True,
            "approval_resume": True,
            "question_requests": True,
            "question_resume": True,
            "runtime_tool_state": True,
            "task_tracking": True,
            "plan_mode": True,
            "worktree_state": True,
        },
        "runtime_fields": [
            "loaded_tools",
            "plan_mode",
            "worktree",
            "tasks",
            "pending_questions",
            "pending_approvals",
        ],
        "server_only_validation": True,
        "harness_test_env": "OV_AGENT_RUN_HARNESS_TESTS",
    }


def build_harness_initialize_payload(session_id: str = "") -> dict:
    """Return the initialize handshake payload, including runtime state."""
    session = session_manager.get_session(session_id) if session_id else None
    runtime = session_manager.get_runtime_state(session_id) if session_id else session_manager.get_runtime_state("")
    if session_id:
        core_runtime = runtime_state.get_summary(session_id)
        runtime = {
            **runtime,
            "loaded_tools": list(core_runtime.get("loaded_tools", runtime.get("loaded_tools", []))),
            "plan_mode": "on" if (core_runtime.get("plan_mode") or {}).get("enabled") else runtime.get("plan_mode", "off"),
            "worktree": core_runtime.get("worktree") or runtime.get("worktree", {}),
            "working_directory": core_runtime.get("working_directory", runtime.get("working_directory", "")),
        }
    return {
        "capabilities": get_harness_capabilities(),
        "session": session.to_summary() if session is not None else None,
        "runtime": runtime,
    }


def load_trace(trace_id: str) -> Optional[dict]:
    """Load a persisted harness trace by id."""
    try:
        return _TRACE_STORE.load(trace_id)
    except Exception:
        return None


def run_agent_stream(agent, prompt, adata, *, session_id: str = ""):
    """Run agent with streaming support for code generation.

    Args:
        agent: OmicVerse Agent instance
        prompt: User prompt for code generation
        adata: AnnData object for analysis
        session_id: Optional session identifier for structured logging

    Returns:
        Dictionary with code, llm_text, result_adata, result_shape
    """
    turn_id = make_turn_id()
    t0 = time.time()
    logger.info("agent_stream_start", extra={
        "turn_id": turn_id, "session_id": session_id,
        "prompt_len": len(prompt),
    })

    async def _runner():
        code = None
        result_adata = None
        result_shape = None
        llm_text = ''
        trace_id = ''
        tool_calls_seen = []
        async for event in agent.stream_async(prompt, adata):
            etype = event.get('type')
            trace_id = trace_id or event.get('trace_id', '')
            if etype == 'llm_chunk':
                llm_text += event.get('content', '')
            elif etype == 'tool_call':
                tool_calls_seen.append(event.get('content', {}).get('name'))
            elif etype == 'code':
                code = event.get('content')
            elif etype == 'result':
                result_adata = event.get('content')
                result_shape = event.get('shape')
            elif etype == 'error':
                elapsed = time.time() - t0
                logger.error("agent_stream_error", extra={
                    "turn_id": turn_id, "session_id": session_id,
                    "latency_s": round(elapsed, 2),
                    "error": event.get('content', 'Agent error'),
                })
                raise RuntimeError(event.get('content', 'Agent error'))
        elapsed = time.time() - t0
        logger.info("agent_stream_done", extra={
            "turn_id": turn_id, "session_id": session_id,
            "latency_s": round(elapsed, 2),
            "tool_calls": tool_calls_seen,
            "has_code": code is not None,
            "data_updated": result_adata is not None,
        })
        return {
            'code': code,
            'llm_text': llm_text,
            'result_adata': result_adata,
            'result_shape': result_shape,
            'trace_id': trace_id,
        }

    def _log_and_raise(exc):
        elapsed = time.time() - t0
        logger.error("agent_stream_error", extra={
            "turn_id": turn_id, "session_id": session_id,
            "latency_s": round(elapsed, 2),
            "error": str(exc),
        })
        raise exc

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop and loop.is_running():
            result_container = {}
            error_container = {}

            def _run_in_thread():
                try:
                    result_container['value'] = asyncio.run(_runner())
                except BaseException as exc:
                    error_container['error'] = exc

            thread = threading.Thread(target=_run_in_thread, name='OmicVerseAgentRunner')
            thread.start()
            thread.join()
            if 'error' in error_container:
                raise error_container['error']
            return result_container.get('value')

        return asyncio.run(_runner())
    except RuntimeError:
        raise  # already logged inside _runner for event-level errors
    except Exception as exc:
        _log_and_raise(exc)


def run_agent_chat(agent, prompt, *, session_id: str = ""):
    """Run agent in chat mode for natural language responses.

    Args:
        agent: OmicVerse Agent instance
        prompt: User prompt for chat
        session_id: Optional session identifier for structured logging

    Returns:
        String response from agent
    """
    turn_id = make_turn_id()
    t0 = time.time()
    logger.info("agent_chat_start", extra={
        "turn_id": turn_id, "session_id": session_id,
        "prompt_len": len(prompt),
    })

    async def _runner():
        if not agent._llm:
            raise RuntimeError("LLM backend is not initialized")
        chat_prompt = (
            "You are an OmicVerse assistant. Answer in natural language only, "
            "avoid code unless explicitly requested.\n\nUser: " + prompt
        )
        result = await agent._llm.run(chat_prompt)
        elapsed = time.time() - t0
        logger.info("agent_chat_done", extra={
            "turn_id": turn_id, "session_id": session_id,
            "latency_s": round(elapsed, 2),
            "reply_len": len(result) if result else 0,
        })
        return result

    def _log_and_raise(exc):
        elapsed = time.time() - t0
        logger.error("agent_chat_error", extra={
            "turn_id": turn_id, "session_id": session_id,
            "latency_s": round(elapsed, 2),
            "error": str(exc),
        })
        raise exc

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop and loop.is_running():
            result_container = {}
            error_container = {}

            def _run_in_thread():
                try:
                    result_container['value'] = asyncio.run(_runner())
                except BaseException as exc:
                    error_container['error'] = exc

            thread = threading.Thread(target=_run_in_thread, name='OmicVerseAgentChatRunner')
            thread.start()
            thread.join()
            if 'error' in error_container:
                raise error_container['error']
            return result_container.get('value')

        return asyncio.run(_runner())
    except RuntimeError:
        raise  # already logged inside _runner
    except Exception as exc:
        _log_and_raise(exc)


def agent_requires_adata(prompt):
    """Check if prompt requires adata based on keywords.

    Args:
        prompt: User prompt to analyze

    Returns:
        Boolean indicating if adata is likely required
    """
    if not prompt:
        return False
    lowered = prompt.lower()
    keywords = [
        'adata', 'qc', 'quality', 'cluster', 'clustering', 'umap', 'tsne', 'pca',
        'embedding', 'neighbors', 'leiden', 'louvain', 'marker', 'differential',
        'hvg', 'highly variable', 'preprocess', 'normalize', 'visualize', 'plot',
        '降维', '聚类', '可视化', '差异', '标记', '质控', '预处理'
    ]
    return any(keyword in lowered for keyword in keywords)


# ---------------------------------------------------------------------------
# Turn event buffer (server-side, for reconnection support)
# ---------------------------------------------------------------------------

# Keyed by turn_id → list[dict].  Kept in memory; cleared on done/error.
_turn_buffers: dict[str, list[dict]] = {}
_TURN_BUFFER_MAX = 50  # max buffered turns to prevent unbounded growth


def _buffer_event(turn_id: str, event_dict: dict) -> None:
    """Append an event to the per-turn buffer."""
    if turn_id not in _turn_buffers:
        # Evict oldest if at capacity
        if len(_turn_buffers) >= _TURN_BUFFER_MAX:
            oldest = next(iter(_turn_buffers))
            del _turn_buffers[oldest]
        _turn_buffers[turn_id] = []
    _turn_buffers[turn_id].append(event_dict)


def get_turn_buffer(turn_id: str) -> list[dict]:
    """Return buffered events for a turn (empty list if none)."""
    return list(_turn_buffers.get(turn_id, []))


def clear_turn_buffer(turn_id: str) -> None:
    """Remove a turn's buffer (call after client confirms receipt)."""
    _turn_buffers.pop(turn_id, None)


# ---------------------------------------------------------------------------
# Pending approval broker (server-side pause/resume for execute_code)
# ---------------------------------------------------------------------------

_APPROVAL_TIMEOUT = 300.0
_pending_approvals: dict[str, dict] = {}
_pending_approvals_lock = threading.Lock()
_pending_questions: dict[str, dict] = {}
_pending_questions_lock = threading.Lock()


def resolve_pending_approval(request_id: str, decision: bool) -> bool:
    """Resolve a pending approval request."""
    with _pending_approvals_lock:
        entry = _pending_approvals.get(request_id)
        if entry is None:
            return False
        entry["decision"] = bool(decision)
        entry["resolved"].set()
        return True


def resolve_pending_question(request_id: str, answer: str) -> bool:
    """Resolve a pending question request."""
    with _pending_questions_lock:
        entry = _pending_questions.get(request_id)
        if entry is None:
            return False
        entry["answer"] = answer
        entry["resolved"].set()
        _pending_questions.pop(request_id, None)
        return True


def get_pending_approval(request_id: str) -> Optional[dict]:
    """Return a serializable view of a pending approval request."""
    with _pending_approvals_lock:
        entry = _pending_approvals.get(request_id)
        if entry is None:
            return None
        payload = dict(entry["payload"])
        payload["request_id"] = request_id
        payload["turn_id"] = entry["turn_id"]
        payload["session_id"] = entry["session_id"]
        payload["created_at"] = entry["created_at"]
        return payload


def get_pending_question(request_id: str) -> Optional[dict]:
    """Return a serializable view of a pending question request."""
    with _pending_questions_lock:
        entry = _pending_questions.get(request_id)
        if entry is None:
            return None
        payload = dict(entry["payload"])
        payload["request_id"] = request_id
        payload["turn_id"] = entry["turn_id"]
        payload["session_id"] = entry["session_id"]
        payload["created_at"] = entry["created_at"]
        return payload


# ---------------------------------------------------------------------------
# Streaming generator for SSE (Phase 1)
# ---------------------------------------------------------------------------

_HEARTBEAT_TIMEOUT = 5.0   # seconds between heartbeats
_SENTINEL = None            # marks end of stream


def _build_data_info(adata_obj: Any) -> Optional[dict]:
    """Extract sidebar-friendly metadata from an AnnData/MuData object."""
    if adata_obj is None or not hasattr(adata_obj, 'n_obs'):
        return None
    try:
        embeddings = [
            k.replace('X_', '') for k in adata_obj.obsm.keys()
        ] if hasattr(adata_obj, 'obsm') else []
        return {
            'n_cells': int(adata_obj.n_obs),
            'n_genes': int(adata_obj.n_vars),
            'embeddings': embeddings,
            'obs_columns': list(adata_obj.obs.columns) if hasattr(adata_obj.obs, 'columns') else [],
            'var_columns': list(adata_obj.var.columns) if hasattr(adata_obj.var, 'columns') else [],
        }
    except Exception:
        return None


def _serialize_sse(event_dict: dict) -> str:
    """Format a dict as an SSE ``data:`` line."""
    return f"data: {json.dumps(event_dict, default=str)}\n\n"


# ---------------------------------------------------------------------------
# Active handle tracking (for server-side cancel endpoint)
# ---------------------------------------------------------------------------

_active_handles: dict[str, "AgentStreamHandle"] = {}
_active_handles_lock = threading.Lock()


def _register_handle(handle: "AgentStreamHandle") -> None:
    with _active_handles_lock:
        _active_handles[handle.turn_id] = handle


def _unregister_handle(turn_id: str) -> None:
    with _active_handles_lock:
        _active_handles.pop(turn_id, None)


def cancel_active_turn(turn_id: str) -> bool:
    """Cancel a running turn by its turn_id.

    Returns True if a cancel signal was sent.
    """
    with _active_handles_lock:
        handle = _active_handles.get(turn_id)
    if handle is None:
        return False
    handle.cancel()
    return True


def get_active_turn_for_session(session_id: str) -> Optional[str]:
    """Return the turn_id of the active turn for a session, or None."""
    with _active_handles_lock:
        for tid, h in _active_handles.items():
            if h.session_id == session_id:
                return tid
    return None


class AgentStreamHandle:
    """Holds a per-request SSE generator and its result context.

    Usage in the route::

        handle = stream_agent_events(agent, prompt, adata, ...)
        resp = Response(stream_with_context(handle), ...)
        # after streaming finishes:
        ctx = handle.ctx   # {"result_adata": ..., "data_updated": ..., ...}

    Cancellation:
    - **Client disconnect**: Flask calls ``close()``, which sets ``cancelled``
      so that the ``on_complete`` callback can skip adata commits.
    - **Server-side cancel**: the ``/api/agent/chat/cancel`` route calls
      ``cancel()``, which sets both ``cancelled`` (skip commit) and
      ``agent_cancel`` (stops the agentic loop at the next checkpoint).
    """

    def __init__(self, gen: Generator, ctx: dict, turn_id: str,
                 session_id: str = ""):
        self._gen = gen
        self.ctx = ctx
        self.turn_id = turn_id
        self.session_id = session_id
        self.cancelled = threading.Event()
        self.agent_cancel = threading.Event()  # forwarded to _run_agentic_loop

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._gen)
        except StopIteration:
            _unregister_handle(self.turn_id)
            raise

    def cancel(self):
        """Server-side cancel: stop the agent loop AND skip on_complete."""
        self.cancelled.set()
        self.agent_cancel.set()

    def close(self):
        """Called by Flask when the client disconnects."""
        self.cancelled.set()
        self.agent_cancel.set()
        _unregister_handle(self.turn_id)
        if hasattr(self._gen, 'close'):
            self._gen.close()


def stream_agent_events(
    agent,
    prompt: str,
    adata: Any,
    *,
    session_id: str = "",
    history: Optional[list] = None,
    on_complete: Optional[callable] = None,
    on_finally: Optional[callable] = None,
) -> AgentStreamHandle:
    """Create an SSE stream handle for a single agent turn.

    This is the bridge between the async ``agent.stream_async()`` generator
    and Flask's synchronous ``stream_with_context`` response.

    Architecture:
        1. A background thread runs a new asyncio event loop that consumes
           ``agent.stream_async()``.
        2. Each raw event is normalized, **buffered in the producer**, and put
           onto a ``queue.Queue``.
        3. The main (Flask) thread reads from the queue and yields SSE lines.
        4. Heartbeats are emitted on queue timeout to keep the connection alive.
        5. On producer completion, ``on_complete(ctx)`` is called (if provided)
           **in the producer thread**, so that state commits happen even when
           the SSE client disconnects early.

    Parameters
    ----------
    agent : OmicVerseAgent
        Initialized agent instance.
    prompt : str
        User prompt.
    adata : Any
        AnnData / MuData object (or None for chat-only mode).
    session_id : str
        Session identifier for logging / reconnection.
    on_complete : callable, optional
        ``on_complete(ctx)`` is called in the producer thread after the agent
        finishes **only when the stream was not cancelled**.  ``ctx`` is the
        shared dict with ``result_adata``, ``data_updated``, ``error``.
        The route uses this to commit adata independently of client
        consumption.
    on_finally : callable, optional
        ``on_finally(ctx)`` is called in the producer thread **unconditionally**
        after the agent finishes (whether completed, errored, or cancelled).
        Use this for cleanup that must always run (e.g. clearing active-turn
        tracking on the session).

    Returns
    -------
    AgentStreamHandle
        Iterable of SSE ``data: {...}\\n\\n`` lines.  After exhaustion,
        ``handle.ctx`` contains ``result_adata``, ``data_updated``, ``error``.
    """
    turn_id = make_turn_id()
    t0 = time.time()
    q: queue.Queue = queue.Queue()
    cancelled = threading.Event()      # set when SSE client disconnects
    agent_cancel = threading.Event()   # forwarded to _run_agentic_loop

    # Shared mutable state between threads — returned to caller via handle.ctx
    ctx: dict = {
        "result_adata": None,
        "result_shape": None,
        "data_updated": False,
        "data_info": None,
        "error": None,
        "llm_text": "",   # accumulated LLM text for history persistence
        "summary": "",    # final summary from done event
        "trace_id": "",
        "approval_ids": [],
    }

    logger.info("agent_stream_sse_start", extra={
        "turn_id": turn_id, "session_id": session_id,
        "prompt_len": len(prompt), "has_adata": adata is not None,
    })

    def _tag_and_buffer(payload: dict) -> dict:
        """Tag event with turn metadata and buffer it (producer-side)."""
        payload["turn_id"] = turn_id
        payload["session_id"] = session_id
        payload["ts"] = time.time()
        _buffer_event(turn_id, payload)
        return payload

    def _approval_handler(payload: dict):
        interaction_kind = (payload.get("kind") or "approval").strip().lower()
        if interaction_kind == "question":
            question_id = payload.get("question_id") or payload.get("request_id") or ("question_" + uuid.uuid4().hex[:12])
            entry = {
                "turn_id": turn_id,
                "session_id": session_id,
                "payload": dict(payload),
                "answer": "",
                "resolved": threading.Event(),
                "created_at": time.time(),
            }
            with _pending_questions_lock:
                _pending_questions[question_id] = entry

            question = QuestionRequest(
                question_id=question_id,
                turn_id=turn_id,
                session_id=session_id,
                title=payload.get("header", payload.get("title", "Question for user")),
                message=payload.get("question", payload.get("message", "")),
                placeholder=payload.get("placeholder", ""),
                options=list(payload.get("options", [])),
                metadata={
                    "trace_id": payload.get("trace_id", ""),
                    "tool_name": payload.get("tool_name", "AskUserQuestion"),
                },
            )
            session_manager.register_question(session_id, question)

            q.put(_tag_and_buffer({
                "type": "question_request",
                "content": {
                    "request_id": question_id,
                    "question_id": question_id,
                    "title": question.title,
                    "message": question.message,
                    "placeholder": question.placeholder,
                    "options": question.options,
                },
                "trace_id": payload.get("trace_id", ""),
                "category": "question",
            }))

            answer = ""
            if entry["resolved"].wait(_APPROVAL_TIMEOUT):
                answer = str(entry["answer"] or "")

            with _pending_questions_lock:
                _pending_questions.pop(question_id, None)

            q.put(_tag_and_buffer({
                "type": "question_resolved",
                "content": {
                    "request_id": question_id,
                    "question_id": question_id,
                    "answer": answer,
                },
                "trace_id": payload.get("trace_id", ""),
                "category": "question",
            }))
            return {"answer": answer}

        request_id = payload.get("request_id") or make_turn_id()
        entry = {
            "turn_id": turn_id,
            "session_id": session_id,
            "payload": dict(payload),
            "decision": False,
            "resolved": threading.Event(),
            "created_at": time.time(),
        }
        with _pending_approvals_lock:
            _pending_approvals[request_id] = entry

        approval = ApprovalRequest(
            approval_id=request_id,
            turn_id=turn_id,
            session_id=session_id,
            title=payload.get("title", "Execution approval required"),
            message=payload.get("message", ""),
            code=payload.get("code", ""),
            violations=list(payload.get("violations", [])),
            metadata={
                "trace_id": payload.get("trace_id", ""),
                "approval_mode": payload.get("approval_mode", "never"),
            },
        )
        session_manager.register_approval(session_id, approval)

        approval_event = _tag_and_buffer({
            "type": "approval_request",
            "content": {
                "request_id": request_id,
                "approval_id": request_id,
                "code": payload.get("code", ""),
                "violations": payload.get("violations", []),
                "approval_mode": payload.get("approval_mode", "never"),
                "title": payload.get("title", "Execution approval required"),
                "message": payload.get("message", ""),
            },
            "trace_id": payload.get("trace_id", ""),
            "category": "approval",
        })
        q.put(approval_event)

        approved = False
        if entry["resolved"].wait(_APPROVAL_TIMEOUT):
            approved = bool(entry["decision"])

        with _pending_approvals_lock:
            _pending_approvals.pop(request_id, None)

        q.put(_tag_and_buffer({
            "type": "approval_resolved",
            "content": {
                "request_id": request_id,
                "approval_id": request_id,
                "decision": "approved" if approved else "denied",
            },
            "trace_id": payload.get("trace_id", ""),
            "category": "approval",
        }))
        return approved

    def _register_question_event(content: dict, trace_id: str = "", step_id: str = "") -> dict:
        question_id = (content or {}).get("question_id") or (content or {}).get("request_id") or ("question_" + uuid.uuid4().hex[:12])
        entry = {
            "turn_id": turn_id,
            "session_id": session_id,
            "payload": dict(content or {}),
            "answer": "",
            "resolved": threading.Event(),
            "created_at": time.time(),
        }
        with _pending_questions_lock:
            _pending_questions[question_id] = entry

        question = QuestionRequest(
            question_id=question_id,
            turn_id=turn_id,
            session_id=session_id,
            title=(content or {}).get("title", "Question for user"),
            message=(content or {}).get("message", ""),
            placeholder=(content or {}).get("placeholder", ""),
            options=list((content or {}).get("options", [])),
            metadata=dict((content or {}).get("metadata", {})),
        )
        session_manager.register_question(session_id, question)

        return {
            "type": "question_request",
            "content": {
                **(content or {}),
                "request_id": question_id,
                "question_id": question_id,
            },
            "trace_id": trace_id,
            "step_id": step_id,
            "category": "question",
        }

    # --- background thread: async consumer → queue producer ----------------

    def _run_async_loop():
        async def _consume():
            try:
                setattr(agent, "_web_session_id", session_id)
                # Always use the full agentic loop — even without adata,
                # the agent still has web_fetch, web_search, web_download,
                # search_functions, search_skills, and conversation history.
                async for event in agent.stream_async(
                        prompt, adata,
                        cancel_event=agent_cancel,
                        history=history,
                        approval_handler=_approval_handler,
                    ):
                        etype = event.get("type")
                        content = event.get("content")
                        ctx["trace_id"] = ctx["trace_id"] or event.get("trace_id", "")

                        # Build normalized payload
                        payload: dict = {"type": etype}
                        for key in ("trace_id", "step_id", "category", "latency_ms", "artifact_refs", "metadata"):
                            if key in event:
                                payload[key] = event[key]

                        if etype == "result":
                            ctx["result_adata"] = content
                            ctx["result_shape"] = event.get("shape")
                            ctx["data_updated"] = True
                            # Include data_info so the frontend can
                            # refresh sidebar without a separate fetch.
                            data_info = _build_data_info(content)
                            ctx["data_info"] = data_info
                            payload["content"] = {
                                "shape": event.get("shape"),
                                "data_info": data_info,
                            }
                        elif etype == "tool_call":
                            # Truncate large argument values for SSE safety
                            tc = content or {}
                            args = tc.get("arguments", {})
                            safe_args = {}
                            for k, v in args.items():
                                sv = str(v)
                                safe_args[k] = sv if len(sv) <= 500 else sv[:500] + "…"
                            payload["content"] = {
                                "name": tc.get("name"),
                                "arguments": safe_args,
                            }
                        elif etype == "usage":
                            # Normalize usage to plain dict
                            if hasattr(content, "__dict__"):
                                payload["content"] = {
                                    k: v for k, v in content.__dict__.items()
                                    if not k.startswith("_")
                                }
                            else:
                                payload["content"] = content
                        elif etype == "llm_chunk":
                            ctx["llm_text"] += (content or "")
                            payload["content"] = content
                        elif etype == "done":
                            ctx["summary"] = content or ""
                            payload["content"] = content
                        elif etype == "error":
                            ctx["error"] = content
                            payload["content"] = content
                        elif etype in ("approval_request", "approval_required"):
                            approval_id = (content or {}).get("request_id") or ("approval_" + uuid.uuid4().hex[:12])
                            approval = ApprovalRequest(
                                approval_id=approval_id,
                                turn_id=turn_id,
                                session_id=session_id,
                                title=(content or {}).get("title", "Execution approval required"),
                                message=(content or {}).get("message", ""),
                                code=(content or {}).get("code", ""),
                                violations=list((content or {}).get("violations", [])),
                                metadata=dict((content or {}).get("metadata", {})),
                            )
                            session_manager.register_approval(session_id, approval)
                            ctx["approval_ids"].append(approval_id)
                            payload["content"] = {
                                **(content or {}),
                                "approval_id": approval_id,
                            }
                        elif etype == "question_request":
                            payload = _register_question_event(content or {}, event.get("trace_id", ""), event.get("step_id", ""))
                        elif etype == "question_resolved":
                            question_id = (content or {}).get("question_id") or (content or {}).get("request_id") or ""
                            answer = str((content or {}).get("answer", ""))
                            if question_id:
                                session_manager.resolve_question(session_id, question_id, answer)
                                with _pending_questions_lock:
                                    _pending_questions.pop(question_id, None)
                            payload["content"] = {
                                **(content or {}),
                                "question_id": question_id,
                                "request_id": question_id,
                            }
                        else:
                            payload["content"] = content

                        session_manager.apply_runtime_event(session_id, {
                            **payload,
                            "turn_id": turn_id,
                            "session_id": session_id,
                        })
                        q.put(_tag_and_buffer(payload))

            except Exception as exc:
                ctx["error"] = str(exc)
                q.put(_tag_and_buffer({"type": "error", "content": str(exc)}))
            finally:
                try:
                    setattr(agent, "_web_session_id", "")
                except Exception:
                    pass
                # Buffer stream_end in the producer so reconnection always
                # sees the complete turn, even if the SSE client disconnected.
                elapsed = time.time() - t0
                stream_end = {
                    "type": "stream_end",
                    "latency_s": round(elapsed, 2),
                    "data_updated": ctx["data_updated"],
                    "error": ctx["error"],
                }
                _tag_and_buffer(stream_end)

                logger.info("agent_stream_sse_done", extra={
                    "turn_id": turn_id, "session_id": session_id,
                    "latency_s": round(elapsed, 2),
                    "data_updated": ctx["data_updated"],
                    "error": ctx["error"],
                })

                # Commit adata only if the client is still connected.
                # When the user clicks Stop or navigates away, Flask
                # closes the generator which sets ``cancelled``.
                if on_complete is not None and not cancelled.is_set():
                    try:
                        on_complete(ctx)
                    except Exception:
                        logger.exception("on_complete callback failed",
                                         extra={"turn_id": turn_id})
                elif cancelled.is_set():
                    logger.info("agent_stream_sse_cancelled", extra={
                        "turn_id": turn_id, "session_id": session_id,
                    })

                # Always run on_finally (cleanup that must happen regardless
                # of completion/cancel state — e.g. clearing active turn).
                if on_finally is not None:
                    try:
                        on_finally(ctx)
                    except Exception:
                        logger.exception("on_finally callback failed",
                                         extra={"turn_id": turn_id})

                q.put(stream_end)
                q.put(_SENTINEL)
                _unregister_handle(turn_id)

        asyncio.run(_consume())

    bg = threading.Thread(target=_run_async_loop, name="AgentSSEProducer", daemon=True)
    bg.start()

    # --- inner generator: queue consumer → SSE yield ----------------------
    # Events are already tagged and buffered in the producer; the consumer
    # only serializes and yields them over the HTTP connection.

    def _generate():
        # Emit opening event with turn metadata
        opening = _tag_and_buffer({
            "type": "status",
            "content": "started",
        })
        yield _serialize_sse(opening)

        while True:
            try:
                event = q.get(timeout=_HEARTBEAT_TIMEOUT)
            except queue.Empty:
                # Keep connection alive
                yield _serialize_sse({"type": "heartbeat"})
                continue

            if event is _SENTINEL:
                break

            yield _serialize_sse(event)

        bg.join(timeout=2.0)

    handle = AgentStreamHandle(_generate(), ctx, turn_id, session_id=session_id)
    handle.cancelled = cancelled
    handle.agent_cancel = agent_cancel
    _register_handle(handle)
    return handle
