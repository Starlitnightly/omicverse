"""
Services Package - Business Logic Layer
========================================
Service modules for OmicVerse web application.
"""

from omicverse.utils.harness import (
    HARNESS_EVENT_TYPES as AGENT_EVENT_TYPES,
    HarnessEvent as AgentEvent,
    HarnessEventType as AgentEventType,
    make_turn_id,
)

from .kernel_service import (
    InProcessKernelExecutor,
    normalize_kernel_id,
    build_kernel_namespace,
    reset_kernel_namespace,
    get_kernel_context,
    get_execution_state,
    request_interrupt,
    execution_state,
    execution_state_lock,
)

from .agent_service import (
    get_agent_instance,
    get_harness_capabilities,
    load_trace,
    run_agent_stream,
    run_agent_chat,
    agent_requires_adata,
    stream_agent_events,
    get_turn_buffer,
    clear_turn_buffer,
    cancel_active_turn,
    get_active_turn_for_session,
    get_pending_approval,
    resolve_pending_approval,
)

from .agent_session_service import (
    session_manager,
    SessionManager,
    AgentSession,
    ChatMessage,
    ApprovalRequest,
)

__all__ = [
    # Kernel service
    'InProcessKernelExecutor',
    'normalize_kernel_id',
    'build_kernel_namespace',
    'reset_kernel_namespace',
    'get_kernel_context',
    'get_execution_state',
    'request_interrupt',
    'execution_state',
    'execution_state_lock',
    # Agent service
    'get_agent_instance',
    'get_harness_capabilities',
    'load_trace',
    'run_agent_stream',
    'run_agent_chat',
    'agent_requires_adata',
    'stream_agent_events',
    'get_turn_buffer',
    'clear_turn_buffer',
    'cancel_active_turn',
    'get_active_turn_for_session',
    'get_pending_approval',
    'resolve_pending_approval',
    'AgentEvent',
    'AgentEventType',
    'AGENT_EVENT_TYPES',
    'make_turn_id',
    # Agent session service
    'session_manager',
    'SessionManager',
    'AgentSession',
    'ChatMessage',
    'ApprovalRequest',
]
