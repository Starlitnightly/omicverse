from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol

from ..agent_bridge import AgentRunResult, AgentBridge


ProgressCallback = Callable[[str], Awaitable[None]]
ChunkCallback = Callable[[str], Awaitable[None]]


@dataclass
class ExecutionCallbacks:
    progress_cb: Optional[ProgressCallback] = None
    llm_chunk_cb: Optional[ChunkCallback] = None


class ExecutionAdapter(Protocol):
    async def run(
        self,
        session: Any,
        request: str,
        *,
        adata: Optional[Any],
        callbacks: ExecutionCallbacks,
        history: Optional[list] = None,
        request_content: Optional[list] = None,
    ) -> AgentRunResult:
        ...


class AgentBridgeExecutionAdapter:
    """Execution shim that preserves the current AgentBridge behavior."""

    async def run(
        self,
        session: Any,
        request: str,
        *,
        adata: Optional[Any],
        callbacks: ExecutionCallbacks,
        history: Optional[list] = None,
        request_content: Optional[list] = None,
    ) -> AgentRunResult:
        bridge = AgentBridge(
            session.agent,
            callbacks.progress_cb,
            callbacks.llm_chunk_cb,
        )
        return await bridge.run(
            request,
            adata,
            history=history or [],
            request_content=request_content or [],
        )
