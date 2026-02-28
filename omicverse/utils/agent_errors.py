"""
Structured error hierarchy for OmicVerse Agent (P0-3).

Replaces bare RuntimeError / ValueError with typed exceptions that carry
context (provider, retryable, status_code) and separate control-flow
signals (WorkflowNeedsFallback) from real bugs.
"""

from __future__ import annotations


class OVAgentError(Exception):
    """Base class for all OV Agent errors."""
    pass


class WorkflowNeedsFallback(OVAgentError):
    """Deprecated: kept for backward compatibility only.

    Previously used as a signal for the legacy Priority 1/2 fallback system,
    which has been removed. The agentic tool-calling loop is now the only
    execution architecture.
    """
    pass


class ProviderError(OVAgentError):
    """Error communicating with an LLM provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int = 0,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class ConfigError(OVAgentError):
    """Configuration error (missing API key, unsupported model, bad params)."""
    pass


class ExecutionError(OVAgentError):
    """Error during generated-code execution."""
    pass


class SandboxDeniedError(ExecutionError):
    """Notebook / sandbox execution failed or was denied."""
    pass


class SecurityViolationError(SandboxDeniedError):
    """Pre-execution security scan detected dangerous code patterns.

    Attributes
    ----------
    violations : list
        List of ``SecurityViolation`` objects describing each finding.
    """

    def __init__(self, message: str, *, violations: list | None = None) -> None:
        super().__init__(message)
        self.violations = violations or []
