"""Tests for task-009: backend common hygiene fixes.

Covers:
1. Thread-safe shared executor creation (AC-001 item 1)
2. _retry_with_backoff keyword-only signature (AC-001 item 2)
3. WorkflowNeedsFallback compatibility alias (AC-001 item 3)
4. No provider-specific behavior changes (AC-001 item 5)
"""

from __future__ import annotations

import concurrent.futures
import inspect
import threading
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# 1. Thread-safe shared executor singleton
# ===================================================================


class TestExecutorThreadSafety:
    """Executor creation must be protected against races."""

    def _reset_executor_state(self):
        import omicverse.utils.agent_backend_common as mod
        old_exc = mod._SHARED_EXECUTOR
        old_flag = mod._EXECUTOR_ATEXIT_REGISTERED
        mod._SHARED_EXECUTOR = None
        mod._EXECUTOR_ATEXIT_REGISTERED = False
        return mod, old_exc, old_flag

    def _restore_executor_state(self, mod, old_exc, old_flag):
        mod._SHARED_EXECUTOR = old_exc
        mod._EXECUTOR_ATEXIT_REGISTERED = old_flag

    def test_uses_threading_lock(self):
        """The module must define a threading.Lock for executor creation."""
        import omicverse.utils.agent_backend_common as mod
        assert hasattr(mod, "_EXECUTOR_LOCK")
        assert isinstance(mod._EXECUTOR_LOCK, type(threading.Lock()))

    def test_concurrent_calls_return_same_instance(self):
        """Multiple threads calling _get_shared_executor get the same object."""
        mod, old_exc, old_flag = self._reset_executor_state()
        try:
            results = []
            barrier = threading.Barrier(8)

            def worker():
                barrier.wait()
                results.append(mod._get_shared_executor())

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            assert len(results) == 8
            assert all(r is results[0] for r in results)
        finally:
            self._restore_executor_state(mod, old_exc, old_flag)

    def test_only_one_executor_constructed(self):
        """Even under contention, only one ThreadPoolExecutor is created."""
        mod, old_exc, old_flag = self._reset_executor_state()
        try:
            construction_count = 0
            original_init = concurrent.futures.ThreadPoolExecutor.__init__

            def counting_init(self_exc, *a, **kw):
                nonlocal construction_count
                construction_count += 1
                original_init(self_exc, *a, **kw)

            barrier = threading.Barrier(8)

            def worker():
                barrier.wait()
                mod._get_shared_executor()

            with patch.object(
                concurrent.futures.ThreadPoolExecutor, "__init__", counting_init
            ):
                threads = [threading.Thread(target=worker) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=5)

            assert construction_count == 1, (
                f"Expected 1 executor construction, got {construction_count}"
            )
        finally:
            self._restore_executor_state(mod, old_exc, old_flag)

    def test_lock_in_source_code(self):
        """Source code must acquire _EXECUTOR_LOCK inside _get_shared_executor."""
        import omicverse.utils.agent_backend_common as mod
        source = inspect.getsource(mod._get_shared_executor)
        assert "_EXECUTOR_LOCK" in source


# ===================================================================
# 2. _retry_with_backoff keyword-friendly signature
# ===================================================================


class TestRetrySignature:
    """_retry_with_backoff accepts a zero-arg callable; config is positional-or-keyword."""

    def test_no_varargs_in_signature(self):
        """The helper must NOT accept *args/**kwargs (prevents silent forwarding)."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff
        sig = inspect.signature(_retry_with_backoff)
        kinds = {p.kind for p in sig.parameters.values()}
        assert inspect.Parameter.VAR_POSITIONAL not in kinds, (
            "_retry_with_backoff must not accept *args"
        )
        assert inspect.Parameter.VAR_KEYWORD not in kinds, (
            "_retry_with_backoff must not accept **kwargs"
        )

    def test_config_params_accept_keyword(self):
        """max_attempts, base_delay, factor, jitter are accepted as keywords."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff
        sig = inspect.signature(_retry_with_backoff)
        for name in ("max_attempts", "base_delay", "factor", "jitter"):
            param = sig.parameters[name]
            assert param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ), f"{name} should be callable by keyword, got {param.kind.name}"

    def test_legacy_positional_config_works(self):
        """Positional config (old calling convention) is interpreted as config, not forwarded."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1
            return "ok"

        # Legacy style: _retry_with_backoff(func, max_attempts)
        result = _retry_with_backoff(counter, 1)
        assert result == "ok"
        assert call_count == 1  # exactly 1 attempt, not forwarded as arg

    def test_func_called_with_zero_args(self):
        """func is called as func() — callers must bind args before passing."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff

        def zero_arg():
            return "zero"

        result = _retry_with_backoff(zero_arg, max_attempts=1)
        assert result == "zero"

    def test_bound_lambda_pattern(self):
        """Standard pattern: wrap func+args in a lambda before passing."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff

        def add(a, b):
            return a + b

        result = _retry_with_backoff(lambda: add(3, 4), max_attempts=1)
        assert result == 7

    def test_retry_still_retries(self):
        """Basic retry behavior is preserved."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("transient")
            return "ok"

        result = _retry_with_backoff(
            flaky, max_attempts=3, base_delay=0.0, jitter=0.0
        )
        assert result == "ok"
        assert call_count == 3

    def test_retry_exhaustion_raises(self):
        """All attempts exhausted raises RuntimeError."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff

        def always_fail():
            raise TimeoutError("always")

        with pytest.raises(RuntimeError, match="All 2 attempts failed"):
            _retry_with_backoff(
                always_fail, max_attempts=2, base_delay=0.0, jitter=0.0
            )

    def test_backend_retry_method_forwards_args(self, monkeypatch):
        """OmicVerseLLMBackend._retry forwards positional/keyword args to func."""
        from omicverse.utils.agent_backend import OmicVerseLLMBackend
        from omicverse.utils.model_config import ModelConfig

        monkeypatch.setattr(
            ModelConfig, "get_provider_from_model", lambda *a, **kw: "openai"
        )
        backend = OmicVerseLLMBackend(
            system_prompt="test", model="gpt-4o", api_key="k"
        )

        def add(a, b):
            return a + b

        result = backend._retry(add, 3, 4)
        assert result == 7

    def test_backend_retry_method_forwards_kwargs(self, monkeypatch):
        """OmicVerseLLMBackend._retry forwards keyword args to func."""
        from omicverse.utils.agent_backend import OmicVerseLLMBackend
        from omicverse.utils.model_config import ModelConfig

        monkeypatch.setattr(
            ModelConfig, "get_provider_from_model", lambda *a, **kw: "openai"
        )
        backend = OmicVerseLLMBackend(
            system_prompt="test", model="gpt-4o", api_key="k"
        )

        def greet(name="world"):
            return f"hello {name}"

        result = backend._retry(greet, name="claude")
        assert result == "hello claude"

    def test_backend_retry_zero_arg_func(self, monkeypatch):
        """OmicVerseLLMBackend._retry works with zero-arg callables."""
        from omicverse.utils.agent_backend import OmicVerseLLMBackend
        from omicverse.utils.model_config import ModelConfig

        monkeypatch.setattr(
            ModelConfig, "get_provider_from_model", lambda *a, **kw: "openai"
        )
        backend = OmicVerseLLMBackend(
            system_prompt="test", model="gpt-4o", api_key="k"
        )

        result = backend._retry(lambda: 42)
        assert result == 42


# ===================================================================
# 3. WorkflowNeedsFallback compatibility alias
# ===================================================================


class TestWorkflowNeedsFallbackCompat:
    """WorkflowNeedsFallback must remain importable and well-documented."""

    def test_importable_from_agent_errors(self):
        from omicverse.utils.agent_errors import WorkflowNeedsFallback
        assert WorkflowNeedsFallback is not None

    def test_reexported_in_utils_init_source(self):
        """The utils __init__.py re-exports WorkflowNeedsFallback."""
        from pathlib import Path
        source = (
            Path(__file__).resolve().parents[2]
            / "omicverse" / "utils" / "__init__.py"
        ).read_text()
        assert "WorkflowNeedsFallback" in source

    def test_is_subclass_of_ovagent_error(self):
        from omicverse.utils.agent_errors import OVAgentError, WorkflowNeedsFallback
        assert issubclass(WorkflowNeedsFallback, OVAgentError)

    def test_is_subclass_of_exception(self):
        from omicverse.utils.agent_errors import WorkflowNeedsFallback
        assert issubclass(WorkflowNeedsFallback, Exception)

    def test_can_be_raised_and_caught(self):
        from omicverse.utils.agent_errors import WorkflowNeedsFallback
        with pytest.raises(WorkflowNeedsFallback):
            raise WorkflowNeedsFallback("fallback needed")

    def test_catchable_as_ovagent_error(self):
        from omicverse.utils.agent_errors import OVAgentError, WorkflowNeedsFallback
        with pytest.raises(OVAgentError):
            raise WorkflowNeedsFallback("compat")

    def test_docstring_mentions_deprecated(self):
        from omicverse.utils.agent_errors import WorkflowNeedsFallback
        assert "deprecated" in (WorkflowNeedsFallback.__doc__ or "").lower()

    def test_identity_matches_across_import_paths(self):
        """The class is the same object whether imported directly or via module."""
        from omicverse.utils.agent_errors import WorkflowNeedsFallback as direct
        import omicverse.utils.agent_errors as mod
        assert direct is mod.WorkflowNeedsFallback

    def test_in_utils_all_as_intentional_export(self):
        """WorkflowNeedsFallback is an intentional public export, not accidental."""
        from omicverse.utils import __all__ as exports
        assert "WorkflowNeedsFallback" in exports

    def test_docstring_documents_no_runtime_usage(self):
        """The docstring explicitly states the shim has no runtime usage."""
        from omicverse.utils.agent_errors import WorkflowNeedsFallback
        doc = WorkflowNeedsFallback.__doc__ or ""
        assert "compatibility" in doc.lower()
        assert "no code path" in doc.lower() or "no runtime" in doc.lower()

    def test_not_raised_in_runtime_source(self):
        """No runtime module raises WorkflowNeedsFallback."""
        from pathlib import Path
        import re
        utils_dir = Path(__file__).resolve().parents[2] / "omicverse" / "utils"
        pattern = re.compile(r"\braise\s+WorkflowNeedsFallback\b")
        for py_file in utils_dir.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8", errors="replace")
            assert not pattern.search(source), (
                f"Runtime code raises WorkflowNeedsFallback in {py_file.name}"
            )

    def test_not_caught_in_runtime_source(self):
        """No runtime module catches WorkflowNeedsFallback."""
        from pathlib import Path
        import re
        utils_dir = Path(__file__).resolve().parents[2] / "omicverse" / "utils"
        pattern = re.compile(r"\bexcept\b.*\bWorkflowNeedsFallback\b")
        for py_file in utils_dir.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8", errors="replace")
            assert not pattern.search(source), (
                f"Runtime code catches WorkflowNeedsFallback in {py_file.name}"
            )


# ===================================================================
# 3b. agent_mode legacy handling (no-op compatibility path)
# ===================================================================


class TestAgentModeLegacyHandling:
    """agent_mode='legacy' is a documented no-op — warn and continue."""

    def test_agent_mode_parameter_exists_on_init(self):
        """OmicVerseAgent.__init__ accepts agent_mode with default 'agentic'."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        sig = inspect.signature(OmicVerseAgent.__init__)
        param = sig.parameters["agent_mode"]
        assert param.default == "agentic"

    def test_agent_mode_parameter_exists_on_factory(self):
        """Agent() factory accepts agent_mode with default 'agentic'."""
        from omicverse.utils.smart_agent import Agent
        sig = inspect.signature(Agent)
        param = sig.parameters["agent_mode"]
        assert param.default == "agentic"

    def test_agent_mode_not_forwarded_to_config(self):
        """agent_mode is never passed to AgentConfig.from_flat_kwargs."""
        from omicverse.utils.agent_config import AgentConfig
        sig = inspect.signature(AgentConfig.from_flat_kwargs)
        assert "agent_mode" not in sig.parameters, (
            "agent_mode must not be a config parameter — it is a no-op shim"
        )

    def test_agent_mode_not_stored_on_config_class(self):
        """AgentConfig has no agent_mode field — the parameter is truly discarded."""
        from omicverse.utils.agent_config import AgentConfig
        assert not hasattr(AgentConfig, "agent_mode"), (
            "AgentConfig must not store agent_mode"
        )
        # Also check dataclass fields if present
        if hasattr(AgentConfig, "__dataclass_fields__"):
            assert "agent_mode" not in AgentConfig.__dataclass_fields__

    def test_legacy_warning_source_guard(self):
        """__init__ source emits DeprecationWarning for non-agentic agent_mode."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        source = inspect.getsource(OmicVerseAgent.__init__)
        assert "agent_mode" in source
        assert "DeprecationWarning" in source
        assert "deprecated" in source.lower()


# ===================================================================
# 4. No provider-specific behavior changes
# ===================================================================


class TestNoProviderBehaviorChange:
    """Verify hygiene fixes don't alter provider dispatch or types."""

    def test_dispatch_tables_unchanged(self):
        from omicverse.utils.agent_backend_common import _SYNC_DISPATCH, _STREAM_DISPATCH
        from omicverse.utils.model_config import WireAPI

        assert _SYNC_DISPATCH[WireAPI.CHAT_COMPLETIONS] == "_chat_via_openai_compatible"
        assert _SYNC_DISPATCH[WireAPI.ANTHROPIC_MESSAGES] == "_chat_via_anthropic"
        assert _SYNC_DISPATCH[WireAPI.GEMINI_GENERATE] == "_chat_via_gemini"
        assert _SYNC_DISPATCH[WireAPI.DASHSCOPE] == "_chat_via_dashscope"
        assert _SYNC_DISPATCH[WireAPI.LOCAL] == "_run_python_local"

    def test_backend_config_fields_unchanged(self):
        from omicverse.utils.agent_backend_common import BackendConfig
        fields = set(BackendConfig.__dataclass_fields__.keys())
        expected = {
            "model", "api_key", "endpoint", "provider", "system_prompt",
            "max_tokens", "temperature", "max_retry_attempts",
            "retry_base_delay", "retry_backoff_factor", "retry_jitter",
        }
        assert fields == expected

    def test_all_exports_preserved(self):
        from omicverse.utils.agent_backend_common import __all__ as exports
        expected = {
            "BackendConfig", "Usage", "ToolCall", "ChatResponse",
            "_coerce_int", "_compute_total", "_should_retry",
            "_retry_with_backoff", "_request_timeout_seconds",
            "_get_shared_executor",
        }
        assert expected == set(exports)


# ===================================================================
# 5. _retry_with_backoff public contract (PR #603 reconciliation)
# ===================================================================


class TestRetryWithBackoffPublicContract:
    """Explicit regression tests for the _retry_with_backoff public contract.

    These tests freeze the contract that PR #603 was trying to establish:
    _retry_with_backoff accepts a zero-arg callable and explicit config
    parameters. No silent arg-forwarding is possible.
    """

    def test_signature_has_exactly_five_params(self):
        """_retry_with_backoff has exactly 5 parameters: func + 4 config."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff
        sig = inspect.signature(_retry_with_backoff)
        assert len(sig.parameters) == 5, (
            f"Expected 5 params (func + 4 config), got {len(sig.parameters)}: "
            f"{list(sig.parameters.keys())}"
        )

    def test_func_is_first_parameter(self):
        """'func' must be the first positional parameter."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff
        sig = inspect.signature(_retry_with_backoff)
        first_param = list(sig.parameters.keys())[0]
        assert first_param == "func"

    def test_config_params_have_defaults(self):
        """All config parameters must have defaults (backward-compat)."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff
        sig = inspect.signature(_retry_with_backoff)
        for name in ("max_attempts", "base_delay", "factor", "jitter"):
            param = sig.parameters[name]
            assert param.default is not inspect.Parameter.empty, (
                f"{name} must have a default value"
            )

    def test_func_invoked_as_zero_arg(self):
        """func() is called with zero args; positionals are config."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff

        received_args = []

        def spy(*args):
            received_args.extend(args)
            return "done"

        result = _retry_with_backoff(spy, 1)  # 1 = max_attempts
        assert result == "done"
        assert received_args == [], (
            f"func received args {received_args}; "
            "positional config was silently forwarded"
        )

    def test_backend_retry_wraps_args_in_closure(self):
        """_retry wraps func+args into a lambda."""
        from omicverse.utils.agent_backend import OmicVerseLLMBackend
        source = inspect.getsource(OmicVerseLLMBackend._retry)
        # The implementation must use lambda or partial to bind args
        assert "lambda" in source or "partial" in source, (
            "OmicVerseLLMBackend._retry must wrap func+args into a closure"
        )
