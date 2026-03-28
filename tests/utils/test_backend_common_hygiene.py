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
    """Config params must be keyword-only; *args/**kwargs forward to func."""

    def test_config_params_are_keyword_only(self):
        """max_attempts, base_delay, factor, jitter must be keyword-only."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff
        sig = inspect.signature(_retry_with_backoff)
        for name in ("max_attempts", "base_delay", "factor", "jitter"):
            param = sig.parameters[name]
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"{name} should be KEYWORD_ONLY, got {param.kind.name}"
            )

    def test_args_forwarded_to_func(self):
        """Positional args after func are forwarded, not consumed by config."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff
        sentinel = object()

        def echo(*args, **kwargs):
            return args, kwargs

        result_args, result_kwargs = _retry_with_backoff(
            echo, sentinel, "b", max_attempts=1, key="val"
        )
        assert result_args == (sentinel, "b")
        assert result_kwargs == {"key": "val"}

    def test_kwargs_not_consumed_by_config(self):
        """Keyword args not matching config params forward to func."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff

        def capture(**kw):
            return kw

        result = _retry_with_backoff(capture, max_attempts=1, foo="bar", baz=42)
        assert result == {"foo": "bar", "baz": 42}

    def test_positional_config_raises_typeerror(self):
        """Passing config params positionally (old style) now goes to *args."""
        from omicverse.utils.agent_backend_common import _retry_with_backoff

        def echo(*args):
            return args

        # With old signature, 5 would fill max_attempts. Now it goes to *args.
        result = _retry_with_backoff(echo, 5, max_attempts=1)
        assert result == (5,)

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

    def test_backend_retry_method_compatible(self, monkeypatch):
        """OmicVerseLLMBackend._retry still works with the new signature."""
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
