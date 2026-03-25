"""Tests for the CodegenPipeline extracted module.

Verifies that:
1. The code generation path is owned by a dedicated module (CodegenPipeline).
2. smart_agent.py delegates rather than implementing extraction/review/reflection inline.
3. Code-only and execute_code-adjacent behavior remains compatible for claw.py.
"""

import ast
import asyncio
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import MethodType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set up lightweight module stubs (same pattern as test_smart_agent.py)
_ORIGINAL_MODULES = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]
}
for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]:
    sys.modules.pop(name, None)

omicverse_pkg = types.ModuleType("omicverse")
omicverse_pkg.__path__ = [str(PACKAGE_ROOT)]
omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = omicverse_pkg

utils_pkg = types.ModuleType("omicverse.utils")
utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = utils_pkg
omicverse_pkg.utils = utils_pkg

smart_agent_spec = importlib.util.spec_from_file_location(
    "omicverse.utils.smart_agent", PACKAGE_ROOT / "utils" / "smart_agent.py"
)
smart_agent_module = importlib.util.module_from_spec(smart_agent_spec)
sys.modules["omicverse.utils.smart_agent"] = smart_agent_module
assert smart_agent_spec.loader is not None
smart_agent_spec.loader.exec_module(smart_agent_module)

OmicVerseAgent = smart_agent_module.OmicVerseAgent
from omicverse.utils.ovagent.codegen_pipeline import CodegenPipeline
from omicverse.utils.ovagent.tool_runtime import ToolRuntime
from omicverse.utils.harness import build_stream_event

for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


# ---------------------------------------------------------------------------
# AC-1: Code generation path is owned by CodegenPipeline
# ---------------------------------------------------------------------------


class TestCodegenPipelineOwnership:
    """Verify that CodegenPipeline is a standalone module that owns the codegen path."""

    def test_codegen_pipeline_module_exists(self):
        """CodegenPipeline is importable from ovagent subpackage."""
        assert CodegenPipeline is not None

    def test_pipeline_has_core_extraction_methods(self):
        """Pipeline exposes the code extraction surface."""
        pipeline = CodegenPipeline.__new__(CodegenPipeline)
        assert callable(getattr(pipeline, "extract_python_code", None))
        assert callable(getattr(pipeline, "extract_python_code_strict", None))
        assert callable(getattr(pipeline, "gather_code_candidates", None))
        assert callable(getattr(pipeline, "normalize_code_candidate", None))
        assert callable(getattr(pipeline, "detect_direct_python_request", None))

    def test_pipeline_has_generation_methods(self):
        """Pipeline exposes the code generation surface."""
        pipeline = CodegenPipeline.__new__(CodegenPipeline)
        assert callable(getattr(pipeline, "generate_code_async", None))
        assert callable(getattr(pipeline, "generate_code", None))
        assert callable(getattr(pipeline, "generate_code_via_agentic_loop", None))

    def test_pipeline_has_review_reflection_methods(self):
        """Pipeline exposes the review and reflection surface."""
        pipeline = CodegenPipeline.__new__(CodegenPipeline)
        assert callable(getattr(pipeline, "review_result", None))
        assert callable(getattr(pipeline, "reflect_on_code", None))
        assert callable(getattr(pipeline, "review_generated_code_lightweight", None))
        assert callable(getattr(pipeline, "rewrite_code_without_scanpy", None))


# ---------------------------------------------------------------------------
# AC-2: smart_agent.py delegates rather than implementing inline
# ---------------------------------------------------------------------------


class TestSmartAgentDelegation:
    """Verify that OmicVerseAgent delegates codegen methods to CodegenPipeline."""

    def test_agent_has_codegen_property(self):
        """Agent exposes lazy _codegen property."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        pipeline = agent._codegen
        assert isinstance(pipeline, CodegenPipeline)

    def test_agent_extract_python_code_delegates(self):
        """_extract_python_code delegates to pipeline."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)

        response = "```python\nimport omicverse as ov\nov.pp.pca(adata)\n```"
        code = agent._extract_python_code(response)

        # Verify the code was extracted and transformed
        assert "ov.pp.pca" in code
        ast.parse(code)

    def test_agent_gather_code_candidates_delegates(self):
        """_gather_code_candidates delegates to pipeline."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        candidates = agent._gather_code_candidates(
            "```python\nimport ov\n```"
        )
        assert len(candidates) >= 1

    def test_agent_extract_strict_delegates(self):
        """_extract_python_code_strict delegates to pipeline."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        code = agent._extract_python_code_strict(
            "```python\nimport omicverse as ov\n```"
        )
        assert "import omicverse" in code

    def test_agent_detect_direct_python_delegates(self):
        """_detect_direct_python_request delegates to pipeline."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        agent.provider = "openai"
        result = agent._detect_direct_python_request("import omicverse as ov\nov.pp.pca(adata)")
        assert result is not None
        assert "ov.pp.pca" in result

    def test_agent_normalize_code_candidate_delegates(self):
        """_normalize_code_candidate delegates to pipeline."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        normalized = agent._normalize_code_candidate("print('hello')")
        assert "import omicverse as ov" in normalized

    def test_agent_looks_like_python_delegates(self):
        """_looks_like_python delegates to pipeline."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        assert agent._looks_like_python("import numpy as np\nfor x in range(10):\n    pass") is True
        assert agent._looks_like_python("this is plain text") is False


# ---------------------------------------------------------------------------
# AC-3: Code-only and execute_code-adjacent behavior compatible for claw.py
# ---------------------------------------------------------------------------


class TestClawCompatibility:
    """Verify that the claw.py consumer contract is preserved."""

    def test_generate_code_async_captures_execute_code(self):
        """generate_code_async delegates to agentic loop and captures code-only output."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        agent.provider = "openai"
        agent._code_only_mode = False
        agent._code_only_captured_code = ""
        agent._code_only_captured_history = []

        seen = {}

        async def _fake_run_agentic_loop(
            self, request, adata,
            event_callback=None, cancel_event=None,
            history=None, approval_handler=None,
        ):
            seen["request"] = request
            assert self._code_only_mode is True
            self._capture_code_only_snippet("import omicverse as ov\nov.pp.qc(adata)")
            await event_callback(build_stream_event(
                "tool_call",
                {"name": "execute_code", "arguments": {"description": "qc"}},
                turn_id="turn-1", trace_id="trace-1",
                session_id="session-1", category="tool",
            ))
            await event_callback(build_stream_event(
                "done", "captured",
                turn_id="turn-1", trace_id="trace-1",
                session_id="session-1", category="lifecycle",
            ))

        agent._run_agentic_loop = MethodType(_fake_run_agentic_loop, agent)

        result = asyncio.run(agent.generate_code_async("basic qc", None))

        assert result == "import omicverse as ov\nov.pp.qc(adata)"
        assert "CLAW REQUEST MODE" in seen["request"]
        assert agent._code_only_mode is False

    def test_code_only_mode_restored_after_generation(self):
        """Code-only mode state is cleanly restored even if loop fails."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        agent.provider = "openai"
        agent._code_only_mode = False
        agent._code_only_captured_code = "original"
        agent._code_only_captured_history = [{"code": "original"}]

        async def _failing_loop(
            self, request, adata,
            event_callback=None, cancel_event=None,
            history=None, approval_handler=None,
        ):
            raise RuntimeError("simulated failure")

        agent._run_agentic_loop = MethodType(_failing_loop, agent)

        with pytest.raises(RuntimeError, match="simulated failure"):
            asyncio.run(agent.generate_code_async("fail test", None))

        assert agent._code_only_mode is False
        assert agent._code_only_captured_code == "original"

    def test_tool_runtime_code_only_capture_uses_pipeline(self):
        """ToolRuntime._tool_execute_code captures via agent._capture_code_only_snippet."""
        runtime = ToolRuntime.__new__(ToolRuntime)
        captured = {}

        class _Ctx:
            _code_only_mode = True

            def _capture_code_only_snippet(self, code, description=""):
                captured["code"] = code
                captured["description"] = description

        class _Executor:
            def check_code_prerequisites(self, code, adata):
                raise AssertionError("should not check in code-only mode")

            def execute_generated_code(self, code, adata, capture_stdout=True):
                raise AssertionError("should not execute in code-only mode")

        runtime._ctx = _Ctx()
        runtime._executor = _Executor()

        result = runtime._tool_execute_code(
            "import omicverse as ov\nov.pp.pca(adata)", "pca", None
        )

        assert "captured generated Python code" in result["output"]
        assert captured["description"] == "pca"
        assert "ov.pp.pca" in captured["code"]

    def test_direct_python_detection_bypasses_llm(self):
        """Direct Python requests are detected and returned without LLM calls."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        agent.provider = "openai"

        code = agent._detect_direct_python_request(
            "```python\nimport omicverse as ov\nov.pp.pca(adata)\n```"
        )
        assert code is not None
        assert "ov.pp.pca" in code

    def test_generate_code_async_direct_python_skips_loop(self):
        """generate_code_async returns directly for Python snippets without calling the loop."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        agent.provider = "openai"
        agent.last_usage = "should_be_cleared"
        agent.last_usage_breakdown = {}

        async def _should_not_be_called(
            self, request, adata,
            event_callback=None, cancel_event=None,
            history=None, approval_handler=None,
        ):
            raise AssertionError("Loop should not be called for direct Python")

        agent._run_agentic_loop = MethodType(_should_not_be_called, agent)

        result = asyncio.run(agent.generate_code_async(
            "```python\nimport omicverse as ov\nov.pp.pca(adata)\n```",
            None,
        ))

        assert "ov.pp.pca" in result
        assert agent.last_usage is None


# ---------------------------------------------------------------------------
# CodegenPipeline standalone extraction tests
# ---------------------------------------------------------------------------


class TestCodegenPipelineExtraction:
    """Verify code extraction methods work standalone through the pipeline."""

    def _make_pipeline(self):
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        return CodegenPipeline(agent)

    def test_extract_fenced_python_block(self):
        pipeline = self._make_pipeline()
        response = "Here is the code:\n```python\nimport omicverse as ov\nov.pp.pca(adata)\n```"
        code = pipeline.extract_python_code(response)
        assert "ov.pp.pca" in code
        ast.parse(code)

    def test_extract_strict_raises_on_no_code(self):
        pipeline = self._make_pipeline()
        with pytest.raises(ValueError, match="no code candidates"):
            pipeline.extract_python_code_strict("no code here at all")

    def test_normalize_adds_import(self):
        normalized = CodegenPipeline.normalize_code_candidate("print('hello')")
        assert "import omicverse as ov" in normalized

    def test_normalize_preserves_existing_import(self):
        code = "import omicverse as ov\nprint('hello')"
        normalized = CodegenPipeline.normalize_code_candidate(code)
        assert normalized.count("import omicverse") == 1

    def test_looks_like_python_positive(self):
        assert CodegenPipeline.looks_like_python("import numpy as np\nfor x in data:\n    pass")

    def test_looks_like_python_negative(self):
        assert not CodegenPipeline.looks_like_python("this is just a sentence")

    def test_contains_forbidden_scanpy(self):
        assert CodegenPipeline.contains_forbidden_scanpy_usage("import scanpy as sc\nsc.pp.pca(adata)")
        assert not CodegenPipeline.contains_forbidden_scanpy_usage("import omicverse as ov\nov.pp.pca(adata)")

    def test_build_code_only_request_with_adata(self):
        request = CodegenPipeline.build_code_only_agentic_request("do qc", object())
        assert "CLAW REQUEST MODE" in request
        assert "live dataset" in request

    def test_build_code_only_request_without_adata(self):
        request = CodegenPipeline.build_code_only_agentic_request("do qc", None)
        assert "CLAW REQUEST MODE" in request
        assert "already exists" in request


# ---------------------------------------------------------------------------
# Task-013: Structured self-healing repair loop tests
# ---------------------------------------------------------------------------

from omicverse.utils.ovagent.contracts import (
    ExecutionFailureEnvelope,
    FailurePhase,
    RepairHint,
)
from omicverse.utils.ovagent.repair_loop import (
    HintDrivenRepairStrategy,
    RegexRepairStrategy,
    RepairLoop,
    RepairResult,
    build_default_repair_loop,
    normalize_execution_failure,
)


class TestFailureNormalization:
    """AC: Execution failures are normalized into structured repair inputs."""

    def test_normalize_produces_envelope(self):
        """Raw exceptions are normalized into ExecutionFailureEnvelope."""
        try:
            raise ValueError("test error message")
        except ValueError as exc:
            envelope = normalize_execution_failure(
                tool_name="execute_code",
                exception=exc,
                code="print('hello')",
            )

        assert type(envelope).__name__ == "ExecutionFailureEnvelope"
        assert envelope.tool_name == "execute_code"
        assert envelope.exception_type == "ValueError"
        assert envelope.message == "test error message"
        assert envelope.phase.value == "execution"
        assert envelope.retry_count == 0
        assert envelope.max_retries == 3

    def test_envelope_has_required_fields(self):
        """Failure envelope includes phase, tool/code context, traceback
        summary, retry count, and repair hints (interface contract)."""
        try:
            raise RuntimeError("something broke")
        except RuntimeError as exc:
            envelope = normalize_execution_failure(
                tool_name="execute_code",
                exception=exc,
                code="adata = ov.pp.pca(adata)",
                retry_count=1,
                max_retries=5,
            )

        # phase
        assert envelope.phase.value in ("pre_exec", "execution", "post_exec", "normalization", "timeout")
        # tool/code context
        assert envelope.tool_name == "execute_code"
        # traceback summary
        assert len(envelope.traceback_summary) > 0
        assert "RuntimeError" in envelope.traceback_summary
        # retry count
        assert envelope.retry_count == 1
        assert envelope.max_retries == 5
        # repair hints (list, may be empty for generic errors)
        assert isinstance(envelope.repair_hints, list)

    def test_envelope_to_llm_message_format(self):
        """to_llm_message() produces a human/LLM readable message."""
        envelope = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="TypeError",
            message="bad argument",
            retry_count=0,
            max_retries=3,
            repair_hints=[
                RepairHint(
                    strategy="test", description="try something else"
                )
            ],
        )
        msg = envelope.to_llm_message()
        assert "execute_code" in msg
        assert "execution" in msg
        assert "TypeError" in msg
        assert "bad argument" in msg
        assert "try something else" in msg
        assert "Attempt 1/3" in msg

    def test_timeout_classified_correctly(self):
        """Timeout exceptions are classified as TIMEOUT phase."""
        try:
            raise TimeoutError("operation timed out")
        except TimeoutError as exc:
            envelope = normalize_execution_failure(
                tool_name="execute_code",
                exception=exc,
                code="import time; time.sleep(999)",
            )
        assert envelope.phase == FailurePhase.TIMEOUT

    def test_security_classified_as_pre_exec(self):
        """Security/sandbox errors are classified as PRE_EXEC phase."""
        try:
            raise PermissionError("sandbox denied this operation")
        except PermissionError as exc:
            envelope = normalize_execution_failure(
                tool_name="execute_code",
                exception=exc,
                code="import os; os.system('rm -rf /')",
            )
        assert envelope.phase == FailurePhase.PRE_EXEC

    def test_missing_module_generates_hint(self):
        """ModuleNotFoundError generates an auto_install repair hint."""
        try:
            raise ModuleNotFoundError("No module named 'coolpackage'")
        except ModuleNotFoundError as exc:
            envelope = normalize_execution_failure(
                tool_name="execute_code",
                exception=exc,
                code="import coolpackage",
            )
        hint_strategies = [h.strategy for h in envelope.repair_hints]
        assert "auto_install" in hint_strategies
        auto_hint = next(
            h for h in envelope.repair_hints if h.strategy == "auto_install"
        )
        assert auto_hint.metadata["package"] == "coolpackage"

    def test_dtype_error_generates_rename_hint(self):
        """AttributeError for .dtype generates rename_attribute hint."""
        try:
            raise AttributeError("has no attribute 'dtype'")
        except AttributeError as exc:
            envelope = normalize_execution_failure(
                tool_name="execute_code",
                exception=exc,
                code="df.dtype",
            )
        hint_strategies = [h.strategy for h in envelope.repair_hints]
        assert "rename_attribute" in hint_strategies

    def test_inplace_assignment_generates_remove_hint(self):
        """NoneType error with in-place function generates remove_assignment."""
        try:
            raise AttributeError("'NoneType' object has no attribute 'obs'")
        except AttributeError as exc:
            envelope = normalize_execution_failure(
                tool_name="execute_code",
                exception=exc,
                code="adata = ov.pp.pca(adata)",
            )
        hint_strategies = [h.strategy for h in envelope.repair_hints]
        assert "remove_assignment" in hint_strategies

    def test_name_error_generates_undefined_hint(self):
        """NameError generates an undefined_name repair hint."""
        try:
            raise NameError("name 'foo' is not defined")
        except NameError as exc:
            envelope = normalize_execution_failure(
                tool_name="execute_code",
                exception=exc,
                code="print(foo)",
            )
        hint_strategies = [h.strategy for h in envelope.repair_hints]
        assert "undefined_name" in hint_strategies


class TestRepairStrategies:
    """AC: Regex transforms are demoted from primary recovery path."""

    def test_hint_driven_rename_attribute(self):
        """HintDrivenRepairStrategy applies rename_attribute hints."""
        strategy = HintDrivenRepairStrategy()
        envelope = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="AttributeError",
            message="has no attribute 'dtype'",
            repair_hints=[
                RepairHint(
                    strategy="rename_attribute",
                    description="Use .dtypes instead of .dtype",
                    metadata={"old": ".dtype", "new": ".dtypes"},
                )
            ],
        )
        code = "print(df.dtype)"
        result = strategy.attempt_repair(envelope, code, None)
        assert result is not None
        assert ".dtypes" in result
        assert ".dtype" not in result.replace(".dtypes", "")

    def test_hint_driven_add_import(self):
        """HintDrivenRepairStrategy applies add_import hints."""
        strategy = HintDrivenRepairStrategy()
        envelope = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="NameError",
            message="name 'pd' is not defined",
            repair_hints=[
                RepairHint(
                    strategy="add_import",
                    description="Add pandas import",
                    metadata={"import_line": "import pandas as pd"},
                )
            ],
        )
        code = "df = pd.DataFrame()"
        result = strategy.attempt_repair(envelope, code, None)
        assert result is not None
        assert result.startswith("import pandas as pd")

    def test_hint_driven_remove_assignment(self):
        """HintDrivenRepairStrategy applies remove_assignment hints."""
        strategy = HintDrivenRepairStrategy()
        envelope = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="AttributeError",
            message="'NoneType' object has no attribute 'obs'",
            repair_hints=[
                RepairHint(
                    strategy="remove_assignment",
                    description="Remove assignment from in-place call",
                    metadata={
                        "pattern": r"adata\s*=\s*(ov\.pp\.pca\s*\([^)]*\))",
                        "replacement": r"\1",
                    },
                )
            ],
        )
        code = "adata = ov.pp.pca(adata)"
        result = strategy.attempt_repair(envelope, code, None)
        assert result is not None
        assert result.strip() == "ov.pp.pca(adata)"

    def test_hint_driven_returns_none_for_no_matching_hints(self):
        """HintDrivenRepairStrategy returns None when no hints match."""
        strategy = HintDrivenRepairStrategy()
        envelope = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="RuntimeError",
            message="something weird",
            repair_hints=[
                RepairHint(
                    strategy="unknown_strategy",
                    description="no handler for this",
                )
            ],
        )
        result = strategy.attempt_repair(envelope, "print('hi')", None)
        assert result is None

    def test_regex_strategy_wraps_executor(self):
        """RegexRepairStrategy wraps apply_execution_error_fix."""

        class FakeExecutor:
            def apply_execution_error_fix(self, code, error_msg):
                if "dtype" in error_msg.lower():
                    return code.replace(".dtype", ".dtypes")
                return None

        strategy = RegexRepairStrategy(FakeExecutor())
        envelope = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="AttributeError",
            message="has no attribute 'dtype'",
        )
        result = strategy.attempt_repair(envelope, "df.dtype", None)
        assert result == "df.dtypes"

    def test_regex_strategy_returns_none_when_no_fix(self):
        """RegexRepairStrategy returns None when executor has no fix."""

        class FakeExecutor:
            def apply_execution_error_fix(self, code, error_msg):
                return None

        strategy = RegexRepairStrategy(FakeExecutor())
        envelope = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="RuntimeError",
            message="unknown error",
        )
        result = strategy.attempt_repair(envelope, "print('hi')", None)
        assert result is None


class TestRepairLoop:
    """AC: Multiple repair attempts are supported; regex demoted."""

    def test_successful_repair_on_first_attempt(self):
        """Repair loop succeeds when first strategy works."""
        call_log = []

        class FixStrategy:
            name = "always_fix"

            def attempt_repair(self, envelope, code, adata):
                return code.replace("broken", "fixed")

        def executor(code, adata):
            call_log.append(code)
            if "broken" in code:
                raise ValueError("still broken")
            return {"output": "ok", "adata": adata}

        loop = RepairLoop(
            strategies=[FixStrategy()],
            executor_fn=executor,
            max_retries=3,
        )
        result = loop.run(
            code="broken code",
            adata=None,
            initial_exception=ValueError("broken"),
        )
        assert result.success is True
        assert result.winning_strategy == "always_fix"
        assert result.total_attempts == 1

    def test_multiple_attempts_before_success(self):
        """Repair loop tries multiple attempts when first fix fails."""
        attempt_count = 0

        class IncrementalFix:
            name = "incremental"

            def attempt_repair(self, envelope, code, adata):
                # Each attempt adds a fix marker
                return code + "\n# fix"

        def executor(code, adata):
            nonlocal attempt_count
            attempt_count += 1
            # Only succeeds on 3rd attempt (code has 3+ fix markers)
            if code.count("# fix") < 3:
                raise ValueError(f"not enough fixes ({code.count('# fix')})")
            return {"output": "success", "adata": adata}

        loop = RepairLoop(
            strategies=[IncrementalFix()],
            executor_fn=executor,
            max_retries=5,
        )
        result = loop.run(
            code="base",
            adata=None,
            initial_exception=ValueError("initial"),
        )
        assert result.success is True
        assert result.total_attempts >= 3

    def test_exhausts_retries_and_fails(self):
        """Repair loop fails gracefully when max retries exhausted."""

        class BadFix:
            name = "bad_fix"

            def attempt_repair(self, envelope, code, adata):
                return code + "# attempted"

        def executor(code, adata):
            raise RuntimeError("always fails")

        loop = RepairLoop(
            strategies=[BadFix()],
            executor_fn=executor,
            max_retries=2,
        )
        result = loop.run(
            code="code",
            adata=None,
            initial_exception=RuntimeError("initial"),
        )
        assert result.success is False
        assert result.total_attempts == 2
        assert result.final_envelope is not None

    def test_no_strategy_matches(self):
        """Loop terminates early when no strategy can produce a fix."""

        class NoFix:
            name = "no_fix"

            def attempt_repair(self, envelope, code, adata):
                return None

        loop = RepairLoop(
            strategies=[NoFix()],
            executor_fn=lambda c, a: {"output": "ok", "adata": a},
            max_retries=3,
        )
        result = loop.run(
            code="code",
            adata=None,
            initial_exception=ValueError("error"),
        )
        assert result.success is False
        assert result.total_attempts == 1  # only one "none" attempt logged

    def test_strategy_order_hint_before_regex(self):
        """Default loop tries HintDrivenRepairStrategy before RegexRepairStrategy."""

        class FakeExecutor:
            def apply_execution_error_fix(self, code, error_msg):
                return code.replace(".dtype", ".dtypes_regex")

        def executor(code, adata):
            return {"output": "ok", "adata": adata}

        loop = build_default_repair_loop(
            executor=FakeExecutor(),
            executor_fn=executor,
        )
        # Verify strategy ordering
        assert loop.strategy_names[0] == "hint_driven_repair"
        assert loop.strategy_names[1] == "regex_pattern_fix"

    def test_default_loop_hint_driven_takes_priority(self):
        """When hint-driven strategy produces a fix, regex is not consulted."""
        regex_called = False

        class TrackingExecutor:
            def apply_execution_error_fix(self, code, error_msg):
                nonlocal regex_called
                regex_called = True
                return code.replace(".dtype", ".dtypes_regex")

        def executor(code, adata):
            return {"output": "ok", "adata": adata}

        loop = build_default_repair_loop(
            executor=TrackingExecutor(),
            executor_fn=executor,
        )

        try:
            raise AttributeError("has no attribute 'dtype'")
        except AttributeError as exc:
            result = loop.run(
                code="print(df.dtype)",
                adata=None,
                initial_exception=exc,
            )

        assert result.success is True
        assert result.winning_strategy == "hint_driven_repair"
        assert not regex_called

    def test_default_loop_falls_back_to_regex(self):
        """When hint-driven strategy has no match, regex fallback runs."""

        class FakeExecutor:
            def apply_execution_error_fix(self, code, error_msg):
                if "custom error" in error_msg.lower():
                    return "# regex fixed\n" + code
                return None

        def executor(code, adata):
            return {"output": "ok", "adata": adata}

        loop = build_default_repair_loop(
            executor=FakeExecutor(),
            executor_fn=executor,
        )

        try:
            raise RuntimeError("custom error XYZ")
        except RuntimeError as exc:
            result = loop.run(
                code="print('original')",
                adata=None,
                initial_exception=exc,
            )

        assert result.success is True
        assert result.winning_strategy == "regex_pattern_fix"

    def test_repair_result_tracks_all_attempts(self):
        """RepairResult records every attempt with strategy and outcome."""
        attempt_num = 0

        class FailThenFix:
            name = "fail_then_fix"

            def attempt_repair(self, envelope, code, adata):
                return "# fixed\n" + code

        def executor(code, adata):
            nonlocal attempt_num
            attempt_num += 1
            if attempt_num < 2:
                raise ValueError("retry needed")
            return {"output": "ok", "adata": adata}

        loop = RepairLoop(
            strategies=[FailThenFix()],
            executor_fn=executor,
            max_retries=3,
        )
        result = loop.run(
            code="code",
            adata=None,
            initial_exception=ValueError("initial"),
        )
        assert result.success is True
        assert result.total_attempts == 2
        assert result.attempts[0].succeeded is False
        assert result.attempts[1].succeeded is True

    def test_envelope_retryable_property(self):
        """ExecutionFailureEnvelope.retryable reflects retry budget."""
        env = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="ValueError",
            message="err",
            retry_count=2,
            max_retries=3,
        )
        assert env.retryable is True

        env_exhausted = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="ValueError",
            message="err",
            retry_count=3,
            max_retries=3,
        )
        assert env_exhausted.retryable is False

    def test_envelope_to_dict_roundtrip(self):
        """ExecutionFailureEnvelope serializes to dict correctly."""
        env = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="TypeError",
            message="bad type",
            retry_count=1,
            max_retries=3,
            repair_hints=[
                RepairHint(
                    strategy="test_strat",
                    description="test desc",
                    confidence=0.9,
                    metadata={"key": "val"},
                )
            ],
        )
        d = env.to_dict()
        assert d["tool_name"] == "execute_code"
        assert d["phase"] == "execution"
        assert d["retry_count"] == 1
        assert d["retryable"] is True
        assert len(d["repair_hints"]) == 1
        assert d["repair_hints"][0]["strategy"] == "test_strat"


class TestToolRuntimeRepairIntegration:
    """Verify tool_runtime._tool_execute_code uses the repair loop."""

    def test_execute_code_error_uses_structured_recovery(self):
        """_tool_execute_code feeds failures through RepairLoop, not bare regex."""
        runtime = ToolRuntime.__new__(ToolRuntime)
        execution_log = []

        class _Ctx:
            _code_only_mode = False
            _config = None

        class _Executor:
            def check_code_prerequisites(self, code, adata):
                return ""

            def execute_generated_code(self, code, adata, capture_stdout=True):
                execution_log.append(code)
                if len(execution_log) == 1:
                    raise AttributeError("has no attribute 'dtype'")
                return {"stdout": "", "adata": adata}

            def apply_execution_error_fix(self, code, error_msg):
                # This should NOT be the primary path anymore
                return code.replace(".dtype", ".dtypes_via_regex")

        runtime._ctx = _Ctx()
        runtime._executor = _Executor()

        result = runtime._tool_execute_code("df.dtype", "test", None)

        assert isinstance(result, dict)
        assert "RECOVERED" in result["output"]
        # The hint_driven strategy should win because it handles .dtype
        assert "hint_driven_repair" in result["output"]

    def test_execute_code_success_skips_repair_loop(self):
        """When code executes successfully, repair loop is never invoked."""
        runtime = ToolRuntime.__new__(ToolRuntime)

        class _Ctx:
            _code_only_mode = False
            _config = None

        class _Executor:
            def check_code_prerequisites(self, code, adata):
                return ""

            def execute_generated_code(self, code, adata, capture_stdout=True):
                return {"stdout": "hello", "adata": adata}

        runtime._ctx = _Ctx()
        runtime._executor = _Executor()

        result = runtime._tool_execute_code("print('hi')", "test", None)
        assert "RECOVERED" not in result["output"]
        assert "stdout:" in result["output"]

    def test_execute_code_all_repairs_fail_returns_structured_error(self):
        """When all repairs fail, returns structured error from envelope."""
        runtime = ToolRuntime.__new__(ToolRuntime)

        class _Ctx:
            _code_only_mode = False
            _config = None

        class _Executor:
            def check_code_prerequisites(self, code, adata):
                return ""

            def execute_generated_code(self, code, adata, capture_stdout=True):
                raise RuntimeError("unfixable error 42")

            def apply_execution_error_fix(self, code, error_msg):
                return None

        runtime._ctx = _Ctx()
        runtime._executor = _Executor()

        result = runtime._tool_execute_code("bad_code()", "test", None)
        assert isinstance(result, dict)
        output = result["output"]
        # Should contain structured envelope message, not raw traceback
        assert "execute_code" in output
        assert "RuntimeError" in output
        assert "unfixable error 42" in output
