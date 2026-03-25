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

    def test_fallback_minimal_workflow_uses_ov_not_scanpy(self):
        """Fallback code should not generate scanpy calls that violate later guards."""
        code = CodegenPipeline._fallback_minimal_workflow()
        assert "import scanpy as sc" not in code
        assert "sc.pp." not in code
        assert "sc.tl." not in code
        assert "adata = adata" not in code
        assert "ov.pp.normalize_total" in code
        assert "ov.pp.log1p" in code
        assert "ov.pp.highly_variable_genes" in code


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
            request_content=None,
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
            request_content=None,
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
            request_content=None,
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
