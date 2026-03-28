"""Codegen / tool-dispatch / repair-loop facade mixin — extracted from OmicVerseAgent.

Owns the thin delegation layer between the public ``OmicVerseAgent`` surface
and the extracted subsystem classes:

* **CodegenPipeline** — code extraction, review, reflection, code-only mode
* **ToolRuntime**     — tool visibility, plan-mode gating, dispatch routing
* **AnalysisExecutor** — code execution, error repair, package management
* **RegistryScanner** — static AST registry search and scoring
* **FollowUpGate**    — turn follow-up heuristics (tool-choice, promissory detection)

The concrete agent class inherits this mixin so that the delegate methods
appear on ``OmicVerseAgent`` without bloating its class body.

The mixin relies on attributes set by the host ``__init__`` (e.g.
``_tool_runtime``, ``_codegen_pipeline``, ``_analysis_executor``,
``_registry_scanner``) and never stores new top-level attributes beyond
the lazy-init properties it manages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from ..agent_backend import Usage
    from ..skill_registry import SkillMatch
    from .codegen_pipeline import CodegenPipeline
    from .registry_scanner import RegistryScanner
    from .turn_controller import FollowUpGate

logger = logging.getLogger(__name__)


class CodegenToolDispatchFacadeMixin:
    """Mixin providing codegen, tool-dispatch, and analysis-executor delegation.

    Expects the host class to set the following attributes before any
    delegate method is called:

    * ``_tool_runtime`` — :class:`ToolRuntime` instance
    * ``_codegen_pipeline`` — :class:`CodegenPipeline` instance
    * ``_analysis_executor`` — :class:`AnalysisExecutor` instance
    * ``_registry_scanner`` — :class:`RegistryScanner` instance
    """

    # ================================================================
    # Lazy service constructors (for __new__-based instances in tests)
    # ================================================================

    @property
    def _codegen(self) -> "CodegenPipeline":
        """Lazily create CodegenPipeline for agents constructed via __new__."""
        from .codegen_pipeline import CodegenPipeline as _CodegenPipeline

        pipeline = getattr(self, "_codegen_pipeline", None)
        if pipeline is None:
            pipeline = _CodegenPipeline(self)
            self._codegen_pipeline = pipeline  # type: ignore[attr-defined]
        return pipeline

    @property
    def _scanner(self) -> "RegistryScanner":
        """Lazily create RegistryScanner for agents constructed via __new__."""
        from .registry_scanner import RegistryScanner as _RegistryScanner

        scanner = getattr(self, "_registry_scanner", None)
        if scanner is None:
            scanner = _RegistryScanner()
            self._registry_scanner = scanner  # type: ignore[attr-defined]
        return scanner

    # ================================================================
    # Tool visibility delegates (ToolRuntime)
    # ================================================================

    def _get_visible_agent_tools(
        self, *, allowed_names: Optional[set[str]] = None
    ) -> list[dict[str, Any]]:
        """Return the currently visible tool schemas."""
        return self._tool_runtime.get_visible_agent_tools(  # type: ignore[attr-defined]
            allowed_names=allowed_names
        )

    def _get_loaded_tool_names(self) -> list[str]:
        """Return loaded tool names."""
        return self._tool_runtime.get_loaded_tool_names()  # type: ignore[attr-defined]

    def _tool_blocked_in_plan_mode(self, tool_name: str) -> bool:
        """Check plan-mode blocking."""
        return self._tool_runtime.tool_blocked_in_plan_mode(tool_name)  # type: ignore[attr-defined]

    # ================================================================
    # Tool dispatch delegate (ToolRuntime)
    # ================================================================

    async def _dispatch_tool(
        self, tool_call: Any, current_adata: Any, request: str
    ) -> Any:
        return await self._tool_runtime.dispatch_tool(  # type: ignore[attr-defined]
            tool_call, current_adata, request
        )

    # ================================================================
    # FollowUp gate delegates (TurnController / FollowUpGate)
    # ================================================================

    def _request_requires_tool_action(self, request: str, adata: Any) -> bool:
        from .turn_controller import FollowUpGate as _FollowUpGate

        return _FollowUpGate.request_requires_tool_action(request, adata)

    def _response_is_promissory(self, content: str) -> bool:
        from .turn_controller import FollowUpGate as _FollowUpGate

        return _FollowUpGate.response_is_promissory(content)

    def _select_agent_tool_choice(
        self,
        *,
        request: str,
        adata: Any,
        turn_index: int,
        had_meaningful_tool_call: bool,
        forced_retry: bool,
    ) -> str:
        from .turn_controller import FollowUpGate as _FollowUpGate

        return _FollowUpGate.select_tool_choice(
            request=request,
            adata=adata,
            turn_index=turn_index,
            had_meaningful_tool_call=had_meaningful_tool_call,
            forced_retry=forced_retry,
        )

    # ================================================================
    # Registry scanner delegates
    # ================================================================

    def _load_static_registry_entries(self) -> List[Dict[str, Any]]:
        """Parse @register_function metadata plus nested method/branch capabilities."""
        return self._scanner.load_static_entries()

    def _collect_relevant_registry_entries(
        self, request: str, max_entries: int = 8
    ) -> List[Dict[str, Any]]:
        """Return a compact set of registry entries relevant to a free-form request."""
        return self._scanner.collect_relevant_entries(request, max_entries)

    def _collect_static_registry_entries(
        self, request: str, max_entries: int = 8
    ) -> List[Dict[str, Any]]:
        """Search the static AST-derived registry snapshot."""
        return self._scanner.collect_static_entries(request, max_entries)

    def _score_registry_entry_for_codegen(
        self, request: str, entry: Dict[str, Any]
    ) -> float:
        """Score a registry entry for lightweight code generation retrieval."""
        from .registry_scanner import RegistryScanner as _RegistryScanner

        return _RegistryScanner.score_entry(request, entry)

    # ================================================================
    # Codegen pipeline delegates
    # ================================================================

    def _extract_python_code(self, response_text: str) -> str:
        """Extract executable Python code from the agent response using AST validation."""
        return self._codegen.extract_python_code(response_text)

    def _normalize_registry_entry_for_codegen(
        self, entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert registry entries to public-facing ov.* names for code generation."""
        from .registry_scanner import RegistryScanner as _RegistryScanner

        return _RegistryScanner.normalize_entry(entry)

    def _capture_code_only_snippet(
        self, code: str, description: str = ""
    ) -> None:
        """Store the latest code snippet captured from execute_code in code-only mode."""
        self._codegen.capture_code_only_snippet(code, description)

    def _select_codegen_skill_matches(
        self, request: str, top_k: int = 2
    ) -> List["SkillMatch"]:
        return self._codegen.select_codegen_skill_matches(request, top_k)

    def _format_registry_context_for_codegen(
        self, entries: List[Dict[str, Any]]
    ) -> str:
        return self._codegen.format_registry_context_for_codegen(entries)

    @staticmethod
    def _format_prerequisites_for_codegen_entry(
        entry: Dict[str, Any],
    ) -> str:
        from .codegen_pipeline import CodegenPipeline as _CodegenPipeline

        return _CodegenPipeline.format_prerequisites_for_codegen_entry(entry)

    def _build_code_generation_system_prompt(self, adata: Any) -> str:
        return self._codegen.build_code_generation_system_prompt(adata)

    @staticmethod
    def _build_code_generation_user_prompt(request: str, adata: Any) -> str:
        from .codegen_pipeline import CodegenPipeline as _CodegenPipeline

        return _CodegenPipeline.build_code_generation_user_prompt(request, adata)

    @staticmethod
    def _contains_forbidden_scanpy_usage(code: str) -> bool:
        from .codegen_pipeline import CodegenPipeline as _CodegenPipeline

        return _CodegenPipeline.contains_forbidden_scanpy_usage(code)

    def _rewrite_scanpy_calls_with_registry(
        self, code: str, entries: List[Dict[str, Any]]
    ) -> str:
        return self._codegen.rewrite_scanpy_calls_with_registry(code, entries)

    async def _rewrite_code_without_scanpy(
        self,
        code: str,
        request: str,
        adata: Any,
        registry_context: str = "",
        skill_guidance: str = "",
    ) -> tuple:
        return await self._codegen.rewrite_code_without_scanpy(
            code, request, adata, registry_context, skill_guidance
        )

    async def _review_generated_code_lightweight(
        self, code: str, request: str, adata: Any
    ) -> tuple:
        return await self._codegen.review_generated_code_lightweight(
            code, request, adata
        )

    @staticmethod
    def _build_code_only_agentic_request(request: str, adata: Any) -> str:
        from .codegen_pipeline import CodegenPipeline as _CodegenPipeline

        return _CodegenPipeline.build_code_only_agentic_request(request, adata)

    async def _generate_code_via_agentic_loop(
        self,
        request: str,
        adata: Any,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        return await self._codegen.generate_code_via_agentic_loop(
            request, adata, progress_callback=progress_callback
        )

    def _gather_code_candidates(self, response_text: str) -> List[str]:
        return self._codegen.gather_code_candidates(response_text)

    @staticmethod
    def _looks_like_python(code: str) -> bool:
        from .codegen_pipeline import CodegenPipeline as _CodegenPipeline

        return _CodegenPipeline.looks_like_python(code)

    @staticmethod
    def _extract_inline_python(response_text: str) -> str:
        from .codegen_pipeline import CodegenPipeline as _CodegenPipeline

        return _CodegenPipeline.extract_inline_python(response_text)

    @staticmethod
    def _normalize_code_candidate(code: str) -> str:
        from .codegen_pipeline import CodegenPipeline as _CodegenPipeline

        return _CodegenPipeline.normalize_code_candidate(code)

    def _extract_python_code_strict(self, response_text: str) -> str:
        return self._codegen.extract_python_code_strict(response_text)

    async def _review_result(
        self,
        original_adata: Any,
        result_adata: Any,
        request: str,
        code: str,
    ) -> Dict[str, Any]:
        return await self._codegen.review_result(
            original_adata, result_adata, request, code
        )

    async def _reflect_on_code(
        self,
        code: str,
        request: str,
        adata: Any,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        return await self._codegen.reflect_on_code(
            code, request, adata, iteration
        )

    def _detect_direct_python_request(self, request: str) -> Optional[str]:
        return self._codegen.detect_direct_python_request(request)

    @staticmethod
    def _merge_usage_stats(usages: List[Optional["Usage"]]) -> Optional["Usage"]:
        """Merge usage records from multiple lightweight codegen calls."""
        from ..agent_backend import Usage

        valid = [usage for usage in usages if usage is not None]
        if not valid:
            return None
        return Usage(
            input_tokens=sum(max(0, usage.input_tokens) for usage in valid),
            output_tokens=sum(max(0, usage.output_tokens) for usage in valid),
            total_tokens=sum(max(0, usage.total_tokens) for usage in valid),
            model=valid[-1].model,
            provider=valid[-1].provider,
        )

    # ================================================================
    # Analysis executor delegates (includes repair-loop adjacent)
    # ================================================================

    def _check_code_prerequisites(self, code: str, adata: Any) -> str:
        return self._analysis_executor.check_code_prerequisites(code, adata)  # type: ignore[attr-defined]

    def _apply_execution_error_fix(
        self, code: str, error_msg: str
    ) -> Optional[str]:
        return self._analysis_executor.apply_execution_error_fix(code, error_msg)  # type: ignore[attr-defined]

    @staticmethod
    def _extract_package_name(error_msg: str) -> Optional[str]:
        from .analysis_executor import AnalysisExecutor as _AnalysisExecutor

        return _AnalysisExecutor.extract_package_name(error_msg)

    def _auto_install_package(self, package_name: str) -> bool:
        return self._analysis_executor.auto_install_package(package_name)  # type: ignore[attr-defined]

    async def _diagnose_error_with_llm(
        self,
        code: str,
        error_msg: str,
        traceback_str: str,
        adata: Any,
    ) -> Optional[str]:
        return await self._analysis_executor.diagnose_error_with_llm(  # type: ignore[attr-defined]
            code, error_msg, traceback_str, adata
        )

    def _validate_outputs(
        self, code: str, output_dir: Optional[str] = None
    ) -> List[str]:
        return self._analysis_executor.validate_outputs(code, output_dir)  # type: ignore[attr-defined]

    async def _generate_completion_code(
        self,
        original_code: str,
        missing_files: List[str],
        adata: Any,
        request: str,
    ) -> Optional[str]:
        return await self._analysis_executor.generate_completion_code(  # type: ignore[attr-defined]
            original_code, missing_files, adata, request
        )

    def _request_approval(self, code: str, violations: list) -> bool:
        return self._analysis_executor.request_approval(code, violations)  # type: ignore[attr-defined]

    def _execute_generated_code(
        self, code: str, adata: Any, capture_stdout: bool = False
    ) -> Any:
        return self._analysis_executor.execute_generated_code(  # type: ignore[attr-defined]
            code, adata, capture_stdout
        )

    @staticmethod
    def _normalize_doublet_obs(adata: Any) -> None:
        from .analysis_executor import AnalysisExecutor as _AnalysisExecutor

        _AnalysisExecutor.normalize_doublet_obs(adata)

    def _process_context_directives(
        self, code: str, local_vars: Dict[str, Any]
    ) -> None:
        self._analysis_executor.process_context_directives(code, local_vars)  # type: ignore[attr-defined]

    def _build_sandbox_globals(self) -> Dict[str, Any]:
        return self._analysis_executor.build_sandbox_globals()  # type: ignore[attr-defined]
