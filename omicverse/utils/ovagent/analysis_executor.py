"""Code execution engine for OVAgent — sandbox, code transform, error recovery.

Extracted from ``smart_agent.py``.  ``AnalysisExecutor`` wraps an
:class:`AgentContext` and provides:

* ``ProactiveCodeTransformer`` — regex-based LLM code fix-ups
* ``execute_generated_code`` — sandbox execution with security scan
* ``build_sandbox_globals`` — restricted namespace construction
* ``apply_execution_error_fix`` — pattern-based error recovery (Stage A)
* Helper methods for auto-install, LLM diagnosis, output validation, etc.

After Phase 4 decomposition the heavy implementations live in:

- ``analysis_transformer``   — ProactiveCodeTransformer
- ``analysis_diagnostics``   — prerequisite checks, error recovery, LLM diagnosis
- ``analysis_sandbox``       — sandbox globals, execution, context directives

``AnalysisExecutor`` remains the stable compatibility facade used by
``ToolRuntime`` and the repair loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Re-export the transformer class so existing imports keep working
from .analysis_transformer import ProactiveCodeTransformer  # noqa: F401

# Re-export the package alias map so existing imports keep working
from .analysis_diagnostics import _PACKAGE_ALIASES  # noqa: F401

from . import analysis_diagnostics, analysis_sandbox

if TYPE_CHECKING:
    from .protocol import AgentContext

__all__ = ["AnalysisExecutor", "ProactiveCodeTransformer"]


class AnalysisExecutor:
    """Sandbox execution engine for OVAgent-generated code.

    This class is a compatibility facade.  The underlying implementations
    live in ``analysis_transformer``, ``analysis_diagnostics``, and
    ``analysis_sandbox``.
    """

    def __init__(self, ctx: "AgentContext") -> None:
        self._ctx = ctx

    # -- prerequisite checks ------------------------------------------------

    def check_code_prerequisites(self, code: str, adata: Any) -> str:
        return analysis_diagnostics.check_code_prerequisites(code, adata)

    # -- pattern-based error recovery (Stage A) -----------------------------

    def apply_execution_error_fix(self, code: str, error_msg: str) -> Optional[str]:
        return analysis_diagnostics.apply_execution_error_fix(self._ctx, code, error_msg)

    # -- package management -------------------------------------------------

    @staticmethod
    def extract_package_name(error_msg: str) -> Optional[str]:
        return analysis_diagnostics.extract_package_name(error_msg)

    def auto_install_package(self, package_name: str) -> bool:
        return analysis_diagnostics.auto_install_package(self._ctx, package_name)

    # -- LLM-based error diagnosis ------------------------------------------

    async def diagnose_error_with_llm(
        self,
        code: str,
        error_msg: str,
        traceback_str: str,
        adata: Any,
    ) -> Optional[str]:
        return await analysis_diagnostics.diagnose_error_with_llm(
            self._ctx, code, error_msg, traceback_str, adata,
        )

    # -- output validation --------------------------------------------------

    def validate_outputs(self, code: str, output_dir: Optional[str] = None) -> List[str]:
        return analysis_diagnostics.validate_outputs(code, output_dir)

    async def generate_completion_code(
        self,
        original_code: str,
        missing_files: List[str],
        adata: Any,
        request: str,
    ) -> Optional[str]:
        return await analysis_diagnostics.generate_completion_code(
            self._ctx, original_code, missing_files, adata, request,
        )

    # -- approval gate ------------------------------------------------------

    def request_approval(self, code: str, violations: list) -> bool:
        return analysis_sandbox.request_approval(self._ctx, code, violations)

    # -- read-only snippet execution ----------------------------------------

    def execute_snippet_readonly(self, code: str, adata: Any) -> str:
        return analysis_sandbox.execute_snippet_readonly(self._ctx, code, adata)

    # -- main execution entry point -----------------------------------------

    def execute_generated_code(
        self, code: str, adata: Any, capture_stdout: bool = False,
    ) -> Any:
        return analysis_sandbox.execute_generated_code(
            self._ctx, code, adata, capture_stdout, _executor=self,
        )

    # -- doublet harmonization ----------------------------------------------

    @staticmethod
    def normalize_doublet_obs(adata: Any) -> None:
        analysis_sandbox.normalize_doublet_obs(adata)

    # -- context directives -------------------------------------------------

    def process_context_directives(self, code: str, local_vars: Dict[str, Any]) -> None:
        analysis_sandbox.process_context_directives(self._ctx, code, local_vars)

    # -- sandbox globals ----------------------------------------------------

    def build_sandbox_globals(self) -> Dict[str, Any]:
        return analysis_sandbox.build_sandbox_globals(self._ctx)

    # -- private helpers (kept on facade for backward compat) ---------------

    def _figure_autosave_dir(self):
        return analysis_sandbox.figure_autosave_dir(self._ctx)

    def _inject_figure_autosave(self, code: str) -> str:
        return analysis_sandbox.inject_figure_autosave(self._ctx, code)

    def _handle_context_write(self, directive: str, local_vars: Dict[str, Any]) -> None:
        analysis_sandbox._handle_context_write(self._ctx, directive, local_vars)

    def _handle_context_update(self, directive: str) -> None:
        analysis_sandbox._handle_context_update(self._ctx, directive)

    @staticmethod
    def _parse_plan_step(step_text: str) -> Optional[Dict[str, Any]]:
        return analysis_sandbox._parse_plan_step(step_text)
