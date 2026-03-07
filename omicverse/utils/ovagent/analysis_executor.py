"""Code execution engine for OVAgent — sandbox, code transform, error recovery.

Extracted from ``smart_agent.py``.  ``AnalysisExecutor`` wraps an
:class:`AgentContext` and provides:

* ``ProactiveCodeTransformer`` — regex-based LLM code fix-ups
* ``execute_generated_code`` — sandbox execution with security scan
* ``build_sandbox_globals`` — restricted namespace construction
* ``apply_execution_error_fix`` — pattern-based error recovery (Stage A)
* Helper methods for auto-install, LLM diagnosis, output validation, etc.
"""

from __future__ import annotations

import ast
import builtins
import json
import logging
import os
import re
import sys
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..agent_errors import SandboxDeniedError, SecurityViolationError
from ..agent_sandbox import ApprovalMode, SafeOsProxy
from ..agent_config import SandboxFallbackPolicy
from ..agent_reporter import EventLevel
from ..._registry import _global_registry

if TYPE_CHECKING:
    from .protocol import AgentContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProactiveCodeTransformer (standalone — no agent state dependency)
# ---------------------------------------------------------------------------

class ProactiveCodeTransformer:
    """Transform LLM-generated code to prevent common errors before execution."""

    INPLACE_FUNCTIONS = {
        "pca", "scale", "neighbors", "leiden", "umap", "tsne", "sude",
        "scrublet", "mde", "louvain", "phate",
    }

    KWARG_RENAMES = {
        (r"mu(?:on)?\.atac\.tl\.lsi", "n_components"): "n_comps",
    }

    def transform(self, code: str) -> str:
        try:
            code = self._fix_inplace_assignments_regex(code)
            code = self._fix_fstring_print_regex(code)
            code = self._fix_cat_accessor_regex(code)
            code = self._fix_kwarg_renames(code)
            ast.parse(code)
            return code
        except SyntaxError:
            logger.debug("ProactiveCodeTransformer: transformation produced invalid syntax, returning original")
            return code
        except Exception as e:
            logger.debug("ProactiveCodeTransformer: unexpected error %s, returning original", e)
            return code

    def _fix_inplace_assignments_regex(self, code: str) -> str:
        inplace_pattern = "|".join(self.INPLACE_FUNCTIONS)
        pattern = r"adata\s*=\s*(ov\.pp\.(?:" + inplace_pattern + r")\s*\([^)]*\))"
        fixed = re.sub(pattern, r"\1", code)
        if fixed != code:
            logger.debug("ProactiveCodeTransformer: fixed in-place function assignment")
        return fixed

    def _fix_fstring_print_regex(self, code: str) -> str:
        lines = code.split("\n")
        fixed_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('print(f"') or stripped.startswith("print(f'"):
                try:
                    fixed_line = self._convert_fstring_line(line)
                    if fixed_line != line:
                        logger.debug("ProactiveCodeTransformer: converted f-string in print")
                    fixed_lines.append(fixed_line)
                except Exception:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        return "\n".join(fixed_lines)

    def _convert_fstring_line(self, line: str) -> str:
        indent = len(line) - len(line.lstrip())
        indent_str = line[:indent]
        content = line.strip()
        match = re.match(r"print\(f([\"'])(.*)\1\)", content)
        if not match:
            return line
        fstring_content = match.group(2)
        parts: list[str] = []
        last_end = 0
        for m in re.finditer(r"\{([^}:]+)(?::[^}]*)?\}", fstring_content):
            if m.start() > last_end:
                text_part = fstring_content[last_end : m.start()]
                if text_part:
                    parts.append(f'"{text_part}"')
            var_name = m.group(1).strip()
            parts.append(f"str({var_name})")
            last_end = m.end()
        if last_end < len(fstring_content):
            remaining = fstring_content[last_end:]
            if remaining:
                parts.append(f'"{remaining}"')
        if not parts:
            return line
        concatenated = " + ".join(parts)
        return f"{indent_str}print({concatenated})"

    def _fix_cat_accessor_regex(self, code: str) -> str:
        return re.sub(r"\.cat\.categories", ".value_counts().index.tolist()", code)

    def _fix_kwarg_renames(self, code: str) -> str:
        for (func_pat, old_kw), new_kw in self.KWARG_RENAMES.items():
            pattern = rf"({func_pat}\s*\([^)]*)\b{old_kw}\s*="
            replacement = rf"\1{new_kw}="
            new_code = re.sub(pattern, replacement, code, flags=re.DOTALL)
            if new_code != code:
                logger.debug("ProactiveCodeTransformer: renamed kwarg %s -> %s", old_kw, new_kw)
                code = new_code
        return code


# ---------------------------------------------------------------------------
# AnalysisExecutor
# ---------------------------------------------------------------------------

_PACKAGE_ALIASES: Dict[str, str] = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "PIL": "Pillow",
    "Bio": "biopython",
    "umap": "umap-learn",
    "leidenalg": "leidenalg",
    "louvain": "louvain",
    "harmonypy": "harmonypy",
    "scanorama": "scanorama",
    "scvi": "scvi-tools",
    "scarches": "scarches",
    "bbknn": "bbknn",
    "scrublet": "scrublet",
    "magic": "magic-impute",
}


class AnalysisExecutor:
    """Sandbox execution engine for OVAgent-generated code."""

    def __init__(self, ctx: "AgentContext") -> None:
        self._ctx = ctx

    # -- prerequisite checks ------------------------------------------------

    def check_code_prerequisites(self, code: str, adata: Any) -> str:
        warnings_list: list[str] = []
        func_patterns = {
            "ov.pp.pca": "pca",
            "ov.pp.scale": "scale",
            "ov.pp.neighbors": "neighbors",
            "ov.pp.umap": "umap",
            "ov.pp.tsne": "tsne",
            "ov.pp.leiden": "leiden",
            "ov.pp.louvain": "louvain",
            "ov.pp.sude": "sude",
            "ov.pp.mde": "mde",
            "ov.single.leiden": "leiden",
            "ov.single.louvain": "louvain",
        }
        for pattern, func_name in func_patterns.items():
            if pattern in code:
                try:
                    result = _global_registry.check_prerequisites(func_name, adata)
                    if not result["satisfied"]:
                        missing = ", ".join(result["missing_structures"][:3])
                        rec = result["recommendation"]
                        warnings_list.append(f"{func_name}: missing {missing}. {rec}")
                except Exception:
                    pass
        return "; ".join(warnings_list)

    # -- pattern-based error recovery (Stage A) -----------------------------

    def apply_execution_error_fix(self, code: str, error_msg: str) -> Optional[str]:
        error_str = str(error_msg).lower()

        # Fix 0: Missing package -> auto-install and retry same code
        cfg = self._ctx._config
        auto_install = cfg.execution.auto_install_packages if cfg else True
        if auto_install and ("no module named" in error_str or "modulenotfounderror" in error_str):
            pkg = self.extract_package_name(error_msg)
            if pkg and self.auto_install_package(pkg):
                return code

        # Fix 1: .dtype -> .dtypes for DataFrames
        if "has no attribute 'dtype'" in error_str or "'dtype'" in error_str:
            fixed_code = re.sub(r"\.dtype\b", ".dtypes", code)
            if fixed_code != code:
                logger.debug("Applied fix: .dtype -> .dtypes")
                return fixed_code

        # Fix 2: seurat_v3 LOESS error -> seurat fallback
        if "extrapolation" in error_str or "loess" in error_str or "blending" in error_str:
            fixed_code = code.replace("flavor='seurat_v3'", "flavor='seurat'")
            fixed_code = fixed_code.replace('flavor="seurat_v3"', 'flavor="seurat"')
            if fixed_code != code:
                logger.debug("Applied fix: seurat_v3 -> seurat for HVG")
                return fixed_code

        # Fix 3: Categorical batch column errors
        if (
            "cannot setitem on a categorical" in error_str
            or "new category" in error_str
            or ("nan" in error_str and ("batch" in error_str or "categorical" in error_str))
        ):
            prep_code = (
                "import pandas as pd\n"
                "if 'batch' in adata.obs.columns:\n"
                "    if pd.api.types.is_categorical_dtype(adata.obs['batch']):\n"
                "        adata.obs['batch'] = adata.obs['batch'].astype(str)\n"
                "    adata.obs['batch'] = adata.obs['batch'].fillna('unknown')\n"
                "    adata.obs['batch'] = adata.obs['batch'].astype('category')\n"
            )
            logger.debug("Applied fix: Categorical batch column handling")
            return prep_code + "\n" + code

        # Fix 4: In-place function assignment error
        if "'nonetype' object has no attribute" in error_str:
            inplace_funcs = ["pca", "scale", "neighbors", "leiden", "umap", "tsne", "sude", "scrublet", "mde"]
            pattern = r"adata\s*=\s*ov\.pp\.(" + "|".join(inplace_funcs) + r")\s*\("
            if re.search(pattern, code):
                fixed_code = re.sub(
                    r"adata\s*=\s*(ov\.pp\.(?:" + "|".join(inplace_funcs) + r")\s*\([^)]*\))",
                    r"\1",
                    code,
                )
                if fixed_code != code:
                    logger.debug("Applied fix: Removed assignment from in-place function call")
                    return fixed_code

        return None

    # -- package management -------------------------------------------------

    @staticmethod
    def extract_package_name(error_msg: str) -> Optional[str]:
        m = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(error_msg))
        if m:
            return m.group(1).split(".")[0]
        return None

    def auto_install_package(self, package_name: str) -> bool:
        import subprocess as _subprocess

        cfg = self._ctx._config
        blocklist = ["os", "sys", "subprocess", "shutil", "signal", "ctypes"]
        if cfg and hasattr(cfg, "execution"):
            blocklist = cfg.execution.package_blocklist

        if package_name in blocklist:
            logger.warning("Package %r is on the blocklist - skipping auto-install", package_name)
            return False

        pip_name = _PACKAGE_ALIASES.get(package_name, package_name)
        print(f"   📦 Auto-installing missing package: {pip_name}")

        try:
            result = _subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                print(f"   ✅ Successfully installed {pip_name}")
                for key in list(sys.modules.keys()):
                    if key == package_name or key.startswith(package_name + "."):
                        del sys.modules[key]
                return True
            else:
                print(f"   ❌ pip install failed: {result.stderr[:200]}")
                return False
        except Exception as exc:
            logger.warning("Auto-install of %r failed: %s", pip_name, exc)
            return False

    # -- LLM-based error diagnosis ------------------------------------------

    async def diagnose_error_with_llm(
        self,
        code: str,
        error_msg: str,
        traceback_str: str,
        adata: Any,
    ) -> Optional[str]:
        if self._ctx._llm is None:
            return None

        dataset_summary = ""
        if adata is not None and hasattr(adata, "shape"):
            dataset_summary = f"Dataset: {adata.shape[0]} cells x {adata.shape[1]} genes"
            if hasattr(adata, "obs") and hasattr(adata.obs, "columns"):
                cols = list(adata.obs.columns[:20])
                dataset_summary += f"\nobs columns: {cols}"

        diagnosis_prompt = (
            "The following OmicVerse agent-generated Python code failed during execution.\n\n"
            f"--- CODE ---\n{code}\n\n"
            f"--- ERROR ---\n{error_msg}\n\n"
            f"--- TRACEBACK ---\n{traceback_str[-1500:]}\n\n"
            f"--- DATASET ---\n{dataset_summary}\n\n"
            "Your task:\n"
            "1. Diagnose the root cause of the error.\n"
            "2. Generate a CORRECTED version of the full code that fixes the issue.\n"
            "3. Wrap the corrected code in ```python ... ``` markers.\n\n"
            "Important rules:\n"
            "- Fix ONLY the error. Do not change logic that already works.\n"
            "- If a variable is undefined, define it or remove the reference.\n"
            "- If a module is unavailable, use an alternative or add a try/except.\n"
            "- Preserve all file output operations (savefig, to_csv, json.dump, etc.).\n"
        )

        try:
            print("   🔬 LLM diagnosing execution error...")
            response = await self._ctx._llm.run(diagnosis_prompt)
            diagnosed_code = self._ctx._extract_python_code(response)
            if diagnosed_code and diagnosed_code.strip():
                print(f"   💡 LLM generated fix ({len(diagnosed_code)} chars)")
                return diagnosed_code
        except Exception as exc:
            logger.warning("LLM error diagnosis failed: %s", exc)

        return None

    # -- output validation --------------------------------------------------

    def validate_outputs(self, code: str, output_dir: Optional[str] = None) -> List[str]:
        missing: List[str] = []
        file_patterns = [
            r'\.savefig\s*\(\s*["\']([^"\']+)["\']',
            r'\.to_csv\s*\(\s*["\']([^"\']+)["\']',
            r'\.write_h5ad\s*\(\s*["\']([^"\']+)["\']',
            r'\.write\s*\(\s*["\']([^"\']+\.h5ad)["\']',
            r'open\s*\(\s*["\']([^"\']+\.json)["\']',
            r'\.to_excel\s*\(\s*["\']([^"\']+)["\']',
            r'\.to_parquet\s*\(\s*["\']([^"\']+)["\']',
        ]
        for pattern in file_patterns:
            for m in re.finditer(pattern, code):
                fpath = m.group(1)
                if output_dir and not os.path.isabs(fpath):
                    fpath = os.path.join(output_dir, fpath)
                if not os.path.exists(fpath):
                    missing.append(fpath)
        return missing

    async def generate_completion_code(
        self,
        original_code: str,
        missing_files: List[str],
        adata: Any,
        request: str,
    ) -> Optional[str]:
        if self._ctx._llm is None or not missing_files:
            return None

        prompt = (
            "The following code was executed successfully but some output files were NOT created:\n\n"
            f"--- ORIGINAL CODE ---\n{original_code}\n\n"
            f"--- MISSING FILES ---\n{json.dumps(missing_files)}\n\n"
            f"--- ORIGINAL REQUEST ---\n{request}\n\n"
            "Generate a SHORT Python snippet that creates ONLY the missing files listed above.\n"
            "- The `adata` variable is already available with the processed data.\n"
            "- Reuse any variables/imports from the original code.\n"
            "- Wrap the code in ```python ... ``` markers.\n"
        )
        try:
            response = await self._ctx._llm.run(prompt)
            return self._ctx._extract_python_code(response)
        except Exception as exc:
            logger.warning("Completion code generation failed: %s", exc)
            return None

    # -- approval gate ------------------------------------------------------

    def request_approval(self, code: str, violations: list) -> bool:
        from ..harness import make_turn_id

        if self._ctx._approval_handler is not None:
            trace = self._ctx._last_run_trace
            payload = {
                "request_id": make_turn_id(),
                "title": "Execution approval required",
                "message": "Generated code requires approval before execution.",
                "code": code,
                "violations": [v.__dict__ if hasattr(v, "__dict__") else str(v) for v in violations],
                "trace_id": getattr(trace, "trace_id", ""),
                "session_id": self._ctx._get_harness_session_id(),
                "approval_mode": self._ctx._security_config.approval_mode.value,
            }
            try:
                return bool(self._ctx._approval_handler(payload))
            except Exception as exc:
                logger.warning("Approval handler failed, denying execution: %s", exc)
                return False

        print("\n" + "=" * 60)
        print("GENERATED CODE REVIEW")
        print("=" * 60)
        display = code if len(code) < 2000 else code[:2000] + "\n... (truncated)"
        for i, line in enumerate(display.split("\n"), 1):
            print(f"  {i:3d} | {line}")
        if violations:
            print()
            print(self._ctx._security_scanner.format_report(violations))
        print("=" * 60)
        try:
            response = input("Execute this code? [y/N]: ").strip().lower()
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    # -- main execution entry point -----------------------------------------

    def execute_generated_code(
        self, code: str, adata: Any, capture_stdout: bool = False
    ) -> Any:
        # --- Pre-execution security scan ---
        try:
            violations = self._ctx._security_scanner.scan(code)
        except SyntaxError:
            violations = []

        if violations:
            report = self._ctx._security_scanner.format_report(violations)
            logger.warning("Security scan report:\n%s", report)
            if self._ctx._security_scanner.has_critical(violations):
                raise SecurityViolationError(
                    f"Code blocked by security scanner:\n{report}",
                    violations=violations,
                )

        # --- Approval gate ---
        approval_mode = self._ctx._security_config.approval_mode
        if approval_mode == ApprovalMode.ALWAYS:
            if not self.request_approval(code, violations):
                raise SecurityViolationError("User declined code execution.")
        elif approval_mode == ApprovalMode.ON_VIOLATION and violations:
            if not self.request_approval(code, violations):
                raise SecurityViolationError(
                    "User declined code execution after security warnings."
                )

        # Use notebook execution if enabled
        if self._ctx.use_notebook_execution and self._ctx._notebook_executor is not None:
            try:
                result_adata = self._ctx._notebook_executor.execute(code, adata)
                if hasattr(result_adata, "uns"):
                    result_adata.uns["_ovagent_session"] = {
                        "session_id": self._ctx._notebook_executor.current_session["session_id"],
                        "notebook_path": str(self._ctx._notebook_executor.current_session["notebook_path"]),
                        "prompt_number": self._ctx._notebook_executor.session_prompt_count,
                    }
                if self._ctx.enable_filesystem_context and self._ctx._filesystem_context:
                    self.process_context_directives(code, {})
                return result_adata

            except Exception as e:
                policy = getattr(getattr(self._ctx, "_config", None), "execution", None)
                fb = getattr(policy, "sandbox_fallback_policy", SandboxFallbackPolicy.WARN_AND_FALLBACK)
                if fb == SandboxFallbackPolicy.RAISE:
                    raise SandboxDeniedError(
                        f"Notebook execution failed and fallback is disabled: {e}"
                    ) from e
                elif fb == SandboxFallbackPolicy.WARN_AND_FALLBACK:
                    if hasattr(self._ctx, "_emit"):
                        self._ctx._emit(EventLevel.WARNING, f"Session execution failed: {e}", "execution")
                        self._ctx._emit(EventLevel.INFO, "Falling back to in-process execution...", "execution")
                    else:
                        print(f"\u26a0\ufe0f  Session execution failed: {e}")
                        print("   Falling back to in-process execution...")

        # Legacy in-process execution
        compiled = compile(code, "<omicverse-agent>", "exec")
        sandbox_globals = self.build_sandbox_globals()
        sandbox_locals: Dict[str, Any] = {"adata": adata}
        try:
            if hasattr(adata, "obs_names"):
                sandbox_locals.setdefault("obs_names", adata.obs_names)
            if hasattr(adata, "var_names"):
                sandbox_locals.setdefault("var_names", adata.var_names)
        except Exception:
            pass

        _is_mudata = type(adata).__name__ == "MuData"

        # Normalize HVG column naming
        try:
            if not _is_mudata and hasattr(adata, "var") and adata.var is not None:
                if "highly_variable" not in adata.var.columns and "highly_variable_features" in adata.var.columns:
                    adata.var["highly_variable"] = adata.var["highly_variable_features"]
            if hasattr(adata, "uns"):
                adata.uns.setdefault("initial_cells", getattr(adata, "n_obs", None) or getattr(adata, "shape", [None])[0])
                adata.uns.setdefault("initial_genes", getattr(adata, "n_vars", None) or getattr(adata, "shape", [None, None])[1])
                adata.uns.setdefault("omicverse_qc_original_cells", getattr(adata, "n_obs", None))
            if not _is_mudata and getattr(adata, "raw", None) is None:
                try:
                    adata.raw = adata
                except Exception:
                    pass
            if not _is_mudata and hasattr(adata, "layers") and getattr(adata, "layers", None) is not None and "scaled" not in adata.layers:
                try:
                    adata.layers["scaled"] = adata.X.copy()
                except Exception:
                    pass
            try:
                import pandas as _pd
                if not _is_mudata and hasattr(adata, "obs"):
                    for col in adata.obs.columns:
                        col_data = adata.obs[col]
                        if _pd.api.types.is_numeric_dtype(col_data):
                            if col_data.isna().any():
                                adata.obs[col] = col_data.fillna(0)
                    if "total_counts" not in adata.obs.columns:
                        try:
                            import numpy as _np
                            data_matrix = None
                            if hasattr(adata, "layers") and getattr(adata, "layers", None) is not None and "counts" in adata.layers:
                                data_matrix = adata.layers["counts"]
                            else:
                                data_matrix = adata.X
                            sums = _np.asarray(data_matrix.sum(axis=1)).ravel()
                            adata.obs["total_counts"] = sums
                        except Exception:
                            adata.obs["total_counts"] = 0
                    if "n_counts" not in adata.obs.columns:
                        adata.obs["n_counts"] = adata.obs.get("total_counts", 0)
                    if "n_genes_by_counts" not in adata.obs.columns:
                        try:
                            import numpy as _np
                            data_matrix = None
                            if hasattr(adata, "layers") and getattr(adata, "layers", None) is not None and "counts" in adata.layers:
                                data_matrix = adata.layers["counts"]
                            else:
                                data_matrix = adata.X
                            adata.obs["n_genes_by_counts"] = _np.asarray((data_matrix > 0).sum(axis=1)).ravel()
                        except Exception:
                            adata.obs["n_genes_by_counts"] = 0
                    if "pct_counts_mito" not in adata.obs.columns and "pct_counts_mt" in adata.obs.columns:
                        adata.obs["pct_counts_mito"] = adata.obs["pct_counts_mt"]
                    elif "pct_counts_mito" not in adata.obs.columns:
                        adata.obs["pct_counts_mito"] = 0
                if hasattr(adata, "var") and adata.var is not None:
                    if "mito" not in adata.var.columns and "mt" in adata.var.columns:
                        adata.var["mito"] = adata.var["mt"]
                    elif "mito" not in adata.var.columns:
                        adata.var["mito"] = False
                try:
                    if not getattr(_pd.cut, "_ov_wrapped", False):
                        _orig_cut = _pd.cut

                        def _safe_cut(*args, **kwargs):
                            kwargs.setdefault("duplicates", "drop")
                            return _orig_cut(*args, **kwargs)

                        _safe_cut._ov_wrapped = True  # type: ignore[attr-defined]
                        _pd.cut = _safe_cut
                except Exception:
                    pass
            except Exception:
                pass
            try:
                Path("genesets").mkdir(exist_ok=True)
            except Exception:
                pass
        except Exception as exc:
            warnings.warn(
                f"Failed to normalize HVG columns for agent execution: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

        warnings.warn(
            "Executing agent-generated code. Ensure the input model and prompts come from a trusted source.",
            RuntimeWarning,
            stacklevel=2,
        )

        import io as _io

        stdout_buffer = _io.StringIO() if capture_stdout else None

        with self._ctx._temporary_api_keys():
            if capture_stdout:
                old_stdout = sys.stdout
                sys.stdout = stdout_buffer  # type: ignore[assignment]
                try:
                    exec(compiled, sandbox_globals, sandbox_locals)
                finally:
                    sys.stdout = old_stdout
            else:
                exec(compiled, sandbox_globals, sandbox_locals)

        result_adata = sandbox_locals.get("adata", adata)
        self.normalize_doublet_obs(result_adata)

        if self._ctx.enable_filesystem_context and self._ctx._filesystem_context:
            self.process_context_directives(code, sandbox_locals)

        if capture_stdout:
            stdout_text = stdout_buffer.getvalue()  # type: ignore[union-attr]
            return {"adata": result_adata, "stdout": stdout_text}

        return result_adata

    # -- doublet harmonization ----------------------------------------------

    @staticmethod
    def normalize_doublet_obs(adata: Any) -> None:
        try:
            obs = getattr(adata, "obs", None)
            if obs is None:
                return
            if "predicted_doublet" not in obs.columns and "predicted_doublets" in obs.columns:
                obs["predicted_doublet"] = obs["predicted_doublets"]
            col = None
            for c in ("predicted_doublet", "predicted_doublets", "doublet", "doublets"):
                if c in obs.columns:
                    col = c
                    break
            if col is None:
                return
            try:
                rate = float(obs[col].mean()) * 100.0
                if hasattr(adata, "uns"):
                    adata.uns.setdefault("doublet_summary", {})["rate_percent"] = rate
            except Exception:
                pass
        except Exception:
            return

    # -- context directives -------------------------------------------------

    def process_context_directives(self, code: str, local_vars: Dict[str, Any]) -> None:
        fc = self._ctx._filesystem_context
        if not fc:
            return
        try:
            lines = code.split("\n")
            collecting_plan = False
            plan_steps: list[dict[str, Any]] = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# CONTEXT_WRITE:"):
                    self._handle_context_write(stripped, local_vars)
                elif stripped.startswith("# CONTEXT_PLAN:"):
                    collecting_plan = True
                    plan_steps = []
                elif collecting_plan:
                    if stripped.startswith("# - "):
                        step_info = self._parse_plan_step(stripped[4:])
                        if step_info:
                            plan_steps.append(step_info)
                    elif stripped.startswith("#"):
                        if not stripped.startswith("# CONTEXT_"):
                            continue
                        else:
                            if plan_steps:
                                fc.write_plan(plan_steps)
                            collecting_plan = False
                            plan_steps = []
                    else:
                        if plan_steps:
                            fc.write_plan(plan_steps)
                        collecting_plan = False
                        plan_steps = []
                elif stripped.startswith("# CONTEXT_UPDATE:"):
                    self._handle_context_update(stripped)

            if collecting_plan and plan_steps:
                fc.write_plan(plan_steps)

        except Exception as e:
            logger.debug("Error processing context directives: %s", e)

    def _handle_context_write(self, directive: str, local_vars: Dict[str, Any]) -> None:
        fc = self._ctx._filesystem_context
        if not fc:
            return
        try:
            content = directive.replace("# CONTEXT_WRITE:", "").strip()
            if " -> " in content:
                key, value_expr = content.split(" -> ", 1)
                key = key.strip()
                value_expr = value_expr.strip()
                try:
                    if value_expr in local_vars:
                        value = local_vars[value_expr]
                    else:
                        value = eval(value_expr, {"__builtins__": {}}, local_vars)
                except Exception:
                    value = value_expr
                category = "notes"
                if any(kw in key.lower() for kw in ["result", "stats", "metrics", "output"]):
                    category = "results"
                elif any(kw in key.lower() for kw in ["decision", "choice", "why"]):
                    category = "decisions"
                elif any(kw in key.lower() for kw in ["error", "fail", "exception"]):
                    category = "errors"
                fc.write_note(key, value, category)
                logger.debug("Context write: %s -> %s", key, category)
        except Exception as e:
            logger.debug("Failed to process CONTEXT_WRITE: %s", e)

    def _handle_context_update(self, directive: str) -> None:
        fc = self._ctx._filesystem_context
        if not fc:
            return
        try:
            content = directive.replace("# CONTEXT_UPDATE:", "").strip()
            parts: dict[str, str] = {}
            for part in content.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    parts[k.strip()] = v.strip().strip('"').strip("'")
            step = int(parts.get("step", "0"))
            status = parts.get("status", "completed")
            result = parts.get("result")
            fc.update_plan_step(step, status, result)
            logger.debug("Context update: step %d -> %s", step, status)
        except Exception as e:
            logger.debug("Failed to process CONTEXT_UPDATE: %s", e)

    @staticmethod
    def _parse_plan_step(step_text: str) -> Optional[Dict[str, Any]]:
        try:
            status = "pending"
            if "[" in step_text and "]" in step_text:
                status_start = step_text.rfind("[")
                status_end = step_text.rfind("]")
                status = step_text[status_start + 1 : status_end].strip().lower()
                step_text = step_text[:status_start].strip()
            if step_text.lower().startswith("step "):
                colon_idx = step_text.find(":")
                if colon_idx > 0:
                    step_text = step_text[colon_idx + 1 :].strip()
            return {"description": step_text, "status": status}
        except Exception:
            return None

    # -- sandbox globals ----------------------------------------------------

    def build_sandbox_globals(self) -> Dict[str, Any]:
        allowed_builtins = [
            "abs", "all", "any", "bool", "dict", "enumerate", "Exception",
            "float", "int", "isinstance", "iter", "len", "list", "map",
            "max", "min", "next", "pow", "print", "range", "round", "set",
            "sorted", "str", "sum", "tuple", "zip", "filter", "type",
            "hasattr", "getattr", "setattr",
            "ImportError", "AttributeError", "IndexError",
            "FileNotFoundError", "OSError", "Exception", "ValueError",
            "RuntimeError", "TypeError", "KeyError", "AssertionError",
        ]
        if not self._ctx._security_config.restrict_introspection:
            allowed_builtins.extend(["locals", "globals"])

        safe_builtins = {name: getattr(builtins, name) for name in allowed_builtins if hasattr(builtins, name)}
        allowed_modules: Dict[str, Any] = {}
        deny_roots = {
            "subprocess", "socket", "ssl", "urllib", "http",
            "ftplib", "smtplib", "telnetlib", "paramiko", "requests",
            "importlib", "ctypes", "multiprocessing",
        }
        deny_roots |= self._ctx._security_config.extra_blocked_modules
        core_modules = (
            "omicverse", "numpy", "pandas", "scanpy",
            "time", "math", "json", "re", "pathlib",
            "itertools", "functools", "collections",
            "statistics", "random", "warnings", "datetime", "typing",
        )
        skill_modules = (
            "openpyxl", "reportlab", "matplotlib", "seaborn",
            "scipy", "statsmodels", "sklearn",
        )
        for module_name in core_modules + skill_modules:
            try:
                allowed_modules[module_name] = __import__(module_name)
            except ImportError:
                warnings.warn(
                    f"Module '{module_name}' is not available inside the agent sandbox.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        allowed_modules["os"] = SafeOsProxy()

        def _apply_scvi_shims() -> None:
            try:
                import functools
                from scvi.model import MULTIVI
            except Exception:
                return
            try:
                already = getattr(MULTIVI.train, "_ovbench_patched", False)
            except Exception:
                already = False
            if already:
                return
            orig_train = MULTIVI.train

            @functools.wraps(orig_train)
            def _train_wrapper(self_m, *args, **kwargs):
                kwargs.pop("early_stopping_patience", None)
                return orig_train(self_m, *args, **kwargs)

            _train_wrapper._ovbench_patched = True  # type: ignore[attr-defined]
            MULTIVI.train = _train_wrapper  # type: ignore[assignment]

        def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
            root_name = name.split(".")[0]
            if root_name in deny_roots:
                caller_pkg = (globals or {}).get("__package__", "") or ""
                if not caller_pkg.startswith("omicverse"):
                    raise ImportError(
                        f"Module '{name}' is blocked inside the OmicVerse agent sandbox."
                    )
            if root_name not in allowed_modules:
                allowed_modules[root_name] = __import__(root_name)
            if root_name == "scvi":
                _apply_scvi_shims()
            root_module = allowed_modules[root_name]
            if not fromlist:
                return root_module
            if name == root_name:
                return root_module
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = limited_import

        sandbox_globals: Dict[str, Any] = {"__builtins__": safe_builtins}
        sandbox_globals.update(allowed_modules)
        if "pandas" in allowed_modules:
            sandbox_globals.setdefault("pd", allowed_modules["pandas"])
        if "numpy" in allowed_modules:
            sandbox_globals.setdefault("np", allowed_modules["numpy"])
        if "scanpy" in allowed_modules:
            sandbox_globals.setdefault("sc", allowed_modules["scanpy"])
        if "omicverse" in allowed_modules:
            sandbox_globals.setdefault("ov", allowed_modules["omicverse"])
        return sandbox_globals
