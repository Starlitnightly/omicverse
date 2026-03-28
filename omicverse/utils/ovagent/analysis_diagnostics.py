"""Diagnostic and repair helpers for agent-generated code.

Extracted from ``analysis_executor.py`` during Phase 4 decomposition.
Contains prerequisite checking, pattern-based error recovery (Stage A),
package management, LLM-based error diagnosis, and output validation.

All functions receive an explicit ``ctx`` (AgentContext) parameter where
state access is needed, rather than relying on ``self``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..._registry import _global_registry

if TYPE_CHECKING:
    from .protocol import AgentContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Package alias map (module-level constant)
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


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

def check_code_prerequisites(code: str, adata: Any) -> str:
    """Check whether *adata* satisfies the prerequisites for functions in *code*."""
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


# ---------------------------------------------------------------------------
# Pattern-based error recovery (Stage A)
# ---------------------------------------------------------------------------

def apply_execution_error_fix(
    ctx: "AgentContext", code: str, error_msg: str,
) -> Optional[str]:
    """Apply heuristic regex fixes for common execution errors."""
    error_str = str(error_msg).lower()

    # Fix 0: Missing package -> auto-install and retry same code
    cfg = ctx._config
    auto_install = cfg.execution.auto_install_packages if cfg else True
    if auto_install and ("no module named" in error_str or "modulenotfounderror" in error_str):
        pkg = extract_package_name(error_msg)
        if pkg and auto_install_package(ctx, pkg):
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


# ---------------------------------------------------------------------------
# Package management
# ---------------------------------------------------------------------------

_VALID_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")


def _is_valid_package_name(name: str) -> bool:
    """Check that *name* looks like a legitimate Python/pip package name.

    Rejects names containing shell metacharacters, path separators, flags
    (leading ``-``), or whitespace to prevent command-injection via
    crafted error messages.
    """
    if not name or len(name) > 128:
        return False
    return _VALID_PACKAGE_NAME_RE.match(name) is not None


def extract_package_name(error_msg: str) -> Optional[str]:
    """Extract the missing package name from a ModuleNotFoundError message."""
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(error_msg))
    if m:
        candidate = m.group(1).split(".")[0]
        if _is_valid_package_name(candidate):
            return candidate
        logger.warning("Extracted package name %r failed validation — skipping", candidate)
    return None


def auto_install_package(ctx: "AgentContext", package_name: str) -> bool:
    """Auto-install a missing package via pip if allowed by config."""
    import subprocess as _subprocess

    if not _is_valid_package_name(package_name):
        logger.warning("Package name %r failed validation - skipping auto-install", package_name)
        return False

    cfg = ctx._config
    _ALWAYS_BLOCKED = {"os", "sys", "subprocess", "shutil", "signal", "ctypes"}
    blocklist = set(_ALWAYS_BLOCKED)
    if cfg and hasattr(cfg, "execution"):
        blocklist = _ALWAYS_BLOCKED | set(cfg.execution.package_blocklist)

    if package_name in blocklist:
        logger.warning("Package %r is on the blocklist - skipping auto-install", package_name)
        return False

    pip_name = _PACKAGE_ALIASES.get(package_name, package_name)
    if not _is_valid_package_name(pip_name):
        logger.warning("Resolved pip name %r failed validation - skipping auto-install", pip_name)
        return False
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


# ---------------------------------------------------------------------------
# LLM-based error diagnosis
# ---------------------------------------------------------------------------

async def diagnose_error_with_llm(
    ctx: "AgentContext",
    code: str,
    error_msg: str,
    traceback_str: str,
    adata: Any,
    *,
    extract_code_fn: Optional[Any] = None,
) -> Optional[str]:
    """Ask the LLM to diagnose and fix an execution error."""
    if ctx._llm is None:
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

    if extract_code_fn is None:
        from .codegen_pipeline import CodegenPipeline
        extract_code_fn = CodegenPipeline(ctx).extract_python_code

    try:
        print("   🔬 LLM diagnosing execution error...")
        response = await ctx._llm.run(diagnosis_prompt)
        diagnosed_code = extract_code_fn(response)
        if diagnosed_code and diagnosed_code.strip():
            print(f"   💡 LLM generated fix ({len(diagnosed_code)} chars)")
            return diagnosed_code
    except Exception as exc:
        logger.warning("LLM error diagnosis failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Output validation and completion
# ---------------------------------------------------------------------------

def validate_outputs(code: str, output_dir: Optional[str] = None) -> List[str]:
    """Check whether files written by *code* actually exist on disk."""
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
    ctx: "AgentContext",
    original_code: str,
    missing_files: List[str],
    adata: Any,
    request: str,
    *,
    extract_code_fn: Optional[Any] = None,
) -> Optional[str]:
    """Generate a completion snippet to create missing output files."""
    if ctx._llm is None or not missing_files:
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
    if extract_code_fn is None:
        from .codegen_pipeline import CodegenPipeline
        extract_code_fn = CodegenPipeline(ctx).extract_python_code

    try:
        response = await ctx._llm.run(prompt)
        return extract_code_fn(response)
    except Exception as exc:
        logger.warning("Completion code generation failed: %s", exc)
        return None
