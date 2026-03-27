"""Proactive code transformer for LLM-generated code.

Extracted from ``analysis_executor.py`` during Phase 4 decomposition.
``ProactiveCodeTransformer`` applies regex-based fixups to agent-generated
code *before* sandbox execution — preventing common errors such as
matplotlib GUI hangs, in-place assignment bugs, and f-string scoping
issues.

The class is stateless (no ``AgentContext`` dependency) and can be
constructed with zero arguments.
"""

from __future__ import annotations

import ast
import logging
import re

logger = logging.getLogger(__name__)


class ProactiveCodeTransformer:
    """Transform LLM-generated code to prevent common errors before execution."""

    INPLACE_FUNCTIONS = {
        "pca", "scale", "neighbors", "leiden", "umap", "tsne", "sude",
        "scrublet", "mde", "louvain", "phate",
    }

    KWARG_RENAMES = {
        (r"mu(?:on)?\.atac\.tl\.lsi", "n_components"): "n_comps",
    }

    # Prepended to all agent-generated code to prevent GUI windows
    _MATPLOTLIB_PREAMBLE = (
        "import matplotlib as _mpl; _mpl.use('Agg')\n"
        "import matplotlib.pyplot as _plt; _plt.ioff()\n"
    )

    def transform(self, code: str) -> str:
        try:
            code = self._prepend_matplotlib_noninteractive(code)
            code = self._fix_show_true(code)
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

    def _prepend_matplotlib_noninteractive(self, code: str) -> str:
        """Ensure matplotlib uses non-interactive backend to prevent GUI hang."""
        if "_mpl.use('Agg')" in code:
            return code
        return self._MATPLOTLIB_PREAMBLE + code

    def _fix_show_true(self, code: str) -> str:
        """Replace show=True with show=False in plotting calls to avoid blocking."""
        return re.sub(r'\bshow\s*=\s*True\b', 'show=False', code)

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
                except Exception as e:
                    logger.debug("ProactiveCodeTransformer: f-string conversion failed (%s), keeping original line", e)
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
