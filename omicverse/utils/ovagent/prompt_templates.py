"""Template-based prompt composition engine for OVAgent.

Implements the ``PromptComposer`` protocol from ``contracts.py``,
replacing monolithic string concatenation with explicit, layered
prompt assembly.  Each logical section of the system prompt is a
named block constant; the ``PromptTemplateEngine`` composes them
into a final prompt string in priority order.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .contracts import PromptLayer, PromptLayerKind


# =====================================================================
# Prompt block constants — canonical content for each logical section
# =====================================================================

IDENTITY_BLOCK = (
    "You are OmicVerse Agent, an expert bioinformatics assistant that processes "
    "single-cell, bulk RNA-seq, and spatial transcriptomics data."
)

TOOL_CATALOG_BLOCK = (
    "You have OmicVerse data tools plus a Claude-style coding tool catalog.\n"
    "Core Claude tools are ToolSearch, Bash, and Read. Deferred Claude tools "
    "must be loaded through ToolSearch before use. High-risk tools require approval.\n\n"
    "Use the legacy OmicVerse tools for dataset inspection and execution, and "
    "use Claude-style tools for files, tasks, planning, worktrees, and questions.\n\n"
    "Do not narrate future actions without taking them in the same turn. "
    "If a tool is needed, call it now instead of saying that you will call it later."
)

CODING_FILESYSTEM_BLOCK = (
    "CODING / FILESYSTEM:\n"
    "- Use ToolSearch when you need deferred tools such as Edit, Write, Glob, "
    "Grep, NotebookEdit, Task*, or EnterWorktree\n"
    "- Use Read for local files instead of shelling out to cat/head/sed when possible\n"
    "- Use AskUserQuestion when a turn genuinely needs user clarification"
)

WEB_ACCESS_BLOCK = (
    "WEB ACCESS:\n"
    "- Use WebFetch(url) or web_fetch(url) to read any web page "
    "(GEO datasets, papers, docs)\n"
    "- Use WebSearch(query) or web_search(query) to search the internet "
    "for information\n"
    "- Use web_download(url) to download files (datasets, h5ad, csv, gz) to disk\n"
    "- When the user provides a URL, ALWAYS fetch it first instead of asking "
    "them to paste content\n"
    "- When you need background info (gene functions, diseases, methods), "
    "search the web\n"
    "- To download GEO data, fetch the GEO page first to find download links, "
    "then use web_download"
)

ANALYSIS_WORKFLOW_BLOCK = (
    "WORKFLOW:\n"
    "1. If the user provides a URL, use web_fetch to read it\n"
    "2. Use inspect_data to understand the dataset structure\n"
    "3. Use search_functions to find relevant OmicVerse functions "
    "(includes prerequisite chains and examples)\n"
    "4. For complex multi-step workflows, use search_skills for domain guidance\n"
    "5. Execute code step by step, checking results between steps\n"
    "6. Call finish() when the task is complete"
)

CODE_QUALITY_BLOCK = (
    "MANDATORY CODE QUALITY RULES:\n"
    "- NEVER use f-strings in print() - use string concatenation: "
    "print('Result: ' + str(value))\n"
    "- NEVER assign result of in-place OmicVerse functions. Call them directly:\n"
    "  ov.pp.pca(adata, n_pcs=50)   # CORRECT\n"
    "  adata = ov.pp.pca(adata)      # WRONG - returns None!\n"
    "  In-place functions: pca, scale, neighbors, leiden, umap, tsne, "
    "sude, scrublet, mde, louvain, phate\n"
    "- NEVER use .cat.categories - use .value_counts() instead\n"
    "- ALWAYS wrap HVG selection in try/except with fallback:\n"
    "  try:\n"
    "      sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)\n"
    "  except ValueError:\n"
    "      sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)\n"
    "- ALWAYS validate batch column before batch operations "
    "(check existence, fillna, astype('category'))"
)

GUIDELINES_BLOCK = (
    "Guidelines:\n"
    "- ALWAYS inspect data before writing code that depends on column names "
    "or structure\n"
    "- Execute code in logical steps, not one giant block\n"
    "- If code fails, read the error carefully, diagnose the issue, "
    "and try a different approach\n"
    "- Use run_snippet() for exploration (won't modify data)\n"
    "- Use execute_code() for actual processing (modifies data)\n"
    "- Available packages: ov (omicverse), sc (scanpy), np (numpy), pd (pandas), "
    "matplotlib, seaborn, scipy, sklearn\n"
    "- All OmicVerse functions are called via ov.* "
    "(e.g. ov.pp.preprocess, ov.pp.scale)\n"
    "- When saving files, use the output directory or current working directory"
)

DELEGATION_BLOCK = (
    "DELEGATION STRATEGY (use the delegate tool for complex tasks):\n"
    "- Agent(subagent_type='explore', ...) or delegate('explore', ...) "
    "— read-only data characterization\n"
    "- Agent(subagent_type='plan', ...) or delegate('plan', ...) "
    "— design workflow before execution\n"
    "- Agent(subagent_type='execute', ...) or delegate('execute', ...) "
    "— focused execution step\n"
    "- For simple single-step operations, execute directly without delegation\n"
    "- Subagents have their own context window (prevents context overflow)"
)

CLAW_CODE_ONLY_BLOCK = (
    "CLAW CODE-ONLY MODE:\n"
    "- Reuse the normal OmicVerse Agent workflow, tools, search_functions, "
    "and search_skills logic.\n"
    "- In this mode, execute_code captures the final Python code instead "
    "of running it.\n"
    "- Use the same planning and tool-calling behavior you would use in Jarvis.\n"
    "- When ready, call execute_code with the final OmicVerse Python snippet, "
    "then call finish.\n"
    "- Do not stop at a prose-only plan when executable code is requested."
)

# -- Subagent block constants --------------------------------------------

EXPLORE_IDENTITY_BLOCK = (
    "You are a bioinformatics data inspector for OmicVerse. "
    "Your job is to thoroughly characterize the dataset and report findings.\n\n"
    "You CANNOT modify the data. Use inspect_data and run_snippet to explore.\n"
    "Use search_functions to understand what OmicVerse functions are available."
)

EXPLORE_REPORT_BLOCK = (
    "Report:\n"
    "1. Dataset dimensions (cells x genes)\n"
    "2. Available metadata columns (obs, var) with example values\n"
    "3. Existing embeddings (obsm) and layers\n"
    "4. Data quality indicators (sparsity, mito%, batch columns, "
    "cell type annotations)\n"
    "5. Potential issues or recommendations\n\n"
    "Available packages: ov (omicverse), sc (scanpy), np (numpy), pd (pandas)\n\n"
    "Call finish() with a comprehensive summary when done."
)

PLAN_IDENTITY_BLOCK = (
    "You are a bioinformatics workflow architect for OmicVerse. "
    "Design a step-by-step execution plan. Do NOT execute code — only plan.\n\n"
    "Use search_functions to find OmicVerse functions and their prerequisites.\n"
    "Use search_skills for domain-specific workflow guidance.\n"
    "Use inspect_data and run_snippet to understand current data state."
)

PLAN_OUTPUT_BLOCK = (
    "Your plan should include:\n"
    "1. Ordered steps with specific OmicVerse function calls and parameters\n"
    "2. Quality check points between steps\n"
    "3. Fallback strategies for known failure modes\n"
    "4. Expected outputs at each step (obsm, obs columns, layers)"
)

EXECUTE_IDENTITY_BLOCK = (
    "You are a focused bioinformatics code executor for OmicVerse. "
    "Complete the specific sub-task described. Execute code step by step.\n\n"
    "Guidelines:\n"
    "- Inspect data before writing code that depends on column names\n"
    "- Execute in logical steps, not one giant block\n"
    "- If code fails, diagnose and try a different approach\n"
    "- Use run_snippet() for exploration, execute_code() for processing\n"
    "- Call finish() when done with a summary of what was accomplished"
)

EXECUTE_PACKAGES_BLOCK = (
    "Available packages: ov (omicverse), sc (scanpy), np (numpy), pd (pandas), "
    "matplotlib, seaborn, scipy, sklearn\n"
    "All OmicVerse functions are called via ov.* "
    "(e.g. ov.pp.preprocess, ov.pp.scale)"
)


# =====================================================================
# PromptTemplateEngine
# =====================================================================

class PromptTemplateEngine:
    """Concrete prompt composer using layered ``PromptLayer`` composition.

    Satisfies the ``PromptComposer`` protocol from ``contracts.py``.
    Layers are added with :meth:`add_layer` (or the convenience
    :meth:`add`) and composed in priority order (lower = earlier).
    """

    def __init__(self) -> None:
        self._layers: List[PromptLayer] = []

    # -- PromptComposer protocol -------------------------------------------

    def add_layer(self, layer: PromptLayer) -> None:
        """Add a prompt layer to the composition stack."""
        self._layers.append(layer)

    def compose(self) -> str:
        """Assemble all layers into the final prompt string.

        Layers are sorted by priority (ascending) and joined with
        double newlines.
        """
        sorted_layers = self.layers()
        parts = [layer.content for layer in sorted_layers if layer.content.strip()]
        return "\n\n".join(parts)

    def layers(self) -> Sequence[PromptLayer]:
        """Return current layers sorted by priority (ascending)."""
        return sorted(self._layers, key=lambda l: l.priority)

    def total_tokens(self) -> int:
        """Estimated total token count of all layers."""
        return sum(layer.estimated_tokens() for layer in self._layers)

    # -- Extended API (beyond protocol) ------------------------------------

    def add(
        self,
        kind: PromptLayerKind,
        content: str,
        *,
        priority: int = 0,
        source: str = "",
    ) -> None:
        """Convenience: create and add a ``PromptLayer`` in one call.

        Empty or whitespace-only content is silently ignored.
        """
        if not content or not content.strip():
            return
        self.add_layer(PromptLayer(
            kind=kind,
            content=content,
            priority=priority,
            source=source,
        ))

    def clear(self) -> None:
        """Remove all layers."""
        self._layers.clear()

    def remove_kind(self, kind: PromptLayerKind) -> int:
        """Remove all layers of a given kind.  Returns count removed."""
        before = len(self._layers)
        self._layers = [l for l in self._layers if l.kind != kind]
        return before - len(self._layers)

    def has_kind(self, kind: PromptLayerKind) -> bool:
        """Check whether any layer of *kind* exists."""
        return any(l.kind == kind for l in self._layers)

    def layer_count(self) -> int:
        """Number of layers currently in the stack."""
        return len(self._layers)


# =====================================================================
# Helper: build a skill overlay layer
# =====================================================================

def build_skill_layer(
    skill_metadata: dict,
    *,
    priority: int = 85,
    header: str = "Available domain skills (use search_skills for detailed guidance):",
) -> Optional[PromptLayer]:
    """Build a SKILL prompt layer from a skill_metadata mapping.

    Parameters
    ----------
    skill_metadata : dict
        Mapping of slug → SkillMetadata with ``.slug`` and ``.description``.
    priority : int
        Layer priority (default 85, after base system blocks).
    header : str
        Header line for the skill list.

    Returns
    -------
    PromptLayer or None
        ``None`` when *skill_metadata* is empty.
    """
    if not skill_metadata:
        return None
    lines = [header]
    for meta in sorted(skill_metadata.values(), key=lambda s: s.slug):
        lines.append(f"  - {meta.slug}: {meta.description[:80]}")
    return PromptLayer(
        kind=PromptLayerKind.SKILL,
        content="\n".join(lines),
        priority=priority,
        source="skills",
    )


# =====================================================================
# Exports
# =====================================================================

__all__ = [
    # Engine
    "PromptTemplateEngine",
    # Block constants
    "IDENTITY_BLOCK",
    "TOOL_CATALOG_BLOCK",
    "CODING_FILESYSTEM_BLOCK",
    "WEB_ACCESS_BLOCK",
    "ANALYSIS_WORKFLOW_BLOCK",
    "CODE_QUALITY_BLOCK",
    "GUIDELINES_BLOCK",
    "DELEGATION_BLOCK",
    "CLAW_CODE_ONLY_BLOCK",
    "EXPLORE_IDENTITY_BLOCK",
    "EXPLORE_REPORT_BLOCK",
    "PLAN_IDENTITY_BLOCK",
    "PLAN_OUTPUT_BLOCK",
    "EXECUTE_IDENTITY_BLOCK",
    "EXECUTE_PACKAGES_BLOCK",
    # Helpers
    "build_skill_layer",
]
