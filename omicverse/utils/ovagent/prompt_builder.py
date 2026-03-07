"""Prompt construction for OVAgent agentic and subagent loops.

Extracted from ``smart_agent.py`` to keep prompt assembly in one place.
All methods are pure string construction with no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import AgentContext


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CODE_QUALITY_RULES = (
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
    "(check existence, fillna, astype('category'))\n"
)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Build system / user prompts for the agentic loop and subagents."""

    def __init__(self, ctx: "AgentContext") -> None:
        self._ctx = ctx

    # -- subagent prompts ---------------------------------------------------

    def build_explore_prompt(self, context: str) -> str:
        prompt = (
            "You are a bioinformatics data inspector for OmicVerse. "
            "Your job is to thoroughly characterize the dataset and report findings.\n\n"
            "You CANNOT modify the data. Use inspect_data and run_snippet to explore.\n"
            "Use search_functions to understand what OmicVerse functions are available.\n\n"
            "Report:\n"
            "1. Dataset dimensions (cells x genes)\n"
            "2. Available metadata columns (obs, var) with example values\n"
            "3. Existing embeddings (obsm) and layers\n"
            "4. Data quality indicators (sparsity, mito%, batch columns, cell type annotations)\n"
            "5. Potential issues or recommendations\n\n"
            "Available packages: ov (omicverse), sc (scanpy), np (numpy), pd (pandas)\n\n"
            "Call finish() with a comprehensive summary when done.\n"
        )
        if context:
            prompt += f"\nAdditional context from parent: {context}"
        return prompt

    def build_plan_prompt(self, context: str) -> str:
        prompt = (
            "You are a bioinformatics workflow architect for OmicVerse. "
            "Design a step-by-step execution plan. Do NOT execute code — only plan.\n\n"
            "Use search_functions to find OmicVerse functions and their prerequisites.\n"
            "Use search_skills for domain-specific workflow guidance.\n"
            "Use inspect_data and run_snippet to understand current data state.\n\n"
            "Your plan should include:\n"
            "1. Ordered steps with specific OmicVerse function calls and parameters\n"
            "2. Quality check points between steps\n"
            "3. Fallback strategies for known failure modes\n"
            "4. Expected outputs at each step (obsm, obs columns, layers)\n\n"
        )
        prompt += CODE_QUALITY_RULES
        prompt += "\nCall finish() with the complete plan when done.\n"
        if context:
            prompt += f"\nAdditional context from parent: {context}"
        registry = self._ctx.skill_registry
        if registry is not None and getattr(registry, "skill_metadata", None):
            prompt += "\nAvailable domain skills (use search_skills for details):\n"
            for meta in sorted(registry.skill_metadata.values(), key=lambda s: s.slug):
                prompt += f"  - {meta.slug}: {meta.description[:80]}\n"
        return prompt

    def build_execute_prompt(self, context: str) -> str:
        prompt = (
            "You are a focused bioinformatics code executor for OmicVerse. "
            "Complete the specific sub-task described. Execute code step by step.\n\n"
            "Guidelines:\n"
            "- Inspect data before writing code that depends on column names\n"
            "- Execute in logical steps, not one giant block\n"
            "- If code fails, diagnose and try a different approach\n"
            "- Use run_snippet() for exploration, execute_code() for processing\n"
            "- Call finish() when done with a summary of what was accomplished\n\n"
        )
        prompt += CODE_QUALITY_RULES
        prompt += (
            "\nAvailable packages: ov (omicverse), sc (scanpy), np (numpy), pd (pandas), "
            "matplotlib, seaborn, scipy, sklearn\n"
            "All OmicVerse functions are called via ov.* (e.g. ov.pp.preprocess, ov.pp.scale)\n"
        )
        if context:
            prompt += f"\nContext from parent: {context}"
        return prompt

    def build_subagent_system_prompt(self, agent_type: str, context: str = "") -> str:
        if agent_type == "explore":
            return self.build_explore_prompt(context)
        elif agent_type == "plan":
            return self.build_plan_prompt(context)
        elif agent_type == "execute":
            return self.build_execute_prompt(context)
        raise ValueError(f"Unknown subagent type: {agent_type}")

    def build_subagent_user_message(self, task: str, adata: Any) -> str:
        msg = f"Task: {task}\n\n"
        if adata is not None and hasattr(adata, "shape"):
            msg += f"Dataset: {adata.shape[0]} cells x {adata.shape[1]} genes\n"
        return msg

    # -- main agentic prompt ------------------------------------------------

    def build_agentic_system_prompt(self) -> str:
        prompt = (
            "You are OmicVerse Agent, an expert bioinformatics assistant that processes "
            "single-cell, bulk RNA-seq, and spatial transcriptomics data.\n\n"
            "You have OmicVerse data tools plus a Claude-style coding tool catalog.\n"
            "Core Claude tools are ToolSearch, Bash, and Read. Deferred Claude tools "
            "must be loaded through ToolSearch before use. High-risk tools require approval.\n\n"
            "Use the legacy OmicVerse tools for dataset inspection and execution, and "
            "use Claude-style tools for files, tasks, planning, worktrees, and questions.\n\n"
            "Do not narrate future actions without taking them in the same turn. "
            "If a tool is needed, call it now instead of saying that you will call it later.\n\n"
            "CODING / FILESYSTEM:\n"
            "- Use ToolSearch when you need deferred tools such as Edit, Write, Glob, Grep, NotebookEdit, Task*, or EnterWorktree\n"
            "- Use Read for local files instead of shelling out to cat/head/sed when possible\n"
            "- Use AskUserQuestion when a turn genuinely needs user clarification\n\n"
            "WEB ACCESS:\n"
            "- Use WebFetch(url) or web_fetch(url) to read any web page (GEO datasets, papers, docs)\n"
            "- Use WebSearch(query) or web_search(query) to search the internet for information\n"
            "- Use web_download(url) to download files (datasets, h5ad, csv, gz) to disk\n"
            "- When the user provides a URL, ALWAYS fetch it first instead of asking them to paste content\n"
            "- When you need background info (gene functions, diseases, methods), search the web\n"
            "- To download GEO data, fetch the GEO page first to find download links, then use web_download\n\n"
            "WORKFLOW:\n"
            "1. If the user provides a URL, use web_fetch to read it\n"
            "2. Use inspect_data to understand the dataset structure\n"
            "3. Use search_functions to find relevant OmicVerse functions "
            "(includes prerequisite chains and examples)\n"
            "4. For complex multi-step workflows, use search_skills for domain guidance\n"
            "5. Execute code step by step, checking results between steps\n"
            "6. Call finish() when the task is complete\n\n"
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
            "(check existence, fillna, astype('category'))\n\n"
            "Guidelines:\n"
            "- ALWAYS inspect data before writing code that depends on column names or structure\n"
            "- Execute code in logical steps, not one giant block\n"
            "- If code fails, read the error carefully, diagnose the issue, and try a different approach\n"
            "- Use run_snippet() for exploration (won't modify data)\n"
            "- Use execute_code() for actual processing (modifies data)\n"
            "- Available packages: ov (omicverse), sc (scanpy), np (numpy), pd (pandas), "
            "matplotlib, seaborn, scipy, sklearn\n"
            "- All OmicVerse functions are called via ov.* (e.g. ov.pp.preprocess, ov.pp.scale)\n"
            "- When saving files, use the output directory or current working directory\n\n"
            "DELEGATION STRATEGY (use the delegate tool for complex tasks):\n"
            "- Agent(subagent_type='explore', ...) or delegate('explore', ...) — read-only data characterization\n"
            "- Agent(subagent_type='plan', ...) or delegate('plan', ...) — design workflow before execution\n"
            "- Agent(subagent_type='execute', ...) or delegate('execute', ...) — focused execution step\n"
            "- For simple single-step operations, execute directly without delegation\n"
            "- Subagents have their own context window (prevents context overflow)\n"
        )

        registry = self._ctx.skill_registry
        if registry is not None and getattr(registry, "skill_metadata", None):
            prompt += "\nAvailable domain skills (use search_skills for detailed guidance):\n"
            for meta in sorted(registry.skill_metadata.values(), key=lambda s: s.slug):
                prompt += f"  - {meta.slug}: {meta.description[:80]}\n"

        if self._ctx._ov_runtime is not None:
            prompt = self._ctx._ov_runtime.compose_system_prompt(prompt)

        return prompt

    def build_initial_user_message(self, request: str, adata: Any) -> str:
        msg = f"Task: {request}\n\n"
        if adata is not None:
            dtype = type(adata).__name__
            if hasattr(adata, "shape"):
                msg += f"Dataset ({dtype}): {adata.shape[0]} cells x {adata.shape[1]} features\n"
            if dtype == "MuData" and hasattr(adata, "mod"):
                msg += f"Modalities: {list(adata.mod.keys())}\n"
            msg += "Use inspect_data to learn more about the dataset structure before writing code."
        else:
            msg += (
                "No dataset is loaded yet. You can still help with:\n"
                "- Use web_fetch/web_search to look up information\n"
                "- Use web_download to download datasets from URLs\n"
                "- Answer bioinformatics questions\n"
                "- Plan analysis workflows\n"
            )
        return msg
