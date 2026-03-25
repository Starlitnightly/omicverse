"""Prompt construction for OVAgent agentic and subagent loops.

Extracted from ``smart_agent.py`` to keep prompt assembly in one place.
All methods are pure string construction with no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

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

        if getattr(self._ctx, "_code_only_mode", False):
            prompt += (
                "\nCLAW CODE-ONLY MODE:\n"
                "- Reuse the normal OmicVerse Agent workflow, tools, search_functions, and search_skills logic.\n"
                "- In this mode, execute_code captures the final Python code instead of running it.\n"
                "- Use the same planning and tool-calling behavior you would use in Jarvis.\n"
                "- When ready, call execute_code with the final OmicVerse Python snippet, then call finish.\n"
                "- Do not stop at a prose-only plan when executable code is requested.\n"
            )

        registry = self._ctx.skill_registry
        if registry is not None and getattr(registry, "skill_metadata", None):
            prompt += "\nAvailable domain skills (use search_skills for detailed guidance):\n"
            for meta in sorted(registry.skill_metadata.values(), key=lambda s: s.slug):
                prompt += f"  - {meta.slug}: {meta.description[:80]}\n"

        if self._ctx._ov_runtime is not None:
            prompt = self._ctx._ov_runtime.compose_system_prompt(prompt)

        return prompt

    def build_initial_user_message(
        self,
        request: str,
        adata: Any,
        *,
        extra_content: Optional[List[dict]] = None,
    ) -> Any:
        msg = f"Task: {request}\n\n"
        if getattr(self._ctx, "_code_only_mode", False):
            if adata is not None:
                dtype = type(adata).__name__
                if hasattr(adata, "shape"):
                    msg += f"Dataset ({dtype}): {adata.shape[0]} cells x {adata.shape[1]} features\n"
                if dtype == "MuData" and hasattr(adata, "mod"):
                    msg += f"Modalities: {list(adata.mod.keys())}\n"
                msg += (
                    "This is a Claw code-only request. Inspect/search as needed, then call "
                    "execute_code with the final script. The code will be captured, not run."
                )
            else:
                msg += (
                    "No live dataset object is loaded for this Claw code-only request.\n"
                    "Assume an AnnData object named `adata` already exists unless the user "
                    "explicitly asks to load data from disk.\n"
                    "Use the normal Jarvis tool workflow, then call execute_code with the "
                    "final OmicVerse Python script. The code will be captured, not run."
                )
            if extra_content:
                return [{"type": "input_text", "text": msg}] + list(extra_content)
            return msg
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
        if extra_content:
            return [{"type": "input_text", "text": msg}] + list(extra_content)
        return msg


# ---------------------------------------------------------------------------
# Filesystem context instructions (extracted from smart_agent._build_*)
# ---------------------------------------------------------------------------

def build_filesystem_context_instructions(session_id: str = "N/A") -> str:
    """Build instructions for using the filesystem context workspace.

    This teaches LLMs how to use the filesystem-based context management
    system for offloading intermediate results, plans, and notes.

    Parameters
    ----------
    session_id : str
        Current filesystem context session identifier.

    Returns
    -------
    str
        Instructions for filesystem context usage.
    """
    return f"""

## Context Engineering with Filesystem Workspace

You have access to a **filesystem-based context workspace** that allows you to:
- Offload intermediate results to reduce memory/context usage
- Save and track execution plans across multiple steps
- Search for relevant context using patterns
- Share context with sub-agents

**Current Session**: `{session_id}`

### Why Use the Workspace?

1. **Reduce Context Window Usage**: Instead of keeping all results in memory, write them to disk
2. **Track Multi-Step Workflows**: Save plans and update progress as you complete steps
3. **Retrieve Relevant Context**: Search for notes when you need specific information
4. **Debug and Audit**: All notes are persisted for later review

### Available Context Operations

#### 1. Writing Notes (Offload Results)
Use `# CONTEXT_WRITE:` comments in your code to indicate what should be saved:

```python
# After completing a step, offload the result
# CONTEXT_WRITE: qc_result -> {{"n_cells_before": original_count, "n_cells_after": adata.n_obs, "removed": removed_count}}

# Example: Save intermediate statistics
qc_stats = {{
    "n_cells": adata.n_obs,
    "n_genes": adata.n_vars,
    "mito_pct_mean": float(adata.obs['pct_counts_mt'].mean()) if 'pct_counts_mt' in adata.obs else None
}}
# CONTEXT_WRITE: qc_stats -> qc_stats
```

#### 2. Saving Execution Plans
For multi-step workflows, define a plan upfront:

```python
# CONTEXT_PLAN:
# - Step 1: Quality Control [pending]
# - Step 2: Normalization [pending]
# - Step 3: Feature Selection [pending]
# - Step 4: Dimensionality Reduction [pending]
# - Step 5: Clustering [pending]
```

#### 3. Updating Plan Progress
As you complete steps, update the plan:

```python
# CONTEXT_UPDATE: step=0, status=completed, result="QC removed 500 low-quality cells"
```

#### 4. Searching for Context
When you need to reference previous results:

```python
# CONTEXT_SEARCH: pattern="qc*", type="glob"
# Or for content search:
# CONTEXT_SEARCH: pattern="resolution", type="grep"
```

### Context Categories

Organize your notes by category:
- **notes**: General observations and comments
- **results**: Computation results (statistics, parameters)
- **decisions**: Important choices and their rationale
- **snapshots**: Data state at key points
- **errors**: Error logs and debugging information

### Best Practices

1. **Write Early, Write Often**: Offload results as soon as they're computed
2. **Use Descriptive Keys**: `clustering_leiden_res1.0` is better than `result1`
3. **Include Metadata**: Add function names, parameters, timestamps
4. **Reference Previous Context**: Check workspace before repeating computations
5. **Update Plans Promptly**: Mark steps complete immediately after finishing

### Example: Multi-Step Workflow with Context

```python
import omicverse as ov

# CONTEXT_PLAN:
# - Step 1: Quality Control [in_progress]
# - Step 2: Preprocessing [pending]
# - Step 3: Clustering [pending]

# Step 1: QC
original_cells = adata.n_obs
adata = ov.pp.qc(adata, tresh={{'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}})
removed = original_cells - adata.n_obs

# CONTEXT_WRITE: qc_result -> {{"original": original_cells, "remaining": adata.n_obs, "removed": removed}}
# CONTEXT_UPDATE: step=0, status=completed, result="Removed " + str(removed) + " cells"

print("QC completed: " + str(adata.n_obs) + " cells remaining")
```

### Automatic Context Injection

The workspace context is automatically searched and injected into prompts when relevant.
You can reference previous results without explicitly searching:

- Recent notes are included automatically
- Plan status is always visible
- Relevant context is retrieved based on the current task
"""
