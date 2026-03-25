"""Prompt construction for OVAgent agentic and subagent loops.

Extracted from ``smart_agent.py`` to keep prompt assembly in one place.
Uses :class:`PromptTemplateEngine` for block-based composition instead
of monolithic string concatenation.  Each logical section is a named
block constant defined in :mod:`prompt_templates`; overlays (provider,
workflow, skill) are explicit ``PromptLayer`` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .contracts import PromptLayerKind
from .prompt_templates import (
    PromptTemplateEngine,
    build_skill_layer,
    # Main agentic blocks
    IDENTITY_BLOCK,
    TOOL_CATALOG_BLOCK,
    CODING_FILESYSTEM_BLOCK,
    WEB_ACCESS_BLOCK,
    ANALYSIS_WORKFLOW_BLOCK,
    CODE_QUALITY_BLOCK,
    GUIDELINES_BLOCK,
    DELEGATION_BLOCK,
    CLAW_CODE_ONLY_BLOCK,
    # Subagent blocks
    EXPLORE_IDENTITY_BLOCK,
    EXPLORE_REPORT_BLOCK,
    PLAN_IDENTITY_BLOCK,
    PLAN_OUTPUT_BLOCK,
    EXECUTE_IDENTITY_BLOCK,
    EXECUTE_PACKAGES_BLOCK,
)

if TYPE_CHECKING:
    from .protocol import AgentContext


# ---------------------------------------------------------------------------
# Backward-compatible constant (imported by session_context.py and others)
# ---------------------------------------------------------------------------

CODE_QUALITY_RULES = CODE_QUALITY_BLOCK + "\n"


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Build system / user prompts for the agentic loop and subagents.

    Internally uses :class:`PromptTemplateEngine` for layered composition.
    Each section of the prompt is an explicit ``PromptLayer`` that can be
    inspected, replaced, or removed before final composition.
    """

    def __init__(self, ctx: "AgentContext") -> None:
        self._ctx = ctx

    @staticmethod
    def _new_engine() -> PromptTemplateEngine:
        """Create a fresh engine for prompt composition."""
        return PromptTemplateEngine()

    # -- subagent prompts ---------------------------------------------------

    def build_explore_prompt(self, context: str) -> str:
        """Build system prompt for the *explore* subagent."""
        engine = self._new_engine()
        engine.add(PromptLayerKind.BASE_SYSTEM, EXPLORE_IDENTITY_BLOCK,
                   priority=0, source="explore_identity")
        engine.add(PromptLayerKind.BASE_SYSTEM, EXPLORE_REPORT_BLOCK,
                   priority=10, source="explore_report")
        if context:
            engine.add(PromptLayerKind.CONTEXT,
                       f"Additional context from parent: {context}",
                       priority=90, source="parent_context")
        return engine.compose()

    def build_plan_prompt(self, context: str) -> str:
        """Build system prompt for the *plan* subagent."""
        engine = self._new_engine()
        engine.add(PromptLayerKind.BASE_SYSTEM, PLAN_IDENTITY_BLOCK,
                   priority=0, source="plan_identity")
        engine.add(PromptLayerKind.BASE_SYSTEM, PLAN_OUTPUT_BLOCK,
                   priority=10, source="plan_output")
        engine.add(PromptLayerKind.BASE_SYSTEM, CODE_QUALITY_BLOCK,
                   priority=20, source="code_quality")
        engine.add(PromptLayerKind.BASE_SYSTEM,
                   "Call finish() with the complete plan when done.",
                   priority=30, source="plan_finish")
        if context:
            engine.add(PromptLayerKind.CONTEXT,
                       f"Additional context from parent: {context}",
                       priority=90, source="parent_context")
        # Skill overlay
        registry = self._ctx.skill_registry
        if registry is not None and getattr(registry, "skill_metadata", None):
            layer = build_skill_layer(
                registry.skill_metadata, priority=80,
                header="Available domain skills (use search_skills for details):",
            )
            if layer is not None:
                engine.add_layer(layer)
        return engine.compose()

    def build_execute_prompt(self, context: str) -> str:
        """Build system prompt for the *execute* subagent."""
        engine = self._new_engine()
        engine.add(PromptLayerKind.BASE_SYSTEM, EXECUTE_IDENTITY_BLOCK,
                   priority=0, source="execute_identity")
        engine.add(PromptLayerKind.BASE_SYSTEM, CODE_QUALITY_BLOCK,
                   priority=10, source="code_quality")
        engine.add(PromptLayerKind.BASE_SYSTEM, EXECUTE_PACKAGES_BLOCK,
                   priority=20, source="execute_packages")
        if context:
            engine.add(PromptLayerKind.CONTEXT,
                       f"Context from parent: {context}",
                       priority=90, source="parent_context")
        return engine.compose()

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
        """Build the full system prompt by composing all layers.

        Equivalent to the previous monolithic string but assembled from
        explicit, inspectable ``PromptLayer`` objects.
        """
        return self.build_agentic_engine().compose()

    def build_agentic_engine(self) -> PromptTemplateEngine:
        """Build and return the engine with all layers for the main prompt.

        Callers that need to inspect, modify, or extend the prompt can
        use this method to obtain the engine before calling
        ``engine.compose()``.
        """
        engine = self._new_engine()

        # ── Base system layers ──────────────────────────────────────
        engine.add(PromptLayerKind.BASE_SYSTEM, IDENTITY_BLOCK,
                   priority=0, source="identity")
        engine.add(PromptLayerKind.BASE_SYSTEM, TOOL_CATALOG_BLOCK,
                   priority=10, source="tool_catalog")
        engine.add(PromptLayerKind.BASE_SYSTEM, CODING_FILESYSTEM_BLOCK,
                   priority=20, source="coding_filesystem")
        engine.add(PromptLayerKind.BASE_SYSTEM, WEB_ACCESS_BLOCK,
                   priority=30, source="web_access")
        engine.add(PromptLayerKind.BASE_SYSTEM, ANALYSIS_WORKFLOW_BLOCK,
                   priority=40, source="analysis_workflow")
        engine.add(PromptLayerKind.BASE_SYSTEM, CODE_QUALITY_BLOCK,
                   priority=50, source="code_quality")
        engine.add(PromptLayerKind.BASE_SYSTEM, GUIDELINES_BLOCK,
                   priority=60, source="guidelines")
        engine.add(PromptLayerKind.BASE_SYSTEM, DELEGATION_BLOCK,
                   priority=70, source="delegation")

        # ── Provider overlay: Claw code-only mode ───────────────────
        if getattr(self._ctx, "_code_only_mode", False):
            engine.add(PromptLayerKind.PROVIDER, CLAW_CODE_ONLY_BLOCK,
                       priority=80, source="claw_code_only")

        # ── Skill overlay ───────────────────────────────────────────
        registry = self._ctx.skill_registry
        if registry is not None and getattr(registry, "skill_metadata", None):
            layer = build_skill_layer(registry.skill_metadata, priority=85)
            if layer is not None:
                engine.add_layer(layer)

        # ── Workflow overlay (from OmicVerseRuntime) ────────────────
        if self._ctx._ov_runtime is not None:
            workflow = self._ctx._ov_runtime.reload_workflow()
            if workflow.body or workflow.config.default_tools:
                engine.add(PromptLayerKind.WORKFLOW,
                           workflow.build_prompt_block(),
                           priority=90, source="workflow")

        return engine

    def build_initial_user_message(self, request: str, adata: Any) -> str:
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
